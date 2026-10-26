"""Minimal GeoTIFF reading/writing with tifffile (north-up rasters, EPSG CRS).

Drone orthomosaics from photogrammetry tools (OpenDroneMap, Pix4D, Agisoft
Metashape) are north-up GeoTIFFs: the pixel grid is defined by one tie point
and a pixel size. That is all this module handles. For rotated grids,
reprojection or other formats use rasterio/GDAL (see README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tifffile

TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEOKEYS, TAG_GDAL_NODATA = 33550, 33922, 34735, 42113
KEY_MODEL_TYPE, KEY_RASTER_TYPE, KEY_GEOGRAPHIC_CRS, KEY_PROJECTED_CRS = 1024, 1025, 2048, 3072


@dataclass(frozen=True)
class GeoGrid:
    x0: float  # map x of the upper-left corner of pixel (0, 0)
    y0: float  # map y of that corner
    pixel_x: float
    pixel_y: float
    epsg: int | None

    def xy(self, col: float, row: float) -> tuple[float, float]:
        """Map coordinates of a pixel corner (col, row)."""
        return self.x0 + col * self.pixel_x, self.y0 - row * self.pixel_y


def read(path: str) -> tuple[np.ndarray, GeoGrid, float | None]:
    """Return (bands as array [band, row, col], grid, nodata value)."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        data = page.asarray()
        tags = page.tags
        if TAG_PIXEL_SCALE not in tags or TAG_TIEPOINT not in tags:
            raise ValueError(f"{path} has no GeoTIFF georeferencing tags")
        sx, sy = tags[TAG_PIXEL_SCALE].value[:2]
        i, j, _, x, y, _ = tags[TAG_TIEPOINT].value[:6]
        epsg = None
        if TAG_GEOKEYS in tags:
            keys = tags[TAG_GEOKEYS].value
            entries = {keys[n]: keys[n + 3] for n in range(4, 4 + 4 * keys[3], 4)}
            epsg = entries.get(KEY_PROJECTED_CRS) or entries.get(KEY_GEOGRAPHIC_CRS)
        nodata = float(tags[TAG_GDAL_NODATA].value.strip("\x00 ")) if TAG_GDAL_NODATA in tags else None
    if data.ndim == 2:
        data = data[np.newaxis]
    elif page.planarconfig == tifffile.PLANARCONFIG.CONTIG:
        data = np.moveaxis(data, -1, 0)  # (rows, cols, bands) -> (bands, rows, cols)
    grid = GeoGrid(float(x - i * sx), float(y + j * sy), float(sx), float(sy), epsg)
    return data, grid, nodata


def write(path: str, data: np.ndarray, grid: GeoGrid, nodata: float | None = None) -> None:
    """Write [band, row, col] or [row, col] data as a GeoTIFF."""
    geographic = grid.epsg is not None and grid.epsg == 4326
    geokeys = [1, 1, 0, 3, KEY_MODEL_TYPE, 0, 1, 2 if geographic else 1, KEY_RASTER_TYPE, 0, 1, 1,
               KEY_GEOGRAPHIC_CRS if geographic else KEY_PROJECTED_CRS, 0, 1, grid.epsg or 32767]
    extratags = [
        (TAG_PIXEL_SCALE, "d", 3, (grid.pixel_x, grid.pixel_y, 0.0), True),
        (TAG_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, grid.x0, grid.y0, 0.0), True),
        (TAG_GEOKEYS, "H", len(geokeys), geokeys, True),
    ]
    if nodata is not None:
        extratags.append((TAG_GDAL_NODATA, "s", 0, str(nodata), True))
    tifffile.imwrite(path, data, photometric="minisblack", planarconfig="separate" if data.ndim == 3 else None,
                     compression="zlib", extratags=extratags)
