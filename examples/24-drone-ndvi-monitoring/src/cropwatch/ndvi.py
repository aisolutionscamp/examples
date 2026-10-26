"""NDVI from a multispectral drone orthomosaic and per-zone field statistics.

NDVI = (NIR - Red) / (NIR + Red), from -1 to 1. Dense, healthy canopy is
typically above ~0.6, bare soil around 0.1-0.2. A low or falling NDVI says
"something is wrong here", not what: water stress, nutrient deficiency,
disease and thin stands all look similar. Zones are flagged for scouting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geotiff
from .geotiff import GeoGrid


@dataclass
class Thresholds:
    # Default classes for a closed canopy in mid season; adjust per crop and growth stage.
    critical: float = 0.2
    stressed: float = 0.4
    watch: float = 0.6
    drop_alert: float = 0.1  # NDVI drop vs. previous flight that raises an alert
    min_valid_share: float = 0.5  # zones with less valid data are reported as "no data"


@dataclass
class Raster:
    ndvi: np.ndarray  # float32, NaN = no data
    grid: GeoGrid


def compute_ndvi(nir: np.ndarray, red: np.ndarray, nodata: np.ndarray | None = None) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(total > 0, (nir - red) / total, np.nan)
    if nodata is not None:
        ndvi[nodata] = np.nan
    return np.clip(ndvi, -1, 1).astype(np.float32)


def read_ndvi(path: str, red_band: int, nir_band: int) -> Raster:
    """Read a multiband GeoTIFF (band numbers start at 1) and compute NDVI."""
    bands, grid, nodata_value = geotiff.read(path)
    red, nir = bands[red_band - 1], bands[nir_band - 1]
    nodata = None
    if nodata_value is not None:
        nodata = (red == nodata_value) | (nir == nodata_value)
    return Raster(compute_ndvi(nir, red, nodata), grid)


def write_ndvi(path: str, raster: Raster) -> None:
    geotiff.write(path, raster.ndvi, raster.grid, nodata=float("nan"))


@dataclass
class Zone:
    row: int
    col: int
    polygon: list[tuple[float, float]]  # ring in the raster CRS
    mean_ndvi: float | None
    low_share: float  # share of valid pixels below Thresholds.stressed
    status: str
    change: float | None = None
    alerts: list[str] = field(default_factory=list)


def classify(mean: float | None, t: Thresholds) -> str:
    if mean is None:
        return "no data"
    if mean < t.critical:
        return "critical"
    if mean < t.stressed:
        return "stressed"
    if mean < t.watch:
        return "watch"
    return "healthy"


def zone_stats(raster: Raster, zone_size_m: float, t: Thresholds | None = None,
               previous: Raster | None = None) -> list[Zone]:
    """Split the field into square zones of zone_size_m and summarize each one.

    `previous` must be on the same grid (co-registered orthomosaic of an earlier flight).
    """
    t = t or Thresholds()
    if previous is not None and previous.ndvi.shape != raster.ndvi.shape:
        raise ValueError("previous raster must be co-registered (same shape and grid)")
    pixel = raster.grid.pixel_x
    step = max(1, round(zone_size_m / pixel))
    h, w = raster.ndvi.shape
    zones = []
    for r0 in range(0, h, step):
        for c0 in range(0, w, step):
            block = raster.ndvi[r0:r0 + step, c0:c0 + step]
            valid = block[~np.isnan(block)]
            enough = valid.size >= t.min_valid_share * block.size
            mean = float(valid.mean()) if enough else None
            low = float((valid < t.stressed).mean()) if valid.size else 0.0
            r1, c1 = r0 + block.shape[0], c0 + block.shape[1]
            ring = [raster.grid.xy(c, r) for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1), (c0, r0))]
            zone = Zone(r0 // step, c0 // step, ring, None if mean is None else round(mean, 3),
                        round(low, 3), classify(mean, t))
            if previous is not None and mean is not None:
                before = previous.ndvi[r0:r1, c0:c1]
                if np.isfinite(before).sum() >= t.min_valid_share * before.size:
                    zone.change = round(mean - float(np.nanmean(before)), 3)
            if zone.status in ("critical", "stressed"):
                zone.alerts.append(f"mean NDVI {zone.mean_ndvi} ({zone.status}): scout this zone")
            if zone.change is not None and zone.change <= -t.drop_alert:
                zone.alerts.append(f"NDVI dropped by {-zone.change:.2f} since the previous flight")
            zones.append(zone)
    return zones


def to_geojson(zones: list[Zone], epsg: int | None) -> dict:
    """FeatureCollection in the raster CRS (reproject to EPSG:4326 for web maps if needed)."""
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}} if epsg else None,
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in z.polygon]]},
            "properties": {"zone": f"{z.row}-{z.col}", "mean_ndvi": z.mean_ndvi, "low_share": z.low_share,
                           "status": z.status, "change": z.change, "alerts": z.alerts},
        } for z in zones],
    }
