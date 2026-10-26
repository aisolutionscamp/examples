# NDVI field zones and alerts from drone imagery

Takes a multispectral drone orthomosaic (GeoTIFF with red and near-infrared bands), computes NDVI, splits the
field into square zones and flags zones with weak or declining vegetation for scouting.

NDVI = (NIR - Red) / (NIR + Red). Dense, healthy canopy is usually above about 0.6, bare soil around 0.1 to 0.2.
A low or falling NDVI shows *where* plants are weaker, not *why*: water stress, nitrogen deficiency, disease,
pests and thin stands all look alike. The alerts therefore ask for a field check instead of recommending a
treatment.

## Architecture

```
drone images ──► orthomosaic GeoTIFF (OpenDroneMap / Pix4D / Metashape)
   ──► read red + NIR bands, no-data mask ──► NDVI per pixel
   ──► zones of N m x N m: mean NDVI, share below "stressed", status
   ──► (optional) change vs. earlier flight on the same grid
   ──► out/ndvi.tif, out/zones.geojson, alerts (JSON)
```

- `ndvi.py`: `compute_ndvi`, `zone_stats`, `classify`, `to_geojson`.
- `geotiff.py`: minimal GeoTIFF read/write with `tifffile` (north-up grid, EPSG code).
- `__main__.py`: command line.

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m cropwatch /data/field_2026-06-01.tif --red 3 --nir 5 --zone-size 10 --out-dir ../out
python -m cropwatch /data/field_2026-06-15.tif --red 3 --nir 5 --previous /data/field_2026-06-01.tif
```

Band numbers start at 1 and depend on the camera and the export (for a 5-band stack ordered blue, green, red,
red edge, NIR: `--red 3 --nir 5`). Output: `ndvi.tif` (float32, NaN = no data), `zones.geojson` (one polygon per
zone in the raster CRS with `status`, `mean_ndvi`, `low_share`, `change`, `alerts`) and a JSON summary on stdout.

Tests (synthetic GeoTIFFs written in a temp folder): `pytest`.

## Configuration

`Thresholds` in `ndvi.py`:

| Field | Default | Meaning |
|---|---|---|
| `critical` | 0.2 | below: bare soil or dead vegetation |
| `stressed` | 0.4 | below: clearly weak vegetation, alert |
| `watch` | 0.6 | below: worth a look, no alert |
| `drop_alert` | 0.1 | alert when a zone's mean NDVI fell this much since the previous flight |
| `min_valid_share` | 0.5 | zones with less valid data are "no data" |

The defaults suit a closed canopy in mid season. Early in the season, or for row crops with visible soil, NDVI is
naturally lower; set thresholds per crop and growth stage, or compare zones with the field median instead.

## Limitations

- `geotiff.py` handles north-up GeoTIFFs with an EPSG code (what photogrammetry tools export). Rotated grids,
  reprojection and other formats need rasterio/GDAL. rasterio was not used here because its current wheels need
  the system library `libexpat`, which the `python:3.12-slim` test image does not contain (in a full image or with
  `apt-get install libexpat1` it works).
- Change detection needs both flights on the same pixel grid (co-registered). Reproject/resample first if not.
- NDVI depends on radiometric calibration (reflectance panel or sunlight sensor) and on light conditions.
  Uncalibrated digital numbers from different flights are not comparable.
- NDVI saturates for very dense canopy; NDRE (red edge) can be more sensitive late in the season.

## For production

- Calibrate every flight and process with the same settings, so values are comparable over the season.
- Reproject zones to EPSG:4326 for web maps, or load them into the farm management system's field boundaries.
- Keep scouting results per zone and use them to tune thresholds, or train a classifier on NDVI, NDRE and
  weather features once enough labeled zones exist.
- Feed irrigation decisions with soil moisture sensors or evapotranspiration data, not NDVI alone.
