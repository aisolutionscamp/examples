import json

import numpy as np
import pytest

from cropwatch import geotiff
from cropwatch.__main__ import run
from cropwatch.geotiff import GeoGrid
from cropwatch.ndvi import Thresholds, classify, compute_ndvi, read_ndvi, zone_stats

ORIGIN = (500000.0, 5300000.0)  # UTM 32N, meters
PIXEL = 0.5


def write_field(path, ndvi_target, nodata_rows=0):
    """Synthetic 5-band orthomosaic (B, G, R, RedEdge, NIR) with a given NDVI map."""
    h, w = ndvi_target.shape
    red = np.full((h, w), 1000, np.uint16)
    nir = (red * (1 + ndvi_target) / (1 - ndvi_target)).astype(np.uint16)  # solves NDVI for NIR
    bands = np.stack([red // 2, red, red, nir // 2, nir])
    if nodata_rows:
        bands[:, :nodata_rows] = 0
    geotiff.write(str(path), bands, GeoGrid(*ORIGIN, PIXEL, PIXEL, 32632), nodata=0)
    return path


def field_map(stressed_value=0.3):
    ndvi = np.full((80, 80), 0.8)
    ndvi[:20, :20] = stressed_value  # north-west zone (10 m x 10 m at 0.5 m/px)
    return ndvi


def test_compute_ndvi_values_and_division_by_zero():
    out = compute_ndvi(np.array([[3, 0, 1]]), np.array([[1, 0, 3]]))
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[0, 1])
    assert out[0, 2] == pytest.approx(-0.5)


def test_geotiff_roundtrip_keeps_grid_and_bands(tmp_path):
    bands = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    grid = GeoGrid(10.0, 20.0, 0.25, 0.25, 25832)
    geotiff.write(str(tmp_path / "r.tif"), bands, grid, nodata=0)
    data, back, nodata = geotiff.read(str(tmp_path / "r.tif"))
    assert (data == bands).all() and back == grid and nodata == 0
    assert back.xy(4, 2) == (11.0, 19.5)


def test_classify_thresholds():
    t = Thresholds()
    assert [classify(v, t) for v in (0.1, 0.3, 0.5, 0.7, None)] == \
        ["critical", "stressed", "watch", "healthy", "no data"]


def test_zone_stats_flags_stressed_corner(tmp_path):
    raster = read_ndvi(str(write_field(tmp_path / "f.tif", field_map())), red_band=3, nir_band=5)
    zones = zone_stats(raster, zone_size_m=10)
    assert len(zones) == 16
    nw = next(z for z in zones if (z.row, z.col) == (0, 0))
    assert nw.status == "stressed" and nw.mean_ndvi == pytest.approx(0.3, abs=0.01)
    assert nw.low_share == 1.0 and nw.alerts
    assert all(z.status == "healthy" and not z.alerts for z in zones if z is not nw)
    # polygon in map coordinates: 10 m square at the raster origin
    xs, ys = zip(*nw.polygon)
    assert (min(xs), max(xs), min(ys), max(ys)) == (ORIGIN[0], ORIGIN[0] + 10, ORIGIN[1] - 10, ORIGIN[1])


def test_nodata_is_ignored(tmp_path):
    raster = read_ndvi(str(write_field(tmp_path / "f.tif", np.full((40, 40), 0.7), nodata_rows=15)), 3, 5)
    assert np.isnan(raster.ndvi[:15]).all()
    zones = zone_stats(raster, zone_size_m=10)
    assert [z.status for z in zones if z.row == 0] == ["no data", "no data"]
    assert all(z.status == "healthy" for z in zones if z.row == 1)


def test_change_detection_against_previous_flight(tmp_path):
    before = read_ndvi(str(write_field(tmp_path / "a.tif", np.full((80, 80), 0.8))), 3, 5)
    after = read_ndvi(str(write_field(tmp_path / "b.tif", field_map(stressed_value=0.55))), 3, 5)
    zones = zone_stats(after, 10, previous=before)
    nw = zones[0]
    assert nw.status == "watch"
    assert nw.change == pytest.approx(-0.25, abs=0.01)
    assert any("dropped" in a for a in nw.alerts)
    with pytest.raises(ValueError):
        zone_stats(after, 10, previous=read_ndvi(str(write_field(tmp_path / "c.tif", np.full((40, 40), 0.8))), 3, 5))


def test_cli_writes_geotiff_and_geojson(tmp_path):
    report = run(str(write_field(tmp_path / "f.tif", field_map())), 3, 5, 10, tmp_path / "out")
    assert report["by_status"] == {"stressed": 1, "healthy": 15}
    assert report["alerts"][0]["zone"] == "0-0"
    data, grid, _ = geotiff.read(str(tmp_path / "out" / "ndvi.tif"))
    assert grid == GeoGrid(*ORIGIN, PIXEL, PIXEL, 32632) and data.shape == (1, 80, 80)
    geo = json.loads((tmp_path / "out" / "zones.geojson").read_text())
    assert len(geo["features"]) == 16
