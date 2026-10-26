"""Analyze one flight (optionally against an earlier one).

    python -m cropwatch field.tif --red 3 --nir 5 --zone-size 10 \
        --previous last_week.tif --out-dir out/
Writes out/ndvi.tif, out/zones.geojson and prints the alerts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ndvi import Thresholds, read_ndvi, to_geojson, write_ndvi, zone_stats


def run(path: str, red: int, nir: int, zone_size: float, out_dir: Path,
        previous: str | None = None, thresholds: Thresholds | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    current = read_ndvi(path, red, nir)
    before = read_ndvi(previous, red, nir) if previous else None
    zones = zone_stats(current, zone_size, thresholds, before)
    write_ndvi(str(out_dir / "ndvi.tif"), current)
    (out_dir / "zones.geojson").write_text(json.dumps(to_geojson(zones, current.grid.epsg)))
    counts: dict[str, int] = {}
    for z in zones:
        counts[z.status] = counts.get(z.status, 0) + 1
    return {"zones": len(zones), "by_status": counts,
            "alerts": [{"zone": f"{z.row}-{z.col}", "messages": z.alerts} for z in zones if z.alerts]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="NDVI zones and alerts from a multispectral GeoTIFF.")
    p.add_argument("raster")
    p.add_argument("--red", type=int, required=True, help="band number of red (1-based)")
    p.add_argument("--nir", type=int, required=True, help="band number of near infrared (1-based)")
    p.add_argument("--zone-size", type=float, default=10.0, help="zone edge length in meters")
    p.add_argument("--previous", help="co-registered GeoTIFF of an earlier flight")
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    a = p.parse_args(argv)
    print(json.dumps(run(a.raster, a.red, a.nir, a.zone_size, a.out_dir, a.previous), indent=2))


if __name__ == "__main__":
    main()
