"""CLI: train on synthetic data (default) or a C-MAPSS file and save the model.

    python -m pdm.train --out model.joblib
    python -m pdm.train --cmapss data/train_FD001.txt --out model.joblib
"""
from __future__ import annotations

import argparse
import json

from .data import SENSORS, generate_fleet, load_cmapss
from .model import train

# C-MAPSS sensors that change over the life of the engine in FD001 (the others are constant).
CMAPSS_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s20", "s21"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cmapss", help="path to a C-MAPSS train_FD00x.txt file")
    p.add_argument("--machines", type=int, default=60, help="synthetic machines if no file is given")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--out", default="model.joblib")
    args = p.parse_args(argv)

    if args.cmapss:
        df, sensors = load_cmapss(args.cmapss), CMAPSS_SENSORS
    else:
        df, sensors = generate_fleet(args.machines), SENSORS
    model = train(df, sensors, window=args.window, horizon=args.horizon)
    model.save(args.out)
    print(json.dumps(model.metrics, indent=2))


if __name__ == "__main__":
    main()
