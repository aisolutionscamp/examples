"""Synthetic run-to-failure sensor data and a loader for the NASA C-MAPSS text format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["temperature", "vibration", "pressure", "rpm"]


def generate_fleet(n_machines: int = 40, seed: int = 0, min_life: int = 150, max_life: int = 350) -> pd.DataFrame:
    """Simulate machines that run until failure.

    Each machine has a hidden wear level that grows from 0 to 1 over its life.
    Wear raises temperature and vibration and lowers pressure and rpm, with noise
    on top. The returned frame has one row per machine and cycle plus the true
    remaining useful life (RUL), which a real plant only knows after the fact.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for machine in range(n_machines):
        life = int(rng.integers(min_life, max_life + 1))
        cycle = np.arange(life)
        wear = (cycle / (life - 1)) ** 2.5  # slow at first, fast near the end
        base_temp = rng.normal(70, 2)
        frames.append(pd.DataFrame({
            "machine_id": f"M{machine:03d}",
            "cycle": cycle,
            "temperature": base_temp + 15 * wear + rng.normal(0, 0.8, life),
            "vibration": 0.5 + 2.0 * wear**1.5 + rng.normal(0, 0.08, life),
            "pressure": 30 - 5 * wear + rng.normal(0, 0.4, life),
            "rpm": 1500 - 60 * wear + rng.normal(0, 6, life),
            "rul": life - 1 - cycle,
        }))
    return pd.concat(frames, ignore_index=True)


def add_failure_label(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Label = 1 if the machine fails within the next `horizon` cycles."""
    out = df.copy()
    out["fails_soon"] = (out["rul"] <= horizon).astype(int)
    return out


CMAPSS_COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS training file (e.g. train_FD001.txt) and add the RUL column.

    The files are space separated without a header: unit, cycle, 3 settings, 21 sensors.
    Training trajectories run to failure, so RUL = last cycle of the unit - cycle.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, usecols=range(26))
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df.rename(columns={"unit": "machine_id"})
