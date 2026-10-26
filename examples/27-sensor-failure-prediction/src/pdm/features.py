"""Rolling-window features. Each row only uses readings up to and including its own cycle."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_features(df: pd.DataFrame, sensors: list[str], window: int = 10) -> pd.DataFrame:
    """Per machine and sensor: last value, rolling mean, rolling std and slope over `window` cycles.

    Rows without a full window (the first `window - 1` cycles of a machine) are dropped.
    """
    out = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    grouped = out.groupby("machine_id", sort=False)
    for s in sensors:
        roll = grouped[s].rolling(window, min_periods=window)
        out[f"{s}_last"] = out[s]
        out[f"{s}_mean"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{s}_std"] = roll.std().reset_index(level=0, drop=True)
        # Slope of a least-squares line through the window (change per cycle).
        out[f"{s}_slope"] = roll.apply(_slope, raw=True).reset_index(level=0, drop=True)
    return out.dropna(subset=[f"{sensors[0]}_mean"]).reset_index(drop=True)


def feature_columns(sensors: list[str]) -> list[str]:
    return [f"{s}_{kind}" for s in sensors for kind in ("last", "mean", "std", "slope")]


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float((x * (values - values.mean())).sum() / (x * x).sum())
