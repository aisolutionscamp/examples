"""Synthetic sales history in the Autopilot training layout. Replace with your sales export."""
import numpy as np
import pandas as pd

from .features import FEATURES, TARGET


def make_training_frame(days: int = 365, skus=("SKU-1", "SKU-2"), stores=("S01", "S02"), seed: int = 0,
                        elasticity: float = -1.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    frames = []
    for i, sku in enumerate(skus):
        for j, store in enumerate(stores):
            list_price = 10.0 + 5 * i
            price = np.round(list_price * rng.choice([0.85, 0.95, 1.0, 1.05], size=days), 2)
            competitor = np.round(list_price * (1 + rng.normal(0, 0.05, days)), 2)
            holiday = (dates.strftime("%m-%d").isin(["01-01", "12-24", "12-25", "12-26"])).astype(int)
            weekly = np.where(dates.day_of_week >= 4, 1.3, 1.0)
            demand = (30 + 10 * j) * weekly * (1 + 0.5 * holiday) * (price / competitor) ** elasticity
            frames.append(pd.DataFrame({
                "sku": sku, "store_id": store, "day_of_week": dates.day_of_week, "month": dates.month,
                "is_holiday": holiday, "price": price, "competitor_price": competitor,
                "price_ratio": np.round(price / competitor, 4), TARGET: rng.poisson(demand),
            }))
    return pd.concat(frames, ignore_index=True)[[*FEATURES, TARGET]]
