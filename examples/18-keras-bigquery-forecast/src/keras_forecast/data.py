"""Synthetic daily sales in the layout returned by bigquery_io.load_daily_sales."""
import numpy as np
import pandas as pd


def make_daily_sales(days: int = 400, start: str = "2025-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.day_of_year.to_numpy()
    weekly = np.array([0.9, 0.85, 0.9, 1.0, 1.25, 1.4, 0.7])[dates.day_of_week]
    temperature = 12 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2, days)
    is_holiday = dates.strftime("%m-%d").isin(["01-01", "05-01", "12-24", "12-25", "12-26"]).astype(int)
    price = np.where(rng.random(days) < 0.1, 8.5, 10.0)
    demand = 50 * weekly * (1 + 0.01 * (temperature - 12)) * (1 + 0.6 * is_holiday) * (price / 10) ** -1.5
    return pd.DataFrame({"date": dates, "units": rng.poisson(demand).astype(float), "avg_price": price,
                         "temperature": temperature.round(1), "is_holiday": is_holiday})
