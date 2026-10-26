"""Synthetic daily sales for one SKU in one store.

The generator is deliberately simple but has the effects the forecaster has to
learn: trend, weekly and yearly seasonality, temperature, holidays and a
constant price elasticity. Replace it with your POS/ERP export in production.
"""
import numpy as np
import pandas as pd

HOLIDAYS = ["01-01", "04-18", "05-01", "10-03", "12-24", "12-25", "12-26", "12-31"]


def make_sales(days: int = 730, start: str = "2024-01-01", seed: int = 0, base_units: float = 40.0,
               list_price: float = 10.0, elasticity: float = -1.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    doy = dates.day_of_year.to_numpy()

    weekly = np.where(np.isin(dates.day_of_week, [4, 5]), 1.3, 1.0)  # Friday, Saturday
    yearly = 1 + 0.2 * np.sin(2 * np.pi * doy / 365.25)
    temperature = 12 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2, days)
    is_holiday = dates.strftime("%m-%d").isin(HOLIDAYS).astype(int)
    promo = rng.random(days) < 0.08
    our_price = np.round(list_price * np.where(promo, 0.85, 1.0), 2)
    competitor_price = np.round(list_price * (1 + rng.normal(0, 0.05, days)), 2)

    demand = (base_units * (1 + 0.0005 * t) * weekly * yearly
              * (1 + 0.01 * (temperature - 12)) * (1 + 0.5 * is_holiday)
              * (our_price / list_price) ** elasticity)
    return pd.DataFrame({
        "date": dates,
        "sales": rng.poisson(demand).astype(float),
        "our_price": our_price,
        "competitor_price": competitor_price,
        "temperature": np.round(temperature, 1),
        "is_holiday": is_holiday,
    })
