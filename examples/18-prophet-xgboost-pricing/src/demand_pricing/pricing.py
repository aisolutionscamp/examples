"""Price recommendation from a demand forecast.

Demand response uses a constant price elasticity estimated from history.
The search is a grid over a price corridor set by business rules (minimum
margin, maximum gap to the competitor, maximum change per update), and sales
are capped by the stock on hand.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PriceDecision:
    current_price: float
    recommended_price: float
    expected_units: float
    expected_profit: float
    profit_at_current_price: float
    price_floor: float
    price_ceiling: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_elasticity(history: pd.DataFrame) -> float:
    """Log-log regression of sales on price, with day-of-week dummies as controls."""
    df = history[history["sales"] > 0]
    dow = pd.get_dummies(pd.to_datetime(df["date"]).dt.day_of_week, drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(df)), np.log(df["our_price"].to_numpy()), dow.to_numpy(),
                         df["is_holiday"].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(x, np.log(df["sales"].to_numpy()), rcond=None)
    return float(coef[1])


def recommend_price(forecast_units: float, current_price: float, unit_cost: float, inventory: float,
                    competitor_price: float, elasticity: float, min_margin: float = 0.10,
                    max_above_competitor: float = 0.15, max_change: float = 0.20,
                    steps: int = 201) -> PriceDecision:
    """Pick the price with the highest expected profit inside the allowed corridor.

    forecast_units is the forecast demand for the horizon at the current price.
    """
    if forecast_units < 0 or inventory < 0 or current_price <= 0:
        raise ValueError("forecast_units and inventory must be >= 0, current_price > 0")
    floor = max(unit_cost * (1 + min_margin), current_price * (1 - max_change))
    ceiling = min(competitor_price * (1 + max_above_competitor), current_price * (1 + max_change))
    ceiling = max(ceiling, floor)  # rules conflict: the margin floor wins

    prices = np.unique(np.append(np.linspace(floor, ceiling, steps),
                                 np.clip(current_price, floor, ceiling)))
    demand = forecast_units * (prices / current_price) ** elasticity
    units = np.minimum(demand, inventory)
    profit = (prices - unit_cost) * units
    best = int(np.argmax(profit))
    price = float(prices[best])

    current_units = min(forecast_units, inventory)
    current_profit = (current_price - unit_cost) * current_units
    if forecast_units > inventory * 1.1:
        why = f"forecast demand ({forecast_units:.0f}) exceeds stock ({inventory:.0f})"
    elif forecast_units < inventory * 0.5:
        why = f"stock ({inventory:.0f}) is well above forecast demand ({forecast_units:.0f})"
    else:
        why = "demand and stock are balanced"
    direction = "raise" if price > current_price + 0.005 else "lower" if price < current_price - 0.005 else "keep"
    reason = (f"{direction.capitalize()} price: {why}; elasticity {elasticity:.2f}; "
              f"allowed range {floor:.2f}-{ceiling:.2f}.")
    return PriceDecision(current_price, round(price, 2), round(float(units[best]), 1),
                         round(float(profit[best]), 2), round(current_profit, 2),
                         round(floor, 2), round(ceiling, 2), reason)
