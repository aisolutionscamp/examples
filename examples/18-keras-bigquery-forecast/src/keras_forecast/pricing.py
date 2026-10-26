"""Pick a price for the forecast horizon: highest expected profit inside a corridor."""
import numpy as np


def recommend_price(forecast_units: float, current_price: float, unit_cost: float, inventory: float,
                    competitor_price: float, elasticity: float = -1.5, min_margin: float = 0.10,
                    max_above_competitor: float = 0.15, max_change: float = 0.20) -> dict:
    floor = max(unit_cost * (1 + min_margin), current_price * (1 - max_change))
    ceiling = max(floor, min(competitor_price * (1 + max_above_competitor), current_price * (1 + max_change)))
    prices = np.linspace(floor, ceiling, 101)
    units = np.minimum(forecast_units * (prices / current_price) ** elasticity, inventory)
    profit = (prices - unit_cost) * units
    best = int(np.argmax(profit))
    return {"recommended_price": round(float(prices[best]), 2), "expected_units": round(float(units[best]), 1),
            "expected_profit": round(float(profit[best]), 2), "price_range": [round(floor, 2), round(ceiling, 2)],
            "elasticity": elasticity}
