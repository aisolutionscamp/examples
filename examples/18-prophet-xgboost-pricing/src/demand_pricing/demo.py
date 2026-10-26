"""Train on synthetic history, forecast the next 7 days and recommend a price.

    python -m demand_pricing.demo
"""
import json

import pandas as pd

from .data import make_sales
from .ensemble import EnsembleForecaster
from .pricing import estimate_elasticity, recommend_price


def run(horizon: int = 7, seed: int = 0) -> dict:
    data = make_sales(days=800 + horizon, seed=seed)
    history, future = data.iloc[:-horizon], data.iloc[-horizon:]

    forecaster = EnsembleForecaster().fit(history)
    # Future rows: planned price, competitor price and weather forecast are known inputs
    forecast = forecaster.predict(future.drop(columns=["sales"]))

    elasticity = estimate_elasticity(history)
    decision = recommend_price(
        forecast_units=float(forecast["forecast"].sum()),
        current_price=float(history["our_price"].iloc[-1]),
        unit_cost=6.0,
        inventory=250,
        competitor_price=float(future["competitor_price"].mean()),
        elasticity=elasticity,
    )
    return {
        "validation": forecaster.validation_,
        "prophet_weight": round(forecaster.prophet_weight, 3),
        "forecast": forecast.assign(date=forecast["date"].dt.date.astype(str)).round(1).to_dict("records"),
        "actual_sales": future["sales"].tolist(),
        "pricing": decision.to_dict(),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print(json.dumps(run(), indent=2, default=float))
