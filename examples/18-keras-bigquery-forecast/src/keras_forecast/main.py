"""HTTP function (Cloud Run functions / functions-framework) that serves a 7-day forecast.

    functions-framework --source src/keras_forecast/main.py --target forecast_http
Request: {"sku": "SKU-1", "store_id": "S01"} plus optional pricing inputs
{"current_price", "unit_cost", "inventory", "competitor_price", "elasticity"}.
"""
import os
from datetime import date, timedelta

import functions_framework
from google.cloud import bigquery

from keras_forecast.bigquery_io import load_daily_sales, write_forecast
from keras_forecast.model import WINDOW, Forecaster
from keras_forecast.pricing import recommend_price

_state: dict = {}
PRICING_KEYS = ("current_price", "unit_cost", "inventory", "competitor_price")


def _get(name, factory):
    if name not in _state:  # loaded once per instance, reused across requests
        _state[name] = factory()
    return _state[name]


def handle(payload: dict, today: date | None = None) -> tuple[dict, int]:
    sku, store_id = payload.get("sku"), payload.get("store_id")
    if not isinstance(sku, str) or not isinstance(store_id, str) or not sku or not store_id:
        return {"error": "sku and store_id are required strings"}, 400
    client = _get("bq", bigquery.Client)
    forecaster = _get("model", lambda: Forecaster.load(os.environ.get("MODEL_DIR", "models/latest")))
    dataset = os.environ["BQ_DATASET"]

    end = (today or date.today()) - timedelta(days=1)
    recent = load_daily_sales(client, dataset, sku, store_id, end - timedelta(days=WINDOW - 1), end)
    forecast = forecaster.predict(recent)
    version = os.environ.get("MODEL_VERSION", "local")
    write_forecast(client, dataset, sku, store_id, forecast, version)

    body = {"sku": sku, "store_id": store_id, "model_version": version,
            "forecast": [{"date": str(r.date.date()), "units": round(float(r.units), 1),
                          "lower": round(float(r.lower), 1), "upper": round(float(r.upper), 1)}
                         for r in forecast.itertuples()]}
    if all(isinstance(payload.get(k), (int, float)) for k in PRICING_KEYS):
        body["pricing"] = recommend_price(float(forecast["units"].sum()),
                                          *(float(payload[k]) for k in PRICING_KEYS),
                                          elasticity=float(payload.get("elasticity", -1.5)))
    return body, 200


@functions_framework.http
def forecast_http(request):
    body, status = handle(request.get_json(silent=True) or {})
    return body, status
