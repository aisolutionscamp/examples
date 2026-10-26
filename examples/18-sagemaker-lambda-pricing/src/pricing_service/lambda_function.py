"""AWS Lambda handler: price recommendation from a SageMaker demand endpoint.

For each candidate price in the allowed corridor, the endpoint predicts the
units sold for the requested day (one CSV row per candidate, one request).
Sales are capped by stock and the price with the highest expected profit is
returned. Works behind API Gateway (proxy event with a JSON body) or with a
direct invocation.
"""
import json
import logging
import os
from datetime import date

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

from .features import feature_row, to_csv_rows

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REQUIRED = {"sku": str, "store_id": str, "date": str, "current_price": (int, float),
            "unit_cost": (int, float), "inventory": (int, float), "competitor_price": (int, float)}
_clients: dict = {}


def client(name: str):
    if name not in _clients:  # reuse clients across warm invocations
        _clients[name] = boto3.client(name)
    return _clients[name]


def parse_event(event: dict) -> dict:
    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(body)
    for key, kind in REQUIRED.items():
        if not isinstance(body.get(key), kind) or isinstance(body.get(key), bool):
            raise TypeError(f"field '{key}' is missing or has the wrong type")
    if body["current_price"] <= 0 or body["competitor_price"] <= 0 or body["inventory"] < 0:
        raise ValueError("prices must be > 0 and inventory >= 0")
    body["day"] = date.fromisoformat(body["date"])
    return body


def price_corridor(req: dict, min_margin=0.10, max_above_competitor=0.15, max_change=0.20, steps=21):
    floor = max(req["unit_cost"] * (1 + min_margin), req["current_price"] * (1 - max_change))
    ceiling = max(floor, min(req["competitor_price"] * (1 + max_above_competitor),
                             req["current_price"] * (1 + max_change)))
    return np.round(np.linspace(floor, ceiling, steps), 2)


def predict_units(endpoint: str, req: dict, prices: np.ndarray) -> np.ndarray:
    rows = [feature_row(req["sku"], req["store_id"], req["day"], req.get("is_holiday", False), p,
                        req["competitor_price"]) for p in prices]
    resp = client("sagemaker-runtime").invoke_endpoint(
        EndpointName=endpoint, ContentType="text/csv", Accept="text/csv", Body=to_csv_rows(rows))
    lines = resp["Body"].read().decode().strip().splitlines()
    units = np.clip(np.array([float(x.split(",")[0]) for x in lines]), 0, None)
    if len(units) != len(prices):
        raise RuntimeError(f"endpoint returned {len(units)} predictions for {len(prices)} rows")
    # demand must not rise with price; flatten model noise that says otherwise
    return np.minimum.accumulate(units)


def choose_price(req: dict, prices: np.ndarray, units: np.ndarray) -> dict:
    sold = np.minimum(units, req["inventory"])
    profit = (prices - req["unit_cost"]) * sold
    best = int(np.argmax(profit))
    return {"sku": req["sku"], "store_id": req["store_id"], "date": req["date"],
            "current_price": req["current_price"], "recommended_price": float(prices[best]),
            "predicted_units": round(float(units[best]), 1), "expected_profit": round(float(profit[best]), 2),
            "price_range": [float(prices[0]), float(prices[-1])],
            "stock_limited": bool(units[best] > req["inventory"])}


def lambda_handler(event, context):
    try:
        req = parse_event(event)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"statusCode": 400, "body": json.dumps({"error": str(exc)})}

    endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "demand-forecast")
    prices = price_corridor(req)
    try:
        units = predict_units(endpoint, req, prices)
    except (ClientError, BotoCoreError, RuntimeError, ValueError) as exc:
        logger.error("forecast failed: %s", exc)
        return {"statusCode": 502, "body": json.dumps({"error": "demand forecast unavailable"})}

    decision = choose_price(req, prices, units)
    logger.info(json.dumps({"event": "price_decision", **decision}))
    change_pct = 100 * (decision["recommended_price"] / req["current_price"] - 1)
    try:
        client("cloudwatch").put_metric_data(
            Namespace=os.environ.get("METRICS_NAMESPACE", "Example/DynamicPricing"),
            MetricData=[{"MetricName": "PriceChangePercent", "Value": change_pct, "Unit": "Percent",
                         "Dimensions": [{"Name": "StoreId", "Value": req["store_id"]}]}])
    except (ClientError, BotoCoreError) as exc:  # metrics must never block a decision
        logger.warning("metric not sent: %s", exc)
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(decision)}
