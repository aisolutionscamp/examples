"""Optional adapters: Redis forecast cache and Kafka sales-event consumer.

The functions take client objects, so tests can pass fakes and the
forecasting core does not depend on either service.
"""
import json
import os
from collections.abc import Callable, Iterable

import pandas as pd

REQUIRED_EVENT_KEYS = {"sku", "store_id", "date", "units", "price"}


def forecast_key(sku: str, store_id: str) -> str:
    return f"forecast:{store_id}:{sku}"


def cache_forecast(redis_client, sku: str, store_id: str, forecast: pd.DataFrame,
                   ttl_seconds: int = 4 * 3600) -> str:
    """Store the forecast as JSON with a TTL that matches the refresh interval."""
    records = forecast.assign(date=forecast["date"].astype(str)).to_dict(orient="records")
    key = forecast_key(sku, store_id)
    redis_client.setex(key, ttl_seconds, json.dumps(records))
    return key


def get_cached_forecast(redis_client, sku: str, store_id: str) -> pd.DataFrame | None:
    raw = redis_client.get(forecast_key(sku, store_id))
    if raw is None:
        return None
    df = pd.DataFrame(json.loads(raw))
    df["date"] = pd.to_datetime(df["date"])
    return df


def consume_sales_events(messages: Iterable, on_event: Callable[[dict], None],
                         max_events: int | None = None) -> dict:
    """Parse sales events (Kafka messages with a JSON value) and hand valid ones to on_event."""
    stats = {"processed": 0, "rejected": 0}
    for message in messages:
        try:
            event = json.loads(message.value)
            if not REQUIRED_EVENT_KEYS <= event.keys():
                raise ValueError("missing keys")
        except (ValueError, TypeError):
            stats["rejected"] += 1
        else:
            on_event(event)
            stats["processed"] += 1
        if max_events is not None and stats["processed"] + stats["rejected"] >= max_events:
            break
    return stats


def make_redis():  # pragma: no cover - needs a running Redis
    import redis
    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def make_consumer():  # pragma: no cover - needs a running Kafka broker
    from kafka import KafkaConsumer
    return KafkaConsumer(os.environ.get("KAFKA_TOPIC", "sales_events"),
                         bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(","),
                         group_id="demand-forecaster", auto_offset_reset="earliest")
