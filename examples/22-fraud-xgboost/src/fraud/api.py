"""HTTP scoring API.

    FRAUD_MODEL_PATH=model/fraud_model.json uvicorn fraud.api:app --port 8000
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import InMemoryFeatureStore, Transaction
from .scoring import Scorer


class TransactionIn(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    country: str = Field(min_length=2, max_length=2)
    device_id: str
    merchant_category: str
    timestamp: float | None = None


def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="Fraud scoring")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/score")
    def score(body: TransactionIn) -> dict:
        data = body.model_dump()
        data["timestamp"] = data["timestamp"] or time.time()
        return scorer.score(Transaction(**data)).as_dict()

    return app


def _from_env() -> FastAPI:
    from .model import load

    clf, trained_threshold = load(os.environ.get("FRAUD_MODEL_PATH", "model/fraud_model.json"))
    review = float(os.environ.get("FRAUD_REVIEW_THRESHOLD") or trained_threshold)
    block = float(os.environ.get("FRAUD_BLOCK_THRESHOLD") or 0.9)
    store = InMemoryFeatureStore()
    if os.environ.get("REDIS_URL"):
        from .redis_store import RedisFeatureStore

        store = RedisFeatureStore.from_url(os.environ["REDIS_URL"])
    return create_app(Scorer(clf, store, min(review, block), block))


def __getattr__(name: str):  # lazy, so importing the module needs no model file
    if name == "app":
        return _from_env()
    raise AttributeError(name)
