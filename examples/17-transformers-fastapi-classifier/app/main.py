"""FastAPI microservice: POST /classify with Redis caching and simple in-process metrics."""

import hashlib
import json
import os
import time
from collections import Counter
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .classifier import EmailClassifier

CACHE_TTL_SECONDS = 3600


class EmailIn(BaseModel):
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=20000)


class Metrics:
    """Counters for /metrics. Per process; use Prometheus in production."""

    def __init__(self):
        self.requests = 0
        self.cache_hits = 0
        self.needs_review = 0
        self.total_ms = 0.0
        self.urgency = Counter()
        self.department = Counter()

    def record(self, result: dict, ms: float, cached: bool):
        self.requests += 1
        self.cache_hits += cached
        self.needs_review += result["needs_review"]
        self.total_ms += ms
        self.urgency[result["urgency"]] += 1
        self.department[result["department"]] += 1

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "needs_review": self.needs_review,
            "avg_latency_ms": round(self.total_ms / self.requests, 2) if self.requests else 0,
            "urgency": dict(self.urgency),
            "department": dict(self.department),
        }


def make_cache():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    return redis.Redis.from_url(url, socket_timeout=0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set app.state.classifier / app.state.cache before startup; otherwise load from env.
    if getattr(app.state, "classifier", None) is None:
        app.state.classifier = EmailClassifier(
            os.getenv("URGENCY_MODEL_DIR", "models/urgency"),
            os.getenv("DEPARTMENT_MODEL_DIR", "models/department"),
            float(os.getenv("MIN_CONFIDENCE", "0.6")),
        )
    if not hasattr(app.state, "cache"):
        app.state.cache = make_cache()
    app.state.metrics = Metrics()
    yield


app = FastAPI(title="Email Classification API", version="1.0.0", lifespan=lifespan)


def cache_key(email: EmailIn) -> str:
    digest = hashlib.sha256(f"{email.subject}\x00{email.body}".encode()).hexdigest()
    return f"email-classification:v1:{digest}"


@app.post("/classify")
def classify(email: EmailIn, request: Request) -> dict:
    state = request.app.state
    start = time.perf_counter()
    key, result, cached = cache_key(email), None, False
    if state.cache is not None:
        try:
            raw = state.cache.get(key)
            if raw:
                result, cached = json.loads(raw), True
        except (redis.RedisError, ValueError):  # a broken cache must not break classification
            pass
    if result is None:
        result = state.classifier.classify(email.subject, email.body)
        if state.cache is not None:
            try:
                state.cache.set(key, json.dumps(result), ex=CACHE_TTL_SECONDS)
            except redis.RedisError:
                pass
    state.metrics.record(result, (time.perf_counter() - start) * 1000, cached)
    return result | {"cached": cached}


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "urgency_labels": request.app.state.classifier.urgency.labels,
            "department_labels": request.app.state.classifier.department.labels}


@app.get("/metrics")
def metrics(request: Request) -> dict:
    return request.app.state.metrics.as_dict()
