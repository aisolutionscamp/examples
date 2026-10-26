"""Redis-backed feature store: shared state for several scoring workers.

Per user: a sorted set of recent transactions (score = timestamp) and a hash
with the running profile. Requires `pip install -r requirements-adapters.txt`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .features import WINDOW_SECONDS, Transaction, UserProfile


class RedisFeatureStore:
    def __init__(self, client, ttl_seconds: int = 7 * 86400) -> None:
        self.r = client
        self.ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str) -> RedisFeatureStore:
        import redis

        return cls(redis.Redis.from_url(url, decode_responses=True))

    def recent(self, user_id: str, since: float) -> list[Transaction]:
        raw = self.r.zrangebyscore(f"txns:{user_id}", since, "+inf")
        return [Transaction(**json.loads(item)) for item in raw]

    def profile(self, user_id: str) -> UserProfile | None:
        data = self.r.hgetall(f"profile:{user_id}")
        if not data:
            return None
        return UserProfile(home_country=data["home_country"], txn_count=int(data["txn_count"]),
                           amount_mean=float(data["amount_mean"]),
                           known_devices=set(json.loads(data["known_devices"])))

    def record(self, txn: Transaction) -> None:
        key = f"txns:{txn.user_id}"
        profile = self.profile(txn.user_id) or UserProfile(home_country=txn.country)
        profile.update(txn)
        pipe = self.r.pipeline()
        pipe.zadd(key, {json.dumps(asdict(txn)): txn.timestamp})
        pipe.zremrangebyscore(key, "-inf", txn.timestamp - WINDOW_SECONDS)
        pipe.expire(key, self.ttl)
        pipe.hset(f"profile:{txn.user_id}", mapping={
            "home_country": profile.home_country,
            "txn_count": profile.txn_count,
            "amount_mean": profile.amount_mean,
            "known_devices": json.dumps(sorted(profile.known_devices)),
        })
        pipe.execute()
