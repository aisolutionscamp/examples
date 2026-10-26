"""Point-in-time feature engineering for card-not-present transactions.

Features are computed only from what was known *before* the transaction
(recent transactions of the same user and a running user profile), so the
same code is used for training (replaying history) and for live scoring.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Protocol

HIGH_RISK_CATEGORIES = {"electronics", "gift_cards", "crypto", "money_transfer"}
WINDOW_SECONDS = 24 * 3600

FEATURE_NAMES = [
    "amount_log",
    "amount_to_user_mean",
    "txn_count_10m",
    "txn_count_1h",
    "amount_sum_1h_log",
    "foreign_country",
    "distinct_countries_24h",
    "new_device",
    "seconds_since_last_log",
    "hour_of_day",
    "user_history_len_log",
    "high_risk_category",
]


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    country: str
    device_id: str
    merchant_category: str
    timestamp: float  # unix seconds


@dataclass
class UserProfile:
    home_country: str
    txn_count: int = 0
    amount_mean: float = 0.0
    known_devices: set[str] = field(default_factory=set)

    def update(self, txn: Transaction) -> None:
        self.txn_count += 1
        self.amount_mean += (txn.amount - self.amount_mean) / self.txn_count
        self.known_devices.add(txn.device_id)


class FeatureStore(Protocol):
    def recent(self, user_id: str, since: float) -> list[Transaction]: ...
    def profile(self, user_id: str) -> UserProfile | None: ...
    def record(self, txn: Transaction) -> None: ...


class InMemoryFeatureStore:
    """Feature store for tests, training replays and single-process demos."""

    def __init__(self, max_per_user: int = 200) -> None:
        self._txns: dict[str, deque[Transaction]] = defaultdict(lambda: deque(maxlen=max_per_user))
        self._profiles: dict[str, UserProfile] = {}

    def recent(self, user_id: str, since: float) -> list[Transaction]:
        return [t for t in self._txns.get(user_id, ()) if t.timestamp >= since]

    def profile(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def record(self, txn: Transaction) -> None:
        self._txns[txn.user_id].append(txn)
        profile = self._profiles.setdefault(txn.user_id, UserProfile(home_country=txn.country))
        profile.update(txn)


def compute_features(txn: Transaction, recent: list[Transaction], profile: UserProfile | None) -> list[float]:
    """Return the feature vector in FEATURE_NAMES order."""
    prior = [t for t in recent if t.timestamp < txn.timestamp]
    last_10m = [t for t in prior if txn.timestamp - t.timestamp <= 600]
    last_1h = [t for t in prior if txn.timestamp - t.timestamp <= 3600]
    countries_24h = {t.country for t in prior} | {txn.country}

    if profile is None or profile.txn_count == 0:
        amount_ratio, foreign, new_device, history = 1.0, 0.0, 1.0, 0
    else:
        amount_ratio = txn.amount / max(profile.amount_mean, 1.0)
        foreign = float(txn.country != profile.home_country)
        new_device = float(txn.device_id not in profile.known_devices)
        history = profile.txn_count

    since_last = txn.timestamp - max(t.timestamp for t in prior) if prior else WINDOW_SECONDS
    hour = (txn.timestamp % 86400) / 3600

    return [
        math.log1p(txn.amount),
        amount_ratio,
        float(len(last_10m)),
        float(len(last_1h)),
        math.log1p(sum(t.amount for t in last_1h)),
        foreign,
        float(len(countries_24h)),
        new_device,
        math.log1p(since_last),
        hour,
        math.log1p(history),
        float(txn.merchant_category in HIGH_RISK_CATEGORIES),
    ]


def features_for(txn: Transaction, store: FeatureStore) -> list[float]:
    return compute_features(txn, store.recent(txn.user_id, txn.timestamp - WINDOW_SECONDS),
                            store.profile(txn.user_id))
