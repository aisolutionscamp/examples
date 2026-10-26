"""Synthetic transaction stream with labeled fraud, for tests and the demo.

Legitimate behavior: each user shops a few times per week in the home
country, on one or two devices, with amounts around a personal mean. Some
legitimate users travel or make one large purchase (so the task is not trivial).
Fraud: an account takeover produces a short burst of high-value purchases from
a new device, often abroad and in high-risk categories.
"""

from __future__ import annotations

import numpy as np

from .features import Transaction

COUNTRIES = ["DE", "FR", "NL", "US", "GB", "BR", "NG", "IN"]
CATEGORIES = ["grocery", "fashion", "restaurants", "travel", "electronics", "gift_cards", "books"]
DAY = 86400.0


def generate(n_users: int = 300, days: int = 30, fraud_user_share: float = 0.05,
             seed: int = 7) -> tuple[list[Transaction], list[int]]:
    """Return transactions sorted by time and their labels (1 = fraud)."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[Transaction, int]] = []
    start = 1_767_225_600.0  # 2026-01-01T00:00:00Z
    for u in range(n_users):
        user = f"u{u:04d}"
        home = COUNTRIES[int(rng.integers(0, 4))]
        mean = float(rng.lognormal(3.5, 0.5))
        devices = [f"{user}-d{i}" for i in range(int(rng.integers(1, 3)))]
        n = int(rng.poisson(days * 0.6)) + 3
        for i, ts in enumerate(np.sort(rng.uniform(0, days * DAY, n))):
            country = home
            if rng.random() < 0.03:  # legitimate travel
                country = COUNTRIES[int(rng.integers(0, len(COUNTRIES)))]
            amount = float(rng.lognormal(np.log(mean), 0.4))
            if rng.random() < 0.02:  # legitimate large purchase
                amount *= float(rng.uniform(4, 10))
            category = CATEGORIES[int(rng.integers(0, 4))] if rng.random() < 0.9 else "electronics"
            rows.append((Transaction(f"{user}-t{i}", user, round(amount, 2), country,
                                     devices[int(rng.integers(0, len(devices)))], category,
                                     start + float(ts)), 0))
        if rng.random() < fraud_user_share:
            t0 = start + float(rng.uniform(days * DAY * 0.2, days * DAY))
            country = COUNTRIES[int(rng.integers(3, len(COUNTRIES)))] if rng.random() < 0.8 else home
            device = f"{user}-stolen"
            for j in range(int(rng.integers(2, 7))):
                category = ["electronics", "gift_cards", "fashion"][int(rng.integers(0, 3))]
                amount = float(mean * rng.uniform(3, 15))
                rows.append((Transaction(f"{user}-f{j}", user, round(amount, 2), country, device,
                                         category, t0 + j * float(rng.uniform(30, 240))), 1))
    rows.sort(key=lambda r: r[0].timestamp)
    return [r[0] for r in rows], [r[1] for r in rows]
