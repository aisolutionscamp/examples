"""Real-time scoring: features from the store, XGBoost score, decision, reasons."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import xgboost as xgb

from .features import FEATURE_NAMES, FeatureStore, Transaction, features_for


@dataclass
class Decision:
    transaction_id: str
    score: float
    action: str  # "approve" | "review" | "block"
    reasons: list[str]
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


class Scorer:
    def __init__(self, model: xgb.XGBClassifier, store: FeatureStore,
                 review_threshold: float, block_threshold: float = 0.9) -> None:
        if not 0 <= review_threshold <= block_threshold <= 1:
            raise ValueError("expected 0 <= review_threshold <= block_threshold <= 1")
        self.booster = model.get_booster()
        self.booster.set_param({"nthread": 1})  # one row at a time: threads only add overhead
        self.store = store
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold

    def score(self, txn: Transaction) -> Decision:
        started = time.perf_counter()
        row = np.asarray([features_for(txn, self.store)], dtype=np.float32)
        matrix = xgb.DMatrix(row, feature_names=FEATURE_NAMES)
        score = float(self.booster.predict(matrix)[0])
        # SHAP-style contributions: which features pushed the score up the most.
        contribs = self.booster.predict(matrix, pred_contribs=True)[0][:-1]
        top = np.argsort(contribs)[::-1][:3]
        reasons = [FEATURE_NAMES[i] for i in top if contribs[i] > 0]

        if score >= self.block_threshold:
            action = "block"
        elif score >= self.review_threshold:
            action = "review"
        else:
            action = "approve"
        # Record after scoring, so the transaction never sees itself in its features.
        self.store.record(txn)
        return Decision(txn.transaction_id, round(score, 4), action, reasons,
                        round((time.perf_counter() - started) * 1000, 3))
