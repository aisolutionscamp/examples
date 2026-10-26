"""Training, threshold selection and evaluation of the XGBoost fraud model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURE_NAMES, InMemoryFeatureStore, Transaction, features_for


def build_dataset(transactions: list[Transaction]) -> np.ndarray:
    """Replay transactions in time order so every row only sees its past."""
    store = InMemoryFeatureStore()
    rows = []
    for txn in transactions:
        rows.append(features_for(txn, store))
        store.record(txn)
    return np.asarray(rows, dtype=np.float32)


def train(x: np.ndarray, y: np.ndarray, seed: int = 0, n_jobs: int = 2) -> xgb.XGBClassifier:
    positives = max(int(y.sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(len(y) - positives) / positives,  # class imbalance
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,  # explicit: "all cores" thrashes on busy or shared machines
    )
    model.fit(x, y)
    model.get_booster().feature_names = FEATURE_NAMES
    return model


def threshold_for_fpr(y: np.ndarray, proba: np.ndarray, max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y, proba) is <= max_fpr."""
    negatives = np.sort(proba[y == 0])[::-1]
    allowed = int(np.floor(max_fpr * len(negatives)))
    if allowed >= len(negatives):
        return 0.0
    return float(np.nextafter(negatives[allowed], 1.0))


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    false_positive_rate: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float) -> Metrics:
    flagged = proba >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    return Metrics(
        roc_auc=float(roc_auc_score(y, proba)),
        pr_auc=float(average_precision_score(y, proba)),
        precision=tp / max(int(flagged.sum()), 1),
        recall=tp / max(int((y == 1).sum()), 1),
        false_positive_rate=fp / max(int((y == 0).sum()), 1),
        threshold=threshold,
    )


def save(model: xgb.XGBClassifier, path: str | Path, threshold: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    path.with_suffix(".meta.json").write_text(json.dumps({"review_threshold": threshold,
                                                          "features": FEATURE_NAMES}))


def load(path: str | Path) -> tuple[xgb.XGBClassifier, float]:
    path = Path(path)
    model = xgb.XGBClassifier()
    model.load_model(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta["features"] != FEATURE_NAMES:
        raise ValueError("model was trained with a different feature set")
    return model, float(meta["review_threshold"])
