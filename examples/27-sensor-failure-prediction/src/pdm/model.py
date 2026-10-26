"""Train and apply the three models: failure classifier, RUL regressor, anomaly detector."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .data import add_failure_label
from .features import feature_columns, rolling_features


@dataclass
class MaintenanceModel:
    sensors: list[str]
    window: int
    horizon: int
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    detector: IsolationForest
    anomaly_threshold: float  # 99th percentile of anomaly scores on healthy training windows
    alert_probability: float = 0.5
    metrics: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return feature_columns(self.sensors)

    def score(self, readings: pd.DataFrame) -> pd.DataFrame:
        """Score every cycle that has a full window of history."""
        X = rolling_features(readings, self.sensors, self.window)
        out = X[["machine_id", "cycle"]].copy()
        feats = X[self.features]
        out["failure_probability"] = self.classifier.predict_proba(feats)[:, 1]
        out["rul_estimate"] = np.clip(self.regressor.predict(feats), 0, None)
        out["anomaly_score"] = -self.detector.score_samples(feats)  # higher = more unusual
        out["anomaly"] = out["anomaly_score"] > self.anomaly_threshold
        out["alert"] = (out["failure_probability"] >= self.alert_probability) | out["anomaly"]
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> MaintenanceModel:
        return joblib.load(path)


def train(df: pd.DataFrame, sensors: list[str], window: int = 10, horizon: int = 30,
          healthy_rul: int = 100, test_size: float = 0.25, seed: int = 0) -> MaintenanceModel:
    """Train on run-to-failure data with a `rul` column.

    The split is by machine, so test machines are never seen during training.
    The anomaly detector only sees windows far from failure (RUL > healthy_rul).
    """
    data = add_failure_label(rolling_features(df, sensors, window), horizon)
    cols = feature_columns(sensors)
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(split.split(data, groups=data["machine_id"]))
    tr, te = data.iloc[train_idx], data.iloc[test_idx]

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    clf.fit(tr[cols], tr["fails_soon"])
    # RUL is capped: far from failure the sensors carry little information.
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
    reg.fit(tr[cols], tr["rul"].clip(upper=150))

    healthy = tr[tr["rul"] > healthy_rul][cols]
    det = IsolationForest(n_estimators=200, random_state=seed).fit(healthy)
    threshold = float(np.quantile(-det.score_samples(healthy), 0.99))

    proba = clf.predict_proba(te[cols])[:, 1]
    metrics = {
        "test_machines": int(te["machine_id"].nunique()),
        "roc_auc": float(roc_auc_score(te["fails_soon"], proba)),
        "average_precision": float(average_precision_score(te["fails_soon"], proba)),
        "rul_mae_capped": float(mean_absolute_error(te["rul"].clip(upper=150), reg.predict(te[cols]))),
    }
    return MaintenanceModel(sensors, window, horizon, clf, reg, det, threshold, metrics=metrics)
