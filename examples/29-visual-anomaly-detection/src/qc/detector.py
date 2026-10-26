"""Patch-level anomaly detector: StandardScaler + IsolationForest trained on good images only.

Image score = highest patch score, so one small defect is enough to flag a part.
The patch scores also form a coarse heatmap that shows where the defect is.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import IMAGE_SIZE, PATCH, STRIDE, patch_features, patch_grid, preprocess


@dataclass
class Threshold:
    value: float
    false_reject_rate: float  # share of good parts above the threshold (on calibration data)
    defect_recall: float | None  # share of defective parts above it, if defects were given


class PatchAnomalyDetector:
    def __init__(self, n_estimators: int = 200, max_samples: int = 1024, seed: int = 0):
        # More samples per tree = deeper trees = a wider score range for strong outliers.
        self.model: Pipeline = make_pipeline(StandardScaler(), IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed))
        self.threshold: Threshold | None = None

    # --- training -------------------------------------------------------------
    def fit(self, good_images: list[np.ndarray]) -> PatchAnomalyDetector:
        X = np.vstack([patch_features(preprocess(img)) for img in good_images])
        self.model.fit(X)
        return self

    def calibrate(self, good_images: list[np.ndarray], defect_images: list[np.ndarray] | None = None,
                  max_false_reject: float = 0.01) -> Threshold:
        """Pick the decision threshold on images that were NOT used for training.

        With good images only: the (1 - max_false_reject) quantile of their scores.
        With labeled defects too: the lowest threshold whose false-reject rate stays within
        the limit, which maximises defect recall under that constraint.
        """
        good = np.array([self.score(img) for img in good_images])
        recall = None
        if not defect_images:
            value = float(np.quantile(good, 1 - max_false_reject))
        else:
            bad = np.array([self.score(img) for img in defect_images])
            y = np.r_[np.zeros(len(good)), np.ones(len(bad))]
            fpr, tpr, thr = roc_curve(y, np.r_[good, bad])  # rule: defect if score >= thr
            ok = np.where((fpr <= max_false_reject) & np.isfinite(thr))[0]
            value = float(thr[ok[np.argmax(tpr[ok])]]) if len(ok) else float(np.nextafter(good.max(), np.inf))
            recall = float((bad >= value).mean())
        self.threshold = Threshold(value, float((good >= value).mean()), recall)
        return self.threshold

    # --- inference ------------------------------------------------------------
    def patch_scores(self, image: np.ndarray) -> np.ndarray:
        """Anomaly score per patch, higher = more unusual (negated IsolationForest decision)."""
        return -self.model.decision_function(patch_features(preprocess(image)))

    def score(self, image: np.ndarray) -> float:
        return float(self.patch_scores(image).max())

    def heatmap(self, image: np.ndarray) -> np.ndarray:
        """Patch scores as a 2-D grid (rows x cols) in image layout."""
        n = int(np.sqrt(len(patch_grid())))
        return self.patch_scores(image).reshape(n, n)

    def predict(self, image: np.ndarray) -> dict:
        if self.threshold is None:
            raise RuntimeError("call calibrate() first")
        scores = self.patch_scores(image)
        y, x = patch_grid()[int(scores.argmax())]
        return {"score": float(scores.max()), "threshold": self.threshold.value,
                "defect": bool(scores.max() >= self.threshold.value),
                # location of the most unusual patch, in pixels of the resized image
                "box": {"x": x, "y": y, "w": PATCH, "h": PATCH, "image_size": IMAGE_SIZE},
                "stride": STRIDE}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> PatchAnomalyDetector:
        return joblib.load(path)
