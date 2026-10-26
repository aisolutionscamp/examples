"""Export the trained patch model to ONNX and score with onnxruntime.

Feature extraction stays in OpenCV/NumPy; only the scaler + IsolationForest go to ONNX.
That keeps the runtime small (onnxruntime instead of scikit-learn) on an edge PC.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .detector import PatchAnomalyDetector
from .features import patch_features, preprocess


def export_onnx(detector: PatchAnomalyDetector, path: str) -> None:
    n_features = detector.model.named_steps["standardscaler"].n_features_in_
    onx = convert_sklearn(detector.model, initial_types=[("patches", FloatTensorType([None, n_features]))],
                          target_opset={"": 17, "ai.onnx.ml": 3})
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())


class OnnxScorer:
    def __init__(self, path: str, threshold: float):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.threshold = threshold

    def patch_scores(self, image: np.ndarray) -> np.ndarray:
        feats = patch_features(preprocess(image))
        _labels, scores = self.session.run(None, {"patches": feats})
        return -np.asarray(scores).ravel()  # same sign convention as PatchAnomalyDetector

    def predict(self, image: np.ndarray) -> dict:
        score = float(self.patch_scores(image).max())
        return {"score": score, "threshold": self.threshold, "defect": score >= self.threshold}
