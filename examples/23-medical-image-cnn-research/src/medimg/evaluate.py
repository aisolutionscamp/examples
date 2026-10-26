"""Evaluation metrics and a Grad-CAM heatmap for inspecting what the model looks at."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    pred = (np.asarray(prob).ravel() >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "threshold": threshold,
    }


def grad_cam(model: keras.Model, image: np.ndarray, layer_name: str = "last_conv") -> np.ndarray:
    """Heatmap in [0, 1] with the image's height/width (for the small_cnn model)."""
    conv = model.get_layer(layer_name)
    grad_model = keras.Model(model.inputs, [conv.output, model.output])
    x = tf.convert_to_tensor(image[None, ..., None] if image.ndim == 2 else image[None], tf.float32)
    with tf.GradientTape() as tape:
        features, prob = grad_model(x, training=False)
        score = prob[:, 0]
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(features * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2])[..., 0]
    return cam.numpy()
