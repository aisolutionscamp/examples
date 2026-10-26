"""Hand-crafted patch features with OpenCV.

The image is converted to grayscale, resized to a fixed size and cut into overlapping
patches. Each patch is described by brightness, contrast, sharpness and edge
statistics. A defect (scratch, dent, stain) changes these numbers locally.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 256
PATCH = 32
STRIDE = 16


def preprocess(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """BGR or gray uint8 image -> gray float32 in [0, 1], size x size, median shifted to 0.5."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # Remove global brightness differences between shots (lighting, exposure).
    return np.clip(image - np.median(image) + 0.5, 0.0, 1.0)


def patch_grid(size: int = IMAGE_SIZE, patch: int = PATCH, stride: int = STRIDE) -> list[tuple[int, int]]:
    starts = range(0, size - patch + 1, stride)
    return [(y, x) for y in starts for x in starts]


def patch_features(gray: np.ndarray, patch: int = PATCH, stride: int = STRIDE) -> np.ndarray:
    """Return an array of shape (n_patches, n_features) for a preprocessed image."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0

    rows = []
    for y, x in patch_grid(gray.shape[0], patch, stride):
        p = gray[y:y + patch, x:x + patch]
        g = grad[y:y + patch, x:x + patch]
        lo, hi = np.percentile(p, [2, 98])
        rows.append([p.mean(), p.std(), lo, hi, lap[y:y + patch, x:x + patch].var(),
                     g.mean(), g.max(), edges[y:y + patch, x:x + patch].mean()])
    return np.asarray(rows, dtype=np.float32)
