"""Synthetic brushed-metal style images with optional defects, for tests and a first try-out."""
from __future__ import annotations

import cv2
import numpy as np


def good_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    base = rng.uniform(120, 150)
    img = np.full((size, size), base, np.float32)
    img += rng.normal(0, 6, (size, size)).astype(np.float32)
    # horizontal brushing texture
    img += np.repeat(rng.normal(0, 4, (size, 1)), size, axis=1).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_defect(img: np.ndarray, rng: np.random.Generator, kind: str | None = None) -> tuple[np.ndarray, tuple]:
    """Draw a scratch, spot or crack. Returns the image and the defect centre (x, y)."""
    out = img.copy()
    size = img.shape[0]
    kind = kind or rng.choice(["scratch", "spot", "crack"])
    cx, cy = (int(v) for v in rng.integers(40, size - 40, 2))
    if kind == "scratch":
        angle = rng.uniform(0, np.pi)
        dx, dy = int(25 * np.cos(angle)), int(25 * np.sin(angle))
        cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), 230, 2)
    elif kind == "spot":
        cv2.circle(out, (cx, cy), int(rng.integers(5, 10)), 40, -1)
    else:
        pts = np.cumsum(rng.integers(-4, 5, (12, 2)), axis=0) + [cx, cy]
        cv2.polylines(out, [pts.astype(np.int32)], False, 30, 2)
    return out, (cx, cy)
