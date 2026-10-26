"""Dataset helpers: folder of DICOM files per class, and synthetic images for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from . import dicom_io


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return tf.image.resize(img[..., None], (size, size), antialias=True).numpy()[..., 0]


def load_folder(root: str | Path, size: int, classes: tuple[str, str] = ("normal", "finding")):
    """root/normal/*.dcm and root/finding/*.dcm -> (x [n, size, size, 1], y [n])."""
    xs, ys = [], []
    for label, name in enumerate(classes):
        for path in sorted((Path(root) / name).glob("*.dcm")):
            xs.append(resize(dicom_io.load(path), size))
            ys.append(label)
    if not xs:
        raise ValueError(f"no .dcm files found under {root}/{{{','.join(classes)}}}")
    return np.stack(xs)[..., None].astype(np.float32), np.asarray(ys, np.float32)


def synthetic(n: int, size: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noisy gradient 'images'; label 1 has a bright round blob. Not medical data."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    xs, ys = [], []
    for i in range(n):
        img = 0.3 + 0.2 * yy + rng.normal(0, 0.05, (size, size))
        label = i % 2
        if label:
            cy, cx = rng.uniform(0.25, 0.75, 2)
            r = rng.uniform(0.06, 0.12)
            img += 0.4 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        xs.append(np.clip(img, 0, 1))
        ys.append(label)
    return np.stack(xs)[..., None].astype(np.float32), np.asarray(ys, np.float32)
