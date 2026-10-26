"""Remove the background of product photos with OpenCV GrabCut.

GrabCut needs a first guess of what is foreground. Product photos are usually
shot on a plain, light background, so the guess comes from the image border:
pixels that look like the border are "probably background", the rest
"probably foreground", and a thin frame around the image is "surely
background". GrabCut then refines this guess with color models (GMMs) and a
graph cut. The result is cleaned up and returned as a soft alpha channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Settings:
    border: float = 0.02  # share of width/height treated as sure background
    color_tolerance: float = 30.0  # Lab distance from the background color
    iterations: int = 5
    feather: int = 3  # px of soft edge in the alpha channel; 0 = hard edge
    keep_largest: bool = False  # keep only the largest object (single products)
    fill_holes: bool = True  # light labels/reflections inside the product stay opaque
    max_side: int = 1024  # GrabCut runs on a downscaled copy for speed


def _background_color(lab: np.ndarray, frame: int) -> np.ndarray:
    border = np.concatenate([lab[:frame].reshape(-1, 3), lab[-frame:].reshape(-1, 3),
                             lab[:, :frame].reshape(-1, 3), lab[:, -frame:].reshape(-1, 3)])
    return np.median(border, axis=0)


def initial_mask(image: np.ndarray, s: Settings) -> np.ndarray:
    h, w = image.shape[:2]
    frame = max(2, round(min(h, w) * s.border))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - _background_color(lab, frame), axis=2)
    mask = np.where(distance > s.color_tolerance, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[:frame], mask[-frame:], mask[:, :frame], mask[:, -frame:] = (cv2.GC_BGD,) * 4
    return mask


def foreground_mask(image: np.ndarray, s: Settings | None = None) -> np.ndarray:
    """Return a uint8 mask (255 = product, 0 = background) for a BGR image."""
    s = s or Settings()
    h, w = image.shape[:2]
    scale = min(1.0, s.max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else image
    mask = initial_mask(small, s)
    if not np.any(mask == cv2.GC_PR_FGD):
        return np.zeros((h, w), np.uint8)  # nothing differs from the background

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgd, fgd, s.iterations, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # remove specks
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill pinholes
    if s.keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    if s.fill_holes:
        binary = _fill_holes(binary)
    if scale < 1:
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = np.where(binary > 127, 255, 0).astype(np.uint8)
    return binary


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Set background regions that do not touch the image border to foreground."""
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 255)
    holes = cv2.bitwise_not(outside)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def to_rgba(image: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
    """BGR image + mask -> BGRA image with (optionally soft) alpha."""
    alpha = mask
    if feather > 0:
        k = 2 * feather + 1
        alpha = cv2.GaussianBlur(mask, (k, k), 0)
    return np.dstack([image, alpha])


def on_white(image: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
    """Composite onto pure white, as many marketplaces require for main images."""
    alpha = to_rgba(image, mask, feather)[:, :, 3:].astype(np.float32) / 255.0
    return (image * alpha + 255 * (1 - alpha)).astype(np.uint8)


def remove_background(image: np.ndarray, s: Settings | None = None) -> np.ndarray:
    s = s or Settings()
    return to_rgba(image, foreground_mask(image, s), s.feather)
