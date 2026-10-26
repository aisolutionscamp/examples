"""Anonymous counting: short-lived track ids, line crossings, zone occupancy.

Track ids are integers that exist only in memory while an object is in view.
They are built from box overlap alone (no appearance features, no faces), so
the same person gets a new id after leaving the frame or on another camera.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    label: str
    box: tuple[float, float, float, float]
    history: list[tuple[float, float]] = field(default_factory=list)
    missed: int = 0


class IouTracker:
    def __init__(self, min_iou: float = 0.3, max_missed: int = 10) -> None:
        self.min_iou, self.max_missed = min_iou, max_missed
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = list(range(len(detections)))
        for track in self.tracks:
            best, best_iou = None, self.min_iou
            for i in unmatched:
                d = detections[i]
                score = iou(track.box, d.box) if d.label == track.label else 0.0
                if score >= best_iou:
                    best, best_iou = i, score
            if best is None:
                track.missed += 1
                continue
            unmatched.remove(best)
            track.box, track.missed = detections[best].box, 0
            track.history = (track.history + [detections[best].center])[-30:]
        for i in unmatched:
            d = detections[i]
            self.tracks.append(Track(self._next, d.label, d.box, [d.center]))
            self._next += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.missed == 0]


class LineCounter:
    """Counts tracks whose center crosses the line p1 -> p2.

    Direction: for a line drawn top -> bottom, moving left -> right is "in".
    Swap p1 and p2 to swap "in" and "out".
    """

    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        self.p1, self.p2 = np.asarray(p1, float), np.asarray(p2, float)
        self.counts: Counter[tuple[str, str]] = Counter()
        self._side: dict[int, float] = {}

    def _side_of(self, point) -> float:
        d, v = self.p2 - self.p1, np.asarray(point, float) - self.p1
        return float(np.sign(d[0] * v[1] - d[1] * v[0]))

    def update(self, tracks: list[Track]) -> None:
        for t in tracks:
            side = self._side_of(t.history[-1])
            before = self._side.get(t.id)
            if before is not None and side != 0 and before != 0 and side != before:
                self.counts[(t.label, "in" if side < 0 else "out")] += 1
            if side != 0:
                self._side[t.id] = side


class ZoneOccupancy:
    """How many objects of each class are inside a polygon right now."""

    def __init__(self, polygon: list[tuple[float, float]]) -> None:
        self.polygon = np.asarray(polygon, np.float32).reshape(-1, 1, 2)

    def count(self, tracks: list[Track]) -> Counter[str]:
        inside = Counter()
        for t in tracks:
            x, y = t.history[-1]
            if cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0:
                inside[t.label] += 1
        return inside
