"""Frame loop: detect -> track -> count -> aggregate. Frames never leave this loop.

Privacy by design:
- frames are processed in memory and dropped; nothing is written to disk,
- no face detection, face recognition or re-identification of individuals,
- only per-interval aggregate counts are emitted,
- the optional preview pixelates every person box before it is shown.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection
from .tracking import IouTracker, LineCounter, ZoneOccupancy


@dataclass
class Interval:
    camera_id: str
    start: float
    end: float
    crossings: dict[str, int] = field(default_factory=dict)  # "person_in": 12, ...
    occupancy_max: dict[str, int] = field(default_factory=dict)
    occupancy_mean: dict[str, float] = field(default_factory=dict)
    frames: int = 0


def pixelate(frame: np.ndarray, detections: list[Detection], labels=("person",), blocks: int = 8) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        if d.label not in labels:
            continue
        x1, y1, x2, y2 = (int(v) for v in d.box)
        roi = out[y1:y2, x1:x2]
        if roi.size:
            small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def run(frames: Iterable[tuple[float, np.ndarray]], detector: Callable[[np.ndarray], list[Detection]],
        camera_id: str, interval_s: float = 60.0, line: tuple | None = None, zone: list | None = None,
        preview: Callable[[np.ndarray], None] | None = None) -> Iterator[Interval]:
    """frames: (timestamp, BGR image) pairs. Yields one Interval per interval_s."""
    tracker = IouTracker()
    counter = LineCounter(*line) if line else None
    occupancy = ZoneOccupancy(zone) if zone else None
    current: Interval | None = None
    occ_sum: Counter[str] = Counter()
    for ts, frame in frames:
        if current is None:
            current = Interval(camera_id, ts, ts + interval_s)
        while ts >= current.end:
            yield _close(current, counter, occ_sum)
            current, occ_sum = Interval(camera_id, current.end, current.end + interval_s), Counter()
        detections = detector(frame)
        tracks = tracker.update(detections)
        if counter:
            counter.update(tracks)
        inside = occupancy.count(tracks) if occupancy else Counter(t.label for t in tracks)
        for label, n in inside.items():
            current.occupancy_max[label] = max(current.occupancy_max.get(label, 0), n)
            occ_sum[label] += n
        current.frames += 1
        if preview:
            preview(pixelate(frame, detections))
        del frame  # the frame is not kept beyond this iteration
    if current is not None:
        yield _close(current, counter, occ_sum)


def _close(interval: Interval, counter: LineCounter | None, occ_sum: Counter) -> Interval:
    if counter:
        interval.crossings = {f"{label}_{direction}": n for (label, direction), n in sorted(counter.counts.items())}
        counter.counts.clear()
    if interval.frames:
        interval.occupancy_mean = {k: round(v / interval.frames, 2) for k, v in occ_sum.items()}
    return interval
