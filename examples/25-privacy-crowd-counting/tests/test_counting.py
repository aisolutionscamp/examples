import json
from dataclasses import asdict

import numpy as np

from counting.detector import Detection
from counting.pipeline import pixelate, run
from counting.tracking import IouTracker, LineCounter, ZoneOccupancy


def person_at(x, y, label="person"):
    return Detection(label, 0.9, (x - 10, y - 20, x + 10, y + 20))


def test_tracker_keeps_id_while_object_moves_and_drops_it_later():
    tracker = IouTracker(max_missed=2)
    ids = [tracker.update([person_at(100 + 3 * i, 100)])[0].id for i in range(5)]
    assert len(set(ids)) == 1
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == []
    assert tracker.update([person_at(110, 100)])[0].id != ids[0]  # new anonymous id


def test_line_counter_counts_each_crossing_once_by_direction():
    tracker, line = IouTracker(), LineCounter((200, 0), (200, 400))
    for x in range(150, 260, 5):  # walks left -> right across x = 200
        line.update(tracker.update([person_at(x, 100), person_at(400 - x, 300, "car")]))
    # Line drawn top -> bottom: left-to-right is "in", right-to-left is "out".
    assert dict(line.counts) == {("person", "in"): 1, ("car", "out"): 1}


def test_zone_occupancy():
    tracker = IouTracker()
    tracks = tracker.update([person_at(50, 50), person_at(150, 50), person_at(500, 500), person_at(60, 60, "car")])
    zone = ZoneOccupancy([(0, 0), (200, 0), (200, 200), (0, 200)])
    assert zone.count(tracks) == {"person": 2, "car": 1}


def test_pixelate_only_changes_person_boxes():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = pixelate(frame, [Detection("person", 0.9, (10, 10, 50, 50)), Detection("car", 0.9, (60, 60, 90, 90))])
    assert not np.array_equal(out[10:50, 10:50], frame[10:50, 10:50])
    assert np.array_equal(out[60:90, 60:90], frame[60:90, 60:90])
    assert np.array_equal(out[:10], frame[:10])


def test_pipeline_emits_only_aggregates_per_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # 3 people walk across a vertical line at x = 200, one per 20 s; 1 frame per second, 60 s intervals.
    def detector(frame):
        t = int(frame[0, 0, 0])  # synthetic frame carries its timestamp in one pixel
        dets = []
        for k in range(3):
            start = 20 * k
            if start <= t < start + 30:
                dets.append(person_at(100 + 8 * (t - start), 100 + 60 * k))
        return dets

    def frames():
        for t in range(120):
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[0, 0, 0] = t
            yield float(t), frame

    previews = []
    intervals = list(run(frames(), detector, "gate-1", interval_s=60, line=((200, 0), (200, 480)),
                         preview=previews.append))
    assert [(i.start, i.end, i.frames) for i in intervals] == [(0, 60, 60), (60, 120, 60)]
    total = sum(sum(i.crossings.values()) for i in intervals)
    assert total == 3
    assert intervals[0].occupancy_max["person"] == 2
    assert len(previews) == 120
    record = json.loads(json.dumps(asdict(intervals[0])))  # plain JSON, no images or ids
    assert set(record) == {"camera_id", "start", "end", "crossings", "occupancy_max", "occupancy_mean", "frames"}
    assert list(tmp_path.iterdir()) == []  # nothing written to disk
