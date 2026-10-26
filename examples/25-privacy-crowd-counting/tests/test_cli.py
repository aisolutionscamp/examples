import cv2
import numpy as np
import pytest

from counting.__main__ import _points, video_frames


def test_points_parsing():
    assert _points("0,360,1280,360") == [(0.0, 360.0), (1280.0, 360.0)]
    assert _points(None) is None


def test_video_frames_reads_every_nth_frame(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG writer not available in this OpenCV build")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    frames = list(video_frames(path, every_nth=2))
    assert len(frames) == 5
    assert frames[0][1].shape == (48, 64, 3)


def test_video_frames_rejects_missing_source(tmp_path):
    with pytest.raises(SystemExit):
        list(video_frames(str(tmp_path / "missing.mp4")))
