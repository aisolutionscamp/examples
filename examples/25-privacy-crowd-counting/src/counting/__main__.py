"""Count people and vehicles in a video file or stream; print JSON lines per interval.

    python -m counting --model models/yolo11n.onnx --source rtsp://camera.example.com/stream \
        --camera-id gate-1 --interval 60 --line 0,360,1280,360
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import cv2

from .detector import OnnxYoloDetector
from .pipeline import run


def video_frames(source: str, every_nth: int = 1):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {source}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % every_nth == 0:
                yield time.time(), frame
    finally:
        cap.release()


def _points(text: str | None):
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    return list(zip(values[::2], values[1::2]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Anonymous people/vehicle counting (YOLO + ONNX Runtime).")
    p.add_argument("--model", default=os.environ.get("YOLO_ONNX_MODEL", "models/yolo11n.onnx"))
    p.add_argument("--source", required=True, help="video file, RTSP URL or camera index")
    p.add_argument("--camera-id", default="camera-1")
    p.add_argument("--interval", type=float, default=60.0, help="aggregation interval in seconds")
    p.add_argument("--line", help="counting line x1,y1,x2,y2 in pixels")
    p.add_argument("--zone", help="occupancy polygon x1,y1,x2,y2,... in pixels")
    p.add_argument("--every-nth", type=int, default=2, help="process every n-th frame")
    p.add_argument("--conf", type=float, default=0.35)
    a = p.parse_args(argv)
    line = _points(a.line)
    detector = OnnxYoloDetector(a.model, conf=a.conf)
    for interval in run(video_frames(a.source, a.every_nth), detector, a.camera_id, a.interval,
                        tuple(line) if line else None, _points(a.zone)):
        print(json.dumps(asdict(interval)), flush=True)


if __name__ == "__main__":
    main()
