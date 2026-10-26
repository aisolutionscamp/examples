"""Train, calibrate, export and score.

    python -m qc.cli train --good data/train/good --val-good data/val/good \
        [--val-defect data/val/defect] --out model.joblib --onnx model.onnx
    python -m qc.cli score --model model.joblib part1.png part2.png
    python -m qc.cli demo        # everything on synthetic images
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import PatchAnomalyDetector
from .onnx_export import export_onnx
from .synthetic import add_defect, good_image


def _read(folder: str | None) -> list[np.ndarray]:
    if not folder:
        return []
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--good", required=True)
    t.add_argument("--val-good", required=True)
    t.add_argument("--val-defect")
    t.add_argument("--max-false-reject", type=float, default=0.01)
    t.add_argument("--out", default="model.joblib")
    t.add_argument("--onnx")
    s = sub.add_parser("score")
    s.add_argument("--model", default="model.joblib")
    s.add_argument("images", nargs="+")
    sub.add_parser("demo")
    args = p.parse_args(argv)

    if args.cmd == "train":
        det = PatchAnomalyDetector().fit(_read(args.good))
        th = det.calibrate(_read(args.val_good), _read(args.val_defect), args.max_false_reject)
        det.save(args.out)
        if args.onnx:
            export_onnx(det, args.onnx)
        print(json.dumps(th.__dict__))
    elif args.cmd == "score":
        det = PatchAnomalyDetector.load(args.model)
        for path in args.images:
            print(json.dumps({"image": path, **det.predict(cv2.imread(path, cv2.IMREAD_GRAYSCALE))}))
    else:
        rng = np.random.default_rng(0)
        det = PatchAnomalyDetector().fit([good_image(rng) for _ in range(20)])
        th = det.calibrate([good_image(rng) for _ in range(20)],
                           [add_defect(good_image(rng), rng)[0] for _ in range(20)])
        print("threshold:", th)
        print("good part:", det.predict(good_image(rng)))
        print("defect:   ", det.predict(add_defect(good_image(rng), rng)[0]))


if __name__ == "__main__":
    main()
