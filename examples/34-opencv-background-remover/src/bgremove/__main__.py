"""Batch CLI.

    python -m bgremove photos/ out/              # transparent PNGs
    python -m bgremove photos/ out/ --white      # JPGs on pure white
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .remover import Settings, foreground_mask, on_white, to_rgba

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_folder(src: Path, dst: Path, settings: Settings, white: bool = False) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in EXTENSIONS):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipped (not readable): {path.name}", file=sys.stderr)
            continue
        mask = foreground_mask(image, settings)
        if white:
            out = dst / f"{path.stem}.jpg"
            cv2.imwrite(str(out), on_white(image, mask, settings.feather), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            out = dst / f"{path.stem}.png"
            cv2.imwrite(str(out), to_rgba(image, mask, settings.feather))
        written.append(out)
        print(f"{path.name} -> {out.name} ({mask.mean() / 2.55:.0f}% foreground)")
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Remove backgrounds from product photos (OpenCV GrabCut).")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--white", action="store_true", help="write JPGs on a white background instead of PNGs")
    p.add_argument("--tolerance", type=float, default=Settings.color_tolerance)
    p.add_argument("--feather", type=int, default=Settings.feather)
    p.add_argument("--single", action="store_true", help="keep only the largest object")
    a = p.parse_args(argv)
    s = Settings(color_tolerance=a.tolerance, feather=a.feather, keep_largest=a.single)
    process_folder(a.src, a.dst, s, a.white)


if __name__ == "__main__":
    main()
