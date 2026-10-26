"""File lists for the MVTec AD dataset layout (download it yourself from mvtec.com).

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png

MVTec AD is licensed CC BY-NC-SA 4.0: research and non-commercial use only.
"""
from __future__ import annotations

from pathlib import Path


def mvtec_split(root: str | Path, category: str) -> dict[str, list[Path]]:
    base = Path(root) / category
    test = base / "test"
    return {
        "train_good": sorted((base / "train" / "good").glob("*.png")),
        "test_good": sorted((test / "good").glob("*.png")),
        "test_defect": sorted(p for d in test.iterdir() if d.is_dir() and d.name != "good"
                              for p in d.glob("*.png")),
    }
