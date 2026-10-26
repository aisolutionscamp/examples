"""Train and evaluate on a DICOM folder (or on synthetic images with --synthetic).

    python -m medimg.train --data data/ --model small --size 128 --epochs 20 --out model.keras
    python -m medimg.train --synthetic --epochs 5

Research/education code: not a medical device, not for clinical use.
"""

from __future__ import annotations

import argparse
import json

import keras
import numpy as np

from . import data
from .evaluate import binary_metrics
from .model import resnet50_transfer, small_cnn


def split(x, y, test_share: float = 0.2, seed: int = 0):
    """Random split. With real data split by patient, never by image (see README)."""
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(len(y) * (1 - test_share))
    return x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]]


def run(x, y, model_name: str = "small", epochs: int = 10, seed: int = 0, weights: str | None = "imagenet",
        out: str | None = None) -> dict:
    keras.utils.set_random_seed(seed)
    x_train, y_train, x_test, y_test = split(x, y, seed=seed)
    size = x.shape[1]
    model = small_cnn(size) if model_name == "small" else resnet50_transfer(size, weights=weights)
    model.fit(x_train, y_train, validation_split=0.15, epochs=epochs, batch_size=16, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    prob = model.predict(x_test, verbose=0)
    if out:
        model.save(out)
    return {"train_images": len(y_train), "test_images": len(y_test), **binary_metrics(y_test, prob)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", help="folder with normal/ and finding/ subfolders of .dcm files")
    p.add_argument("--synthetic", action="store_true", help="use generated images instead of --data")
    p.add_argument("--model", choices=["small", "resnet50"], default="small")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", default="model.keras")
    a = p.parse_args()
    x, y = data.synthetic(200, a.size) if a.synthetic else data.load_folder(a.data, a.size)
    print(json.dumps(run(x, y, a.model, a.epochs, out=a.out), indent=2))


if __name__ == "__main__":
    main()
