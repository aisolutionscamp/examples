"""Train on the synthetic stream with a time-based split and save the model.

    python -m fraud.train --out model/fraud_model.json --max-fpr 0.005
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import model as fm
from .synth import generate


def run(out: str, max_fpr: float, n_users: int = 600, seed: int = 7) -> dict:
    transactions, labels = generate(n_users=n_users, seed=seed)
    x, y = fm.build_dataset(transactions), np.asarray(labels)
    # Time-based split: train on the past, pick the threshold on the next
    # period, report on the most recent period (no shuffling across time).
    a, b = int(len(y) * 0.6), int(len(y) * 0.8)
    clf = fm.train(x[:a], y[:a], seed=seed)
    threshold = fm.threshold_for_fpr(y[a:b], clf.predict_proba(x[a:b])[:, 1], max_fpr)
    metrics = fm.evaluate(y[b:], clf.predict_proba(x[b:])[:, 1], threshold)
    fm.save(clf, out, threshold)
    return {"rows": len(y), "fraud_rows": int(y.sum()), "test": metrics.as_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="model/fraud_model.json")
    parser.add_argument("--max-fpr", type=float, default=0.005)
    parser.add_argument("--users", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.max_fpr, args.users), indent=2))


if __name__ == "__main__":
    main()
