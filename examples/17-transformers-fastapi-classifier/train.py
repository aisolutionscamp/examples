"""Fine-tune the urgency and department classifiers on a CSV of labeled support emails.

    python train.py data/sample_emails.csv --out models --epochs 4

CSV columns: subject, body, urgency, department. Each model is saved with its tokenizer
to <out>/urgency and <out>/department, ready for app/classifier.py.
"""

import argparse
import csv
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.classifier import MAX_TOKENS, clean_email

DEFAULT_BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"  # covers English, Spanish, German


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["body"].strip()]


def split(rows: list[dict], test_share: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_share))
    return rows[n_test:], rows[:n_test]


def fine_tune(model, tokenizer, texts: list[str], label_ids: list[int], epochs: int = 3,
              lr: float = 5e-5, batch_size: int = 16, seed: int = 42) -> list[float]:
    """Plain PyTorch training loop. Returns the mean loss per epoch."""
    torch.manual_seed(seed)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    order = list(range(len(texts)))
    epoch_losses = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_TOKENS,
                              padding=True, return_tensors="pt")
            out = model(**batch, labels=torch.tensor([label_ids[i] for i in idx]))
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(out.loss.item())
        epoch_losses.append(sum(losses) / len(losses))
    model.eval()
    return epoch_losses


@torch.inference_mode()
def evaluate(model, tokenizer, texts: list[str], label_ids: list[int]) -> dict:
    batch = tokenizer(texts, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt")
    pred = model(**batch).logits.argmax(dim=-1).tolist()
    labels = sorted(set(label_ids) | set(pred))
    f1s = []
    for c in labels:
        tp = sum(p == c and y == c for p, y in zip(pred, label_ids, strict=True))
        fp = sum(p == c and y != c for p, y in zip(pred, label_ids, strict=True))
        fn = sum(p != c and y == c for p, y in zip(pred, label_ids, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    accuracy = sum(p == y for p, y in zip(pred, label_ids, strict=True)) / len(label_ids)
    return {"accuracy": round(accuracy, 3), "macro_f1": round(sum(f1s) / len(f1s), 3), "n": len(label_ids)}


def train_task(rows_train, rows_test, column: str, base_model: str, out_dir: Path, epochs: int) -> dict:
    labels = sorted({r[column] for r in rows_train + rows_test})
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
    texts = [clean_email(f"{r['subject']}\n{r['body']}") for r in rows_train]
    losses = fine_tune(model, tokenizer, texts, [label2id[r[column]] for r in rows_train], epochs=epochs)
    metrics = evaluate(model, tokenizer, [clean_email(f"{r['subject']}\n{r['body']}") for r in rows_test],
                       [label2id[r[column]] for r in rows_test])
    model.save_pretrained(out_dir / column)
    tokenizer.save_pretrained(out_dir / column)
    return {"task": column, "labels": labels, "losses": [round(x, 4) for x in losses], "test": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--test-share", type=float, default=0.2)
    args = parser.parse_args()

    train_rows, test_rows = split(read_csv(args.csv), args.test_share)
    for column in ("urgency", "department"):
        print(train_task(train_rows, test_rows, column, args.base_model, args.out, args.epochs))


if __name__ == "__main__":
    main()
