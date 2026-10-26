"""Builds tiny, randomly initialized BERT models offline and fine-tunes them on the sample CSV.

No download from the Hugging Face Hub: the vocabulary is built from the sample data and the model
has one layer with 32 hidden units, so the whole test suite runs on a CPU in seconds.
"""

import re
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from app.classifier import clean_email
from train import fine_tune, read_csv

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_emails.csv"
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def tiny_tokenizer(texts: list[str]) -> BertTokenizer:
    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    return BertTokenizer(vocab={tok: i for i, tok in enumerate(SPECIAL + words)}, do_lower_case=True)


def tiny_model(vocab_size: int, labels: list[str]) -> BertForSequenceClassification:
    torch.manual_seed(0)  # deterministic random initialization
    config = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=512, hidden_dropout_prob=0.0,
                        attention_probs_dropout_prob=0.0, num_labels=len(labels),
                        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)})
    return BertForSequenceClassification(config)


@pytest.fixture(scope="session")
def rows():
    return read_csv(SAMPLE)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, rows):
    """Fine-tune one tiny model per task; returns {task: (model_dir, losses)}."""
    texts = [clean_email(f"{r['subject']}\n{r['body']}") for r in rows]
    out = {}
    for task in ("urgency", "department"):
        labels = sorted({r[task] for r in rows})
        tokenizer = tiny_tokenizer(texts)
        model = tiny_model(len(tokenizer.get_vocab()), labels)
        losses = fine_tune(model, tokenizer, texts, [labels.index(r[task]) for r in rows],
                           epochs=40, lr=2e-3, batch_size=8)
        model_dir = tmp_path_factory.mktemp(task)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        out[task] = (model_dir, losses, model, tokenizer, texts, [labels.index(r[task]) for r in rows])
    return out
