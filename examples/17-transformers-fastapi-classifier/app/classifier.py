"""Two fine-tuned Transformer classifiers (urgency and department) behind one predict() call."""

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TOKENS = 256


def clean_email(text: str) -> str:
    """Drop quoted replies and signatures, collapse whitespace."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^(On .+ wrote:|-{2,}\s*Original Message|From: )", s, re.IGNORECASE) or s in {"--", "-- "}:
            break
        if not s.startswith(">"):
            kept.append(s)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class TextClassifier:
    """One fine-tuned sequence classification model loaded from a local directory."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        inputs = self.tokenizer(texts, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt")
        logits = self.model(**inputs.to(self.device)).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        results = []
        for row in probs:
            best = int(row.argmax())
            results.append(Prediction(self.labels[best], round(float(row[best]), 4),
                                      {label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)}))
        return results


class EmailClassifier:
    def __init__(self, urgency_dir: str | Path, department_dir: str | Path, min_confidence: float = 0.6):
        self.urgency = TextClassifier(urgency_dir)
        self.department = TextClassifier(department_dir)
        self.min_confidence = min_confidence

    def classify(self, subject: str, body: str) -> dict:
        text = clean_email(f"{subject}\n{body}")
        urgency = self.urgency.predict_batch([text])[0]
        department = self.department.predict_batch([text])[0]
        overall = min(urgency.confidence, department.confidence)
        return {
            "urgency": urgency.label,
            "department": department.label,
            "urgency_confidence": urgency.confidence,
            "department_confidence": department.confidence,
            "overall_confidence": overall,
            "needs_review": overall < self.min_confidence,
            "scores": {"urgency": urgency.scores, "department": department.scores},
        }
