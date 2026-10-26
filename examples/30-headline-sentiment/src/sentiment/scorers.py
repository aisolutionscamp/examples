"""Two sentiment scorers with the same interface.

FinBertScorer: ProsusAI/finbert, a BERT model fine-tuned on financial news sentences
(labels positive / negative / neutral). Runs locally on CPU via transformers.
VaderScorer: rule-based lexicon (VADER), no model download, a quick baseline. It was built
for social media text and misreads finance terms ("short", "liability", "beat").
Both return a score in [-1, 1] per text.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Sentiment:
    label: str  # positive | negative | neutral
    score: float  # P(positive) - P(negative) for FinBERT, VADER compound for VADER
    confidence: float


class FinBertScorer:
    def __init__(self, model: str | None = None, batch_size: int = 16, pipeline_factory=None):
        self.model = model or os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
        self.batch_size = batch_size
        self._factory = pipeline_factory
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            factory = self._factory
            if factory is None:
                from transformers import pipeline as factory
            # top_k=None returns the probability of every label, not only the best one
            self._pipe = factory("text-classification", model=self.model, top_k=None, device=-1)
        return self._pipe

    def score(self, texts: list[str]) -> list[Sentiment]:
        if not texts:
            return []
        outputs = self._pipeline()(texts, batch_size=self.batch_size, truncation=True)
        results = []
        for label_scores in outputs:
            probs = {d["label"].lower(): float(d["score"]) for d in label_scores}
            best = max(probs, key=probs.get)
            results.append(Sentiment(best, probs.get("positive", 0.0) - probs.get("negative", 0.0), probs[best]))
        return results


class VaderScorer:
    def __init__(self, threshold: float = 0.05):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def score(self, texts: list[str]) -> list[Sentiment]:
        out = []
        for text in texts:
            c = self.analyzer.polarity_scores(text)["compound"]
            label = "positive" if c >= self.threshold else "negative" if c <= -self.threshold else "neutral"
            out.append(Sentiment(label, c, abs(c)))
        return out
