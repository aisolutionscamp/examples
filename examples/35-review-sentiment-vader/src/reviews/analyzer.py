"""Restaurant review sentiment with VADER, overall and per aspect.

VADER is a rule-based sentiment model with a built-in lexicon (shipped inside
the `vaderSentiment` package, no download). It handles negation ("not good"),
intensifiers ("very"), capitals and "but" clauses, and needs no training data.
Aspects (food, service, ...) come from a small keyword list: every clause that
mentions an aspect gets that clause's sentiment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Standard VADER cut-offs for the compound score (-1 .. 1).
POSITIVE, NEGATIVE = 0.05, -0.05

ASPECTS: dict[str, set[str]] = {
    "food": {"food", "dish", "dishes", "meal", "pizza", "pasta", "burger", "burgers", "steak", "dessert", "taste",
             "tasty", "flavor", "flavour", "menu", "portion", "portions", "fries", "soup", "salad", "fish"},
    "service": {"service", "staff", "waiter", "waitress", "server", "host", "friendly", "rude",
                "attentive", "manager", "bartender"},
    "ambience": {"ambience", "atmosphere", "noisy", "loud", "music", "decor", "cozy", "crowded",
                 "quiet", "view", "terrace"},
    "price": {"price", "prices", "expensive", "cheap", "overpriced", "value", "bill", "cost", "pricey"},
    "wait": {"wait", "waited", "waiting", "slow", "quick", "fast", "minutes", "hour", "delay", "late"},
    "cleanliness": {"clean", "dirty", "hygiene", "toilet", "toilets", "restroom", "sticky", "smell"},
}

# Restaurant words that the general-purpose VADER lexicon scores as neutral.
# Scale is VADER's: -4 (most negative) .. +4 (most positive). Extend for your venue.
DOMAIN_LEXICON: dict[str, float] = {
    "overpriced": -2.0, "pricey": -1.0, "tiny": -1.0, "slow": -1.5, "waited": -0.8, "lukewarm": -1.2,
    "bland": -1.8, "greasy": -1.5, "soggy": -1.5, "stale": -1.5, "undercooked": -2.0,
    "overcooked": -1.5, "burnt": -1.8, "sticky": -1.0, "crowded": -0.8, "cramped": -1.0,
    "fresh": 1.2, "crispy": 1.0, "juicy": 1.2, "tender": 1.2, "cozy": 1.5, "attentive": 1.5,
}

_CLAUSE_SPLIT = re.compile(r"[.!?;\n]+|\s+but\s+|\s+however\s+|\s+although\s+", re.IGNORECASE)
_WORD = re.compile(r"[a-z']+")


def label(compound: float) -> str:
    if compound >= POSITIVE:
        return "positive"
    if compound <= NEGATIVE:
        return "negative"
    return "neutral"


@dataclass
class ReviewResult:
    text: str
    compound: float
    sentiment: str
    aspects: dict[str, float] = field(default_factory=dict)  # aspect -> mean clause compound


class ReviewAnalyzer:
    def __init__(self, aspects: dict[str, set[str]] | None = None,
                 extra_lexicon: dict[str, float] | None = None) -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(DOMAIN_LEXICON if extra_lexicon is None else extra_lexicon)
        self.aspects = aspects or ASPECTS

    def score(self, text: str) -> float:
        return float(self.vader.polarity_scores(text)["compound"])

    def analyze(self, text: str) -> ReviewResult:
        compound = self.score(text)
        per_aspect: dict[str, list[float]] = defaultdict(list)
        for clause in filter(None, (c.strip() for c in _CLAUSE_SPLIT.split(text))):
            words = set(_WORD.findall(clause.lower()))
            hits = [a for a, keywords in self.aspects.items() if words & keywords]
            if hits:
                clause_score = self.score(clause)
                for aspect in hits:
                    per_aspect[aspect].append(clause_score)
        aspects = {a: round(sum(v) / len(v), 3) for a, v in per_aspect.items()}
        return ReviewResult(text, round(compound, 3), label(compound), aspects)


def summarize(results: list[ReviewResult], worst: int = 3) -> dict:
    """Counts, shares and per-aspect averages, plus the most negative reviews to read first."""
    total = len(results)
    counts = {s: sum(r.sentiment == s for r in results) for s in ("positive", "neutral", "negative")}
    aspect_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for aspect, value in r.aspects.items():
            aspect_scores[aspect].append(value)
    aspects = {
        a: {"mentions": len(v), "mean": round(sum(v) / len(v), 3),
            "negative_share": round(sum(x <= NEGATIVE for x in v) / len(v), 3)}
        for a, v in sorted(aspect_scores.items(), key=lambda kv: -len(kv[1]))
    }
    return {
        "total_reviews": total,
        "counts": counts,
        "shares": {s: round(n / total, 3) if total else 0.0 for s, n in counts.items()},
        "mean_compound": round(sum(r.compound for r in results) / total, 3) if total else 0.0,
        "aspects": aspects,
        "most_negative": [r.text for r in sorted(results, key=lambda r: r.compound)[:worst]
                          if r.sentiment == "negative"],
    }
