import csv
import json
from pathlib import Path

import pytest

from reviews.__main__ import run
from reviews.analyzer import ReviewAnalyzer, label, summarize

DATA = Path(__file__).resolve().parents[1] / "samples" / "sample_reviews.csv"


@pytest.fixture(scope="module")
def analyzer():
    return ReviewAnalyzer()


@pytest.mark.parametrize("text,expected", [
    ("Absolutely delicious food and lovely staff.", "positive"),
    ("Terrible service, the soup was cold and the waiter was rude.", "negative"),
    ("We had lunch there on Tuesday.", "neutral"),
    ("The pizza was not good.", "negative"),
])
def test_overall_label(analyzer, text, expected):
    assert analyzer.analyze(text).sentiment == expected


def test_mixed_review_splits_aspects(analyzer):
    r = analyzer.analyze("The food was amazing and the service was quick, "
                         "but the restaurant was quite noisy and hard to have a conversation.")
    assert r.aspects["food"] > 0
    assert r.aspects["ambience"] < 0
    assert set(r.aspects) >= {"food", "service", "ambience", "wait"}


def test_domain_lexicon_catches_restaurant_complaints():
    text = "Overpriced for tiny portions."
    assert ReviewAnalyzer(extra_lexicon={}).analyze(text).sentiment == "neutral"  # plain VADER
    assert ReviewAnalyzer().analyze(text).sentiment == "negative"


def test_label_thresholds():
    assert label(0.05) == "positive" and label(-0.05) == "negative" and label(0.0) == "neutral"


def test_summary_counts_and_worst(analyzer):
    texts = ["Great food.", "Awful, dirty toilets.", "Rude waiter and horrible food.", "It is a restaurant."]
    report = summarize([analyzer.analyze(t) for t in texts])
    assert report["total_reviews"] == 4
    assert report["counts"] == {"positive": 1, "neutral": 1, "negative": 2}
    assert report["most_negative"][0] in texts[1:3]
    assert report["aspects"]["food"]["mentions"] == 2


def test_sample_csv_mostly_agrees_with_star_ratings(analyzer):
    """Sanity check on the bundled examples: 4-5 stars should not be negative, 1-2 not positive.

    Known VADER limitation: VADER weights the clause after "but" higher, so the
    mixed 4-star review ("food amazing ... but quite noisy") comes out negative
    overall. The aspect scores still show the positive food rating.
    """
    disagreements = []
    with DATA.open() as f:
        for row in csv.DictReader(f):
            s = analyzer.analyze(row["review"]).sentiment
            stars = int(row["stars"])
            if (stars >= 4 and s == "negative") or (stars <= 2 and s == "positive"):
                disagreements.append(row["review"])
    assert len(disagreements) <= 1, disagreements


def test_cli_writes_outputs(tmp_path):
    report = run(DATA, "review", tmp_path / "r.csv", tmp_path / "r.json")
    assert report["total_reviews"] == 10
    assert json.loads((tmp_path / "r.json").read_text())["counts"] == report["counts"]
    assert len((tmp_path / "r.csv").read_text().splitlines()) == 11


def test_cli_rejects_unknown_column(tmp_path):
    with pytest.raises(SystemExit):
        run(DATA, "text", None, None)
