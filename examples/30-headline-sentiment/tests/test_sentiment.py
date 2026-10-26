import pandas as pd
import pytest

from sentiment.aggregate import assign_day, daily_sentiment, extract_tickers
from sentiment.cli import main, score_frame
from sentiment.scorers import FinBertScorer, VaderScorer

# What transformers' text-classification pipeline returns with top_k=None: one list per input.
CANNED = {
    "beats": [{"label": "positive", "score": 0.90}, {"label": "neutral", "score": 0.07},
              {"label": "negative", "score": 0.03}],
    "lawsuit": [{"label": "negative", "score": 0.80}, {"label": "neutral", "score": 0.15},
                {"label": "positive", "score": 0.05}],
    "meeting": [{"label": "neutral", "score": 0.85}, {"label": "positive", "score": 0.10},
                {"label": "negative", "score": 0.05}],
}


def fake_pipeline_factory(calls):
    def factory(task, model, top_k, device):
        calls["init"] = (task, model, top_k, device)

        def run(texts, batch_size, truncation):
            calls["batch_size"] = batch_size
            return [next(v for k, v in CANNED.items() if k in t.lower()) for t in texts]
        return run
    return factory


def test_finbert_scorer_converts_probabilities():
    calls = {}
    scorer = FinBertScorer(model="ProsusAI/finbert", pipeline_factory=fake_pipeline_factory(calls))
    res = scorer.score(["Acme beats estimates", "Acme faces lawsuit", "Acme annual meeting on May 3"])
    assert [r.label for r in res] == ["positive", "negative", "neutral"]
    assert res[0].score == pytest.approx(0.87) and res[1].score == pytest.approx(-0.75)
    assert res[2].confidence == pytest.approx(0.85)
    assert calls["init"] == ("text-classification", "ProsusAI/finbert", None, -1)
    assert scorer.score([]) == []


def test_model_name_from_env(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL", "my-org/finbert-tuned")
    assert FinBertScorer().model == "my-org/finbert-tuned"


def test_vader_baseline_runs_offline():
    res = VaderScorer().score(["Great results, strong growth", "Terrible losses and a scandal", "The report is out"])
    assert [r.label for r in res] == ["positive", "negative", "neutral"]


def test_extract_tickers():
    assert extract_tickers("$AAPL and $MSFT rally", "NVDA; aapl") == ["AAPL", "MSFT", "NVDA"]
    assert extract_tickers("No tickers here, costs $5 more", float("nan")) == []
    assert extract_tickers("$BRK.B holders meet") == ["BRK.B"]


def test_after_close_news_counts_for_next_day():
    days = assign_day(pd.Series(["2026-03-02T15:59:00-05:00", "2026-03-02T21:30:00Z", "2026-03-03T01:00:00Z"]))
    assert [str(d) for d in days] == ["2026-03-02", "2026-03-03", "2026-03-03"]


def test_daily_aggregation_dedups_and_splits_tickers():
    scored = pd.DataFrame({
        "published_at": ["2026-03-02T14:00:00Z", "2026-03-02T15:00:00Z", "2026-03-02T15:05:00Z",
                         "2026-03-02T16:00:00Z"],
        "headline": ["Acme beats estimates", "Acme faces lawsuit", "ACME faces lawsuit!", "Acme and Bolt merge"],
        "tickers": [["ACME"], ["ACME"], ["ACME"], ["ACME", "BOLT"]],
        "label": ["positive", "negative", "negative", "neutral"],
        "score": [0.8, -0.6, -0.6, 0.1],
    })
    out = daily_sentiment(scored)
    acme = out[out["ticker"] == "ACME"].iloc[0]
    assert acme["headlines"] == 3  # the syndicated copy is dropped
    assert acme["mean_score"] == pytest.approx(0.1)
    assert acme["positive_share"] == pytest.approx(0.333) and acme["negative_share"] == pytest.approx(0.333)
    assert out[out["ticker"] == "BOLT"]["headlines"].item() == 1


def test_cli_end_to_end(tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.DataFrame({
        "published_at": ["2026-03-02T14:00:00Z", "2026-03-02T15:00:00Z"],
        "headline": ["$ACME beats estimates", "Annual meeting set"],
        "tickers": [None, "ACME"],
    }).to_csv(src, index=False)
    calls = {}
    daily = main([str(src), str(tmp_path / "out.csv")],
                 scorer=FinBertScorer(pipeline_factory=fake_pipeline_factory(calls)))
    assert daily["headlines"].tolist() == [2]
    assert pd.read_csv(tmp_path / "out.csv")["ticker"].tolist() == ["ACME"]
    assert "Not investment advice" in capsys.readouterr().out


def test_rows_without_ticker_are_dropped():
    df = score_frame(pd.DataFrame({"published_at": ["2026-03-02T14:00:00Z"], "headline": ["Markets are calm"]}),
                     VaderScorer())
    assert daily_sentiment(df).empty
