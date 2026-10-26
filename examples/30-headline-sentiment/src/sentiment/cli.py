"""Score a CSV of headlines and write daily sentiment per ticker.

    python -m sentiment.cli headlines.csv daily.csv [--scorer finbert|vader]

Input columns: published_at (ISO time with offset or UTC), headline, tickers (optional, "AAPL;MSFT").
The output describes the tone of the headlines. It is not investment advice and not a
price forecast.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .aggregate import daily_sentiment, extract_tickers
from .scorers import FinBertScorer, VaderScorer

DISCLAIMER = "Headline tone only. Not investment advice, not a price forecast."


def score_frame(df: pd.DataFrame, scorer) -> pd.DataFrame:
    df = df.copy()
    listed = df["tickers"] if "tickers" in df else [None] * len(df)
    df["tickers"] = [extract_tickers(h, t) for h, t in zip(df["headline"], listed, strict=True)]
    results = scorer.score(df["headline"].tolist())
    df["label"] = [r.label for r in results]
    df["score"] = [r.score for r in results]
    return df


def main(argv: list[str] | None = None, scorer=None) -> pd.DataFrame:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--scorer", choices=["finbert", "vader"], default="finbert")
    p.add_argument("--tz", default="America/New_York")
    p.add_argument("--cutoff-hour", type=int, default=16)
    args = p.parse_args(argv)

    scorer = scorer or (FinBertScorer() if args.scorer == "finbert" else VaderScorer())
    daily = daily_sentiment(score_frame(pd.read_csv(args.input), scorer), args.tz, args.cutoff_hour)
    daily.to_csv(args.output, index=False)
    print(f"{len(daily)} ticker-days written to {args.output}. {DISCLAIMER}")
    return daily


if __name__ == "__main__":
    main()
