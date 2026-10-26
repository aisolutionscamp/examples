"""Turn scored headlines into one row per ticker and trading day."""
from __future__ import annotations

import re

import pandas as pd

CASHTAG = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b")


def extract_tickers(headline: str, listed: str | float | None = None) -> list[str]:
    """Tickers from an explicit column ("AAPL;MSFT") plus cashtags in the text ("$AAPL")."""
    found = [t.strip().upper() for t in str(listed).replace(",", ";").split(";")
             if isinstance(listed, str) and t.strip()]
    found += CASHTAG.findall(headline)
    return sorted(set(found))


def assign_day(published_at: pd.Series, tz: str = "America/New_York", cutoff_hour: int = 16) -> pd.Series:
    """Calendar day in the exchange time zone. News after the close (cutoff_hour) counts for the
    next day, because it can only move prices from the next session on. Weekends and holidays
    are not shifted here; join with an exchange calendar if you need trading days."""
    ts = pd.to_datetime(published_at, utc=True).dt.tz_convert(tz)
    day = ts.dt.normalize() + pd.to_timedelta((ts.dt.hour >= cutoff_hour).astype(int), unit="D")
    return day.dt.date


def daily_sentiment(scored: pd.DataFrame, tz: str = "America/New_York", cutoff_hour: int = 16) -> pd.DataFrame:
    """Input columns: published_at, headline, tickers (list), label, score.

    Output: ticker, day, headlines, mean_score, positive_share, negative_share.
    Identical headlines for the same ticker and day (syndicated copies) are counted once.
    """
    df = scored.explode("tickers").dropna(subset=["tickers"]).rename(columns={"tickers": "ticker"})
    df["day"] = assign_day(df["published_at"], tz, cutoff_hour)
    df["norm"] = df["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["ticker", "day", "norm"])
    out = df.groupby(["ticker", "day"]).agg(
        headlines=("score", "size"),
        mean_score=("score", "mean"),
        positive_share=("label", lambda s: (s == "positive").mean()),
        negative_share=("label", lambda s: (s == "negative").mean()),
    ).reset_index()
    return out.round({"mean_score": 3, "positive_share": 3, "negative_share": 3})
