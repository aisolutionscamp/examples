# Financial news headline sentiment per ticker and day (FinBERT + VADER baseline)

Use case: [Financial Market Sentiment Analysis](https://aisolutionscamp.io/projects/30)

> **Not investment advice.** This example measures the tone of news headlines. It does not
> forecast prices, it is not a trading signal, and no claim is made that headline sentiment
> predicts market movements.

## What it does

1. Reads a CSV of headlines: `published_at`, `headline`, optional `tickers` (`"AAPL;MSFT"`).
2. Finds tickers in that column and as cashtags in the text (`$AAPL`, `$BRK.B`).
3. Scores each headline:
   - **FinBERT** ([ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)) via the
     `transformers` text-classification pipeline with `top_k=None`; score = P(positive) - P(negative).
   - or **VADER** (`--scorer vader`), a rule-based lexicon baseline, compound score in [-1, 1].
4. Assigns each headline to a day in the exchange time zone (default `America/New_York`);
   headlines at or after 16:00 count for the next day.
5. Drops syndicated duplicates (same normalised headline, ticker and day) and writes one row per
   ticker and day: `headlines, mean_score, positive_share, negative_share`.

```
headlines.csv -> tickers/cashtags -> FinBERT | VADER -> exchange day (+ after-close shift)
  -> dedup + groupby(ticker, day) -> daily.csv
```

## Setup and run

```bash
pip install -r requirements.txt     # installs the CPU build of PyTorch (see the index URL in the file)
cd src
python -m sentiment.cli ../headlines.csv ../daily.csv                  # FinBERT
python -m sentiment.cli ../headlines.csv ../daily.csv --scorer vader   # baseline, no download
```

The FinBERT model (about 440 MB) is downloaded from Hugging Face on first use and cached.
For a GPU, install the matching CUDA build of PyTorch instead of the CPU one.

| Option / variable | Default | Meaning |
| --- | --- | --- |
| `SENTIMENT_MODEL` | `ProsusAI/finbert` | any text-classification model with positive/negative/neutral labels |
| `HF_TOKEN` | empty | only for gated or private models |
| `--tz` | `America/New_York` | exchange time zone for the daily bucket |
| `--cutoff-hour` | 16 | headlines at or after this hour count for the next day |

## Tests

```bash
pytest     # or: scripts/test-example.sh examples/30-headline-sentiment
```

The tests never download a model: the transformers pipeline is replaced by a function that
returns the same structure the real pipeline returns with `top_k=None` (a list of
`{"label", "score"}` per input). VADER runs for real (its lexicon ships with the package).
Covered: probability-to-score conversion, model name from env, ticker and cashtag extraction,
time zone and after-close handling, de-duplication, aggregation and the CLI.

Checked manually (not part of the tests): with the real `ProsusAI/finbert` model,
"Acme shares fall after regulator opens probe" came out negative and "Acme to hold annual
meeting on May 3" neutral.

## Limitations

- Headlines only. Tone of a headline is not the same as its effect on a company.
- FinBERT works on single sentences in English. Long articles must be split, and sarcasm or
  context ("beats lowered expectations") is often missed.
- VADER is a general-purpose lexicon and misreads finance terms ("short", "liability", "beat").
- Weekends and exchange holidays are not shifted to the next trading day; join with an exchange
  calendar if you need that.
- Ticker extraction is naive: company names without a cashtag or ticker column are not mapped.
- License: the Hugging Face model card lists `ProsusAI/finbert` under CC BY-NC-SA 4.0 at the
  time of writing. Check the current license before commercial use, or fine-tune your own model.

## For production

- Use a news source whose terms allow automated processing and storage.
- Map company names to tickers with a maintained reference list (an entity linker), not only cashtags.
- Hand-label a sample of your own headlines and measure agreement for FinBERT and VADER before using the series.
- Store model name and version with every score so the series stays comparable over time.
- Keep the disclaimer in every report or dashboard built on the output.
