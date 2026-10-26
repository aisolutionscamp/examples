"""Train and save the forecaster.

    python -m keras_forecast.train --synthetic --out models/latest
    python -m keras_forecast.train --sku SKU-1 --store S01 --out models/latest   # reads BigQuery
"""
import argparse
import json
import os
from datetime import date, timedelta

from keras_forecast.data import make_daily_sales
from keras_forecast.model import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--sku")
    p.add_argument("--store")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default="models/latest")
    args = p.parse_args()
    if args.synthetic:
        history = make_daily_sales(days=args.days)
    else:  # pragma: no cover - needs Google Cloud credentials
        from google.cloud import bigquery

        from keras_forecast.bigquery_io import load_daily_sales
        end = date.today() - timedelta(days=1)
        history = load_daily_sales(bigquery.Client(), os.environ["BQ_DATASET"], args.sku, args.store,
                                   end - timedelta(days=args.days - 1), end)
    forecaster = train(history, epochs=args.epochs)
    forecaster.save(args.out)
    print(json.dumps(forecaster.metrics, indent=2))


if __name__ == "__main__":
    main()
