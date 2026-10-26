"""Analyze a CSV of reviews.

    python -m reviews samples/sample_reviews.csv --text-column review \
        --out-csv results.csv --out-json report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .analyzer import ReviewAnalyzer, summarize


def run(path: Path, text_column: str, out_csv: Path | None, out_json: Path | None) -> dict:
    analyzer = ReviewAnalyzer()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and text_column not in rows[0]:
        raise SystemExit(f"column '{text_column}' not found; columns: {', '.join(rows[0])}")
    results = [analyzer.analyze(row[text_column]) for row in rows if row[text_column].strip()]
    if out_csv:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["review", "compound", "sentiment", "aspects"])
            for r in results:
                writer.writerow([r.text, r.compound, r.sentiment, json.dumps(r.aspects)])
    report = summarize(results)
    if out_json:
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sentiment of restaurant reviews (VADER).")
    p.add_argument("csv", type=Path)
    p.add_argument("--text-column", default="review")
    p.add_argument("--out-csv", type=Path)
    p.add_argument("--out-json", type=Path)
    a = p.parse_args(argv)
    json.dump(run(a.csv, a.text_column, a.out_csv, a.out_json), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
