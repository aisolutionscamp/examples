"""Feature layout shared by the training CSV and the Lambda inference requests.

SageMaker Autopilot's inference container expects CSV rows with exactly the
training columns (minus the target) in the same order and without a header.
Only the standard library is used here, so the Lambda package stays small.
"""
from datetime import date

TARGET = "units_sold"
FEATURES = ["sku", "store_id", "day_of_week", "month", "is_holiday", "price", "competitor_price", "price_ratio"]


def feature_row(sku: str, store_id: str, day: date, is_holiday: bool, price: float,
                competitor_price: float) -> list:
    return [sku, store_id, day.weekday(), day.month, int(is_holiday), round(price, 2),
            round(competitor_price, 2), round(price / competitor_price, 4)]


def to_csv_rows(rows: list[list]) -> str:
    for row in rows:
        if len(row) != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} values per row, got {len(row)}")
        if any("," in str(v) or "\n" in str(v) for v in row):
            raise ValueError("values must not contain commas or newlines")
    return "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
