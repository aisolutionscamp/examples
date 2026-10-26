"""Read daily sales features from BigQuery and write forecasts back.

Values go in as query parameters (no string formatting of user input).
Table names cannot be parameters, so the dataset id is validated instead.
"""
import re
from datetime import date

import pandas as pd
from google.cloud import bigquery

DATASET_RE = re.compile(r"^[a-z][a-z0-9-]{4,29}\.[A-Za-z0-9_]{1,1024}$")

SALES_SQL = """
SELECT
  s.sale_date AS date,
  SUM(s.units) AS units,
  SAFE_DIVIDE(SUM(s.revenue), SUM(s.units)) AS avg_price,
  ANY_VALUE(w.avg_temp_c) AS temperature,
  IF(ANY_VALUE(h.holiday_date) IS NULL, 0, 1) AS is_holiday
FROM `{ds}.sales_daily` AS s
JOIN `{ds}.stores` AS st USING (store_id)
LEFT JOIN `{ds}.weather_daily` AS w ON w.region = st.region AND w.day = s.sale_date
LEFT JOIN `{ds}.holidays` AS h ON h.region = st.region AND h.holiday_date = s.sale_date
WHERE s.sku = @sku AND s.store_id = @store_id AND s.sale_date BETWEEN @start_date AND @end_date
GROUP BY date
ORDER BY date
"""


def check_dataset(dataset: str) -> str:
    if not DATASET_RE.match(dataset):
        raise ValueError("BQ_DATASET must look like 'project-id.dataset_name'")
    return dataset


def load_daily_sales(client: bigquery.Client, dataset: str, sku: str, store_id: str,
                     start: date, end: date) -> pd.DataFrame:
    """One row per calendar day; days without sales get 0 units."""
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("sku", "STRING", sku),
        bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
        bigquery.ScalarQueryParameter("start_date", "DATE", start),
        bigquery.ScalarQueryParameter("end_date", "DATE", end),
    ])
    sql = SALES_SQL.format(ds=check_dataset(dataset))
    df = client.query(sql, job_config=job_config).to_dataframe()
    days = pd.date_range(start, end, freq="D")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").reindex(days)
    df["units"] = df["units"].fillna(0.0)
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df[["avg_price", "temperature"]] = df[["avg_price", "temperature"]].ffill().bfill()
    return df.rename_axis("date").reset_index()


def write_forecast(client: bigquery.Client, dataset: str, sku: str, store_id: str,
                   forecast: pd.DataFrame, model_version: str) -> int:
    """Append forecast rows to `<dataset>.demand_forecasts` (streaming insert)."""
    rows = [{"sku": sku, "store_id": store_id, "forecast_date": str(r.date.date()),
             "units": float(r.units), "lower": float(r.lower), "upper": float(r.upper),
             "model_version": model_version} for r in forecast.itertuples()]
    errors = client.insert_rows_json(f"{check_dataset(dataset)}.demand_forecasts", rows)
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")
    return len(rows)
