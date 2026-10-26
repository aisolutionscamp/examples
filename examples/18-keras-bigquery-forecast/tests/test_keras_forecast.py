from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from google.cloud import bigquery

from keras_forecast import main
from keras_forecast.bigquery_io import check_dataset, load_daily_sales, write_forecast
from keras_forecast.data import make_daily_sales
from keras_forecast.model import HORIZON, WINDOW, Forecaster, mape, train
from keras_forecast.pricing import recommend_price

DATASET = "example-project.retail"


@pytest.fixture(scope="module")
def forecaster():
    return train(make_daily_sales(days=420, seed=3), epochs=40)


def fake_bq(frame: pd.DataFrame) -> MagicMock:
    client = MagicMock(spec=bigquery.Client)
    client.query.return_value.to_dataframe.return_value = frame
    client.insert_rows_json.return_value = []
    return client


def test_query_uses_parameters_and_fills_missing_days():
    rows = pd.DataFrame({"date": [date(2026, 1, 1), date(2026, 1, 3)], "units": [5.0, 7.0],
                         "avg_price": [9.99, 9.99], "temperature": [3.0, 4.0], "is_holiday": [1, 0]})
    client = fake_bq(rows)
    df = load_daily_sales(client, DATASET, "SKU-1'; DROP TABLE x;--", "S01", date(2026, 1, 1), date(2026, 1, 3))
    sql, kwargs = client.query.call_args.args[0], client.query.call_args.kwargs
    assert "DROP TABLE" not in sql and "@sku" in sql
    params = {p.name: p.value for p in kwargs["job_config"].query_parameters}
    assert params["sku"] == "SKU-1'; DROP TABLE x;--" and params["start_date"] == date(2026, 1, 1)
    assert df["units"].tolist() == [5.0, 0.0, 7.0]
    assert df["temperature"].tolist() == [3.0, 3.0, 4.0]


def test_dataset_name_is_validated():
    assert check_dataset(DATASET) == DATASET
    with pytest.raises(ValueError):
        check_dataset("x`; DROP")


def test_lstm_learns_weekly_pattern(forecaster):
    m = forecaster.metrics
    # the LSTM must beat "same weekday last week" on the held-out 8 weeks
    assert m["val_mape"] < m["seasonal_naive_mape"]
    assert m["val_mape"] < 35


def test_forecast_shape_intervals_and_roundtrip(forecaster, tmp_path):
    recent = make_daily_sales(days=60, seed=9)
    fc = forecaster.predict(recent)
    assert len(fc) == HORIZON and (fc["units"] >= 0).all()
    assert (fc["lower"] <= fc["units"]).all() and (fc["units"] <= fc["upper"]).all()
    assert fc["date"].iloc[0] == recent["date"].max() + pd.Timedelta(days=1)
    forecaster.save(tmp_path)
    again = Forecaster.load(tmp_path).predict(recent)
    assert np.allclose(again["units"], fc["units"], atol=1e-4)
    with pytest.raises(ValueError):
        forecaster.predict(recent.head(WINDOW - 1))


def test_write_forecast_reports_bigquery_errors(forecaster):
    fc = forecaster.predict(make_daily_sales(days=40))
    client = fake_bq(pd.DataFrame())
    assert write_forecast(client, DATASET, "SKU-1", "S01", fc, "v1") == HORIZON
    table, rows = client.insert_rows_json.call_args.args
    assert table == f"{DATASET}.demand_forecasts" and rows[0]["model_version"] == "v1"
    client.insert_rows_json.return_value = [{"index": 0, "errors": ["bad"]}]
    with pytest.raises(RuntimeError):
        write_forecast(client, DATASET, "SKU-1", "S01", fc, "v1")


def test_http_handler_end_to_end(forecaster, monkeypatch):
    history = make_daily_sales(days=WINDOW, start="2026-10-01")
    client = fake_bq(history)
    monkeypatch.setenv("BQ_DATASET", DATASET)
    monkeypatch.setattr(main, "_state", {"bq": client, "model": forecaster})
    body, status = main.handle({"sku": "SKU-1", "store_id": "S01", "current_price": 10, "unit_cost": 6,
                                "inventory": 300, "competitor_price": 10}, today=date(2026, 10, 29))
    assert status == 200 and len(body["forecast"]) == HORIZON
    assert body["forecast"][0]["date"] == "2026-10-29"
    assert 8.0 <= body["pricing"]["recommended_price"] <= 11.5
    assert client.insert_rows_json.called
    assert main.handle({"store_id": "S01"})[1] == 400


def test_pricing_and_mape():
    r = recommend_price(400, 10, 6, 200, 10)
    assert r["recommended_price"] > 10 and r["expected_units"] <= 200
    assert mape([0, 10, 20], [1, 11, 18]) == pytest.approx(10.0)
