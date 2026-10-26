# Keras LSTM demand forecast on BigQuery data

Reads daily sales, weather and holidays for one SKU/store from **Google BigQuery**, trains a small **TensorFlow/Keras** LSTM that predicts the next 7 days, serves forecasts from an HTTP function (Cloud Run functions, via `functions-framework`) and writes them back to BigQuery, where Looker Studio or any BI tool can read them. An optional pricing step turns the 7-day forecast into a price recommendation.

## How it works

```
BigQuery: sales_daily + stores + weather_daily + holidays
   │  parameterized SQL (bigquery_io.load_daily_sales), missing days filled with 0
   ▼
features: units, price, temperature, holiday, day-of-week (sin/cos), standardized
   ▼
Keras: LSTM(32) ─► Dense(32) ─► Dense(7)       (train.py, model saved as .keras + meta.json)
   ▼
HTTP function forecast_http ─► 7-day forecast with 10/90 % error band ─► BigQuery demand_forecasts
                            └─► optional price recommendation (corridor + elasticity)
```

- **Model** (`model.py`): input is the last 28 days of 6 features, output the next 7 days of units. Loss is MAE, early stopping on the last 8 weeks. The forecast band is the 10th/90th percentile of the validation errors per horizon day, not a model assumption.
- **Baseline**: training reports the validation MAPE of the LSTM and of a seasonal-naive forecast ("same weekday last week"). Only use the model where it beats the baseline.
- **BigQuery** (`bigquery_io.py`): all values (SKU, store, dates) are query parameters; the dataset name is validated because table names cannot be parameters. Forecasts are appended with `insert_rows_json`.
- **Pricing** (`pricing.py`): grid search for the most profitable price with constant elasticity, capped by stock, inside a corridor (10 % minimum margin, at most 15 % above competitor, at most 20 % change).

Expected BigQuery tables (in `BQ_DATASET`): `sales_daily(sku, store_id, sale_date DATE, units, revenue)`, `stores(store_id, region)`, `weather_daily(region, day DATE, avg_temp_c)`, `holidays(region, holiday_date DATE)`, `demand_forecasts(sku, store_id, forecast_date DATE, units, lower, upper, model_version)`.

## Run locally (synthetic data, no Google account)

```bash
pip install -r requirements.txt
cd src
python -m keras_forecast.train --synthetic --out models/latest   # prints val MAPE vs seasonal naive
```

## Run against BigQuery (needs a Google Cloud project)

```bash
cp .env.example .env && set -a && . ./.env && set +a
gcloud auth application-default login
cd src
python -m keras_forecast.train --sku SKU-1 --store S01 --out models/latest
PYTHONPATH=. MODEL_DIR=models/latest functions-framework --source keras_forecast/main.py --target forecast_http
curl -X POST localhost:8080 -H 'Content-Type: application/json' \
  -d '{"sku":"SKU-1","store_id":"S01","current_price":10,"unit_cost":6,"inventory":300,"competitor_price":10}'
```

To deploy as a Cloud Run function, put a `main.py` at the source root with `from keras_forecast.main import forecast_http`, include the model directory, and give the service account `roles/bigquery.jobUser` plus `roles/bigquery.dataEditor` on the dataset. TensorFlow makes the image large (about 1.8 GB of Python packages); give the function at least 2 GiB memory and expect slower cold starts.

## Tests (offline)

```bash
scripts/test-example.sh examples/18-keras-bigquery-forecast   # from the repo root, runs in Docker
```

The BigQuery client is a mock. The tests train the real LSTM on CPU on 420 days of synthetic data (about 15 seconds) and check that it beats the seasonal-naive baseline on the hold-out period, that intervals are ordered, that save/load gives the same forecast, that SQL values are passed as parameters, and that the HTTP handler returns and stores a forecast.

## Why TensorFlow

TensorFlow/Keras is the point of this solution, so it stays, as `tensorflow-cpu` (no GPU needed). If install size matters more than the framework, the same model shape can be trained with scikit-learn or LightGBM on lag features; see the Prophet + XGBoost solution for that route.

## Limitations

- One model per SKU/store. For many SKUs, train a global model with SKU and store embeddings, or use BigQuery ML `ARIMA_PLUS` for a SQL-only baseline.
- Future prices, weather and holidays are not model inputs, only the past 28 days are. Planned promotions are therefore not anticipated.
- `insert_rows_json` uses the streaming API (small cost, rows visible within seconds). For large batches use a load job.

## For production

- Retrain on a schedule (Cloud Scheduler + Cloud Run job or Vertex AI Pipelines), store models in Cloud Storage with a version, and log `val_mape` next to the baseline.
- Authenticate the HTTP function (no unauthenticated invocations) and validate SKU/store against your master data.
- Alert in Cloud Monitoring when forecast error over the last weeks exceeds the baseline.
