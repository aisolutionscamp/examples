# SageMaker Autopilot demand model + Lambda pricing

Trains a demand model with **Amazon SageMaker Autopilot** (AutoML V2, tabular regression), deploys the best candidate to a real-time endpoint, and serves price recommendations from an **AWS Lambda** function behind API Gateway. Every decision is logged to CloudWatch Logs and a price-change metric goes to **Amazon CloudWatch**.

## How it works

```
sales export ─► S3 (training CSV) ─► SageMaker Autopilot job ─► best candidate ─► SageMaker endpoint
                                                                                       ▲
API Gateway ─► Lambda: validate request ─► 21 candidate prices ─► one CSV request ─────┘
                       └─► predicted units per price ─► cap at stock ─► max profit ─► JSON response
                                                                                  └─► CloudWatch metric + log
```

1. `training_data.py` creates synthetic sales history (2 SKUs x 2 stores x 365 days) in the training layout: `sku, store_id, day_of_week, month, is_holiday, price, competitor_price, price_ratio, units_sold`. Replace it with your sales export.
2. `automl.py upload` writes the CSV to S3. `automl.py start` calls `create_auto_ml_job_v2` with a `TabularJobConfig` (regression on `units_sold`, objective MAE, `ENSEMBLING` mode, 1-hour budget, 20 % validation split).
3. `automl.py deploy <job>` waits for the job, then creates a model from the best candidate's inference containers, an endpoint config (`ml.m5.large`) and the endpoint (or updates it if it exists).
4. `lambda_function.lambda_handler` accepts an API Gateway proxy event or a direct invocation:

   ```json
   {"sku": "SKU-1", "store_id": "S01", "date": "2026-11-27", "current_price": 10.0,
    "unit_cost": 6.0, "inventory": 80, "competitor_price": 10.0, "is_holiday": false}
   ```

   It builds a price corridor (at least 10 % margin, at most 15 % above the competitor, at most 20 % change), asks the endpoint for the predicted units at 21 prices in one `invoke_endpoint` call (CSV in, CSV out), forces predicted demand to be non-increasing in price, caps sales at stock, and returns the price with the highest expected profit. Bad input gives 400, endpoint errors give 502; a failed metric write never blocks the answer.

## Setup (needs an AWS account)

```bash
pip install -r requirements.txt
cp .env.example .env          # region, IAM role, bucket, endpoint name
set -a; . ./.env; set +a
cd src
python -m pricing_service.automl upload
python -m pricing_service.automl start          # prints the job name
python -m pricing_service.automl deploy demand-20261127-120000
```

Lambda: package `src/pricing_service/` with numpy (for example with the AWS SDK for pandas layer or a zip built in a Lambda base image), set the handler to `pricing_service.lambda_function.lambda_handler`, set `SAGEMAKER_ENDPOINT_NAME` and `METRICS_NAMESPACE`, and give the function role `sagemaker:InvokeEndpoint` on the endpoint and `cloudwatch:PutMetricData`. Put an API Gateway HTTP API in front of it.

Costs: the Autopilot job runs training instances for up to the configured budget, and the endpoint is billed per hour while it runs. Delete the endpoint when you are done.

## Tests (offline)

```bash
scripts/test-example.sh examples/18-sagemaker-lambda-pricing   # from the repo root, runs in Docker
```

The SageMaker and S3 calls are checked with botocore's `Stubber`, which validates every request against the real API model (wrong parameter names or types fail the test). The endpoint is replaced by a fake with a known demand curve, so the tests can check the chosen price, the corridor, the CSV sent to the model, the 400/502 paths and the metric call.

## Limitations

- One prediction per day and SKU/store. For a multi-day horizon, send one row per day and price, or train an Autopilot time-series forecasting job instead.
- The model only knows price through the training data. If prices hardly varied in the past, the learned price response is weak; run controlled price tests first.
- `np.minimum.accumulate` is a simple guard against non-monotone predictions, not a substitute for a model with a sound price response.

## For production

- Authenticate API Gateway (IAM, Cognito or an authorizer) and add throttling.
- Retrain on a schedule (EventBridge + Step Functions or SageMaker Pipelines) and compare the new candidate's MAE with the live model before switching the endpoint.
- Enable SageMaker Model Monitor data capture to see input drift.
- Store each decision (inputs, prices tried, chosen price) in S3 or DynamoDB for audits and A/B analysis.
