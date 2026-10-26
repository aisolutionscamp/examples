# Prophet + XGBoost demand forecast with price recommendation

Forecasts daily demand for one product in one store with a blend of Prophet and XGBoost, then recommends a price for the forecast horizon inside a corridor set by business rules. Optional adapters cache forecasts in Redis and read sales events from Kafka.

## What it does

1. `data.py` generates two years of synthetic daily sales with trend, weekly and yearly seasonality, temperature, holidays, promotions and a price elasticity of -1.5. Replace it with your own sales export.
2. `ensemble.py` adds calendar and price features, then trains:
   - **Prophet** with trend, weekly and yearly seasonality and three regressors (temperature, holiday flag, our price / competitor price);
   - **XGBoost** on calendar, weather and price features.
   The last 28 days are held out once to measure each model's MAPE; the blend weight is the inverse-error weight of the two models. Both models are then retrained on all history. Prediction intervals come from Prophet and are re-centred on the blended forecast.
3. `pricing.py` estimates the price elasticity from history (log-log regression with day-of-week and holiday controls) and searches a price grid. Expected units are `forecast * (price / current_price) ^ elasticity`, capped by stock on hand; the price with the highest expected profit wins. The corridor: at least 10 % margin over unit cost, at most 15 % above the competitor, at most 20 % change per update. Every decision comes with a one-line reason for store managers.
4. `adapters.py` (optional): `cache_forecast` / `get_cached_forecast` store forecasts in Redis with a 4-hour TTL; `consume_sales_events` validates JSON sales events from a Kafka topic and hands them to a callback (for example "append to history and re-forecast this SKU").

```
sales history + weather + prices ─► features ─┬─► Prophet ─┐
                                              └─► XGBoost ─┴─► weighted blend ─► price search ─► decision + reason
                                                                     │
                                                                     └─► Redis cache (optional)
Kafka sales events (optional) ─► append to history ─► re-forecast
```

## Run

```bash
pip install -r requirements.txt
cd src && python -m demand_pricing.demo      # trains on synthetic data, prints forecast and price decision
```

Tests (offline, no Redis or Kafka needed):

```bash
scripts/test-example.sh examples/18-prophet-xgboost-pricing   # from the repo root, runs in Docker
```

The tests check that the ensemble beats a naive "mean of the last 28 days" forecast on a held-out window, that the elasticity estimate recovers the value used to generate the data, that the price search respects the corridor and stock, and that the adapters serialize and validate correctly.

## Configuration

- `EnsembleForecaster(validation_days=28, prophet_weight=None, interval_width=0.8, xgb_params=...)`: pass `prophet_weight` to fix the blend instead of learning it.
- `recommend_price(..., min_margin=0.10, max_above_competitor=0.15, max_change=0.20)`: your pricing policy.
- `.env.example`: `REDIS_URL`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC` for the adapters.

## Limitations

- One model per SKU/store. For 10,000+ SKUs, train in parallel jobs (one process per SKU batch) or switch to a global model across SKUs.
- Constant elasticity is a strong simplification. Promotions, cross-price effects between products and reference-price effects are not modelled.
- Future regressors must be known: planned prices, a weather forecast and the holiday calendar. Forecast errors in the weather feed add to the demand error.
- The synthetic data is easy compared with real sales (no stock-outs, no product launches, no missing days).

## For production

- Read history from your data warehouse / ERP and write decisions back through its API; keep a human approval step for large price changes.
- Backtest with rolling origins (several cut-off dates), not one hold-out window, and track MAPE and bias per SKU.
- Run the forecast job on a schedule (for example every 4 hours) and serve decisions from the Redis cache.
- Log every price decision with inputs and reason for auditing; A/B test pricing rules before rolling them out.
