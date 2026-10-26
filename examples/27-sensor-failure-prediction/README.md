# Predictive maintenance: failure risk and anomaly score from sensor time series

Use case: [Predictive Maintenance for Manufacturing Equipment](https://aisolutionscamp.io/projects/27)

Trains three scikit-learn models on run-to-failure sensor history and serves them with FastAPI:

| Output | Model | Meaning |
| --- | --- | --- |
| `failure_probability` | `HistGradientBoostingClassifier` | probability of failure within the next `horizon` cycles (default 30) |
| `rul_estimate` | `HistGradientBoostingRegressor` | rough remaining useful life in cycles, capped at 150 |
| `anomaly_score`, `anomaly` | `IsolationForest` on healthy windows | how unusual the recent behaviour is; flagged above the 99th percentile of healthy windows |
| `alert` | rule | `failure_probability >= 0.5` or `anomaly` |

## Architecture

```
sensor readings (machine_id, cycle, sensors...)
  -> rolling-window features per machine (last, mean, std, slope over 10 cycles; past only)
  -> training, split by machine:
       classifier (fails within 30 cycles)
       regressor (capped RUL)
       IsolationForest (healthy windows only)
  -> model.joblib
  -> FastAPI POST /score -> alerts / work orders
```

- `src/pdm/data.py` synthetic fleet generator and a C-MAPSS loader
- `src/pdm/features.py` rolling features
- `src/pdm/model.py` training, evaluation, scoring
- `src/pdm/api.py` FastAPI app
- `src/pdm/train.py` CLI

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m pdm.train --out ../model.joblib            # synthetic data, prints test metrics
MODEL_PATH=../model.joblib uvicorn pdm.api:app --port 8000
```

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "machine_id": "M001",
  "readings": [{"cycle": 0, "temperature": 70.1, "vibration": 0.52, "pressure": 30.2, "rpm": 1501}, ...]
}'
```

At least `window` (10) readings are needed. The response describes the latest cycle.

### Try it on NASA C-MAPSS

Download the Turbofan Engine Degradation Simulation data set (C-MAPSS) from the
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and run:

```bash
python -m pdm.train --cmapss data/train_FD001.txt --out ../model.joblib
```

This example has not been benchmarked on C-MAPSS; the printed metrics are what you get.
Note that the C-MAPSS test files are truncated before failure and need the separate
`RUL_FD00x.txt` files for evaluation; the loader here only reads the training files.

## Configuration

| CLI flag | Default | Meaning |
| --- | --- | --- |
| `--window` | 10 | cycles per rolling window |
| `--horizon` | 30 | "fails soon" means RUL <= horizon |
| `--machines` | 60 | synthetic machines when no file is given |
| `MODEL_PATH` (env, API) | `model.joblib` | model file loaded on startup |

## Tests

```bash
pytest           # or from the repo root: scripts/test-example.sh examples/27-sensor-failure-prediction
```

The tests check the generator, exact rolling-feature values, that features never use future
readings, the C-MAPSS parser, that risk and anomaly scores rise towards failure on an unseen
synthetic machine, and the API (including input validation).

## Limitations

- The synthetic wear pattern is simple and makes the models look better than they will be on real data.
- Run-to-failure labels are rare in practice: machines get serviced before they fail. Treat
  preventive replacements as censored data, not as failures.
- One model for all machines assumes they are comparable. Different machine types or operating
  modes need normalisation per type or separate models.
- No handling of missing readings, sensor drift or recalibration.

## For production

- Read from your historian or IoT platform (OPC UA, MQTT, a time-series DB) and resample to a fixed interval.
- Add authentication to the API and log every score with the model version.
- Set the alert threshold with the maintenance team based on the cost of a missed failure versus a needless inspection.
- Monitor the alert rate and the share of alerts confirmed by technicians; retrain when new failures are recorded.
