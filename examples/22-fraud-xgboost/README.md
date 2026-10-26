# Real-time fraud scoring with XGBoost

Scores card-not-present payments before authorization. For every transaction the service looks up the user's
recent transactions and profile in a feature store, computes velocity and behavior features, scores them with an
XGBoost model and returns `approve`, `review` (step-up verification) or `block`, with the features that pushed the
score up as reasons.

The model is trained and tested on a **synthetic** transaction stream (`src/fraud/synth.py`). The numbers it
prints are not a statement about real fraud detection quality.

## Architecture

```
payment service ──► POST /score (FastAPI) ─┐
Kafka "transactions" ──► stream worker ────┤
                                           ├─► feature store (memory or Redis): recent txns + user profile
                                           ├─► features: velocity 10 min / 1 h, amount vs. user mean,
                                           │   foreign country, new device, countries in 24 h, ...
                                           ├─► XGBoost score + SHAP contributions (reasons)
                                           └─► approve / review / block ──► response, Kafka "fraud-decisions",
                                                                           Elasticsearch index (optional)
labeled history ──► replay in time order (same feature code) ──► time-based split ──► train + threshold
```

- `features.py`: transaction/profile types, the feature store interface, `compute_features`. Features use only
  transactions *before* the current one (point-in-time correct), and the same code serves training and scoring.
- `model.py`: build the training set by replaying history, train `XGBClassifier` (`scale_pos_weight` for class
  imbalance), pick the threshold for a target false-positive rate, evaluate, save/load.
- `scoring.py`: the real-time `Scorer` (score, reasons from `pred_contribs`, decision, then record the transaction).
- `api.py`: FastAPI app (`/score`, `/health`).
- `redis_store.py`, `stream.py`: optional Redis feature store and Kafka/Elasticsearch worker.

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m fraud.train --out ../model/fraud_model.json --max-fpr 0.005   # synthetic data, prints metrics
FRAUD_MODEL_PATH=../model/fraud_model.json uvicorn fraud.api:app --port 8000
curl -s localhost:8000/score -H 'content-type: application/json' -d '{"transaction_id":"t1","user_id":"u1",
  "amount":49.9,"country":"DE","device_id":"phone-1","merchant_category":"fashion"}'
```

Response: `{"transaction_id": "t1", "score": 0.0012, "action": "approve", "reasons": [...], "latency_ms": 1.2}`.

With Redis and Kafka (`pip install -r requirements-adapters.txt`, see `.env.example`):

```bash
REDIS_URL=redis://localhost:6379/0 KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python -m fraud.stream
```

Tests (offline; Kafka, Redis and Elasticsearch are replaced by small fakes): `pytest`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FRAUD_MODEL_PATH` | `model/fraud_model.json` | XGBoost model; the threshold is stored next to it in `*.meta.json` |
| `FRAUD_REVIEW_THRESHOLD` | from training | score at which step-up verification is requested |
| `FRAUD_BLOCK_THRESHOLD` | `0.9` | score at which the payment is declined |
| `REDIS_URL` | unset | use the Redis feature store instead of process memory |
| `KAFKA_*`, `ELASTICSEARCH_*` | see `.env.example` | stream worker topics and decision index |

`train(..., n_jobs=2)` sets XGBoost threads explicitly; "all cores" can be very slow on shared machines.

## Limitations

- Synthetic data: fraud bursts with new devices abroad are easy to separate, so ROC AUC near 1.0 is expected here
  and meaningless for production.
- The in-memory store is per process and lost on restart. Use Redis (or another shared store) for more than one
  worker.
- Labels in real life arrive late (chargebacks after weeks). Train only on periods whose labels are mature.
- No merchant-level, card-BIN, IP or device-fingerprint features; these usually matter a lot.

## For production

- Train on your own labeled transactions and chargebacks; validate on a later time period, never a random split.
- Add features from your payment gateway (IP geolocation, BIN country, device fingerprint, merchant history).
- Monitor score distribution, review rate and confirmed fraud per day; retrain on a schedule and when they drift.
- Keep a human review queue for `review` decisions and feed outcomes back as labels.
- Check regulatory duties for automated decisions (for example GDPR Art. 22 and PSD2 strong customer
  authentication rules) and keep the reasons for each decision.
