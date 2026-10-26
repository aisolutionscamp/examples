# Email classification microservice with Hugging Face Transformers + FastAPI

A self-hosted service that classifies support emails by **urgency** (HIGH, MEDIUM, LOW) and **department** (TECHNICAL, BILLING, SALES, SUPPORT) with two fine-tuned Transformer models. No external AI API is called; the emails stay on your servers.

## Architecture

```
POST /classify {subject, body}
  -> Redis cache lookup (sha256 of subject + body, 1 h TTL; skipped if REDIS_URL is not set)
  -> clean_email           drop quoted replies and signature, collapse whitespace
  -> urgency model         AutoModelForSequenceClassification (fine-tuned DistilBERT multilingual)
  -> department model      AutoModelForSequenceClassification
  -> merge                 labels, confidences, needs_review = min(confidence) < MIN_CONFIDENCE
  -> JSON response, metrics counters (GET /metrics)

train.py: CSV (subject, body, urgency, department) -> fine-tune both models -> models/urgency, models/department
```

- `app/classifier.py`: model loading and prediction
- `app/main.py`: FastAPI app, Redis cache, `/classify`, `/health`, `/metrics`
- `train.py`: fine-tuning with a plain PyTorch loop, held-out accuracy and macro F1
- `data/sample_emails.csv`: 24 synthetic emails in English, Spanish and German, only to show the format

## Train

You need labeled emails: at least a few hundred per task, better a few thousand, exported from your ticket system. The 24 sample rows are far too few for a useful model.

```bash
pip install -r requirements.txt
python train.py data/your_emails.csv --out models --epochs 4
```

The base model `distilbert/distilbert-base-multilingual-cased` (about 540 MB) is downloaded from the Hugging Face Hub on the first run. Training on a CPU works for a few thousand emails; a GPU is much faster (install the CUDA build of PyTorch instead of the CPU wheel in `requirements.txt`). The script prints the loss per epoch and accuracy and macro F1 on a 20 % held-out split.

## Run

```bash
docker compose up --build            # API on http://localhost:8000, Redis next to it
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"subject": "API 500 errors", "body": "Every call to /v2/orders fails since this morning."}'
```

Interactive API docs: `http://localhost:8000/docs`. Without Docker: `uvicorn app.main:app` (no cache unless `REDIS_URL` is set).

## Tests

```bash
pytest
```

The tests do not download anything. They build a tiny BERT model (1 layer, 32 hidden units) with a vocabulary made from the sample CSV, fine-tune it with the same `fine_tune()` function, save and reload it, and call the API with an in-memory Redis (`fakeredis`). They check that training lowers the loss, that the service returns valid labels and confidences, that the cache works, and that a broken Redis does not break classification. They run on a CPU in under a minute.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `URGENCY_MODEL_DIR` | `models/urgency` | fine-tuned urgency model |
| `DEPARTMENT_MODEL_DIR` | `models/department` | fine-tuned department model |
| `REDIS_URL` | not set (no cache) | for example `redis://localhost:6379/0` |
| `MIN_CONFIDENCE` | `0.6` | below this, `needs_review` is true |
| `BASE_MODEL` | `distilbert/distilbert-base-multilingual-cased` | base model for `train.py` |

## Limitations

- Model quality depends entirely on your training data. No accuracy figure is given here because none has been measured on real support emails.
- The softmax confidence of a fine-tuned model is often overconfident. Check `needs_review` against a validation set before you route automatically.
- Inputs are truncated to 256 tokens (subject plus the start of the body).
- Metrics are per process and reset on restart.

## For production

- Export the models to ONNX (`optimum`) for faster CPU inference, or run several workers behind a load balancer.
- Replace `/metrics` with Prometheus (`prometheus-fastapi-instrumentator`) and track the label distribution over time to notice drift.
- Store agent corrections from your ticket system and retrain regularly with `train.py`.
- Put the API behind authentication; it is open by default.
