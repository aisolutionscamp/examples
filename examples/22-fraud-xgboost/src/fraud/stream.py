"""Kafka worker: read transactions, score them, publish decisions, index them.

    python -m fraud.stream

Consumer/producer follow the confluent-kafka interface (poll/produce/flush);
the Elasticsearch sink is optional. Requires requirements-adapters.txt.
"""

from __future__ import annotations

import json
import os

from .features import Transaction
from .scoring import Scorer


def process_message(value: bytes, scorer: Scorer) -> dict:
    return scorer.score(Transaction(**json.loads(value))).as_dict()


def run(consumer, producer, scorer: Scorer, output_topic: str, es=None, es_index: str = "",
        max_messages: int | None = None) -> int:
    handled = 0
    while max_messages is None or handled < max_messages:
        msg = consumer.poll(1.0)
        if msg is None:
            if max_messages is not None:
                break
            continue
        if msg.error():
            continue
        decision = process_message(msg.value(), scorer)
        producer.produce(output_topic, key=decision["transaction_id"], value=json.dumps(decision))
        if es is not None:
            es.index(index=es_index, id=decision["transaction_id"], document=decision)
        handled += 1
    producer.flush()
    return handled


def main() -> None:
    from confluent_kafka import Consumer, Producer

    from .model import load
    from .redis_store import RedisFeatureStore

    clf, threshold = load(os.environ.get("FRAUD_MODEL_PATH", "model/fraud_model.json"))
    block = float(os.environ.get("FRAUD_BLOCK_THRESHOLD") or 0.9)
    scorer = Scorer(clf, RedisFeatureStore.from_url(os.environ["REDIS_URL"]), min(threshold, block), block)
    servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    consumer = Consumer({"bootstrap.servers": servers, "group.id": "fraud-scoring",
                         "auto.offset.reset": "latest"})
    consumer.subscribe([os.environ.get("KAFKA_INPUT_TOPIC", "transactions")])
    es = None
    if os.environ.get("ELASTICSEARCH_URL"):
        from elasticsearch import Elasticsearch

        es = Elasticsearch(os.environ["ELASTICSEARCH_URL"])
    try:
        run(consumer, Producer({"bootstrap.servers": servers}), scorer,
            os.environ.get("KAFKA_OUTPUT_TOPIC", "fraud-decisions"), es,
            os.environ.get("ELASTICSEARCH_INDEX", "fraud-decisions"))
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
