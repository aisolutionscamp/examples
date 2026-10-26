import json

from fastapi.testclient import TestClient

from fraud.api import create_app
from fraud.features import InMemoryFeatureStore, Transaction
from fraud.scoring import Scorer
from fraud.stream import run as run_stream

T0 = 1_767_225_600.0


def history(store, n=20):
    for i in range(n):
        store.record(Transaction(f"h{i}", "u1", 35.0 + i % 5, "DE", "phone", "grocery",
                                 T0 - 86400 * (n - i) / 2))


def test_burst_scores_higher_than_normal_purchase(trained):
    clf = trained[0]
    store = InMemoryFeatureStore()
    history(store)
    scorer = Scorer(clf, store, review_threshold=0.5, block_threshold=0.9)
    normal = scorer.score(Transaction("n1", "u1", 38.0, "DE", "phone", "grocery", T0 + 3600))
    burst = [scorer.score(Transaction(f"f{i}", "u1", 450.0, "NG", "unknown", "gift_cards",
                                      T0 + 7200 + 60 * i)) for i in range(4)]
    assert normal.action == "approve"
    assert burst[-1].score > normal.score
    assert burst[-1].action in {"review", "block"}
    assert burst[-1].reasons
    assert burst[-1].latency_ms >= 0


def test_api_scores_and_validates(trained):
    store = InMemoryFeatureStore()
    history(store)
    client = TestClient(create_app(Scorer(trained[0], store, 0.5, 0.9)))
    assert client.get("/health").json() == {"status": "ok"}
    body = {"transaction_id": "x1", "user_id": "u1", "amount": 40, "country": "DE",
            "device_id": "phone", "merchant_category": "grocery", "timestamp": T0 + 10}
    r = client.post("/score", json=body)
    assert r.status_code == 200
    assert set(r.json()) == {"transaction_id", "score", "action", "reasons", "latency_ms"}
    assert client.post("/score", json={**body, "amount": -1}).status_code == 422


class FakeMsg:
    def __init__(self, value):
        self._v = value

    def error(self):
        return None

    def value(self):
        return self._v


class FakeConsumer:
    def __init__(self, values):
        self.values = list(values)

    def poll(self, timeout):
        return FakeMsg(self.values.pop(0)) if self.values else None


class FakeProducer:
    def __init__(self):
        self.sent, self.flushed = [], False

    def produce(self, topic, key, value):
        self.sent.append((topic, key, json.loads(value)))

    def flush(self):
        self.flushed = True


class FakeES:
    def __init__(self):
        self.docs = {}

    def index(self, index, id, document):
        self.docs[(index, id)] = document


def test_stream_worker_publishes_and_indexes(trained):
    msgs = [json.dumps({"transaction_id": f"k{i}", "user_id": "u9", "amount": 20.0, "country": "FR",
                        "device_id": "d", "merchant_category": "books", "timestamp": T0 + i}).encode()
            for i in range(3)]
    producer, es = FakeProducer(), FakeES()
    handled = run_stream(FakeConsumer(msgs), producer, Scorer(trained[0], InMemoryFeatureStore(), 0.5, 0.9),
                         "decisions", es, "idx", max_messages=10)
    assert handled == 3 and producer.flushed
    assert [s[1] for s in producer.sent] == ["k0", "k1", "k2"]
    assert ("idx", "k2") in es.docs
