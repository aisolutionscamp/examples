import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient

from app.classifier import EmailClassifier, clean_email
from app.main import app

URGENCY = {"HIGH", "MEDIUM", "LOW"}
DEPARTMENTS = {"TECHNICAL", "BILLING", "SALES", "SUPPORT"}


@pytest.fixture(scope="module")
def classifier(trained):
    return EmailClassifier(trained["urgency"][0], trained["department"][0], min_confidence=0.6)


@pytest.fixture
def client(classifier):
    app.state.classifier = classifier
    app.state.cache = fakeredis.FakeRedis()
    with TestClient(app) as c:
        yield c


def test_clean_email():
    text = "Hello,\n  the API fails.\n> old quote\nThanks\n--\nJane\nOn Mon someone wrote:\nold"
    assert clean_email(text) == "Hello, the API fails. Thanks"


def test_classifier_loads_saved_models_and_predicts(classifier):
    result = classifier.classify("API 500 errors", "Since the last release every call returns HTTP 500.")
    assert result["urgency"] in URGENCY and result["department"] in DEPARTMENTS
    assert abs(sum(result["scores"]["department"].values()) - 1) < 1e-3
    assert result["overall_confidence"] == min(result["urgency_confidence"], result["department_confidence"])
    assert result["needs_review"] == (result["overall_confidence"] < 0.6)
    # A training example is classified as labeled (the tiny model memorized the sample set).
    assert result["department"] == "TECHNICAL"


def test_classify_endpoint_caches_results(client):
    email = {"subject": "Charged twice", "body": "I was charged twice for the March invoice."}
    first = client.post("/classify", json=email).json()
    second = client.post("/classify", json=email).json()
    assert first["cached"] is False and second["cached"] is True
    assert {k: v for k, v in first.items() if k != "cached"} == {k: v for k, v in second.items() if k != "cached"}
    m = client.get("/metrics").json()
    assert m["requests"] == 2 and m["cache_hits"] == 1 and sum(m["department"].values()) == 2


def test_validation_and_health(client):
    assert client.post("/classify", json={"subject": "x", "body": ""}).status_code == 422
    health = client.get("/health").json()
    assert set(health["urgency_labels"]) == URGENCY and set(health["department_labels"]) == DEPARTMENTS


class BrokenCache:
    def get(self, key):
        raise redis.ConnectionError("down")

    def set(self, *args, **kwargs):
        raise redis.ConnectionError("down")


def test_broken_cache_does_not_break_classification(classifier):
    app.state.classifier = classifier
    app.state.cache = BrokenCache()
    with TestClient(app) as c:
        resp = c.post("/classify", json={"body": "How do I upgrade to the business plan?"})
    assert resp.status_code == 200 and resp.json()["cached"] is False
