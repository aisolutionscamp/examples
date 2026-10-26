import pytest
from fastapi.testclient import TestClient

from pdm.api import create_app
from pdm.data import SENSORS, generate_fleet
from pdm.model import MaintenanceModel, train


@pytest.fixture(scope="module")
def model():
    return train(generate_fleet(n_machines=24, seed=0), SENSORS, window=10, horizon=30)


def test_model_separates_healthy_from_failing_on_unseen_machines(model):
    # Synthetic data is easy; this guards against broken features or label leakage bugs,
    # it says nothing about accuracy on real equipment.
    assert model.metrics["test_machines"] == 6
    assert model.metrics["roc_auc"] > 0.9


def test_scores_rise_towards_failure(model):
    machine = generate_fleet(n_machines=1, seed=99)
    scored = model.score(machine)
    early, late = scored.head(50), scored.tail(10)
    assert late["failure_probability"].mean() > 0.7 > early["failure_probability"].mean()
    assert late["anomaly_score"].mean() > early["anomaly_score"].mean()
    assert late["rul_estimate"].mean() < early["rul_estimate"].mean()
    assert late["alert"].all()
    assert early["anomaly"].mean() < 0.1


def test_save_and_load(model, tmp_path):
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = MaintenanceModel.load(path)
    assert loaded.window == model.window and loaded.features == model.features


def _readings(df):
    return [{k: float(v) for k, v in r.items()} for r in df[["cycle", *SENSORS]].to_dict("records")]


def test_api_scores_latest_cycle(model):
    machine = generate_fleet(n_machines=1, seed=7)
    with TestClient(create_app(model)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        healthy = client.post("/score", json={"machine_id": "M7", "readings": _readings(machine.head(30))})
        failing = client.post("/score", json={"machine_id": "M7", "readings": _readings(machine.tail(30))})
    assert healthy.status_code == 200 and failing.status_code == 200
    assert healthy.json()["cycle"] == 29
    assert failing.json()["failure_probability"] > healthy.json()["failure_probability"]
    assert failing.json()["alert"] is True


def test_api_validates_input(model):
    machine = generate_fleet(n_machines=1, seed=7)
    with TestClient(create_app(model)) as client:
        short = client.post("/score", json={"machine_id": "x", "readings": _readings(machine.head(5))})
        missing = client.post("/score", json={"machine_id": "x", "readings": [{"cycle": i} for i in range(20)]})
    assert short.status_code == 422 and "at least 10" in short.json()["detail"]
    assert missing.status_code == 422 and "missing" in missing.json()["detail"]
