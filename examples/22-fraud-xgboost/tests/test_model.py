import numpy as np

from fraud import model as fm
from fraud.train import run


def test_model_separates_synthetic_fraud(trained):
    clf, x_test, y_test = trained
    proba = clf.predict_proba(x_test)[:, 1]
    metrics = fm.evaluate(y_test, proba, 0.5)
    assert y_test.sum() > 10
    assert metrics.roc_auc > 0.95
    assert metrics.recall > 0.7


def test_threshold_respects_max_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(50)]
    proba = np.r_[rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1, 50)]
    t = fm.threshold_for_fpr(y, proba, 0.01)
    fpr = ((proba >= t) & (y == 0)).sum() / 1000
    assert fpr <= 0.01
    assert fm.threshold_for_fpr(y, proba, 0.0) > proba[y == 0].max()


def test_train_cli_saves_loadable_model(tmp_path):
    out = tmp_path / "m.json"
    report = run(str(out), max_fpr=0.01, n_users=200, seed=5)
    assert report["test"]["false_positive_rate"] <= 0.05
    clf, threshold = fm.load(out)
    assert 0 < threshold <= 1
    assert clf.predict_proba(np.zeros((1, 12), dtype=np.float32)).shape == (1, 2)
