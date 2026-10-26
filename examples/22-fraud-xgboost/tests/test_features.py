import pytest

from fraud.features import FEATURE_NAMES, InMemoryFeatureStore, Transaction, compute_features, features_for

T0 = 1_767_225_600.0


def txn(i, amount=40.0, country="DE", device="d1", category="grocery", ts=0.0):
    return Transaction(f"t{i}", "u1", amount, country, device, category, T0 + ts)


def feat(vector, name):
    return vector[FEATURE_NAMES.index(name)]


def test_first_transaction_has_neutral_history():
    v = compute_features(txn(0), [], None)
    assert len(v) == len(FEATURE_NAMES)
    assert feat(v, "txn_count_1h") == 0
    assert feat(v, "foreign_country") == 0
    assert feat(v, "amount_to_user_mean") == 1.0


def test_burst_abroad_on_new_device_is_visible_in_features():
    store = InMemoryFeatureStore()
    for i in range(10):
        store.record(txn(i, ts=-86400 * (10 - i)))
    for i in range(3):
        store.record(txn(100 + i, amount=500, country="BR", device="new", ts=i * 60))
    v = features_for(txn(200, amount=600, country="BR", device="new", category="electronics", ts=240), store)
    assert feat(v, "txn_count_10m") == 3
    assert feat(v, "foreign_country") == 1
    assert feat(v, "distinct_countries_24h") == 1  # earlier home txns are older than 24 h
    assert feat(v, "new_device") == 0  # device was seen in the burst already
    assert feat(v, "high_risk_category") == 1
    assert feat(v, "amount_to_user_mean") > 3


def test_point_in_time_ignores_future_transactions():
    later = txn(1, amount=999, ts=100)
    v = compute_features(txn(0, ts=0), [later], None)
    assert feat(v, "txn_count_1h") == 0


def test_profile_running_mean():
    store = InMemoryFeatureStore()
    for i, amount in enumerate([10, 20, 30]):
        store.record(txn(i, amount=amount, ts=i))
    p = store.profile("u1")
    assert p.txn_count == 3 and p.amount_mean == pytest.approx(20)
    assert p.home_country == "DE"
