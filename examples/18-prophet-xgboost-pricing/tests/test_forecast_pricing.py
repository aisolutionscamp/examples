import json
from types import SimpleNamespace

import numpy as np
import pytest

from demand_pricing.adapters import cache_forecast, consume_sales_events, get_cached_forecast
from demand_pricing.data import make_sales
from demand_pricing.ensemble import EnsembleForecaster, add_features, mape
from demand_pricing.pricing import estimate_elasticity, recommend_price


@pytest.fixture(scope="module")
def trained():
    data = make_sales(days=800, seed=1)
    history, future = data.iloc[:-14], data.iloc[-14:]
    model = EnsembleForecaster(validation_days=28).fit(history)
    return model, future


def test_add_features_requires_inputs():
    with pytest.raises(ValueError):
        add_features(make_sales(days=10).drop(columns=["temperature"]))
    feats = add_features(make_sales(days=10))
    assert {"day_of_week", "price_ratio", "is_weekend"} <= set(feats.columns)


def test_mape_ignores_zero_days():
    assert mape([0, 10], [5, 12]) == pytest.approx(20.0)


def test_ensemble_forecast_beats_naive_baseline(trained):
    model, future = trained
    forecast = model.predict(future.drop(columns=["sales"]))
    assert len(forecast) == 14
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert 0 <= model.prophet_weight <= 1
    assert set(model.validation_) >= {"prophet_mape", "xgboost_mape", "ensemble_mape"}
    # a model that learned seasonality should beat "same as the mean of the last 28 days"
    naive = np.full(14, make_sales(days=800, seed=1).iloc[-42:-14]["sales"].mean())
    assert mape(future["sales"], forecast["forecast"]) < mape(future["sales"], naive)


def test_elasticity_recovered_from_synthetic_history():
    assert estimate_elasticity(make_sales(days=730, elasticity=-1.5)) == pytest.approx(-1.5, abs=0.35)


def test_price_goes_up_when_demand_exceeds_stock():
    d = recommend_price(forecast_units=400, current_price=10, unit_cost=6, inventory=200,
                        competitor_price=10, elasticity=-1.5)
    assert d.recommended_price > 10
    assert d.recommended_price <= d.price_ceiling == pytest.approx(11.5)
    assert d.expected_profit >= d.profit_at_current_price
    assert "exceeds stock" in d.reason


def test_price_drop_is_limited_by_max_change_when_overstocked():
    # profit-optimal price would be 7.50, but the corridor allows at most 20 % down
    d = recommend_price(forecast_units=50, current_price=10, unit_cost=5, inventory=500,
                        competitor_price=10, elasticity=-3.0)
    assert d.price_floor == pytest.approx(8.0)
    assert d.recommended_price == pytest.approx(8.0)
    assert "above forecast" in d.reason


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        recommend_price(-1, 10, 6, 10, 10, -1.5)


class FakeRedis:
    def __init__(self):
        self.store, self.ttl = {}, {}

    def setex(self, key, ttl, value):
        self.store[key], self.ttl[key] = value, ttl

    def get(self, key):
        return self.store.get(key)


def test_redis_cache_roundtrip(trained):
    model, future = trained
    forecast = model.predict(future.drop(columns=["sales"]))
    r = FakeRedis()
    key = cache_forecast(r, "SKU1", "S01", forecast)
    assert key == "forecast:S01:SKU1" and r.ttl[key] == 4 * 3600
    back = get_cached_forecast(r, "SKU1", "S01")
    assert np.allclose(back["forecast"], forecast["forecast"])
    assert get_cached_forecast(r, "SKU2", "S01") is None


def test_consume_sales_events_validates_messages():
    good = {"sku": "A", "store_id": "S01", "date": "2026-01-01", "units": 3, "price": 9.99}
    msgs = [SimpleNamespace(value=json.dumps(good).encode()), SimpleNamespace(value=b"not json"),
            SimpleNamespace(value=json.dumps({"sku": "A"}).encode())]
    seen = []
    stats = consume_sales_events(msgs, seen.append)
    assert stats == {"processed": 1, "rejected": 2} and seen == [good]
    assert consume_sales_events(msgs, seen.append, max_events=1) == {"processed": 1, "rejected": 0}
