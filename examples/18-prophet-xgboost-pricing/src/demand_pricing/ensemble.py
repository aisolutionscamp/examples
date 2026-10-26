"""Prophet + XGBoost ensemble for daily demand of one SKU/store.

Prophet models trend, weekly/yearly seasonality and the external regressors.
XGBoost learns non-linear effects of calendar, weather and price features.
The two forecasts are blended with weights taken from a hold-out window.
"""
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

PROPHET_REGRESSORS = ["temperature", "is_holiday", "price_ratio"]
XGB_FEATURES = ["day_of_week", "month", "day_of_year", "is_weekend", "temperature",
                "is_holiday", "our_price", "price_ratio"]
INPUT_COLUMNS = ["date", "our_price", "competitor_price", "temperature", "is_holiday"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and price features. Future rows need planned prices and a weather forecast."""
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.day_of_week
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.day_of_year
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_holiday"] = out["is_holiday"].astype(int)
    out["price_ratio"] = out["our_price"] / out["competitor_price"]
    return out


def mape(actual, predicted) -> float:
    """Mean absolute percentage error in %, ignoring days with zero sales."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = actual > 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


class EnsembleForecaster:
    def __init__(self, validation_days: int = 28, prophet_weight: float | None = None,
                 interval_width: float = 0.8, xgb_params: dict | None = None):
        self.validation_days = validation_days
        self.prophet_weight = prophet_weight
        self.interval_width = interval_width
        # n_jobs=1: on small data, OpenMP threads cost more than they gain (and stall on busy hosts)
        self.xgb_params = xgb_params or {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                                         "subsample": 0.9, "random_state": 0, "n_jobs": 1}
        self.validation_: dict | None = None

    def _fit_models(self, df: pd.DataFrame):
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                          changepoint_prior_scale=0.05, interval_width=self.interval_width)
        for name in PROPHET_REGRESSORS:
            prophet.add_regressor(name)
        prophet.fit(df[["date", "sales", *PROPHET_REGRESSORS]].rename(columns={"date": "ds", "sales": "y"}))
        booster = xgb.XGBRegressor(objective="reg:squarederror", **self.xgb_params)
        booster.fit(df[XGB_FEATURES], df["sales"])
        return prophet, booster

    @staticmethod
    def _predict_models(models, df: pd.DataFrame):
        prophet, booster = models
        p = prophet.predict(df[["date", *PROPHET_REGRESSORS]].rename(columns={"date": "ds"}))
        return (p["yhat"].to_numpy(), booster.predict(df[XGB_FEATURES]).astype(float),
                p["yhat_lower"].to_numpy(), p["yhat_upper"].to_numpy())

    def fit(self, history: pd.DataFrame) -> "EnsembleForecaster":
        df = add_features(history)
        if self.prophet_weight is None:
            train, val = df.iloc[:-self.validation_days], df.iloc[-self.validation_days:]
            p, x, _, _ = self._predict_models(self._fit_models(train), val)
            mp, mx = mape(val["sales"], p), mape(val["sales"], x)
            # inverse-error weighting: the better model gets the larger weight
            self.prophet_weight = mx / (mp + mx) if mp + mx > 0 else 0.5
            blend = self.prophet_weight * p + (1 - self.prophet_weight) * x
            self.validation_ = {"prophet_mape": mp, "xgboost_mape": mx,
                                "ensemble_mape": mape(val["sales"], blend), "days": self.validation_days}
        self.models_ = self._fit_models(df)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        df = add_features(future)
        p, x, lower, upper = self._predict_models(self.models_, df)
        w = self.prophet_weight
        forecast = np.clip(w * p + (1 - w) * x, 0, None)
        return pd.DataFrame({
            "date": df["date"].to_numpy(),
            "prophet": p,
            "xgboost": x,
            "forecast": forecast,
            # Prophet's interval width, centred on the blended forecast
            "lower": np.clip(forecast - (p - lower), 0, None),
            "upper": forecast + (upper - p),
        })
