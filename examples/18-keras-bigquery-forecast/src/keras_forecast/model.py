"""Small Keras LSTM: last 28 days of features in, next 7 days of units out.

Intervals come from the empirical 10th/90th percentile of validation errors
per horizon day, so they reflect how wrong the model actually was.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

WINDOW, HORIZON = 28, 7
FEATURES = ["units", "avg_price", "temperature", "is_holiday", "dow_sin", "dow_cos"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    dow = pd.to_datetime(df["date"]).dt.day_of_week
    return df.assign(dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7))


def make_windows(values: np.ndarray, target: np.ndarray):
    x = [values[i - WINDOW:i] for i in range(WINDOW, len(values) - HORIZON + 1)]
    y = [target[i:i + HORIZON] for i in range(WINDOW, len(values) - HORIZON + 1)]
    return np.asarray(x, "float32"), np.asarray(y, "float32")


def mape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = actual > 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def build_model(n_features: int, units: int = 32) -> keras.Model:
    model = keras.Sequential([
        keras.Input((WINDOW, n_features)),
        keras.layers.LSTM(units),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dense(HORIZON),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


@dataclass
class Forecaster:
    model: keras.Model
    mean: np.ndarray
    std: np.ndarray
    residual_q: np.ndarray  # shape (2, HORIZON): 10th and 90th percentile of errors
    metrics: dict

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        return (add_calendar(df)[FEATURES].to_numpy(float) - self.mean) / self.std

    def predict(self, recent: pd.DataFrame) -> pd.DataFrame:
        """recent: at least WINDOW days ending yesterday; returns the next HORIZON days."""
        if len(recent) < WINDOW:
            raise ValueError(f"need {WINDOW} days of history, got {len(recent)}")
        x = self._scale(recent.tail(WINDOW))[None].astype("float32")
        y = self.model.predict(x, verbose=0)[0] * self.std[0] + self.mean[0]
        y = np.clip(y, 0, None)
        start = pd.to_datetime(recent["date"]).max() + pd.Timedelta(days=1)
        return pd.DataFrame({"date": pd.date_range(start, periods=HORIZON, freq="D"), "units": y,
                             "lower": np.clip(y + self.residual_q[0], 0, None), "upper": y + self.residual_q[1]})

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        (directory / "meta.json").write_text(json.dumps({
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "residual_q": self.residual_q.tolist(), "metrics": self.metrics}))

    @classmethod
    def load(cls, directory: str | Path) -> "Forecaster":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        return cls(keras.models.load_model(directory / "model.keras"), np.array(meta["mean"]),
                   np.array(meta["std"]), np.array(meta["residual_q"]), meta["metrics"])


def train(history: pd.DataFrame, epochs: int = 40, val_days: int = 56, seed: int = 0) -> Forecaster:
    keras.utils.set_random_seed(seed)
    values = add_calendar(history)[FEATURES].to_numpy(float)
    split = len(values) - val_days
    mean, std = values[:split].mean(0), values[:split].std(0) + 1e-6
    x, y = make_windows((values - mean) / std, (values[:, 0] - mean[0]) / std[0])
    n_val = val_days - HORIZON + 1  # windows whose targets fall in the validation period
    x_tr, y_tr, x_val, y_val = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]

    model = build_model(len(FEATURES))
    model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    pred = np.clip(model.predict(x_val, verbose=0) * std[0] + mean[0], 0, None)
    actual = y_val * std[0] + mean[0]
    naive = x_val[:, -HORIZON:, 0] * std[0] + mean[0]  # same weekday one week earlier
    residual_q = np.percentile(actual - pred, [10, 90], axis=0)
    metrics = {"val_mape": mape(actual, pred), "seasonal_naive_mape": mape(actual, naive),
               "val_windows": int(n_val)}
    return Forecaster(model, mean, std, residual_q, metrics)
