import numpy as np
import pandas as pd

from pdm.data import SENSORS, add_failure_label, generate_fleet, load_cmapss
from pdm.features import feature_columns, rolling_features


def test_generator_runs_to_failure():
    df = generate_fleet(n_machines=3, seed=1)
    assert set(df.columns) >= {"machine_id", "cycle", "rul", *SENSORS}
    for _, g in df.groupby("machine_id"):
        assert g["rul"].iloc[-1] == 0
        # wear makes the end of life hotter and shakier than the start
        assert g["temperature"].tail(10).mean() > g["temperature"].head(10).mean() + 5
        assert g["vibration"].tail(10).mean() > g["vibration"].head(10).mean() + 1


def test_failure_label_uses_horizon():
    df = add_failure_label(generate_fleet(n_machines=1, seed=2), horizon=30)
    assert df["fails_soon"].sum() == 31  # rul 0..30


def test_rolling_features_drop_incomplete_windows_and_match_manual_values():
    df = pd.DataFrame({"machine_id": ["A"] * 6 + ["B"] * 6, "cycle": list(range(6)) * 2,
                       "x": [0, 1, 2, 3, 4, 5, 10, 10, 10, 10, 10, 10.0]})
    out = rolling_features(df, ["x"], window=3)
    assert len(out) == 8  # 4 full windows per machine
    a = out[out["machine_id"] == "A"]
    assert a["x_mean"].tolist() == [1, 2, 3, 4]
    assert np.allclose(a["x_slope"], 1.0)
    b = out[out["machine_id"] == "B"]
    assert np.allclose(b["x_std"], 0) and np.allclose(b["x_slope"], 0)


def test_features_do_not_look_into_the_future():
    df = generate_fleet(n_machines=2, seed=3)
    full = rolling_features(df, SENSORS, window=5)
    cut = rolling_features(df[df["cycle"] <= 50], SENSORS, window=5)
    cols = feature_columns(SENSORS)
    merged = cut.merge(full, on=["machine_id", "cycle"], suffixes=("_cut", "_full"))
    for c in cols:
        assert np.allclose(merged[f"{c}_cut"], merged[f"{c}_full"])


def test_load_cmapss_format(tmp_path):
    rows = []
    for unit, life in [(1, 4), (2, 3)]:
        for cycle in range(1, life + 1):
            rows.append(" ".join(str(v) for v in [unit, cycle, 0.1, 0.2, 100] + [cycle * 1.0] * 21) + " ")
    path = tmp_path / "train_FD001.txt"
    path.write_text("\n".join(rows))
    df = load_cmapss(path)
    assert df.shape == (7, 27)
    assert df[df["machine_id"] == 1]["rul"].tolist() == [3, 2, 1, 0]
