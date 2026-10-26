import numpy as np
import pytest

from fraud import model as fm
from fraud.synth import generate


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    transactions, labels = generate(n_users=300, seed=3)
    x, y = fm.build_dataset(transactions), np.asarray(labels)
    split = int(len(y) * 0.7)
    clf = fm.train(x[:split], y[:split])
    return clf, x[split:], y[split:]
