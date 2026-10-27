import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from possession.deadball_model import fit, load, save


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0, 6, n)
    noise = rng.normal(0, 1, n)
    X = pd.DataFrame({"player_speed_mean": speed, "noise": noise}, dtype=np.float32)
    y = (speed < 2.0).astype(int)  # slow players -> dead ball
    return X, y


def test_fit_learns_separable_rule():
    X_tr, y_tr = synthetic(5000, 0)
    X_va, y_va = synthetic(2000, 1)
    booster = fit(X_tr, y_tr, X_va, y_va)
    assert roc_auc_score(y_va, booster.predict(X_va)) > 0.95


def test_save_load_identical_predictions(tmp_path):
    X_tr, y_tr = synthetic(3000, 2)
    X_va, y_va = synthetic(1000, 3)
    booster = fit(X_tr, y_tr, X_va, y_va)
    path = str(tmp_path / "model.txt")
    save(booster, path)
    np.testing.assert_array_equal(load(path).predict(X_va), booster.predict(X_va))
