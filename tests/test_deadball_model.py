import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from possession.deadball_model import fit, load, predict, save


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
    model = fit(X_tr, y_tr, X_va, y_va)
    assert roc_auc_score(y_va, predict(model, X_va)) > 0.95


def test_fit_is_deterministic():
    X_tr, y_tr = synthetic(3000, 4)
    X_va, y_va = synthetic(1000, 5)
    np.testing.assert_array_equal(predict(fit(X_tr, y_tr, X_va, y_va), X_va),
                                  predict(fit(X_tr, y_tr, X_va, y_va), X_va))


def test_save_load_identical_predictions(tmp_path):
    X_tr, y_tr = synthetic(3000, 2)
    X_va, y_va = synthetic(1000, 3)
    model = fit(X_tr, y_tr, X_va, y_va)
    path = str(tmp_path / "model.joblib")
    save(model, path)
    np.testing.assert_array_equal(predict(load(path), X_va), predict(model, X_va))


def test_missing_values_allowed():
    X_tr, y_tr = synthetic(3000, 6)
    X_va, y_va = synthetic(1000, 7)
    X_tr.iloc[::7, 0] = np.nan
    model = fit(X_tr, y_tr, X_va, y_va)
    assert not np.isnan(predict(model, X_va.assign(player_speed_mean=np.nan))).any()
