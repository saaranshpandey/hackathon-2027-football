"""P(ball is dead) per frame: gradient-boosted trees learned from the scrambled DFL training matches.

    python -m possession.deadball_model      # train on TRAIN, pick rounds on VAL, save models/deadball_hgb.joblib

scikit-learn's HistGradientBoostingClassifier (the LightGBM-style algorithm) is used rather than LightGBM
itself: LightGBM loads Homebrew's OpenMP and PyTorch its own, and the two crash in one process on macOS.
TEST is never touched here. The saved model is trained on public DFL data only.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from evaluation import possession_metrics as pm
from possession.features import deadball_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "deadball_hgb.joblib")
SUBSAMPLE = 5
MAX_ROUNDS = 600
BASE = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=MAX_ROUNDS, max_leaf_nodes=31,
                                      min_samples_leaf=200, max_features=0.8, early_stopping=False,
                                      random_state=0)


def fit(X_train, y_train, X_val, y_val):
    """Fit MAX_ROUNDS rounds, keep the round count with the lowest validation log-loss, refit to it."""
    full = clone(BASE).fit(X_train, y_train)
    losses = [log_loss(y_val, p[:, 1], labels=[0, 1]) for p in full.staged_predict_proba(X_val)]
    return clone(BASE).set_params(max_iter=int(np.argmin(losses)) + 1).fit(X_train, y_train)


def predict(model, X):
    return model.predict_proba(X)[:, 1].astype(np.float32)


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path=MODEL_PATH):
    return joblib.load(path)


def p_dead_model(trk, model):
    """P(dead) for every frame (frames outside a period get a value too; the decoder ignores them)."""
    return predict(model, deadball_features(trk))


def _rows(match_ids):
    from possession.scramble import load_scrambled

    X, y = [], []
    for match in match_ids:
        trk, gt = load_scrambled(match)
        keep = np.flatnonzero((trk.period > 0) & (gt != pm.X))[::SUBSAMPLE]
        X.append(deadball_features(trk).iloc[keep])
        y.append((gt[keep] == pm.D).astype(int))
    return pd.concat(X, ignore_index=True), np.concatenate(y)


def train(train_ids, val_ids):
    X_tr, y_tr = _rows(train_ids)
    X_va, y_va = _rows(val_ids)
    return fit(X_tr, y_tr, X_va, y_va), (X_va, y_va)


def main():
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    from possession.external import TRAIN, VAL

    model, (X_va, y_va) = train(TRAIN, VAL)
    save(model, MODEL_PATH)
    print("rounds", model.n_iter_, " val AUC %.4f" % roc_auc_score(y_va, predict(model, X_va)))
    sample = np.arange(0, len(y_va), 10)
    imp = permutation_importance(model, X_va.iloc[sample], y_va[sample], scoring="roc_auc", n_repeats=3,
                                 random_state=0)
    top = pd.Series(imp.importances_mean, index=X_va.columns).sort_values(ascending=False).head(8)
    print("top features (AUC drop when shuffled):", top.round(4).to_dict())


if __name__ == "__main__":
    main()
