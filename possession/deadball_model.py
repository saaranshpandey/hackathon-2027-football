"""P(ball is dead) per frame, learned with LightGBM from the scrambled DFL training matches.

    python -m possession.deadball_model      # train on TRAIN, early-stop on VAL, save models/deadball_lgbm.txt

TEST is never touched here. The saved model is trained on public DFL data only.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from possession.features import deadball_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "deadball_lgbm.txt")
SUBSAMPLE = 5
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "deterministic": True, "force_col_wise": True, "num_threads": 4,
    "seed": 0, "verbose": -1,
}
MAX_ROUNDS = 2000
EARLY_STOP = 50


def fit(X_train, y_train, X_val, y_val):
    train = lgb.Dataset(X_train, label=y_train)
    val = lgb.Dataset(X_val, label=y_val, reference=train)
    return lgb.train(PARAMS, train, num_boost_round=MAX_ROUNDS, valid_sets=[val],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def save(booster, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path, num_iteration=booster.best_iteration)


def load(path=MODEL_PATH):
    return lgb.Booster(model_file=path)


def p_dead_model(trk, booster):
    """P(dead) for every frame (frames outside a period get a value too; the decoder ignores them)."""
    return booster.predict(deadball_features(trk)).astype(np.float32)


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
    from sklearn.metrics import roc_auc_score

    from possession.external import TRAIN, VAL

    booster, (X_va, y_va) = train(TRAIN, VAL)
    save(booster, MODEL_PATH)
    print("rounds", booster.best_iteration, " val AUC %.4f" % roc_auc_score(y_va, booster.predict(X_va)))
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    print("top features by gain:", (gain / gain.sum()).sort_values(ascending=False).head(10).round(3).to_dict())


if __name__ == "__main__":
    main()
