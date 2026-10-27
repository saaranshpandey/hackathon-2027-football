import numpy as np

from evaluation import possession_metrics as pm
from possession.evaluate import score


def two_run_truth():
    # run 1: H H H A A A D D, gap X X, run 2: A A A H H H
    letters = list("HHHAAADD") + ["X", "X"] + list("AAAHHH")
    return pm.label_codes(np.array(letters, dtype=object))


def test_score_perfect():
    gt = two_run_truth()
    result = score(gt, gt)
    assert result["spell_f1"] == 1.0
    assert result["n_gt"] == result["n_pred"] == 4


def test_score_constant():
    gt = two_run_truth()
    pred = np.where(gt == pm.X, pm.X, pm.D).astype(np.int8)
    assert score(gt, pred)["spell_f1"] == 0.0
