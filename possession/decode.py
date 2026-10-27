"""Per-frame probabilities -> clean H/A/D possessions, by Viterbi over three states.

Changing state costs a fixed penalty, so a few odd frames cannot split a possession, and a
frame with no information (p_home = 0.5) simply keeps the current state: a loose ball stays
with the last team.
"""

import numpy as np

from evaluation import possession_metrics as pm

STATES = np.array([pm.H, pm.A, pm.D], dtype=np.int8)


def _viterbi(emit, cost):
    """Most likely state path for emissions (T, 3) and transition costs (3, 3)."""
    n = len(emit)
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(p_dead, p_home, runs, cost_switch=8.0, cost_dead=4.0):
    """Label codes (T,) from P(dead) and P(home | alive); X outside the half-open runs."""
    p_dead = np.asarray(p_dead, dtype=np.float64)
    p_home = np.asarray(p_home, dtype=np.float64)
    if not (np.isfinite(p_dead).all() and np.isfinite(p_home).all()):
        # A NaN would silently turn the rest of its run into H (argmax over NaN is 0).
        raise ValueError("decode needs finite probabilities; a producer emitted NaN or inf")
    p_dead = np.clip(np.round(p_dead, 3), 0.001, 0.999)
    p_home = np.clip(np.round(p_home, 3), 0.001, 0.999)
    emit = np.log(np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1))
    cost = np.array([[0.0, cost_switch, cost_dead],
                     [cost_switch, 0.0, cost_dead],
                     [cost_dead, cost_dead, 0.0]])

    codes = np.full(len(p_dead), pm.X, dtype=np.int8)
    for start, stop in runs:
        if stop > start:
            codes[start:stop] = STATES[_viterbi(emit[start:stop], cost)]
    return codes
