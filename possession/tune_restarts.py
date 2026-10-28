"""Tune the restart rule (window, value) on VAL with DFL restarts standing in for Smart Tagging; report TEST.

    python -m possession.tune_restarts

German restarts are subsampled to Smart Tagging's apparent recall (Barcelona tag count / mean DFL restart count),
because Smart Tagging misses restarts. Scored as set-up 4 (2 s answer key, 2 s rule on predictions), approach DB.
"""

import numpy as np

from possession.controlled import controlled_labels
from possession.decode import decode
from possession.evaluate import answer_key, score, signals
from possession.external import ALL_MATCHES, TEST, VAL, load_dfl
from possession.restarts import apply_restart_rule, dfl_restarts, smart_tag_restarts
from possession.scramble import load_scrambled
from possession.tracking import runs_from_period

WINDOWS = (25, 50, 75, 125, 200)   # frames: 1, 2, 3, 5, 8 s
VALUES = (0.6, 0.8, 0.95)
SEED = 2026


def tag_recall():
    per_match = np.mean([len(dfl_restarts(m, load_dfl(m)[0])) for m in ALL_MATCHES])
    return min(1.0, len(smart_tag_restarts()) / per_match)


def spell_f1(match, window, value, recall):
    trk, gt = load_scrambled(match)
    runs = runs_from_period(trk.period)
    p_dead, p_home = signals("DB", match, trk)
    if window:
        restarts = dfl_restarts(match, trk)
        keep = np.random.default_rng(SEED).random(len(restarts)) < recall
        p_dead = apply_restart_rule(p_dead, restarts[keep], window, value)
    codes = controlled_labels(decode(p_dead, p_home, runs), runs, 50)
    return score(answer_key(trk, gt, "controlled2"), codes)


def main():
    recall = tag_recall()
    print("simulated tag recall {:.2f}\n".format(recall))
    base = spell_f1(VAL[0], 0, 0, recall)["spell_f1"]
    grid = {(w, v): spell_f1(VAL[0], w, v, recall)["spell_f1"] for w in WINDOWS for v in VALUES}
    best = max(grid, key=grid.get)
    print("| window (s) | " + " | ".join("value {}".format(v) for v in VALUES) + " |")
    print("|---|" + "---|" * len(VALUES))
    for w in WINDOWS:
        print("| {:g} | ".format(w / 25) + " | ".join("{:.4f}".format(grid[(w, v)]) for v in VALUES) + " |")
    print("\nVAL no rule {:.4f} · best {:g} s / {} → {:.4f}".format(base, best[0] / 25, best[1], grid[best]))
    for label, (w, v) in (("no rule", (0, 0)), ("best on VAL", best)):
        r = spell_f1(TEST[0], w, v, recall)
        print("TEST {}: spellF1 {:.4f} · dead-ball F1 {:.4f} · stoppages {}/{} · possessions {}/{}".format(
            label, r["spell_f1"], r["dead_f1"], r["stops_pred"], r["stops_gt"], r["n_pred"], r["n_gt"]))


if __name__ == "__main__":
    main()
