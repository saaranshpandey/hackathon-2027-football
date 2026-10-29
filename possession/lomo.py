"""Leave-one-match-out scores on all 7 DFL matches (set-up 4: 2 s key, 2 s rule).

    python -m possession.lomo

For each held-out match: the dead-ball model and the TCN are retrained on 5 matches, with the match before it
(sorted order, wrapping) as validation, as PathCRF's own LOMOCV does. PathCRF itself is NOT retrained (it would take
days): on its 5 training matches its team signal is in-sample, so those rows are marked and optimistic for DB/DBC.
"""

import numpy as np

from possession.controlled import controlled_labels
from possession.deadball import p_dead_rules
from possession.deadball_model import train as train_deadball
from possession.decode import decode
from possession.evaluate import answer_key, score
from possession.external import ALL_MATCHES, TRAIN
from possession.nn import approach_c
from possession.nn import train as train_tcn
from possession.pathcrf import approach_db, load_model
from possession.scramble import load_scrambled
from possession.team import p_home_nearest
from possession.tracking import runs_from_period


def folds():
    for i, test in enumerate(ALL_MATCHES):
        val = ALL_MATCHES[i - 1]
        yield test, val, [m for m in ALL_MATCHES if m not in (test, val)]


def main():
    pathcrf = load_model("cpu")
    rows = []
    for test, val, train in folds():
        deadball, _ = train_deadball(train, [val])
        tcn = train_tcn(train, [val])
        trk, gt = load_scrambled(test)
        runs = runs_from_period(trk.period)
        key = answer_key(trk, gt, "controlled2")
        db_dead, db_home = approach_db(trk, model=pathcrf, deadball=deadball)
        c_dead, c_home = approach_c(trk, tcn)
        signals = {"A": (p_dead_rules(trk), p_home_nearest(trk)), "DB": (db_dead, db_home),
                   "C": (c_dead, c_home), "DBC": ((db_dead + c_dead) / 2, (db_home + c_home) / 2)}
        r = {name: score(key, controlled_labels(decode(pd_, ph, runs), runs, 50))["spell_f1"]
             for name, (pd_, ph) in signals.items()}
        rows.append((test, test in TRAIN, r))
        print("{} {}: ".format(test, "(PathCRF in-sample)" if test in TRAIN else "(clean)") +
              "  ".join("{} {:.4f}".format(k, v) for k, v in r.items()), flush=True)

    print("\n| match | PathCRF trained on it? | A | C | DB | DBC |\n|---|---|---|---|---|---|")
    for test, in_sample, r in rows:
        print("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            test, "yes" if in_sample else "no", r["A"], r["C"], r["DB"], r["DBC"]))
    for label, keep in (("mean, all 7", lambda s: True), ("mean, 2 clean", lambda s: not s),
                        ("mean, 5 in-sample", lambda s: s)):
        sel = [r for _, s, r in rows if keep(s)]
        print("| **{}** | | ".format(label) + " | ".join(
            "{:.3f} ± {:.3f}".format(np.mean([x[k] for x in sel]), np.std([x[k] for x in sel]))
            for k in ("A", "C", "DB", "DBC")) + " |")
    wins = sum(r["DBC"] > r["DB"] for _, _, r in rows)
    print("\nDBC beats DB on {}/7 matches".format(wins))


if __name__ == "__main__":
    main()
