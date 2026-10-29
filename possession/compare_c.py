"""Compare approach C combinations with DB on VAL and TEST (set-ups 1 and 4).

    python -m possession.compare_c

Rows: DB · C (both heads) · C-dead + PathCRF team · dead-ball model + C-team · averages of DB and C.
"""

import numpy as np

from possession.controlled import controlled_labels
from possession.decode import decode
from possession.evaluate import answer_key, score, signals
from possession.external import TEST, VAL
from possession.scramble import load_scrambled
from possession.tracking import runs_from_period


def combos(match, trk):
    db_dead, db_home = signals("DB", match, trk)
    c_dead, c_home = signals("C", match, trk)
    cd_dead, cd_home = signals("C_dead", match, trk)
    return {
        "DB (current best)": (db_dead, db_home),
        "C both heads": (c_dead, c_home),
        "C-dead + PathCRF team": (cd_dead, cd_home),
        "dead-ball model + C-team": (db_dead, c_home),
        "average of DB and C": ((db_dead + c_dead) / 2, (db_home + c_home) / 2),
    }


def main():
    rows = {}
    for match in VAL + TEST:
        trk, gt = load_scrambled(match)
        runs = runs_from_period(trk.period)
        for name, (p_dead, p_home) in combos(match, trk).items():
            codes = decode(p_dead, p_home, runs)
            for su, labels, post in ((1, "original", 0), (4, "controlled2", 50)):
                pred = controlled_labels(codes, runs, post) if post else codes
                rows.setdefault((name, su), {})[match] = score(answer_key(trk, gt, labels), pred)
    print("| variant | set-up | spellF1 val | spellF1 test | macroF1 test | dead-ball F1 test | stoppages test | possessions test |")
    print("|---|---|---|---|---|---|---|---|")
    for (name, su), r in rows.items():
        v, t = r[VAL[0]], r[TEST[0]]
        print("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {}/{} | {}/{} |".format(
            name, su, v["spell_f1"], t["spell_f1"], t["macro_f1"], t["dead_f1"], t["stops_pred"], t["stops_gt"],
            t["n_pred"], t["n_gt"]))


if __name__ == "__main__":
    main()
