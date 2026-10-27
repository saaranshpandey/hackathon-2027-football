"""Score predictors on the DFL validation matches with the organizers' metric code.

    python -m possession.evaluate --predictors raw smooth25 smooth50 smooth125 A [--clean]

Scrambled matches (default) look like our match; --clean uses the original DFL tracking.
"""

import argparse

import numpy as np

from evaluation import possession_metrics as pm
from possession.deadball import p_dead_rules
from possession.decode import decode
from possession.external import ALL_MATCHES, TEST, VAL, load_dfl
from possession.scramble import load_scrambled
from possession.team import nearest_team_codes, p_home_nearest
from possession.tracking import runs_from_period
from starter.baseline import smooth


def score(gt, pred):
    result = pm.evaluate(gt, pred)
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_gt": result["n_gt_spells"], "n_pred": result["n_pred_spells"]}


def _baseline(window):
    def predict(trk):
        codes = pm.label_codes(smooth(nearest_team_codes(trk), window))
        codes[trk.period == 0] = pm.X
        return codes
    return predict


def _approach_a(trk):
    return decode(p_dead_rules(trk), p_home_nearest(trk), runs_from_period(trk.period))


PREDICTORS = {
    "raw": _baseline(0),
    "smooth25": _baseline(25),
    "smooth50": _baseline(50),
    "smooth125": _baseline(125),
    "A": _approach_a,
}


def main():
    parser = argparse.ArgumentParser(description="Score predictors on DFL validation matches")
    parser.add_argument("--predictors", nargs="+", default=list(PREDICTORS))
    parser.add_argument("--clean", action="store_true", help="original DFL tracking, not scrambled")
    args = parser.parse_args()

    rows = {name: {} for name in args.predictors}
    for match in ALL_MATCHES:
        trk, gt = load_dfl(match) if args.clean else load_scrambled(match)
        for name in args.predictors:
            rows[name][match] = score(gt, PREDICTORS[name](trk))

    print("data: {}\n".format("clean DFL" if args.clean else "scrambled DFL"))
    print("| predictor | match | spellF1 | macroF1 | n_gt | n_pred |")
    print("|---|---|---|---|---|---|")
    for name in args.predictors:
        for match in ALL_MATCHES:
            r = rows[name][match]
            role = "test" if match in TEST else "val" if match in VAL else "train"
            print("| {} | {} ({}) | {:.4f} | {:.4f} | {} | {} |".format(
                name, match, role, r["spell_f1"], r["macro_f1"], r["n_gt"], r["n_pred"]))
        for label, group in (("VAL+TEST", VAL + TEST), ("ALL", ALL_MATCHES)):
            print("| {} | **mean {}** | {:.4f} | {:.4f} | | |".format(
                name, label, np.mean([rows[name][m]["spell_f1"] for m in group]),
                np.mean([rows[name][m]["macro_f1"] for m in group])))


if __name__ == "__main__":
    main()
