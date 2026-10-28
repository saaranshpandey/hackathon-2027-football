"""Score predictors on the DFL validation matches with the organizers' metric code.

    python -m possession.evaluate --predictors smooth50 A D DB --matches J03WQQ J03WR9 \\
        [--labels original|controlled2|controlled3] [--post FRAMES] [--cost-switch C] [--cost-dead C] [--clean]

Scrambled matches (default) look like our match; --clean uses the original DFL tracking.
--labels picks the answer key: the DFL labels as they are, or with the controlled-possession rule applied.
--post applies the same rule to the predictions (0 = off).
"""

import argparse
import hashlib
import os

import numpy as np

from evaluation import possession_metrics as pm
from possession.controlled import controlled_labels
from possession.deadball import p_dead_rules
from possession.decode import decode
from possession.external import ALL_MATCHES, DFL_DIR, TEST, VAL, load_dfl
from possession.pathcrf import approach_d, approach_db
from possession.scramble import load_scrambled
from possession.team import nearest_team_codes, p_home_nearest
from possession.tracking import runs_from_period
from starter.baseline import smooth

LABEL_SETS = {"original": 0, "controlled2": 50, "controlled3": 75}
SIGNAL_DIR = os.path.join(DFL_DIR, "signals")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P = os.path.join(_ROOT, "possession")
_PATHCRF = os.path.join(_ROOT, "third_party", "pathcrf")
_RULES = [os.path.join(_P, f) for f in ("tracking.py", "scramble.py", "deadball.py", "team.py")] + [
    os.path.join(_ROOT, "data", "cache", "defects.json")]
_PATHCRF_CHAIN = [os.path.join(_P, f) for f in ("decode.py", "stitch.py", "pathcrf.py")] + [
    os.path.join(_PATHCRF, "inference.py"), os.path.join(_PATHCRF, "datatools", "utils.py"),
    os.path.join(_PATHCRF, "saved", "140", "model", "state_dict_best_acc.pt")]
# Every file a cached signal depends on: a change to any of them gives a new cache key.
SIGNAL_SOURCES = {
    "A": _RULES,
    "D": _RULES + _PATHCRF_CHAIN,
    "DB": _RULES + _PATHCRF_CHAIN + [os.path.join(_P, "features.py"), os.path.join(_P, "deadball_model.py"),
                                     os.path.join(_ROOT, "models", "deadball_hgb.joblib")],
}


def _fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()[:12]


def _stoppages(codes, runs):
    return sum(1 for _, _, k in pm.segments(codes, runs) if k == pm.D)


def score(gt, pred):
    result = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_gt": result["n_gt_spells"], "n_pred": result["n_pred_spells"],
            "dead_f1": result["alive_dead_f1"], "stops_gt": _stoppages(gt, runs),
            "stops_pred": _stoppages(pm.fill_abstentions(pred, runs), runs)}


def _baseline(window):
    def predict(trk):
        codes = pm.label_codes(smooth(nearest_team_codes(trk), window))
        codes[trk.period == 0] = pm.X
        return codes
    return predict


BASELINES = {"raw": _baseline(0), "smooth25": _baseline(25), "smooth50": _baseline(50),
             "smooth125": _baseline(125)}
SIGNALS = {
    "A": lambda trk: (p_dead_rules(trk), p_home_nearest(trk)),
    # PathCRF was trained on the 5 train matches: only its val/test rows are fair.
    "D": approach_d,
    "DB": approach_db,
}
PREDICTORS = list(BASELINES) + list(SIGNALS)


def signals(name, match, trk, clean=False):
    """(P(dead), P(home)) of a signal-based approach on one DFL match, cached on disk."""
    tag = "clean" if clean else "scrambled"
    key = _fingerprint(SIGNAL_SOURCES[name])
    path = os.path.join(SIGNAL_DIR, "{}_{}_{}_{}.npz".format(name, match, tag, key))
    if os.path.exists(path):
        with np.load(path) as f:
            p_dead, p_home = f["p_dead"], f["p_home"]
        if len(p_dead) == len(p_home) == len(trk.period):
            return p_dead, p_home
    p_dead, p_home = SIGNALS[name](trk)
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    with open(path + ".part", "wb") as f:  # write then rename: an interrupted run leaves no corrupt cache
        np.savez(f, p_dead=p_dead, p_home=p_home)
    os.replace(path + ".part", path)
    return p_dead, p_home


def predict(name, match, trk, clean=False, cost_switch=8.0, cost_dead=4.0, post=0):
    runs = runs_from_period(trk.period)
    if name in BASELINES:
        codes = BASELINES[name](trk)
    else:
        codes = decode(*signals(name, match, trk, clean), runs, cost_switch=cost_switch, cost_dead=cost_dead)
    return controlled_labels(codes, runs, post) if post else codes


def answer_key(trk, gt, labels):
    threshold = LABEL_SETS[labels]
    return controlled_labels(gt, runs_from_period(trk.period), threshold) if threshold else gt


def main():
    parser = argparse.ArgumentParser(description="Score predictors on DFL validation matches")
    parser.add_argument("--predictors", nargs="+", default=PREDICTORS, choices=PREDICTORS)
    parser.add_argument("--clean", action="store_true", help="original DFL tracking, not scrambled")
    parser.add_argument("--matches", nargs="+", default=VAL + TEST, choices=ALL_MATCHES)
    parser.add_argument("--labels", default="original", choices=sorted(LABEL_SETS))
    parser.add_argument("--post", type=int, default=0, help="apply the controlled rule to predictions (frames)")
    parser.add_argument("--cost-switch", type=float, default=8.0)
    parser.add_argument("--cost-dead", type=float, default=4.0)
    args = parser.parse_args()

    rows = {name: {} for name in args.predictors}
    for match in args.matches:
        trk, gt = load_dfl(match) if args.clean else load_scrambled(match)
        gt = answer_key(trk, gt, args.labels)
        for name in args.predictors:
            rows[name][match] = score(gt, predict(name, match, trk, args.clean, args.cost_switch,
                                                  args.cost_dead, args.post))

    print("data: {} DFL · answer key: {} · post: {} · costs switch/dead: {}/{}\n".format(
        "clean" if args.clean else "scrambled", args.labels, args.post, args.cost_switch, args.cost_dead))
    print("| predictor | match | spellF1 | macroF1 | dead-ball F1 | possessions pred/true | stoppages pred/true |")
    print("|---|---|---|---|---|---|---|")
    for name in args.predictors:
        for match in args.matches:
            r = rows[name][match]
            role = "test" if match in TEST else "val" if match in VAL else "train"
            print("| {} | {} ({}) | {:.4f} | {:.4f} | {:.4f} | {}/{} | {}/{} |".format(
                name, match, role, r["spell_f1"], r["macro_f1"], r["dead_f1"], r["n_pred"], r["n_gt"],
                r["stops_pred"], r["stops_gt"]))
        for label, group in (("VAL+TEST", VAL + TEST), ("ALL", ALL_MATCHES)):
            if not set(group) <= set(args.matches):
                continue
            print("| {} | **mean {}** | {:.4f} | {:.4f} | | | |".format(
                name, label, np.mean([rows[name][m]["spell_f1"] for m in group]),
                np.mean([rows[name][m]["macro_f1"] for m in group])))


if __name__ == "__main__":
    main()
