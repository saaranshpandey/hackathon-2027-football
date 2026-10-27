"""Step 2 go/no-go for PathCRF on the DFL test match (J03WR9).

    python -m possession.gonogo

Team accuracy = share of truly alive frames where the decoded label names the right team, with the
true dead-ball labels given to the decoder so that only the team signal is compared.
"""

import numpy as np

from evaluation import possession_metrics as pm
from possession.deadball import p_dead_rules
from possession.decode import decode
from possession.evaluate import score
from possession.external import TEST, load_dfl
from possession.pathcrf import load_model, p_home_pathcrf
from possession.scramble import load_scrambled
from possession.stitch import stitch
from possession.team import p_home_nearest
from possession.tracking import runs_from_period

PITCH_CENTRE_X = 52.5


def keeper_column_share(trk):
    """Per team and period: share of frames where the column deepest on average is the deepest player."""
    shares = []
    for a, b in runs_from_period(trk.period):
        for team in (0, 1):
            x = trk.players[a:b, trk.team == team, 0]
            depth = np.abs(x - PITCH_CENTRE_X)
            keeper = np.nanargmax(np.nanmean(depth, axis=0))
            shares.append(float((np.nanargmax(depth, axis=1) == keeper).mean()))
    return shares


def main():
    match = TEST[0]
    clean, gt = load_dfl(match)
    scrambled, _ = load_scrambled(match)
    stitched = stitch(scrambled)
    runs = runs_from_period(clean.period)
    alive = (gt == pm.H) | (gt == pm.A)
    true_dead = np.where(alive, 0.05, 0.95)
    model = load_model("cpu")

    signals = {
        "1 PathCRF clean": (clean, p_home_pathcrf(clean, alive, model=model)),
        "2 PathCRF scrambled+stitched": (scrambled, p_home_pathcrf(stitched, alive, model=model)),
        "3 A (nearest) scrambled": (scrambled, p_home_nearest(scrambled)),
    }
    acc = {}
    print("match", match, "\n")
    print("| run | team acc (alive frames, true dead) | spellF1 (A dead-ball rules) | possessions (true {}) |".format(
        len(pm.possession_spells(gt, runs))))
    print("|---|---|---|---|")
    for name, (trk, p_home) in signals.items():
        codes = decode(true_dead, p_home, runs)
        acc[name] = float((codes[alive] == gt[alive]).mean())
        r = score(gt, decode(p_dead_rules(trk), p_home, runs))
        print("| {} | {:.4f} | {:.4f} | {} |".format(name, acc[name], r["spell_f1"], r["n_pred"]))

    print("\nkeeper column stays the deepest player (stitched, per team & period):",
          [round(s, 3) for s in keeper_column_share(stitched)])
    clean_acc, scr_acc, a_acc = acc.values()
    drop = 100 * (clean_acc - scr_acc)
    verdict = ("NO-GO (scrambled not above A)" if scr_acc <= a_acc else
               "GO, use as-is" if drop <= 3 else "GO after fine-tuning (drop > 3 points)")
    print("drop clean -> scrambled: {:.1f} points; verdict: {}".format(drop, verdict))


if __name__ == "__main__":
    main()
