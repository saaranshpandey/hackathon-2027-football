"""One command from the provided data to a submission file.

    python -m possession.run --approach A --name A      ->  submissions/A.csv

Prints only aggregates of the prediction (label shares, possession count).
"""

import argparse
import os
import sys

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks
from possession.data import ROOT, load_match
from possession.deadball import p_dead_rules
from possession.pathcrf import approach_d, approach_db
from possession.decode import decode
from possession.team import p_home_nearest
from possession.tracking import runs_from_period
from starter.baseline import write_submission
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES

KICKOFF = {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}

APPROACHES = {
    "A": lambda trk, device: (p_dead_rules(trk), p_home_nearest(trk)),
    "D": lambda trk, device: approach_d(trk, device=device),
    "DB": lambda trk, device: approach_db(trk, device=device),
}


def predict(approach, device="cpu"):
    trk = load_match()
    p_dead, p_home = APPROACHES[approach](trk, device)
    for half, first, _ in half_video_blocks(GAME):
        p_dead[first:KICKOFF[half]] = 0.999  # graded but before kickoff: the ball is not in play
    return decode(p_dead, p_home, runs_from_period(trk.period)), trk


def main():
    parser = argparse.ArgumentParser(description="Write a submission file")
    parser.add_argument("--approach", choices=sorted(APPROACHES), default="A")
    parser.add_argument("--name", default=None, help="file name stem (default: the approach)")
    parser.add_argument("--device", default="cpu", help="torch device for approach D: cpu, mps or cuda")
    args = parser.parse_args()

    codes, trk = predict(args.approach, args.device)
    path = os.path.join(ROOT, "submissions", "{}.csv".format(args.name or args.approach))
    write_submission(path, np.array(pm.LABELS)[codes])

    reason = pm.validate_submission(pm.read_label_csv(path), len(codes))
    if reason:
        sys.exit("invalid submission: " + reason)

    graded = trk.period > 0
    shares = {k: round(float((codes[graded] == getattr(pm, k)).mean()), 3) for k in ("H", "A", "D")}
    spells = pm.possession_spells(codes, runs_from_period(trk.period))
    print("{} -> {}  shares {}  possessions {}".format(
        args.approach, os.path.relpath(path, ROOT), shares, len(spells)))


if __name__ == "__main__":
    main()
