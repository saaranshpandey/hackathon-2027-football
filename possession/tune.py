"""Grid-search the decoder's switching costs on the VAL match only.

    python -m possession.tune --predictors A D DB --labels controlled2 --post 50

Uses cached signals (evaluate.signals), so each grid point only re-decodes. TEST is never used here.
Prints the best costs per predictor and how flat the score is around them.
"""

import argparse
import itertools

import numpy as np

from possession.evaluate import LABEL_SETS, SIGNALS, answer_key, predict, score
from possession.external import VAL
from possession.scramble import load_scrambled

COST_SWITCH = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
COST_DEAD = (1.0, 2.0, 4.0, 8.0, 16.0)


def grid(name, labels, post):
    """{(cost_switch, cost_dead): VAL spellF1}."""
    match = VAL[0]
    trk, gt = load_scrambled(match)
    gt = answer_key(trk, gt, labels)
    return {(cs, cd): score(gt, predict(name, match, trk, cost_switch=cs, cost_dead=cd, post=post))["spell_f1"]
            for cs, cd in itertools.product(COST_SWITCH, COST_DEAD)}


def neighbours(costs, key):
    i, j = COST_SWITCH.index(key[0]), COST_DEAD.index(key[1])
    return [costs[(COST_SWITCH[a], COST_DEAD[b])] for a in (i - 1, i, i + 1) for b in (j - 1, j, j + 1)
            if 0 <= a < len(COST_SWITCH) and 0 <= b < len(COST_DEAD) and (a, b) != (i, j)]


def main():
    parser = argparse.ArgumentParser(description="Tune decoder costs on VAL")
    parser.add_argument("--predictors", nargs="+", default=list(SIGNALS), choices=list(SIGNALS))
    parser.add_argument("--labels", default="controlled2", choices=sorted(LABEL_SETS))
    parser.add_argument("--post", type=int, default=0)
    args = parser.parse_args()

    print("VAL {} · answer key {} · post {}\n".format(VAL[0], args.labels, args.post))
    print("| predictor | default 8/4 | best switch/dead | best VAL spellF1 | neighbours min–max |")
    print("|---|---|---|---|---|")
    for name in args.predictors:
        costs = grid(name, args.labels, args.post)
        best = max(costs, key=lambda k: (round(costs[k], 4), -abs(np.log(k[0] / 8)) - abs(np.log(k[1] / 4))))
        near = neighbours(costs, best)
        print("| {} | {:.4f} | {:g}/{:g} | {:.4f} | {:.4f}–{:.4f} |".format(
            name, costs[(8.0, 4.0)], best[0], best[1], costs[best], min(near), max(near)))


if __name__ == "__main__":
    main()
