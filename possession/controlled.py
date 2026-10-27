"""Controlled possession: a change of team only counts if the new team keeps the ball long enough.

CHALLENGE.md: "A change of possession occurs when a player from the opposing team gains controlled
possession... a simple touch without control, a deflection that does not intercept, an aerial duel won
without subsequent control" do not count, and "possible proxies could include: a player keeps the ball
for a few seconds". The threshold (2 s) was fixed with the user before looking at any result.

Inside each live stretch, a spell of the other team shorter than the threshold goes back to the team that
had the ball before. The first spell after a restart (a stoppage or the start of a run) is always kept:
"whenever play restarts after a stoppage, there is a clear possessor".
"""

import numpy as np

from evaluation import possession_metrics as pm

TWO_SECONDS = 50


def controlled_labels(codes, runs, min_frames=TWO_SECONDS):
    """Copy of codes with short uncontrolled team spells relabelled; D and X unchanged."""
    out = np.array(codes, copy=True)
    for run_start, run_stop in runs:
        current = None
        for start, stop, code in pm.segments(out, [(run_start, run_stop)]):
            if code not in pm.POSSESSION_LABELS:
                current = None  # a stoppage: the restart decides the next possessor
            elif current is None or code == current or stop - start >= min_frames:
                current = code
            else:
                out[start:stop] = current
    return out
