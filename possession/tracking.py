"""One container for tracking data from any source: our match's ATD or DFL open data.

Everything is in metres with the origin at a pitch corner, x along the length in [0, L] and
y across in [0, W]. Missing observations are NaN, for the ball and players alike.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Tracking:
    ball: np.ndarray     # (T, 2) float32
    players: np.ndarray  # (T, S, 2) float32; a slot is a track, not a person
    team: np.ndarray     # (S,) int8, 0 home, 1 away
    period: np.ndarray   # (T,) int8, 0 outside a period of play
    pitch: tuple         # (L, W) metres


def runs_from_period(period):
    """Half-open [start, stop) blocks where period != 0, split wherever the period changes."""
    period = np.asarray(period)
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(period)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if period[a] != 0]


def true_runs(mask):
    """(starts, stops) of True runs in a 1-D bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def save(trk, path):
    np.savez(path, ball=trk.ball, players=trk.players, team=trk.team, period=trk.period,
             pitch=np.array(trk.pitch, dtype=np.float64))


def load(path):
    with np.load(path) as f:
        return Tracking(ball=f["ball"], players=f["players"], team=f["team"],
                        period=f["period"], pitch=tuple(float(v) for v in f["pitch"]))
