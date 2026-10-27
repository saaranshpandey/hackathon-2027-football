"""P(home has the ball | ball alive) per frame. Approach A: nearest player to the ball."""

import numpy as np

DISTANCE_SCALE_M = 1.5


def _team_distances(trk):
    """(T, 2) distance from the ball to the nearest visible player of each team; NaN if none."""
    dist = np.linalg.norm(trk.players - trk.ball[:, None, :], axis=2)
    out = np.full((len(dist), 2), np.nan)
    for team in (0, 1):
        cols = dist[:, trk.team == team]
        has = (~np.isnan(cols)).any(axis=1)
        out[has, team] = np.nanmin(cols[has], axis=1)
    return out


def p_home_nearest(trk):
    d = _team_distances(trk)
    known = ~np.isnan(d).any(axis=1)
    p = np.full(len(d), 0.5)
    p[known] = 1.0 / (1.0 + np.exp(-(d[known, 1] - d[known, 0]) / DISTANCE_SCALE_M))
    return p.astype(np.float32)


def nearest_team_codes(trk):
    """0/1 team of the nearest visible player, -1 where the ball or every player is missing."""
    d = _team_distances(trk)
    d = np.where(np.isnan(d), np.inf, d)
    codes = np.where(d[:, 0] <= d[:, 1], 0, 1).astype(np.int8)
    codes[np.isinf(d).all(axis=1)] = -1
    return codes
