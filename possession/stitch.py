"""Anonymous, fragmented track slots -> a fixed number of identity-continuous players per team.

PathCRF needs 11 players per team that keep their identity. Our tracking has more slots than
players, and a person's track jumps to a new slot whenever the tracker loses them. Within each
period and team: a slot keeps its column while it stays visible; slots that appear take free
columns, nearest to the column's last known position first; slots beyond the column count are
dropped; gaps in a column are filled by straight-line interpolation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from possession.tracking import Tracking, runs_from_period

# A new slot prefers a never-used column over a free column whose last position is further than this.
NEW_COLUMN_COST_M = 50.0


def _stitch_team(pos, per_team):
    """(n, S, 2) slot positions of one team in one period -> (n, per_team, 2) column positions."""
    n = len(pos)
    vis = ~np.isnan(pos[:, :, 0])
    out = np.full((n, per_team, 2), np.nan)
    col_slot = np.full(per_team, -1)
    last = np.full((per_team, 2), np.nan)

    changes = np.flatnonzero((vis[1:] != vis[:-1]).any(axis=1)) + 1
    bounds = np.concatenate(([0], changes, [n]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        visible = np.flatnonzero(vis[a])
        col_slot[~np.isin(col_slot, visible)] = -1
        new = visible[~np.isin(visible, col_slot)]
        free = np.flatnonzero(col_slot < 0)
        if len(new) and len(free):
            cost = np.linalg.norm(last[free][:, None, :] - pos[a, new][None, :, :], axis=2)
            rows, cols = linear_sum_assignment(np.where(np.isnan(cost), NEW_COLUMN_COST_M, cost))
            col_slot[free[rows]] = new[cols]
        assigned = np.flatnonzero(col_slot >= 0)
        out[a:b, assigned] = pos[a:b, col_slot[assigned]]
        last[assigned] = pos[b - 1, col_slot[assigned]]

    frames = np.arange(n)
    for c in range(per_team):
        seen = ~np.isnan(out[:, c, 0])
        if seen.any() and not seen.all():
            for d in range(2):
                out[:, c, d] = np.interp(frames, frames[seen], out[seen, c, d])
    return out


def stitch(trk, per_team=11):
    """Tracking with 2 * per_team identity-continuous slots (home first); same ball and periods."""
    players = np.full((len(trk.period), 2 * per_team, 2), np.nan, dtype=np.float32)
    for a, b in runs_from_period(trk.period):
        for team in (0, 1):
            slots = np.flatnonzero(trk.team == team)
            columns = slice(team * per_team, (team + 1) * per_team)
            players[a:b, columns] = _stitch_team(trk.players[a:b, slots].astype(np.float64), per_team)
    return Tracking(ball=trk.ball.copy(), players=players,
                    team=np.array([0] * per_team + [1] * per_team, dtype=np.int8),
                    period=trk.period.copy(), pitch=trk.pitch)
