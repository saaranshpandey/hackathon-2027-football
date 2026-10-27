"""Per-frame features for the dead-ball model, from tracking only (so they exist for DFL and our match).

Everything is computed inside one period run at a time, so no window reaches across halftime or the
DFL period separator. Missing values stay NaN; LightGBM handles them natively.
"""

import warnings

import numpy as np
import pandas as pd

from possession.deadball import centred_mean, speeds
from possession.scramble import _true_runs
from possession.tracking import runs_from_period

SLOW_MS = 1.0
NEAR_BALL_M = 5.0
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
WINDOWS = (12, 50, 125)  # +/- 0.5 s, 2 s, 5 s


def _one_sided_mean(values, width, past):
    """NaN-aware mean over [t - width, t] (past) or [t, t + width] (future)."""
    seen = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - width) if past else index
    high = index + 1 if past else np.minimum(len(values), index + width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def _gap_features(visible):
    """Current missing-run length, frames since the ball was last seen, frames until it is next seen."""
    n = len(visible)
    run_len = np.zeros(n)
    since = np.where(visible, 0.0, np.nan)
    until = np.where(visible, 0.0, np.nan)
    for start, stop in zip(*_true_runs(~visible)):
        run_len[start:stop] = stop - start
        if start > 0:
            since[start:stop] = np.arange(1, stop - start + 1)
        if stop < n:
            until[start:stop] = np.arange(stop - start, 0, -1)
    return run_len, since, until


def _run_features(trk, a, b):
    length, width = trk.pitch
    ball = trk.ball[a:b].astype(np.float64)
    players = trk.players[a:b].astype(np.float64)
    x, y = ball[:, 0], ball[:, 1]
    visible = ~np.isnan(x)
    f = {}

    f["ball_visible"] = visible.astype(float)
    f["ball_speed"] = np.where(visible, centred_mean(speeds(ball), 12), np.nan)
    f["ball_line_dist"] = np.minimum.reduce([x, length - x, y, width - y])
    corners = np.array([[0, 0], [0, width], [length, 0], [length, width]])
    f["ball_corner_dist"] = np.linalg.norm(ball[:, None, :] - corners[None], axis=2).min(axis=1)
    f["ball_missing_run"], f["since_ball_seen"], f["until_ball_seen"] = _gap_features(visible)

    seen = ~np.isnan(players[:, :, 0])
    player_speed = speeds(players)
    known = ~np.isnan(player_speed)
    f["visible_players"] = seen.sum(axis=1).astype(float)
    f["player_speed_mean"] = np.nanmean(player_speed, axis=1)
    f["player_speed_median"] = np.nanmedian(player_speed, axis=1)
    f["slow_players"] = (known & (np.nan_to_num(player_speed, nan=np.inf) < SLOW_MS)).sum(axis=1).astype(float)
    f["spread_x"] = np.nanstd(players[:, :, 0], axis=1)
    f["spread_y"] = np.nanstd(players[:, :, 1], axis=1)

    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    f["players_near_ball"] = np.where(visible, (dist < NEAR_BALL_M).sum(axis=1), np.nan)
    px, py = players[:, :, 0], players[:, :, 1]
    in_box = ((px < BOX_DEPTH_M) | (px > length - BOX_DEPTH_M)) & (np.abs(py - width / 2) < BOX_HALF_WIDTH_M)
    f["players_in_boxes"] = in_box.sum(axis=1).astype(float)
    f["touchline_min_dist"] = np.nanmin(np.minimum(py, width - py), axis=1)

    for w in WINDOWS:
        f["player_speed_mean_w{}".format(w)] = centred_mean(f["player_speed_mean"], w)
        f["slow_players_w{}".format(w)] = centred_mean(f["slow_players"], w)
        f["ball_visible_w{}".format(w)] = centred_mean(f["ball_visible"], w)
    f["ball_speed_w50"] = centred_mean(f["ball_speed"], 50)
    f["player_speed_past_w50"] = _one_sided_mean(f["player_speed_mean"], 50, past=True)
    f["player_speed_future_w50"] = _one_sided_mean(f["player_speed_mean"], 50, past=False)
    return f


def deadball_features(trk):
    """(T, F) float32 feature table; rows outside a period are all NaN."""
    table = None
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, b in runs_from_period(trk.period):
            f = _run_features(trk, a, b)
            if table is None:
                table = pd.DataFrame(np.nan, index=np.arange(len(trk.period)), columns=list(f), dtype=np.float32)
            table.iloc[a:b] = np.column_stack([f[c] for c in table.columns]).astype(np.float32)
    return table
