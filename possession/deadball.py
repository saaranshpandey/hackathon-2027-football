"""P(ball is dead) per frame. Approach A: hand-written rules."""

import numpy as np

from possession.tracking import true_runs


FPS = 25
OUT_MARGIN_M = 0.5
SLOW_PLAYERS_MS = 2.0
STATIC_BALL_MS = 0.5
LONG_GAP_FRAMES = 75


def centred_mean(values, half_width):
    """NaN-aware mean over a centred window of +/- half_width frames. NaN where nothing seen."""
    seen = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def speeds(positions):
    """Speed in m/s from (T, ..., 2) positions; NaN on the first frame and wherever a frame is missing."""
    step = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * FPS
    return np.concatenate([np.full((1,) + step.shape[1:], np.nan), step])


def p_dead_rules(trk):
    length, width = trk.pitch
    x, y = trk.ball[:, 0], trk.ball[:, 1]
    with np.errstate(invalid="ignore"):
        ball_out = ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
                    | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))

        player_speed = speeds(trk.players.astype(np.float64))
        seen = ~np.isnan(player_speed)
        mean_speed = np.where(seen.any(axis=1),
                              np.nansum(player_speed, axis=1) / np.maximum(seen.sum(axis=1), 1), np.nan)
        slow = centred_mean(mean_speed, 50) < SLOW_PLAYERS_MS

        ball_static = ~np.isnan(x) & (centred_mean(speeds(trk.ball.astype(np.float64)), 12) < STATIC_BALL_MS)

    ball_gap_long = np.zeros(len(x), dtype=bool)
    for start, stop in zip(*true_runs(np.isnan(x))):
        if stop - start > LONG_GAP_FRAMES:
            ball_gap_long[start:stop] = True

    dead = ball_out | (slow & (ball_static | ball_gap_long))
    return np.where(dead, 0.9, 0.1).astype(np.float32)
