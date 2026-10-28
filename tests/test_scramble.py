import numpy as np
import pytest

from possession.scramble import DefectStats, measure, scramble
from possession.tracking import Tracking

T = 5000


def random_walk(rng, n, start, step, low, high):
    moves = rng.normal(0.0, step, size=(n, 2))
    return np.clip(start + np.cumsum(moves, axis=0), low, high)


@pytest.fixture(scope="module")
def clean():
    rng = np.random.default_rng(0)
    low, high = np.array([0.0, 0.0]), np.array([105.0, 68.0])
    players = np.stack([random_walk(rng, T, rng.uniform(low, high), 0.2, low, high)
                        for _ in range(22)], axis=1)
    ball = random_walk(rng, T, np.array([52.5, 34.0]), 0.5, low, high)
    return Tracking(ball=ball.astype(np.float32), players=players.astype(np.float32),
                    team=np.array([0] * 11 + [1] * 11, dtype=np.int8),
                    period=np.ones(T, dtype=np.int8), pitch=(105.0, 68.0))


@pytest.fixture(scope="module")
def stats():
    return DefectStats(ball_missing_frac=0.4,
                       ball_gap_hist=np.bincount(np.full(10, 50)),
                       slot_life_hist=np.bincount(np.full(10, 200)),
                       mean_visible=20.5, jump_rate=0.002, slots_per_team=(23, 23))


@pytest.fixture(scope="module")
def scrambled(clean, stats):
    return scramble(clean, stats, seed=7)


def period_mask(trk):
    return trk.period > 0


def test_scramble_shapes(scrambled):
    assert scrambled.players.shape == (T, 46, 2)
    assert scrambled.team.tolist() == [0] * 23 + [1] * 23


def test_scramble_ball_missing(scrambled):
    missing = np.isnan(scrambled.ball[period_mask(scrambled), 0]).mean()
    assert abs(missing - 0.40) <= 0.02


def test_scramble_visible(scrambled):
    visible = (~np.isnan(scrambled.players[period_mask(scrambled), :, 0])).sum(axis=1).mean()
    assert abs(visible - 20.5) <= 0.5


def test_scramble_deterministic(clean, stats):
    a = scramble(clean, stats, seed=7)
    b = scramble(clean, stats, seed=7)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.players, b.players, equal_nan=True)


def test_measure_clean(clean):
    s = measure(clean)
    assert s.ball_missing_frac == 0
    assert s.jump_rate == 0
    assert s.slots_per_team == (11, 11)
    assert abs(s.mean_visible - 22.0) < 1e-9


def test_calibrate_rate_reaches_tiny_targets():
    from possession.scramble import _calibrate_rate

    # Our match's jump rate is ~1e-5, which needs a swap rate near 1e-4 per frame.
    rate = _calibrate_rate(lambda r: r / 10, target=1e-5)
    assert abs(rate / 10 - 1e-5) <= 1e-6


def test_situations_bits():
    from possession.scramble import CROWD_BIT, LINE_BIT, SLOW_BIT, situations

    n = 120
    players = np.full((n, 4, 2), np.nan, dtype=np.float32)
    ball = np.tile([52.0, 34.0], (n, 1)).astype(np.float32)
    ball[:60] = [104.0, 34.0]                  # near the right goal line for the first half of the clip
    players[:, 0] = [103.0, 34.0]              # two players next to it, standing still
    players[:, 1] = [104.5, 35.0]
    players[60:, 0] += np.arange(60)[:, None] * 0.2   # in the second half they run (5 m/s), ball mid-pitch
    trk = Tracking(ball=ball, players=players, team=np.array([0, 0, 1, 1], np.int8),
                   period=np.ones(n, np.int8), pitch=(105.0, 68.0))
    trk.ball[100:] = np.nan
    s = situations(trk)
    assert s[30] == LINE_BIT | CROWD_BIT | SLOW_BIT
    assert s[90] == 0
    assert (s[100:] == -1).all()


def test_measure_hazard_follows_where_gaps_start():
    from possession.scramble import LINE_BIT, measure

    n = 4000
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = np.stack([20 + np.arange(n) * 0.2 % 60, np.full(n, 20.0)], axis=1)
    ball = np.stack([20 + np.arange(n) * 0.3 % 60 + 10, np.full(n, 30.0)], axis=1).astype(np.float32)
    for start in range(200, n, 400):           # every gap starts right after the ball was at the line
        ball[start - 1] = [104.0, 30.0]
        ball[start:start + 50] = np.nan
    trk = Tracking(ball=ball, players=players, team=np.array([0, 1], np.int8),
                   period=np.ones(n, np.int8), pitch=(105.0, 68.0))
    hazard = np.array(measure(trk).gap_hazard)
    near = [c for c in range(8) if c & LINE_BIT]
    assert hazard[near].max() > 0
    assert hazard[[c for c in range(8) if not c & LINE_BIT]].max() == 0


def test_scramble_starts_gaps_in_hazard_situations(clean, stats):
    from possession.scramble import LINE_BIT, situations

    hazard = [1.0 if c & LINE_BIT else 0.0 for c in range(8)]
    clean2 = Tracking(ball=clean.ball.copy(), players=clean.players, team=clean.team,
                      period=clean.period, pitch=clean.pitch)
    clean2.ball[::50] = [104.5, 30.0]           # the ball touches the line now and then
    s2 = DefectStats(**{**stats.__dict__, "gap_hazard": hazard,
                        "gap_hist_line": stats.ball_gap_hist, "gap_hist_other": stats.ball_gap_hist})
    out = scramble(clean2, s2, seed=3)
    starts = np.flatnonzero(np.isnan(out.ball[1:, 0]) & ~np.isnan(out.ball[:-1, 0])) + 1
    sit = situations(clean2)
    assert len(starts) > 0
    assert all(sit[t - 1] & LINE_BIT or sit[t] & LINE_BIT for t in starts)
