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
