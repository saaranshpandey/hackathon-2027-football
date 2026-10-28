import numpy as np

from possession.restarts import apply_restart_rule


def test_rule_raises_only_the_window_before_each_restart():
    p = np.full(100, 0.2, dtype=np.float32)
    out = apply_restart_rule(p, np.array([50]), window=10, value=0.8)
    assert (out[40:50] == np.float32(0.8)).all()
    assert (out[:40] == np.float32(0.2)).all() and (out[50:] == np.float32(0.2)).all()


def test_rule_never_lowers_and_clips_at_start():
    p = np.full(30, 0.2, dtype=np.float32)
    p[2] = 0.95
    out = apply_restart_rule(p, np.array([5]), window=10, value=0.8)
    assert out[2] == np.float32(0.95)
    assert (out[[0, 1, 3, 4]] == np.float32(0.8)).all()
    assert (p[[0, 1, 3, 4]] == np.float32(0.2)).all()  # input untouched


def test_no_restarts_no_change():
    p = np.random.default_rng(0).random(50).astype(np.float32)
    np.testing.assert_array_equal(apply_restart_rule(p, np.array([], dtype=int), window=10, value=0.8), p)
