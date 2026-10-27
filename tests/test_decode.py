import numpy as np

from evaluation import possession_metrics as pm
from possession.decode import decode

T = 300


def inputs(p_dead=0.1, p_home=0.9):
    return np.full(T, p_dead), np.full(T, p_home)


def test_constant_home():
    p_dead, p_home = inputs()
    assert (decode(p_dead, p_home, [(0, T)]) == pm.H).all()


def test_single_frame_flip_ignored():
    p_dead, p_home = inputs()
    p_home[150] = 0.1
    assert (decode(p_dead, p_home, [(0, T)]) == pm.H).all()


def test_long_switch_at_boundary():
    p_dead, p_home = inputs()
    p_home[150:] = 0.1
    codes = decode(p_dead, p_home, [(0, T)])
    assert (codes[:150] == pm.H).all()
    assert (codes[150:] == pm.A).all()


def test_dead_block():
    p_dead, p_home = inputs()
    p_dead[100:200] = 0.95
    codes = decode(p_dead, p_home, [(0, T)])
    assert (codes[100:200] == pm.D).all()
    assert (codes[:100] == pm.H).all() and (codes[200:] == pm.H).all()


def test_runs_and_x():
    p_dead, p_home = inputs()
    runs = [(10, 100), (150, 290)]
    codes = decode(p_dead, p_home, runs)
    assert (codes[:10] == pm.X).all()
    assert (codes[100:150] == pm.X).all()
    assert (codes[290:] == pm.X).all()

    flipped = p_home.copy()
    flipped[150:290] = 0.1
    codes2 = decode(p_dead, flipped, runs)
    assert np.array_equal(codes2[10:100], codes[10:100])
    assert (codes2[150:290] == pm.A).all()


def test_uninformative_keeps_state():
    p_dead, p_home = inputs()
    p_home[100:] = 0.5
    assert (decode(p_dead, p_home, [(0, T)]) == pm.H).all()


def test_nan_input_rejected():
    import pytest

    p_dead, p_home = inputs()
    p_home[50] = np.nan
    with pytest.raises(ValueError):
        decode(p_dead, p_home, [(0, T)])
