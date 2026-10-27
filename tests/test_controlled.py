import numpy as np

from evaluation import possession_metrics as pm
from possession.controlled import controlled_labels

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def seq(*parts):
    return np.concatenate([np.full(n, code, dtype=np.int8) for code, n in parts])


def test_short_spell_relabelled_to_previous_team():
    codes = seq((H, 100), (A, 20), (H, 100))
    assert (controlled_labels(codes, [(0, 220)]) == H).all()


def test_long_spell_kept():
    codes = seq((H, 100), (A, 60), (H, 100))
    np.testing.assert_array_equal(controlled_labels(codes, [(0, 260)]), codes)


def test_short_spell_before_dead_relabelled():
    codes = seq((H, 100), (A, 20), (D, 50))
    np.testing.assert_array_equal(controlled_labels(codes, [(0, 170)]), seq((H, 120), (D, 50)))


def test_first_spell_after_restart_kept_even_if_short():
    codes = seq((H, 100), (D, 50), (A, 20), (H, 100))
    np.testing.assert_array_equal(controlled_labels(codes, [(0, 270)]), codes)


def test_runs_are_independent_and_x_untouched():
    codes = seq((H, 100), (X, 10), (A, 20), (H, 100))
    out = controlled_labels(codes, [(0, 100), (110, 230)])
    np.testing.assert_array_equal(out, codes)  # A is the first spell of run 2, so it is kept


def test_chain_of_short_spells():
    codes = seq((H, 100), (A, 20), (H, 10), (A, 30), (H, 100))
    assert (controlled_labels(codes, [(0, 260)]) == H).all()


def test_input_not_modified():
    codes = seq((H, 100), (A, 20), (H, 100))
    before = codes.copy()
    controlled_labels(codes, [(0, 220)])
    np.testing.assert_array_equal(codes, before)
