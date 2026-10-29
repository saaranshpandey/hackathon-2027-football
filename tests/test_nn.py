import numpy as np
import torch

from possession.nn import TCN, mirror, nn_features, receptive_steps
from possession.tracking import Tracking


def make(home_x, away_x, n=60):
    players = np.zeros((n, 4, 2), np.float32)
    players[:, :2, 0], players[:, 2:, 0] = home_x, away_x
    players[:, :, 1] = 34.0
    return Tracking(ball=np.tile([52.0, 34.0], (n, 1)).astype(np.float32), players=players,
                    team=np.array([0, 0, 1, 1], np.int8), period=np.ones(n, np.int8), pitch=(105.0, 68.0))


def test_orientation_home_attacks_right():
    left = nn_features(make([20, 30], [70, 80]))
    right = nn_features(make([75, 85], [25, 35]))  # home defends the right goal -> mirrored
    np.testing.assert_allclose(left["home_cx"].iloc[5], 25.0)
    np.testing.assert_allclose(right["home_cx"].iloc[5], 105.0 - 80.0)


def test_mirror_swaps_teams():
    trk = make([20, 30], [70, 80])
    m = mirror(trk)
    assert m.team.tolist() == [1, 1, 0, 0]
    # old away becomes home and is re-oriented to attack right: x -> L - x
    np.testing.assert_allclose(nn_features(m)["home_cx"].iloc[5], 105.0 - nn_features(trk)["away_cx"].iloc[5], atol=1e-4)


def test_tcn_keeps_length_and_receptive_field():
    torch.manual_seed(0)
    net = TCN(n_features=7).eval()
    x = torch.zeros(1, 7, 300)
    y0 = net(x)
    assert y0.shape == (1, 2, 300)
    x2 = x.clone()
    x2[0, :, 150] = 5.0
    diff = (net(x2) - y0).abs().sum(dim=(0, 1))
    reach = receptive_steps()
    assert diff[150 - reach:150 + reach + 1].sum() > 0
    assert diff[:150 - reach].sum() == 0 and diff[150 + reach + 1:].sum() == 0


def test_approach_c_refuses_mismatched_feature_columns():
    import pytest

    from possession.nn import approach_c

    trk = make([20, 30], [70, 80])
    cols = list(nn_features(trk).columns)
    net = TCN(len(cols))
    bundle = {"state": net.state_dict(), "mean": np.zeros(len(cols)), "std": np.ones(len(cols)),
              "columns": cols[::-1]}  # same count, different order
    with pytest.raises(ValueError, match="columns"):
        approach_c(trk, bundle)
    bundle["columns"] = cols
    p_dead, p_home = approach_c(trk, bundle)
    assert len(p_dead) == len(p_home) == len(trk.period)


def test_upsample_interpolates_between_steps():
    from possession.nn import _upsample

    np.testing.assert_allclose(_upsample(np.array([0.0, 1.0]), 10), [0, .2, .4, .6, .8, 1, 1, 1, 1, 1])


def test_speed_not_taken_across_period_edge():
    trk = make([20, 30], [70, 80], n=60)
    trk.period[:10] = 0
    trk.players[:10, :, 0] = 104.0  # positions just before the period, far away
    f = nn_features(trk)
    assert f["home_speed"].iloc[10:].max() < 1.0


def test_orientation_rotates_y_too():
    trk = make([75, 85], [25, 35])  # home defends the right goal -> rotated 180 degrees
    trk.players[:, :2, 1] = 10.0
    np.testing.assert_allclose(nn_features(trk)["home_cy"].iloc[5], 68.0 - 10.0)
