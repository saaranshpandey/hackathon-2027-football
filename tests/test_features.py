import numpy as np

from possession.features import deadball_features
from possession.tracking import Tracking

PITCH = (105.0, 68.0)


def make(players, ball, period):
    return Tracking(ball=np.asarray(ball, dtype=np.float32), players=np.asarray(players, dtype=np.float32),
                    team=np.array([0] * (players.shape[1] // 2) + [1] * (players.shape[1] - players.shape[1] // 2),
                                  dtype=np.int8),
                    period=np.asarray(period, dtype=np.int8), pitch=PITCH)


def static_players(n_frames, n_players=4):
    pos = np.stack([np.linspace(20, 80, n_players), np.full(n_players, 30.0)], axis=1)
    return np.repeat(pos[None], n_frames, axis=0)


def test_static_players_have_zero_speed():
    n = 300
    f = deadball_features(make(static_players(n), np.tile([50.0, 30.0], (n, 1)), np.ones(n)))
    assert len(f) == n
    np.testing.assert_allclose(f["player_speed_mean"].iloc[1:], 0.0, atol=1e-5)
    np.testing.assert_allclose(f["slow_players"].iloc[1:], 4.0)


def test_ball_missing_run_length():
    n = 100
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[40:70] = np.nan
    f = deadball_features(make(static_players(n), ball, np.ones(n)))
    assert f["ball_missing_run"].iloc[50] == 30
    assert f["ball_missing_run"].iloc[10] == 0
    assert f["since_ball_seen"].iloc[45] == 6       # last seen at frame 39
    assert f["until_ball_seen"].iloc[45] == 25      # next seen at frame 70
    assert np.isnan(f["ball_speed"].iloc[50])
    assert np.isnan(f["ball_line_dist"].iloc[50])


def test_windows_do_not_cross_periods():
    n = 400
    players = static_players(n)
    players[200:] += np.arange(200)[:, None, None] * np.array([0.2, 0.0])  # period 2 players run at 5 m/s
    period = np.array([1] * 199 + [0] + [2] * 200)
    f = deadball_features(make(players, np.tile([50.0, 30.0], (n, 1)), period))
    # last frames of period 1 must not see period 2's running players in their +/-5 s window
    np.testing.assert_allclose(f["player_speed_mean_w125"].iloc[190:199], 0.0, atol=1e-5)
    assert f.iloc[199].isna().all()  # outside a period


def test_ball_out_has_negative_line_distance():
    n = 50
    f = deadball_features(make(static_players(n), np.tile([PITCH[0] + 2.0, 30.0], (n, 1)), np.ones(n)))
    assert (f["ball_line_dist"] < 0).all()
