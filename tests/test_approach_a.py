import numpy as np

from evaluation import possession_metrics as pm
from possession.deadball import p_dead_rules
from possession.decode import decode
from possession.team import nearest_team_codes, p_home_nearest
from possession.tracking import Tracking

T = 200
PITCH = (105.0, 68.0)


def make(ball, players, team=(0, 1)):
    return Tracking(ball=np.asarray(ball, dtype=np.float32),
                    players=np.asarray(players, dtype=np.float32),
                    team=np.array(team, dtype=np.int8),
                    period=np.ones(len(ball), dtype=np.int8), pitch=PITCH)


def line(start, velocity):
    """(T, 2) positions moving at `velocity` m/s from `start`."""
    return np.asarray(start) + np.arange(T)[:, None] * np.asarray(velocity) / 25.0


def running_players():
    return np.stack([line([30, 30], [4, 0]), line([40, 40], [-4, 0])], axis=1)


def test_ball_out_is_dead():
    trk = make(np.tile([PITCH[0] + 2, 30.0], (T, 1)), running_players())
    assert (p_dead_rules(trk) == np.float32(0.9)).all()


def test_running_play_is_alive():
    trk = make(line([35, 30], [5, 0]), running_players())
    assert (p_dead_rules(trk) == np.float32(0.1)).all()


def test_nearest_home():
    ball = np.tile([50.0, 30.0], (T, 1))
    players = np.stack([np.tile([51.0, 30.0], (T, 1)), np.tile([55.0, 30.0], (T, 1))], axis=1)
    trk = make(ball, players)
    assert (p_home_nearest(trk) > 0.9).all()
    assert (nearest_team_codes(trk) == 0).all()


def test_ball_missing_uninformative():
    trk = make(np.full((T, 2), np.nan), running_players())
    assert (p_home_nearest(trk) == 0.5).all()
    assert (nearest_team_codes(trk) == -1).all()


def test_team_absent():
    players = running_players()
    players[:, 1] = np.nan
    trk = make(line([35, 30], [5, 0]), players)
    p = p_home_nearest(trk)
    assert not np.isnan(p).any()
    assert (p == 0.5).all()


def test_never_visible_period_decodes():
    trk = make(np.full((T, 2), np.nan), running_players())
    codes = decode(p_dead_rules(trk), p_home_nearest(trk), [(0, T)])
    assert np.isin(codes, [pm.H, pm.A, pm.D]).all()
