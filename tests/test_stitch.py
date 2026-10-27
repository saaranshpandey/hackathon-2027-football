import numpy as np

from possession.stitch import stitch
from possession.tracking import Tracking

T = 200
SLOTS_HOME, SLOTS_AWAY = 14, 14


def base(n_home=11, n_away=11):
    """Players walking slowly; home on x in [5, 45], away on x in [60, 100]. Slot i = player i."""
    rng = np.random.default_rng(1)
    players = np.full((T, SLOTS_HOME + SLOTS_AWAY, 2), np.nan, dtype=np.float32)
    steps = np.arange(T)[:, None] * 0.05
    for i in range(n_home):
        players[:, i] = np.stack([5 + 3.5 * i + steps[:, 0], 10 + 4 * i + 0 * steps[:, 0]], axis=1)
    for i in range(n_away):
        players[:, SLOTS_HOME + i] = np.stack([60 + 3.5 * i + steps[:, 0], 12 + 4 * i + 0 * steps[:, 0]], axis=1)
    players += rng.normal(0, 0.01, players.shape).astype(np.float32)
    team = np.array([0] * SLOTS_HOME + [1] * SLOTS_AWAY, dtype=np.int8)
    return Tracking(ball=np.zeros((T, 2), dtype=np.float32), players=players, team=team,
                    period=np.ones(T, dtype=np.int8), pitch=(105.0, 68.0))


def column_of(out, source, frame):
    """Index of the output column holding `source` position at `frame`."""
    d = np.linalg.norm(out.players[frame] - source, axis=1)
    return int(np.nanargmin(d))


def test_stable_slots_keep_columns():
    trk = base()
    out = stitch(trk)
    assert out.players.shape == (T, 22, 2)
    assert out.team.tolist() == [0] * 11 + [1] * 11
    for s in list(range(11)) + list(range(SLOTS_HOME, SLOTS_HOME + 11)):
        c = column_of(out, trk.players[0, s], 0)
        np.testing.assert_allclose(out.players[:, c], trk.players[:, s], atol=1e-5)


def test_fragment_relinked_to_same_column():
    trk = base()
    # home player 3 drops out of slot 3 at frame 100 and reappears in new slot 12 at frame 110
    trk.players[110:, 12] = trk.players[110:, 3]
    trk.players[100:, 3] = np.nan
    out = stitch(trk)
    c = column_of(out, trk.players[50, 3], 50)
    np.testing.assert_allclose(out.players[110:, c], trk.players[110:, 12], atol=1e-5)
    gap = out.players[100:110, c, 0]
    assert not np.isnan(gap).any()
    assert (gap >= out.players[99, c, 0] - 1e-4).all() and (gap <= out.players[110, c, 0] + 1e-4).all()


def test_extra_slot_dropped():
    trk = base()
    trk.players[:, 11] = trk.players[:, 0] + 0.3  # a duplicate of home player 0
    out = stitch(trk)
    assert out.players.shape == (T, 22, 2)
    assert not np.isnan(out.players).any()


def test_no_nan_inside_period():
    trk = base()
    trk.players[20:60, 5] = np.nan
    trk.players[150:, SLOTS_HOME + 2] = np.nan
    out = stitch(trk)
    assert not np.isnan(out.players).any()


def test_teams_never_mix():
    trk = base()
    trk.players[100:, 3] = np.nan  # free a home column while away slots keep appearing
    trk.players[120:, SLOTS_HOME + 12] = trk.players[120:, SLOTS_HOME + 0]
    trk.players[120:, SLOTS_HOME + 0] = np.nan
    out = stitch(trk)
    assert (out.players[:, :11, 0] < 55).all()
    assert (out.players[:, 11:, 0] >= 55).all()


def test_far_extra_slot_cannot_steal_freed_column():
    trk = base()
    trk.players[:, 11] = [100.0, 67.0]          # a persistent spurious home track far away
    trk.players[110:, 12] = trk.players[110:, 3]  # player 3 breaks at 100, returns in slot 12 at 110
    trk.players[100:, 3] = np.nan
    out = stitch(trk)
    c = column_of(out, trk.players[50, 3], 50)
    assert np.nanmax(np.abs(out.players[:, c, 0] - trk.players[50, 3, 0])) < 20  # never jumps to x=100
    np.testing.assert_allclose(out.players[110:, c], trk.players[110:, 12], atol=1e-5)
