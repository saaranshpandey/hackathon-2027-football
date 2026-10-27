import numpy as np

from possession.data import parse_atd
from possession.tracking import Tracking, load, runs_from_period, save

META_XML = """<?xml version="1.0" encoding="utf-8"?>
<Main>
  <Teams>
    <Team id="T1"><Name>Home FC</Name></Team>
    <Team id="T2"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T1"><Name>Track 1</Name></Player>
    <Player id="P2" teamId="T1"><Name>Track 2</Name></Player>
    <Player id="P3" teamId="T2"><Name>Track 3</Name></Player>
  </Players>
</Main>
"""

RAW = (
    "1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,0.0:0.4,0.5,3.0\n"
    "2:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n"
)


def test_runs_from_period():
    assert runs_from_period(np.array([0, 1, 1, 0, 2, 2, 2, 0])) == [(1, 3), (4, 7)]


def test_parse_atd_synthetic(tmp_path):
    meta = tmp_path / "meta.xml"
    raw = tmp_path / "raw.txt"
    meta.write_text(META_XML)
    raw.write_text(RAW)

    trk = parse_atd(str(raw), str(meta), n_frames=2, pitch=(100.0, 50.0))

    np.testing.assert_allclose(trk.ball[0], [40.0, 25.0])
    assert np.isnan(trk.ball[1]).all()
    np.testing.assert_allclose(trk.players[0, 0], [50.0, 25.0])
    np.testing.assert_allclose(trk.players[0, 2], [10.0, 10.0])
    assert np.isnan(trk.players[0, 1]).all()
    assert trk.team.tolist() == [0, 0, 1]
    assert trk.players.shape == (2, 3, 2)
    assert trk.pitch == (100.0, 50.0)


def test_save_load_roundtrip(tmp_path):
    trk = Tracking(
        ball=np.array([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32),
        players=np.full((2, 3, 2), np.nan, dtype=np.float32),
        team=np.array([0, 0, 1], dtype=np.int8),
        period=np.array([1, 1], dtype=np.int8),
        pitch=(105.3, 68.0),
    )
    path = tmp_path / "trk.npz"
    save(trk, str(path))
    back = load(str(path))

    np.testing.assert_array_equal(back.ball, trk.ball)
    np.testing.assert_array_equal(back.players, trk.players)
    np.testing.assert_array_equal(back.team, trk.team)
    np.testing.assert_array_equal(back.period, trk.period)
    assert back.pitch == trk.pitch


def test_load_match_recomputes_period_from_games(tmp_path, monkeypatch):
    import possession.data as data

    n = 199319
    stale = Tracking(ball=np.full((n, 2), np.nan, dtype=np.float32),
                     players=np.full((n, 1, 2), np.nan, dtype=np.float32),
                     team=np.array([0], dtype=np.int8),
                     period=np.zeros(n, dtype=np.int8), pitch=(105.3, 68.0))
    save(stale, str(tmp_path / "atd_tactical.npz"))
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))

    trk = data.load_match()
    assert int((trk.period > 0).sum()) == 158857
    assert trk.period[11185] == 1 and trk.period[90651] == 1 and trk.period[90652] == 0
    assert trk.period[110808] == 2 and trk.period[190197] == 2 and trk.period[190198] == 0
