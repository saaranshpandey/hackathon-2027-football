import numpy as np

from evaluation import possession_metrics as pm
from possession.evaluate import score


def two_run_truth():
    # run 1: H H H A A A D D, gap X X, run 2: A A A H H H
    letters = list("HHHAAADD") + ["X", "X"] + list("AAAHHH")
    return pm.label_codes(np.array(letters, dtype=object))


def test_score_perfect():
    gt = two_run_truth()
    result = score(gt, gt)
    assert result["spell_f1"] == 1.0
    assert result["n_gt"] == result["n_pred"] == 4


def test_score_constant():
    gt = two_run_truth()
    pred = np.where(gt == pm.X, pm.X, pm.D).astype(np.int8)
    assert score(gt, pred)["spell_f1"] == 0.0


def _fake_signal(monkeypatch, tmp_path, calls):
    import possession.evaluate as ev
    from possession.tracking import Tracking

    source = tmp_path / "fake_source.py"
    source.write_text("version 1")
    monkeypatch.setattr(ev, "SIGNAL_DIR", str(tmp_path / "signals"))
    monkeypatch.setitem(ev.SIGNALS, "FAKE", lambda trk: (calls.append(1) or np.full(len(trk.period), 0.2),
                                                         np.full(len(trk.period), 0.7)))
    monkeypatch.setitem(ev.SIGNAL_SOURCES, "FAKE", [str(source)])
    trk = Tracking(ball=np.zeros((10, 2), np.float32), players=np.zeros((10, 2, 2), np.float32),
                   team=np.array([0, 1], np.int8), period=np.ones(10, np.int8), pitch=(105.0, 68.0))
    return ev, trk, source


def test_signal_cache_reused_when_nothing_changed(monkeypatch, tmp_path):
    calls = []
    ev, trk, _ = _fake_signal(monkeypatch, tmp_path, calls)
    ev.signals("FAKE", "M1", trk)
    p_dead, p_home = ev.signals("FAKE", "M1", trk)
    assert len(calls) == 1
    np.testing.assert_allclose(p_home, 0.7)


def test_signal_cache_recomputed_when_a_source_changes(monkeypatch, tmp_path):
    calls = []
    ev, trk, source = _fake_signal(monkeypatch, tmp_path, calls)
    ev.signals("FAKE", "M1", trk)
    source.write_text("version 2")
    ev.signals("FAKE", "M1", trk)
    assert len(calls) == 2


def test_signal_cache_rejects_wrong_length(monkeypatch, tmp_path):
    calls = []
    ev, trk, _ = _fake_signal(monkeypatch, tmp_path, calls)
    ev.signals("FAKE", "M1", trk)
    trk.period = np.ones(12, np.int8)
    p_dead, _ = ev.signals("FAKE", "M1", trk)
    assert len(calls) == 2 and len(p_dead) == 12
