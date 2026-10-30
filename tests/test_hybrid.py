import numpy as np

from possession import evaluate, hybrid, run
from tests.test_nn import make


def test_hybrid_heads_cache_and_submission_agree(tmp_path, monkeypatch):
    from possession import nn, pathcrf, transformer
    trk = make([20, 30], [70, 80])
    values = {'T': (.8, .99), 'DB': (.1, .2), 'C': (.3, .6)}
    calls = []
    monkeypatch.setattr(evaluate, 'SIGNAL_DIR', str(tmp_path))
    monkeypatch.setattr(evaluate, '_fingerprint', lambda paths: 'test')
    for name, pair in values.items():
        def predictor(track, name=name, pair=pair, **kwargs):
            calls.append(name)
            return tuple(np.full(len(track.period), v) for v in pair)
        monkeypatch.setitem(evaluate.SIGNALS, name, predictor)
        module, function = {'T': (transformer, 'approach_t'), 'DB': (pathcrf, 'approach_db'),
                            'C': (nn, 'approach_c')}[name]
        monkeypatch.setattr(module, function, predictor)
    for _ in range(2):
        result = evaluate.signals('T_DBC', 'synthetic', trk)
        np.testing.assert_allclose(result[0], .8)
        np.testing.assert_allclose(result[1], .4)
    assert calls == ['T', 'DB', 'C']
    actual = run.APPROACHES['T_DBC'](trk, 'cpu')
    np.testing.assert_allclose(actual, result)
