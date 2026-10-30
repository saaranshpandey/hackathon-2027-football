import numpy as np
import pytest
import torch

from possession.transformer import (Config, TemporalTransformer, WindowBank, masked_loss,
                                    window_probabilities, approach_t, save_checkpoint)
from possession.nn import nn_features, load_bundle
from tests.test_nn import make


def test_masked_loss_and_backward_with_no_live_labels():
    net = TemporalTransformer(4, Config(window=16, stride=4))
    y = torch.ones(2, 2, 16)
    y[:, 1] = float('nan')
    loss = masked_loss(net(torch.randn(2, 4, 16)), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    assert masked_loss(torch.zeros_like(y), torch.full_like(y, float('nan'))) == 0


def test_bank_never_crosses_half_and_masks_short_padding():
    seqs = [(np.full((3, n), value, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32))
            for n, value in [(5, 1), (23, 2)]]
    bank = WindowBank(seqs, 8, 'cpu')
    x, y = bank.sample(100)
    assert (x == x[:, :, :1]).all()
    short = x[:, 0, 0] == 1
    assert short.any()
    assert torch.isnan(y[short, :, 5:]).all()
    assert torch.isfinite(y[short, :, :5]).all()


@pytest.mark.parametrize('n', [1, 7, 16, 17, 53])
def test_inference_covers_tail_and_is_batch_independent(n):
    net = TemporalTransformer(4, Config(window=16, stride=4)).eval()
    x = torch.randn(4, n)
    a = window_probabilities(net, x, batch=1)
    b = window_probabilities(net, x, batch=3)
    assert a.shape == (2, n)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_checkpoint_and_period_isolation(tmp_path):
    trk = make([20, 30], [70, 80], n=120)
    trk.period[50:60] = 0
    trk.period[60:] = 2
    cols = list(nn_features(trk).columns)
    cfg = Config(window=16, stride=4)
    net = TemporalTransformer(len(cols), cfg)
    from dataclasses import asdict
    bundle = dict(state=net.state_dict(), config=asdict(cfg), columns=cols,
                  mean=np.zeros(len(cols)), std=np.ones(len(cols)))
    path = tmp_path / 'test.pt'
    save_checkpoint(bundle, path)
    bundle = load_bundle(path)
    a = approach_t(trk, bundle, device='cpu')
    assert np.all(np.array(a)[:, 50:60] == .5)
    trk.players[60:] += 10
    b = approach_t(trk, bundle, device='cpu')
    np.testing.assert_allclose(np.array(a)[:, :50], np.array(b)[:, :50])
    bundle['columns'] = cols[::-1]
    with pytest.raises(ValueError, match='columns'):
        approach_t(trk, bundle, device='cpu')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_saves_loadable_checkpoint_without_test_leakage(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from argparse import Namespace
    from possession import transformer, external, scramble
    from evaluation import possession_metrics as pm
    seen = []
    def fake_load(match):
        seen.append(match)
        trk = make([20, 30], [70, 80], n=100)
        gt = np.full(100, pm.H, dtype=np.int8)
        gt[40:60] = pm.D
        return trk, gt
    monkeypatch.setattr(scramble, 'load_scrambled', fake_load)
    path = tmp_path / 'trained.pt'
    args = Namespace(device=device, seed=0, window=16, stride=4, lr=3e-4,
                     fp32=False, epochs=1, steps=2, batch=2, patience=1, output=str(path))
    transformer.train(args)
    assert set(seen) == set(external.TRAIN + external.VAL)
    assert not set(seen) & set(external.TEST)
    bundle = load_bundle(path)
    assert np.isfinite(bundle['val_loss'])
    trk, _ = fake_load('synthetic')
    result = approach_t(trk, bundle, device=device)
    assert np.isfinite(result).all()


def test_ensemble_uses_component_signal_caches(tmp_path, monkeypatch):
    from possession import evaluate
    trk = make([20, 30], [70, 80])
    calls = []
    monkeypatch.setattr(evaluate, 'SIGNAL_DIR', str(tmp_path))
    monkeypatch.setattr(evaluate, '_fingerprint', lambda paths: 'test')
    for name, value in [('DB', .2), ('T', .8)]:
        def predict(track, name=name, value=value):
            calls.append(name)
            return (np.full(len(track.period), value), np.full(len(track.period), value))
        monkeypatch.setitem(evaluate.SIGNALS, name, predict)
    for _ in range(2):
        result = evaluate.signals('DBT', 'synthetic', trk)
        np.testing.assert_allclose(result, .5)
    assert calls == ['DB', 'T']
