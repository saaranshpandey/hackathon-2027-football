"""Approach C: a temporal convolutional network (TCN) with two heads, P(dead) and P(home | alive).

    python -m possession.nn        # train on scrambled TRAIN (+ team-swapped mirrors), early-stop on VAL

Inputs are tracking-only features at 5 frames/s; outputs go to the same decoder and 2 s rule as DB.
TEST is never used here. Training runs on the Mac GPU if available; predictions always run on CPU.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation import possession_metrics as pm
from possession.deadball import speeds
from possession.features import deadball_features
from possession.tracking import Tracking, runs_from_period

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "tcn.pt")
STEP = 5                       # 25 fps -> 5 fps
DILATIONS = (1, 2, 4, 8, 16, 32)
CHANNELS = 64
NEAR_M = 10.0
WINDOW, BATCH, BATCHES_PER_EPOCH, MAX_EPOCHS, PATIENCE = 600, 16, 200, 60, 5
SEED = 0


def receptive_steps(dilations=DILATIONS):
    """How many steps each side an output can see (kernel 3: +/- dilation per block)."""
    return int(sum(dilations))


class _Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(torch.relu(self.conv(x)))


class TCN(nn.Module):
    """(batch, features, time) -> (batch, 2, time) logits: [dead, home]."""

    def __init__(self, n_features, channels=CHANNELS, dilations=DILATIONS):
        super().__init__()
        self.inp = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.Sequential(*[_Block(channels, d) for d in dilations])
        self.head = nn.Conv1d(channels, 2, 1)

    def forward(self, x):
        return self.head(self.blocks(self.inp(x)))


def mirror(trk):
    """The same match with home and away swapped (labels must swap H and A too)."""
    return Tracking(ball=trk.ball, players=trk.players, team=(1 - trk.team).astype(np.int8),
                    period=trk.period, pitch=trk.pitch)


def nn_features(trk):
    """(T, F) float32 table: dead-ball features + per-team summaries, x oriented so home attacks right."""
    length, width = trk.pitch
    base = deadball_features(trk)
    cols = {}
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        players = trk.players.astype(np.float64).copy()
        ball = trk.ball.astype(np.float64).copy()
        for a, b in runs_from_period(trk.period):
            if np.nanmean(players[a:b, trk.team == 0, 0]) > np.nanmean(players[a:b, trk.team == 1, 0]):
                # teams changed ends: rotate the pitch 180 degrees so home always attacks right
                players[a:b, :, 0], players[a:b, :, 1] = length - players[a:b, :, 0], width - players[a:b, :, 1]
                ball[a:b, 0], ball[a:b, 1] = length - ball[a:b, 0], width - ball[a:b, 1]
        speed = np.full(players.shape[:2], np.nan)
        for a, b in runs_from_period(trk.period):  # never difference across a period edge
            speed[a:b] = speeds(players[a:b])
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
        for team, name in ((0, "home"), (1, "away")):
            p, sp, d = players[:, trk.team == team], speed[:, trk.team == team], dist[:, trk.team == team]
            cols[name + "_cx"] = np.nanmean(p[:, :, 0], axis=1)
            cols[name + "_cy"] = np.nanmean(p[:, :, 1], axis=1)
            cols[name + "_sx"] = np.nanstd(p[:, :, 0], axis=1)
            cols[name + "_sy"] = np.nanstd(p[:, :, 1], axis=1)
            cols[name + "_speed"] = np.nanmean(sp, axis=1)
            cols[name + "_near_ball"] = np.nanmin(d, axis=1)
            cols[name + "_near10"] = np.where(np.isnan(ball[:, 0]), np.nan, (d < NEAR_M).sum(axis=1))
        cols["ball_xo"] = ball[:, 0]
        cols["ball_y"] = ball[:, 1]
    extra = pd.DataFrame(cols, index=base.index, dtype=np.float32)
    extra[trk.period == 0] = np.nan
    return pd.concat([base, extra], axis=1)


def _sequences(trk, gt):
    """[(X (F, T5), dead (T5,), home (T5,) with NaN where not alive)] per period run, at 5 fps."""
    feats = nn_features(trk).to_numpy(np.float32)
    out = []
    for a, b in runs_from_period(trk.period):
        idx = np.arange(a, b, STEP)
        g = gt[idx]
        home = np.where(g == pm.H, 1.0, np.where(g == pm.A, 0.0, np.nan)).astype(np.float32)
        out.append((feats[idx].T, (g == pm.D).astype(np.float32), home))
    return out


def _upsample(prob, n):
    """5 fps outputs (step k computed at frame STEP*k) -> n frames, linearly interpolated (no hold-forward lag)."""
    steps = np.arange(len(prob)) * STEP
    return np.interp(np.arange(n), steps, prob).astype(np.float32)


def _standardise(x, mean, std):
    return np.nan_to_num((x - mean[:, None]) / std[:, None], nan=0.0).astype(np.float32)


def _swap_labels(gt):
    out = gt.copy()
    out[gt == pm.H], out[gt == pm.A] = pm.A, pm.H
    return out


def _loss(logits, dead, home):
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = bce(logits[:, 0], dead)
    alive = ~torch.isnan(home)
    if alive.any():
        loss = loss + bce(logits[:, 1][alive], home[alive])
    return loss


def train(train_ids, val_ids, device=None):
    from possession.evaluate import answer_key
    from possession.scramble import load_scrambled

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def load(ids, with_mirror):
        seqs = []
        for m in ids:
            trk, gt = load_scrambled(m)
            gt = answer_key(trk, gt, "controlled2")
            seqs += _sequences(trk, gt)
            if with_mirror:
                seqs += _sequences(mirror(trk), _swap_labels(gt))
        return seqs

    tr, va = load(train_ids, True), load(val_ids, False)
    allx = np.concatenate([x for x, _, _ in tr], axis=1)
    mean, std = np.nanmean(allx, axis=1), np.nanstd(allx, axis=1) + 1e-6
    tr = [(_standardise(x, mean, std), d, h) for x, d, h in tr]
    va = [(_standardise(x, mean, std), d, h) for x, d, h in va]

    net = TCN(len(mean)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        net.train()
        for _ in range(BATCHES_PER_EPOCH):
            xs, ds, hs = [], [], []
            for _ in range(BATCH):
                x, d, h = tr[rng.integers(len(tr))]
                s = rng.integers(0, max(1, x.shape[1] - WINDOW))
                xs.append(x[:, s:s + WINDOW]), ds.append(d[s:s + WINDOW]), hs.append(h[s:s + WINDOW])
            n = min(a.shape[1] for a in xs)
            xb = torch.tensor(np.stack([a[:, :n] for a in xs]), device=device)
            db = torch.tensor(np.stack([a[:n] for a in ds]), device=device)
            hb = torch.tensor(np.stack([a[:n] for a in hs]), device=device)
            opt.zero_grad()
            _loss(net(xb), db, hb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(np.mean([_loss(net(torch.tensor(x[None], device=device)), torch.tensor(d[None], device=device),
                                       torch.tensor(h[None], device=device)).item() for x, d, h in va]))
        print("epoch {:2d}  val loss {:.4f}".format(epoch, val), flush=True)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    columns = list(nn_features(load_scrambled(train_ids[0])[0]).columns)
    return {"state": best_state, "mean": mean, "std": std, "columns": columns, "val_loss": best}


def save(bundle, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(bundle, path)


def load_bundle(path=MODEL_PATH):
    return torch.load(path, map_location="cpu", weights_only=False)


def approach_c(trk, bundle=None):
    """(P(dead), P(home | alive)) per 25 fps frame from the TCN, computed on CPU."""
    bundle = bundle if bundle is not None else load_bundle()
    net = TCN(len(bundle["mean"]))
    net.load_state_dict(bundle["state"])
    net.eval()
    table = nn_features(trk)
    if list(table.columns) != list(bundle["columns"]):
        raise ValueError("feature columns differ from the ones the TCN was trained on; retrain: python -m possession.nn")
    feats = table.to_numpy(np.float32)
    p_dead, p_home = np.full(len(trk.period), 0.5, np.float32), np.full(len(trk.period), 0.5, np.float32)
    with torch.no_grad():
        for a, b in runs_from_period(trk.period):
            x = _standardise(feats[a:b:STEP].T, bundle["mean"], bundle["std"])
            prob = torch.sigmoid(net(torch.tensor(x[None]))).numpy()[0]
            p_dead[a:b] = _upsample(prob[0], b - a)
            p_home[a:b] = _upsample(prob[1], b - a)
    return p_dead, p_home


def main():
    from possession.external import TRAIN, VAL

    bundle = train(TRAIN, VAL)
    save(bundle)
    print("saved", MODEL_PATH, "best val loss {:.4f}".format(bundle["val_loss"]))


if __name__ == "__main__":
    main()
