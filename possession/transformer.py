"""Small bidirectional temporal Transformer; CUDA-resident windows, no full-half attention.

python -m possession.transformer --device cuda
python -m possession.transformer --device cuda --benchmark
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from possession.nn import (STEP, _sequences, _standardise, _swap_labels, _upsample,
                           load_bundle, mirror, nn_features)
from possession.tracking import runs_from_period

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'transformer.pt'


@dataclass
class Config:
    width: int = 64
    heads: int = 4
    layers: int = 2
    window: int = 128
    stride: int = 32
    dropout: float = 0.1

    def __post_init__(self):
        if (self.width < 2 or self.heads < 1 or self.width % self.heads or
                self.width % 2 or self.layers < 1 or not 0 < self.stride <= self.window
                or not 0 <= self.dropout < 1):
            raise ValueError('Invalid Transformer dimensions/window/stride/dropout')


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 2 * cfg.width), nn.GELU(),
                                nn.Linear(2 * cfg.width, cfg.width))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # PyTorch selects an available fused SDPA kernel. Residual dropout keeps
        # attention dropout zero, allowing fast kernels on more Windows builds.
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, c)))
        return x + self.drop(self.ff(self.norm2(x)))


class TemporalTransformer(nn.Module):
    """(batch, features, time) -> (batch, 2, time), dead/home logits."""
    def __init__(self, n_features, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.inp = nn.Linear(n_features, c.width)
        pos = torch.arange(c.window)[:, None]
        freq = torch.exp(torch.arange(0, c.width, 2) * (-math.log(10000) / c.width))
        pe = torch.zeros(c.window, c.width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer('position', pe)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.layers)])
        self.head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, 2))

    def forward(self, x):
        if not 0 < x.shape[-1] <= self.cfg.window:
            raise ValueError('Input must fit the configured window')
        z = self.inp(x.transpose(1, 2))
        z = z + self.position[:z.shape[1]].to(z.dtype)
        return self.head(self.blocks(z)).transpose(1, 2)


def device_for(name='auto'):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch; see TRANSFORMER.md.')
    return device


def autocast(device, enabled=True):
    if device.type != 'cuda' or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast('cuda', dtype=dtype)


def masked_loss(logits, targets):
    """Equal head weights; ignore unknown labels without CUDA-to-CPU branching."""
    valid = torch.isfinite(targets)
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.nan_to_num(), reduction='none')
    return ((losses * valid).sum((0, 2)) / valid.sum((0, 2)).clamp_min(1)).sum()


class WindowBank:
    """Store sequences once on GPU; vectorized sampling never crosses a half."""
    def __init__(self, sequences, window, device):
        xs, ys, starts, counts = [], [], [], []
        offset = 0
        for x, d, h in sequences:
            n = x.shape[1]
            if not n:
                continue
            pad = max(0, window - n)
            xs.append(np.pad(x, ((0, 0), (0, pad)), mode='edge'))
            ys.append(np.pad(np.stack([d, h]), ((0, 0), (0, pad)), constant_values=np.nan))
            starts.append(offset)
            counts.append(max(1, n - window + 1))
            offset += n + pad
        if not xs:
            raise ValueError('No training sequences')
        self.x = torch.as_tensor(np.concatenate(xs, axis=1).T.copy(), device=device)
        self.y = torch.as_tensor(np.concatenate(ys, axis=1).T.copy(), device=device)
        self.starts = torch.tensor(starts, device=device)
        self.counts = torch.tensor(counts, device=device)
        self.steps = torch.arange(window, device=device)

    def sample(self, batch):
        seq = torch.randint(len(self.starts), (batch,), device=self.x.device)
        start = self.starts[seq] + (torch.rand(batch, device=self.x.device) * self.counts[seq]).long()
        idx = start[:, None] + self.steps
        return self.x[idx].transpose(1, 2), self.y[idx].transpose(1, 2)


@torch.inference_mode()
def window_probabilities(net, x, batch=128, amp=True):
    """Overlap-add probabilities on device; linear memory, complete tail coverage."""
    if batch < 1 or x.shape[1] < 1:
        raise ValueError('Batch and sequence length must be positive')
    net.eval()
    n = x.shape[1]
    window = min(n, net.cfg.window)
    starts = list(range(0, n - window + 1, net.cfg.stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    total = torch.zeros(2, n, device=x.device)
    count = torch.zeros(n, device=x.device)
    # Favor predictions with more context on both sides; endpoints stay nonzero.
    weight = 1 - torch.abs(torch.linspace(-1, 1, window, device=x.device)) * .95
    steps = torch.arange(window, device=x.device)
    for s in range(0, len(starts), batch):
        idx = torch.tensor(starts[s:s + batch], device=x.device)[:, None] + steps
        xb = x.T[idx].transpose(1, 2)
        with autocast(x.device, amp):
            prob = net(xb).float().sigmoid()
        total.index_add_(1, idx.flatten(), (prob * weight).permute(1, 0, 2).reshape(2, -1))
        count.index_add_(0, idx.flatten(), weight.expand(len(idx), -1).reshape(-1))
    return total / count


def approach_t(trk, bundle=None, device='auto', batch=128):
    bundle = load_bundle(MODEL_PATH) if bundle is None else bundle
    device = device_for(device)
    net = TemporalTransformer(len(bundle['mean']), Config(**bundle['config'])).to(device)
    net.load_state_dict(bundle['state'])
    table = nn_features(trk)
    if list(table.columns) != list(bundle['columns']):
        raise ValueError('Transformer feature columns differ; retrain')
    feats = table.to_numpy(np.float32)
    out = np.full((2, len(trk.period)), .5, dtype=np.float32)
    for a, b in runs_from_period(trk.period):
        x = _standardise(feats[a:b:STEP].T, bundle['mean'], bundle['std'])
        prob = window_probabilities(net, torch.as_tensor(x, device=device), batch).cpu().numpy()
        out[:, a:b] = np.stack([_upsample(p, b - a) for p in prob])
    return out[0], out[1]


def approach_dbt(trk, device='auto'):
    from possession.pathcrf import approach_db
    device = str(device_for(device))
    db = approach_db(trk, device=device)
    t = approach_t(trk, device=device)
    return tuple((a + b) / 2 for a, b in zip(db, t))


def save_checkpoint(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    torch.save(bundle, temp)
    temp.replace(path)


def train(args):
    from evaluation import possession_metrics as pm
    from possession.evaluate import answer_key
    from possession.external import TRAIN, VAL
    from possession.scramble import load_scrambled

    device = device_for(args.device)
    torch.manual_seed(args.seed)
    cfg = Config(window=args.window, stride=args.stride)
    columns = None
    def load(ids, augment):
        nonlocal columns
        seqs = []
        for match in ids:
            print(f'Preparing {match} (CPU tracking features)', flush=True)
            trk, gt = load_scrambled(match)
            if columns is None:
                columns = list(nn_features(trk).columns)
            gt = answer_key(trk, gt, 'controlled2')
            for track, labels in [(trk, gt)] + ([(mirror(trk), _swap_labels(gt))] if augment else []):
                chunks = _sequences(track, labels)
                for (a, b), (x, d, h) in zip(runs_from_period(track.period), chunks):
                    d[labels[a:b:STEP] == pm.X] = np.nan
                    seqs.append((x, d, h))
        return seqs
    tr, va = load(TRAIN, True), load(VAL, False)
    allx = np.concatenate([x for x, _, _ in tr], axis=1)
    mean = np.nan_to_num(np.nanmean(allx, axis=1))
    std = np.nan_to_num(np.nanstd(allx, axis=1), nan=1.) + 1e-6
    del allx
    bank = WindowBank([(_standardise(x, mean, std), d, h) for x, d, h in tr], cfg.window, device)
    validation = [(torch.as_tensor(_standardise(x, mean, std), device=device),
                   torch.as_tensor(np.stack([d, h]), device=device)) for x, d, h in va]
    net = TemporalTransformer(len(mean), cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01, fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported() and not args.fp32)
    best, bad = math.inf, 0
    print(f'{device}: {sum(p.numel() for p in net.parameters()):,} parameters; training data resident on device', flush=True)
    for epoch in range(args.epochs):
        start = time.perf_counter()
        net.train()
        for _ in range(args.steps):
            x, y = bank.sample(args.batch)
            opt.zero_grad(set_to_none=True)
            with autocast(device, not args.fp32):
                loss = masked_loss(net(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sums, counts = torch.zeros(2, device=device), torch.zeros(2, device=device)
        for x, y in validation:
            prob = window_probabilities(net, x, args.batch, not args.fp32).clamp(1e-6, 1 - 1e-6)
            valid = torch.isfinite(y)
            sums += (F.binary_cross_entropy(prob, y.nan_to_num(), reduction='none') * valid).sum(1)
            counts += valid.sum(1)
        val = (sums / counts.clamp_min(1)).sum().item()
        if not math.isfinite(val):
            raise RuntimeError('Non-finite validation loss; retry --fp32 and inspect data')
        elapsed = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0
        print(f'epoch {epoch + 1}: val={val:.5f}, {elapsed:.1f}s, peak GPU={memory:.0f} MiB', flush=True)
        if val < best - 1e-4:
            best, bad = val, 0
            save_checkpoint({'state': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                             'config': asdict(cfg), 'mean': mean, 'std': std, 'columns': columns,
                             'val_loss': best, 'epoch': epoch + 1, 'train_ids': TRAIN, 'val_ids': VAL,
                             'args': vars(args)}, args.output)
        else:
            bad += 1
            if bad >= args.patience:
                break
    print(f'Saved best checkpoint: {args.output} (validation loss {best:.5f})')


def benchmark(args):
    """Synthetic compute benchmark, excluding preprocessing and validation."""
    device = device_for(args.device)
    net = TemporalTransformer(44, Config(window=args.window, stride=args.stride)).to(device)
    opt = torch.optim.AdamW(net.parameters(), fused=device.type == 'cuda')
    x = torch.randn(args.batch, 44, args.window, device=device)
    y = torch.randint(2, (args.batch, 2, args.window), device=device).float()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported() and not args.fp32)
    def step():
        opt.zero_grad(set_to_none=True)
        with autocast(device, not args.fp32):
            loss = masked_loss(net(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss
    for _ in range(10):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if not torch.isfinite(loss).item():
        raise RuntimeError('Benchmark loss is not finite')
    print(f'{device}: {args.steps / seconds:.1f} batches/s; {seconds:.2f}s for {args.steps} steps; loss={loss.item():.4f}')
    if device.type == 'cuda':
        print(f'{torch.cuda.get_device_name(device)}; torch={torch.__version__}; peak allocated={torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--window', type=int, default=128)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default=str(MODEL_PATH))
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()
    if min(args.batch, args.epochs, args.steps, args.patience) < 1 or args.lr <= 0:
        p.error('batch, epochs, steps, patience and learning rate must be positive')
    (benchmark if args.benchmark else train)(args)


if __name__ == '__main__':
    main()
