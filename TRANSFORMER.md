# Temporal Transformer experiment

Branch: `experiment/temporal-transformer`. The TCN and its checkpoint remain available.
`T` predicts both probabilities; `DBT` averages DB and T before the existing decoder.

`T_DBC` is a fixed hybrid: dead-ball probability comes entirely from T; home-given-live
probability is the average of DB and C, as in DBC. DB's PathCRF live stretches remain
defined by its original dead-ball model. The final decoder and two-second rule are unchanged.
No retraining is needed; evaluation reuses all three component signal caches.

```powershell
.venv/Scripts/python.exe -m possession.evaluate --predictors DBC DBT T_DBC --matches J03WQQ --labels controlled2 --post 50
```

This hybrid was proposed after inspecting the previous test results. Treat it as an
exploratory experiment selected on VAL; the already-inspected test match is no longer
an independent confirmation for subsequent changes.

## Windows / NVIDIA setup

From the repository root in PowerShell (Python 3.12, uv):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe numpy==2.5.4 pandas scipy scikit-learn==1.9.1 pyarrow kloppy==3.18.0 torch-geometric matplotlib tqdm pytest
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This explicitly selects CUDA wheels instead of relying on platform defaults. See
[PyTorch installation guidance](https://pytorch.org/get-started/locally/).
Use this virtual environment directly; `uv sync` can replace its packages with the
repository's existing lock resolution. The existing lock has not been changed.

## Run locally

```powershell
# No match data needed: 10 warmup steps, then 200 timed synthetic training steps.
.venv/Scripts/python.exe -m possession.transformer --benchmark --device cuda

# Train on five TRAIN matches; early-stop on the separate VAL match, never TEST.
.venv/Scripts/python.exe -m possession.transformer --device cuda

# Compare on VAL only while making choices. Includes decoding and the two-second rule.
.venv/Scripts/python.exe -m possession.evaluate --predictors C T DB DBC DBT --matches J03WQQ --labels controlled2 --post 50

# After selecting settings, evaluate once on the held-out match.
.venv/Scripts/python.exe -m possession.evaluate --predictors C T DB DBC DBT --matches J03WR9 --labels controlled2 --post 50

# Produce a submission when the Barcelona match assets are installed.
.venv/Scripts/python.exe -m possession.run --approach DBT --device cuda --post 50 --name DBT
```

Best checkpoints are written atomically to `models/transformer.pt` after each improvement.
`--output models/transformer-seed1.pt --seed 1` keeps another run separately; evaluation
and submission commands load the default checkpoint. Copy a selected checkpoint to
that default path before evaluating it. Signal caches fingerprint the checkpoint.

Training needs the existing scrambled DFL data pipeline. With no cached data, its
loader downloads DFL tracking and requires the Barcelona-derived defect profile
(see `possession.scramble`). Feature preparation runs on CPU and may dominate the
first run. This checkout does not include those match assets.

After installing the Barcelona assets as described in `DATA.md`, create the defect
profile once if `data/cache/defects.json` is absent:

```powershell
.venv/Scripts/python.exe -c "from possession.data import load_match, CACHE_DIR; from possession.scramble import measure; from pathlib import Path; t = load_match(); measure(t).to_json(str(Path(CACHE_DIR) / 'defects.json'))"
```

Local verification on the RTX 5070 Ti with PyTorch 2.11.0+cu128: a 100-step
synthetic benchmark at batch 128 measured about 294 batches/s and 99 MiB peak
allocated GPU memory. This is a compute smoke check, not a match-training time
estimate or an accuracy result.

## Implementation and limits

- Two pre-norm layers, width 64, four attention heads, sinusoidal positions, FFN width 128.
- Same 44 tracking features, 5 fps, two BCE heads and team-swapping augmentation as C.
  Unknown labels are masked. Normalization uses TRAIN only.
- 128-step windows (25.6 seconds), stride 32 at inference. This matches the TCN's
  approximate *total* context length, not its centered receptive field at every
  position: Transformer context varies with position inside each window.
- Normalized training sequences and validation sequences stay on the GPU. Random
  windows are gathered there without per-batch CPU loading or transfers. Windows
  cannot cross half boundaries. Short sequences replicate feature edges and mask
  padded targets. Training samples halves uniformly, matching the TCN sampler.
- CUDA BF16 autocast when supported, otherwise FP16 with gradient scaling;
  `--fp32` disables autocast. Fused AdamW, gradient clipping, zero-grad with
  `set_to_none`, and PyTorch scaled-dot-product attention. SDPA chooses available
  kernels automatically; fused attention is not assumed on every Windows build.
- Inference uses batches of overlapping windows and weighted probability averaging.
  No quadratic full-half attention. Predictions interpolate back to 25 fps.
- Defaults: batch 128, 200 steps/epoch, 60 epochs maximum, patience 5, LR 0.0003.
  `--batch 64` reduces memory. `--steps 10 --epochs 1` is a data-pipeline smoke run.
  Epoch logs report elapsed time and peak allocated GPU memory. Synthetic benchmark
  timing excludes preprocessing, validation, and checkpoint writes.
- No compilation dependency or Windows DataLoader workers. CPU inference is available
  with `--device cpu` on submission generation; evaluation auto-selects CUDA for T.
  DB evaluation retains its existing CPU execution and cache behavior.

Early stopping uses validation probability loss. Model selection should use decoded
spellF1 on VAL. A faster run or lower BCE does not establish a better possession model.
The training budget and optimizer differ from the original TCN; keep that distinction
in mind when attributing improvements to attention alone.

## Full-match visual review

After producing `submissions/T_DBC.csv` and `submissions/DBC.csv`:

```powershell
.venv/Scripts/python.exe -m possession.review
Start-Process .\out\review\index.html
```

Open the local player in Edge or Chrome. It uses the original full tactical MP4,
with a synchronized possession label and top-down tracking map. It requires no
server or video re-encoding. Keep the player folder in place because the video path
is relative to it. Labels update at 25 fps; the map holds observations sampled at
5 fps. The overlay remains visible using the player's Fullscreen button.

Filter the sidebar for ball gaps, short spells, restarts, model disagreements, or
turnovers near gaps. Clicking an item seeks five seconds before it. The complete
candidate list is also saved to `out/review/review-candidates.csv`. These flags are
diagnostics, not verified errors. The public five-minute ground truth is displayed alongside predictions only where
labelled; ungraded frames are marked unavailable. The Public ground truth button
jumps to its start. Private labels are never read, and the frozen submission is
never edited. Review candidates remain tracking/model diagnostics, independent
of ground truth. Player code is versioned; generated video-derived data stays
under ignored `out/` and is for local inspection.
