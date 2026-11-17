# Ball possession from tracking: the T_DBC hybrid

Technical document for More than a Hack 2027, Phase 1. Submission file: `submissions/T_DBC.csv`,
produced by `python -m possession.run --approach T_DBC --device cuda --post 50`.

## 1. Approach

The submission is produced from the provided tactical tracking (ATD) only. Three learned signals
estimate, for every frame, two probabilities: **P(dead)**, that the ball is out of play, and
**P(home | alive)**, that Barcelona has the ball given play is live. A decoder turns those
probabilities into clean H / A / D possessions, then a controlled-possession rule removes team
changes that never became control.

```
ATD tracking ──► 44 features @ 5 fps ──► T  (Transformer) ─────────────► P(dead)
            │                       └──► C  (TCN) ──────────┐
            └──► stitched 11+11 players ► DB (PathCRF) ─────┴─ average ─► P(home | alive)
                                                                              │
             Viterbi decoder (H/A/D, switch costs) ◄──────────────────────────┘
                              │
             2 s controlled-possession rule ──► 25 fps labels
```

**Why this split.** Whether the ball is dead and who has it are different problems with different
evidence. Dead balls show up as global patterns (players slowing down, walking back, the ball
leaving the field or disappearing). Possession needs the ball-player relationship, but the ball is
often unobserved in this tracking. Keeping the two heads separate lets each come from a different
model. The combination was chosen by decoded spellF1 on the validation match: dead balls from the
Transformer, the team from an average of PathCRF and a TCN.

### Components

- **Features (44 per frame, 5 fps).** 28 dead-ball features: ball visibility, speed, distance to
  the touchline and corners, length of the current missing-ball run, player speed statistics,
  count of slow players, team spread, players near the ball and in the boxes, and rolling versions
  over 0.5, 2 and 5 s. 16 team summaries: centroid, spread, mean speed, nearest-player distance
  and players within 10 m of the ball for each team, plus ball position. Each half is rotated so
  home always attacks right. Nothing is computed across a half boundary.
- **T, temporal Transformer** (`possession/transformer.py`): 2 pre-norm layers, width 64, 4
  heads, feed-forward 128, sinusoidal positions, 128-step windows (25.6 s). Inference uses
  overlapping windows (stride 32) with weighted averaging, interpolated back to 25 fps. Two
  binary heads (dead; home given live), unknown labels masked, team-swap augmentation. Trained on
  5 DFL matches and early-stopped on validation loss (best epoch 14). **Only its P(dead) is used.**
- **C, temporal convolutional network** (`possession/nn.py`): residual dilated 1-D convolutions
  (dilations 1-32, 64 channels), same features and heads. **Its P(home | alive) is used.**
- **DB, PathCRF + learned dead-ball model** (`possession/pathcrf.py`): PathCRF (Kim et al., KDD
  2026) infers the ball possessor from player trajectories alone. We use the authors' published
  checkpoint (vendored in `third_party/pathcrf`, MPL-2.0) without retraining. Because it needs 11
  identity-continuous players per team, our anonymous, fragmented track slots are first stitched
  into 11+11 continuous players (`possession/stitch.py`). It runs on live stretches defined by a
  gradient-boosted dead-ball model (scikit-learn HistGradientBoosting on the 28 dead-ball
  features). **Its P(home | alive) is used.**
- **Fusion** (`possession/hybrid.py`): P(dead) = T; P(home | alive) = ½ (DB + C).
- **Decoder** (`possession/decode.py`): three-state Viterbi over H, A, D, run separately in each
  half. Changing between H and A costs 8, entering or leaving D costs 4 (log-probability units),
  so a few noisy frames cannot split a possession.

### Making German training data look like this match

The DFL training tracking is clean; this match's tracking has anonymous, fragmented tracks,
hidden players, identity swaps and long missing-ball stretches. `possession/scramble.py` measures
**aggregate defect statistics** of the provided tracking (no labels, no frame-specific values)
and applies them to the DFL matches, so models are trained and validated on input that looks like
the test match. This is the only use of the provided game during training.

## 2. Assets used

| Asset | Used? | How |
|---|---|---|
| Tactical ATD tracking | **Yes** | Sole input to every model; aggregate defect statistics drive the scrambling |
| Kickoff frames (`starter/load_smart_tagging.py`) | **Yes** | Graded frames before each half's kickoff are set to dead |
| Smart Tagging restarts | Tried, dropped | Extra dead-ball evidence before tagged restarts: no gain on validation (0.7459 vs 0.7466) |
| Homography | No | |
| Tactical video | Not in the pipeline | Watched in a local review player to debug; an audio whistle detector was built and dropped (section 5) |
| Broadcast video/tracking | No | |
| Public 5-minute ground truth | Pipeline check only | Never used for training, tuning or model selection |
| DFL open data (external) | **Yes** | Training and validation labels (section 3) |
| PathCRF checkpoint (external) | **Yes** | Team signal, used as published |

## 3. Training data

No labels from the provided game were used, by hand or otherwise. We created no labels of our own.

| Dataset | Source | Labelled by | Amount | Used for |
|---|---|---|---|---|
| Sportec Open DFL Dataset (Bassek et al., *Scientific Data* 2025, CC-BY 4.0) | Hugging Face mirror `pysport/idsse-data` | DFL / Sportec Solutions: per-frame ball status (alive/dead) and possessing team in the official tracking feed | 7 Bundesliga / 2. Bundesliga matches | Training T, C and the dead-ball model; validation and test |
| Same dataset, event data synchronised with ELASTIC | PathCRF repository (`third_party/pathcrf/data/sportec_synced`) | DFL event data (manually annotated), synchronised by the PathCRF authors | 7 matches | Training of the published PathCRF checkpoint (by its authors, fold: test J03WR9, val J03WQQ). We used these files only to locate German restarts in the dropped restart experiment |

Split: train J03WMX, J03WN1, J03WOH, J03WOY, J03WPY · validation J03WQQ · test J03WR9.
The DFL labels are converted to H/A/D directly (dead → D, otherwise the possessing team).

## 4. Possession rules implemented

- **Dead ball → D.** Learned from DFL's ball-status labels: out of play, stoppages and the frames
  before each kickoff. Frames outside the two halves are X; no graded frame is left as X.
- **Loose balls stay with the last team.** When the team signal is uninformative, the decoder's
  switching cost keeps the current possessor until another team's evidence is strong.
- **Touches, deflections and challenges are not changes of possession.** A change of team only
  counts if the new team keeps the ball for at least **2 seconds** (50 frames); shorter spells
  are returned to the team that had it before (`possession/controlled.py`). This follows the
  challenge definition of *controlled* possession; the 2 s threshold was fixed before looking at
  any result, and the same rule is applied to the DFL answer key used for validation.
- **Restarts have a clear possessor.** The first spell after any stoppage is always kept, however
  short, because the restarting team has the ball.

## 5. Results and model selection

spellF1 (ranking metric) and macroF1, scrambled DFL matches with the 2 s answer key, and the
public 5-minute window of this match.

| Model | Validation J03WQQ | Test J03WR9 | Mean | Public window spellF1 / macroF1 |
|---|---:|---:|---:|---:|
| **T_DBC (submitted)** | **0.7466** | **0.7900** | **0.7683** | **0.6857 / 0.7957** |
| T_DBC + outside-pitch rule | 0.7466 | 0.7892 | 0.7679 | |
| DBC (½ DB + ½ C, both heads) | 0.7349 | 0.7878 | 0.7614 | 0.6378 / 0.7935 |
| DB | 0.7030 | 0.7256 | 0.7143 | |
| C | 0.7028 | 0.7153 | 0.7091 | |
| T alone | 0.6861 | 0.6980 | 0.6921 | |
| baseline_smooth50 (organisers) | | | | 0.3256 / 0.6207 |

On the public window T_DBC predicts 18 possessions against 17 real ones.

**Tried and dropped.** Hand-written dead-ball rules with a nearest-player team signal (approach
A); PathCRF with rule-based dead balls (D); averaging DB with T on both heads (DBT); an explicit
ball-outside-the-pitch rule (no measurable gain, see `BOUNDARY.md`); Smart Tagging restarts as
dead-ball evidence (no gain); an audio whistle detector using an AudioSet model on the match audio
(too unreliable to use without synchronised labels, see `AUDIO_REVIEW.md`).

## 6. Limitations

- **Small margin over the previous model.** T_DBC beats DBC by 0.007 spellF1 on the DFL matches,
  well inside the ~0.04 noise band in `EVALUATION.md`.
- **Optimistic validation numbers.** The T_DBC combination was chosen after the test match had
  been inspected for earlier models, so the test score is no longer independent. The published
  PathCRF checkpoint holds out J03WR9 but, under its authors' leave-one-match-out protocol, used
  J03WQQ to select the checkpoint, so DB, DBC and T_DBC validation scores slightly flatter the
  team signal.
- **Domain shift.** Five German matches with simulated defects stand in for a LaLiga match from a
  tactical camera; the scrambling copies aggregate defect statistics, not the exact failure modes.
- **The 2 s rule removes genuine quick turnovers**, such as an interception lost again within two
  seconds.
- **Stoppages inside the pitch** (fouls, injuries) are harder than balls leaving the field, and
  long missing-ball stretches leave the team signal to player movement alone.
- **With more time:** retrain PathCRF without the validation match; adapt to this match's tracking
  with self-supervised learning; average several training seeds; detect the ball in the video to
  fill tracking gaps.

## 7. Reproduction

Environment used: Windows 11, Python 3.12.12, uv 0.10.6, NVIDIA GeForce RTX 5070 Ti, PyTorch
2.11.0 with CUDA 12.8. All trained weights are committed: `models/transformer.pt`,
`models/tcn.pt`, `models/deadball_hgb.joblib` and the vendored PathCRF checkpoint.

```powershell
# 1. Environment (exact versions)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-submission.txt --index-strategy unsafe-best-match

# 2. Data: unpack the match bundle into data/ as described in data/README.md
#    (data/tactical/BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt are what is read)

# 3. Generate the submission -> submissions/T_DBC.csv
.venv/Scripts/python.exe -m possession.run --approach T_DBC --device cuda --post 50 --name T_DBC
.venv/Scripts/python.exe evaluation/validate_submission.py submissions/T_DBC.csv
```

Expected SHA-256 of `submissions/T_DBC.csv`:
`3f4878340177bee5d22d825d4370710567ade8d617032cf6aee54e0ac4aa20bc`.
This was reproduced byte for byte from the committed code and weights. **Use `--device cuda`:**
the CPU path gives the same public-window score but differs on 111 of 199,319 frames (0.06%)
through floating-point differences.

Optional, to retrain or re-evaluate (needs the DFL download, done automatically, and the defect
profile from `TRANSFORMER.md`; retrained weights will not be bit-identical to the committed ones):

```powershell
.venv/Scripts/python.exe -m possession.deadball_model          # dead-ball model
.venv/Scripts/python.exe -m possession.nn                      # TCN
.venv/Scripts/python.exe -m possession.transformer --device cuda
.venv/Scripts/python.exe -m possession.evaluate --predictors DB C DBC T T_DBC --matches J03WQQ J03WR9 --labels controlled2 --post 50
```
