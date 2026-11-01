# Outside-pitch evidence experiment

`T_DBC_B` adds geometric dead-ball evidence to the original T_DBC hybrid. It does
not use the restart rule, Smart Tagging, audio, or Barcelona ground truth.

```powershell
cd G:\hackathon-2027-football

# Fixed-rule validation and isolated coordinate-noise stress tests (five seeds).
.\.venv\Scripts\python.exe -m possession.compare_boundary

# Standard side-by-side scorer, on VAL only.
.\.venv\Scripts\python.exe -m possession.evaluate --predictors T_DBC T_DBC_B --matches J03WQQ --labels controlled2 --post 50

# Optional: generate a separate Barcelona prediction file.
.\.venv\Scripts\python.exe -m possession.run --approach T_DBC_B --device cuda --post 50 --name T_DBC_B
```

No retraining is required. The implementation is `possession/boundary.py`.
Defaults are exploratory fixed settings, not tuned to the public Barcelona window.

## Evidence and safeguards

- Calculate signed distance to the rectangular pitch boundary in metres: positive
  inside, zero on the line, negative outside.
- Require a ball centre more than 0.5 m outside for at least 10 consecutive observed
  frames (0.4 seconds at 25 fps). Missing observations break the run.
- Reject both endpoints of consecutive observed displacements exceeding 45 m/s,
  and positions more than 8 m outside. These conservative heuristics can miss real
  events; they do not prove that every remaining track is correct.
- Keep evidence separate within each half; never fill through a missing ball or
  halftime. Evidence disappears when the reliable outside run ends. Being inside
  does not force a live-play prediction: many stoppages occur inside the pitch.
- Raise P(dead) using `p + 0.5 * strength * (1-p)`. Strength rises from 0.5 to 1
  over the first metre beyond the margin. It is a heuristic, not a calibrated
  probability. The possession head and existing decoder/postprocessing are unchanged.
- This is offline evidence: when a run meets the minimum duration, its first frame
  also receives evidence. A live implementation would need a confirmation delay.

The 0.5 m margin exceeds the approximate ball radius but does not remove tracking
or pitch-calibration uncertainty. The method uses 2D positions and cannot verify
physical whole-ball crossing exactly. Persistent mistracking may still pass the
speed check. High-speed kicks can be rejected. The existing neural model already
receives `ball_line_dist`, so this experiment tests an explicit prior on information
already present in the model's input, not a new independent source.

## Validation results

J03WQQ only; controlled2 answer key; post=50; decoder switch/dead costs 8/4.

| Scenario | spellF1 mean [min,max] | macroF1 | Dead-ball F1 |
|---|---:|---:|---:|
| T_DBC baseline | 0.7466 | 0.9193 | 0.9714 |
| Boundary evidence, observed tracking | 0.7466 | 0.9195 | 0.9715 |
| Added 0.15 m coordinate jitter | 0.7466 [0.7466,0.7466] | 0.9193 | 0.9714 |
| Added 0.50 m coordinate jitter | 0.7466 [0.7466,0.7466] | 0.9193 | 0.9714 |
| Fixed 0.75 m calibration offset | 0.7462 [0.7459,0.7466] | 0.9194 | 0.9714 |
| Sporadic jumps on 0.2% of observations | 0.7466 [0.7466,0.7466] | 0.9195 | 0.9715 |

Baseline and the observed-tracking variant both predict 285 possessions, versus
284 actual. The rule activates on 3,591 frames; 97.0% of activated, graded frames
have dead-ball ground truth. This is an evidence precision diagnostic, not overall
dead-ball accuracy or stoppage recall. No improved spellF1 has been demonstrated.

Noise tests perturb only the extra boundary signal while keeping T_DBC's original
probabilities fixed, isolating the added rule's sensitivity. They are not full
pipeline robustness tests. Jitter is independent Gaussian noise per coordinate;
calibration offset is one random direction per match; jumps add random offsets
up to 20 m per coordinate. Missing observations remain missing. Five seeds are
saved separately in `out/boundary-validation.csv`. No test match or Barcelona
public-window score is used by the default command.
