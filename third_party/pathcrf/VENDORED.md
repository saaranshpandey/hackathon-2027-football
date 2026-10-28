# Vendored: PathCRF

Source: https://github.com/hyunsungkim-ds/pathcrf at commit 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
(Kim et al., "PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player
Trajectories", KDD 2026, arXiv 2602.12080). Licence: MPL-2.0 (see LICENSE). Files are covered by MPL-2.0.

Copied: inference.py, dataset.py, datatools/, models/, set_transformer/, saved/140/ (args.json,
model/state_dict_best_acc.pt — the paper's Dynamic Masked CRF, LOMOCV fold 6: test J03WR9, val J03WQQ).
Not copied: data/, docs/, scripts/, applications/, notebooks, other checkpoints.

Modifications:
- inference.py `_downsample_possession`: `.to_numpy()` -> `.to_numpy(copy=True)` (twice). pandas 3 returns
  read-only arrays; the original assigns into them.

Also copied: data/sportec_synced/*.parquet — DFL open-data events synchronised to tracking by the PathCRF
authors (ELASTIC). Used only to locate German restarts (throw-ins, corners, goal kicks, free kicks).
