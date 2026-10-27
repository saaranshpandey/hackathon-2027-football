"""Approach D: PathCRF (vendored in third_party/pathcrf) as the team signal P(home | alive).

PathCRF infers who has the ball from player movement alone. Its code uses top-level imports
(datatools, models, inference), so its folder is put on sys.path before importing it.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHCRF_DIR = os.path.join(ROOT, "third_party", "pathcrf")
TRIAL_DIR = os.path.join(PATHCRF_DIR, "saved", "140")
WEIGHTS = "state_dict_best_acc.pt"


def _import_pathcrf():
    if PATHCRF_DIR not in sys.path:
        sys.path.insert(0, PATHCRF_DIR)


def load_model(device="cpu"):
    """The paper's Dynamic Masked CRF (trial 140) with its best-accuracy weights, in eval mode."""
    _import_pathcrf()
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path

    args = load_trial_args(TRIAL_DIR)
    model = build_model(args, device=device)
    state = torch.load(resolve_model_path(TRIAL_DIR, WEIGHTS), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model
