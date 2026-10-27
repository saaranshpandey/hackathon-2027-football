"""Approach D: PathCRF (vendored in third_party/pathcrf) as the team signal P(home | alive).

PathCRF infers who has the ball from player movement alone. Its code uses top-level imports
(datatools, models, inference), so its folder is put on sys.path before importing it.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from possession.tracking import runs_from_period

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


FPS = 25


def base_table(trk, alive):
    """PathCRF's input columns for frames inside a period, before running features.

    frame_id is our frame index, so PathCRF's per-frame output maps straight back.
    Slot s of a team becomes "<team>_<k>", k counting from 1 within the team.
    """
    rows = np.flatnonzero(trk.period != 0)
    start = np.zeros(len(trk.period), dtype=np.int64)
    for a, b in runs_from_period(trk.period):
        start[a:b] = a

    columns = {
        "frame_id": rows,
        "period_id": trk.period[rows].astype(int),
        "timestamp": np.round((rows - start[rows]) / FPS, 2),
        "ball_state": np.where(alive[rows], "alive", "dead"),
        "ball_owning_team_id": np.full(len(rows), np.nan),
        "ball_x": trk.ball[rows, 0].astype(float),
        "ball_y": trk.ball[rows, 1].astype(float),
    }
    for team, name in ((0, "home"), (1, "away")):
        for k, s in enumerate(np.flatnonzero(trk.team == team), start=1):
            columns["{}_{}_x".format(name, k)] = trk.players[rows, s, 0].astype(float)
            columns["{}_{}_y".format(name, k)] = trk.players[rows, s, 1].astype(float)
    return pd.DataFrame(columns)


def to_table(trk, alive):
    """base_table plus PathCRF's own running features (vx, vy, speed, accel), episodes and phases."""
    _import_pathcrf()
    from datatools.utils import calculate_running_features, label_frames_and_episodes, label_phases

    # Label episodes and phases here: calculate_running_features would call
    # label_frames_and_episodes itself but treats its (tracking, events) tuple as a table.
    table, _ = label_frames_and_episodes(base_table(trk, alive))
    table, _ = label_phases(table)
    return calculate_running_features(table)


def team_probability(micro_pred, n_frames):
    """P(home) per frame from PathCRF's predicted possessor: home 0.95, away 0.05, else 0.5."""
    p = np.full(n_frames, 0.5)
    team = micro_pred["edge_src"].astype(str).str[:4]
    frames = micro_pred.index.to_numpy()
    p[frames[(team == "home").to_numpy()]] = 0.95
    p[frames[(team == "away").to_numpy()]] = 0.05
    return p


def p_home_pathcrf(trk, alive, device="cpu", model=None):
    """P(home | alive) per frame from PathCRF, for a Tracking with identity-continuous 11+11 players."""
    _import_pathcrf()
    from inference import inference

    with open(os.path.join(TRIAL_DIR, "args.json")) as f:
        args = json.load(f)
    model = model or load_model(device)
    _, _, micro_pred, _ = inference(model, to_table(trk, alive), use_crf=True, decode="indep",
                                    window_seconds=args["window_seconds"], correct_episode_lasts=True,
                                    evaluate=False)
    return team_probability(micro_pred, len(trk.period))


def approach_d(trk, device="cpu", model=None):
    """(P(dead), P(home | alive)) for approach D: A's dead-ball rules, PathCRF's team signal.

    PathCRF works on live-ball stretches; those come from decoding A's signals, so D needs no
    ground truth. Tracks are stitched into 11+11 identity-continuous players first.
    """
    from evaluation import possession_metrics as pm
    from possession.deadball import p_dead_rules
    from possession.decode import decode
    from possession.stitch import stitch
    from possession.team import p_home_nearest

    p_dead = p_dead_rules(trk)
    alive = (decode(p_dead, p_home_nearest(trk), runs_from_period(trk.period)) != pm.D) & (trk.period > 0)
    return p_dead, p_home_pathcrf(stitch(trk), alive, device=device, model=model)
