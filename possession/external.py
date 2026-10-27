"""DFL open data (Bassek et al. 2025, CC-BY 4.0): seven German matches with per-frame labels.

The tracking marks every frame with the ball state (alive/dead) and the team owning the ball,
which converts directly into the challenge's H/A/D labels. Files come from the Hugging Face
mirror pysport/idsse-data, because the figshare original now refuses scripted downloads.
"""

import os
import urllib.request

import numpy as np

from evaluation import possession_metrics as pm
from possession.tracking import Tracking, load, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
RAW_DIR = os.path.join(DFL_DIR, "raw")
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"

# PathCRF's split: its files sorted by name, first 5 train, next 1 val, last 1 test.
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
VAL = ["J03WQQ"]
TEST = ["J03WR9"]
ALL_MATCHES = sorted(TRAIN + VAL + TEST)

# Competition id in each match's file names (1. Bundesliga = 000001, 2. Bundesliga = 000002).
COMPETITION = {"J03WMX": "000001", "J03WN1": "000001"}


def labels_from_state(alive, owner):
    """Label codes from ball state and owning team (0 home, 1 away, -1 none)."""
    codes = np.full(len(alive), pm.D, dtype=np.int8)
    codes[alive & (owner == 0)] = pm.H
    codes[alive & (owner == 1)] = pm.A
    codes[alive & (owner < 0)] = pm.X
    return codes


def _raw_paths(match_id):
    """Download (once) and return (meta_path, positions_path) for one match."""
    stem = "DFL-COM-{}_DFL-MAT-{}.xml".format(COMPETITION.get(match_id, "000002"), match_id)
    names = ("DFL_02_01_matchinformation_" + stem, "DFL_04_03_positions_raw_observed_" + stem)
    os.makedirs(RAW_DIR, exist_ok=True)
    paths = []
    for name in names:
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(MIRROR + name, path + ".part")
            os.replace(path + ".part", path)
        paths.append(path)
    return tuple(paths)


def _build(match_id):
    from kloppy import sportec

    meta_path, raw_path = _raw_paths(match_id)
    dataset = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)
    meta = dataset.metadata
    length, width = meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width
    home, away = meta.teams
    df = dataset.to_df()

    slots = [p.player_id for p in home.players] + [p.player_id for p in away.players]
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)

    ball = np.stack([df["ball_x"].to_numpy(float) * length,
                     df["ball_y"].to_numpy(float) * width], axis=1)
    players = np.full((len(df), len(slots), 2), np.nan)
    for s, pid in enumerate(slots):
        if pid + "_x" in df:
            players[:, s, 0] = df[pid + "_x"].to_numpy(float) * length
            players[:, s, 1] = df[pid + "_y"].to_numpy(float) * width

    alive = df["ball_state"].astype(str).str.lower().str.contains("alive").to_numpy()
    owner_id = df["ball_owning_team_id"].astype(str).to_numpy()
    owner = np.where(owner_id == home.team_id, 0, np.where(owner_id == away.team_id, 1, -1))
    codes = labels_from_state(alive, owner.astype(np.int8))
    period = df["period_id"].to_numpy().astype(np.int8)

    # One separator frame between periods, so nothing can run across the break.
    keep, sep = [], []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        if keep:
            sep.append(sum(len(k) for k in keep) + len(sep))
        keep.append(idx)
    order = np.concatenate(keep)
    n = len(order) + len(sep)
    is_sep = np.zeros(n, dtype=bool)
    is_sep[sep] = True

    def place(values, fill):
        out = np.full((n,) + values.shape[1:], fill, dtype=values.dtype)
        out[~is_sep] = values[order]
        return out

    trk = Tracking(ball=place(ball.astype(np.float32), np.nan),
                   players=place(players.astype(np.float32), np.nan),
                   team=team, period=place(period, 0), pitch=(float(length), float(width)))
    return trk, place(codes, pm.X)


def load_dfl(match_id):
    """(clean Tracking, label codes) for one DFL match, cached under data/external/dfl/."""
    trk_path = os.path.join(DFL_DIR, match_id + ".npz")
    lab_path = os.path.join(DFL_DIR, match_id + "_labels.npy")
    if os.path.exists(trk_path) and os.path.exists(lab_path):
        return load(trk_path), np.load(lab_path)
    trk, codes = _build(match_id)
    save(trk, trk_path)
    np.save(lab_path, codes)
    return trk, codes
