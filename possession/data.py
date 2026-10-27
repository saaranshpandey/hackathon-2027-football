"""Our match's tracking (ATD) as a Tracking, parsed once and cached.

Data rule: nothing in here prints match data. Callers print aggregates only.
"""

import os

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from possession.tracking import Tracking, load, save
from starter.baseline import team_of_each_track

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")


def parse_atd(raw_path, meta_path, n_frames, pitch):
    """Raw ATD lines "<frame_id>:<x,y,s;...>:<x,y,s>" -> Tracking in metres. period is all 0."""
    team = team_of_each_track(meta_path)
    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            groups = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
            players[row] = groups.reshape(-1, 3)[:, :2] * scale
            ball[row] = np.array(parts[2].split(",")[:2], dtype=np.float64) * scale

    return Tracking(ball=ball, players=players, team=team,
                    period=np.zeros(n_frames, dtype=np.int8), pitch=(float(length), float(width)))


def load_match(game=GAME):
    """Tactical ATD of the match, with period = half number on graded frames, else 0."""
    path = os.path.join(CACHE_DIR, "atd_tactical.npz")
    if os.path.exists(path):
        return load(path)

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    trk = parse_atd(raw_path, meta_path, source_frames("tactical", game), GAMES[game]["pitch"])
    for half, first, last in half_video_blocks(game):
        trk.period[first:last + 1] = half

    os.makedirs(CACHE_DIR, exist_ok=True)
    save(trk, path)
    return trk
