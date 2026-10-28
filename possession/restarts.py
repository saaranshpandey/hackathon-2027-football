"""Restarts (throw-ins, corners, goal kicks, free kicks) as extra dead-ball evidence.

A restart can only happen after the ball was dead, so P(dead) is raised in the seconds before each one.
Barcelona: restarts from Smart Tagging (automatic, provided by the organizers; FAQ: "use it").
German matches: the official DFL events (synchronised to tracking by the PathCRF authors) stand in for tags.
Data rule: restart frame indices of our match are never printed.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNCED_DIR = os.path.join(ROOT, "third_party", "pathcrf", "data", "sportec_synced")
FPS = 25

# Smart Tagging "Type" values and the DFL (SPADL) event types that mean the same restart.
SMART_TAG_TYPES = {"throw-in", "corner kick", "goal kick", "free kick"}
DFL_TYPES = {"throw_in", "corner_crossed", "corner_short", "goalkick",
             "freekick_short", "freekick_crossed", "shot_freekick"}


def apply_restart_rule(p_dead, restarts, window, value):
    """Copy of p_dead with P(dead) >= value on the `window` frames before each restart frame."""
    out = np.array(p_dead, copy=True)
    for t in np.asarray(restarts, dtype=np.int64):
        a = max(0, t - window)
        out[a:t] = np.maximum(out[a:t], value)
    return out


def dfl_restarts(match_id, trk):
    """Our frame indices of the DFL restarts of one match (period start + timestamp within the period)."""
    from possession.tracking import runs_from_period

    events = pd.read_parquet(os.path.join(SYNCED_DIR, match_id + ".parquet"))
    events = events[events["spadl_type"].isin(DFL_TYPES)].dropna(subset=["timestamp"])
    starts = {p: a for p, (a, _) in enumerate(runs_from_period(trk.period), start=1)}
    minutes, seconds = events["timestamp"].str.split(":", expand=True).astype(float).T.values
    frames = np.round((minutes * 60 + seconds) * FPS).astype(np.int64)
    return np.array([starts[int(p)] + f for p, f in zip(events["period_id"], frames)], dtype=np.int64)


def smart_tag_restarts():
    """Video frames where Smart Tagging marks a restart of a type in SMART_TAG_TYPES (our match)."""
    from starter.load_smart_tagging import load

    return np.array([e["start_frame"] for e in load()
                     if (e["tags"].get("Type") or "").strip().lower() in SMART_TAG_TYPES], dtype=np.int64)
