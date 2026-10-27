"""Give clean tracking (DFL) the defects of our match's tracking, so validation looks like the test.

measure() reduces a Tracking to aggregate defect statistics -- the only thing taken from our
match. scramble() applies those statistics to a clean Tracking: anonymous fragmented tracks,
hidden players, identity swaps and a missing ball. Labels are never touched.
"""

import json
from dataclasses import dataclass

import numpy as np

from possession.tracking import Tracking, runs_from_period

CAP = 2500       # run lengths are histogrammed up to 100 s
JUMP_M = 3.0     # a slot moving more than this in one frame has switched to someone else
CROWD_M = 5.0    # players this close to the ball count as crowding it


@dataclass
class DefectStats:
    ball_missing_frac: float
    ball_gap_hist: np.ndarray   # bincount of ball-missing run lengths (frames)
    slot_life_hist: np.ndarray  # bincount of slot visible-run lengths (frames)
    mean_visible: float
    jump_rate: float            # jumps > JUMP_M per pair of consecutive visible slot-frames
    slots_per_team: tuple

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"ball_missing_frac": self.ball_missing_frac,
                       "ball_gap_hist": self.ball_gap_hist.tolist(),
                       "slot_life_hist": self.slot_life_hist.tolist(),
                       "mean_visible": self.mean_visible, "jump_rate": self.jump_rate,
                       "slots_per_team": list(self.slots_per_team)}, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["ball_missing_frac"], np.array(d["ball_gap_hist"]),
                   np.array(d["slot_life_hist"]), d["mean_visible"], d["jump_rate"],
                   tuple(d["slots_per_team"]))


def _true_runs(mask):
    """(starts, stops) of True runs in a 1-D bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _hist(lengths):
    lengths = np.minimum(np.asarray(lengths, dtype=np.int64), CAP)
    return np.bincount(lengths, minlength=1)


def measure(trk):
    """Aggregate defect statistics over frames inside a period."""
    gaps, lives = [], []
    missing = frames = visible = jumps = pairs = 0
    for a, b in runs_from_period(trk.period):
        ball_missing = np.isnan(trk.ball[a:b, 0])
        missing += int(ball_missing.sum())
        starts, stops = _true_runs(ball_missing)
        gaps.extend((stops - starts).tolist())

        vis = ~np.isnan(trk.players[a:b, :, 0])
        visible += int(vis.sum())
        frames += b - a
        for s in range(vis.shape[1]):
            starts, stops = _true_runs(vis[:, s])
            lives.extend((stops - starts).tolist())

        both = vis[1:] & vis[:-1]
        step = np.linalg.norm(np.diff(trk.players[a:b], axis=0), axis=2)
        jumps += int((step[both] > JUMP_M).sum())
        pairs += int(both.sum())

    return DefectStats(ball_missing_frac=missing / frames, ball_gap_hist=_hist(gaps),
                       slot_life_hist=_hist(lives), mean_visible=visible / frames,
                       jump_rate=jumps / pairs if pairs else 0.0,
                       slots_per_team=tuple(int(c) for c in np.bincount(trk.team, minlength=2)))


def _percentile(hist, q):
    if hist.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(hist), q / 100.0 * hist.sum()))


def summary(stats):
    """The aggregates compared between our match and a scrambled DFL match."""
    return {"ball_missing": round(stats.ball_missing_frac, 3),
            "mean_visible": round(stats.mean_visible, 2),
            "slot_life_med": _percentile(stats.slot_life_hist, 50),
            "slot_life_p90": _percentile(stats.slot_life_hist, 90),
            "ball_gap_med": _percentile(stats.ball_gap_hist, 50),
            "ball_gap_p90": _percentile(stats.ball_gap_hist, 90),
            "jump_rate": round(stats.jump_rate, 5)}


def _fragments(clean, stats, rng):
    """[(player, start, stop)]: each visible span cut into shown runs separated by hidden runs."""
    life_p = stats.slot_life_hist / stats.slot_life_hist.sum()
    mean_life = float((np.arange(len(life_p)) * life_p).sum())
    hidden = max(0.0, 1.0 - stats.mean_visible / measure(clean).mean_visible)
    mean_gap = hidden * mean_life / (1.0 - hidden)

    frags = []
    for p in range(clean.players.shape[1]):
        for a, b in runs_from_period(clean.period):
            starts, stops = _true_runs(~np.isnan(clean.players[a:b, p, 0]))
            for t, end in zip(starts + a, stops + a):
                while t < end:
                    stop = min(t + max(1, int(rng.choice(len(life_p), p=life_p))), end)
                    frags.append((p, int(t), int(stop)))
                    t = stop + (int(rng.geometric(1.0 / max(mean_gap, 1.0))) if mean_gap > 0 else 0)
    return frags


def _swap(clean, frags, rate, u, pick):
    """Copy of clean positions with identity swaps at frames where u < rate."""
    pos = clean.players.copy()
    n_frames, n_players = pos.shape[:2]
    shown = np.zeros((n_frames, n_players), dtype=bool)
    frag_stop = np.zeros((n_frames, n_players), dtype=np.int64)
    for p, a, b in frags:
        shown[a:b, p] = True
        frag_stop[a:b, p] = b

    for t in np.flatnonzero((u < rate) & (clean.period > 0)):
        cand = np.flatnonzero(shown[t])
        if len(cand) < 2:
            continue
        p = cand[int(pick[t] * len(cand))]
        mates = cand[(clean.team[cand] == clean.team[p]) & (cand != p)]
        if not len(mates):
            continue
        q = mates[np.argmin(np.linalg.norm(pos[t, mates] - pos[t, p], axis=1))]
        e = min(frag_stop[t, p], frag_stop[t, q])
        pos[t:e, [p, q]] = pos[t:e, [q, p]]
    return pos


def _assign_slots(clean, frags, pos, slots_per_team):
    """Fragments -> anonymous slots: lowest free slot of the team, never continuing a slot directly."""
    team_out = np.array([0] * slots_per_team[0] + [1] * slots_per_team[1], dtype=np.int8)
    out = np.full((pos.shape[0], len(team_out), 2), np.nan, dtype=np.float32)
    free_at = np.full(len(team_out), -1, dtype=np.int64)
    for p, a, b in sorted(frags, key=lambda f: (f[1], f[0])):
        cand = np.flatnonzero((team_out == clean.team[p]) & (free_at < a))
        if not len(cand):
            continue
        out[a:b, cand[0]] = pos[a:b, p]
        free_at[cand[0]] = b
    return team_out, out


def _remove_ball(clean, stats, rng):
    ball = clean.ball.copy()
    in_period = clean.period > 0
    target = stats.ball_missing_frac * in_period.sum()
    run_end = np.zeros(len(ball), dtype=np.int64)
    for a, b in runs_from_period(clean.period):
        run_end[a:b] = b

    with np.errstate(invalid="ignore"):
        crowd = (np.linalg.norm(clean.players - clean.ball[:, None, :], axis=2) < CROWD_M).sum(1)
    gap_p = stats.ball_gap_hist / stats.ball_gap_hist.sum()

    missing = int((np.isnan(ball[:, 0]) & in_period).sum())
    while missing < target:
        weights = np.where(~np.isnan(ball[:, 0]) & in_period, 1.0 + crowd, 0.0)
        if weights.sum() == 0:
            break
        starts = rng.choice(len(ball), size=50, p=weights / weights.sum())
        lengths = rng.choice(len(gap_p), size=50, p=gap_p)
        for t, length in zip(starts, lengths):
            if missing >= target:
                break
            stop = min(t + max(1, int(length)), run_end[t], t + int(np.ceil(target - missing)))
            missing += int((~np.isnan(ball[t:stop, 0])).sum())
            ball[t:stop] = np.nan
    return ball


def _calibrate_rate(jump_rate_of, target, low=1e-7, high=0.05, steps=24):
    """Swap rate whose jump rate is within 10% of target, by bisection in log space.

    Log space because our match's jump rate is tiny (~1e-5): linear halving from 0.05 cannot
    reach the ~1e-4 swaps per frame that needs in a handful of steps.
    """
    rate = high
    for _ in range(steps):
        rate = float(np.sqrt(low * high))
        got = jump_rate_of(rate)
        if abs(got - target) <= 0.1 * target:
            break
        low, high = (rate, high) if got < target else (low, rate)
    return rate


def scramble(clean, stats, seed):
    """Clean Tracking -> Tracking with our match's defects. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    frags = _fragments(clean, stats, rng)
    u, pick = rng.random(len(clean.period)), rng.random(len(clean.period))

    def build(rate):
        pos = _swap(clean, frags, rate, u, pick)
        return _assign_slots(clean, frags, pos, stats.slots_per_team)

    def jump_rate_of(rate):
        team, players = build(rate)
        return measure(Tracking(clean.ball, players, team, clean.period, clean.pitch)).jump_rate

    rate = _calibrate_rate(jump_rate_of, stats.jump_rate) if stats.jump_rate > 0 else 0.0
    team, players = build(rate)

    return Tracking(ball=_remove_ball(clean, stats, rng), players=players, team=team,
                    period=clean.period.copy(), pitch=clean.pitch)


def load_scrambled(match_id):
    """(scrambled Tracking, label codes) for one DFL match, cached next to the clean one."""
    import os

    from possession.data import CACHE_DIR
    from possession.external import ALL_MATCHES, DFL_DIR, load_dfl
    from possession.tracking import load, save

    path = os.path.join(DFL_DIR, match_id + "_scrambled.npz")
    clean, codes = load_dfl(match_id)
    if os.path.exists(path):
        return load(path), codes
    stats = DefectStats.from_json(os.path.join(CACHE_DIR, "defects.json"))
    trk = scramble(clean, stats, seed=2026 + ALL_MATCHES.index(match_id))
    save(trk, path)
    return trk, codes
