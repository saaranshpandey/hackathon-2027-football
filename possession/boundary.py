"""Conservative outside-pitch evidence added to T_DBC, independently of tags/audio."""
from dataclasses import dataclass

import numpy as np

from possession.tracking import runs_from_period, true_runs


@dataclass(frozen=True)
class Config:
    margin_m: float = .5
    min_frames: int = 10  # 0.4 seconds at 25 fps
    max_speed_ms: float = 45.
    max_outside_m: float = 8.
    weight: float = .5

    def __post_init__(self):
        if not (0 <= self.margin_m < self.max_outside_m and self.min_frames >= 1
                and self.max_speed_ms > 0 and 0 <= self.weight <= 1):
            raise ValueError('Invalid boundary configuration')


def signed_distance(ball, pitch):
    """Positive inside, zero on the boundary, negative outside; metres."""
    x,y=np.asarray(ball,dtype=float).T
    length,width=pitch
    return np.minimum.reduce([x,length-x,y,width-y])


def evidence(trk, cfg=None):
    """Outside confidence only on observed contiguous plausible runs within a half.

    Offline: once a run reaches min_frames, evidence covers its first frame too.
    Missing observations always break runs; no out-of-play state is carried forward.
    """
    cfg=cfg or Config()
    dist=signed_distance(trk.ball,trk.pitch)
    strength=np.zeros(len(dist),dtype=np.float32)
    visible=np.isfinite(trk.ball).all(axis=1)
    for a,b in runs_from_period(trk.period):
        good=visible[a:b].copy()
        delta=np.linalg.norm(np.diff(trk.ball[a:b].astype(float),axis=0),axis=1)*25
        jump=np.isfinite(delta)&(delta>cfg.max_speed_ms)
        # Reject both endpoints of an implausible observed displacement.
        good[:-1]&=~jump
        good[1:]&=~jump
        outside=good&(dist[a:b]<-cfg.margin_m)&(dist[a:b]>=-cfg.max_outside_m)
        for start,stop in zip(*true_runs(outside)):
            if stop-start>=cfg.min_frames:
                # The geometric margin is not a calibrated uncertainty model.
                # Stronger distances receive a larger (still bounded) boost.
                depth=-dist[a+start:a+stop]-cfg.margin_m
                strength[a+start:a+stop]=.5+.5*np.clip(depth,0,1)
    return dist,strength


def adjust(trk,p_dead,p_home,cfg=None):
    cfg=cfg or Config()
    distance,strength=evidence(trk,cfg)
    dead=np.asarray(p_dead).copy()
    dead+=cfg.weight*strength*(1-dead)
    return dead,np.asarray(p_home).copy(),dict(distance=distance,strength=strength)


def approach_t_dbc_b(trk,device='cpu'):
    from possession.hybrid import approach_t_dbc
    dead,home=approach_t_dbc(trk,device=device)
    dead,home,_=adjust(trk,dead,home)
    return dead,home
