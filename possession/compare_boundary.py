"""Fixed boundary-evidence experiment on VAL, including isolated add-on noise tests."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from possession.boundary import adjust
from possession.controlled import controlled_labels
from possession.decode import decode
from possession.evaluate import answer_key, score, signals
from possession.external import VAL, TEST
from possession.scramble import load_scrambled
from possession.tracking import runs_from_period


def perturb(trk,seed,sigma=0.,bias=0.,jump_fraction=0.):
    rng=np.random.default_rng(seed)
    ball=trk.ball.copy()
    visible=np.isfinite(ball).all(axis=1)
    ball[visible]+=rng.normal(0,sigma,(visible.sum(),2))
    # One fixed calibration offset per match, rather than framewise jitter.
    if bias:
        angle=rng.uniform(0,2*np.pi)
        ball[visible]+=bias*np.array([np.cos(angle),np.sin(angle)])
    jumps=visible & (rng.random(len(ball))<jump_fraction)
    ball[jumps]+=rng.uniform(-20,20,(jumps.sum(),2))
    return replace(trk,ball=ball)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',nargs='+',choices=VAL+TEST,default=VAL)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--out',default='out/boundary-validation.csv')
    args=parser.parse_args()
    if args.seeds<1:parser.error('seeds must be positive')
    scenarios={'observed tracking':{}, 'jitter 0.15m':dict(sigma=.15),
               'jitter 0.50m':dict(sigma=.5), 'calibration bias 0.75m':dict(bias=.75),
               'jumps 0.2pct':dict(jump_fraction=.002)}
    rows=[]
    for match in args.matches:
        trk,gt=load_scrambled(match)
        gt=answer_key(trk,gt,'controlled2')
        runs=runs_from_period(trk.period)
        dead,home=signals('T_DBC',match,trk)
        def evaluate(d,h):return score(gt,controlled_labels(decode(d,h,runs),runs,50))
        rows.append(dict(match=match,scenario='baseline T_DBC',seed=-1,evidence_frames=0,
                         evidence_dead_fraction=np.nan,changed_frames=0,**evaluate(dead,home)))
        for name,settings in scenarios.items():
            for seed in range(args.seeds if settings else 1):
                variant=perturb(trk,seed,**settings) if settings else trk
                d,h,diag=adjust(variant,dead,home)
                support=diag['strength']>0
                graded=support & (gt!=pm.X)
                rows.append(dict(match=match,scenario=name,seed=seed,evidence_frames=int(support.sum()),
                                 evidence_dead_fraction=float(np.mean(gt[graded]==pm.D)) if graded.any() else np.nan,
                                 changed_frames=int(np.sum(d!=dead)),**evaluate(d,h)))
    df=pd.DataFrame(rows)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False)
    print('VAL default; fixed rule; controlled2 labels, post=50, decoder costs=8/4.')
    print('Noise affects ONLY boundary evidence; baseline model probabilities stay fixed. Not an end-to-end robustness test.')
    print('| match | scenario | spellF1 mean [min,max] | macroF1 | dead F1 | possessions | evidence frames | evidence on GT dead |')
    print('|---|---|---|---|---|---|---|---|')
    for (m,s),g in df.groupby(['match','scenario'],sort=False):
        fraction=g.evidence_dead_fraction.mean()
        fraction=f'{fraction:.1%}' if np.isfinite(fraction) else 'n/a'
        print(f'| {m} | {s} | {g.spell_f1.mean():.4f} [{g.spell_f1.min():.4f}, {g.spell_f1.max():.4f}] | {g.macro_f1.mean():.4f} | {g.dead_f1.mean():.4f} | {g.n_pred.mean():.1f}/{g.n_gt.iloc[0]} | {g.evidence_frames.mean():.0f} | {fraction} |')
    print(f'Saved per-seed results to {path}')


if __name__=='__main__':main()
