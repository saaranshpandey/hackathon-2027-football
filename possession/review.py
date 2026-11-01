"""Build a local, full-match video review player without re-encoding the footage.

python -m possession.review
Uses frozen submission labels and public-only ground truth for display; never changes predictions.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from possession.data import load_match
from possession.tracking import true_runs

ROOT = Path(__file__).resolve().parents[1]
FPS = 25


def spans(values):
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [[int(a), int(b), str(values[a])] for a, b in zip(edges[:-1], edges[1:])]


def read_labels(path, n):
    df = pd.read_csv(path)
    if list(df.columns) != ['frame', 'label'] or len(df) != n or not np.array_equal(df.frame, np.arange(n)):
        raise ValueError(f'Invalid frame alignment in {path}')
    if not df.label.isin(['H', 'A', 'D', 'X']).all():
        raise ValueError(f'Invalid labels in {path}')
    return df.label.to_numpy()


def public_truth(path, n):
    """X in the public file means ungraded/unavailable, not a known match state."""
    return [r for r in spans(read_labels(path, n)) if r[2] != 'X']


def candidates(labels, original, missing, period):
    """Tracking/model diagnostics only, with explicit half boundaries."""
    out = []
    def add(a, b, kind, detail):
        out.append(dict(frame=int(a), end=int(b), seconds=round(a / FPS, 2),
                        duration=round((b - a) / FPS, 2), kind=kind, detail=detail))
    active = period > 0
    # Enumerate separately by half; never join intervals across a period change.
    for p in np.unique(period[active]):
        for mask, minimum, kind, detail in [
            (missing, 5 * FPS, 'Ball missing', 'Ball absent from tracking for at least 5 seconds'),
            (labels != original, FPS, 'Models disagree', 'Hybrid and DBC final labels disagree for at least 1 second'),
        ]:
            starts, ends = true_runs(mask & (period == p))
            for a, b in zip(starts, ends):
                if b - a >= minimum:
                    add(a, b, kind, detail)
    runs = spans(labels)
    for i, (a, b, label) in enumerate(runs):
        if label != 'X' and b - a <= 3 * FPS:
            add(a, b, 'Short spell', f'{label} lasts {(b - a) / FPS:.2f}s after decoding')
        if i and label in ('H', 'A') and runs[i - 1][2] == 'D':
            add(a, min(b, a + 5 * FPS), 'Restart', f'Dead ball changes to {label}')
        if i and label in ('H', 'A') and runs[i - 1][2] in ('H', 'A'):
            if missing[max(0, a - 2 * FPS):min(len(missing), a + 2 * FPS)].any():
                add(a, min(b, a + 3 * FPS), 'Turnover near gap', 'Team changes within 2 seconds of a ball-tracking gap')
    return sorted(out, key=lambda e: (e['frame'], e['kind']))


def build(output, prediction='T_DBC'):
    from games import asset_paths
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trk = load_match()
    n = len(trk.period)
    source = ROOT / 'submissions' / f'{prediction}.csv'
    comparison='DBC'
    baseline = ROOT / 'submissions' / f'{comparison}.csv'
    labels, original = read_labels(source, n), read_labels(baseline, n)
    truth = public_truth(ROOT / 'evaluation' / 'gt' / 'BAR-ATH_possession_gt_public.csv', n)
    missing = ~np.isfinite(trk.ball).all(axis=1)
    events = candidates(labels, original, missing, trk.period)
    audio_review = None
    audio_file = ROOT / 'out' / 'audio' / 'analysis.json'
    if audio_file.exists():
        audio_review = json.loads(audio_file.read_text(encoding='utf-8'))
        current_video = Path(asset_paths()['video'])
        if (audio_review['source_size'] != current_video.stat().st_size or
                audio_review['source_mtime_ns'] != current_video.stat().st_mtime_ns):
            raise ValueError('Audio analysis is stale for this video; rerun possession.audio_review')
        for event in audio_review['events']:
            a = round(event['start_seconds'] * FPS)
            b = min(n, round(event['end_seconds'] * FPS))
            events.append(dict(frame=a, end=b, seconds=event['start_seconds'], duration=(b-a)/FPS,
                               kind='Whistle-like audio', detail=f"Audio score {event['peak_score']:.2f}; coarse 10s windows, not confirmed referee whistle"))
        for window in audio_review.get('inspection_windows', []):
            a, b = round(window['start_seconds']*FPS), min(n, round(window['end_seconds']*FPS))
            events.append(dict(frame=a, end=b, seconds=window['start_seconds'], duration=(b-a)/FPS,
                               kind='Audio inspection (ranked)',
                               detail=f"Whistle score {window['score']:.4f}; ranked listening sample, NOT a detection"))
    events.sort(key=lambda e: (e['frame'], e['kind']))
    # Centimetre precision, 5 fps map; labels and visibility remain frame-exact at 25 fps.
    # Sentinel -32768 denotes missing observations; never interpolate through gaps.
    coordinates = np.concatenate([trk.players[::5], trk.ball[::5, None]], axis=1)
    valid = np.isfinite(coordinates).all(axis=2)
    packed = np.full(coordinates.shape, -32768, dtype='<i2')
    packed[valid] = np.clip(np.rint(coordinates[valid] * 100), -32767, 32767).astype('<i2')
    import os
    video = os.path.relpath(asset_paths()['video'], output).replace('\\', '/')
    if audio_review is not None:
        audio_review['amplified'] = os.path.relpath(ROOT/'out'/'audio'/'amplified.m4a',output).replace('\\','/')
    data = dict(fps=FPS, frames=n, step=5, pitch=list(trk.pitch), teams=trk.team.tolist(),
                video=video, labels=spans(labels), baseline=spans(original), groundTruth=truth,
                missing=[[int(a), int(b)] for a, b in zip(*true_runs(missing))],
                periods=spans(trk.period.astype(str)), events=events, audioReview=audio_review,
                coordinates=base64.b64encode(packed.tobytes()).decode('ascii'),
                model=prediction,
                comparisonModel=comparison,
                labelSha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output / 'review-data.js').write_text('window.REVIEW_DATA=' + json.dumps(data, separators=(',', ':')) + ';', encoding='utf-8')
    template=Path(__file__).with_name('review_player.html').read_text(encoding='utf-8')
    template=template.replace('MODEL PREDICTION · T_DBC','MODEL PREDICTION · '+prediction)
    template=template.replace('review · T_DBC hybrid','review · '+prediction+' hybrid')
    (output/'index.html').write_text(template,encoding='utf-8')
    pd.DataFrame(events).to_csv(output / 'review-candidates.csv', index=False)
    (output / 'README.txt').write_text(
        'Open index.html in Edge or Chrome. Keep this folder in the repository: it links to the original tactical video.\n'
        f'Labels: frozen submissions/{prediction}.csv (25 fps). Map: tracking sampled at 5 fps, held between samples.\n'
        'Blue = Barcelona; coral = Athletic; yellow = observed ball. Tracking slots are not player identities.\n'
        'Space = play/pause; arrows = +/-5s; comma/period = one-frame steps while paused.\n'
        'Click a review candidate to seek five seconds before it. Public five-minute ground truth is shown only where available.\n'
        'X in the public ground-truth file is treated as unavailable; private ground truth is never read.\n'
        'Flags are possible edge cases, not verified errors. Use for inspection, not manual corrections to match labels.\n'
        'This player displays the full original video; it is not a separately encoded MP4.\n', encoding='utf-8')
    print(f'Created {output / "index.html"}: {n:,} frames, {len(events)} review candidates', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'out' / 'review'))
    p.add_argument('--prediction',choices=['T_DBC','DBC'],default='T_DBC')
    args=p.parse_args()
    build(args.out,args.prediction)


if __name__ == '__main__':
    main()
