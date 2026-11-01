"""Exploratory audio-only whistle candidates, never possession labels.

python -m possession.audio_review --device cuda
No tracking, predictions, or ground truth are used by this detector.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION = 'f826b80d28226b62986cc218e5cec390b1096902'
RATE = 16000


def window_starts(n, window, hop):
    if min(n, window, hop) <= 0 or hop > window:
        raise ValueError('Positive lengths and hop <= window required')
    starts = list(range(0, max(1, n - window + 1), hop))
    if n > window and starts[-1] != n - window:
        starts.append(n - window)
    return starts


def merge_candidates(rows, threshold):
    """Merge overlapping high-scoring windows; scores are uncalibrated audio scores."""
    events = []
    for row in rows:
        if row['whistle_score'] < threshold:
            continue
        a, b, score = row['start_seconds'], row['end_seconds'], row['whistle_score']
        if events and a <= events[-1]['end_seconds']:
            event = events[-1]
            event['end_seconds'] = max(b, event['end_seconds'])
            event['peak_score'] = max(score, event['peak_score'])
            event['windows'] += 1
        else:
            events.append(dict(start_seconds=a, end_seconds=b, peak_score=score, windows=1))
    return events


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def inspection_windows(rows, count=20):
    """Nonoverlapping top-ranked windows for debugging, never thresholded detections."""
    chosen = []
    for r in sorted(rows, key=lambda r: r['whistle_score'], reverse=True):
        if all(r['end_seconds'] <= c['start_seconds'] or r['start_seconds'] >= c['end_seconds'] for c in chosen):
            chosen.append(dict(start_seconds=r['start_seconds'], end_seconds=r['end_seconds'],
                               score=r['whistle_score']))
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda r: r['start_seconds'])


def run(args):
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from games import asset_paths

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    video = Path(asset_paths()['video'])
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(video)]))['streams']
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    picture = next(s for s in streams if s['codec_type'] == 'video')
    # The current tactical asset starts both streams at zero. Fail rather than
    # silently shifting candidates if another source needs timestamp handling.
    if abs(float(audio.get('start_time', 0))) > .001 or abs(float(picture.get('start_time', 0))) > .001:
        raise ValueError('Nonzero stream start timestamps require explicit alignment')
    pcm = out / 'audio-16k.f32'
    print('Decoding full audio to 16 kHz mono (local only)', flush=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-map', '0:a:0', '-vn',
                    '-ac', '1', '-ar', str(RATE), '-f', 'f32le', str(pcm)], check=True)
    waveform = np.memmap(pcm, dtype='<f4', mode='r')
    peak = 0.
    energy = 0.
    for i in range(0, len(waveform), RATE * 60):
        chunk = waveform[i:i + RATE * 60]
        peak = max(peak, float(np.abs(chunk).max()))
        energy += float(np.square(chunk, dtype=np.float64).sum())
    # One global gain, no label-driven normalization; preserve relative levels.
    gain = min(100., .95 / max(peak, 1e-8))
    mean_db = 10 * np.log10(max(energy / len(waveform), 1e-16))
    print(f'Audio RMS {mean_db:.1f} dBFS; global gain {20*np.log10(gain):.1f} dB; peak after gain {peak*gain:.3f}', flush=True)
    model_path = Path(args.model)
    extractor = ASTFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    net = ASTForAudioClassification.from_pretrained(model_path, local_files_only=True,
                                                     attn_implementation='sdpa').to(device).eval()
    ids = {v: int(k) for k, v in net.config.id2label.items()}
    if 'Whistle' not in ids:
        raise ValueError('Expected AudioSet Whistle class missing')
    window, hop = RATE * 10, RATE * 5
    starts = window_starts(len(waveform), window, hop)
    rows = []
    began = time.perf_counter()
    for i in range(0, len(starts), args.batch):
        batch_starts = starts[i:i + args.batch]
        waves = [np.asarray(waveform[s:s + window]).copy() * gain for s in batch_starts]
        inputs = extractor(waves, sampling_rate=RATE, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()
        with torch.inference_mode(), ctx:
            scores = net(**inputs).logits.float().sigmoid().cpu().numpy()
        for s, score in zip(batch_starts, scores):
            top = int(np.argmax(score))
            rows.append(dict(start_seconds=s/RATE, end_seconds=min(s+window,len(waveform))/RATE,
                             whistle_score=float(score[ids['Whistle']]),
                             human_whistling_score=float(score[ids['Whistling']]),
                             train_whistle_score=float(score[ids['Train whistle']]),
                             steam_whistle_score=float(score[ids['Steam whistle']]),
                             top_class=net.config.id2label[top], top_score=float(score[top])))
        if i == 0 or (i // args.batch) % 10 == 0 or i + args.batch >= len(starts):
            print(f'Audio windows {min(i+args.batch,len(starts))}/{len(starts)}; {time.perf_counter()-began:.1f}s', flush=True)
    events = merge_candidates(rows, args.threshold)
    pd.DataFrame(rows).to_csv(out / 'window-scores.csv', index=False)
    pd.DataFrame(events, columns=['start_seconds','end_seconds','peak_score','windows']).to_csv(out / 'whistle-candidates.csv', index=False)
    summary = dict(model=MODEL_ID, revision=REVISION, model_sha256=fingerprint(model_path/'model.safetensors'),
                   source_video=str(video), source_size=video.stat().st_size, source_mtime_ns=video.stat().st_mtime_ns,
                   sample_rate=RATE, audio_duration=len(waveform)/RATE, audio_rms_dbfs=mean_db,
                   global_gain=gain, gain_db=float(20*np.log10(gain)), window_seconds=10, hop_seconds=5,
                   threshold=args.threshold, windows=len(rows), events=events,
                   inspection_windows=inspection_windows(rows),
                   score_windows=[[r['start_seconds'],r['end_seconds'],r['whistle_score']] for r in rows],
                   warning='Whistle-like sound scores, NOT referee identity, exact whistle onset, or P(dead). No ground truth used.',
                   code_sha256=fingerprint(__file__))
    (out/'analysis.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    print('Writing amplified listening track', flush=True)
    subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar',str(RATE),'-ac','1','-i',str(pcm),
                    '-af',f'volume={gain}', '-c:a','aac','-b:a','96k',str(out/'amplified.m4a')],check=True)
    print(f'Done: {len(events)} whistle-like candidate intervals at score >= {args.threshold}. Not validated as stoppages.',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threshold',type=float,default=.2)
    p.add_argument('--out',default=str(ROOT/'out'/'audio'))
    p.add_argument('--model',default=str(ROOT/'out'/'audio'/'model'))
    a=p.parse_args()
    if a.batch < 1 or not 0 <= a.threshold <= 1:
        p.error('Positive batch and threshold in [0,1] required')
    run(a)


if __name__=='__main__':
    main()
