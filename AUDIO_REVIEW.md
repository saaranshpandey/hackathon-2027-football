# Audio feasibility experiment

This is an audio-only diagnostic. It does not modify possession predictions, train
on Barcelona labels, or infer that a detected sound means play is stopped.

## Reproduce

Use the existing CUDA virtual environment. Install the optional dependencies:

```powershell
uv pip install --python .venv/Scripts/python.exe transformers==4.57.6
uv pip install --python .venv/Scripts/python.exe torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$env:HF_HOME = "$PWD\.uv-cache\huggingface"
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('MIT/ast-finetuned-audioset-10-10-0.4593', revision='f826b80d28226b62986cc218e5cec390b1096902', local_dir='out/audio/model', allow_patterns=['config.json','preprocessor_config.json','model.safetensors','README.md'])"
.venv/Scripts/python.exe -m possession.audio_review --device cuda
.venv/Scripts/python.exe -m possession.review
Start-Process .\out\review\index.html
```

TorchAudio must match the installed PyTorch version and CUDA wheel family. The
commands above match the project's tested torch 2.11.0+cu128 environment. FFmpeg
and FFprobe must be on PATH. Downloads retrieve public model files only; audio is
processed locally. The model is an Audio Spectrogram Transformer, not the temporal
tracking Transformer used for possession.

Sources: [model and model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593),
[AST documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/audio-spectrogram-transformer).

## Fixed analysis

- Decode the full tactical audio to mono float32 at 16 kHz, using stream zero.
- Verify audio and video start timestamps are zero. Other alignments fail explicitly.
- Apply one global peak-limited gain, capped at +40 dB. This makes the very quiet
  source louder while preserving relative levels; it amplifies noise too.
- Use the pretrained model's supplied filterbank normalization, 10-second windows,
  5-second hops, GPU batches of 8 and BF16 where supported. The final window covers
  the tail. Apply sigmoid to multilabel logits, not softmax.
- Record Whistle, Whistling, Train whistle, Steam whistle, and the top audio class.
- Fixed candidate threshold: Whistle score >= 0.20. Merge overlapping positive
  windows. Scores are uncalibrated and the threshold is an exploratory convention,
  not a validated operating point. No possession labels select this threshold.
- Separately expose the top 20 nonoverlapping windows as **ranked listening samples**,
  regardless of score. They are not detections. Onsets within these ten-second
  windows are unknown; these are unsuitable for direct frame-level boundary shifts.

The review player adds the current whistle score, purple thresholded detection
intervals, an `Audio inspection (ranked)` filter, and an optional amplified mono
listening track. The original video is muted while that track is enabled. Playback,
seeking, and speed changes synchronize the two elements. Original audio remains
available by unchecking the option.

## First full-match result

The source contains 48 kHz stereo audio. Both streams start at zero. Decoded mono
RMS was approximately -54.6 dBFS; the applied gain was +40 dB, with a resulting
peak of about 0.945 (no hard clipping in the float waveform).

Across 1,594 windows, the highest Whistle score was **0.0159**. **Zero candidate
intervals** exceeded 0.20. The most common top classes were Speech, Music, and
Cheering. These are model outputs, not verified descriptions of the recording.

This run does **not** establish a reliable whistle detector or dead-ball predictor.
It cannot distinguish lack of audible whistles from domain mismatch/model misses.
Inspect the amplified ranked clips to assess audio feasibility. Do not lower the
threshold merely to manufacture detections or tune against the public five-minute
labels. Reliable precision/recall and effects on spellF1 require synchronized audio
and labelled stoppages from other matches, with separate validation and test data.

Outputs under ignored `out/audio/`: raw decoded audio, amplified listening track,
all window scores, thresholded intervals, and `analysis.json` containing model
revision, hashes, audio settings, source identity, and ranked listening windows.
These media-derived files are local review artifacts and must not be published.
