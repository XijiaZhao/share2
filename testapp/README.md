# AudioAction — test app

A lightweight browser tool to try the model: pick a preset wav or upload your own, see the predicted
lip motion and the model latency, and download the resulting 13-DoF commands.

## Run

```bash
# 1) start the model server first (parent folder), on the port in config.yml -> model_url
python ../serve.py --config ../config.yml

# 2) start this test app
python app_server.py --config config.yml
# open  http://localhost:8026/
```

## What it shows
- **Pick a preset wav** (from `testwav/`) or **upload your own**.
- The predicted motion: each command frame is matched to the nearest reference pose and shown as a
  lip photo, played in sync with the audio.
- The **model latency** (compute time, real-time factor, speedup) and active model.
- **Download** the full 13-DoF commands as CSV, JSON, or NPY.

## Visualization assets

The lip-photo preview uses a reference-pose pack bundled in `assets/`:
- `assets/commands.npy` — `(N,13)` reference command table
- `assets/lip_128/NNNNN_lip.jpg` — the matching lip photos

These are included. (Commands + latency + download work even without them; only the photo preview
needs them.)

## Config (`config.yml`)
- `model_url` — the running model server (default `http://localhost:8025`).
- `port` — this app's port (default 8026).
- `assets_dir`, `lip_subdir`, `commands_npy` — the pose pack for the preview.
- `testwav_dir` — preset wavs.
