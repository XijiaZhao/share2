# AudioAction — deployment

AudioAction maps speech audio to robot lip-servo commands, in real time:

```
INPUT   16 kHz mono audio   ────────►   OUTPUT   13-DoF servo commands, streaming @ 10 fps
```

## Contents

```
audioaction_a.pt / audioaction_b.pt   the model files (TorchScript; pick one in config.yml)
config.yml                  host, port, device, and which model to serve
meta.json                   input/output contract
serve.py                    runs the model on a port
runtime.py / audio_frontend.py   loader + audio pre-processing
mel_filters.npz             audio pre-processing data
requirements.txt
Dockerfile
testapp/                    optional browser tester (pick/upload a wav, see commands + latency)
```

## Requirements

- **OS:** Windows or Linux.
- **Python 3.9–3.12** + the packages in `requirements.txt` (or just use Docker).
- **GPU optional:** an NVIDIA GPU (CUDA) is much faster, but the model also runs on **CPU**. `device:
  auto` in `config.yml` uses the GPU if present, otherwise CPU.

The model files are single TorchScript `.pt` files — they load with plain PyTorch on any of the above.

## Run

### Windows (Git Bash or PowerShell)
```bash
python --version          # must be 3.9–3.12
# GPU (NVIDIA):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# …or CPU-only:
pip install torch==2.5.1
pip install numpy soundfile pyyaml
python serve.py --config config.yml          # -> http://0.0.0.0:8025
```

### Linux
Same as above (use the cu121 index for GPU, or `pip install torch==2.5.1` for CPU).

### Docker (Linux host)
```bash
docker build -t audioaction:1.0 .
docker run --gpus all -p 8025:8025 audioaction:1.0     # with GPU
docker run -p 8025:8025 audioaction:1.0                # CPU only (omit --gpus)
```

## Choosing the model

There are two model files. Select which one to serve in `config.yml`:

```yaml
device: auto        # auto | cuda | cpu
model: a            # or: b
models:
  a: audioaction_a.pt
  b: audioaction_b.pt
```

or override at launch: `python serve.py --model b`. Check the active model with `GET /healthz`.

## API

| method | path | body | returns |
|---|---|---|---|
| GET  | `/healthz` | — | `{status, model, fps:10, dof:13, device}` |
| GET  | `/meta` | — | input/output contract |
| POST | `/infer` | audio bytes (wav/flac/… or raw PCM, see below) | `{n_frames, fps, dof, commands:[[13]…], timing}` |
| POST | `/infer?format=npy` | audio | `(T,13)` float32 `.npy` download (timing in `X-Timing` header) |
| POST | `/infer?format=csv` | audio | `time_s` + 13 columns CSV |
| POST | `/stream` | audio | NDJSON stream, one `{i,t,cmd[13]}` per frame; `?paced=1` emits at 10 fps |

Raw PCM input (for a hardware audio pipeline) is accepted via header
`X-Audio-Format: pcm_f32_16k` (float32) or `pcm_s16_16k` (int16), 16 kHz mono.

```bash
curl -s -X POST --data-binary @clip.wav http://localhost:8025/infer | jq '.timing, .n_frames'
curl -s -X POST --data-binary @clip.wav "http://localhost:8025/infer?format=npy" -o commands.npy
curl -s -N -X POST --data-binary @clip.wav "http://localhost:8025/stream?paced=1"
```

**Output:** `commands[t]` is the 13-DoF servo vector for 10 fps frame `t` (covering audio seconds
`t/10 … (t+1)/10`), each value in `[0,1]`. Stream the rows straight to the servos at 10 fps. On a GPU,
compute is a few milliseconds per clip (far faster than real time); on CPU it's slower but still well
within real time for short clips.

## Quick check without the server

```bash
python runtime.py --wav testapp/testwav/demo.wav            # auto GPU/CPU
python runtime.py --wav testapp/testwav/demo.wav --device cpu
```

## Test app

A small browser tool to try the model — see `testapp/README.md`.
