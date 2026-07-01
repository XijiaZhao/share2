# AudioAction — deployment

AudioAction maps speech audio to robot lip-servo commands, in real time:

```
INPUT   16 kHz mono audio   ────────►   OUTPUT   13-DoF servo commands @ 10 fps (25 fps for `fps25`)
```

## Contents

```
audioaction_*.pt            the model files (TorchScript; pick one in config.yml):
                              a, b   English             (output [0,1])
                              zha    English + Chinese    (output [0,1])
                              xba    English + Chinese, ROUND-2 WIDE RANGE (output TRUE ~[-1.0,1.6], 10 fps)
                              fps25  English + Chinese, WIDE RANGE, NATIVE 25 fps (output TRUE ~[-0.74,1.6])
                              fps10630 English + Chinese, WIDE RANGE, NATIVE 10 fps, ref_9364 (output TRUE ~[-0.69,1.15])
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

There are six model files. Select which one to serve in `config.yml`:

```yaml
device: auto        # auto | cuda | cpu
model: zha          # a | b | zha | xba | fps25 | fps10630   (default zha)
models:
  a:        audioaction_a.pt        # English                      output [0,1]   10 fps
  b:        audioaction_b.pt        # English, held-out variant    output [0,1]   10 fps
  zha:      audioaction_zha.pt      # English+Chinese              output [0,1]   10 fps
  xba:      audioaction_xba.pt      # English+Chinese, WIDE RANGE  output TRUE ~[-1.0,1.6]  10 fps
  fps25:    audioaction_fps25.pt    # English+Chinese, WIDE RANGE, NATIVE 25 fps  output TRUE ~[-0.74,1.6]
  fps10630: audioaction_fps10630.pt # English+Chinese, WIDE RANGE, NATIVE 10 fps, ref_9364  output TRUE ~[-0.69,1.15]
```

or override at launch: `python serve.py --model fps25`. Check the active model (and its fps) with
`GET /healthz`. **Default stays `zha` (`[0,1]`, 10 fps)** so existing `[0,1]→PWM` integrations are
unaffected; switch to `xba`/`fps25` only once your PWM map covers the wide range (next section).

**Frame rate is auto-detected** — the runtime reads it from the model (10 fps for a/b/zha/xba/fps10630,
25 fps for `fps25`) and reports it in `/healthz`, `/infer`, and the stream pacing. No config knob to set; a
25 fps clip just returns 2.5× as many frames over the same duration.

## Output range — the wide-range `xba` model (read before mapping to PWM)

`xba` is the shipped round-2 model: combined English + Chinese (voice `zf_xiaobei`), trained on all
data. It deliberately **widens the motion**, so unlike `a/b/zha` it does **not** output `[0,1]`.

**Normalized vs. true range — what actually comes out of the wire.** The network core always emits a
**normalized** value in **[0,1]** per independent DoF (a sigmoid). The robot's **true** command is
recovered by a *fixed* per-DoF affine map `true = lo + norm·(hi − lo)` plus reconstruction of the 4
slaved (right-side mirror) DoF. For `a/b/zha` that map is the identity (`lo=0, hi=1`), so their output
is already the true command in **[0,1]**. For **`xba`** the `lo/hi` span a wider range, so the true
command runs roughly **[−1.0 … +1.6]**.

We **bake that de-normalization into `audioaction_xba.pt`**, so it outputs the **TRUE wide-range
command directly** — your PWM map consumes real commands, no extra step. Per-DoF motor name and true
range (these are the PWM anchors):

| DoF | motor | lo | hi | kind |
|---|---|---|---|---|
| 0  | upperlip_center_fwd | 0.33  | 0.90 | indep |
| 1  | upperlip_center_up  | 0.33  | 0.33 | const |
| 2  | lowerlip_center_fwd | 0.00  | 0.69 | indep |
| 3  | lowerlip_center_up  | 0.33  | 0.33 | const |
| 4  | upperlip_fwd_L      | 0.00  | 0.27 | indep |
| 5  | upperlip_fwd_R      | 0.00  | 0.27 | slaved = 0.272 − c4 |
| 6  | lowerlip_up_L       | 0.30  | 1.20 | indep |
| 7  | lowerlip_up_R       | −0.26 | 0.64 | slaved = 0.944 − c6 |
| 8  | corner_frontback_L  | −1.00 | 0.23 | indep |
| 9  | corner_updown_L     | −0.60 | 0.90 | indep |
| 10 | corner_frontback_R  | 0.24  | 1.47 | slaved = 0.467 − c8 |
| 11 | corner_updown_R     | 0.10  | 1.60 | slaved = 1.0 − c9 |
| 12 | jaw_fwd             | 0.00  | 1.00 | indep |

Overall envelope **[−1.0, +1.6]**. DoF **1 & 3 are constant** at 0.33; DoF **5, 7, 10, 11 are the
right-side mirrors** (slaved: `right = zone_max − left`, derived internally from 4, 6, 8, 9). Output
**shape is unchanged: `(T, 13)` at 10 fps** — only the numeric range is wider, and unlike `a/b/zha` it
**can go negative and exceed 1.0**.

> **⚠ PWM mapping differs for `xba`.** `a/b/zha` emit `[0,1]`, so an existing `[0,1]→PWM` table works
> for them. **`xba` emits the wider `[−1.0…+1.6]`** (per-DoF as above), so its PWM table must cover that
> range — the natural mapping is per-DoF `lo → min-PWM`, `hi → max-PWM` using the table.
> **Do not feed `xba` output into a `[0,1]`-only PWM map** — values would clip or wrap.

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
`t/10 … (t+1)/10`). For `a/b/zha` each value is in `[0,1]`; for the wide-range `xba` model each
value is the **TRUE** command in roughly `[−1.0…+1.6]` (per-DoF ranges and the PWM note are in the
[Output range](#output-range--the-wide-range-xba-model-read-before-mapping-to-pwm)
section above). Stream the rows straight to the servos at 10 fps. On a GPU, compute is a few
milliseconds per clip (far faster than real time); on CPU it's slower but still well within real time
for short clips.

## Quick check without the server

```bash
python runtime.py --wav testapp/testwav/demo.wav            # auto GPU/CPU
python runtime.py --wav testapp/testwav/demo.wav --device cpu
```

## Test app

A small browser tool to try the model — see `testapp/README.md`.
