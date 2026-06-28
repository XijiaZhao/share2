"""AudioAction runtime: load a model file and map audio -> 13-DoF commands @ 10 fps.

A model file (e.g. audioaction_a.pt) is a single self-contained TorchScript file. This module loads it
with `torch.jit.load` (works on Windows or Linux, on CPU or CUDA) and exposes the audio->commands call;
it holds no model definition itself.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audio_frontend as af

DOF = 13
FPS = 10


def resample_commands(cmd: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a (T, DOF) command array from src_fps to dst_fps via per-DoF linear
    interpolation, preserving wall-clock duration (T / src_fps seconds).

    Upsampling smooths motion with interpolated frames; downsampling decimates to save
    bandwidth. Returns the input unchanged when rates match or there are <2 frames.
    """
    cmd = np.asarray(cmd, dtype=np.float32)
    T = cmd.shape[0]
    if dst_fps <= 0 or dst_fps == src_fps or T < 2:
        return cmd
    dur = T / src_fps
    T2 = max(1, int(round(dur * dst_fps)))
    if T2 == T:
        return cmd
    src_t = np.arange(T, dtype=np.float64) / src_fps
    dst_t = np.arange(T2, dtype=np.float64) / dst_fps
    out = np.empty((T2, cmd.shape[1]), dtype=np.float32)
    for j in range(cmd.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, cmd[:, j])
    return out


def pick_device(device=None):
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[runtime] CUDA not available — falling back to CPU", flush=True)
        return "cpu"
    return device


class Model:
    def __init__(self, model_path, device=None):
        self.device = pick_device(device)
        self.m = torch.jit.load(str(model_path), map_location=self.device).eval()

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def warmup(self, n=2):
        z = torch.zeros(1, af.N_MELS, af.CHUNK_FRAMES, device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self.m(z)
        self._sync()

    @torch.no_grad()
    def infer(self, audio: np.ndarray) -> np.ndarray:
        """float32 mono @16 kHz waveform -> (T, 13) commands in [0,1] @10 fps."""
        chunks, nvalid, T = af.wav_to_chunks(audio, self.device)
        parts, total, K = [], 0, len(chunks)
        for i, (seg, nv) in enumerate(zip(chunks, nvalid)):
            out = self.m(seg.to(self.device).float())[0].detach().cpu().numpy()   # (300,13)
            ncmd = max(1, nv // 5)
            take = ncmd if i < K - 1 else max(1, min(out.shape[0], T - total))
            parts.append(out[:take]); total += take
        cmd = np.concatenate(parts, 0)
        if len(cmd) < T:
            cmd = np.concatenate([cmd, np.repeat(cmd[-1:], T - len(cmd), 0)], 0)
        return cmd[:T].astype(np.float32)

    @torch.no_grad()
    def infer_timed(self, audio: np.ndarray):
        self._sync(); t0 = time.time()
        cmd = self.infer(audio)
        self._sync(); dt = time.time() - t0
        dur = len(cmd) / FPS
        timing = {"compute_ms": round(dt * 1000, 2), "audio_s": round(dur, 3),
                  "n_frames": int(len(cmd)), "fps": FPS,
                  "rtf": round(dt / max(dur, 1e-6), 5),
                  "speedup_x": round(max(dur, 1e-6) / max(dt, 1e-9), 1)}
        return cmd, timing

    def infer_wav(self, path) -> np.ndarray:
        return self.infer(af.load_wav(path))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model", default=str(HERE / "audioaction_a.pt"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()
    m = Model(args.model, device=args.device); m.warmup()
    cmd, t = m.infer_timed(af.load_wav(args.wav))
    print(f"{Path(args.wav).name}: {cmd.shape}  device={m.device}  compute={t['compute_ms']}ms  "
          f"audio={t['audio_s']}s  speedup={t['speedup_x']}x")
