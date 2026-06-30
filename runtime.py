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
FPS = 10            # legacy default; the real rate is auto-detected per model (see Model.fps)


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
        # auto-detect the model's frame rate from its output width on a full chunk:
        # a 30 s mel chunk -> 1500 features -> 1500/pool command frames; pool 5 -> 300 @10fps,
        # pool 2 -> 750 @25fps. Keeps the runtime model-agnostic (no fps in config).
        self.pool, self.fps = 5, FPS
        self._detect()

    def _detect(self):
        with torch.no_grad():
            z = torch.zeros(1, af.N_MELS, af.CHUNK_FRAMES, device=self.device)
            nout = int(self.m(z).shape[1])
        self._sync()
        self.pool = max(1, round(1500 / nout))           # features-per-command-frame
        self.fps = max(1, round(nout / 30.0))            # 30 s chunk -> nout frames

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
        """float32 mono @16 kHz waveform -> (T, 13) commands at the model's native fps.
        [0,1] for a/b/zha; TRUE wide-range for xba/fps25."""
        chunks, nvalid, T = af.wav_to_chunks(audio, self.device, self.fps)
        parts, total, K = [], 0, len(chunks)
        for i, (seg, nv) in enumerate(zip(chunks, nvalid)):
            out = self.m(seg.to(self.device).float())[0].detach().cpu().numpy()   # (nout,13)
            ncmd = max(1, nv // self.pool)
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
        dur = len(cmd) / self.fps
        timing = {"compute_ms": round(dt * 1000, 2), "audio_s": round(dur, 3),
                  "n_frames": int(len(cmd)), "fps": self.fps,
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
