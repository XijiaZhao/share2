"""Test app for AudioAction (talks to the running model port and visualizes the result).

Pick a preset wav or upload your own -> the audio is sent to the model port -> the returned 13-DoF
commands are matched to the nearest reference pose and shown as a lip photo per frame, with the model
latency. No model is contained here; this is a pure client + visualizer.

  python app_server.py [--config config.yml]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CFG = {}
POSES = None            # (N,13) reference pose table for nearest-neighbour visualization


def load_config(path):
    cfg = {"host": "0.0.0.0", "port": 8026, "model_url": "http://localhost:8025",
           "assets_dir": "assets", "lip_subdir": "lip_128", "commands_npy": "commands.npy",
           "testwav_dir": "testwav"}
    p = Path(path)
    if p.is_file():
        cfg.update(yaml.safe_load(p.read_text()) or {})
    return cfg


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else (HERE / p)


def poses():
    global POSES
    if POSES is None:
        POSES = np.load(_abs(CFG["assets_dir"]) / CFG["commands_npy"]).astype(np.float32)
    return POSES


def nearest_frames(cmds):
    nd = poses()
    d = ((cmds[:, None, :] - nd[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    return (nn + 1).astype(int).tolist(), int(len(np.unique(nn)))


def call_model(wav_bytes):
    url = CFG["model_url"].rstrip("/") + "/infer"
    req = urllib.request.Request(url, data=wav_bytes, method="POST",
                                 headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    if "error" in res:
        raise RuntimeError(res["error"])
    return np.asarray(res["commands"], np.float32), res.get("timing", {}), res.get("model")


def find_wav(name):
    p = _abs(CFG["testwav_dir"]) / Path(name).name
    return p if p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), ctype="text/html")
        if path == "/api/presets":
            wavs = []
            d = _abs(CFG["testwav_dir"])
            for p in sorted(d.glob("*.wav")) if d.is_dir() else []:
                txt = p.with_suffix(".txt")
                wavs.append({"name": p.name,
                             "label": txt.read_text().strip()[:60] if txt.is_file() else p.stem})
            return self._send(200, {"wavs": wavs, "model_url": CFG["model_url"]})
        if path.startswith("/lip/"):
            img = _abs(CFG["assets_dir"]) / CFG["lip_subdir"] / f"{int(path[5:]):05d}_lip.jpg"
            if not img.is_file():
                return self._send(404, {"error": "no image"})
            return self._send(200, img.read_bytes(), ctype="image/jpeg",
                              extra={"Cache-Control": "max-age=86400"})
        if path.startswith("/twav/"):
            p = find_wav(path[len("/twav/"):])
            if not p:
                return self._send(404, {"error": "wav not found"})
            return self._send(200, p.read_bytes(), ctype="audio/wav")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/infer":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        wav = self.rfile.read(n) if n else b""
        if not wav:
            return self._send(400, {"error": "empty audio"})
        try:
            t0 = time.time()
            cmd, timing, model_id = call_model(wav)
            t_model = time.time() - t0
            t1 = time.time()
            frames, n_unique = nearest_frames(cmd)
            t_nn = time.time() - t1
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(200, {
            "n_frames": int(len(cmd)), "fps": 10, "dof": 13, "model": model_id,
            "commands": np.round(cmd, 5).tolist(), "frames": frames, "n_unique_poses": n_unique,
            "model_timing": timing, "roundtrip_ms": round(t_model * 1000, 1),
            "render_ms": round(t_nn * 1000, 1)})


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yml"))
    args = ap.parse_args()
    CFG = load_config(args.config)
    poses()
    srv = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler)
    print(f"test app -> http://localhost:{CFG['port']}/   (model at {CFG['model_url']})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
