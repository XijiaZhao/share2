"""Serve the AudioAction model on a port (configured in config.yml).

    python serve.py [--config config.yml] [--model a|b]

Endpoints
  GET  /healthz                  -> {status, model, fps, dof, device}
  GET  /meta                     -> I/O contract (meta.json)
  POST /infer                    body = audio file bytes (wav/flac/…) OR raw float32 PCM @16k
                                 (header  X-Audio-Format: pcm_f32_16k / pcm_s16_16k).
                                 ?format=json (default) -> {n_frames, fps, dof, commands, timing}
                                 ?format=npy            -> (T,13) float32 .npy download
                                 ?format=csv            -> time_s + 13 columns
  POST /stream                   same input; streams NDJSON, one {i,t,cmd[13]} per frame
                                 (?paced=1 emits at 10 fps wall-clock).
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audio_frontend as af
from runtime import Model, FPS, DOF

CFG, MODEL, META, ACTIVE = {}, None, {}, None


def load_config(path):
    cfg = {"host": "0.0.0.0", "port": 8025, "device": "cuda", "warmup": True,
           "max_audio_seconds": 300, "model": "a", "models": {"a": "audioaction_a.pt"}}
    p = Path(path)
    if p.is_file():
        cfg.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def _decode(handler):
    n = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(n) if n else b""
    fmt = handler.headers.get("X-Audio-Format", "").lower()
    if fmt == "pcm_f32_16k":
        return np.frombuffer(body, dtype=np.float32).copy()
    if fmt == "pcm_s16_16k":
        return np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    return af.read_audio_bytes(body)


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

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Audio-Format")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/healthz":
            return self._send(200, {"status": "ok", "model": ACTIVE, "fps": FPS, "dof": DOF,
                                    "device": MODEL.device})
        if path == "/meta":
            return self._send(200, {**META, "model": ACTIVE})
        if path in ("/", "/index.html"):
            return self._send(200, "POST audio to /infer (see /meta)\n", ctype="text/plain")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        try:
            audio = _decode(self)
        except Exception as e:
            return self._send(400, {"error": f"bad audio: {type(e).__name__}: {e}"})
        if len(audio) == 0:
            return self._send(400, {"error": "empty audio"})
        if len(audio) / af.SR > CFG.get("max_audio_seconds", 300):
            return self._send(413, {"error": "audio too long"})
        try:
            if path == "/infer":
                return self._infer(audio, q)
            if path == "/stream":
                return self._stream(audio, q)
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "not found"})

    def _infer(self, audio, q):
        cmd, timing = MODEL.infer_timed(audio)
        fmt = (q.get("format", ["json"])[0]).lower()
        if fmt == "npy":
            buf = io.BytesIO(); np.save(buf, cmd.astype(np.float32))
            return self._send(200, buf.getvalue(), ctype="application/octet-stream",
                              extra={"Content-Disposition": "attachment; filename=commands.npy",
                                     "X-Timing": json.dumps(timing)})
        if fmt == "csv":
            t = (np.arange(len(cmd)) / FPS)[:, None]
            hdr = "time_s," + ",".join(f"dof{j}" for j in range(DOF))
            txt = io.StringIO(); np.savetxt(txt, np.concatenate([t, cmd], 1), delimiter=",",
                                            header=hdr, comments="", fmt="%.5f")
            return self._send(200, txt.getvalue(), ctype="text/csv",
                              extra={"Content-Disposition": "attachment; filename=commands.csv",
                                     "X-Timing": json.dumps(timing)})
        return self._send(200, {"n_frames": int(len(cmd)), "fps": FPS, "dof": DOF, "model": ACTIVE,
                                "commands": np.round(cmd, 5).tolist(), "timing": timing})

    def _stream(self, audio, q):
        paced = q.get("paced", ["0"])[0] in ("1", "true", "yes")
        cmd, timing = MODEL.infer_timed(audio)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def chunk(obj):
            line = (json.dumps(obj) + "\n").encode()
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n"); self.wfile.flush()

        chunk({"event": "start", "n_frames": int(len(cmd)), "fps": FPS, "timing": timing})
        t0 = time.time()
        for i in range(len(cmd)):
            if paced:
                dt = t0 + i / FPS - time.time()
                if dt > 0:
                    time.sleep(dt)
            chunk({"i": i, "t": round(i / FPS, 3), "cmd": np.round(cmd[i], 5).tolist()})
        chunk({"event": "end"})
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()


def main():
    global CFG, MODEL, META, ACTIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yml"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default=None, help="override the model id in config (e.g. a, b)")
    args = ap.parse_args()
    CFG = load_config(args.config)
    if args.port:
        CFG["port"] = args.port
    if args.model:
        CFG["model"] = args.model

    ACTIVE = CFG["model"]
    models = CFG.get("models", {})
    if ACTIVE not in models:
        raise SystemExit(f"model '{ACTIVE}' not in config models {list(models)}")
    mp = models[ACTIVE]
    mp = mp if Path(mp).is_absolute() else HERE / mp
    mj = HERE / "meta.json"
    META = json.loads(mj.read_text(encoding="utf-8")) if mj.is_file() else {}

    print(f"loading model '{ACTIVE}': {mp}  (device={CFG['device']})", flush=True)
    MODEL = Model(str(mp), device=CFG["device"])
    if CFG.get("warmup", True):
        MODEL.warmup(); print("ready.", flush=True)

    srv = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler)
    print(f"serving model '{ACTIVE}' on http://{CFG['host']}:{CFG['port']}  "
          f"(POST /infer, /stream ; GET /healthz, /meta)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
