#!/usr/bin/env python3
"""Production SP decision server (MLP distilled from Tev1-4B).

Drop-in replacement for Ollama's `/v1/systemone` endpoint: it accepts the same
request shape and returns the same response shape, so the stock-twin-kr app can
point `sp.symbolModel` at this server with no client changes.

Request  (POST /v1/systemone):
  {"model": "...", "state": {...}, "questions": {"sp": {"type": "choice",
     "instructions": "...", "criteria": {...}, "choices": [...]}}}

Response:
  {"model": "...", "answers": {"sp": {"type": "choice", "choice": "...",
     "probabilities": {...}, "confidence": ...}}, "usage": {...}}

Request  (POST /v1/features):
  {"candles": [{"high": 1, "low": 2, "close": 3}, ...],
   "pivot_type": 0, "analysis_type": "SI"}

  pivot_type:    0=Neely (default, training basis), 1=Williams Fractal N=2,
                 2=Williams N=2 + Alligator — same choices as the GUI.
  analysis_type: SI/DI/DE ("Analysis mode" in the GUI), default SI.
                 `mode` is accepted as a backward-compatible alias.

Response:
  {"prices": [...], "waves": [{"start", "end", "sp", "startPrice",
   "endPrice", "state": {...38 V4 keys...}}], "count": n}

The feature endpoint runs the exact GUI pipeline (Neely with the selected
pivot_type, MacroWave.monoize/analyze, wave chain w_adj_ws, Decimal
ROUND_HALF_UP) so clients never need to reimplement the wave engine in
TypeScript. Only pivot_type=0 is the basis the shipped V4 model was trained
on; 1/2 are provided for GUI-equivalent feature extraction.

The model is a small CPU MLP (no GPU, no Ollama, no llama.cpp). It runs the
standardised monowave feature vector through the network and returns the SP
symbol with per-choice probabilities.

Usage:
  python mlp/serve_mlp.py --model mlp/sp_mlp.pt --host 127.0.0.1 --port 8095
"""
import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn

from v4_features import build_features_response


class MLP(nn.Module):
    # Must match train_mlp.py exactly (Dropout is a no-op in eval mode).
    def __init__(self, d, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, n_out),
        )

    def forward(self, x):
        return self.net(x)


# V4 feature layout — 38-key state, 37 consumed keys (excluding `beyond`).
# Matches generate_v4_dataset.py / train_v4_fast.py / mlp/v4_features.py.
V4_DIRF = ["m0_dir"]
V4_NUMF = [
    "m0_change_pct",
    "rc_la_lc", "rc_lb_lc",
    "m1_retrace_ratio", "m2_retrace_ratio", "m_1_retrace_ratio",
    "m_2_retrace_ratio", "m3_retrace_ratio", "m4_retrace_ratio",
    "m1_time_ratio", "m2_time_ratio", "m_1_time_ratio",
    "m_2_time_ratio", "m3_time_ratio", "m4_time_ratio",
    "m0_time_bars", "m1_time_bars", "m2_time_bars", "m_1_time_bars",
    "m_2_time_bars", "m3_time_bars", "m4_time_bars", "m_3_time_bars",
    "m_4_time_bars",
]
V4_BOOLF = [
    "m_2_vs_m0_overlap", "m_1_vs_m1_overlap", "m0_vs_m2_overlap",
    "m1_vs_m3_overlap", "m2_vs_m4_overlap",
    "m_2_vs_m1_overlap", "m_1_vs_m2_overlap", "m0_vs_m3_overlap",
    "m1_vs_m4_overlap",
]
V4_INTF = ["m_1_mw_count", "m2_mw_count", "m3_mw_count"]


class SPMLP:
    def __init__(self, path):
        art = torch.load(path, map_location="cpu", weights_only=False)
        self.keys = art["keys"]
        self.mu = np.asarray(art["mu"], dtype=np.float32)
        self.sd = np.asarray(art["sd"], dtype=np.float32)
        # Use artifact-provided feature lists if present, else V4 layout
        self.dirf = art.get("dirf", V4_DIRF)
        self.numf = art.get("numf", V4_NUMF)
        self.boolf = art.get("boolf", V4_BOOLF)
        self.intf = art.get("intf", V4_INTF)
        self.model = MLP(len(self.mu), len(self.keys))
        self.model.load_state_dict(art["state"])
        self.model.eval()
        self._x = None  # preallocated batch buffer

    def feature_vector(self, state):
        v = []
        for f in self.dirf:
            v.append(1.0 if state.get(f) == "up" else (-1.0 if state.get(f) == "down" else 0.0))
        for f in self.numf:
            x = state.get(f)
            v.append(0.0 if x is None else float(x))
            v.append(0.0 if x is not None else 1.0)
        for f in self.boolf:
            v.append(1.0 if state.get(f) else 0.0)
        for f in self.intf:
            x = state.get(f)
            v.append(0.0 if x is None else float(x))
        return v

    def predict(self, states):
        """states: list of state dicts. Returns list of (choice, probs, confidence)."""
        n = len(states)
        X = np.asarray([self.feature_vector(s) for s in states], dtype=np.float32)
        Xn = (X - self.mu) / self.sd
        with torch.no_grad():
            logits = self.model(torch.tensor(Xn)).numpy()
        # softmax over the 10 SP keys
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        out = []
        for i in range(n):
            p = probs[i]
            best = int(p.argmax())
            out.append((self.keys[best], {self.keys[j]: float(p[j]) for j in range(len(self.keys))}, float(p[best])))
        return out


def softmax_top(logits):
    pass


class Handler(BaseHTTPRequestHandler):
    sp = None
    model_name = "sp-mlp"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": self.model_name}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/v1/features"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode())
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        if self.path == "/v1/features":
            candles = req.get("candles")
            if not isinstance(candles, list):
                self._send(400, {"error": "candles must be a list of OHLC bars"})
                return
            try:
                mode = req.get("analysis_type") or req.get("mode") or "SI"
                payload = build_features_response(
                    candles,
                    pivot_type=int(req.get("pivot_type", 0)),
                    mode=str(mode).upper(),
                )
            except Exception as e:
                print(f"[features] failed: {type(e).__name__}: {e}", flush=True)
                self._send(400, {"error": f"feature build failed: {e}"})
                return
            self._send(200, payload)
            return

        state = req.get("state", {})
        questions = req.get("questions", {})
        t0 = time.perf_counter()
        results = self.sp.predict([state])
        dt = time.perf_counter() - t0
        choice, probs, confidence = results[0]

        answers = {}
        for qname, q in questions.items():
            answers[qname] = {
                "type": q.get("type", "choice"),
                "choice": choice,
                "probabilities": probs,
                "confidence": confidence,
            }
        self._send(200, {
            "model": req.get("model", self.model_name),
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0, "inference_ms": round(dt * 1000, 3)},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp/sp_mlp.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--name", default="sp-mlp")
    args = ap.parse_args()

    sp = SPMLP(args.model)
    Handler.sp = sp
    Handler.model_name = args.name
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SP MLP server on http://{args.host}:{args.port}  (model={args.model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
