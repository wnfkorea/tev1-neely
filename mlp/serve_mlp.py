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


class SPMLP:
    def __init__(self, path):
        art = torch.load(path, map_location="cpu", weights_only=False)
        self.keys = art["keys"]
        self.mu = np.asarray(art["mu"], dtype=np.float32)
        self.sd = np.asarray(art["sd"], dtype=np.float32)
        self.dirf = art["dirf"]
        self.numf = art["numf"]
        self.boolf = art["boolf"]
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
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode())
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
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
