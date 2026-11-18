#!/usr/bin/env python3
"""Evaluate the REFINED shape-invariant SP decision MLP (no length, keep time_bars).

Usage:
  python mlp/eval_nolen.py --model mlp/sp_mlp_nolen.pt --heldout sp-systemone-multi-heldout.jsonl
"""
import argparse
import collections
import json
import time

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    # Must match train_nolen.py exactly (Dropout is a no-op in eval mode).
    def __init__(self, d, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, n_out),
        )

    def forward(self, x):
        return self.net(x)


# Refined shape-invariant feature layout (must match train_nolen.py).
DIRF = ["m0_dir", "m1_dir", "m2_dir", "m_1_dir", "m_2_dir"]
NUMF = [
    "m0_change_pct", "m1_retrace_ratio", "m1_time_ratio", "m2_retrace_ratio",
    "m0_time_bars", "m1_time_bars", "m2_time_bars", "m_1_time_bars", "m_2_time_bars",
]
BOOLF = ["m0_overlap_m2", "rule_of_equality_satisfied", "channel_break"]


def feature_vector(state):
    v = []
    for f in DIRF:
        v.append(1.0 if state.get(f) == "up" else (-1.0 if state.get(f) == "down" else 0.0))
    for f in NUMF:
        x = state.get(f)
        v.append(0.0 if x is None else float(x))
        v.append(0.0 if x is not None else 1.0)  # null indicator
    for f in BOOLF:
        v.append(1.0 if state.get(f) else 0.0)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp/sp_mlp_nolen.pt")
    ap.add_argument("--heldout", default="sp-systemone-multi-heldout.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    art = torch.load(args.model, map_location="cpu", weights_only=False)
    keys = art["keys"]
    mu, sd = np.asarray(art["mu"], np.float32), np.asarray(art["sd"], np.float32)
    model = MLP(len(mu), len(keys))
    model.load_state_dict(art["state"])
    model.eval()

    rows = []
    with open(args.heldout, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if args.limit and len(rows) >= args.limit:
                    break

    X = np.asarray([feature_vector(r["state"]) for r in rows], dtype=np.float32)
    y = np.asarray([keys.index(r["answer_key"]) for r in rows], dtype=np.int64)
    Xn = (X - mu) / sd

    with torch.no_grad():
        logits = model(torch.tensor(Xn))
        pred = logits.argmax(1).numpy()
    acc = (pred == y).mean()
    print(f"held-out acc = {acc:.4f}  ({(pred == y).sum()}/{len(y)})")

    per = collections.defaultdict(lambda: [0, 0])
    for p, g in zip(pred, y):
        per[keys[g]][1] += 1
        per[keys[g]][0] += int(p == g)
    for k in keys:
        c, t = per[k]
        if t:
            print(f"  {k:34s} {c:4d}/{t:<4d} {c / t:.3f}")

    def run(n, batched):
        Xb = torch.tensor(Xn[:n])
        for _ in range(3):
            with torch.no_grad():
                if batched:
                    model(Xb)
                else:
                    for i in range(min(5, n)):
                        model(Xb[i:i + 1])
        t0 = time.perf_counter()
        with torch.no_grad():
            if batched:
                model(Xb)
            else:
                for i in range(n):
                    model(Xb[i:i + 1])
        return time.perf_counter() - t0

    n = min(200, len(rows))
    single = run(n, False)
    batched = run(n, True)
    print(f"\nCPU inference (n={n}):")
    print(f"  single:  {single / n * 1e6:.1f} us/req  ({n / single:.0f} req/s)")
    print(f"  batched: {batched / n * 1e6:.1f} us/req  ({n / batched:.0f} req/s)")


if __name__ == "__main__":
    main()
