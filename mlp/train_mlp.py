#!/usr/bin/env python3
"""Train the SP decision MLP (distilled from the Tev1-4B decision model).

The Tev1 SP task is a classification over the standardised monowave feature
vector (m0 target + m1/m2/m_1/m_2 neighbours with retrace/time ratios). A small
MLP reaches the same accuracy as the 4B LLM on the held-out set and runs in
microseconds on CPU.

Usage:
  python mlp/train_mlp.py \
      --train sp-systemone-train.jsonl \
      --heldout sp-systemone-heldout.jsonl \
      --out mlp/sp_mlp.pt

The saved artifact contains everything the server needs:
  - MLP state_dict
  - feature mean/std (z-score)
  - the 10 SP keys in fixed order (A..J)
  - the feature column order
"""
import argparse
import collections
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Fixed feature layout (must match serve_mlp.py / eval_mlp.py).
DIRF = ["m0_dir", "m1_dir", "m2_dir", "m_1_dir", "m_2_dir"]
NUMF = [
    "m0_length", "m0_change_pct", "m0_time_bars",
    "m1_length", "m1_time_bars", "m1_retrace_ratio", "m1_time_ratio",
    "m2_length", "m2_time_bars", "m2_retrace_ratio",
    "m_1_length", "m_1_time_bars", "m_2_length", "m_2_time_bars",
]
BOOLF = ["m0_overlap_m2", "rule_of_equality_satisfied", "channel_break"]
# 10 SP keys in the fixed A..J order (matches the systemone template).
KEYS = [
    ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3",
    ":L5", ":L3", ":sL3", "Invalid_Neely_Rule_Violation",
]
key2i = {k: i for i, k in enumerate(KEYS)}


def feature_vector(state: dict) -> list[float]:
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


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, len(KEYS)),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sp-systemone-train.jsonl")
    ap.add_argument("--heldout", default="sp-systemone-heldout.jsonl")
    ap.add_argument("--out", default="mlp/sp_mlp.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    train = load_rows(args.train)
    held = load_rows(args.heldout)
    Xtr = np.asarray([feature_vector(r["state"]) for r in train], dtype=np.float32)
    ytr = np.asarray([key2i[r["answer_key"]] for r in train], dtype=np.int64)
    Xva = np.asarray([feature_vector(r["state"]) for r in held], dtype=np.float32)
    yva = np.asarray([key2i[r["answer_key"]] for r in held], dtype=np.int64)
    print(f"features={Xtr.shape[1]} train={len(train)} held={len(held)}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd

    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(Xtr_n), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(Xva_n)).argmax(1).numpy()
            print(f"epoch {ep:2d} held_acc={(pred == yva).mean():.4f}")
    print(f"train time {time.time() - t0:.0f}s")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva_n))
        pred = logits.argmax(1).numpy()
    acc = (pred == yva).mean()
    print(f"\nFINAL held-out acc = {acc:.4f}  ({(pred == yva).sum()}/{len(yva)})")
    per = collections.defaultdict(lambda: [0, 0])
    for p, g in zip(pred, yva):
        per[KEYS[g]][1] += 1
        per[KEYS[g]][0] += int(p == g)
    for k in KEYS:
        c, t = per[k]
        if t:
            print(f"  {k:34s} {c:4d}/{t:<4d} {c / t:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "mu": mu,
        "sd": sd,
        "keys": KEYS,
        "dirf": DIRF,
        "numf": NUMF,
        "boolf": BOOLF,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
