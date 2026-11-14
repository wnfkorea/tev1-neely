#!/usr/bin/env python3
"""Estimate the achievable accuracy ceiling of the SP decision task.

Trains a small MLP on the *same* monowave feature vector that the 4B model
sees, using a proper stratified 80/20 split (no leakage). If a strong model on
these features cannot exceed ~X%, then X is the feature-imposed ceiling and
more data/epochs on the 4B model cannot help.

Usage: CUDA_VISIBLE_DEVICES= ./venv/bin/python eval/ceiling_probe.py
"""
import json, collections, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PATH = "sp-systemone-train.jsonl"

DIRF = ["m0_dir", "m1_dir", "m2_dir", "m_1_dir", "m_2_dir"]
NUMF = [
    "m0_length", "m0_change_pct", "m0_time_bars",
    "m1_length", "m1_time_bars", "m1_retrace_ratio", "m1_time_ratio",
    "m2_length", "m2_time_bars", "m2_retrace_ratio",
    "m_1_length", "m_1_time_bars", "m_2_length", "m_2_time_bars",
]
BOOLF = ["m0_overlap_m2", "rule_of_equality_satisfied", "channel_break"]

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

rows = load(PATH)
keys = sorted({r["answer_key"] for r in rows})
key2i = {k: i for i, k in enumerate(keys)}
print(f"rows={len(rows)} classes={len(keys)}")

# Build feature matrix
feats = []
for r in rows:
    s = r["state"]
    v = []
    for f in DIRF:
        v.append(1.0 if s.get(f) == "up" else (-1.0 if s.get(f) == "down" else 0.0))
    for f in NUMF:
        x = s.get(f)
        v.append(0.0 if x is None else float(x))
        v.append(0.0 if x is not None else 1.0)  # null indicator
    for f in BOOLF:
        v.append(1.0 if s.get(f) else 0.0)
    feats.append(v)
X = np.asarray(feats, dtype=np.float32)
y = np.asarray([key2i[r["answer_key"]] for r in rows], dtype=np.int64)

# Standardize numerics (skip the 0/1 dir & bool columns is unnecessary; z-score all)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = (X - mu) / sd

# Stratified 80/20 split
rng = np.random.default_rng(0)
idx = np.arange(len(rows)); rng.shuffle(idx)
# stratify: within each class, take 20%
tr, va = [], []
byclass = collections.defaultdict(list)
for i in idx:
    byclass[y[i]].append(i)
for c, ids in byclass.items():
    k = max(1, int(0.2 * len(ids)))
    va.extend(ids[:k]); tr.extend(ids[k:])
tr, va = np.array(tr), np.array(va)
print(f"train={len(tr)} val={len(va)}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.tensor(Xn); yt = torch.tensor(y)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 10),
        )
    def forward(self, x): return self.net(x)

model = MLP(X.shape[1]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
ds = TensorDataset(Xt[tr].to(dev), yt[tr].to(dev))
dl = DataLoader(ds, batch_size=256, shuffle=True)

best = 0.0
for ep in range(60):
    model.train()
    for xb, yb in dl:
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt[va].to(dev)).argmax(1).cpu()
    acc = (pred == yt[va]).float().mean().item()
    best = max(best, acc)
    if ep % 10 == 0 or ep == 59:
        print(f"epoch {ep:2d}  val_acc={acc:.4f}")

# per-class metrics at best-ish (final)
with torch.no_grad():
    pred = model(Xt[va].to(dev)).argmax(1).cpu().numpy()
gt = y[va]
print(f"\nFINAL val acc = {(pred==gt).mean():.4f}  (majority baseline = {max(collections.Counter(gt).values())/len(gt):.4f})")
print("per-class recall (n):")
for c, name in enumerate(keys):
    m = gt == c
    if m.sum():
        print(f"  {name:34s} n={m.sum():4d}  recall={ (pred[m]==c).mean():.3f}")
