#!/usr/bin/env python3
"""Regression audit for Invalid_Neely_Rule_Violation labels in an SFT dataset.

Verifies that every Invalid-labelled sample is *learnable*: the claimed rule
violation must be visible in the target's local feature vector, and the Invalid
class must be separable from the valid labels in feature space.

Usage:
  ./venv/bin/python audit_invalid_labels.py sft-dataset-train4.jsonl
"""
import collections
import json
import sys

import numpy as np

FEATS = [
    "m0_dir", "m1_dir", "m2_dir", "m_1_dir", "m_2_dir",
    "m1_retrace_ratio", "m1_time_ratio", "m2_retrace_ratio",
    "m0_overlap_m2", "rule_of_equality_satisfied", "channel_break",
]
INVALID = "Invalid_Neely_Rule_Violation"


def vec(s):
    out = []
    for k in FEATS:
        v = s.get(k)
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif v is None:
            out.append(-1.0)
        elif v == "up":
            out.append(1.0)
        elif v == "down":
            out.append(0.0)
        else:
            out.append(float(v))
    return out


def has_invalid_signature(s):
    non_alternation = s["m1_dir"] is not None and s["m1_dir"] == s["m0_dir"]
    r = s["m1_retrace_ratio"]
    t = s["m1_time_ratio"]
    extreme_retrace = r is not None and (r > 8 or r < 0.1)
    extreme_time = t is not None and (t < 0.15 or t > 4)
    return non_alternation or extreme_retrace or extreme_time


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sft-dataset-train4.jsonl"
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    inv = [r for r in data if r["answer"] == INVALID]
    val = [r for r in data if r["answer"] != INVALID]
    if not inv:
        sys.exit(f"{path}: no Invalid samples (set SFT_INVALID_WEIGHT>0 to audit)")

    sig_ok = sum(1 for r in inv if has_invalid_signature(r["state"]))

    V = np.array([vec(r["state"]) for r in data])
    mu, sd = V.mean(0), V.std(0) + 1e-9
    Vn = (V - mu) / sd
    lab = np.array([r["answer"] for r in data])
    inv_idx = np.where(lab == INVALID)[0]
    val_idx = np.where(lab != INVALID)[0]
    Vv, Vi = Vn[val_idx], Vn[inv_idx]

    nn_invalid = 0
    ambiguous = 0
    for k, i in enumerate(inv_idx):
        d = ((Vn - Vn[i]) ** 2).sum(1)
        d[i] = 1e9
        if lab[int(d.argmin())] == INVALID:
            nn_invalid += 1
        dval = np.sqrt(((Vv - Vn[i]) ** 2).sum(1)).min()
        do = np.sqrt(((np.delete(Vi, k, axis=0) - Vn[i]) ** 2).sum(1)).min()
        if dval < do:
            ambiguous += 1

    fp = sum(1 for r in val if has_invalid_signature(r["state"]))

    print(f"{path}")
    print(f"  Invalid samples                         : {len(inv)}")
    print(f"  violation signature visible in features : {sig_ok}/{len(inv)} = {sig_ok/len(inv)*100:.1f}%")
    print(f"  1-NN label is Invalid                   : {nn_invalid}/{len(inv)} = {nn_invalid/len(inv)*100:.1f}%")
    print(f"  ambiguous (valid closer than Invalid)   : {ambiguous}/{len(inv)} = {ambiguous/len(inv)*100:.1f}%")
    print(f"  valid samples matching signature (FP)   : {fp}/{len(val)} = {fp/len(val)*100:.2f}%")

    ok = (sig_ok == len(inv)) and (nn_invalid == len(inv)) and (ambiguous == 0)
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
