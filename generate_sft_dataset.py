#!/usr/bin/env python3
"""Standalone synthetic SFT dataset generator for the tev1-neely SP decision model.

This is a dependency-free (stdlib-only) port of the original TypeScript generator
(`stock-twin-kr/scripts/generate-sft-dataset.ts`) so that the whole pipeline can be
reproduced from this repository alone — no Node.js, no other repo required.

It generates monowave feature vectors (the standardized state, see
`src/services/analysis/elliott-wave.service.ts`) paired with ground-truth Neely SP
labels, following the quantitative rules of Glenn Neely's *Mastering Elliott Wave*.

Key design points (see README / PIPELINE.md §2):

* **Label-first sampling.** Every pattern starts with `:F3`, `:5` only appears in
  impulse/complex, `x:c3` only in complex, `:sL3`/`:L3` only in triangle, etc. So
  pattern weights alone cannot reproduce the real marginal distribution. Instead we
  draw the target label from the measured real-world prior (TARGET_LABEL_WEIGHTS)
  and then generate a host pattern that can carry it.
* **Measured prior.** TARGET_LABEL_WEIGHTS is the real SP distribution measured from
  the Kiwoom broker `macro_wave` (400 KOSPI/KOSDAQ symbols, Daily). See
  SP_REAL_DISTRIBUTION.md.
* **Invalid gate.** `Invalid_Neely_Rule_Violation` is a project-invented reject class
  that never occurs in real data (0 / ~77k). A small hedge is kept
  (`--invalid-weight`, default 3%), and every Invalid sample is built so that the
  rule violation is *visible* in the target's local feature vector (outside the
  measured valid bands), otherwise it is dropped.

Output: JSONL, one sample per line:
  {"state": {...}, "question": "...", "options": [...], "answer": "<SP label>"}

Usage:
  python3 generate_sft_dataset.py --count 20000 --seed 20261113 --out sft-dataset-train4.jsonl
  python3 generate_sft_dataset.py --count 3000  --seed 424242   --out sft-dataset-heldout4.jsonl
  SFT_INVALID_WEIGHT=0 python3 generate_sft_dataset.py --count 20000 --seed 1 --out d.jsonl

Note: the PRNG (mulberry32) and the generation logic mirror the TypeScript original,
so the *distribution and algorithm* are identical. Byte-for-byte equality with the
committed `sft-dataset-*.jsonl` is not guaranteed (JS/Python number formatting
differs); the committed files are the reference outputs produced by the original
generator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# SP choices + question (must mirror elliott-wave.service.ts SP_CHOICES)
# ---------------------------------------------------------------------------

SP_CHOICES = [
    ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3",
    "Invalid_Neely_Rule_Violation",
]

QUESTION = (
    "Classify the Neely SP symbol for the target mono wave (m0) given the "
    "standardized monowave feature state"
)


# ---------------------------------------------------------------------------
# Random utilities (seedable; exact port of the TS mulberry32)
# ---------------------------------------------------------------------------

def mulberry32(seed: int) -> Callable[[], float]:
    """32-bit mulberry32 PRNG, matching the TypeScript original bit-for-bit."""
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * ((1 | a) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * ((61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rng


class Rand:
    def __init__(self, rng: Callable[[], float]):
        self._rng = rng

    def range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self._rng()

    def int(self, lo: int, hi: int) -> int:
        return math.floor(self.range(lo, hi + 1))

    def chance(self, p: float) -> bool:
        return self._rng() < p


def js_round(x: float) -> int:
    """Math.round (round half toward +inf)."""
    return math.floor(x + 0.5)


def round3(v: Optional[float]):
    """Number(v.toFixed(3)) — round to 3 decimals, half away from zero.

    Returns an ``int`` for integral values so the JSON output matches
    ``JSON.stringify`` (which prints ``1`` rather than ``1.0``).
    """
    if v is None:
        return None
    q = float(Decimal(v).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return int(q) if q.is_integer() else q


# ---------------------------------------------------------------------------
# Synthetic wave + feature vector
# ---------------------------------------------------------------------------

class Wave(dict):
    pass


def build_feature(i: int, waves: List[dict]) -> dict:
    """Build the standardized monowave feature vector for the target wave at i.

    Mirrors ElliottWaveService.buildMonowaveFeatures on the synthetic sequence.
    """
    nw = waves[i]
    m0_dir = "up" if nw["endPrice"] > nw["startPrice"] else "down"
    m0_length = abs(nw["endPrice"] - nw["startPrice"])
    m0_change_pct = (
        ((nw["endPrice"] - nw["startPrice"]) / nw["startPrice"]) * 100
        if nw["startPrice"] != 0 else 0
    )
    m0_time_bars = nw["timeBars"]

    def get_wave(j: int):
        if j < 0 or j >= len(waves):
            return None
        w = waves[j]
        return {
            "dir": "up" if w["endPrice"] > w["startPrice"] else "down",
            "length": abs(w["endPrice"] - w["startPrice"]),
            "timeBars": w["timeBars"],
            "high": w["high"],
            "low": w["low"],
        }

    w1 = get_wave(i - 1)
    w2 = get_wave(i - 2)
    w_1 = get_wave(i + 1)
    w_2 = get_wave(i + 2)

    m1 = None
    if w1:
        m1 = {
            "dir": w1["dir"],
            "length": w1["length"],
            "timeBars": w1["timeBars"],
            "retraceRatio": (w1["length"] / m0_length) if m0_length != 0 else 0,
            "timeRatio": (w1["timeBars"] / m0_time_bars) if m0_time_bars != 0 else 0,
        }
    m2 = None
    if w2:
        m2 = {
            "dir": w2["dir"],
            "length": w2["length"],
            "timeBars": w2["timeBars"],
            "retraceRatio": (w2["length"] / w1["length"]) if (w1 and w1["length"] != 0) else 0,
        }
    m_1 = {"dir": w_1["dir"], "length": w_1["length"], "timeBars": w_1["timeBars"]} if w_1 else None
    m_2 = {"dir": w_2["dir"], "length": w_2["length"], "timeBars": w_2["timeBars"]} if w_2 else None

    m0_high = nw["high"]
    m0_low = nw["low"]
    m0_overlap_m2 = (m0_low < w2["high"] and m0_high > w2["low"]) if w2 else False

    rule_of_equality = (
        abs(w1["length"] - w2["length"]) / max(w1["length"], w2["length"]) <= 0.15
        if (w1 and w2) else False
    )
    channel_break = (
        m0_high > max(w1["high"], w2["high"]) or m0_low < min(w1["low"], w2["low"])
        if (w1 and w2) else False
    )

    return {
        "m0_dir": m0_dir,
        "m0_length": m0_length,
        "m0_change_pct": m0_change_pct,
        "m0_time_bars": m0_time_bars,
        "m1_dir": m1["dir"] if m1 else None,
        "m1_length": m1["length"] if m1 else None,
        "m1_time_bars": m1["timeBars"] if m1 else None,
        "m1_retrace_ratio": m1["retraceRatio"] if m1 else None,
        "m1_time_ratio": m1["timeRatio"] if m1 else None,
        "m2_dir": m2["dir"] if m2 else None,
        "m2_length": m2["length"] if m2 else None,
        "m2_time_bars": m2["timeBars"] if m2 else None,
        "m2_retrace_ratio": m2["retraceRatio"] if m2 else None,
        "m_1_dir": m_1["dir"] if m_1 else None,
        "m_1_length": m_1["length"] if m_1 else None,
        "m_1_time_bars": m_1["timeBars"] if m_1 else None,
        "m_2_dir": m_2["dir"] if m_2 else None,
        "m_2_length": m_2["length"] if m_2 else None,
        "m_2_time_bars": m_2["timeBars"] if m_2 else None,
        "m0_overlap_m2": m0_overlap_m2,
        "rule_of_equality_satisfied": rule_of_equality,
        "channel_break": channel_break,
    }


# ---------------------------------------------------------------------------
# Pattern generators (each wave carries a ground-truth SP label)
# ---------------------------------------------------------------------------

def make_wave(rand: Rand, start: float, end: float, time_bars: float, label: str) -> dict:
    is_up = end > start
    wick = abs(end - start) * rand.range(0.05, 0.15)
    return {
        "startPrice": start,
        "endPrice": end,
        "high": (end + wick) if is_up else (start + wick),
        "low": (start - wick) if is_up else (end - wick),
        "timeBars": max(1, js_round(time_bars)),
        "label": label,
    }


def build_sequence(rand: Rand, base_price: float, specs: List[tuple]) -> List[dict]:
    """specs: [(dir, lengthFactor, timeFactor, label), ...]"""
    waves: List[dict] = []
    price = base_price
    base_length = base_price * rand.range(0.05, 0.15)
    base_time = rand.int(3, 8)

    for d, length_factor, time_factor, label in specs:
        length = base_length * length_factor * rand.range(0.95, 1.05)
        t = base_time * time_factor * rand.range(0.95, 1.05)
        end = price + d * length
        waves.append(make_wave(rand, price, end, t, label))
        price = end
    return waves


def generate_impulse(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.3, 1.0, ":c3"),
        (1, 1.5, 0.5, ":5"),
        (-1, 0.25, 1.2, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ])


def generate_zigzag(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (-1, 1.0, 1.0, ":F3"),
        (1, 0.5, 1.0, ":c3"),
        (-1, 1.2, 0.7, ":L5"),
    ])


def generate_flat(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (-1, 1.0, 1.0, ":F3"),
        (1, 1.0, 1.0, "b:c3"),
        (-1, 1.2, 0.7, ":L5"),
    ])


def generate_triangle(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.9, 1.0, ":c3"),
        (1, 0.7, 1.0, ":c3"),
        (-1, 0.5, 1.0, ":sL3"),
        (1, 0.3, 1.0, ":L3"),
    ])


def generate_complex(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.4, 0.5, "x:c3"),
        (1, 1.0, 1.0, ":5"),
        (-1, 0.4, 1.0, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ])


def generate_special5(rand: Rand, base_price: float) -> List[dict]:
    return build_sequence(rand, base_price, [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.5, 1.0, ":c3"),
        (1, 1.1, 0.6, ":s5"),
        (-1, 0.4, 1.0, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ])


def generate_invalid(rand: Rand, base_price: float) -> List[dict]:
    """5-wave sequence whose TARGET (index 2) violates a Neely rule.

    Violations sit OUTSIDE the measured valid-label bands (valid samples always
    alternate; m1_retrace_ratio in [0.18, 6.60]; m1_time_ratio in [0.25, 3.00]):
      0: non-alternation     — m1_dir == m0_dir
      1: extreme retrace     — m1_retrace_ratio ~ 13-60
      2: extreme time ratio  — m1_time_ratio ~ 0.03-0.09
      3: compound            — non-alternation + extreme retrace
    """
    base_length = base_price * rand.range(0.05, 0.15)
    base_time = rand.int(3, 8)
    waves: List[dict] = []
    price = base_price

    violation = rand.int(0, 3)

    # m2 (index 0, pattern head) — up
    l0 = base_length * rand.range(0.8, 1.2)
    waves.append(make_wave(rand, price, price + l0, base_time * rand.range(0.8, 1.2), ":F3"))
    price += l0

    # m1 (index 1) — down (alternates with m2)
    l1 = base_length * rand.range(0.8, 1.2)
    waves.append(make_wave(rand, price, price - l1, base_time * rand.range(0.8, 1.2), ":c3"))
    price -= l1

    # m0 (index 2, TARGET) — carries the violation
    if violation == 0:
        dir2, l2, t2 = -1, base_length * rand.range(0.5, 1.0), base_time * rand.range(0.8, 1.2)
    elif violation == 1:
        dir2, l2, t2 = 1, base_length * rand.range(0.02, 0.06), base_time * rand.range(0.8, 1.2)
    elif violation == 2:
        dir2, l2, t2 = 1, base_length * rand.range(0.8, 1.2), base_time * rand.range(8, 16)
    else:
        dir2, l2, t2 = -1, base_length * rand.range(0.02, 0.06), base_time * rand.range(0.8, 1.2)
    waves.append(make_wave(rand, price, price + dir2 * l2, t2, "Invalid_Neely_Rule_Violation"))
    price += dir2 * l2

    # m_1 (index 3) — up
    l3 = base_length * rand.range(0.5, 0.9)
    waves.append(make_wave(rand, price, price + l3, base_time * rand.range(0.8, 1.2), ":c3"))
    price += l3

    # m_2 (index 4) — down
    l4 = base_length * rand.range(0.5, 0.9)
    waves.append(make_wave(rand, price, price - l4, base_time * rand.range(0.8, 1.2), ":L5"))

    return waves


# ---------------------------------------------------------------------------
# Real-world target distribution (measured, not assumed)
# ---------------------------------------------------------------------------

# Real marginal over the valid SP labels (normalized; Invalid handled separately).
# Source: SP_REAL_DISTRIBUTION.md — Kiwoom broker `macro_wave`, 400 KOSPI/KOSDAQ
# symbols, Daily, uniform tie-break marginal over candidate lists.
TARGET_LABEL_WEIGHTS = {
    ":F3": 0.3007,
    ":5": 0.2242,
    ":c3": 0.1595,
    ":L5": 0.1363,
    "x:c3": 0.0598,
    ":s5": 0.0553,
    ":sL3": 0.0232,
    "b:c3": 0.0213,
    ":L3": 0.0197,
}

# Host patterns per label: which generator can emit it, at which target positions.
# `weight` is the relative preference among hosts (feature diversity).
LABEL_HOSTS = {
    ":F3": [
        (generate_impulse, [0], 46), (generate_zigzag, [0], 19),
        (generate_flat, [0], 18), (generate_triangle, [0], 15),
        (generate_complex, [0], 5), (generate_special5, [0], 5),
    ],
    ":5": [
        (generate_impulse, [2], 46), (generate_complex, [2], 5),
    ],
    ":c3": [
        (generate_impulse, [1, 3], 46), (generate_zigzag, [1], 19),
        (generate_triangle, [1, 2], 15), (generate_complex, [3], 5),
        (generate_special5, [1, 3], 5),
    ],
    ":L5": [
        (generate_impulse, [4], 46), (generate_zigzag, [2], 19),
        (generate_flat, [2], 18), (generate_complex, [4], 5),
        (generate_special5, [4], 5),
    ],
    "b:c3": [(generate_flat, [1], 1)],
    "x:c3": [(generate_complex, [1], 1)],
    ":sL3": [(generate_triangle, [3], 1)],
    ":L3": [(generate_triangle, [4], 1)],
    ":s5": [(generate_special5, [2], 1)],
}

# Invalid share (%). Real data never emits Invalid (0 / ~77k). A small hedge keeps
# the reject class learnable. Set SFT_INVALID_WEIGHT=0 to match production exactly.
INVALID_WEIGHT = float(os.environ.get("SFT_INVALID_WEIGHT", "3"))


def feature_to_state(feature: dict) -> dict:
    return {
        "m0_dir": feature["m0_dir"],
        "m0_length": round3(feature["m0_length"]),
        "m0_change_pct": round3(feature["m0_change_pct"]),
        "m0_time_bars": feature["m0_time_bars"],
        "m1_dir": feature["m1_dir"],
        "m1_length": round3(feature["m1_length"]),
        "m1_time_bars": feature["m1_time_bars"],
        "m1_retrace_ratio": round3(feature["m1_retrace_ratio"]),
        "m1_time_ratio": round3(feature["m1_time_ratio"]),
        "m2_dir": feature["m2_dir"],
        "m2_length": round3(feature["m2_length"]),
        "m2_time_bars": feature["m2_time_bars"],
        "m2_retrace_ratio": round3(feature["m2_retrace_ratio"]),
        "m_1_dir": feature["m_1_dir"],
        "m_1_length": round3(feature["m_1_length"]),
        "m_1_time_bars": feature["m_1_time_bars"],
        "m_2_dir": feature["m_2_dir"],
        "m_2_length": round3(feature["m_2_length"]),
        "m_2_time_bars": feature["m_2_time_bars"],
        "m0_overlap_m2": feature["m0_overlap_m2"],
        "rule_of_equality_satisfied": feature["rule_of_equality_satisfied"],
        "channel_break": feature["channel_break"],
    }


def weighted_pick(rand: Rand, items: list):
    total = sum(it[2] for it in items)
    r = rand.range(0, total)
    for it in items:
        if r < it[2]:
            return it
        r -= it[2]
    return items[-1]


def pick_label(rand: Rand) -> str:
    items = [(label, label, weight) for label, weight in TARGET_LABEL_WEIGHTS.items()]
    return weighted_pick(rand, items)[0]


def sample_for_label(rand: Rand, label: str) -> Optional[dict]:
    hosts = LABEL_HOSTS.get(label)
    if not hosts:
        return None
    gen, positions, _ = weighted_pick(rand, hosts)
    base_price = rand.range(5000, 50000)
    waves = gen(rand, base_price)
    target_idx = positions[rand.int(0, len(positions) - 1)]
    if target_idx < 0 or target_idx >= len(waves):
        return None
    feature = build_feature(target_idx, waves)
    return {
        "state": feature_to_state(feature),
        "question": QUESTION,
        "options": list(SP_CHOICES),
        "answer": label,
    }


def is_invalid_signature(f: dict) -> bool:
    non_alternation = f["m1_dir"] is not None and f["m1_dir"] == f["m0_dir"]
    r = f["m1_retrace_ratio"]
    t = f["m1_time_ratio"]
    extreme_retrace = r is not None and (r > 8 or r < 0.1)
    extreme_time = t is not None and (t < 0.15 or t > 4)
    return non_alternation or extreme_retrace or extreme_time


def generate_invalid_sample(rand: Rand) -> Optional[dict]:
    base_price = rand.range(5000, 50000)
    waves = generate_invalid(rand, base_price)
    target_idx = 2  # the wave that carries the violation
    feature = build_feature(target_idx, waves)
    if not is_invalid_signature(feature):
        return None  # gate: signature must be visible in the feature vector
    return {
        "state": feature_to_state(feature),
        "question": QUESTION,
        "options": list(SP_CHOICES),
        "answer": "Invalid_Neely_Rule_Violation",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20000, help="number of samples (default 20000)")
    ap.add_argument("--seed", type=int, default=None, help="PRNG seed (default random)")
    ap.add_argument("--out", default="sft-dataset.jsonl", help="output JSONL path")
    ap.add_argument("--invalid-weight", type=float, default=INVALID_WEIGHT,
                    help="%% share of Invalid_Neely_Rule_Violation samples (default 3)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (os.getpid() * 2654435761) & 0xFFFFFFFF
    rand = Rand(mulberry32(seed))
    invalid_weight = args.invalid_weight

    print(f"[SFT] Generating {args.count} samples (seed={seed}) -> {args.out}")

    lines: List[str] = []
    label_counts: dict = {}
    invalid_count = 0
    skipped = 0

    for _ in range(args.count):
        if invalid_weight > 0 and rand.chance(invalid_weight / 100.0):
            sample = generate_invalid_sample(rand)
        else:
            sample = sample_for_label(rand, pick_label(rand))

        if sample is None:
            skipped += 1
            continue

        if sample["answer"] == "Invalid_Neely_Rule_Violation":
            invalid_count += 1
        label_counts[sample["answer"]] = label_counts.get(sample["answer"], 0) + 1
        lines.append(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    total = len(lines)
    print(f"[SFT] Wrote {total} samples to {args.out} (skipped={skipped})")
    print(f"[SFT] Invalid ratio: {invalid_count / max(1, total) * 100:.1f}%")
    print("[SFT] Label distribution:")
    for label, c in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {c} ({c / max(1, total) * 100:.1f}%)")


if __name__ == "__main__":
    main()
