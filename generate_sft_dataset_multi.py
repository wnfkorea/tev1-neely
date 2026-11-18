#!/usr/bin/env python3
"""Multi-market synthetic SFT dataset generator for the tev1-neely SP decision model.

Extends generate_sft_dataset.py with per-market feature-distribution profiles so a
single MLP generalizes across Korean stocks, US stocks, and coins (24/7).

The SP task is market-independent (same 36-dim feature vector, same 10 SP symbols,
same Neely rules). What differs per market is the FEATURE distribution:
  - volatility  -> base_length_factor  (drives m0_change_pct, retrace_ratio)
  - session     -> base_time            (drives time_bars, time_ratio)
  - price scale -> base_price

Market profiles (weight = traffic share in the mixture prior):

  kr_stock  KOSPI/KOSDAQ Daily      length 5-15%   time 3-8    price 5k-50k
  us_stock  S&P/Nasdaq Daily        length 3-10%   time 3-8    price 10-500
  coin      crypto 24/7             length 10-40%  time 3-30   price 100-50k

Usage:
  python3 generate_sft_dataset_multi.py --count 50000 --seed 20261118 --out sft-dataset-multi-train.jsonl
  python3 generate_sft_dataset_multi.py --count 3000  --seed 424243   --out sft-dataset-multi-heldout.jsonl
  python3 generate_sft_dataset_multi.py --count 3000  --market coin   --out sft-dataset-coin-heldout.jsonl
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
    return math.floor(x + 0.5)


def round3(v: Optional[float]):
    if v is None:
        return None
    q = float(Decimal(v).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return int(q) if q.is_integer() else q


# ---------------------------------------------------------------------------
# Market profiles
# ---------------------------------------------------------------------------

# Per-market feature-distribution parameters.
#   length_factor: fraction of base_price for the base wave length (volatility)
#   base_time:     base bar count (trading session)
#   base_price:    price scale
#   weight:        traffic share in the mixture prior
MARKET_PROFILES = {
    "kr_stock": {
        "weight": 0.40,
        "length_factor": (0.05, 0.15),
        "base_time": (3, 8),
        "base_price": (5000, 50000),
    },
    "us_stock": {
        "weight": 0.30,
        "length_factor": (0.03, 0.10),
        "base_time": (3, 8),
        "base_price": (10, 500),
    },
    "coin": {
        "weight": 0.30,
        "length_factor": (0.10, 0.40),
        "base_time": (3, 30),
        "base_price": (100, 50000),
    },
}

# Real marginal over the valid SP labels (normalized; Invalid handled separately).
# Source: SP_REAL_DISTRIBUTION.md — Kiwoom broker `macro_wave`, 400 KOSPI/KOSDAQ
# symbols, Daily. Used as the base prior; the mixture is formed by weighting this
# prior across market profiles (the SP label marginal is largely market-
# independent, so we reuse it and let the FEATURE distribution carry the market).
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
LABEL_HOSTS = {
    ":F3": [
        ("impulse", [0], 46), ("zigzag", [0], 19),
        ("flat", [0], 18), ("triangle", [0], 15),
        ("complex", [0], 5), ("special5", [0], 5),
    ],
    ":5": [
        ("impulse", [2], 46), ("complex", [2], 5),
    ],
    ":c3": [
        ("impulse", [1, 3], 46), ("zigzag", [1], 19),
        ("triangle", [1, 2], 15), ("complex", [3], 5),
        ("special5", [1, 3], 5),
    ],
    ":L5": [
        ("impulse", [4], 46), ("zigzag", [2], 19),
        ("flat", [2], 18), ("complex", [4], 5),
        ("special5", [4], 5),
    ],
    "b:c3": [("flat", [1], 1)],
    "x:c3": [("complex", [1], 1)],
    ":sL3": [("triangle", [3], 1)],
    ":L3": [("triangle", [4], 1)],
    ":s5": [("special5", [2], 1)],
}

INVALID_WEIGHT = float(os.environ.get("SFT_INVALID_WEIGHT", "3"))


# ---------------------------------------------------------------------------
# Synthetic wave + feature vector
# ---------------------------------------------------------------------------

def build_feature(i: int, waves: List[dict]) -> dict:
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


def build_sequence(rand: Rand, base_price: float, specs: List[tuple],
                   length_factor: tuple, base_time: tuple) -> List[dict]:
    """specs: [(dir, lengthFactor, timeFactor, label), ...]"""
    waves: List[dict] = []
    price = base_price
    base_length = base_price * rand.range(*length_factor)
    base_time = rand.int(*base_time)

    for d, length_factor_i, time_factor, label in specs:
        length = base_length * length_factor_i * rand.range(0.95, 1.05)
        t = base_time * time_factor * rand.range(0.95, 1.05)
        end = price + d * length
        waves.append(make_wave(rand, price, end, t, label))
        price = end
    return waves


PATTERN_SPECS = {
    "impulse": [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.3, 1.0, ":c3"),
        (1, 1.5, 0.5, ":5"),
        (-1, 0.25, 1.2, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ],
    "zigzag": [
        (-1, 1.0, 1.0, ":F3"),
        (1, 0.5, 1.0, ":c3"),
        (-1, 1.2, 0.7, ":L5"),
    ],
    "flat": [
        (-1, 1.0, 1.0, ":F3"),
        (1, 1.0, 1.0, "b:c3"),
        (-1, 1.2, 0.7, ":L5"),
    ],
    "triangle": [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.9, 1.0, ":c3"),
        (1, 0.7, 1.0, ":c3"),
        (-1, 0.5, 1.0, ":sL3"),
        (1, 0.3, 1.0, ":L3"),
    ],
    "complex": [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.4, 0.5, "x:c3"),
        (1, 1.0, 1.0, ":5"),
        (-1, 0.4, 1.0, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ],
    "special5": [
        (1, 1.0, 1.0, ":F3"),
        (-1, 0.5, 1.0, ":c3"),
        (1, 1.1, 0.6, ":s5"),
        (-1, 0.4, 1.0, ":c3"),
        (1, 1.0, 1.0, ":L5"),
    ],
}


def generate_pattern(rand: Rand, base_price: float, name: str,
                     length_factor: tuple, base_time: tuple) -> List[dict]:
    return build_sequence(rand, base_price, PATTERN_SPECS[name], length_factor, base_time)


def generate_invalid(rand: Rand, base_price: float,
                     length_factor: tuple, base_time: tuple) -> List[dict]:
    """5-wave sequence whose TARGET (index 2) violates a Neely rule."""
    base_length = base_price * rand.range(*length_factor)
    base_time = rand.int(*base_time)
    waves: List[dict] = []
    price = base_price

    violation = rand.int(0, 3)

    l0 = base_length * rand.range(0.8, 1.2)
    waves.append(make_wave(rand, price, price + l0, base_time * rand.range(0.8, 1.2), ":F3"))
    price += l0

    l1 = base_length * rand.range(0.8, 1.2)
    waves.append(make_wave(rand, price, price - l1, base_time * rand.range(0.8, 1.2), ":c3"))
    price -= l1

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

    l3 = base_length * rand.range(0.5, 0.9)
    waves.append(make_wave(rand, price, price + l3, base_time * rand.range(0.8, 1.2), ":c3"))
    price += l3

    l4 = base_length * rand.range(0.5, 0.9)
    waves.append(make_wave(rand, price, price - l4, base_time * rand.range(0.8, 1.2), ":L5"))

    return waves


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

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


def pick_market(rand: Rand, markets: List[str]) -> str:
    items = [(m, m, MARKET_PROFILES[m]["weight"]) for m in markets]
    return weighted_pick(rand, items)[0]


def pick_label(rand: Rand) -> str:
    items = [(label, label, weight) for label, weight in TARGET_LABEL_WEIGHTS.items()]
    return weighted_pick(rand, items)[0]


def sample_for_label(rand: Rand, label: str, market: str) -> Optional[dict]:
    hosts = LABEL_HOSTS.get(label)
    if not hosts:
        return None
    gen_name, positions, _ = weighted_pick(rand, hosts)
    prof = MARKET_PROFILES[market]
    base_price = rand.range(*prof["base_price"])
    waves = generate_pattern(rand, base_price, gen_name, prof["length_factor"], prof["base_time"])
    target_idx = positions[rand.int(0, len(positions) - 1)]
    if target_idx < 0 or target_idx >= len(waves):
        return None
    feature = build_feature(target_idx, waves)
    return {
        "state": feature_to_state(feature),
        "question": QUESTION,
        "options": list(SP_CHOICES),
        "answer": label,
        "answer_key": label,
        "market": market,
    }


def is_invalid_signature(f: dict) -> bool:
    non_alternation = f["m1_dir"] is not None and f["m1_dir"] == f["m0_dir"]
    r = f["m1_retrace_ratio"]
    t = f["m1_time_ratio"]
    extreme_retrace = r is not None and (r > 8 or r < 0.1)
    extreme_time = t is not None and (t < 0.15 or t > 4)
    return non_alternation or extreme_retrace or extreme_time


def generate_invalid_sample(rand: Rand, market: str) -> Optional[dict]:
    prof = MARKET_PROFILES[market]
    base_price = rand.range(*prof["base_price"])
    waves = generate_invalid(rand, base_price, prof["length_factor"], prof["base_time"])
    target_idx = 2
    feature = build_feature(target_idx, waves)
    if not is_invalid_signature(feature):
        return None
    return {
        "state": feature_to_state(feature),
        "question": QUESTION,
        "options": list(SP_CHOICES),
        "answer": "Invalid_Neely_Rule_Violation",
        "answer_key": "Invalid_Neely_Rule_Violation",
        "market": market,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="sft-dataset-multi.jsonl")
    ap.add_argument("--market", default=None,
                    help="restrict to a single market (default: mixture over all profiles)")
    ap.add_argument("--invalid-weight", type=float, default=INVALID_WEIGHT)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (os.getpid() * 2654435761) & 0xFFFFFFFF
    rand = Rand(mulberry32(seed))
    markets = [args.market] if args.market else list(MARKET_PROFILES.keys())

    print(f"[SFT-multi] Generating {args.count} samples (seed={seed}, markets={markets}) -> {args.out}")

    lines: List[str] = []
    label_counts: dict = {}
    market_counts: dict = {}
    invalid_count = 0
    skipped = 0

    for _ in range(args.count):
        market = pick_market(rand, markets)
        if args.invalid_weight > 0 and rand.chance(args.invalid_weight / 100.0):
            sample = generate_invalid_sample(rand, market)
        else:
            sample = sample_for_label(rand, pick_label(rand), market)

        if sample is None:
            skipped += 1
            continue

        if sample["answer"] == "Invalid_Neely_Rule_Violation":
            invalid_count += 1
        label_counts[sample["answer"]] = label_counts.get(sample["answer"], 0) + 1
        market_counts[sample["market"]] = market_counts.get(sample["market"], 0) + 1
        lines.append(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    total = len(lines)
    print(f"[SFT-multi] Wrote {total} samples to {args.out} (skipped={skipped})")
    print(f"[SFT-multi] Invalid ratio: {invalid_count / max(1, total) * 100:.1f}%")
    print("[SFT-multi] Market distribution:")
    for m, c in sorted(market_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m}: {c} ({c / max(1, total) * 100:.1f}%)")
    print("[SFT-multi] Label distribution:")
    for label, c in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {c} ({c / max(1, total) * 100:.1f}%)")


if __name__ == "__main__":
    main()
