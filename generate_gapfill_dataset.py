#!/usr/bin/env python3
"""Generate gap-filling training data for the nolen LLM.

The nolen training data has gaps in the m1_retrace_ratio space:
  - Gap 1: [0.50, 0.68] between :s5 (max 0.50) and Invalid (min 0.68)
  - Gap 2: [3.67, 5.45] between :c3 (max 6.62) and x:c3 (min 2.27)
  - Gap 3: [6.62, 13.61] between :c3 (max 6.62) and Invalid (max 57.7)

This script generates data with m1_retrace_ratio in these gap regions,
with SP labels assigned based on the Neely rule-conditions.

Usage:
  python generate_gapfill_dataset.py --count 5000 --out sp-systemone-nolen-gapfill.jsonl
"""
import argparse
import json
import random
import math

# Gap regions for m1_retrace_ratio
GAPS = [
    {"name": "gap1", "min": 0.50, "max": 0.68},
    {"name": "gap2", "min": 3.67, "max": 5.45},
    {"name": "gap3", "min": 6.62, "max": 13.61},
]

# SP rule-conditions (simplified) — assign based on nearest SP in the gap region
def assign_sp(gap_name, m1_retrace, m1_time, m2_retrace, m0_change_pct, m0_dir, m1_dir, m2_dir,
              m0_overlap_m2, rule_of_equality, channel_break, rng):
    """Assign SP label based on the gap region and rule-conditions.
    
    For each gap region, assign SP labels from the neighboring SP symbols
    with a balanced distribution.
    """
    # Gap 1 [0.50, 0.68]: between :s5 and Invalid
    if gap_name == "gap1":
        # :s5 if m1_time ~1.67 and m2_retrace ~2.0
        if 1.3 <= m1_time <= 2.0 and m2_retrace is not None and 1.8 <= m2_retrace <= 2.2:
            return ":s5"
        # :5 if m1_time > 1 and m2_retrace > 2
        if m1_time > 1.0 and m2_retrace is not None and m2_retrace > 2.0:
            return ":5"
        # :c3 if m1_time ~1.0
        if 0.8 <= m1_time <= 1.2:
            return ":c3"
        # Invalid otherwise
        return "Invalid_Neely_Rule_Violation"
    
    # Gap 2 [3.67, 5.45]: between :c3 and x:c3
    if gap_name == "gap2":
        # x:c3 if m1_time ~2.0 and m2_retrace null
        if 1.5 <= m1_time <= 3.0 and (m2_retrace is None or m2_retrace < 1.0):
            return "x:c3"
        # :c3 if m1_time ~1.0
        if 0.8 <= m1_time <= 1.2:
            return ":c3"
        # :5 if m1_time > 1 and m2_retrace > 2
        if m1_time > 1.0 and m2_retrace is not None and m2_retrace > 2.0:
            return ":5"
        # Invalid otherwise
        return "Invalid_Neely_Rule_Violation"
    
    # Gap 3 [6.62, 13.61]: between :c3 and Invalid
    if gap_name == "gap3":
        # :c3 if m1_time ~1.0
        if 0.8 <= m1_time <= 1.2:
            return ":c3"
        # :5 if m1_time > 1 and m2_retrace > 2
        if m1_time > 1.0 and m2_retrace is not None and m2_retrace > 2.0:
            return ":5"
        # x:c3 if m1_time ~2.0
        if 1.5 <= m1_time <= 3.0:
            return "x:c3"
        # Invalid otherwise
        return "Invalid_Neely_Rule_Violation"
    
    # Default: Invalid
    return "Invalid_Neely_Rule_Violation"


def generate_sample(rng, gap):
    """Generate one sample with m1_retrace_ratio in the gap region."""
    m1_retrace = rng.uniform(gap["min"], gap["max"])
    
    # Generate other features
    m0_dir = rng.choice(["up", "down"])
    m1_dir = rng.choice(["up", "down"])
    m2_dir = rng.choice(["up", "down"])
    
    # m1_time_ratio: random in [0.2, 3.0]
    m1_time = rng.uniform(0.2, 3.0)
    
    # m2_retrace_ratio: random in [0.1, 6.0] or None
    if rng.random() < 0.7:
        m2_retrace = rng.uniform(0.1, 6.0)
    else:
        m2_retrace = None
    
    # m0_change_pct: random in [-40, 50]
    m0_change_pct = rng.uniform(-40, 50)
    
    # m0_time_bars: random in [1, 30]
    m0_time_bars = rng.randint(1, 30)
    
    # m1_time_bars: random in [1, 30]
    m1_time_bars = rng.randint(1, 30)
    
    # m2_time_bars: random in [1, 30] or None
    if m2_retrace is not None:
        m2_time_bars = rng.randint(1, 30)
    else:
        m2_time_bars = None
    
    # m_1_dir, m_2_dir: random or None
    m_1_dir = rng.choice(["up", "down", None])
    m_2_dir = rng.choice(["up", "down", None])
    
    # m_1_time_bars, m_2_time_bars: random or None
    m_1_time_bars = rng.randint(1, 30) if m_1_dir else None
    m_2_time_bars = rng.randint(1, 30) if m_2_dir else None
    
    # Bool features
    m0_overlap_m2 = rng.random() < 0.5
    rule_of_equality = rng.random() < 0.3
    channel_break = rng.random() < 0.5
    
    # Assign SP label
    sp = assign_sp(gap["name"], m1_retrace, m1_time, m2_retrace, m0_change_pct, m0_dir, m1_dir, m2_dir,
                   m0_overlap_m2, rule_of_equality, channel_break, rng)
    
    # Build state (17 nolen keys)
    state = {
        "m0_dir": m0_dir,
        "m1_dir": m1_dir,
        "m2_dir": m2_dir,
        "m_1_dir": m_1_dir,
        "m_2_dir": m_2_dir,
        "m0_change_pct": round(m0_change_pct, 3),
        "m1_retrace_ratio": round(m1_retrace, 3),
        "m1_time_ratio": round(m1_time, 3),
        "m2_retrace_ratio": round(m2_retrace, 3) if m2_retrace is not None else None,
        "m0_time_bars": m0_time_bars,
        "m1_time_bars": m1_time_bars,
        "m2_time_bars": m2_time_bars,
        "m_1_time_bars": m_1_time_bars,
        "m_2_time_bars": m_2_time_bars,
        "m0_overlap_m2": m0_overlap_m2,
        "rule_of_equality_satisfied": rule_of_equality,
        "channel_break": channel_break,
    }
    
    return {
        "state": state,
        "answer_key": sp,
        "gap": gap["name"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--out", default="sp-systemone-nolen-gapfill.jsonl")
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()
    
    rng = random.Random(args.seed)
    
    # Generate samples
    samples = []
    sp_counts = {}
    gap_counts = {}
    for i in range(args.count):
        gap = GAPS[i % len(GAPS)]
        sample = generate_sample(rng, gap)
        samples.append(sample)
        sp_counts[sample["answer_key"]] = sp_counts.get(sample["answer_key"], 0) + 1
        gap_counts[sample["gap"]] = gap_counts.get(sample["gap"], 0) + 1
    
    # Write to file
    with open(args.out, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(samples)} samples -> {args.out}")
    print(f"\nGap distribution:")
    for gap, cnt in gap_counts.items():
        print(f"  {gap}: {cnt}")
    print(f"\nSP distribution:")
    for sp, cnt in sorted(sp_counts.items(), key=lambda x: -x[1]):
        print(f"  {sp:<30} {cnt:>6}  {cnt/len(samples)*100:.1f}%")


if __name__ == "__main__":
    main()
