#!/usr/bin/env python3
"""A/B verification on real Upbit ETH data: MLP (8095) vs Ollama tev1-neely (11434).

Reproduces the app's buildMonowaveFeatures (elliott-wave.service.ts) exactly,
builds the standardized state for each ETH monowave, and sends the SAME state
to both models. Compares the chosen SP symbol.
"""
import json
import time
import urllib.request

MLP = "http://127.0.0.1:8095/v1/systemone"
OLLAMA = "http://127.0.0.1:11434/v1/systemone"

SP_CHOICES = [":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3",
             "Invalid_Neely_Rule_Violation"]

INSTRUCTIONS = ("Classify the Neely SP symbol for the target mono wave (m0) "
                "given the standardized monowave feature state")

SP_CRITERIA = {
    ':5': 'Impulse wave (not final): 5-wave pattern that does not end a larger pattern; covers more price in less time than the prior wave (m1_retrace_ratio < 1, m1_time_ratio > 1)',
    ':s5': 'Special 5-wave: part of a complex correction or 3rd wave of an impulse with failed/truncated 5th wave; can occupy :L5 position without reversal confirmation; two waves before it must be 5-F3 or F3-c3',
    ':c3': 'Central 3-wave: middle of a pattern, not initial or final; not immediately followed by a large strong directional move; the prior wave is longer (m1_retrace_ratio > 1) and consumes same or more time (m1_time_ratio >= 1)',
    'b:c3': 'B-wave of a flat correction: the b-wave of a flat (not a zigzag); m1_retrace_ratio near 1.0 with rule_of_equality_satisfied likely true',
    ':F3': 'First 3-wave: head of a pattern, right after x:c3 or between two :5s; first segment of complex corrections or corrections between two impulse waves; often m1/m2 are null (pattern head)',
    'x:c3': 'X-wave position: central 3-wave connecting two standard corrections; same degree as adjacent corrections, must consume less time and be less complex than the corrections before and after it (m1_time_ratio < 1)',
    ':L5': 'Last 5-wave: final wave of a larger pattern; the label right before it must be :F3 or :c3; ends a trending impulse or terminal pattern; often m_1/m_2 are null (pattern end)',
    ':L3': 'Last 3-wave: final 3-segment wave ending a triangle or terminal made of five :3s; if smallest of the five, must be completely retraced within the time it took to form; often m_1/m_2 are null (pattern end)',
    ':sL3': 'Second-to-last 3-wave: always followed by :L3; part of a triangle or terminal pattern made of five consecutive :3s; never starts or ends a pattern',
    'Invalid_Neely_Rule_Violation': 'No valid Neely structure: the monowave sequence violates Neely rules (e.g. m1_retrace_ratio and m2_retrace_ratio both outside valid ranges, rule_of_equality_satisfied false where required, channel_break inconsistent, or overlapping/irregular retracements that fit no SP)',
}


def round3(v):
    if v is None:
        return None
    return float(f"{v:.3f}")


def build_feature(i, waves, ohlc):
    nw = waves[i]
    m0_dir = "up" if nw["endPrice"] > nw["startPrice"] else "down"
    m0_length = abs(nw["endPrice"] - nw["startPrice"])
    m0_change_pct = ((nw["endPrice"] - nw["startPrice"]) / nw["startPrice"]) * 100 if nw["startPrice"] != 0 else 0
    m0_time_bars = nw["end"] - nw["start"] + 1

    def get_wave(j):
        if j < 0 or j >= len(waves):
            return None
        w = waves[j]
        seg = ohlc[w["start"]:w["end"] + 1]
        return {
            "dir": "up" if w["endPrice"] > w["startPrice"] else "down",
            "length": abs(w["endPrice"] - w["startPrice"]),
            "timeBars": w["end"] - w["start"] + 1,
            "high": max(d["high"] for d in seg),
            "low": min(d["low"] for d in seg),
        }

    w1 = get_wave(i - 1)
    w2 = get_wave(i - 2)
    w_1 = get_wave(i + 1)
    w_2 = get_wave(i + 2)

    m1 = None
    if w1:
        m1 = {
            "dir": w1["dir"], "length": w1["length"], "timeBars": w1["timeBars"],
            "retraceRatio": (w1["length"] / m0_length) if m0_length != 0 else 0,
            "timeRatio": (w1["timeBars"] / m0_time_bars) if m0_time_bars != 0 else 0,
        }
    m2 = None
    if w2:
        m2 = {
            "dir": w2["dir"], "length": w2["length"], "timeBars": w2["timeBars"],
            "retraceRatio": (w2["length"] / w1["length"]) if (w1 and w1["length"] != 0) else 0,
        }
    m_1 = {"dir": w_1["dir"], "length": w_1["length"], "timeBars": w_1["timeBars"]} if w_1 else None
    m_2 = {"dir": w_2["dir"], "length": w_2["length"], "timeBars": w_2["timeBars"]} if w_2 else None

    seg0 = ohlc[nw["start"]:nw["end"] + 1]
    m0_high = max(d["high"] for d in seg0)
    m0_low = min(d["low"] for d in seg0)
    m0_overlap_m2 = (m0_low < w2["high"] and m0_high > w2["low"]) if w2 else False
    rule_of_equality = (abs(w1["length"] - w2["length"]) / max(w1["length"], w2["length"]) <= 0.15) if (w1 and w2) else False
    channel_break = (m0_high > max(w1["high"], w2["high"]) or m0_low < min(w1["low"], w2["low"])) if (w1 and w2) else False

    return {
        "m0_dir": m0_dir, "m0_length": round3(m0_length), "m0_change_pct": round3(m0_change_pct),
        "m0_time_bars": m0_time_bars,
        "m1_dir": m1["dir"] if m1 else None, "m1_length": round3(m1["length"]) if m1 else None,
        "m1_time_bars": m1["timeBars"] if m1 else None,
        "m1_retrace_ratio": round3(m1["retraceRatio"]) if m1 else None,
        "m1_time_ratio": round3(m1["timeRatio"]) if m1 else None,
        "m2_dir": m2["dir"] if m2 else None, "m2_length": round3(m2["length"]) if m2 else None,
        "m2_time_bars": m2["timeBars"] if m2 else None,
        "m2_retrace_ratio": round3(m2["retraceRatio"]) if m2 else None,
        "m_1_dir": m_1["dir"] if m_1 else None, "m_1_length": round3(m_1["length"]) if m_1 else None,
        "m_1_time_bars": m_1["timeBars"] if m_1 else None,
        "m_2_dir": m_2["dir"] if m_2 else None, "m_2_length": round3(m_2["length"]) if m_2 else None,
        "m_2_time_bars": m_2["timeBars"] if m_2 else None,
        "m0_overlap_m2": m0_overlap_m2, "rule_of_equality_satisfied": rule_of_equality,
        "channel_break": channel_break,
    }


def call(url, state, model):
    payload = {"model": model, "state": state,
               "questions": {"sp": {"type": "choice", "instructions": INSTRUCTIONS,
                                     "criteria": SP_CRITERIA, "choices": list(SP_CHOICES)}}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read())
    dt = (time.perf_counter() - t0) * 1000
    ans = out.get("answers", {}).get("sp", {})
    return ans.get("choice"), ans.get("confidence"), dt


def main():
    d = json.load(open("/tmp/eth_wave.json"))
    waves = d["data"]["monoWaves"]
    ohlc = d["data"]["ohlc"]
    n = len(waves)
    print(f"ETH monowaves: {n}\n")

    rows = []
    for i in range(n):
        state = build_feature(i, waves, ohlc)
        mlp_c, mlp_conf, mlp_ms = call(MLP, state, "sp-mlp")
        oll_c, oll_conf, oll_ms = call(OLLAMA, state, "tev1-neely:latest")
        match = (mlp_c == oll_c)
        rows.append((i, waves[i]["sp"], mlp_c, oll_c, match, mlp_ms, oll_ms))
        flag = "OK " if match else "DIFF"
        print(f"[{flag}] wave {i:2d} app_sp={waves[i]['sp']:<32} mlp={str(mlp_c):<32} ollama={str(oll_c)}")

    total = len(rows)
    mlp_ms = [r[5] for r in rows]
    oll_ms = [r[6] for r in rows]
    mlp_ms.sort(); oll_ms.sort()

    # app_sp may be a multi-candidate label like ":F3/(:s5)/:c3" — normalize to a set
    def app_set(sp):
        import re
        toks = re.split(r'[/\[\]()\s]+', sp)
        return {t for t in toks if t}

    mlp_hit_app = 0
    oll_hit_app = 0
    mlp_hit_oll = 0
    for r in rows:
        i, app_sp, mlp_c, oll_c, _, _, _ = r
        aset = app_set(app_sp)
        if mlp_c in aset:
            mlp_hit_app += 1
        if oll_c in aset:
            oll_hit_app += 1
        if mlp_c == oll_c:
            mlp_hit_oll += 1

    print(f"\n=== A/B SUMMARY (ETH 1D, {total} waves) ===")
    print(f"mlp == ollama (raw):        {mlp_hit_oll}/{total} = {mlp_hit_oll/total:.4f}")
    print(f"mlp in app_sp candidates:   {mlp_hit_app}/{total} = {mlp_hit_app/total:.4f}")
    print(f"ollama in app_sp candidates:{oll_hit_app}/{total} = {oll_hit_app/total:.4f}")
    print(f"MLP    p50={mlp_ms[total//2]:.1f}ms  mean={sum(mlp_ms)/total:.1f}ms")
    print(f"Ollama p50={oll_ms[total//2]:.1f}ms  mean={sum(oll_ms)/total:.1f}ms")

    # save full results
    with open("/tmp/ab_eth_results.json", "w") as fh:
        json.dump([{"i": r[0], "app_sp": r[1], "mlp": r[2], "ollama": r[3],
                   "mlp_ms": r[5], "oll_ms": r[6]} for r in rows], fh, indent=2)
    print("\nsaved /tmp/ab_eth_results.json")


if __name__ == "__main__":
    main()
