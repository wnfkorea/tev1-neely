#!/usr/bin/env python3
"""
Measure the REAL-WORLD Neely SP symbol distribution from the production data
source used by stock-twin-kr.

The app never computes SP labels from scratch for real symbols: it reads them
from the Kiwoom broker API (`KIWOOM_BROKER_API`, default http://localhost:8000)
via /stock_chart/{code}. The response carries a `macro_wave` array whose `sp`
field is the broker's Neely structure/progression label. That label is a
slash-separated CANDIDATE list, with rarity annotations:
    ":F3/:c3"        -> two normal candidates
    ":c3/(:F3)"      -> first normal, second RARE   (...)
    ":F3/[:L5]"      -> first normal, second VERY RARE  [...]
    ""               -> whitespace: the tev1 model classifies freely

So the production decision mix is:
    single candidate  -> NO model call (label taken directly)
    multi candidate   -> model DISAMBIGUATES within the candidate set
    blank             -> model classifies FREELY over all SP_CHOICES

This script samples real symbols x timeframes and reports:
  1. the decision-type mix (single / multi / blank) -> the real training prior
  2. the candidate-set size distribution (disambiguation arity)
  3. the SP symbol distribution (per candidate mention, and per primary)
  4. the rarity annotation distribution
  5. KOSPI vs KOSDAQ split

Usage:
  python3 measure_sp_distribution.py --limit 250 --timeframes D W M \
      --universe /tmp/opencode/sp_universe.tsv --out sp-real-distribution.json
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import date

BASE = os.environ.get("KIWOOM_BROKER_API", "http://192.168.55.229:8000").rstrip("/")
BASE_DT = date.today().strftime("%Y%m%d")
SP_CHOICES = [':5', ':s5', ':c3', 'b:c3', ':F3', 'x:c3', ':L5', ':L3', ':sL3',
              'Invalid_Neely_Rule_Violation']


def fetch_macro_wave(code, indc_tp, timeout=30):
    url = f"{BASE}/stock_chart/{code}?base_dt={BASE_DT}&indc_tp={indc_tp}"
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data.get("macro_wave") or []


def parse_sp(raw):
    """Return (kind, candidates) where kind in {blank, label}.
    candidates: list of (symbol, rarity) with rarity in {normal, rare, veryRare}."""
    if raw is None:
        return "blank", []
    s = str(raw).strip()
    if s == "":
        return "blank", []
    cands = []
    for p in s.split("/"):
        p = p.strip()
        if not p:
            continue
        rarity = "normal"
        if len(p) >= 2 and p[0] == "(" and p[-1] == ")":
            rarity, p = "rare", p[1:-1].strip()
        elif len(p) >= 2 and p[0] == "[" and p[-1] == "]":
            rarity, p = "veryRare", p[1:-1].strip()
        if p:
            cands.append((p, rarity))
    if not cands:
        return "blank", []
    return "label", cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="/tmp/opencode/sp_universe.tsv")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--timeframes", nargs="+", default=["D"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="sp-real-distribution.json")
    ap.add_argument("--stride", type=int, default=None,
                    help="sample every Nth symbol (default: even spread)")
    args = ap.parse_args()

    rows = []
    with open(args.universe) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                rows.append((parts[0], parts[1]))
    if args.stride:
        universe = rows[::args.stride][: args.limit]
    else:
        step = max(1, len(rows) // args.limit)
        universe = rows[::step][: args.limit]

    print(f"[measure] base={BASE} base_dt={BASE_DT} "
          f"symbols={len(universe)} timeframes={args.timeframes}", file=sys.stderr)

    # per-timeframe aggregation
    per_tf = {tf: {
        "waves": 0, "symbols_ok": 0, "symbols_err": 0,
        "decision": Counter(),          # single / multi / blank
        "set_size": Counter(),          # 1 / 2 / 3+
        "primary": Counter(),           # first candidate
        "mentions": Counter(),          # every distinct candidate symbol
        "mentions_multi_only": Counter(),
        "single_only": Counter(),       # labels of single-candidate waves (certain)
        "tiebreak": defaultdict(float), # uniform tie-break marginal (1/n per candidate)
        "rarity": Counter(),
        "market": defaultdict(Counter),  # market -> decision
        "by_symbol_waves": 0,
    } for tf in args.timeframes}

    tasks = [(code, mkt, tf) for (code, mkt) in universe for tf in args.timeframes]

    def work(t):
        code, mkt, tf = t
        try:
            return t, fetch_macro_wave(code, tf), None
        except Exception as e:
            return t, None, str(e)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            (code, mkt, tf), mw, err = fut.result()
            done += 1
            agg = per_tf[tf]
            if err or mw is None:
                agg["symbols_err"] += 1
                continue
            agg["symbols_ok"] += 1
            for w in mw:
                kind, cands = parse_sp(w.get("sp"))
                agg["waves"] += 1
                if kind == "blank":
                    agg["decision"]["blank"] += 1
                    agg["set_size"]["0"] += 1
                    agg["market"][mkt]["blank"] += 1
                else:
                    n = len(cands)
                    agg["decision"]["single" if n == 1 else "multi"] += 1
                    agg["set_size"][str(n) if n < 3 else "3+"] += 1
                    agg["market"][mkt]["single" if n == 1 else "multi"] += 1
                    agg["primary"][cands[0][0]] += 1
                    if n == 1:
                        agg["single_only"][cands[0][0]] += 1
                    seen = set()
                    for sym, rar in cands:
                        agg["rarity"][rar] += 1
                        if sym not in seen:
                            agg["mentions"][sym] += 1
                            if n > 1:
                                agg["mentions_multi_only"][sym] += 1
                            seen.add(sym)
                    for sym in seen:
                        agg["tiebreak"][sym] += 1.0 / len(seen)
            if done % 50 == 0:
                print(f"[measure] {done}/{len(tasks)} fetched", file=sys.stderr)

    # ---- report ----
    def pct(c, tot):
        return round(100.0 * c / tot, 2) if tot else 0.0

    report = {"base_url": BASE, "base_dt": BASE_DT,
              "universe_symbols": len(universe), "timeframes": args.timeframes,
              "results": {}}
    for tf, agg in per_tf.items():
        tot = agg["waves"]
        res = {
            "symbols_ok": agg["symbols_ok"], "symbols_err": agg["symbols_err"],
            "waves": tot,
            "decision_mix": {k: {"n": v, "pct": pct(v, tot)}
                             for k, v in agg["decision"].most_common()},
            "set_size": {k: {"n": v, "pct": pct(v, tot)}
                         for k, v in sorted(agg["set_size"].items())},
            "primary_distribution": {k: {"n": v, "pct": pct(v, tot)}
                                     for k, v in agg["primary"].most_common()},
            "candidate_mentions": {k: {"n": v, "pct": pct(v, tot)}
                                   for k, v in agg["mentions"].most_common()},
            "multi_mentions": {k: {"n": v, "pct": pct(v, tot)}
                               for k, v in agg["mentions_multi_only"].most_common()},
            "single_only_distribution": {k: {"n": v, "pct": pct(v, sum(agg["single_only"].values()))}
                                         for k, v in agg["single_only"].most_common()},
            "tiebreak_marginal": {k: {"n": round(v, 1), "pct": pct(v, tot)}
                                  for k, v in sorted(agg["tiebreak"].items(), key=lambda kv: -kv[1])},
            "rarity": {k: {"n": v, "pct": pct(v, sum(agg["rarity"].values()))}
                       for k, v in agg["rarity"].most_common()},
            "by_market": {m: dict(c) for m, c in agg["market"].items()},
        }
        report["results"][tf] = res

        print(f"\n================= timeframe {tf} =================")
        print(f"symbols ok/err: {agg['symbols_ok']}/{agg['symbols_err']}   waves: {tot}")
        print("decision mix (real training prior):")
        for k, v in agg["decision"].most_common():
            print(f"  {k:7s} {v:7d}  {pct(v, tot):6.2f}%")
        print("candidate set size:")
        for k, v in sorted(agg["set_size"].items()):
            print(f"  {k:>2s} {v:7d}  {pct(v, tot):6.2f}%")
        print("primary (first) candidate:")
        for k, v in agg["primary"].most_common():
            print(f"  {k:28s} {v:7d}  {pct(v, tot):6.2f}%")
        print("all candidate mentions:")
        for k, v in agg["mentions"].most_common():
            print(f"  {k:28s} {v:7d}  {pct(v, tot):6.2f}%")
        print("single-candidate only (certain labels):")
        st = sum(agg["single_only"].values())
        for k, v in agg["single_only"].most_common():
            print(f"  {k:28s} {v:7d}  {pct(v, st):6.2f}%")
        print("rarity annotations:")
        rt = sum(agg["rarity"].values())
        for k, v in agg["rarity"].most_common():
            print(f"  {k:9s} {v:7d}  {pct(v, rt):6.2f}%")

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[measure] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
