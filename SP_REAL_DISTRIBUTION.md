# Real-world Neely SP symbol distribution (measured, not assumed)

Date: 2026-11-13
Author: analysis for the Tev1 SP decision model (`tev1-neely:latest`)

> **Status (2026-11-14): the distribution prior is applied.** It now drives the label-first
> reweighting in `stock-twin-kr/scripts/generate-sft-dataset.ts`. The `Invalid` problem
> (§4, point 1) was fixed and audited (`audit_invalid_labels.py`, `PIPELINE.md` §2.1). The
> **task-mix** mismatch (§5) is still out of scope (chosen scope: reweight the synthetic
> generator only, no real-data training pipeline). See `ACCURACY_ANALYSIS.md` §0.

## TL;DR

1. **No published empirical frequency table exists** for Neely SP symbols
   (`:5`, `:c3`, `:F3`, `x:c3`, ...). The notation is proprietary to Glenn
   Neely's *Mastering Elliott Wave* (1990); the literature only gives
   *qualitative* rarity statements (see §1).
2. **But we don't need the literature.** The app already ingests real SP labels
   from the production data source — the Kiwoom broker API — and we measured
   them: **~77,000 real labeled monowaves** across 400 KOSPI/KOSDAQ symbols and
   3 timeframes (D/W/M). See §2–§3.
3. **Verdict: the current synthetic training distribution is NOT appropriate.**
   The two biggest problems:
   - **`Invalid_Neely_Rule_Violation` is 34.5% of training but ≈0% in reality.**
     The real source never emits it. This alone explains the Invalid
     over-prediction observed in production.
   - **Task mismatch:** in production only ~20% of waves are free 10-way
     classifications. ~40% are *candidate-set disambiguation* and ~40% need no
     model call at all. The SFT dataset trains 100% free classification.
   See §4–§5.

---

## 1. Web research: what the literature does and does not tell us

Searches run: Neely SP symbol meanings, Elliott-wave impulse-vs-correction
frequency, triangle/zigzag/flat relative frequency, NeoWave rarity statements.

Findings:

- **No authoritative table.** Neely's `:5/:c3/:F3/:L5/:s5/:sL3/:L3/x:c3/b:c3`
  labels are not tabulated by frequency anywhere public. The closest public
  artifacts are third-party TradingView indicators that re-implement the
  labeling rules (e.g. "Structura Wave Engine"), not frequency studies.
- **Qualitative statements only:**
  - `:s5` — "usually an element of a *complex* pattern, in **rare** occasions
    the third segment of a trend impulse" (rare).
  - `x:c3` — the connecting wave between standard patterns; triangles appear in
    X/Z position; "except for **rare** occasions, [a wave] can't have a more
    complex structure than the preceding and proceeding patterns".
  - `:sL3` / `:L3` — "always indicate either a **triangle** or a **terminal**
    pattern forming" → tied to the rarer structures.
  - `:L5` — "always a **conclusion** of a larger pattern".
  - General Elliott lore: zigzag = most common sharp correction; flats common
    (expanding flats most common, regular flats rarer); triangles mainly in
    wave 4 and X/Z; "expanding triangles are extremely rare within a complex
    combination".
  - Corrections are commonly said to occupy ~2/3 of the time.
- **Caveat / criticism:** Batchelor & Ramyar found no significant Fibonacci
  ratios in the DJIA. If price doesn't follow Elliott rules, the "true" SP
  distribution is only defined *relative to a given labeling convention* — which
  is exactly why the broker's own labels are the right ground truth.

**Conclusion of §1:** the literature can only sanity-check *ordering* (zigzag
common, triangles/terminals rare). It cannot supply the numbers. The numbers
have to be measured — and they can be.

---

## 2. Where the real labels come from (code path)

```
elliott-wave.service.ts
  getChartHistoricalData()          -> fetch `${INTERNAL_API_BASE_URL}/api/stock-chart?code=..&indc_tp=..`
  /api/stock-chart/route.ts         -> proxies `${KIWOOM_BROKER_API}/stock_chart/{code}?base_dt=..&indc_tp=..`
  response.macro_wave[].sp          <- the REAL Neely SP label (broker's engine)
  createMonoWavesFromNeely()        -> DP interval scheduling -> main waves -> features
  parseSPCandidates()               -> splits "A/B/(C)/[D]" candidate list, rarity tags
```

Key facts:

- `sp` is a **slash-separated candidate list**, not a single label:
  `":F3/:c3"`, `":c3/(:F3)"`, `":F3/[:L5]"`, `""` (blank).
- `(x)` = **rare**, `[x]` = **very rare** (the vendor's own rarity encoding —
  it is literally an annotation of relative frequency).
- The app's own fallback labeler `assignSPToMonoWaves` only ever emits
  `:5`/`:c3`/`:F3`, so **it must not be used to measure the real distribution.**
- Real labels also include `RN` (reversal) and `:?○:?` (uncertain), which are
  **not** in `SP_CHOICES`.

**Production decision mix** (`createMonoWavesFromNeely`, lines ~1414-1460):

| source SP            | model behaviour                         |
|----------------------|-----------------------------------------|
| single candidate     | **no model call** — label used directly |
| multi candidate      | model **disambiguates** within the set  |
| blank                | model **classifies freely** (10-way)    |

---

## 3. Measured real distribution

Harness: `~/tev1-finetune/measure_sp_distribution.py`
Raw output: `~/tev1-finetune/sp-real-distribution.json`
Sample: 400 symbols evenly drawn from 2,735 (KOSPI 926 / KOSDAQ 1,809),
timeframes D/W/M, Kiwoom broker `macro_wave`.

### 3.1 Decision mix (the prior that actually matters for training)

| timeframe | single (no call) | multi (disambiguate) | blank (free) | waves |
|-----------|------------------|----------------------|--------------|-------|
| Daily     | **40.2%**        | **40.1%**            | **19.7%**    | 41,638 |
| Weekly    | 38.8%            | 37.3%                | 24.0%        | 21,065 |
| Monthly   | 37.0%            | 36.0%                | 27.0%        | 14,410 |

Candidate-set size (daily): 1 → 40.2%, 2 → 31.5%, 3+ → 8.6%, blank → 19.7%.
KOSPI vs KOSDAQ decision mix is essentially identical.

### 3.2 SP symbol distribution

**All candidate mentions** (a wave with `:F3/:c3` contributes to both; sums >100%):

| symbol   | Daily  | Weekly | Monthly |
|----------|--------|--------|---------|
| `:F3`    | 38.44% | 36.46% | 35.35%  |
| `:c3`    | 23.86% | 23.41% | 23.16%  |
| `:5`     | 23.83% | 21.95% | 21.88%  |
| `:L5`    | 14.93% | 13.75% | 12.07%  |
| `:s5`    |  9.82% |  9.29% |  8.47%  |
| `x:c3`   |  8.05% |  7.60% |  7.08%  |
| `:sL3`   |  4.06% |  3.52% |  3.34%  |
| `b:c3`   |  3.41% |  3.36% |  3.12%  |
| `:L3`    |  3.06% |  2.16% |  2.20%  |
| `RN`     |  0.24% |  0.12% |  0.05%  |
| `:?○:?`  |  0.22% |  0.25% |  0.18%  |
| **`Invalid_Neely_Rule_Violation`** | **0%** | **0%** | **0%** |

**Primary (first) candidate** (daily): `:F3` 33.7%, `:5` 19.8%, `:L5` 12.0%,
`:c3` 9.2%, `x:c3` 3.4%, `:sL3` 0.7%, `:s5` 0.6%, `:L3` 0.5%, `RN` 0.24%,
`:?○:?` 0.08%.

**Rarity tags:** normal 93.2%, rare 5.0%, veryRare 1.9% (stable across D/W/M).

The ordering is stable across timeframes and consistent with the literature's
qualitative claims (zigzag/flat common; `:s5`, `:sL3`/`:L3` rarer; x-waves are
the connectors).

---

## 4. Synthetic prior vs real prior

Synthetic `sp-systemone-train.jsonl` (20,000, labels shown as letters A–J):

| label (letter) | synthetic | real (mentions, D, normalized to valid) |
|----------------|-----------|------------------------------------------|
| `Invalid` (J)  | **34.55%**| **0%**                                   |
| `:c3` (C)      | 20.16%    | 18.4%                                    |
| `:F3` (E)      | 15.90%    | 29.7%                                    |
| `:L5` (G)      | 14.36%    | 11.5%                                    |
| `:5` (A)       |  6.17%    | 18.4%                                    |
| `b:c3` (D)     |  3.76%    |  2.6%                                    |
| `:sL3` (I)     |  2.00%    |  3.1%                                    |
| `:L3` (H)      |  1.89%    |  2.4%                                    |
| `x:c3` (F)     |  0.66%    |  6.2%                                    |
| `:s5` (B)      |  0.56%    |  7.6%                                    |

(synthetic normalized over its valid labels only, to make the two comparable)

**Mismatches, in order of severity:**

1. **`Invalid` 34.5% vs 0%.** Catastrophic. The model is explicitly trained to
   answer Invalid a third of the time; the real data never uses it. This is the
   single best explanation for the Invalid over-prediction.
2. **`x:c3` 0.7% vs 6.2%** (~9× under-represented).
3. **`:s5` 0.6% vs 7.6%** (~13× under-represented).
4. **`:5` 6.2% vs 18.4%** (~3× under-represented).
5. **`:L5` 14.4% vs 11.5%**, **`:c3` 20.2% vs 18.4%** — modestly over.
6. **Task structure:** synthetic is 100% free 10-way; real is 40% no-call /
   40% disambiguation / 20% free.
7. **Out-of-vocabulary real labels** (`RN` 0.24%, `:?○:?` 0.22%) are absent
   from `SP_CHOICES`.

The generator's *structural* choices are otherwise sane (impulse/correction
mix, zigzag ≥ flat > triangle ordering, `:sL3` only before `:L3`, etc.) — it is
the **frequencies** and the **Invalid prior** that are wrong, not the rule
modeling.

---

## 5. Recommended actions

**Preferred — train on real data.** The broker `macro_wave` already provides
tens of thousands of real, correctly-structured `(monowave features, SP)` pairs
per timeframe, and it is the exact distribution production sees. Build the SFT
set by running the app's own `createMonoWavesFromNeely` (DP + `buildMonowaveFeatures`)
over a symbol universe and keeping the candidate-set structure:

- free-classification examples from `blank` waves (all 10 options),
- disambiguation examples from multi-candidate waves (**restricted options =
  the candidate set**),
- optionally skip single-candidate waves (no model call in production).

This removes the Invalid prior entirely and matches the real label frequencies
by construction. Scaling: 2,735 symbols × 5 timeframes is cheap (the 400-symbol
D/W/M sweep took ~110 s).

**Fallback — reweight the synthetic prior** if you want to keep the synthetic
generator: drop `INVALID_WEIGHT` from 35 → ~2–5% (or 0), and rebalance so the
valid-label shares track §3.2 (notably raise `:5`, `:s5`, `x:c3`; lower `:c3`,
`:L5`, `b:c3`). `scripts/generate-sft-dataset.ts` already supports weights.

**Regardless of choice:**

- Evaluate with **per-class recall and macro-F1**, not just accuracy (the class
  imbalance otherwise hides the rare-class failures). `finetune_tev1_sp.py` and
  `eval/eval_merged_sp.py` already print per-class recall — extend to macro-F1.
- Consider **inference-time prior correction**: the production decision is often
  restricted to a candidate set, so scoring should be computed *within the
  candidates*, not over all 10 options. The app already does this
  (`resolveSPWithModel`); make sure training matches it.
- Decide how to handle `RN` / `:?○:?` (map to `Invalid`? add to `SP_CHOICES`?).

---

## 6. Reproduce

```bash
# universe (from stock_master)
docker exec kil2_container bash -lc 'mysql -h$MYSQL_HOST -u$MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE -N \
  -e "SELECT stock_code, market FROM stock_master WHERE stock_code REGEXP \"^[0-9]{6}$\";"' \
  2>/dev/null | awk '{print $1"\t"$2}' > /tmp/opencode/sp_universe.tsv

cd ~/tev1-finetune
python3 measure_sp_distribution.py --limit 400 --timeframes D W M \
  --concurrency 10 --out sp-real-distribution.json
```
