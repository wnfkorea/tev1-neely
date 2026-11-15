# Tev1 → Neely (SP) decision model — fine-tune & Ollama registration

> This is the **detailed pipeline / provenance** document. For the repository overview and
> setup, see [`README.md`](README.md); for a hands-on follow-along walkthrough, see
> [`docs/LECTURE.md`](docs/LECTURE.md).

How the `tev1-neely` System-1 **decision model** was produced: a **bf16 LoRA** fine-tune
of `togethercomputer/Tev1-4B-experimental` on a synthetic Neely wave-count (SP symbol)
dataset, merged to bf16, exported to GGUF, given `pplx-decider` decision metadata,
quantized to Q8_0, and registered with Ollama.

The training data is **synthesised** by `generate_sft_dataset.py` (a standalone, stdlib-only
port of the original `stock-twin-kr/scripts/generate-sft-dataset.ts`),
reweighted to the **measured** real SP prior (`SP_REAL_DISTRIBUTION.md`), and every
`Invalid_Neely_Rule_Violation` label is checked by a generation-time audit gate
(`audit_invalid_labels.py`). See §2–§2.2.

The model answers the **SP symbol** question of the Elliott-wave pipeline via Ollama's
`/v1/systemone` endpoint (it is a *decision* model, **not** a text generator).

---

## 0. Prerequisites

| Item | Path / value |
|---|---|
| Base model | `togethercomputer/Tev1-4B-experimental` (Qwen3.5-4B hybrid: linear-attention + full-attention + MTP) |
| HF cache snapshot | `~/.cache/huggingface/hub/models--togethercomputer--Tev1-4B-experimental/snapshots/0b7becf017daa0e5eb222f8ce7483c8c8259c52f` |
| Python env | `~/tev1-finetune/venv` (torch, transformers, peft, bitsandbytes, datasets, triton, fla, gguf, numpy, tqdm) |
| llama.cpp | `~/tev1-finetune/llama.cpp` — used for `convert_hf_to_gguf.py` and the `gguf-py` package |
| Data generator | `generate_sft_dataset.py` — standalone, stdlib-only (no Node.js); CPU only |
| Real SP prior | `SP_REAL_DISTRIBUTION.md` + `sp-real-distribution.json` (measured from the Kiwoom broker `macro_wave` via `measure_sp_distribution.py`) |
| GPU | bf16 LoRA training on the 24 GB 4090; merge/convert/quantize are CPU-friendly. Torch device order is `CUDA_VISIBLE_DEVICES=0` → **4090**, `=1` → 4060 Ti (opposite of `nvidia-smi`). |

### llama.cpp patch (required)

`llama.cpp/conversion/qwen.py` — `Qwen3_5TextModel._SHARED_INPUT_MERGES` must be emptied so
the exported GGUF keeps **separate** `attn_qkv` / `attn_gate` and `ssm_alpha` / `ssm_beta`
tensors. This matches the deployed `tev1` GGUF layout the llama.cpp server loader expects.

```python
class Qwen3_5TextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35
    _SHARED_INPUT_MERGES = ()
```

---

## 1. Pipeline overview

```
generate_sft_dataset.py ──→ sft-dataset-train4.jsonl / sft-dataset-heldout4.jsonl
                            │  (label-first, measured real prior; Invalid gated)
                            └─ build_sp_systemone.py ─→ sp-systemone-{train,heldout}.jsonl

sp-systemone-train.jsonl ── finetune_tev1_sp.py ─→ output/tev1-sp-lora-v4   (LoRA adapter, bf16)
output/tev1-sp-lora-v4 ──── merge_tev1_sp.py ────→ merged-tev1-sp-bf16-v4   (standalone HF bf16)
merged-tev1-sp-bf16-v4 ──── convert_hf_to_gguf.py → tev1-neely-sp-bf16-v4-f16.gguf
tev1-neely-sp-bf16-v4-f16.gguf ── inject_decision_metadata.py → …-f16-decision.gguf
…-f16-decision.gguf ─────── quantize_q8.py ──────→ tev1-neely-sp-bf16-v4-q8_0-decision.gguf
tev1-neely-sp-bf16-v4-q8_0-decision.gguf ─ ollama create -f Modelfile-tev1-neely-v4 → tev1-neely-v4
```

> **Do not `ollama create` the `…-f16.gguf` / `…-f16-decision.gguf` intermediates.** Point
> `FROM` at the **Q8_0** file. Feeding Ollama an f16 GGUF can fail with
> `Error: unsupported file type [123 10 32 32]` (see §8).

---

## 2. Generate the SFT dataset

The SFT dataset is **synthetic**, produced by the standalone `generate_sft_dataset.py`
(stdlib-only; a faithful port of the original `stock-twin-kr/scripts/generate-sft-dataset.ts`
— the same seed reproduces the committed files byte for byte):

```bash
cd ~/tev1-finetune
./venv/bin/python generate_sft_dataset.py --count 20000 --seed 20261113 --out sft-dataset-train4.jsonl
./venv/bin/python generate_sft_dataset.py --count 3000  --seed 424242   --out sft-dataset-heldout4.jsonl
```

Each row is `{state, question, answer}`, where `answer` is one of the 10 SP keys and `state`
is the standardised monowave feature vector of the target wave (m0) plus its ±2 neighbours.
Sampling is **label-first**: the target label is drawn from the **measured real prior**
(`SP_REAL_DISTRIBUTION.md`) and a matching host pattern/position is then generated, so the
marginal label distribution matches production (pattern weights alone cannot — every pattern
starts with `:F3`, `:5` only occurs in impulse/complex, etc.).

* `SFT_INVALID_WEIGHT` (env, default **3** %) — share of `Invalid_Neely_Rule_Violation`
  samples. Real `macro_wave` data never emits it (0 / ~77 k), so `SFT_INVALID_WEIGHT=0`
  matches production exactly; a small hedge keeps the reject class learnable.
* Seeds: train `20261113` (20 000 rows), held-out `424242` (3 000 rows); the two splits are
  disjoint (0 overlap on `state`+`answer`).

### 2.1 Invalid-label audit (generation-time gate)

`Invalid_Neely_Rule_Violation` is a project-invented 10th "reject" class, not a real Neely
SP. An audit of an earlier dataset found the labels were largely **unlearnable**:

| metric | broken (train3) | fixed (train4) |
|---|---|---|
| claimed violation visible in the target feature vector | 28.3 % | **100 %** |
| 1-NN label of an Invalid sample is Invalid | 95.6 % | **100 %** |
| ambiguous (a valid sample is closer than the nearest Invalid) | 4.4 % | **0 %** |
| valid samples matching the Invalid signature (false positives) | 0 % | 0 % |

Two defects caused it: (1) the violating wave sat at index 2 but the target index was picked
at random from {1,2,3}, so ~67 % of Invalid rows labelled a **valid** `:c3` wave; (2) the
"violation" was often invisible in the target's local features. The generator now **fixes the
target to the violating wave** and builds violations **outside** the measured valid bands
(valid samples always alternate; `m1_retrace_ratio` ∈ [0.18, 6.60]; `m1_time_ratio` ∈
[0.25, 3.00]):

| type | violation | signal |
|---|---|---|
| 0 | non-alternation | `m1_dir == m0_dir` (never true for valid) |
| 1 | extreme retrace | `m1_retrace_ratio ≈ 13–60` |
| 2 | extreme time ratio | `m1_time_ratio ≈ 0.03–0.09` |
| 3 | compound | non-alternation + extreme retrace |

A gate (`isInvalidSignature` in the generator) drops any Invalid sample whose violation is not
visible in the feature vector. Re-run the regression audit after any change:

```bash
cd ~/tev1-finetune
./venv/bin/python audit_invalid_labels.py sft-dataset-train4.jsonl   # -> RESULT: PASS
```

### 2.2 Build the systemone dataset

Converts the SFT rows into decision records
(`{state, question, options, answer: "A".."J", answer_key: ":c3", prompt}`):

```bash
cd ~/tev1-finetune
./venv/bin/python build_sp_systemone.py \
  --train sft-dataset-train4.jsonl \
  --heldout sft-dataset-heldout4.jsonl \
  --out-prefix sp-systemone
# -> sp-systemone-train.jsonl (20000), sp-systemone-heldout.jsonl (3000)
```

* The prompt is rendered with the **exact** systemone template injected into the GGUF
  (`systemone-template.txt`) via `tokenizer.apply_chat_template`, so it is byte-identical to
  what `/v1/systemone` feeds the model at inference — including the `<think>` / `</think>`
  added tokens.
* The 10 SP choices map to letters **A–J** in this fixed order:
  `:5 :s5 :c3 b:c3 :F3 x:c3 :L5 :L3 :sL3 Invalid_Neely_Rule_Violation`.

---

## 3. Fine-tune (bf16 LoRA)

```bash
cd ~/tev1-finetune
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python finetune_tev1_sp.py \
  --train sp-systemone-train.jsonl \
  --heldout sp-systemone-heldout.jsonl \
  --output-dir output/tev1-sp-lora-v4 \
  --max-seq-len 1280 --batch-size 1 --grad-accum 16 \
  --epochs 1 --lr 2e-5 --eval-samples 300 \
  --no-grad-checkpointing --optim adamw_torch_fused \
  --precision bf16
```

* `--precision bf16` (recommended) keeps the adapter consistent with the merged/deployed
  bf16 artifact; `--precision qlora` is the legacy 4-bit nf4 path (leaves an nf4→bf16
  fidelity gap).
* LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target_modules="all-linear"`, completion-only
  loss; ~38.99 M trainable params (0.85 %). Training input is the pre-rendered `prompt`; the
  target is the answer letter + `<|im_end|>`.
* `--no-grad-checkpointing --optim adamw_torch_fused` fit on the 24 GB 4090; drop
  `--no-grad-checkpointing` if you OOM.
* Output: `output/tev1-sp-lora-v4/` (adapter + tokenizer).

---

## 4. Merge LoRA → standalone HF model

```bash
cd ~/tev1-finetune
ADAPTER_DIR=output/tev1-sp-lora-v4 MERGED_DIR=merged-tev1-sp-bf16-v4 ./venv/bin/python merge_tev1_sp.py
# -> merged-tev1-sp-bf16-v4/  (bf16, ~8.5 GB, 2 shards)
```

`merge_tev1_sp.py` loads the base in bf16 on CPU, applies the adapter with `PeftModel`,
and calls `merge_and_unload()`.

---

## 5. Export to GGUF (text-only f16)

```bash
cd ~/tev1-finetune
./venv/bin/python llama.cpp/convert_hf_to_gguf.py merged-tev1-sp-bf16-v4 \
  --outfile tev1-neely-sp-bf16-v4-f16.gguf --outtype f16
```

* **No `--mmproj`** — the vision tower is dropped; only the text LLM is exported
  (426 tensors). Requires the `qwen.py` patch from §0.
* Norm / SSM tensors (`attn_norm`, `post_attention_norm`, `ssm_norm`, `ssm_a`,
  `ssm_conv1d`, `ssm_dt.bias`, `output_norm`) are written as **F32**, everything else F16.

---

## 6. Inject decision metadata + systemone template

```bash
cd ~/tev1-finetune
./venv/bin/python inject_decision_metadata.py \
  tev1-neely-sp-bf16-v4-f16.gguf \
  tev1-neely-sp-bf16-v4-f16-decision.gguf \
  systemone-template.txt
```

Adds/overrides (making it a `pplx-decider` model served by `/v1/systemone`):

* `<arch>.decision.type = "pplx-decider"`
* `tokenizer.chat_template.systemone = <contents of systemone-template.txt>`
* `tokenizer.chat_templates = ["systemone"]`

---

## 7. Quantize to Q8_0

```bash
cd ~/tev1-finetune
./venv/bin/python quantize_q8.py \
  tev1-neely-sp-bf16-v4-f16-decision.gguf \
  tev1-neely-sp-bf16-v4-q8_0-decision.gguf
```

* `quantize_q8.py` keeps the norm / SSM tensors in **F32** (`KEEP_FLOAT_NAMES`, matched by
  **exact trailing name** — `ssm_a` must not substring-match `ssm_alpha`). Quantizing these
  to Q8_0 crashes the llama.cpp RMSNorm / SSM binary ops.

---

## 8. Register with Ollama

`Modelfile-tev1-neely-v4`:

```
FROM /home/wnf/tev1-finetune/tev1-neely-sp-bf16-v4-q8_0-decision.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0

TEMPLATE "{{ .Prompt }}"
```

```bash
cd ~/tev1-finetune
export OLLAMA_HOST=http://192.168.55.229:11434
ollama create tev1-neely-v4 -f Modelfile-tev1-neely-v4   # -> tev1-neely-v4  (~4.5 GB)
# after A/B verification, promote to the tag the app uses:
ollama create tev1-neely -f Modelfile-tev1-neely-v4      # -> tev1-neely:latest
```

> **`FROM` must be the Q8_0 file.** Pointing `ollama create` at the f16 intermediate
> (`…-f16.gguf` / `…-f16-decision.gguf`) can fail with
> `Error: unsupported file type [123 10 32 32]` (`[123 10 32 32]` = `{\n  `, i.e. Ollama
> tried to read a JSON-shaped source). Only the quantized `…-q8_0-decision.gguf` is the
> deployable artifact.

Smoke test (the model is a decision model — use `/v1/systemone`, not `/api/generate`):

```bash
curl -s http://localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "tev1-neely:latest",
  "state": {"context": "…"},
  "questions": {"sp": {"type": "choice", "instructions": "…",
    "criteria": {":5": "…", ":c3": "…"},
    "choices": [":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3", "Invalid_Neely_Rule_Violation"]}}}'
# -> {"answers": {"sp": {"choice": "<symbol>", …}}}
```

---

## 9. Wire into the app

In `stock-twin-kr` (`src/lib/config.ts`):

```ts
sentiment.model      = process.env.OLLAMA_MODEL_SENTIMENT  || 'tev1-neely:latest'
sp.model             = process.env.OLLAMA_MODEL_SP         || 'tev1-neely:latest'
sp.symbolModel       = process.env.OLLAMA_MODEL_SP_SYMBOL  || 'tev1-neely:latest'
```

Calls go through `ollamaClient.systemOne(model, state, questions, timeoutMs)`
(`src/llm/ollama.client.ts`) → `POST /v1/systemone`. Unload with
`unloadSystemOne()` (`keep_alive: 0`).

---

## 10. Verification

* `ollama ps` → `tev1-neely:latest … 100% GPU` (it can split to CPU if another model is
  co-resident on the same GPU; reload with VRAM free to get 34/34 layers on GPU).
* `ollama show tev1-neely` → architecture `qwen35`, context 4096.
* Merged-model eval: `CUDA_VISIBLE_DEVICES=1 ./venv/bin/python eval/eval_merged_sp.py
  --model merged-tev1-sp-bf16-v4 --heldout sp-systemone-heldout.jsonl --limit 500`.
* Deployed eval: POST the 100 unseen samples to `tev1-neely-v4` via `/v1/systemone`
  (`eval_ollama_sp.py`) — the model must match the merged model.
* Live `GET /api/elliott-wave?symbol=005930` logs `[ElliottWave] SP inference (tev1-neely)`.

---

## 11. Known issues / caveats

1. **Train/held-out leakage — fixed.** The `train4` / `heldout4` splits share **0** rows
   (`state`+`answer` *and* rendered prompt), and the 100-sample unseen test set is disjoint
   from both. Verify with a set intersection before trusting any accuracy number.
2. **Training eval ≠ deployed model.** `finetune_tev1_sp.py`'s `evaluate()` runs the
   in-memory PEFT model; the deployed artifact is the **merged bf16 / Q8_0** model. Evaluate
   the merged model (`eval/eval_merged_sp.py`) and the Ollama tag.
3. **`<think` typo — fixed.** `build_sp_systemone.py` now renders via
   `tokenizer.apply_chat_template` with `systemone-template.txt`, so the prompt matches the
   deployed template byte-for-byte (including the `<think>` / `</think>` added tokens).
4. **Invalid labels.** `Invalid_Neely_Rule_Violation` is project-invented and never occurs in
   real `macro_wave` data. The generator now emits only *learnable* Invalid samples (§2.1);
   set `SFT_INVALID_WEIGHT=0` to drop the class entirely and match production.

---

## 12. Reproduction file inventory

```
README.md                     repository overview / setup
PIPELINE.md                   this document (detailed pipeline)
docs/LECTURE.md               hands-on follow-along tutorial (Korean)
docs/sp-wave-example.png      example chart analysed with the deployed model
requirements.txt              python environment
patches/
  llama.cpp-qwen35-shared-input-merges.patch   required llama.cpp patch (§0)

generate_sft_dataset.py       standalone synthetic SFT generator (label-first, Invalid gate)
build_sp_systemone.py         dataset builder (SFT → systemone)
finetune_tev1_sp.py           LoRA training (bf16; qlora legacy)
merge_tev1_sp.py              LoRA merge → HF bf16
inject_decision_metadata.py   pplx-decider + systemone template injection
quantize_q8.py                Q8_0 quantizer (F32 keep-float list)
audit_invalid_labels.py       Invalid-label regression audit (§2.1)
Modelfile-tev1-neely-v4       Ollama model definition (Q8_0)
systemone-template.txt        systemone chat template (injected into the GGUF)
measure_sp_distribution.py    measures the real SP prior from Kiwoom macro_wave
SP_REAL_DISTRIBUTION.md       measured real SP prior (reweighting target)
sp-real-distribution.json     raw measured prior

sft-dataset-train4.jsonl      source SFT (20k, seed 20261113)
sft-dataset-heldout4.jsonl    source held-out (3k, seed 424242)
sp-systemone-train.jsonl      generated training rows (20k)
sp-systemone-heldout.jsonl    generated held-out rows (3k)

output/tev1-sp-lora-v4/       final LoRA adapter
merged-tev1-sp-bf16-v4/       merged HF bf16 model
tev1-neely-sp-bf16-v4-q8_0-decision.gguf   deployed GGUF (Q8_0)

llama.cpp/                    converter + gguf-py (cloned + patched; not committed)
venv/                         python environment (not committed)
```

> The standalone `generate_sft_dataset.py` is a port of the original TypeScript generator
> (`stock-twin-kr/scripts/generate-sft-dataset.ts`); the two produce **identical** output for
> the same seed.

> `.gitignore`d (large / derived, regenerate per README): `llama.cpp/`, `venv/`, `output/`,
> `merged-*/`, `*.gguf`, `sp-systemone-*.jsonl`.
