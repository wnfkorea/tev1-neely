# tev1-neely — Neely SP decision model

Fine-tune of **Tev1-4B-experimental** (Qwen3.5-4B hybrid) into a System-1 **decision
model** that answers the Elliott-wave **SP symbol** question of the `stock-twin-kr`
pipeline over Ollama's `/v1/systemone` endpoint.

The model is **not** a text generator. Given a standardised wave state it returns exactly
one of ten fixed Neely SP symbols:

```
:5  :s5  :c3  b:c3  :F3  x:c3  :L5  :L3  :sL3  Invalid_Neely_Rule_Violation
```

- **Base** — `togethercomputer/Tev1-4B-experimental` (Qwen3.5-4B hybrid: linear-attention +
  full-attention + MTP)
- **Method** — bf16 LoRA (`r=16`) on a *synthetic* Neely dataset, **reweighted to the measured
  real SP prior**
- **Deploy** — LoRA merge → HF bf16 → GGUF f16 → `pplx-decider` metadata → Q8_0 →
  `ollama create`
- **Result** — **1.000** accuracy on a disjoint, unseen test set

**New here?** Start with the hands-on tutorial **[`docs/LECTURE.md`](docs/LECTURE.md)** — a
follow-along lecture that walks the whole pipeline from a bare machine to a deployed Ollama
model. The detailed pipeline and provenance live in [`PIPELINE.md`](PIPELINE.md).

---

## Results

All numbers are on **leak-free** splits (0 overlap between train / held-out / test).

| Artifact | Data | Accuracy |
|---|---|---|
| MLP ceiling probe (24-feature) | clean split | 0.999 |
| Merged bf16 (`merged-tev1-sp-bf16`) | clean held-out (500) | **1.000** |
| Deployed `tev1-neely:latest` (`/v1/systemone`) | unseen 100 | **1.000** |
| Retrained `tev1-neely-v4` | unseen 100 (corrected labels) | **1.000** |

The task is essentially perfectly learnable from the 24-dimensional monowave features — the
remaining work was *pipeline fidelity* (prompt rendering, QLoRA→bf16 merge loss, held-out
leakage, label quality), not model capacity. Full analysis:
[`ACCURACY_ANALYSIS.md`](ACCURACY_ANALYSIS.md).

The real SP-symbol distribution was **measured** (not assumed) from ~77 k Kiwoom `macro_wave`
labels: [`SP_REAL_DISTRIBUTION.md`](SP_REAL_DISTRIBUTION.md).

---

## Pipeline at a glance

```
generate_sft_dataset.py                         (synthetic, label-first, Invalid-gated)
      └─→ sft-dataset-train4.jsonl / sft-dataset-heldout4.jsonl
              └─ build_sp_systemone.py ─→ sp-systemone-{train,heldout}.jsonl
                      └─ finetune_tev1_sp.py ─→ output/tev1-sp-lora-v4        (bf16 LoRA)
                              └─ merge_tev1_sp.py ─→ merged-tev1-sp-bf16-v4    (HF bf16)
                                      └─ convert_hf_to_gguf.py ─→ …-f16.gguf   (text-only, patched)
                                              └─ inject_decision_metadata.py ─→ …-f16-decision.gguf
                                                      └─ quantize_q8.py ─→ …-q8_0-decision.gguf
                                                              └─ ollama create ─→ tev1-neely
```

> **`ollama create` `FROM` must point at the Q8_0 file.** Feeding it an f16 intermediate can
> fail with `Error: unsupported file type [123 10 32 32]` (the bytes spell `{\n  `).

---

## Repository layout

```
README.md                       this file
PIPELINE.md                     detailed fine-tune → GGUF → Ollama pipeline
docs/LECTURE.md                 hands-on tutorial (Korean) — follow along end to end
docs/sp-wave-example.png        example chart analysed with the deployed model
ACCURACY_ANALYSIS.md            accuracy analysis (before / after the fixes)
SP_REAL_DISTRIBUTION.md         measured real SP prior (reweighting target)
sp-real-distribution.json       raw measured prior
requirements.txt                Python environment

patches/
  llama.cpp-qwen35-shared-input-merges.patch   required llama.cpp patch (see Setup)

generate_sft_dataset.py         standalone synthetic SFT generator (stdlib only, no Node)
finetune_tev1_sp.py             bf16 LoRA training (--precision bf16 | qlora)
build_sp_systemone.py           SFT rows → systemone decision records
merge_tev1_sp.py                LoRA merge → standalone HF bf16
inject_decision_metadata.py     pplx-decider type + systemone chat template
quantize_q8.py                  Q8_0 quantizer (F32 keep-float list)
audit_invalid_labels.py         Invalid-label regression audit
measure_sp_distribution.py      measures the real prior from Kiwoom macro_wave
systemone-template.txt          systemone chat template (injected into the GGUF)
Modelfile-tev1-neely-v4         Ollama model definition (Q8_0)

sft-dataset-train4.jsonl        source SFT (20 000 rows, seed 20261113)
sft-dataset-heldout4.jsonl      source held-out (3 000 rows, seed 424242)

eval/                           evaluation scripts (ceiling probe, merged-model eval, …)
legacy/                         superseded early scripts (kept for history)
```

### Not committed (regenerate locally)

The following are **large or derived** and are `.gitignore`d — see
[Large artifacts](#large-artifacts-not-committed):

- `llama.cpp/` — cloned + patched (Setup)
- `venv/` — Python environment
- `output/` — LoRA adapters and checkpoints
- `merged-*/` — merged HF models (~8.5 GB each)
- `*.gguf` — f16 / Q8_0 models (4–8 GB each)
- `sp-systemone-*.jsonl` — rendered datasets (derived; the train split exceeds GitHub's 100 MB file limit)

---

## Setup

### 1. llama.cpp (pinned + patched)

The GGUF export needs a **specific llama.cpp revision with a one-line patch** — the
`Qwen3_5TextModel._SHARED_INPUT_MERGES` list must be emptied so the GGUF keeps separate
`attn_qkv`/`attn_gate` and `ssm_alpha`/`ssm_beta` tensors (matching the deployed `tev1` GGUF
layout the server loader expects).

```bash
cd ~/tev1-finetune
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c
git -C llama.cpp apply patches/llama.cpp-qwen35-shared-input-merges.patch
```

The patch is `git apply`-able from the `llama.cpp/` root (`-p1`).

### 2. Python environment

```bash
cd ~/tev1-finetune
python3 -m venv venv
./venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
./venv/bin/pip install -r requirements.txt
```

Tested with Python 3.12 + CUDA 12.4.

### 3. Base model

Download `togethercomputer/Tev1-4B-experimental` (a Hugging Face cache snapshot is fine; the
scripts load it by repo id). The training/eval scripts use the HF cache directly.

### 4. Dataset generator

`generate_sft_dataset.py` is a **standalone, dependency-free** (stdlib-only) generator — no
Node.js and no other repository required. It is a faithful port of the original TypeScript
generator (`stock-twin-kr/scripts/generate-sft-dataset.ts`): given the same seed it reproduces
the committed `sft-dataset-*.jsonl` **byte for byte** (verified for both splits). See
[`PIPELINE.md`](PIPELINE.md) §2 for the algorithm and the measured prior it reweights to.

---

## Reproduce

```bash
cd ~/tev1-finetune

# 1) generate the synthetic SFT data (standalone generator, no Node.js needed)
./venv/bin/python generate_sft_dataset.py --count 20000 --seed 20261113 --out sft-dataset-train4.jsonl
./venv/bin/python generate_sft_dataset.py --count 3000  --seed 424242   --out sft-dataset-heldout4.jsonl

# 2) build systemone decision records + audit the Invalid labels
./venv/bin/python build_sp_systemone.py \
  --train sft-dataset-train4.jsonl --heldout sft-dataset-heldout4.jsonl --out-prefix sp-systemone
./venv/bin/python audit_invalid_labels.py sft-dataset-train4.jsonl      # -> RESULT: PASS

# 3) train (bf16 LoRA) on the 4090
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python finetune_tev1_sp.py \
  --train sp-systemone-train.jsonl --heldout sp-systemone-heldout.jsonl \
  --output-dir output/tev1-sp-lora-v4 \
  --max-seq-len 1280 --batch-size 1 --grad-accum 16 --epochs 1 --lr 2e-5 \
  --eval-samples 300 --no-grad-checkpointing --optim adamw_torch_fused --precision bf16

# 4) merge → GGUF → metadata → Q8_0
ADAPTER_DIR=output/tev1-sp-lora-v4 MERGED_DIR=merged-tev1-sp-bf16-v4 ./venv/bin/python merge_tev1_sp.py
./venv/bin/python llama.cpp/convert_hf_to_gguf.py merged-tev1-sp-bf16-v4 \
  --outfile tev1-neely-sp-bf16-v4-f16.gguf --outtype f16          # no --mmproj: text-only
./venv/bin/python inject_decision_metadata.py \
  tev1-neely-sp-bf16-v4-f16.gguf tev1-neely-sp-bf16-v4-f16-decision.gguf systemone-template.txt
./venv/bin/python quantize_q8.py \
  tev1-neely-sp-bf16-v4-f16-decision.gguf tev1-neely-sp-bf16-v4-q8_0-decision.gguf

# 5) register with Ollama (FROM the Q8_0 file)
export OLLAMA_HOST=http://127.0.0.1:11434
ollama create tev1-neely-v4 -f Modelfile-tev1-neely-v4
```

---

## Deploy to Ollama

`Modelfile-tev1-neely-v4`:

```
FROM /home/wnf/tev1-finetune/tev1-neely-sp-bf16-v4-q8_0-decision.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0

TEMPLATE "{{ .Prompt }}"
```

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama create tev1-neely-v4 -f Modelfile-tev1-neely-v4   # A/B candidate (~4.5 GB)
ollama create tev1-neely    -f Modelfile-tev1-neely-v4   # promote to the tag the app uses
```

Smoke test (a **decision** model — use `/v1/systemone`, not `/api/generate`):

```bash
curl -s "$OLLAMA_HOST/v1/systemone" -H 'Content-Type: application/json' -d '{
  "model": "tev1-neely:latest",
  "state": {"context": "…"},
  "questions": {"sp": {"type": "choice", "instructions": "…",
    "criteria": {":5": "…", ":c3": "…"},
    "choices": [":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3", "Invalid_Neely_Rule_Violation"]}}}'
# -> {"answers": {"sp": {"choice": "<symbol>", …}}}
```

The application side lives in `stock-twin-kr` (`src/lib/config.ts` sets
`sp.model` / `sp.symbolModel` / `sentiment.model` to `tev1-neely:latest`;
calls go through `ollamaClient.systemOne()` → `POST /v1/systemone`).

---

## Large artifacts (not committed)

GitHub rejects files larger than 100 MB, so the trained artifacts are intentionally excluded:

| Artifact | Size | How to obtain |
|---|---|---|
| `merged-tev1-sp-bf16-v4/` | ~8.5 GB | step 4 of *Reproduce* |
| `tev1-neely-sp-bf16-v4-f16.gguf` | ~7.8 GB | step 4 |
| `tev1-neely-sp-bf16-v4-q8_0-decision.gguf` | ~4.2 GB | step 4 (deployable) |
| `output/tev1-sp-lora-v4/` | ~0.1 GB | step 3 |
| `sp-systemone-train.jsonl` | ~126 MB | `build_sp_systemone.py` |

Distribute the GGUF via GitHub Releases or the Hugging Face Hub if you need the binary without
retraining.

---

## Caveats

1. **Train/held-out/test are disjoint.** `train4`/`heldout4` share 0 rows and the unseen test
   set is disjoint from both — verify with a set intersection before trusting any accuracy number.
2. **Evaluate the deployed artifact, not the in-memory PEFT model.** `finetune_tev1_sp.py`'s
   in-training `evaluate()` measures the PEFT model; the deployed model is the *merged bf16 /
   Q8_0* one. Use `eval/eval_merged_sp.py` and the Ollama tag.
3. **`Invalid_Neely_Rule_Violation` is project-invented** and never occurs in real `macro_wave`
   data (~0 / 77 k). The generator emits only *learnable* Invalid samples
   (`audit_invalid_labels.py`); set `SFT_INVALID_WEIGHT=0` to drop the class entirely and match
   production.
4. **Do not quantize the norm / SSM tensors.** `quantize_q8.py` keeps `attn_norm`,
   `post_attention_norm`, `ssm_norm`, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, `output_norm` in
   F32 — quantizing them crashes the llama.cpp RMSNorm / SSM binary ops.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/LECTURE.md`](docs/LECTURE.md) | hands-on follow-along tutorial (Korean) — start here |
| [`PIPELINE.md`](PIPELINE.md) | full pipeline, exact commands, reproduction inventory |
| [`ACCURACY_ANALYSIS.md`](ACCURACY_ANALYSIS.md) | accuracy analysis and the fixes that took it to 1.000 |
| [`SP_REAL_DISTRIBUTION.md`](SP_REAL_DISTRIBUTION.md) | measured real-world SP prior + Neely literature review |

---

## Acknowledgements

- Base model: `togethercomputer/Tev1-4B-experimental`
- Neely SP notation: Glenn Neely, *Mastering Elliott Wave* (1990) — proprietary notation; the
  labels used here are the ones emitted by the production data source.
- Conversion/quantization: [llama.cpp](https://github.com/ggml-org/llama.cpp) (`ggml-org`).

---

## License

MIT — see [`LICENSE`](LICENSE).
