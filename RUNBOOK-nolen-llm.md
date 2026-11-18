# nolen LLM — 4090 fine-tune → Ollama 등록 → A/B 런북

> nolen feature(절대 length 제거, time_bars 유지)로 **tev1-neely LLM**을 재학습하고,
> Ollama decision 모델(`tev1-neely-nolen`)로 등록한 뒤 **nolen MLP(8097)**와 출력을 비교한다.

## 0. 전제

| 항목 | 값 |
|---|---|
| 학습 데이터 | `sp-systemone-nolen-train.jsonl` (49,999) / `sp-systemone-nolen-heldout.jsonl` (9,000) — 이미 생성 완료 |
| GPU | 4090 = torch device 0 (`CUDA_VISIBLE_DEVICES=0`, nvidia-smi index 1과 반대) |
| base | `togethercomputer/Tev1-4B-experimental` (HF cache snapshot) |
| prompt 길이 | min 906 / p50 921 / max 934 tokens → `--max-seq-len 1280` 충분 |

> **4090이 `tqserve`로 98% 점유 중** — 학습 시작 전 GPU를 비워야 한다 (사용자 관리).

---

## 1. Fine-tune (사용자가 4090에서 실행)

```bash
cd ~/tev1-finetune
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python finetune_tev1_sp.py \
  --train sp-systemone-nolen-train.jsonl \
  --heldout sp-systemone-nolen-heldout.jsonl \
  --output-dir output/tev1-sp-nolen \
  --max-seq-len 1280 --batch-size 1 --grad-accum 16 \
  --epochs 1 --lr 2e-5 --eval-samples 300 \
  --no-grad-checkpointing --optim adamw_torch_fused \
  --precision bf16
```

- v4와 동일한 하이퍼파라미터 (bf16 LoRA, r=16/alpha=32, 1 epoch, fused adamw)
- `--eval-samples 300`: 학습 직후 in-memory 모델로 nolen held-out 300개 평가 (참고용)
- 산출물: `output/tev1-sp-nolen/` (adapter + tokenizer)
- OOM 시 `--no-grad-checkpointing` 제거

## 2. Merge → GGUF → decision metadata → Q8_0 (학습 완료 후)

```bash
cd ~/tev1-finetune

# 2.1 merge (CPU)
ADAPTER_DIR=output/tev1-sp-nolen MERGED_DIR=merged-tev1-sp-nolen ./venv/bin/python merge_tev1_sp.py

# 2.2 GGUF (f16, text-only)
./venv/bin/python llama.cpp/convert_hf_to_gguf.py merged-tev1-sp-nolen \
  --outfile tev1-neely-sp-nolen-f16.gguf --outtype f16

# 2.3 decision metadata + systemone template 주입
./venv/bin/python inject_decision_metadata.py \
  tev1-neely-sp-nolen-f16.gguf \
  tev1-neely-sp-nolen-f16-decision.gguf \
  systemone-template.txt

# 2.4 Q8_0 (norm/SSM F32 유지)
./venv/bin/python quantize_q8.py \
  tev1-neely-sp-nolen-f16-decision.gguf \
  tev1-neely-sp-nolen-q8_0-decision.gguf
```

## 3. Ollama 등록

```bash
cd ~/tev1-finetune
export OLLAMA_HOST=http://192.168.55.229:11434
ollama create tev1-neely-nolen -f Modelfile-tev1-neely-nolen
```

- `Modelfile-tev1-neely-nolen`은 이미 준비됨 (FROM = Q8_0 decision gguf, num_ctx 4096, temp 0)
- **FROM은 반드시 Q8_0 파일** (f16 intermediates는 `unsupported file type` 에러)
- 기존 `tev1-neely:latest`(36-dim)는 **그대로 유지** — A/B 대조용

## 4. A/B 비교 (nolen LLM vs nolen MLP)

```bash
cd ~/tev1-finetune
./venv/bin/python ab_eth_nolen_llm.py
```

- 4자: **nolen-llm**(ollama `tev1-neely-nolen`) / **nolen-mlp**(8097) / multi(8095) / ollama(11434)
- **MAIN**: `nolen-llm == nolen-mlp` 일치율
- ETH 1D 87 wave (`/tmp/eth_wave.json`)

## 5. 검증 체크리스트

- [ ] `output/tev1-sp-nolen/adapter_model.safetensors` 존재
- [ ] 학습 로그: held-out 300개 accuracy 출력
- [ ] `ollama show tev1-neely-nolen` → architecture `qwen35`, context 4096
- [ ] `curl http://127.0.0.1:11434/v1/systemone` smoke test (nolen state)
- [ ] `ab_eth_nolen_llm.py` 실행 → nolen-llm vs nolen-mlp 정합률 확인
