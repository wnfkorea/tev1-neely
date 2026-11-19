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

### 3.1 nolen을 새 `tev1-neely`로 승격 (2026-11-19)

nolen redesign 원리(시장 의존 `length` 제거)는 **MLP/LLM 공통**이므로,
nolen LLM을 **새 `tev1-neely`**로 등록한다 (기존 36-dim `tev1-neely` 대체).

```bash
# 기존 36-dim tev1-neely 제거
ollama rm tev1-neely:latest
# nolen을 tev1-neely로 복사 (동일 ID, decision metadata 상속)
ollama cp tev1-neely-nolen:latest tev1-neely:latest
```

- `tev1-neely:latest`와 `tev1-neely-nolen:latest`가 **동일 모델 ID**로 확인
- `tev1-neely-nolen:latest`는 **그대로 유지** — 별칭으로 계속 사용 가능
- **`tev1-neely`는 이제 nolen 17-key state를 받음** (22-key full state는 더 이상 사용 불가)
- A/B 대조용 36-dim 모델은 `tev1-neely` 태그가 nolen으로 교체됨 —
  36-dim 대조가 필요하면 별도 태그(예: `tev1-neely-legacy`)로 유지

### 3.2 gapfill v2를 새 `tev1-neely`로 승격 (2026-11-19)

`m1_retrace_ratio` 공백 3개([0.50,0.68], [3.67,5.45], [6.62,13.61])를
5,000 샘플로 증강한 **v2**를 `tev1-neely`로 승격한다.

```bash
# v2 LLM (사용자 4090 학습 → Ollama 등록 완료: tev1-neely-nolen-v2)
# v2 MLP (동일 combined 54,999 데이터셋으로 재학습)
./venv/bin/python mlp/train_nolen.py \
  --train sp-systemone-nolen-train-v2.jsonl \
  --heldout sp-systemone-nolen-heldout.jsonl \
  --out mlp/sp_mlp_nolen_v2.pt --epochs 60

# 프로덕션 8095 → v2
sudo sed -i 's|sp_mlp_nolen\.pt|sp_mlp_nolen_v2.pt|' /etc/systemd/system/sp-mlp.service
sudo systemctl daemon-reload && sudo systemctl restart sp-mlp

# Ollama: v2를 새 tev1-neely로 승격
ollama rm tev1-neely:latest
ollama cp tev1-neely-nolen-v2:latest tev1-neely:latest
```

- `tev1-neely:latest`와 `tev1-neely-nolen-v2:latest`가 **동일 모델 ID**(`a8ff6497ed03`)
- `tev1-neely-nolen:latest`(v1, `1fcc9f425de5`)는 **그대로 유지** — v1 대조용
- **A/B 결과**: nolen-llm-v2 == nolen-mlp-v2
  - ETH: **46/87 = 0.5287** (v1: 23/87 = 0.2644)
  - ^KS11: **47/89 = 0.5281** (v1: 23/87 = 0.2644)
  - gapfill 증강으로 LLM/MLP 일치율 **2배 상승**
- 남은 ~47% 발산: LLM 의미론적 추론 vs MLP 결정적 패턴매칭 + 실제 시장 분포 차이

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
- [ ] **`tev1-neely` 승격**: `ollama list`에서 `tev1-neely:latest`와 `tev1-neely-nolen:latest`가 동일 ID
- [ ] **`tev1-neely` smoke test**: 17-key nolen state로 `/v1/systemone` 호출 → choice + confidence 반환
