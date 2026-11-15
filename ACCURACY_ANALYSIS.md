# Tev1 Neely SP 의사결정 모델 — 정확도 향상 분석

작성 기준: 2026-11-13, 현재 배포 모델 `tev1-neely:latest` (Tev1-4B QLoRA, SP systemone decision).

---

## 0. 반영 결과 (2026-11-14)

이 문서의 권고안은 모두 반영됐다. 이 절이 최신 상태이며, §1–§5는 **수정 전** 분석을
기록한 역사적 내용이다.

| 항목 | 이전 | 현재 |
|---|---|---|
| 학습 정밀도 | QLoRA (nf4) | **bf16 LoRA** (§4.3) |
| 프롬프트 렌더링 | 수기 `<think` (오타) | `tokenizer.apply_chat_template(systemone-template.txt)` (§4.2) |
| held-out | train의 부분집합 (누수) | 새 seed(424242) 3k, **train과 교집합 0** (§4.1) |
| 학습 중 평가 | in-memory nf4 | 병합 bf16 / 배포 산출물 (`eval/eval_merged_sp.py`) |
| 클래스 분포 | 합성 prior (Invalid 34.5%) | **실측 prior** (`SP_REAL_DISTRIBUTION.md`), 실데이터 Invalid 0% |
| Invalid 라벨 | 66.6%가 유효 `:c3` 파동을 오라벨 | 생성기 수정 + 감사 게이트 (§0.1) |

측정 결과 (누수 없는 split):

| 산출물 | 데이터 | 정확도 |
|---|---|---|
| MLP ceiling | 24-feature | 0.999 |
| 병합 bf16 (`merged-tev1-sp-bf16`) | clean held-out (limit 500) | **1.000** |
| 병합 bf16 | 미학습 신규 100 | **1.000** |
| 배포 `tev1-neely:latest` (`/v1/systemone`) | 미학습 신규 100 | **1.000** |
| 신규 데이터 재학습 (`tev1-neely-v4`) | 미학습 신규 100 (수정된 테스트셋) | **1.000** |

> 주의: 초기 테스트셋(`sp-test100-heldout`, **수정 전** 생성기)의 Invalid 2개 중 1개는
> 위반 시그니처가 없는 **유효 패턴 오라벨**이었다 — 그 샘플 하나가 `99/100`의 원인이다.
> 수정된 생성기로 다시 만든 `sp-test100-fixed`(Invalid 3/3 시그니처 정상, train4/heldout4와
> 교집합 0)에서는 구·신 모델 **모두 100/100**(Invalid 3/3)이다. 즉 라벨이 맞으면 두 모델
> 다 완벽하고, 차이는 "오라벨 샘플을 억지로 맞추는가"뿐이다.

### 0.1 Invalid 라벨 감사 및 수정 (신규)

`Invalid_Neely_Rule_Violation`은 프로젝트가 만든 10번째 "reject" 클래스로, 실제
`macro_wave` 데이터에는 **0회** 등장한다(~77k 파동 측정). 초기 생성기는 위반 파동을
index 2에 두고 target을 {1,2,3}에서 무작위 선택 → **66.6%가 유효 `:c3` 파동을 Invalid로
오라벨**, 위반 시그니처가 target 특징 벡터에 드러난 비율은 **28.3%**에 불과했다
(특징 공간 거리 0.0000의 모순 쌍도 존재).

수정: target을 위반 파동(index 2)으로 **고정**하고, 위반을 유효 분포 **바깥**으로 생성
(비교대 `m1_dir==m0_dir`, 극단 `m1_retrace_ratio`≈13–60, 극단 `m1_time_ratio`≈0.03–0.09).
생성 시 `isInvalidSignature` 게이트로 시그니처가 특징에 드러나지 않으면 폐기한다.

| 지표 | 수정 전 (train3) | 수정 후 (train4) |
|---|---|---|
| 위반 시그니처가 특징에 존재 | 28.3% | **100%** |
| 1-NN 최근접이 Invalid | 95.6% | **100%** |
| 모호 (유효가 더 가까움) | 4.4% | **0%** |
| 유효→Invalid 오탐 (FP) | 0% | 0% |

회귀 감사: `./venv/bin/python audit_invalid_labels.py sft-dataset-train4.jsonl` → **PASS**.
`SFT_INVALID_WEIGHT=0`(env)으로 Invalid 클래스를 완전히 제거할 수도 있다(실데이터 일치).
자세한 내용: `PIPELINE.md` §2.1.

---

## TL;DR

- **이 과제는 사실상 완벽하게 학습 가능하다.** 동일한 24차원 monowave feature로
  2-layer MLP를 학습시켰더니 *누수 없는* clean split에서 **99.9%** (rare class 포함
  per-class recall ≈ 1.000)가 나온다. → `eval/ceiling_probe.py`
- 따라서 현재 **68.3%는 데이터/모델 용량의 한계가 아니라 파이프라인 충실도(fidelity)
  문제**다. "정확도를 높이려면" = "파이프라인 버그를 제거하라"에 가깝다.
- 최우선 3가지:
  1. **QLoRA(nf4) → bf16 병합 충실도 손실 제거** (≈ 27 pp) — 단일 최대 레버
  2. **학습/추론 프롬프트 불일치(`<think` vs `<think>`) 제거**
  3. **누수 없는 held-out으로 "배포 산출물" 자체를 평가** (지금은 진행 상황을 볼 수 없음)

---

## 1. 측정 근거 (재현 가능)

| 실험 | 대상 | 정확도 | 비고 |
|---|---|---|---|
| MLP ceiling probe | 24-feature MLP, 20k의 stratified 80/20 | **0.999** | `eval/ceiling_probe.py`, 누수 없음 |
| 배포 모델 | `tev1-neely:latest` (Q8_0), app 경로 | 0.683 (205/300) | held-out이 **train의 부분집합** |
| 학습 중 eval | nf4 4-bit + LoRA (in-memory) | 1.000 (300/300) | 잘못된 산출물 측정 |
| 격리 실험 | nf4 base + LoRA | 1.000 | |
| 격리 실험 | bf16 base + LoRA (unmerged) | 0.73 | 같은 어댑터, base 표현만 교체 |
| 격리 실험 | bf16 base + LoRA (merged) | 0.69 | `merge_and_unload()` 손실 ~4 pp |
| 격리 실험 | nf4 base, LoRA 없음 | 0.00 | base 단독으로는 과제 수행 불가 |

핵심: **같은 LoRA 가중치가 nf4에서는 1.00, bf16에서는 0.73** → 학습된 어댑터가
nf4 양자화 표현에 맞춰져 있어, 배포 base(bf16)에서 어긋난다.

---

## 2. 원인 (영향도 순)

### A. 평가가 무효 — held-out 누수 (반드시 먼저 고칠 것) — ✅ 수정됨 (§0)

- `sp-systemone-heldout.jsonl` 300행의 **prompt와 정답이 모두** `sp-systemone-train.jsonl`에
  그대로 존재: **300/300** (state 기준 300/300, 서로 다른 정답 충돌 0).
- 즉 지금의 68.3%는 *일반화 성능이 아니라 학습 데이터 재현율*이다. 진짜 일반화 성능은
  **아직 측정된 적이 없다.**
- 수정: 아래 4번의 "held-out 재생성" 참고. 이걸 고치기 전까지 다른 개선의 효과를
  측정할 수 없다.

### B. QLoRA → bf16 병합 충실도 손실 (≈ 27 pp, 최대 레버) — ✅ 수정됨 (§0)

- 원인: 4-bit(nf4) base 위에서 LoRA를 학습하면 어댑터가 nf4 양자화 오차에 맞춰 보정된다.
  이를 bf16 base에 병합하면 base가 달라져 결정경계가 이동한다.
- 이 과제는 `m1_retrace_ratio` 같은 **연속 수치의 미세 비교**가 핵심이라 표현 차이에 민감하다.
- 수정(권장): **bf16 base에 일반 LoRA로 학습** (QLoRA 폐기).
  4B bf16 ≈ 8 GB + LoRA 옵티마이저(수백 MB) + activation → 24 GB(4090)에 gradient
  checkpointing으로 충분. 병합/양자화 후 배포 산출물과 학습 시점 모델이 일치하게 된다.
- 대안: VRAM이 빠듯하면 `bnb_4bit_quant_storage=torch.bfloat16` + `prepare_model_for_kbit_training`
  유지, 또는 병합 없이 bf16 LoRA를 그대로 배포. 근본 해결은 bf16 학습이다.

### C. 학습/추론 프롬프트 불일치 (`<think` 오타) — ✅ 수정됨 (§0)

`build_sp_systemone.py:80`은 `<think`(닫는 `>` 없음)을 렌더링하지만, 배포 systemone
템플릿은 `<think>`/`</think>` **added token**을 쓴다. 실제 토크나이즈 결과가 다르다:

```
학습 : <|im_start|> assistant \n <think \n\n </think \n\n
       [248045, 74455, 198, 13314, 741, 271, 510, 26003, 271]   (9 tokens)
배포 : <|im_start|> assistant \n <think> \n\n </think> \n\n
       [248045, 74455, 198, 248068, 271, 248069, 271]           (7 tokens)
```

정답 letter는 이 prefix **직후**에 생성되므로, prefix가 다르면 시작 상태가 어긋난다.
수정(권장): 프롬프트를 손으로 만들지 말고 **systemone 템플릿으로 렌더링**
(`systemone-template.txt`를 `tokenizer.apply_chat_template(..., chat_template=...)`에 전달).
이러면 학습/추론 프롬프트가 바이트 단위로 일치한다.

### D. 학습 중 평가가 잘못된 산출물을 측정

- `finetune_tev1_sp.py`의 `evaluate()`는 **in-memory 4-bit PEFT 모델**로 생성 평가 →
  100%가 나와도 배포 산출물과 무관.
- 수정: 매 eval마다 **병합한 bf16 모델**(또는 배포 GGUF)로 평가하고, 누수 없는 val을 쓴다.

### E. 프롬프트 구성 (효율 문제)

- 프롬프트 ≈ **971 tokens**: state **216** + 고정 boilerplate(옵션 10개 설명문) **≈ 755**.
- 매 샘플마다 동일한 긴 criteria 문장이 77%를 차지 → attention 희석 + 학습/추론 비용 증가.
- MLP가 raw feature만으로 99.9%를 내므로, criteria 설명문은 **필수 정보가 아니다**.
  단축/제거를 검토(단, 배포 템플릿과 반드시 동기화).
- `max_seq_len 1280`은 최대 1008 tokens보다 크므로 **절단은 없음**(문제 아님).

### F. 클래스 불균형 (부차적)

- 분포: Invalid 34.5%, `:c3` 20.2%, `:F3` 15.9%, `:L5` 14.4%, `:5` 6.2%, `b:c3` 3.8%,
  `:sL3` 2.0%, `:L3` 1.9%, `x:c3` 0.7%, `:s5` 0.6%.
  → ✅ **수정됨 (§0)**: 위 분포는 폐기된 합성 prior다. 현재는 실측 prior
  (`SP_REAL_DISTRIBUTION.md`)로 label-first 재가중하며, 실데이터에 없는 Invalid(0%)는
  소량 hedge(기본 3%, `SFT_INVALID_WEIGHT=0`이면 제거)만 남긴다.
- 그러나 MLP는 **plain CE로도 rare class recall 1.000** → 불균형은 병목이 아니다.
- 단, LLM의 Invalid 과다예측은 B(충실도) 문제의 증상일 가능성이 크다. B를 고친 뒤
  그래도 남으면 weighted loss / calibration을 적용.

---

## 3. 큰 효과가 없을 것들 (오해 방지)

- **데이터 양·epoch만 늘리기**: MLP가 20k로 99.9% → 데이터는 충분. 지금 문제는 충실도.
- **클래스 가중치/리샘플링**: F 참고, 병목 아님(2차).
- **추론 캘리브레이션/앙상블**: B·C를 고치기 전엔 증상 치료.
- **더 큰 base 모델**: 과제가 MLP로 풀리는 수준이라 모델 용량이 병목이 아님.
- **LoRA rank만 키우기**: B를 고치지 않으면 nf4 어댑터에 더 잘 과적합될 뿐.

---

## 4. 다음 학습 런 — 권장 레시피

### 4.1 held-out 재생성 (누수 제거) — 먼저

제너레이터가 seed 기반이므로 **새 seed로 별도 생성**하면 진짜 미학습 held-out이 된다:

```bash
cd ~/tev1-neely
./venv/bin/python generate_sft_dataset.py --count 3000 --seed 424242 --out sft-dataset-heldout4.jsonl
# 학습에는 절대 넣지 않는다. (또는 20k를 90/10 stratified split)
```

### 4.2 프롬프트를 템플릿으로 렌더링 (`<think` 불일치 제거)

`build_sp_systemone.py`에서 수기 문자열 대신 systemone 템플릿을 사용:

```python
# systemone-template.txt (GGUF에 주입한 것과 동일) 을 읽어
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_json}],  # + system instruction
    chat_template=systemone_tpl, tokenize=False, add_generation_prompt=True,
)
# 끝에 정답 letter + <|im_end|> 만 붙인다.
```

### 4.3 bf16 LoRA 학습 (QLoRA 폐기)

`finetune_tev1_sp.py` 변경점:

```python
# (기존) BitsAndBytesConfig(load_in_4bit=True, ...) + prepare_model_for_kbit_training
# (신규) 4-bit 제거
model = AutoModelForImageTextToText.from_pretrained(
    args.base_model, torch_dtype=torch.bfloat16,
    device_map={"": 0}, trust_remote_code=True,
)
model.gradient_checkpointing_enable()
```

권장 하이퍼파라미터(clean val로 조기 종료):

```bash
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python finetune_tev1_sp.py \
  --train sp-systemone-train.jsonl \
  --heldout sp-systemone-heldout2.jsonl \
  --output-dir output/tev1-sp-lora-bf16 \
  --max-seq-len 1280 --batch-size 1 --grad-accum 16 \
  --epochs 3 --lr 1e-4 --lora-r 32 --lora-alpha 64 \
  --eval-samples 300 --optim adamw_torch_fused --no-grad-checkpointing
```

- `epochs 0.5 → 3`, `lr 2e-5 → 1e-4`, `r 16 → 32`: base가 과제를 못 하므로(0%)
  전부 새로 학습해야 함 → 더 많은 스텝/용량 필요.
- 학습 중 val(heldout2)을 **병합 bf16 모델**로 평가해 조기 종료.

### 4.4 병합 → GGUF → 등록은 기존과 동일

`merge_tev1_sp.py` → `convert_hf_to_gguf.py`(f16) → `inject_decision_metadata.py` →
`quantize_q8.py` → `ollama create`. (PIPELINE.md §4–8)

### 4.5 목표 지표

- train 재현율: ~1.00
- clean val: **MLP ceiling(0.999)에 근접**(≥ 0.95) 목표
- 배포 GGUF(`/v1/systemone`)로 최종 확인

---

## 5. 더 근본적인 관점

1. **과제가 MLP 수준으로 쉽다.** 24-feature MLP가 99.9%라면, 이 *합성* 과제 한정으로는
   4B LLM이 과합니다. Tev1을 decision 엔드포인트로 유지해야 하는 제약이 아니라면,
   소형 분류기/소형 LLM이 더 정확·저렴합니다.
2. **합성 → 실데이터 도메인 갭은 미측정.** 데이터셋은 `generate_sft_dataset.py`가
   Neely 규칙으로 *생성*한 합성 데이터다(라벨이 feature의 결정적 함수). 진짜 정확도는
   실 시장 monowave + 전문가 Neely 라벨로 측정해야 한다. 다음 단계로 실데이터 라벨링
   파이프라인을 권장.
3. **회귀 방지 하네스 유지**: `eval/ceiling_probe.py`(MLP ceiling)를 기준선으로 두고,
   학습 런마다 clean val / 배포 산출물 / MLP ceiling을 함께 기록.

---

## 부록: 재현 명령

```bash
cd /home/wnf/tev1-finetune
CUDA_VISIBLE_DEVICES= ./venv/bin/python eval/ceiling_probe.py   # MLP ceiling ≈ 0.999

# Invalid 라벨 회귀 감사 (train4 → PASS, train3 → FAIL)
./venv/bin/python audit_invalid_labels.py sft-dataset-train4.jsonl
```

데이터 재생성 + 재학습 + 배포 전체 절차는 `PIPELINE.md` §2–§8 참고
(현재 데이터: `sft-dataset-train4.jsonl` / `sft-dataset-heldout4.jsonl`).
