### Tev1-0.8B를 Neely SP 데이터로 fine-tune하기 — RTX 4060 Ti + SOUP 실습 강의

#### 목차

- 0. 이 강의에서 만드는 것
- 1. 배경 — 왜 0.8B인가, 왜 SP인가
- 2. 준비물
- 3. 데이터 — V4 RC-primary + b:c3 보강
- 4. 학습 설정
- 5. 학습 실행
- 6. LoRA 병합과 chat template 복원
- 7. 평가 — SP heldout
- 8. 배포 — GGUF → Ollama
- 9. 부수 실험 — 뉴스 sentiment 유지도
- 10. 함정과 교훈
- 11. 산출물
- 12. 재현 명령 전체 순서
- 부록 A. 파일 지도

#### 0. 이 강의에서 만드는 것

이 실습의 목표는 **Tev1-0.8B를 Neely SP(파동기호) 판정기로 파인튜닝하고, 그 결과를 실제 서비스 인터페이스에서 검증하는 것**입니다.

최종적으로 다음 세 가지가 남습니다.

```text
1) SOUP LoRA adapter + bf16 병합 모델
   /tmp/opencode/soup-v4-rc-08b-bc3aug-merged

2) Ollama 배포 모델
   tev1-sp-bc3aug:latest   (Q8_0, decision capability)

3) 두 가지 평가 결과
   - SP heldout  : 92.79%  (b:c3 recall 99.1%)
   - sentiment   : 92.5%   (base 93.0% 대비 -0.5%p)
```

강의의 초점은 “정확도 몇 %”보다 **왜 이런 설정을 골랐는가, 어디서 막혔는가, 어떻게 풀었는가**에 있습니다.

#### 1. 배경 — 왜 0.8B인가, 왜 SP인가

##### 1.1 decision 모델과 systemone

Tev1은 생성형 LLM이면서 동시에 `/v1/systemone`이라는 **의사결정 전용 인터페이스**를 제공합니다. 입력은 `state / question / options` 구조이고, 출력은 선택지의 키(letter) 하나입니다.

```json
{
  "model": "tev1-sp-bc3aug:latest",
  "state": "{... 모노파동 특징 ...}",
  "questions": {
    "sp": {
      "type": "choice",
      "instructions": "...",
      "criteria": {"A": ":5", "B": ":s5", "C": ":c3", "D": "b:c3", "...": "..."}
    }
  }
}
```

파인튜닝은 이 포맷 그대로 SFT 데이터로 만들었습니다. 즉 **모델의 말투를 바꾸는 게 아니라, 같은 decision 슬롯에서 어떤 기호를 고를지 재학습**하는 것입니다.

##### 1.2 Neely SP 라벨

이 과제의 정답은 10개 클래스입니다.

| 문자 | SP 기호 | 의미 |
|---|---|---|
| A | `:5` | Impulse wave (not final) |
| B | `:s5` | Special 5-wave |
| C | `:c3` | Central 3-wave |
| D | `b:c3` | B-wave of a flat correction |
| E | `:F3` | First 3-wave |
| F | `x:c3` | X-wave position |
| G | `:L5` | Last 5-wave |
| H | `:L3` | Last 3-wave |
| I | `:sL3` | Second-to-last 3-wave |
| J | `Invalid_Neely_Rule_Violation` | 규칙 위반 |

이번 증강의 주인공은 **D (`b:c3`)** 입니다. 기존 모델이 이 클래스를 충분히 잡아주지 못해 데이터와 가중치를 모두 손봤습니다.

##### 1.3 왜 0.8B인가

- 4B Neely 실험에서 이미 “특화 파인튜닝이 sentiment를 깎지 않는다”는 신호를 확인했습니다.
- 0.8B는 용량이 작아 forgetting에 더 민감하므로, 같은 현상이 유지되는지 확인할 가치가 있습니다.
- 실 서비스에서 더 싸고 빠르게 돌릴 수 있는 후보이기도 합니다.

#### 2. 준비물

##### 2.1 하드웨어 — 4060 Ti를 고집하는 이유

학습은 **RTX 4060 Ti 15.6GB**에서만 실행했습니다.

```bash
CUDA_VISIBLE_DEVICES=1 /home/wnf/soup-venv/bin/soup train ...
```

이 환경에서 `CUDA_VISIBLE_DEVICES=1`은 4060 Ti로 매핑됩니다. 기본 GPU인 4090에는 다른 프로세스가 이미 19.70GiB를 점유하고 있어, 그 위에 학습을 올리면 즉시 OOM이 납니다.

```text
GPU 0: NVIDIA GeForce RTX 4060 Ti   ← 학습용
GPU 1: NVIDIA GeForce RTX 4090      ← 다른 작업 점유 중
```

이 숫자는 `nvidia-smi`의 물리 인덱스와 CUDA 가시 인덱스가 뒤섞여 보일 수 있으니, 항상 `torch.cuda.get_device_name(0)`으로 실제 대상을 확인하는 습관이 중요합니다.

##### 2.2 소프트웨어

```bash
# SOUP 학습 툴체인
/home/wnf/soup-venv/bin/soup          # soup-cli[train]==0.78.0

# 저장소
/home/wnf/tev1-finetune

# 베이스 모델
togethercomputer/Tev1-0.8B-experimental
revision 6bb2dff14b38fea90ddb14d870166ccaf77374e9
```

##### 2.3 알려진 함정 — `causal_conv1d` ABI

`soup-venv`의 `causal_conv1d`가 깨져 있으면 매 스텝 fallback 경고가 쏟아집니다. 저장소 venv의 정상 `.so`를 복사해 해결했습니다.

```bash
cp <repo-venv>/.../causal_conv1d*.so /home/wnf/soup-venv/lib/python3.12/site-packages/
```

해결 후 스모크 로그의 fallback 경고는 **0건**이었습니다. 이걸 먼저 잡지 않으면 학습은 되지만 속도와 재현성이 무너집니다.

#### 3. 데이터 — V4 RC-primary + b:c3 보강

##### 3.1 V4 RC basis

기본 V4 RC 데이터는 다음과 같습니다.

| split | 행 수 |
|---|---:|
| train | 101,779 |
| heldout | 11,309 |

##### 3.2 b:c3 승격 + 생성

`b:c3`를 강화하기 위해 두 경로를 섞었습니다.

| 구성 | 행 수 |
|---|---:|
| 기존 2차 후보 승격 | 963 |
| 엔진 생성 데이터 | 4,000 |
| **추가 합계** | **4,963** |

최종 SFT 데이터는 다음과 같습니다.

| split | 행 수 |
|---|---:|
| train | 105,369 |
| heldout | 11,253 |
| heldout 중 `b:c3` | 116 |

```text
/tmp/v4_rc_bc3aug_systemone-soup-train.jsonl
/tmp/v4_rc_bc3aug_systemone-soup-heldout.jsonl
```

각 레코드는 `instruction`에 fully-rendered systemone 프롬프트가 들어 있고, `output`은 정답 문자 하나입니다.

##### 3.3 왜 heldout을 따로 봐야 하는가

학습 후 LLM 자체 평가와 MLP 재학습 평가가 서로 다른 숫자를 낼 수 있습니다. 특히 `b:c3`는 **LLM이 115/116을 맞히지만 MLP는 105/116**에 그쳤습니다. 따라서 “모델 성능”과 “프로덕션 분류기 성능”을 구분해서 기록해야 합니다.

#### 4. 학습 설정

##### 4.1 config 요약

실제 설정 파일은 `/tmp/opencode/soup_v4_rc_08b_bc3aug.yaml`입니다. 핵심 값은 다음과 같습니다.

```yaml
base_model: togethercomputer/Tev1-0.8B-experimental
task: sft
backend: transformers

train_data: /tmp/v4_rc_bc3aug_systemone-soup-train.jsonl
heldout_data: /tmp/v4_rc_bc3aug_systemone-soup-heldout.jsonl

batch_size: 1
gradient_accumulation_steps: 16
max_length: 1240
epochs: 1
learning_rate: 2e-5

lora:
  r: 16
  alpha: 32
  target_modules: all-linear

bf16: true
gradient_checkpointing: false
optimizer: adamw_torch_fused
seed: 42
```

`batch_size=1 + accumulation=16`은 4060 Ti의 15.6GB 안에서 1240 토큰 시퀀스를 안정적으로 돌리기 위한 조합입니다. `gradient_checkpointing`을 끈 대신 메모리를 아낀 덕분에 스텝 시간이 짧아졌습니다.

##### 4.2 왜 bf16 LoRA인가

- QLoRA는 4bit 양자화 오차가 decision 경계에 영향을 줄 수 있습니다.
- LoRA `r=16, alpha=32, all-linear`은 전체 파라미터의 약 1.4%만 갱신합니다.
- 1 epoch이면 SP 포맷에 충분히 적응하면서 sentiment 같은 기존 능력을 덜 잃습니다.

##### 4.3 스모크 테스트

본 학습 전에 짧게 돌려 자원을 확인했습니다.

```text
step time      : 5.67 s/step
GPU peak       : 8.0 / 15.6 GB
fallback       : 0건
b:c3 smoke     : 36 / 40 = 90%
```

이 정도면 풀 런을 시작해도 안전합니다.

#### 5. 학습 실행

##### 5.1 명령

```bash
CUDA_VISIBLE_DEVICES=1 \
  /home/wnf/soup-venv/bin/soup train \
  --config /tmp/opencode/soup_v4_rc_08b_bc3aug.yaml \
  --yes
```

##### 5.2 로그와 체크포인트

최종 완료된 실행은 다음과 같습니다.

```text
run_id        : run_20261121_144423_1232b7a2
device        : NVIDIA GeForce RTX 4060 Ti
dtype         : bf16
dataset       : 105,369 train samples
final step    : checkpoint-6586
adapter mtime : 2026-11-22 01:19
```

주의: `/tmp/opencode/soup_bc3aug_full.log`에는 **중지된 이전 실행**(다른 run_id)이 남아 있습니다. 완료 근거로 쓰면 안 됩니다. 항상 adapter 디렉터리의 `soup_run.json`과 mtime을 확인하세요.

##### 5.3 실제 산출물

```text
/tmp/opencode/soup-v4-rc-08b-bc3aug-out/
└── tev1-v4-rc-primary-08b-bc3aug-soup/
    ├── adapter_model.safetensors   (43 MB)
    ├── adapter_config.json
    ├── soup_run.json
    ├── chat_template.jinja
    └── checkpoint-6586/
```

#### 6. LoRA 병합과 chat template 복원

##### 6.1 merge 명령의 함정

SOUP merge에는 두 가지 제약이 있습니다.

1. `--output`은 반드시 **cwd 하위**여야 합니다.
2. `--adapter`는 `adapter_config.json`이 있는 **중첩 디렉터리**여야 합니다.

따라서 다음처럼 실행해야 합니다.

```bash
cd /tmp/opencode

CUDA_VISIBLE_DEVICES=1 \
/home/wnf/soup-venv/bin/soup merge \
  --adapter /tmp/opencode/soup-v4-rc-08b-bc3aug-out/tev1-v4-rc-primary-08b-bc3aug-soup \
  --output soup-v4-rc-08b-bc3aug-merged \
  --dtype bfloat16
```

잘못된 예:

```bash
# --output이 cwd 밖 → 거부
--output /tmp/opencode/soup-v4-rc-08b-bc3aug-merged

# 상위 output dir을 adapter로 지정 → adapter_config.json 없음
--adapter /tmp/opencode/soup-v4-rc-08b-bc3aug-out
```

##### 6.2 chat template 복원

merge 과정에서 chat template이 빠지거나 inline으로 바뀔 수 있습니다. 원본 `chat_template.jinja`를 복원했습니다.

```bash
python3 /tmp/opencode/restore_tev1_chat_template.py \
  /tmp/opencode/soup-v4-rc-08b-bc3aug-merged
```

확인 결과:

```text
chat_template.jinja        존재 (7,651 bytes)
tokenizer_config.json      inline chat_template 없음
```

병합 모델 크기는 약 **1.4GB**입니다.

#### 7. 평가 — SP heldout

##### 7.1 평가 코드의 핵심

전체 logits를 계산하면 4060 Ti에서도 OOM이 납니다. 마지막 토큰만 계산해야 합니다.

```python
logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
probs = torch.softmax(logits.float(), dim=-1)
pred = tok.decode([int(probs.argmax())])
```

배치는 4, 편집은 불가, 입력은 학습과 동일하게 `add_special_tokens=False`, `max_length=1240`입니다.

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/wnf/soup-venv/bin/python /tmp/opencode/eval_bc3aug_merged.py
```

소요 시간은 약 **798.5초**였습니다.

##### 7.2 전체 결과

```text
overall = 10,442 / 11,253 = 92.79%
mean confidence = 0.9528
```

로그에 `invalid = 11253`가 찍히지만, 이는 스크립트의 라벨 목록을 `:5`, `b:c3` 같은 SP 키로 적어둔 탓입니다. 실제 예측은 모두 `A~I`였고 **진짜 invalid는 0건**입니다.

##### 7.3 클래스별 결과와 해석

| SP 기호 | 문자 | n | Recall | Precision |
|---|---|---:|---:|---:|
| `:5` | A | 3,779 | 98.2% | 99.5% |
| `:s5` | B | 99 | 66.7% | 88.0% |
| `:c3` | C | 1,016 | 91.8% | 83.3% |
| **`b:c3`** | **D** | **116** | **99.1% (115/116)** | **85.8%** |
| `:F3` | E | 4,076 | 96.0% | 90.2% |
| `x:c3` | F | 298 | 34.6% | 74.1% |
| `:L5` | G | 1,751 | 89.2% | 94.3% |
| `:L3` | H | 55 | 49.1% | 60.0% |
| `:sL3` | I | 63 | 19.0% | 92.3% |

핵심 해석:

- **`b:c3` recall 99.1%** → 이번 보강의 목표는 달성됐습니다.
- 다만 precision이 85.8%라, `b:c3`로 과 예측하는 경향(134개 예측 중 115개 정답)이 있습니다.
- 희소 클래스 `x:c3`, `:L3`, `:sL3`, `:s5`가 약합니다. 표본이 55~298건이라 통계도 불안정하지만, 데이터 불균형의 영향이 큽니다.
- 가장 잦은 혼동은 `G → E`(:L5 → :F3, 156건), `F → C`(x:c3 → :c3, 106건)입니다.

##### 7.4 함께 진행된 프로덕션 MLP 교체

SP 판정은 LLM만으로 하지 않습니다. CPU MLP 분류기도 같은 데이터로 재학습해 프로덕션 경로를 교체했습니다.

```text
train acc      : 0.9464
heldout acc    : 0.9158
b:c3 recall    : 105 / 116 = 90.5%
b:c3 precision : 105 / 144 = 72.9%

교체 파일 : /home/wnf/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt
md5       : 69ceaf7e18a74211a07a696575600cf3
백업      : sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726
```

서비스 재시작 후 `/health` ok, `:c3/b:c3` → `b:c3` (conf 0.9932), `:5` → `:5` (conf 1.0)를 확인했습니다.

```bash
# 롤백이 필요하면
cp /home/wnf/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726 \
   /home/wnf/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt
sudo systemctl restart sp-mlp.service
```

#### 8. 배포 — GGUF → Ollama

##### 8.1 양자화 선택

비교 대상 `tev1:0.8b`가 **Q8_0**이므로, 공정한 비교를 위해 파인튜닝 모델도 `q4_k_m`이 아니라 **Q8_0**으로 내보냈습니다.

```bash
/home/wnf/soup-venv/bin/soup export \
  --model /tmp/opencode/soup-v4-rc-08b-bc3aug-merged \
  --format gguf --quant q8_0 \
  --output /tmp/opencode/tev1-sp-bc3aug-ollama-q8_0.gguf \
  --llama-cpp /home/wnf/.soup/llama.cpp-ollama
```

##### 8.2 llama.cpp 버전 불일치

첫 export는 실패했습니다. 순서는 다음과 같습니다.

```text
1) llama.cpp 없음 → ~/.soup/llama.cpp 로 clone
2) gguf/sentencepiece/protobuf 자동 설치 실패 (soup-venv에 pip 없음)
   → 하지만 convert 자체는 성공
3) llama-quantize 바이너리 없음 → 직접 빌드
```

빌드 명령:

```bash
cd /home/wnf/.soup/llama.cpp
cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-quantize -j 8
```

하지만 재생성한 GGUF를 Ollama가 로드하지 못했습니다.

```text
error loading model: check_tensor_dims:
tensor 'blk.0.ssm_beta.weight' not found
```

원인은 **컨버터와 런타임의 llama.cpp 버전 불일치**였습니다.

| 구분 | 생성된 텐서 |
|---|---|
| 최신 llama.cpp (1.0.0-dev) | `ssm_beta_alpha.weight [1024, 32]` (합쳐짐) |
| Ollama 0.40.13 런타임 | `ssm_alpha.weight [1024, 16]` + `ssm_beta.weight [1024, 16]` (분리) |

Ollama 0.40.13은 llama.cpp **b11963**을 사용합니다. 해당 태그로 별도 클론을 받아 컨버터를 맞췄습니다.

```bash
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git \
  /home/wnf/.soup/llama.cpp-ollama
cd /home/wnf/.soup/llama.cpp-ollama
git fetch --depth 1 origin tag b11963
git checkout --detach b11963
```

재export 결과 `ssm_alpha`/`ssm_beta`가 분리된 320개 텐서가 만들어졌고 로드됐습니다.

##### 8.3 decision 메타데이터 주입

두 번째 문제는 Ollama가 모델을 일반 completion 모델로 인식한 것이었습니다.

```text
before: Capabilities = tools, thinking, completion
after : Capabilities = decision
```

base GGUF에는 다음 메타데이터가 있었지만, export본에는 없었습니다.

```text
qwen35.decision.type          = pplx-decider
tokenizer.chat_template.systemone = <systemone jinja>
tokenizer.chat_templates      = ['systemone']
general.tags                  = [..., 'decision-model', ...]
```

`gguf==0.19.0`을 설치한 뒤 base GGUF에서 값을 읽어 새 GGUF에 주입했습니다.

```bash
uv pip install --python /home/wnf/soup-venv/bin/python gguf
/home/wnf/soup-venv/bin/python /tmp/opencode/patch_gguf_decision.py
```

##### 8.4 최종 등록과 검증

```bash
printf 'FROM /tmp/opencode/tev1-sp-bc3aug-ollama-decision-q8_0.gguf\n' \
  > /tmp/opencode/Modelfile.bc3aug-final

ollama create tev1-sp-bc3aug:latest -f /tmp/opencode/Modelfile.bc3aug-final
```

검증:

```text
ollama show tev1-sp-bc3aug:latest
  architecture   qwen35
  quantization   Q8_0
  capabilities   decision

/v1/systemone 호출 → 정상 choice + probability 반환
```

깨진 임시 모델 `tev1-sp-bc3aug-dec:latest`는 삭제했습니다.

#### 9. 부수 실험 — 뉴스 sentiment 유지도

##### 9.1 4B 사전 실험

먼저 4B에서 같은 현상이 있는지 확인했습니다.

| 모델 | Accuracy | Bullish P/R/F1 | 신규 오류 |
|---|---:|---:|---:|
| `tev1:latest` | 190/200 = 95.0% | 0.933 / 0.970 / 0.951 | — |
| `tev1-neely:latest` | 192/200 = 96.0% | 0.951 / 0.970 / 0.960 | 0 |

##### 9.2 0.8B 동일 조건 비교

조건은 다음과 같습니다.

```text
데이터  : zeroshot/twitter-financial-news-sentiment validation
표본    : 상승 100 + 하락 100 = 200건 (seed 42)
프롬프트: "Is the sentiment of this stock-related news bullish or bearish?"
디코딩 : temperature=0, enable_thinking=false
양자화 : 양쪽 모두 Q8_0
인터페이스: Ollama /v1/systemone
```

| 모델 | Accuracy | Bullish P/R/F1 | 평균 confidence |
|---|---:|---:|---:|
| `tev1:0.8b` (base) | 186/200 = **93.0%** | 0.898 / 0.970 / 0.933 | 0.816 |
| `tev1-sp-bc3aug:latest` | 185/200 = **92.5%** | 0.890 / 0.970 / 0.928 | 0.969 |

오류 분석:

```text
base 오류          : 14건
파인튜닝 오류      : 15건
공통 오류          : 14건
base-only 해결     : 0건
파인튜닝 신규 오류 : 1건

"BJ's Wholesale lowers 2019 sales outlook ..."
  gold=하락, base=하락, 파인튜닝=상승
```

##### 9.3 결과와 해석

```text
4B  : 95.0% → 96.0%   (+1.0%p, 오히려 개선)
0.8B: 93.0% → 92.5%   (-0.5%p, 신규 오류 1건)
```

즉 **“0.8B도 degradation이 별로 없지 않을까”라는 추측은 맞았습니다.** 다만 4B처럼 개선되진 않았고, 미세한 손실 수준입니다.

##### 9.4 confidence calibration 경고

정확도는 0.5%p 빠졌지만 평균 confidence는 **0.816 → 0.969**로 크게 올라갔습니다. SP 특화 학습이 확률 분포를 더 뾰족하게 만들었습니다.

따라서 이 모델을 sentiment 확률값 자체를 신호로 쓰는 파이프라인에 넣을 때는 **threshold 재보정이 필요**합니다. 정답 여부만 보면 괜찮지만, “얼마나 확신하는가”는 이미 base와 다른 스케일입니다.

주의: Ollama의 `tev1:0.8b`는 25-block + nextn 구조이고, 학습에 쓴 HF `Tev1-0.8B-experimental`은 24-block입니다. 엄밀한 “동일 base 대비”가 아니라 **실제 서비스 중인 0.8B 대비 실용적 degradation**으로 해석해야 합니다.

#### 10. 함정과 교훈

##### 10.1 4090 OOM

4090에 이미 19.70GiB가 점유된 상태에서 학습을 올리면 OOM입니다. `CUDA_VISIBLE_DEVICES=1`로 4060 Ti를 명시하고, 로그의 `Device:` 줄을 반드시 확인하세요.

##### 10.2 stale log

중지된 이전 실행의 로그가 같은 파일에 남아 있었습니다. 완료 판정은 로그가 아니라 **`soup_run.json`의 run_id와 adapter mtime**으로 해야 합니다.

##### 10.3 logits 전체 OOM

`model(**enc).logits[:, -1, :]`는 시퀀스 전체 logits를 만듭니다. `logits_to_keep=1`을 쓰지 않으면 4060 Ti도 버티지 못합니다.

##### 10.4 merge 경로 제약

`--output`은 cwd 하위, `--adapter`는 `adapter_config.json`이 직접 있는 디렉터리. 이 두 가지만 기억하면 merge에서 헤매지 않습니다.

##### 10.5 GGUF 컨버터 ≠ 런타임

“GGUF로 변환했다”가 끝이 아닙니다.

```text
변환 성공 → capability 없음 (400)
capability 주입 → 텐서 이름 불일치 (500)
런타임 버전 맞춤 → 정상
```

세 관문을 모두 통과해야 `/v1/systemone`이 동작합니다.

##### 10.6 클래스 라벨 표기

평가 스크립트에서 예측을 SP 키(`:5`)로 비교하면 실제 예측이 `A`일 때 전부 invalid로 집계됩니다. 데이터의 `output`이 문자라는 사실을 기억하고 라벨 맵을 명시하세요.

#### 11. 산출물

| 산출물 | 경로 / 이름 |
|---|---|
| 학습 config | `/tmp/opencode/soup_v4_rc_08b_bc3aug.yaml` |
| LoRA adapter | `/tmp/opencode/soup-v4-rc-08b-bc3aug-out/tev1-v4-rc-primary-08b-bc3aug-soup` |
| 병합 모델 | `/tmp/opencode/soup-v4-rc-08b-bc3aug-merged` |
| SP 예측 | `/tmp/opencode/eval_bc3aug_merged_preds.jsonl` |
| 최종 GGUF | `/tmp/opencode/tev1-sp-bc3aug-ollama-decision-q8_0.gguf` |
| Ollama 모델 | `tev1-sp-bc3aug:latest` |
| sentiment 상세 | `/tmp/opencode/sentiment_base08_eval.jsonl`, `/tmp/opencode/sentiment_finetuned08_eval.jsonl` |
| 프로덕션 MLP | `/home/wnf/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt` |
| MLP 백업 | `...pt.bak.20261121_171726` |

#### 12. 재현 명령 전체 순서

```bash
# 0) 환경 확인
CUDA_VISIBLE_DEVICES=1 /home/wnf/soup-venv/bin/python -c \
  'import torch; print(torch.cuda.get_device_name(0))'

# 1) 학습
CUDA_VISIBLE_DEVICES=1 /home/wnf/soup-venv/bin/soup train \
  --config /tmp/opencode/soup_v4_rc_08b_bc3aug.yaml --yes

# 2) 병합
cd /tmp/opencode
CUDA_VISIBLE_DEVICES=1 /home/wnf/soup-venv/bin/soup merge \
  --adapter /tmp/opencode/soup-v4-rc-08b-bc3aug-out/tev1-v4-rc-primary-08b-bc3aug-soup \
  --output soup-v4-rc-08b-bc3aug-merged --dtype bfloat16

# 3) chat template 복원
python3 /tmp/opencode/restore_tev1_chat_template.py \
  /tmp/opencode/soup-v4-rc-08b-bc3aug-merged

# 4) SP heldout 평가
CUDA_VISIBLE_DEVICES=1 /home/wnf/soup-venv/bin/python \
  /tmp/opencode/eval_bc3aug_merged.py

# 5) GGUF export (Ollama 런타임 버전!)
/home/wnf/soup-venv/bin/soup export \
  --model /tmp/opencode/soup-v4-rc-08b-bc3aug-merged \
  --format gguf --quant q8_0 \
  --output /tmp/opencode/tev1-sp-bc3aug-ollama-q8_0.gguf \
  --llama-cpp /home/wnf/.soup/llama.cpp-ollama

# 6) decision 메타데이터 주입
uv pip install --python /home/wnf/soup-venv/bin/python gguf
/home/wnf/soup-venv/bin/python /tmp/opencode/patch_gguf_decision.py

# 7) Ollama 등록
ollama create tev1-sp-bc3aug:latest \
  -f /tmp/opencode/Modelfile.bc3aug-final

# 8) sentiment A/B
MODELS=tev1:0.8b OUT=/tmp/opencode/sentiment_base08_eval.jsonl \
  /home/wnf/soup-venv/bin/python /tmp/opencode/eval_sentiment_generic.py
MODELS=tev1-sp-bc3aug:latest OUT=/tmp/opencode/sentiment_finetuned08_eval.jsonl \
  /home/wnf/soup-venv/bin/python /tmp/opencode/eval_sentiment_generic.py
```

#### 부록 A. 파일 지도

```text
/tmp/opencode/
├── soup_v4_rc_08b_bc3aug.yaml
├── soup-v4-rc-08b-bc3aug-out/
│   └── tev1-v4-rc-primary-08b-bc3aug-soup/     # LoRA adapter
├── soup-v4-rc-08b-bc3aug-merged/               # bf16 merged HF
├── soup_bc3aug_merge.log
├── eval_bc3aug_merged.py
├── eval_bc3aug_merged.log
├── eval_bc3aug_merged_preds.jsonl
├── dump_gguf_metadata.py
├── patch_gguf_decision.py
├── tev1-sp-bc3aug-ollama-q8_0.gguf             # 원본 export
├── tev1-sp-bc3aug-ollama-decision-q8_0.gguf    # 최종 GGUF
├── Modelfile.bc3aug-final
├── eval_sentiment_generic.py
├── sentiment_base08_eval.jsonl
├── sentiment_finetuned08_eval.jsonl
├── compare_bc3aug_mlp.py
└── mlp_bc3aug_cpu_train.log

/home/wnf/.soup/
├── llama.cpp/                # 최신 컨버터 (호환 실패)
└── llama.cpp-ollama/         # b11963 컨버터 (최종 사용)

/home/wnf/tev1-finetune/mlp/
├── sp_mlp_nolen_v4_rc_primary.pt
└── sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726
```
