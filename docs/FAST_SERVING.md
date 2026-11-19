# tev1-neely — Ollama 없이 더 빠르게 쓰는 방법 (측정 결과)

작성일: 2026-11-18

## 결론 (TL;DR)

`tev1-neely` decision model은 **965 토큰 prefill + 1글자(A–J) 출력**의 분류 작업이다.
`/v1/systemone` 응답의 `output_tokens: 0` — **전부 prefill**이다.

Ollama를 거치지 않고 빠르게 쓰는 방법, **실측** 결과:

| # | 방법 | 실측 지연 | 정확도 | 판정 |
|---|------|-----------|--------|------|
| 0 | Ollama `/v1/systemone` (baseline) | **200ms** | 1.000 | — |
| 1 | **llama.cpp `llama-server` 직접** (Q8_0, flash-attn) | **176ms** | 1.000 | ✅ Ollama 제거 |
| 2 | Q4_K_M 재양자화 (2.7GB) | 186ms | **0.920** | ❌ 정확도 급락, 속도도 안 빠름 |
| 3 | ONNX (onnxruntime) | **내보내기 실패** | — | ❌ Triton 커널 비호환 |
| 4 | 배치 처리 (N 파동 1 forward) | N배 속도 | 1.000 | ✅ 앱 변경 필요 |
| 5 | **MLP 분류기 증류 (CPU)** | **~1.6ms (순차)** | **1.0000** | ✅✅ **프로덕션 완성** |

**권고: 방법 5 (MLP 증류) — 프로덕션으로 완성됨 (`mlp/`).**
Ollama/llama.cpp/GPU 없이 CPU에서 **~125-200배** 속도, held-out 3000에서 **1.0000**.
방법 1(llama-server)은 "tev1-neely를 그대로 쓰되 Ollama만 제거"할 때 사용.

## 실측 상세 (4060 Ti 머신, held-out set)

| 구성 | mean | p50 | p90 | throughput | acc |
|------|------|-----|-----|------------|-----|
| Ollama `/v1/systemone` | 200ms | 199ms | 202ms | 5.00 req/s | 1.000 |
| llama-server `/v1/systemone` (Q8_0) | **176ms** | 176ms | 178ms | **5.68 req/s** | 1.000 |
| llama-server (Q4_K_M) | 186ms | 186ms | 188ms | 5.37 req/s | **0.920** |
| **SP MLP 서버 (순차)** | **1.6ms** | 1.6ms | 2.1ms | **606 req/s** | **1.0000** |
| SP MLP 서버 (동시 4) | — | — | — | **1060 req/s** | 1.0000 |

llama-server 설정: `-fa on -ctk q4_0 -ctv q4_0 -b 512 -ub 512 --parallel 1 -c 4096`
(`-fa auto`, `-ctk f16`, `-ctk q8_0`도 모두 ~175-177ms — prefill이 하한)

## 왜 빠르냐 / 왜 더 안 빠르냐

- **Ollama 오버헤드 제거**: 스케줄러, 모델 관리, HTTP 레이어, 템플릿 렌더링.
- **prefill이 하한**: 965 토큰 forward. 단일 시퀀스 prefill은 가중치 재독출 bound.
  flash-attn / KV 양자화 / batch 설정으로 더 이상 안 빠짐 (측정 확인).
- **Q4_K_M이 안 된 이유**: (1) 정확도 0.920으로 급락 (SP 분류는 양자화에 민감),
  (2) prefill이 compute+bandwidth 혼합이라 모델이 작아도 크게 안 빠름.

## 방법 1: llama.cpp `llama-server` 직접 (권고)

Ollama 없이 GGUF를 직접 서빙. 이 llama.cpp 버전(2a4daf0)은 **네이티브 `/v1/systemone`
엔드포인트**를 지원한다 (pplx-decider 모델 자동 감지).

```bash
cd ~/tev1-finetune
# llama.cpp 빌드 (pinned + patched, CUDA)
cmake -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF
ninja -C llama.cpp/build llama-server

# 서버 기동 (4060 Ti = torch device 1)
CUDA_VISIBLE_DEVICES=1 ./llama.cpp/build/bin/llama-server \
  --model tev1-neely-sp-bf16-v4-q8_0-decision.gguf \
  --host 127.0.0.1 --port 8090 \
  -c 4096 -fa on -ctk q4_0 -ctv q4_0 -b 512 -ub 512 --parallel 1
```

앱 연동: `OllamaClient`는 이미 `api:'openai'` 모드 지원. `sp.symbolModel`을
llama-server host로 바꾸고 `/v1/systemone`을 그대로 쓰면 된다 (엔드포인트 호환).

## 방법 3: 배치 처리 (N배 속도)

앱이 파동별로 SP를 1개씩 호출한다. N개 파동을 1번 forward(배치)로 처리하면 N배 속도.
`llama-server`의 `--parallel N` + 앱 측 배치 호출로 구현.

## 방법 5: MLP 증류 (프로덕션 완성 — `mlp/`)

Tev1 SP 태스크는 **36차원 monowave feature → 10개 SP 문자 분류**. 4B LLM이 과잉이고,
작은 MLP(512-512-10, ~1.4M 파라미터)가 held-out에서 **1.0000**을 낸다.

- **서버**: `mlp/serve_mlp.py` — Ollama `/v1/systemone`과 **동일 요청/응답 형식** (드롭인 교체)
- **정확도**: held-out 3000/3000 = 1.0000 (10개 클래스 전부 recall 1.000)
- **속도**: CPU에서 **~1.6ms/req (순차)**, 동시 4 시 **~1060 req/s**
- **연동**: `sp.symbolModel`을 MLP 서버 host로 변경 (클라이언트 코드 변경 없음)

자세한 내용: [`mlp/README.md`](../mlp/README.md)

## 방법 6: 멀티마켓 MLP (한국/미국 주식+코인 — 프로덕션)

SP 분류는 **시장 독립적**이다 (같은 36차원 feature, 같은 10 SP 기호, 같은 Neely 규칙).
시장별 차이는 **feature 분포**(변동폭/거래 세션/가격대)뿐이므로, 시장별 프로파일로
데이터 생성기를 확장(`generate_sft_dataset_multi.py`)해 재학습하면 **단일 MLP**가
모든 시장을 커버한다.

| 시장 프로파일 | length_factor | base_time | base_price |
|--------------|---------------|-----------|------------|
| `kr_stock` (KOSPI/KOSDAQ) | 5–15% | 3–8 bars | 5k–50k |
| `us_stock` (S&P/Nasdaq) | 3–10% | 3–8 bars | 10–500 |
| `coin` (crypto 24/7) | 10–40% | 3–30 bars | 100–50k |

- 학습 50k (3시장 혼합) / 평가 9k (3시장 × 3k)
- **시장별 held-out 정확도 전부 1.0000** (kr 3000/3000, us 3000/3000, coin 3000/3000)
- 프로덕션 모델: `mlp/sp_mlp_multi.pt` (서버/서비스가 이 모델을 사용)

**교훈**: 단일 시장 분포로 학습한 MLP는 그 시장 밖 feature(예: 코인의 큰 변동폭)에서
외삽해 오답을 낸다. 모든 목표 시장을 feature 분포로 커버해야 한다.

## 방법 7: nolen — shape-invariant MLP (프로덕션, 2026-11-19 승격)

**설계 원리: feature 벡터는 과제의 불변군과 일치해야 한다.**

Neely SP 규칙은 차트 축(시간+가격) 재스케일에 불변 → feature도 재스케일 불변 집합만
담아야 한다. 정밀한 구분:

| feature | 종류 | 처리 |
|---------|------|------|
| `*_length` (5개) | 절대 가격 값 (시장 종속 — coin의 100 ≠ stock의 100) | **제거** |
| `*_time_bars` (5개) | 캔들 **개수** (시장 종속 아님 — 5개는 5개) | **유지** |
| ratio / pct / dir / bool | 불변 | 유지 |

- feature: **DIRF(5) + NUMF(9, time_bars 포함) + BOOLF(3)** → null indicator 포함 **27차원**
- 학습 50k (3시장 혼합) / 평가 9k — **시장별 held-out 전부 1.0000**
- **ETH 1D 실데이터(87 wave) A/B**:

| 모델 | feature | multi 정합 | p50 |
|------|---------|-----------|-----|
| **nolen (프로덕션)** | 27차원 | **80%** (70/87) | 2.3ms |
| shape (테스트) | 16차원 (time_bars까지 제거) | 70% (61/87) | 2.3ms |
| multi (레거시) | 36차원 | 100% | 1.8ms |
| ollama (LLM) | 36차원 LLM | 39% | 200ms |

- `time_bars`(개수) 유지가 70% → 80% 정합을 냄 — **개수는 신호, 절대 가격은 노이즈**
- **nolen LLM vs nolen MLP**: **26.4% (23/87)** — ETH와 ^KS11(KOSPI 200)에서 **동일 분기**
  - nolen-llm p50 186ms vs nolen-mlp p50 2.4ms (~77배)
  - 분기 원인: **같은 nolen 17-key state를 다르게 읽는 것** (feature 매칭 vs JSON 의미적 추론)
  - nolen-llm은 `:5`/`Invalid` 과다, nolen-mlp는 `:sL3`/`:c3` 과다
- **nolen redesign 원리(시장 의존 length 제거)는 MLP/LLM 공통** — `tev1-neely`가 nolen로 승격됨
- 프로덕션: `mlp/sp_mlp_nolen.pt` (systemd `sp-mlp.service`, 0.0.0.0:8095)

**교훈**: shape-invariant feature는 (1) 시장 독립성, (2) 실데이터 정합, (3) 속도
(2.3ms)를 동시에 확보. 절대 가격 length 제거가 핵심.

### `tev1-neely` 승격 (2026-11-19)

nolen LLM이 **새 `tev1-neely`**로 등록됨 (기존 36-dim `tev1-neely` 대체):

```bash
ollama rm tev1-neely:latest
ollama cp tev1-neely-nolen:latest tev1-neely:latest
```

- `tev1-neely:latest`와 `tev1-neely-nolen:latest`가 **동일 모델 ID**
- `tev1-neely`는 이제 **nolen 17-key state**를 받음 (22-key full state는 사용 불가)
- `tev1-neely-nolen:latest`는 별칭으로 유지

### gapfill v2 승격 (2026-11-19)

`m1_retrace_ratio` 공백 3개([0.50,0.68], [3.67,5.45], [6.62,13.61])를 5,000 샘플로
증강한 **v2**가 프로덕션으로 승격됨. LLM/MLP **동일 combined 54,999 데이터셋**으로 학습.

```bash
# MLP v2 (8095 프로덕션)
./venv/bin/python mlp/train_nolen.py \
  --train sp-systemone-nolen-train-v2.jsonl \
  --heldout sp-systemone-nolen-heldout.jsonl \
  --out mlp/sp_mlp_nolen_v2.pt --epochs 60
sudo sed -i 's|sp_mlp_nolen\.pt|sp_mlp_nolen_v2.pt|' /etc/systemd/system/sp-mlp.service
sudo systemctl daemon-reload && sudo systemctl restart sp-mlp

# Ollama v2 (새 tev1-neely)
ollama rm tev1-neely:latest
ollama cp tev1-neely-nolen-v2:latest tev1-neely:latest
```

- `tev1-neely:latest` = `tev1-neely-nolen-v2:latest` (동일 ID `a8ff6497ed03`)
- `tev1-neely-nolen:latest`(v1, `1fcc9f425de5`)는 v1 대조용으로 유지
- **A/B (v2, 동일 데이터셋)**:
  - ETH: nolen-llm-v2 == nolen-mlp-v2 **46/87 = 0.5287** (v1: 23/87 = 0.2644)
  - ^KS11: **47/89 = 0.5281** (v1: 23/87 = 0.2644)
  - gapfill 증강으로 LLM/MLP 일치율 **2배 상승**
- 남은 ~47% 발산: LLM 의미론적 추론 vs MLP 결정적 패턴매칭 + 실제 시장 분포 차이
- 프로덕션: `mlp/sp_mlp_nolen_v2.pt` (systemd `sp-mlp.service`, 0.0.0.0:8095)

## A/B 검증 (2026-11-18, held-out 3000 전체, 실서버)

| 서버 | 정확도 | p50 | throughput |
|------|--------|-----|------------|
| **MLP `http://192.168.55.229:8095`** | **3000/3000 = 1.0000** | **3.6ms** | **1035 req/s** |
| Ollama `http://192.168.55.229:11434` | 2998/3000 = 0.9993 | 198.4ms | 5.0 req/s |

- Ollama 오답 2건: `b:c3`→`:c3`, `Invalid_Neely_Rule_Violation`→`:c3` (MLP는 모두 정답)
- **결론: MLP가 정확도·속도 모두 우위 → 프로덕션 컷오버 완료**
- 앱 연동: kil2_container `stock-twin-kr`의 SP 호출이 `SP_MLP_HOST=http://192.168.55.229:8095`
  (`config.sp.host` → SP 전용 `OllamaClient`). 실동작 확인: SP batch 157/157 valid,
  전체 `/api/elliott-wave` ~500ms (Ollama 시절엔 SP 호출만 30초+)

## 부록: 실험 산출물 (정리됨)

- `llama.cpp/build/` — 삭제 (소스 `llama.cpp/`만 유지)
- `quantize_q4km.py`, Q4_K_M GGUF — 삭제 (정확도 0.920으로 **폐기**)
- `/tmp` 벤치/로그/ONNX 산출물 — 삭제
- 프로덕션: `mlp/` (train/serve/eval + `sp_mlp_nolen.pt` + `sp-mlp.service`)
- 테스트 서버: `127.0.0.1:8096` (shape), `127.0.0.1:8097` (nolen) — 프로덕션 8095와 분리
