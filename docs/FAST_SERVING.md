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
작은 MLP(512-512-10, ~1.4M 파라미터)가 held-out 3000에서 **1.0000**을 낸다.

- **서버**: `mlp/serve_mlp.py` — Ollama `/v1/systemone`과 **동일 요청/응답 형식** (드롭인 교체)
- **정확도**: held-out 3000/3000 = 1.0000 (10개 클래스 전부 recall 1.000)
- **속도**: CPU에서 **~1.6ms/req (순차)**, 동시 4 시 **~1060 req/s**
- **연동**: `sp.symbolModel`을 MLP 서버 host로 변경 (클라이언트 코드 변경 없음)

자세한 내용: [`mlp/README.md`](../mlp/README.md)

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
- 프로덕션: `mlp/` (train/serve/eval + `sp_mlp.pt` + `sp-mlp.service`)
