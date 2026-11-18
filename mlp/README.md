# SP decision MLP (Tev1-4B decision model 증류)

Tev1 SP decision model(4B LLM)의 결정을 **CPU MLP**로 증류한 프로덕션 모델.
Ollama / llama.cpp / GPU없이, feature 벡터 → SP 문자 분류를 **~1.6ms/req**로 처리한다.

## 멀티마켓 모델 (프로덕션)

`sp_mlp_multi.pt`는 **한국장(주식/코인) + 미국장(주식/코인)** 전체를 커버하는 모델이다.
과제(SP 분류)는 시장 독립적이고, 시장별 차이는 **feature 분포**(변동폭/거래 세션/가격대)뿐이므로,
시장별 프로파일로 데이터 생성기를 확장(`generate_sft_dataset_multi.py`)해 재학습했다.

| 시장 프로파일 | length_factor | base_time | base_price |
|--------------|---------------|-----------|------------|
| `kr_stock` (KOSPI/KOSDAQ) | 5–15% | 3–8 bars | 5k–50k |
| `us_stock` (S&P/Nasdaq) | 3–10% | 3–8 bars | 10–500 |
| `coin` (crypto 24/7) | 10–40% | 3–30 bars | 100–50k |

- 학습: 50k (3시장 혼합), 평가: 9k (3시장 × 3k)
- **시장별 held-out 정확도 전부 1.0000** (kr 3000/3000, us 3000/3000, coin 3000/3000)
- `sp_mlp.pt`(한국장 전용)는 레거시 — 프로덕션은 `sp_mlp_multi.pt` 사용

## 왜 MLP인가

Tev1 SP 태스크는 **standardised monowave feature 벡터(36차원) → 10개 SP 문자 분류**이다.
4B LLM이 과잉이고, 작은 MLP가 held-out set에서 **동일 정확도(1.0000)**를 낸다.
자세한 분석: [`../docs/FAST_SERVING.md`](../docs/FAST_SERVING.md)

## 실측 (4060 Ti 머신, CPU 추론)

| 구성 | 지연 | throughput | 정확도 |
|------|------|------------|--------|
| Ollama `/v1/systemone` (baseline) | 200ms | 5 req/s | 1.000 |
| llama-server 직접 (Q8_0) | 176ms | 5.7 req/s | 1.000 |
| **SP MLP (이 모델, 순차)** | **1.6ms** | **606 req/s** | **1.0000** |
| SP MLP (동시 4) | — | **1060 req/s** | 1.0000 |

held-out 3000개에서 **3000/3000 = 1.0000** (10개 클래스 전부 recall 1.000).

## A/B 검증 (2026-11-18, held-out 3000 전체)

| 서버 | 정확도 | p50 | throughput |
|------|--------|-----|------------|
| **MLP `http://192.168.55.229:8095`** | **3000/3000 = 1.0000** | **3.6ms** | **1035 req/s** |
| Ollama `http://192.168.55.229:11434` | 2998/3000 = 0.9993 | 198.4ms | 5.0 req/s |

- Ollama 오답 2건: `b:c3`→`:c3`, `Invalid_Neely_Rule_Violation`→`:c3` (MLP는 모두 정답)
- **결론: MLP가 정확도·속도 모두 우위 → 프로덕션 컷오버 완료**

## 파일

```
mlp/
  train_mlp.py          MLP 학습
  serve_mlp.py          HTTP 서버 (/v1/systemone 드롭인 교체)
  eval_mlp.py           held-out 평가 + 지연 측정
  sp_mlp.pt             한국장 전용 모델 (레거시)
  sp_mlp_multi.pt       멀티마켓 모델 (프로덕션 — 한국/미국 주식+코인)
  sp-mlp.service        systemd 유닛 (0.0.0.0:8095, sp_mlp_multi.pt)
  README.md             이 문서

generate_sft_dataset_multi.py   멀티마켓 데이터 생성기 (시장 프로파일)
sp-systemone-multi-train.jsonl      50k 학습셋 (3시장 혼합)
sp-systemone-multi-heldout-*.jsonl  시장별 3k held-out
```

## 재현

```bash
cd ~/tev1-finetune
# 1) 멀티마켓 데이터 생성 (3시장 혼합)
./venv/bin/python generate_sft_dataset_multi.py --count 50000 --seed 20261118 \
  --out sp-systemone-multi-train.jsonl
for m in kr_stock us_stock coin; do
  ./venv/bin/python generate_sft_dataset_multi.py --count 3000 --seed 424243 \
    --market $m --out sp-systemone-multi-heldout-$m.jsonl
done

# 2) 학습 (50k train, 9k held-out)
cat sp-systemone-multi-heldout-*.jsonl > sp-systemone-multi-heldout.jsonl
./venv/bin/python mlp/train_mlp.py \
  --train sp-systemone-multi-train.jsonl \
  --heldout sp-systemone-multi-heldout.jsonl \
  --out mlp/sp_mlp_multi.pt

# 3) 평가 (시장별 held-out 정확도 + 지연)
for m in kr_stock us_stock coin; do
  ./venv/bin/python mlp/eval_mlp.py --model mlp/sp_mlp_multi.pt \
    --heldout sp-systemone-multi-heldout-$m.jsonl
done
```

## 서버 기동

```bash
cd ~/tev1-finetune
./venv/bin/python mlp/serve_mlp.py --model mlp/sp_mlp_multi.pt --host 0.0.0.0 --port 8095
```

- `GET /health` → `{"status":"ok"}`
- `POST /v1/systemone` → Ollama와 **동일 요청/응답 형식** (드롭인 교체)

## systemd 서비스 (프로덕션)

`mlp/sp-mlp.service`를 설치:

```bash
sudo cp ~/tev1-finetune/mlp/sp-mlp.service /etc/systemd/system/sp-mlp.service
sudo systemctl daemon-reload
sudo systemctl enable --now sp-mlp
systemctl status sp-mlp
```

- `0.0.0.0:8095` 리스닝, `Restart=always` (자동 재기동)
- 컨테이너(app)는 `http://192.168.55.229:8095`로 접근

### 요청/응답 형식 (Ollama `/v1/systemone` 호환)

```jsonc
// 요청
{ "model": "sp-mlp", "state": { ...monowave feature... },
  "questions": { "sp": { "type": "choice", "instructions": "...",
                         "criteria": {...}, "choices": [...] } } }
// 응답
{ "model": "sp-mlp",
  "answers": { "sp": { "type": "choice", "choice": ":F3",
                       "probabilities": {":F3": 0.99, ...}, "confidence": 0.99 } },
  "usage": { "input_tokens": 0, "output_tokens": 0, "inference_ms": 1.6 } }
```

## 앱 연동 (stock-twin-kr) — 완료 (2026-11-18)

- `src/lib/config.ts`: `sp.host` 추가 — `process.env.SP_MLP_HOST || process.env.OLLAMA_HOST || 'http://localhost:11434'`
- `src/services/analysis/elliott-wave.service.ts`: SP 전용 `OllamaClient`
  (`new OllamaClient({ host: config.sp.host })`)를 만들어 `inferSPSymbolWithModel` /
  `inferSPBatchWithModel` / `classifyPatternWithModel`이 사용. sentiment 등 다른 LLM 호출은
  기존 `ollamaClient` 싱글턴 유지.
- kil2_container `.env`: `SP_MLP_HOST=http://192.168.55.229:8095`
- 실동작 확인: `/api/elliott-wave` 호출 시 SP batch 157/157 valid, 전체 API ~500ms
  (Ollama 시절엔 SP 호출만 30초+)

`unloadSystemOne()`은 MLP 서버에 필요 없음 (GPU 회수 대상 아님).

## 주의

- **모델 대체**: "tev1-neely를 쓰는" 게 아니라 **대체**하는 것이다. feature가 앱에서
  이미 생성되므로 연동은 쉽다.
- **정확도**: held-out 3000에서 1.0000. 실데이터 분포가 변하면 재학습 필요
  (`train_mlp.py` 재실행).
- **feature 일관성**: `train_mlp.py` / `serve_mlp.py` / `eval_mlp.py`의 feature layout
  (DIRF/NUMF/BOOLF)이 동일해야 한다. 변경 시 세 파일 모두 수정.
