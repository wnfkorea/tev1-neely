# SP decision MLP (Tev1-4B decision model 증류)

Tev1 SP decision model(4B LLM)의 결정을 **CPU MLP**로 증류한 프로덕션 모델.
Ollama / llama.cpp / GPU 없이, feature 벡터 → SP 문자 분류를 **~1.6ms/req**로 처리한다.

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
  train_mlp.py    MLP 학습 (sp-systemone-train.jsonl → sp_mlp.pt)
  serve_mlp.py    HTTP 서버 (/v1/systemone 드롭인 교체)
  eval_mlp.py     held-out 평가 + 지연 측정
  sp_mlp.pt       학습된 모델 (state_dict + mu/sd + keys + feature order)
  README.md       이 문서
```

## 재현

```bash
cd ~/tev1-finetune
# 1) 학습 (20k train, 3k held-out)
./venv/bin/python mlp/train_mlp.py \
  --train sp-systemone-train.jsonl \
  --heldout sp-systemone-heldout.jsonl \
  --out mlp/sp_mlp.pt

# 2) 평가 (held-out 정확도 + 지연)
./venv/bin/python mlp/eval_mlp.py --model mlp/sp_mlp.pt --heldout sp-systemone-heldout.jsonl
```

## 서버 기동

```bash
cd ~/tev1-finetune
./venv/bin/python mlp/serve_mlp.py --model mlp/sp_mlp.pt --host 0.0.0.0 --port 8095
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
