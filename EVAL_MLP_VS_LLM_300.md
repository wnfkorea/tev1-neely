### Production MLP vs `tev1-0.8b-neely:latest` — heldout 300개 Inference 비교

비교 시점: 2026-11-22  
대상: production CPU MLP(`sp_mlp_nolen_v4_rc_primary.pt`)와 production Ollama decision 모델(`tev1-0.8b-neely:latest`)

#### 1. 비교 방법

| 항목 | 내용 |
|---|---|
| 데이터 | `datasets/systemone/v4_rc_bc3aug_systemone-heldout.jsonl` 11,253행 중 무작위 300개 |
| 샘플링 | `random.Random(20261122).sample(rows, 300)` |
| 공통 입력 | heldout 레코드의 `state` + SP question/criteria/choices(A~J, `:5`~`Invalid`) |
| LLM endpoint | `http://127.0.0.1:11434/v1/systemone`, model `tev1-0.8b-neely:latest` |
| MLP endpoint | `http://127.0.0.1:8095/v1/systemone`, production MLP 서버 |
| 정답 | heldout 레코드의 `answer` 라벨(A~J) |
| 원본 결과 | `eval_mlp_vs_llm_300.jsonl` (300행) |

양쪽 모두 동일한 `/v1/systemone` 요청 형식을 사용했고, 응답의 `choice`를 label/key 매핑으로 통일해 정답과 비교했습니다.

#### 2. 전체 결과

| 지표 | LLM (`tev1-0.8b-neely`) | MLP (production) |
|---|---:|---:|
| 정확도 | **280/300 = 93.33%** | **283/300 = 94.33%** |
| 평균 confidence | 0.9452 | 0.9219 |
| 정답 시 confidence | 0.9660 | 0.9370 |
| 오답 시 confidence | 0.6530 | 0.6711 |
| 평균 응답 시간 | 105.4 ms | **2.44 ms** |
| 처리 속도 | 1x | **약 43x 빠름** |

| paired 결과 | 건수 |
|---|---:|
| 둘 다 정답 | 274 |
| LLM만 정답 | 6 |
| MLP만 정답 | 9 |
| 둘 다 오답 | 11 |
| 예측 일치 | 282/300 = 94.0% |

MLP가 1%p 높지만, 불일치 15건 중 LLM만 맞힌 6건과 MLP만 맞힌 9건을 비교한 McNemar 검정 p값은 약 0.61입니다. **통계적으로 유의미한 차이는 아닙니다.**

#### 3. 클래스별 정확도

| 라벨 | SP key | 표본 | LLM 정확도 | MLP 정확도 |
|---|---|---:|---:|---:|
| A | `:5` | 96 | 99.0% | 97.9% |
| B | `:s5` | 3 | 100.0% | 100.0% |
| C | `:c3` | 29 | **86.2%** | 82.8% |
| D | `b:c3` | 2 | 100.0% | 100.0% |
| E | `:F3` | 115 | 98.3% | 97.4% |
| F | `x:c3` | 6 | **16.7%** | **83.3%** |
| G | `:L5` | 46 | 87.0% | **91.3%** |
| H | `:L3` | 2 | 50.0% | 50.0% |
| I | `:sL3` | 1 | 0.0% | 0.0% |

`F=x:c3`에서 MLP가 5/6, LLM이 1/6으로 차이가 가장 큽니다. 반대로 `A=:5`, `C=:c3`에서는 LLM이 소폭 앞섭니다. `D/H/I`는 표본이 1~2건이므로 클래스별 결론으로 일반화하기 어렵습니다.

#### 4. 불일치 분석

불일치 18건의 정답 분포:

| 정답 | 불일치 건수 |
|---|---:|
| C `:c3` | 6 |
| F `x:c3` | 4 |
| E `:F3` | 3 |
| G `:L5` | 2 |
| A `:5` | 2 |
| H `:L3` | 1 |

주요 패턴:

| 패턴 | 건수 | 해석 |
|---|---:|---|
| `F → LLM C / MLP F` | 3 | LLM이 `x:c3`를 `:c3`로 오판 |
| `C → LLM C / MLP E` | 2 | MLP가 `:c3`를 `:F3`로 오판 |
| `G → LLM E / MLP G` | 2 | LLM이 `:L5`를 `:F3`로 오판 |

둘 다 틀린 11건 중 8건은 같은 오답 label이었고, 3건은 서로 다른 오답이었습니다.

#### 5. 해석

- **정확도**: MLP가 1%p 앞서지만 오차 범위 내이며, 전체 heldout 기준 수치(이전 측정: LLM 92.79%, MLP 91.58%)와도 일관됩니다.
- **속도**: MLP가 약 43배 빠르므로 production 1차 판정 경로로는 MLP가 적합합니다.
- **confidence**: LLM이 더 높은 confidence를 주지만 정확도는 낮아, 이 샘플에서는 LLM이 다소 과신하는 경향이 있습니다.
- **권고**: MLP를 기본 판정기로 사용하고, LLM은 low-confidence 케이스 또는 MLP/LLM 불일치 케이스의 2차 판정기로 활용하는 조합이 합리적입니다.

#### 6. 재현

```bash
# LLM
curl -s http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"tev1-0.8b-neely:latest","state":{...},"questions":{"sp":{...}}}'

# MLP
curl -s http://127.0.0.1:8095/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"sp-mlp","state":{...},"questions":{"sp":{...}}}'
```

전체 300건의 `gold`, 예측 label/key, confidence, latency는 `eval_mlp_vs_llm_300.jsonl`에 있습니다.
