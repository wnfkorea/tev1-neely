### Tev1 Neely SP 파인튜닝 데이터셋 히스토리

이 문서는 `/tmp`, 저장소 루트, `/tmp/opencode`에 흩어져 있던 학습·평가 데이터셋을 한곳으로 모으고, 각 데이터셋이 어떤 LLM/MLP 산출물을 만들었는지 기록한 히스토리다. 기준 시점은 **2026-11-22**이며, production 모델은 `tev1-0.8b-neely:latest`와 `mlp/sp_mlp_nolen_v4_rc_primary.pt`다.

#### 1. 경로 규칙

| 구분 | 경로 | 비고 |
|---|---|---|
| 작업 디렉터리 | `/home/wnf/tev1-finetune` | production 파일은 실파일, 이전 버전은 심볼 링크 |
| 데이터셋 | `~/tev1-finetune/datasets` → 아카이브의 `datasets/` | 78개 파일을 6개 그룹으로 통합 |
| 아카이브 루트 | `/media/wnf/Tamuz2T/106.Dev/tev1-finetune` | Tamuz2T 디스크 |
| production LLM GGUF | `~/tev1-finetune/tev1-0.8b-neely-q8_0-decision.gguf` | 아카이브 실파일을 가리키는 심볼 링크 |
| production MLP | `~/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt` | 실파일(이동하지 않음) |
| production MLP 서빙 | `~/tev1-finetune/mlp/serve_mlp.py`, `v4_features.py`, `vendor/`, `sp-mlp.service` | 실파일(이동하지 않음) |

아카이브 디렉터리 구조:

```text
106.Dev/tev1-finetune/
├── datasets/{raw,sft,systemone,augment,mlp,eval}/
├── models/llm/{lora,merged,gguf}/
├── models/mlp/
├── scripts/{data,train,eval,serving,tools,legacy}/
├── configs/
├── logs/
├── docs/old-docs/
└── tools/patches/
```

##### 1.1 통합된 데이터셋 목록

행 수는 이동 시점 기준 `wc -l` 값이다. 모든 경로는 `~/tev1-finetune/datasets/` 아래에 있다.

| 그룹 | 파일 | 행 수 |
|---|---|---:|
| `raw/` | `eth_wave.json`, `eth_main_waves.json` | 2,799 / 793 |
| | `gui_features.jsonl`, `gui_features_rc.jsonl` | 87 / 87 |
| | `v4_train.jsonl`, `v4_test.jsonl` | 94,169 / 882 |
| | `v4_train_filtered.jsonl`, `v4_test_filtered.jsonl` | 509 / 60 |
| | `v4_full_rc.jsonl` | 113,088 |
| | `v4_train_rc.jsonl`, `v4_heldout_rc.jsonl`, `v4_heldout.jsonl`, `v4_rc_test.jsonl` | 101,779 / 11,309 / 9,568 / 418 |
| `sft/` | `sft-dataset-train4.jsonl`, `sft-dataset-heldout4.jsonl` | 20,000 / 3,000 |
| | `v4_sft_train.jsonl`, `v4_sft_heldout.jsonl` | 93,787 / 9,536 |
| | `v4_sft_train_rc.jsonl`, `v4_sft_heldout_rc.jsonl` | 101,369 / 11,253 |
| | `v4_rc_sft_train.jsonl`, `v4_rc_sft_heldout.jsonl` | 101,369 / 11,253 |
| | `v4_rc_bc3_sft-train.jsonl`, `v4_rc_bc3_sft-heldout.jsonl` | 101,369 / 11,253 |
| | `v4_rc_bc3aug_sft-train.jsonl` | 105,369 |
| `systemone/` | `sp-systemone-train.jsonl`, `sp-systemone-heldout.jsonl` | 20,000 / 3,000 |
| | `sp-systemone-multi-train.jsonl`, `sp-systemone-multi-heldout*.jsonl` | 49,999 / 9,000~3,000 |
| | `sp-systemone-nolen-train*.jsonl`, `sp-systemone-nolen-heldout.jsonl` | 49,999~54,999 / 9,000 |
| | `sp-systemone-nolen-gapfill*.jsonl` | 5,000 / 5,000 |
| | `sp-systemone-nolen-v3-*.jsonl`, `sp-systemone-v3-*.jsonl` | 49,650~54,650 / 9,920 |
| | `sp-real-distribution.json` | 914 |
| | `v4_systemone-train.jsonl`, `v4_systemone-heldout.jsonl` | 93,787 / 9,536 |
| | `v4_systemone-soup-train.jsonl`, `v4_systemone-soup-heldout.jsonl` | 93,787 / 9,536 |
| | `v4_systemone_rc-train.jsonl`, `v4_systemone_rc-heldout.jsonl` | 101,369 / 11,253 |
| | `v4_rc_systemone-*.jsonl` | 101,369 / 11,253 |
| | `v4_rc_bc3_systemone-*.jsonl` | 101,369 / 11,253 |
| | `v4_rc_bc3aug_systemone-*.jsonl` | 105,369 / 11,253 |
| | `*_soup-smoke2k.jsonl` | 2,000 |
| `augment/` | `v4_rc_bc3_augment.jsonl` | 4,000 |
| `mlp/` | `v4_rc_bc3aug_mlp-train.jsonl`, `v4_rc_bc3aug_mlp-heldout.jsonl` | 105,369 / 11,253 |
| `eval/` | `mlp_preds_200.json`, `llm_preds_200.json` | 801 / 801 |
| | `tev1_neely_sentiment_eval.jsonl`, `sentiment_base08_eval.jsonl`, `sentiment_finetuned08_eval.jsonl` | 200 / 200 / 200 |
| | `eval_bc3aug_merged_preds.jsonl`, `sft-v4-test.jsonl` | 11,253 / 12 |
| | `chart001.json`, `features-req.json`, `api-elliott-001.json` | feature 요청 스냅샷 |

#### 2. 데이터셋 → LLM/MLP 생성 히스토리

| 단계 | 입력 dataset | 생성 LLM | 생성 MLP |
|---|---|---|---|
| 0. 기반 분석 | `eth_wave`, `eth_main_waves`, `gui_features`, `sp-real-distribution` | 없음 | 없음 |
| 1. 4B 초기 SFT | `sft-dataset-{train4,heldout4}`, `sp-systemone-{train,heldout}` | `tev1-sp-lora`, `merged-tev1-sp`, `tev1-neely-sp-q8_0.gguf` | `sp_mlp.pt`, `sp_mlp_multi.pt` |
| 2. nolen 계열 | `sp-systemone-nolen-*`, `sp-systemone-nolen-gapfill*` | `tev1-sp-nolen`, `tev1-sp-nolen-v2`, `merged-tev1-sp-nolen{, -v2}`, `tev1-neely-sp-nolen*{f16,q8_0}` | `sp_mlp_nolen.pt`, `sp_mlp_nolen_v2.pt`, `sp_mlp_shape.pt` |
| 3. V3 실험 | `sp-systemone-v3-*`, `sp-systemone-nolen-v3-*` | 신규 LLM 없음(v2 계열 유지) | `sp_mlp_nolen_v3.pt` |
| 4. V4 | `v4_train/test`, `v4_sft_*`, `v4_systemone-*` | `tev1-v4-lora`, `merged-tev1-sp-bf16-v4`, `tev1-neely-sp-bf16-v4*{f16,q8_0}` | `sp_mlp_nolen_v4.pt`, `sp_mlp_nolen_v4_retrain.pt` |
| 5. V4 RC | `v4_full_rc`, `v4_train_rc`, `v4_heldout_rc`, `v4_rc_sft_*`, `v4_rc_systemone-*` | `tev1-v4-lora-rc`, `merged-tev1-sp-bf16-v4-rc`, `tev1-neely-sp-bf16-v4-rc*{f16,q8_0}` | `sp_mlp_nolen_v4_rc.pt` |
| 6. b:c3 승격·생성 | `v4_rc_bc3_*`, `v4_rc_bc3_augment` | 신규 LLM 없음(데이터 보강 단계) | `sp_mlp_nolen_v4_rc_primary.pt` (초기), `sp_mlp_nolen_v4_rc_primary_bc3aug.pt` |
| 7. 0.8B SOUP 최종 | `v4_rc_bc3aug_systemone-soup-{train,heldout}`, `v4_rc_bc3aug_sft-train`, `v4_rc_bc3aug_mlp-*` | `soup-v4-rc-08b-bc3aug-out/tev1-v4-rc-primary-08b-bc3aug-soup` → `soup-v4-rc-08b-bc3aug-merged` → `tev1-0.8b-neely-q8_0-decision.gguf` → **Ollama `tev1-0.8b-neely:latest`** | **`sp_mlp_nolen_v4_rc_primary.pt` 재학습/교체** (md5 `69ceaf7e18a74211a07a696575600cf3`) |

##### 2.1 단계 0 — 기반 분석

`eth_wave.json`(2,799파동), `eth_main_waves.json`(793), `gui_features*.jsonl`(87), `sp-real-distribution.json`(914)은 파동/feature 분포와 SP 라벨 분포를 측정한 원천 데이터다. 별도 모델을 만들지 않았고, 이후 SFT 데이터의 클래스 가중치와 `b:c3` 보강 필요성을 판단하는 근거로 사용했다.

##### 2.2 단계 1 — 4B 초기 SFT

```text
generate_sft_dataset.py
  → sft-dataset-train4.jsonl (20,000) / sft-dataset-heldout4.jsonl (3,000)
  → build_sp_systemone.py → sp-systemone-train/heldout.jsonl
  → finetune_tev1_sp.py → output/tev1-sp-lora
  → merge_tev1_sp.py → merged-tev1-sp → merged-tev1-sp-bf16
  → tev1-neely-sp-q8_0.gguf / tev1-neely-sp-bf16-q8_0{,-decision}.gguf
  → Ollama tev1-neely:latest (초기)
```

같은 시기에 MLP 증류 실험이 병행되어 `mlp/train_mlp.py`가 `sp_mlp.pt`와 멀티마켓 `sp_mlp_multi.pt`를 생성했다.

##### 2.3 단계 2~3 — nolen / V3

`sp-systemone-nolen-*`은 길이(length) feature를 제거한 계열이고, gapfill 데이터는 누락 feature 보간용이다. `output/tev1-sp-nolen`, `output/tev1-sp-nolen-v2`와 대응 merge 모델, `tev1-neely-sp-nolen*{f16,q8_0}-decision.gguf`가 생성됐다. V3 데이터(`sp-systemone-v3-*`, `sp-systemone-nolen-v3-*`)는 LLM 대신 `sp_mlp_nolen_v3.pt` 재학습에 주로 사용됐다.

##### 2.4 단계 4~5 — V4 / V4 RC

`v4_train.jsonl`(94,169) → `v4_sft_train.jsonl`(93,787) → `v4_systemone-train.jsonl`(93,787) 순으로 정제했고, RC 기준에서는 `v4_full_rc.jsonl`(113,088)를 `v4_train_rc`(101,779)/`v4_heldout_rc`(11,309)로 분리했다. 여기서 `tev1-v4-lora`, `tev1-v4-lora-rc`, `merged-tev1-sp-bf16-v4`, `merged-tev1-sp-bf16-v4-rc`와 각 f16/Q8_0 GGUF가 생성됐다. MLP는 `sp_mlp_nolen_v4.pt`, `sp_mlp_nolen_v4_rc.pt`가 대응한다.

##### 2.5 단계 6 — b:c3 승격 + 생성 보강

| 구성 | 행 수 |
|---|---:|
| 기존 2차 후보 승격 | 963 |
| 엔진 생성 데이터(`v4_rc_bc3_augment.jsonl`) | 4,000 |
| 추가 합계 | 4,963 |
| 최종 train | 105,369 |
| 최종 heldout | 11,253 (그중 `b:c3` 116) |

이 데이터로 만든 파생셋이 `v4_rc_bc3aug_sft-train.jsonl`, `v4_rc_bc3aug_systemone-{train,heldout}.jsonl`, `v4_rc_bc3aug_systemone-soup-{train,heldout}.jsonl`, `v4_rc_bc3aug_mlp-{train,heldout}.jsonl`이다. 먼저 `sp_mlp_nolen_v4_rc_primary.pt`와 `sp_mlp_nolen_v4_rc_primary_bc3aug.pt`가 만들어졌고, 최종적으로 동일 파일명(`sp_mlp_nolen_v4_rc_primary.pt`)에 재학습 결과를 덮어써 production으로 승격했다. 직전 파일은 `sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726`으로 보존했다.

##### 2.6 단계 7 — 0.8B SOUP 최종 (production)

```text
v4_rc_bc3aug_systemone-soup-train.jsonl (105,369)
v4_rc_bc3aug_systemone-soup-heldout.jsonl (11,253)
  → SOUP LoRA (run_20261121_144423_1232b7a2, checkpoint-6586)
  → soup-v4-rc-08b-bc3aug-merged (bf16)
  → tev1-0.8b-neely-q8_0-decision.gguf (llama.cpp b11963 컨버터)
  → decision 메타데이터 4종 주입
  → Ollama tev1-sp-bc3aug:latest → tev1-0.8b-neely:latest (rename)
```

SOUP 설정은 `batch_size=1`, `gradient_accumulation_steps=16`, `max_length=1240`, bf16 LoRA `r=16/alpha=32`, `target_modules=all-linear`, `optimizer=adamw_torch_fused`, 1 epoch이다. 학습은 `CUDA_VISIBLE_DEVICES=1`의 RTX 4060 Ti에서 수행했다.

GGUF 변환 시 최신 llama.cpp는 `ssm_beta_alpha.weight`를 만들어 Ollama 0.40.13 로딩이 실패했으므로, `b11963` 리비전 컨버터를 사용했다. decision capability 복원을 위해 다음 메타데이터를 주입했다.

- `qwen35.decision.type=pplx-decider`
- `tokenizer.chat_template.systemone`
- `tokenizer.chat_templates=['systemone']`
- `general.tags`에 `decision-model`

#### 3. 현재 production 상태

| 항목 | 이름 / 경로 | 상태 |
|---|---|---|
| Ollama LLM | `tev1-0.8b-neely:latest` | `qwen35`, `Q8_0`, Capabilities `decision` |
| LLM GGUF | `~/tev1-finetune/tev1-0.8b-neely-q8_0-decision.gguf` | 아카이브 실파일의 심볼 링크 |
| Modelfile | `~/tev1-finetune/Modelfile-tev1-0.8b-neely` | 실파일 |
| MLP | `~/tev1-finetune/mlp/sp_mlp_nolen_v4_rc_primary.pt` | 실파일, md5 `69ceaf7e18a74211a07a696575600cf3` |
| MLP 서비스 | `mlp/sp-mlp.service` (8095) | production MLP 경로 참조 |

검증 결과:

- SP heldout 전체 10,442 / 11,253 = **92.79%**, `b:c3` recall 115 / 116 = **99.1%**, precision 85.8%.
- 4B sentiment retention: `tev1:latest` 95.0%, `tev1-neely:latest` 96.0%, 신규 오류 0.
- 0.8B sentiment: base `tev1:0.8b` 93.0%, 파인튜닝 92.5%, 신규 오류 1건, confidence 0.816 → 0.969.
- `/v1/systemone` 호출 정상, capability `decision` 확인.

#### 4. 보존·삭제 정책

- 아카이브로 이동: 이전 버전 LLM(LoRA/merge/GGUF), 이전 MLP 가중치, 데이터 생성·학습·평가 스크립트, 설정, 로그, 기존 문서.
- 현재 디렉터리에는 이동한 항목의 **절대경로 심볼 링크만** 유지한다.
- 실파일로 유지: production MLP 가중치와 서빙 코드(`serve_mlp.py`, `v4_features.py`, `vendor/`), `sp-mlp.service`, `venv/`, `.git`, README/LICENSE/requirements, 본 문서와 강의 문서.
- 삭제: `/home/wnf/.soup/llama.cpp`, `/home/wnf/.soup/llama.cpp-ollama`, 저장소의 `llama.cpp` 빌드 트리, llama.cpp clone/build 로그, 브라우저 HTML 스냅샷, health probe JSON, GGUF 메타데이터 임시 txt, `__pycache__`, 중복 실패 GGUF 산출물, 세션 로그 임시 파일.

상세 학습 과정은 `tev1-08b-soup-lecture.md`와 `/home/wnf/posting/tev1-08b-soup-lecture.md`에 정리되어 있다.
