# tev1-neely — production SP decision stack

This repository contains **only the production artifacts** for Neely SP decision serving:

- two Ollama decision models: `tev1-0.8b-neely:latest` and `tev1-neely:latest`
- the production CPU MLP classifier and its `/v1/systemone` drop-in server
- deployment configuration and production-focused documentation

Previous versions, datasets, training scripts, logs and old checkpoints were moved to
`/media/wnf/Tamuz2T/106.Dev/tev1-finetune`. They remain available in the working
tree as **ignored absolute-path symlinks** and are not tracked by Git.

## Production LLM models

| Tag | Base | Blob | Role |
|---|---|---|---|
| `tev1-0.8b-neely:latest` | `Tev1-0.8B-experimental` | ~811 MB, Q8_0 | **Primary** SP decision model, fine-tuned with SOUP on V4 RC + b:c3 augmented data |
| `tev1-neely:latest` | `Tev1-4B-experimental` | ~4.2 GB | Legacy 4B decision model, kept for compatibility |

Both expose the Ollama-specific `decision` capability and answer `POST /v1/systemone`.

Source GGUFs are stored outside Git:

```text
tev1-0.8b-neely-q8_0-decision.gguf
tev1-neely-sp-bf16-v4-rc-q8_0-decision.gguf
```

They are symlinks into the Tamuz2T archive. Ollama owns its own copy of the blob, so
deleting the symlink does not remove the deployed model.

Example request:

```bash
curl -s http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tev1-0.8b-neely:latest",
    "state": {"$BYND": {"wave": "..."}},
    "questions": {"direction": {
      "type": "choice",
      "instructions": "bullish or bearish?",
      "criteria": {"상승": "bullish", "하락": "bearish"}
    }}
  }'
```

## Production MLP

The CPU MLP is a distilled SP classifier and a drop-in replacement for the Ollama
`/v1/systemone` endpoint.

| Path | Purpose |
|---|---|
| `mlp/sp_mlp_nolen_v4_rc_primary.pt` | production checkpoint (md5 `69ceaf7e18a74211a07a696575600cf3`) |
| `mlp/serve_mlp.py` | HTTP server on port 8095 |
| `mlp/v4_features.py` | GUI-equivalent V4 feature extraction |
| `mlp/vendor/` | bundled feature engine |
| `mlp/sp-mlp.service` | systemd unit |

```bash
curl http://127.0.0.1:8095/health
# {"status":"ok"}
```

Switching back to the previous checkpoint (archived, symlinked locally):

```bash
cp mlp/sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726 \
   mlp/sp_mlp_nolen_v4_rc_primary.pt
sudo systemctl restart sp-mlp.service
```

## Results

- SP heldout: **10,442 / 11,253 = 92.79%**
- `b:c3` recall: **115 / 116 = 99.1%** (precision 85.8%)
- 4B sentiment retention: `tev1:latest` 95.0%, `tev1-neely:latest` 96.0%
- 0.8B sentiment: base 93.0%, fine-tuned 92.5%, confidence 0.816 → 0.969

## Documentation

- `DATASET_HISTORY.md` — dataset consolidation and dataset → LLM/MLP lineage
- `tev1-08b-soup-lecture.md` — 0.8B SOUP fine-tuning and deployment walkthrough
- `EVAL_MLP_VS_LLM_300.md` — heldout 300 inference comparison: production MLP vs `tev1-0.8b-neely:latest`
- `eval_mlp_vs_llm_300.jsonl` — raw predictions/confidence/latency for the 300 samples
- `mlp/README.md` — production MLP server details

## Reproducing the environment

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The GGUF files and archived originals are intentionally not committed. See
`DATASET_HISTORY.md` for their locations and the archive layout.
