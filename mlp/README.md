# Production SP MLP server

This directory contains only the production MLP artifacts. Older checkpoints and
training scripts are archived on Tamuz2T and exposed as ignored symlinks.

## Files

| File | Purpose |
|---|---|
| `sp_mlp_nolen_v4_rc_primary.pt` | production checkpoint (v3-hybrid), md5 `a8f1e6cc9e805ee2313ac14d76ed3696` |
| `serve_mlp.py` | `/v1/systemone` drop-in HTTP server |
| `v4_features.py` | V4 wave/feature extraction matching the GUI pipeline |
| `vendor/` | bundled feature engine |
| `sp-mlp.service` | systemd unit (port 8095) |

## Run

```bash
cd /home/wnf/tev1-finetune
./venv/bin/python mlp/serve_mlp.py \
  --model mlp/sp_mlp_nolen_v4_rc_primary.pt \
  --host 0.0.0.0 --port 8095
```

or via systemd:

```bash
sudo systemctl restart sp-mlp.service
curl http://127.0.0.1:8095/health
```

## Rollback

```bash
cp /media/wnf/Tamuz2T/106.Dev/tev1-finetune/models/mlp/sp_mlp_nolen_v4_rc_primary_bc3aug.pt \
   mlp/sp_mlp_nolen_v4_rc_primary.pt
sudo systemctl restart sp-mlp.service
```

The previous (pre-v3) checkpoint is preserved in the archive. The current
checkpoint was retrained on the v3 rare-class data plus all 4,000 generated
`b:c3` rows; see `DATASET_HISTORY.md` for metrics.
