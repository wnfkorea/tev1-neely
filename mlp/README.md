# Production SP MLP server

This directory contains only the production MLP artifacts. Older checkpoints and
training scripts are archived on Tamuz2T and exposed as ignored symlinks.

## Files

| File | Purpose |
|---|---|
| `sp_mlp_nolen_v4_rc_primary.pt` | production checkpoint, md5 `69ceaf7e18a74211a07a696575600cf3` |
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
cp mlp/sp_mlp_nolen_v4_rc_primary.pt.bak.20261121_171726 \
   mlp/sp_mlp_nolen_v4_rc_primary.pt
sudo systemctl restart sp-mlp.service
```

The backup is an ignored symlink into the archive.
