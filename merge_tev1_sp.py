"""Merge the Tev1 SP LoRA adapter into the Tev1-4B base and save a standalone HF model."""
import os
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get(
    "BASE_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--togethercomputer--Tev1-4B-experimental"
        "/snapshots/0b7becf017daa0e5eb222f8ce7483c8c8259c52f"
    ),
)
ADAPTER = os.environ.get("ADAPTER_DIR", "/home/wnf/tev1-finetune/output/tev1-sp-lora")
OUT = os.environ.get("MERGED_DIR", "/home/wnf/tev1-finetune/merged-tev1-sp")

print("base:", BASE)
print("adapter:", ADAPTER)
print("out:", OUT)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(OUT)
print("saved", OUT)
