"""Merge the SP LoRA adapter into the Qwen3-4B base and save a standalone HF model."""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get(
    "BASE_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    ),
)
ADAPTER = os.environ.get("ADAPTER_DIR", "/home/wnf/tev1-finetune/output/checkpoint-3000")
OUT = os.environ.get("MERGED_DIR", "/home/wnf/tev1-finetune/merged-sp")

print("base:", BASE)
print("adapter:", ADAPTER)
print("out:", OUT)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(OUT)
print("saved", OUT)
