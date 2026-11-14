"""Diagnose train/inference mismatch and raw model behavior."""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
ADAPTER = "/home/wnf/tev1-finetune/output"
DATA = "/home/wnf/tev1-finetune/sft-dataset.jsonl"

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def prompt_of(s):
    return f"State:\n{json.dumps(s['state'], ensure_ascii=False)}\n\nQuestion: {s['question']}\nAnswer:"

samples = [json.loads(l) for l in open(DATA)]
samples = samples[:12]

print("=== RAW next-token top-5 after 'Answer:' ===")
for i, s in enumerate(samples):
    p = prompt_of(s)
    ids = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    top = torch.topk(logits, 5).indices.tolist()
    shows = [repr(tokenizer.decode([t])) for t in top]
    print(f"[{i+1}] exp={s['answer']:<30} top5={shows}")

print("\n=== RAW generation (max_new_tokens=12, free) ===")
for i, s in enumerate(samples):
    p = prompt_of(s)
    ids = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    txt = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{i+1}] exp={s['answer']:<30} raw={txt!r}")

print("\n=== RAW generation with trailing newline in prompt ===")
for i, s in enumerate(samples[:6]):
    p = prompt_of(s) + "\n"
    ids = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    txt = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{i+1}] exp={s['answer']:<30} raw={txt!r}")
