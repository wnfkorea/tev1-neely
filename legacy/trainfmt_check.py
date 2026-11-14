"""Replicate training-format input exactly and inspect completion-region predictions."""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
ADAPTER = "/home/wnf/tev1-finetune/output"
DATA = "/home/wnf/tev1-finetune/sft-dataset.jsonl"
MAXLEN = 512

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def build_prompt(s):
    return f"State:\n{json.dumps(s['state'], ensure_ascii=False)}\n\nQuestion: {s['question']}\nAnswer:"

samples = [json.loads(l) for l in open(DATA)][:2]

for i, s in enumerate(samples):
    prompt = build_prompt(s)
    full = prompt + "\n" + s["answer"]
    # exactly like training
    enc = tok(full, truncation=True, max_length=MAXLEN, padding="max_length")
    ids = torch.tensor([enc["input_ids"]], device=model.device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0]
    # find real sequence start (first non-pad)
    seq = enc["input_ids"]
    start = next(j for j, t in enumerate(seq) if t != tok.pad_token_id)
    # true completion tokens in this tokenization: after last 'Answer:'
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    ans_ids = tok(s["answer"], add_special_tokens=False)["input_ids"]
    print(f"\n=== sample {i+1} answer={s['answer']} seq_len={len(seq)-start} ===")
    # Show last 8 real-token predictions
    for pos in range(len(seq) - 9, len(seq) - 1):
        if pos < start:
            continue
        true_next = seq[pos + 1]
        top = torch.topk(logits[pos], 5).indices.tolist()
        shows = [tok.decode([t]) for t in top]
        tstr = tok.decode([true_next])
        mark = "OK " if top[0] == true_next else "   "
        print(f" {mark}pos{pos}: ctx={tok.decode([seq[pos]])!r:10} true={tstr!r:10} top5={shows}")
