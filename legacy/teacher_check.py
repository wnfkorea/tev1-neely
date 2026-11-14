"""Teacher-forcing diagnostic: what does the model predict after 'Answer:'?"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
ADAPTER = "/home/wnf/tev1-finetune/output"
DATA = "/home/wnf/tev1-finetune/sft-dataset.jsonl"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def build_prompt(s):
    return f"State:\n{json.dumps(s['state'], ensure_ascii=False)}\n\nQuestion: {s['question']}\nAnswer:"

samples = [json.loads(l) for l in open(DATA)][:3]

for i, s in enumerate(samples):
    prompt = build_prompt(s)
    full = prompt + "\n" + s["answer"]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    plen = len(p_ids)
    ids = torch.tensor([f_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    # position that predicts the first token after prompt = index plen-1
    print(f"\n=== sample {i+1}  answer={s['answer']}  prompt_len={plen} ===")
    print("last prompt tokens:", [tok.decode([t]) for t in p_ids[-4:]])
    print("completion tokens:", [tok.decode([t]) for t in f_ids[plen:]])
    for offset in range(0, min(4, len(f_ids) - plen)):
        pos = plen - 1 + offset  # logits at pos predict f_ids[pos+1]
        true_tok = f_ids[pos + 1]
        top = torch.topk(logits[pos], 5).indices.tolist()
        shows = [tok.decode([t]) for t in top]
        print(f"  step {offset}: true={tok.decode([true_tok])!r:12} top5={shows}")
