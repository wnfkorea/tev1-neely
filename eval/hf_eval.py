import json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/wnf/tev1-finetune/merged-tev1-sp"
rows = [json.loads(l) for l in open("/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl") if l.strip()]
print("rows:", len(rows))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:1")
model.eval()

KEYS = [":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3", "Invalid_Neely_Rule_Violation"]
letter_ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in "ABCDEFGHIJ"]

TAIL_OLD = "<think\n\n</think\n\n"
TAIL_NEW = "<think>\n\n</think>\n\n"

@torch.no_grad()
def predict(prompt):
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t = torch.tensor([ids], device="cuda:1")
    logits = model(t).logits[0, -1]
    sub = logits[letter_ids]
    return KEYS[int(sub.argmax())]

results = {"train_fmt": 0, "endpoint_fmt": 0}
t0 = time.time()
for i, r in enumerate(rows):
    gold = r["answer_key"]
    p_old = r["prompt"]
    assert p_old.endswith(TAIL_OLD), repr(p_old[-30:])
    p_new = p_old[:-len(TAIL_OLD)] + TAIL_NEW
    if predict(p_old) == gold: results["train_fmt"] += 1
    if predict(p_new) == gold: results["endpoint_fmt"] += 1
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(rows)}  train={results['train_fmt']/(i+1):.3f}  endpoint={results['endpoint_fmt']/(i+1):.3f}  {time.time()-t0:.0f}s")

n = len(rows)
print(f"\ntrain_fmt    (<think ): {results['train_fmt']}/{n} = {results['train_fmt']/n:.4f}")
print(f"endpoint_fmt (<think>): {results['endpoint_fmt']}/{n} = {results['endpoint_fmt']/n:.4f}")
