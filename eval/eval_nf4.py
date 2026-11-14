"""4-bit regime: base-only vs base+adapter, full 300."""
import json, os, torch

BASE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--togethercomputer--Tev1-4B-experimental"
    "/snapshots/0b7becf017daa0e5eb222f8ce7483c8c8259c52f"
)
ADAPTER = "/home/wnf/tev1-finetune/output/tev1-sp-lora"
HELD = "/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl"

from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

rows = [json.loads(l) for l in open(HELD, encoding="utf-8") if l.strip()]
letters = [chr(ord("A") + i) for i in range(10)]
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def ev(model, label, n):
    model.eval()
    c = t = 0
    per = {}
    for row in rows[:n]:
        pids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        inp = torch.tensor([pids], device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][len(pids):], skip_special_tokens=True)
        pred = next((ch for ch in text if ch in letters), None)
        t += 1
        ok = int(pred == row["answer"])
        c += ok
        per.setdefault(row["answer"], [0, 0]); per[row["answer"]][1] += 1; per[row["answer"]][0] += ok
    print(f"\n===== {label}: {c}/{t} = {c/t:.3f} =====", flush=True)
    print("  " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(per.items())), flush=True)


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
m = AutoModelForImageTextToText.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True,
    torch_dtype=torch.bfloat16)
ev(m, "BASE nf4 (no adapter), first 100", 100)
m = PeftModel.from_pretrained(m, ADAPTER)
ev(m, "BASE nf4 + adapter, ALL 300", 300)
