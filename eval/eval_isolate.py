"""Isolate the degradation: base-only vs base+adapter (unmerged) vs merged, all bf16."""
import json, os, torch, warnings

BASE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--togethercomputer--Tev1-4B-experimental"
    "/snapshots/0b7becf017daa0e5eb222f8ce7483c8c8259c52f"
)
ADAPTER = "/home/wnf/tev1-finetune/output/tev1-sp-lora"
MERGED = "/home/wnf/tev1-finetune/merged-tev1-sp"
HELD = "/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl"
N = int(os.environ.get("N", "100"))

from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

rows = [json.loads(l) for l in open(HELD, encoding="utf-8") if l.strip()][:N]
letters = [chr(ord("A") + i) for i in range(10)]
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def ev(model, label):
    model.eval()
    c = t = 0
    for row in rows:
        pids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        inp = torch.tensor([pids], device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][len(pids):], skip_special_tokens=True)
        pred = next((ch for ch in text if ch in letters), None)
        t += 1
        c += int(pred == row["answer"])
    print(f"\n===== {label}: {c}/{t} = {c/t:.3f} =====\n", flush=True)


# 1) base only (bf16)
m = AutoModelForImageTextToText.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
ev(m, "BASE bf16 (no adapter)")

# 2) base + adapter, UNMERGED, bf16
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = PeftModel.from_pretrained(m, ADAPTER)
    for x in w:
        s = str(x.message)
        if "key" in s.lower() or "lora" in s.lower() or "target" in s.lower():
            print("PEFT WARN:", s[:200])
ev(m, "BASE bf16 + adapter (UNMERGED)")

# 3) merged in-memory
m = m.merge_and_unload()
ev(m, "BASE bf16 + adapter (MERGED in-memory)")
