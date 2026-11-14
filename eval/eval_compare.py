"""Head-to-head: merged bf16 vs base+LoRA(4bit), exact training prompt, same rows."""
import json, os, sys, torch

BASE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--togethercomputer--Tev1-4B-experimental"
    "/snapshots/0b7becf017daa0e5eb222f8ce7483c8c8259c52f"
)
ADAPTER = "/home/wnf/tev1-finetune/output/tev1-sp-lora"
MERGED = "/home/wnf/tev1-finetune/merged-tev1-sp"
HELD = "/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl"
N = int(os.environ.get("N", "100"))

from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

rows = [json.loads(l) for l in open(HELD, encoding="utf-8") if l.strip()][:N]
letters = [chr(ord("A") + i) for i in range(10)]


def evaluate(model, tok, label):
    model.eval()
    correct = total = 0
    per = {}
    samples = []
    for row in rows:
        pids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        inp = torch.tensor([pids], device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=inp, max_new_tokens=4, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][len(pids):], skip_special_tokens=True)
        pred = next((ch for ch in text if ch in letters), None)
        total += 1
        ok = int(pred == row["answer"])
        correct += ok
        per.setdefault(row["answer"], [0, 0])
        per[row["answer"]][1] += 1
        per[row["answer"]][0] += ok
        if len(samples) < 5:
            samples.append((row["answer"], pred, repr(text[:40])))
    print(f"\n===== {label}: {correct}/{total} = {correct/total:.3f} =====")
    for k in sorted(per):
        print(f"  {k}: {per[k][0]}/{per[k][1]}")
    print("  samples (expected, pred, text):")
    for s in samples:
        print("   ", s)
    return correct / total


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1) merged bf16
    m = AutoModelForImageTextToText.from_pretrained(
        MERGED, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    evaluate(m, tok, "MERGED bf16 (exact training prompt)")
    del m
    torch.cuda.empty_cache()

    # 2) base 4-bit + adapter
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    b = AutoModelForImageTextToText.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    b = PeftModel.from_pretrained(b, ADAPTER)
    evaluate(b, tok, "BASE 4bit + LoRA (exact training prompt)")


if __name__ == "__main__":
    main()
