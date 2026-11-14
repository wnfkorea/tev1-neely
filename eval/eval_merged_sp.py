#!/usr/bin/env python3
"""Evaluate a *merged* Tev1 SP model (HF dir) on the clean held-out set.

This is the honest metric: it loads the exact artifact that gets converted to
GGUF, not the in-memory training model.

Usage:
  CUDA_VISIBLE_DEVICES=1 ./venv/bin/python eval/eval_merged_sp.py \
      --model merged-tev1-sp-bf16 --heldout sp-systemone-heldout.jsonl --limit 300
"""
import argparse
import collections
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

LETTERS = [chr(ord("A") + i) for i in range(10)]
KEY_BY_LETTER = {chr(ord("A") + i): k for i, k in enumerate([
    ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3",
    "Invalid_Neely_Rule_Violation",
])}


def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--heldout", default="sp-systemone-heldout.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-seq-len", type=int, default=1280)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    rows = load_rows(args.heldout, limit=args.limit or None)
    conf = collections.defaultdict(collections.Counter)
    correct = total = 0
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > args.max_seq_len:
            continue
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([prompt_ids], device="cuda"),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        pred = next((ch for ch in text if ch in LETTERS), None)
        total += 1
        correct += int(pred == row["answer"])
        conf[row["answer"]][pred] += 1

    if not total:
        print("no samples")
        return
    print(f"{args.model}: held-out accuracy {correct}/{total} = {correct/total:.3f}")
    for letter in LETTERS:
        n = sum(conf[letter].values())
        if n:
            print(f"  {letter} {KEY_BY_LETTER[letter]:34s} n={n:4d} recall={conf[letter][letter]/n:.3f}")


if __name__ == "__main__":
    main()
