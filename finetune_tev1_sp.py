#!/usr/bin/env python3
"""QLoRA fine-tune of Tev1 (Qwen3.5-4B) for the SP systemone decision task.

Input:  sp-systemone-train.jsonl  (built by build_sp_systemone.py)
Output: LoRA adapter + tokenizer in --output-dir

The prompt is already fully rendered in the JSONL, including the systemone
assistant prefix (`<|im_start|>assistant\n<think\n\n</think\n\n`), so this
script only appends the answer letter and `<|im_end|>` as the target.

Example (single 24GB GPU):
  ./venv/bin/python finetune_tev1_sp.py \
    --train sp-systemone-train.jsonl \
    --output-dir output/tev1-sp-lora \
    --batch-size 1 --grad-accum 16 --epochs 1 --max-seq-len 2048
"""
import argparse
import collections
import json
import os
import sys
import sysconfig


def _ensure_python_headers():
    """fla/triton JIT needs Python.h; fall back to a uv-managed CPython include dir."""
    inc = sysconfig.get_paths().get("include", "")
    if inc and os.path.exists(os.path.join(inc, "Python.h")):
        return
    import glob

    pattern = os.path.expanduser("~/.local/share/uv/python/cpython-*/include/python*")
    for cand in sorted(glob.glob(pattern), reverse=True):
        if os.path.exists(os.path.join(cand, "Python.h")):
            os.environ["C_INCLUDE_PATH"] = (
                cand + os.pathsep + os.environ.get("C_INCLUDE_PATH", "")
            )
            return


_ensure_python_headers()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "togethercomputer/Tev1-4B-experimental"
IM_END = "<|im_end|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--train", default="/home/wnf/tev1-finetune/sp-systemone-train.jsonl")
    ap.add_argument("--heldout", default="/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl")
    ap.add_argument("--output-dir", default="/home/wnf/tev1-finetune/output/tev1-sp-lora")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-samples", type=int, default=0,
                    help="generate on this many held-out samples after training (0=skip)")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--optim", default="paged_adamw_8bit",
                    help="Trainer optim, e.g. adamw_torch_fused (faster on <24GB)")
    ap.add_argument("--precision", choices=["bf16", "qlora"], default="bf16",
                    help="bf16 = full-precision LoRA on the bf16 base (recommended: the "
                         "trained adapter is consistent with the merged/deployed artifact). "
                         "qlora = 4-bit nf4 QLoRA (legacy; leaves an nf4->bf16 fidelity gap).")
    ap.add_argument("--dry-run", action="store_true",
                    help="load model, run one forward pass, then exit")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


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
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.precision == "qlora":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            quantization_config=bnb,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_grad_checkpointing
        )
    else:
        # Full-precision LoRA on the bf16 base. This keeps the trained adapter
        # consistent with the merged / deployed artifact (no nf4->bf16 gap).
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if not args.no_grad_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = load_rows(args.train)
    skipped = 0

    def encode(row):
        nonlocal skipped
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(
            row["answer"] + IM_END, add_special_tokens=False
        )["input_ids"]
        if len(prompt_ids) + len(answer_ids) > args.max_seq_len:
            skipped += 1
            return None
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {"input_ids": input_ids, "labels": labels}

    encoded = []
    for row in rows:
        item = encode(row)
        if item is not None:
            encoded.append(item)
    if skipped:
        print(f"WARNING: skipped {skipped} samples longer than {args.max_seq_len} tokens")
    if not encoded:
        sys.exit("no training samples after length filtering")

    lengths = [len(e["input_ids"]) for e in encoded]
    print(
        f"train samples: {len(encoded)}  "
        f"tokens min/mean/max: {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}"
    )

    if args.dry_run:
        batch = [encoded[0]]
        pad_id = tokenizer.pad_token_id
        maxlen = max(len(e["input_ids"]) for e in batch)
        input_ids = torch.tensor(
            [e["input_ids"] + [pad_id] * (maxlen - len(e["input_ids"])) for e in batch],
            device="cuda",
        )
        labels = torch.tensor(
            [e["labels"] + [-100] * (maxlen - len(e["labels"])) for e in batch],
            device="cuda",
        )
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
        print(f"dry-run loss: {out.loss.item():.4f}")
        return

    dataset = Dataset.from_list(encoded)

    def collator(batch):
        pad_id = tokenizer.pad_token_id
        maxlen = max(len(e["input_ids"]) for e in batch)
        input_ids, labels, attention = [], [], []
        for e in batch:
            pad = maxlen - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [pad_id] * pad)
            labels.append(e["labels"] + [-100] * pad)
            attention.append([1] * len(e["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }

    # transformers >=5 removed warmup_ratio; compute the equivalent step count.
    steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=not args.no_grad_checkpointing,
        optim=args.optim,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")

    if args.eval_samples > 0:
        evaluate(model, tokenizer, args)


def evaluate(model, tokenizer, args):
    rows = load_rows(args.heldout, limit=args.eval_samples or None)
    model.eval()
    letters = [chr(ord("A") + i) for i in range(10)]
    key_by_letter = {chr(ord("A") + i): k for i, k in enumerate([
        ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3",
        "Invalid_Neely_Rule_Violation",
    ])}
    conf = collections.defaultdict(collections.Counter)  # true -> predicted
    correct = 0
    total = 0
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > args.max_seq_len:
            continue
        input_ids = torch.tensor([prompt_ids], device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        pred = next((ch for ch in text if ch in letters), None)
        total += 1
        correct += int(pred == row["answer"])
        conf[row["answer"]][pred] += 1
    if not total:
        print("evaluate: no samples")
        return
    print(f"held-out accuracy: {correct}/{total} = {correct/total:.3f}")
    print("per-class recall:")
    for letter in letters:
        n = sum(conf[letter].values())
        if n:
            print(f"  {letter} {key_by_letter[letter]:34s} n={n:4d} recall={conf[letter][letter]/n:.3f}")


if __name__ == "__main__":
    main()
