"""
Phase 3: QLoRA fine-tuning of the tev1 decision model (Qwen3-4B base).

Trains a QLoRA adapter on the synthetic SFT dataset (Phase 2) so the model
learns to map the standardized monowave feature state to the Neely SP symbol.

Critical correctness notes (fixed after first run):
  * Prompt and answer are tokenized SEPARATELY and their token IDs are
    concatenated. Re-tokenizing the concatenated string lets BPE merge the
    boundary (e.g. "Answer:" + "\\n" -> "':\\n'"), which breaks inference.
  * The answer is delimited by a prompt that already ends with "Answer:\\n",
    so the model always sees the same context token before the label.
  * Completion-only loss is preserved via a custom collator (the HF
    DataCollatorForLanguageModeling overwrites labels with input_ids).
  * Dynamic padding (max length in batch) keeps attention masks correct and
    is faster than fixed padding.

Usage:
  CUDA_VISIBLE_DEVICES=0 venv/bin/python finetune.py   # 4090
  CUDA_VISIBLE_DEVICES=1 venv/bin/python finetune.py   # 4060 Ti
"""

import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    ),
)
DATASET_PATH = os.environ.get("SFT_DATASET", "/home/wnf/tev1-finetune/sft-dataset.jsonl")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "/home/wnf/tev1-finetune/output")

# Hyperparameters (optimized for 4090 24GB with QLoRA + dynamic padding)
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10
SAVE_STEPS = 500
SEED = 42


def build_prompt(sample: dict) -> str:
    """Prompt ends with 'Answer:\\n' so the label context is identical at inference."""
    state = json.dumps(sample["state"], ensure_ascii=False)
    return (
        "State:\n"
        f"{state}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:\n"
    )


def build_example(tokenizer, sample) -> dict:
    """Tokenize prompt and answer separately, then concatenate token IDs."""
    prompt = build_prompt(sample)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = answer_ids + [eos]

    input_ids = (prompt_ids + answer_ids)[:MAX_SEQ_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_SEQ_LENGTH]
    return {"input_ids": input_ids, "labels": labels}


class CompletionCollator:
    """Dynamic padding that keeps the completion-only label mask and correct attention mask."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    print(f"Base model: {BASE_MODEL}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load + tokenize
    rows = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Dataset size: {len(rows)}")

    examples = [build_example(tokenizer, r) for r in rows]
    lens = [len(e["input_ids"]) for e in examples]
    print(f"Token length: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")
    dataset = Dataset.from_list(examples)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_steps = (len(rows) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        warmup_steps=warmup_steps,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        seed=SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.3,
        dataloader_num_workers=2,
    )

    collator = CompletionCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
