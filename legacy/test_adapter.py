"""
Test the fine-tuned tev1 adapter with sample prompts from the SFT dataset.
Loads the base model + LoRA adapter and runs inference on a few samples.
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    ),
)
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/home/wnf/tev1-finetune/output")
DATASET_PATH = os.environ.get("SFT_DATASET", "/home/wnf/tev1-finetune/sft-dataset.jsonl")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))


def build_prompt(sample: dict) -> str:
    state = json.dumps(sample["state"], ensure_ascii=False)
    return (
        "State:\n"
        f"{state}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )


def main():
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    # Load samples
    samples = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"\nTesting {len(samples)} samples:\n")
    correct = 0
    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)
        expected = sample["answer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        is_correct = generated.lower().startswith(expected.lower())
        correct += int(is_correct)
        status = "✓" if is_correct else "✗"
        print(f"{status} [{i+1}] expected: {expected:<30} generated: {generated}")

    print(f"\nAccuracy: {correct}/{len(samples)} ({100*correct/len(samples):.0f}%)")


if __name__ == "__main__":
    main()
