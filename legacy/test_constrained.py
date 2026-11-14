"""
Test the fine-tuned tev1 adapter with token-level constrained decoding.
Builds a token trie from SP_CHOICES and restricts generation to valid paths.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/home/wnf/tev1-finetune/output")
DATASET_PATH = os.environ.get("SFT_DATASET", "/home/wnf/tev1-finetune/sft-dataset.jsonl")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))

SP_CHOICES = [
    ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3",
    ":L5", ":L3", ":sL3", "Invalid_Neely_Rule_Violation",
]


def build_prompt(sample: dict) -> str:
    state = json.dumps(sample["state"], ensure_ascii=False)
    return f"State:\n{state}\n\nQuestion: {sample['question']}\nAnswer:\n"


class TokenTrie:
    """Token-level trie for constrained decoding."""

    def __init__(self, tokenizer, labels):
        self.tokenizer = tokenizer
        self.root = {}
        self.label_token_seqs = {}
        for label in labels:
            tokens = tokenizer.encode(label, add_special_tokens=False)
            self.label_token_seqs[label] = tokens
            node = self.root
            for t in tokens:
                node = node.setdefault(t, {})
            node["<END>"] = label  # mark end with the label

    def allowed_next_tokens(self, token_prefix: list) -> set:
        """Return allowed next token IDs given the current token prefix."""
        if not token_prefix:
            return set(self.root.keys())
        node = self.root
        for t in token_prefix:
            if t not in node:
                return set()
            node = node[t]
        return set(node.keys())

    def is_complete(self, token_prefix: list) -> str:
        """Return the label if token_prefix is a complete label, else None."""
        node = self.root
        for t in token_prefix:
            if t not in node:
                return None
            node = node[t]
        return node.get("<END>")


def constrained_generate(model, tokenizer, trie, prompt, max_tokens=30):
    """Generate with token-trie constrained decoding (greedy)."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = input_ids["input_ids"].shape[1]

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_tokens):
            all_ids = torch.cat([
                input_ids["input_ids"][0],
                torch.tensor(generated_tokens, device=model.device, dtype=torch.long)
            ], dim=0)
            inputs = {"input_ids": all_ids.unsqueeze(0)}
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

            allowed = trie.allowed_next_tokens(generated_tokens)
            if not allowed:
                break

            # Mask logits: only allow tokens in the trie
            mask = torch.full_like(logits, float("-inf"))
            for tid in allowed:
                mask[tid] = 0
            logits = logits + mask

            next_id = int(torch.argmax(logits).item())
            generated_tokens.append(next_id)

            # Stop if complete label
            label = trie.is_complete(generated_tokens)
            if label:
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return generated_text


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

    trie = TokenTrie(tokenizer, SP_CHOICES)

    samples = []
    with open(DATASET_PATH) as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"\nTesting {len(samples)} samples with constrained decoding:\n")
    correct = 0
    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)
        expected = sample["answer"]
        generated = constrained_generate(model, tokenizer, trie, prompt)
        is_correct = generated.lower() == expected.lower()
        correct += int(is_correct)
        status = "✓" if is_correct else "✗"
        print(f"{status} [{i+1}] expected: {expected:<30} generated: {generated}")

    print(f"\nAccuracy: {correct}/{len(samples)} ({100*correct/len(samples):.0f}%)")


if __name__ == "__main__":
    main()
