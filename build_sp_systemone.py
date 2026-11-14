#!/usr/bin/env python3
"""Convert the SP SFT dataset into Tev1 systemone decision records.

Each output record:

  {"state": {...}, "question": "...", "options": [...],
   "answer": "E", "answer_key": ":F3", "prompt": "<full systemone prompt>"}

The prompt is rendered with the *exact* systemone chat template that is injected
into the deployed GGUF (`systemone-template.txt`, `tokenizer.chat_template.systemone`),
via `tokenizer.apply_chat_template`. This guarantees the training prompt is
byte-identical to what llama.cpp's `/v1/systemone` feeds the model at inference —
including the `<think>` / `</think>` added tokens (the previous hand-rolled
renderer emitted `<think` / `</think` without the `>` and therefore mismatched).
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Keep in sync with src/services/analysis/elliott-wave.service.ts
SP_CHOICES = [
    ":5", ":s5", ":c3", "b:c3", ":F3", "x:c3", ":L5", ":L3", ":sL3",
    "Invalid_Neely_Rule_Violation",
]

SP_CRITERIA = {
    ":5": "Impulse wave (not final): 5-wave pattern that does not end a larger pattern; covers more price in less time than the prior wave (m1_retrace_ratio < 1, m1_time_ratio > 1)",
    ":s5": "Special 5-wave: part of a complex correction or 3rd wave of an impulse with failed/truncated 5th wave; can occupy :L5 position without reversal confirmation; two waves before it must be 5-F3 or F3-c3",
    ":c3": "Central 3-wave: middle of a pattern, not initial or final; not immediately followed by a large strong directional move; the prior wave is longer (m1_retrace_ratio > 1) and consumes same or more time (m1_time_ratio >= 1)",
    "b:c3": "B-wave of a flat correction: the b-wave of a flat (not a zigzag); m1_retrace_ratio near 1.0 with rule_of_equality_satisfied likely true",
    ":F3": "First 3-wave: head of a pattern, right after x:c3 or between two :5s; first segment of complex corrections or corrections between two impulse waves; often m1/m2 are null (pattern head)",
    "x:c3": "X-wave position: central 3-wave connecting two standard corrections; same degree as adjacent corrections, must consume less time and be less complex than the corrections before and after it (m1_time_ratio < 1)",
    ":L5": "Last 5-wave: final wave of a larger pattern; the label right before it must be :F3 or :c3; ends a trending impulse or terminal pattern; often m_1/m_2 are null (pattern end)",
    ":L3": "Last 3-wave: final 3-segment wave ending a triangle or terminal made of five :3s; if smallest of the five, must be completely retraced within the time it took to form; often m_1/m_2 are null (pattern end)",
    ":sL3": "Second-to-last 3-wave: always followed by :L3; part of a triangle or terminal pattern made of five consecutive :3s; never starts or ends a pattern",
    "Invalid_Neely_Rule_Violation": "No valid Neely structure: the monowave sequence violates Neely rules (e.g. m1_retrace_ratio and m2_retrace_ratio both outside valid ranges, rule_of_equality_satisfied false where required, channel_break inconsistent, or overlapping/irregular retracements that fit no SP)",
}

LABELS = [chr(ord("A") + i) for i in range(len(SP_CHOICES))]
LABEL_BY_KEY = dict(zip(SP_CHOICES, LABELS))

# Every sample offers the same option set (mirrors tev1-neely training).
OPTIONS = [
    {"label": LABEL_BY_KEY[key], "key": key, "description": SP_CRITERIA[key]}
    for key in SP_CHOICES
]


def make_renderer(tokenizer, template):
    """Return render(state, question, options) -> prompt string.

    Uses the deployed systemone template so the rendered prompt is identical to
    the one llama.cpp produces for /v1/systemone.
    """
    def render(state, question, options):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            state=state,
            instructions=question,
            options=options,
            type="choice",
            id="sp",
        )
    return render


def convert(src, dst, render):
    rows = []
    counts = collections.Counter()
    with open(src, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            key = sample["answer"]
            if key not in LABEL_BY_KEY:
                raise ValueError(f"{src}:{lineno}: unknown SP answer {key!r}")
            rows.append({
                "state": sample["state"],
                "question": sample["question"],
                "options": OPTIONS,
                "answer": LABEL_BY_KEY[key],
                "answer_key": key,
                "prompt": render(sample["state"], sample["question"], OPTIONS),
            })
            counts[key] += 1

    with open(dst, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{src} -> {dst}: {len(rows)} rows")
    for key in SP_CHOICES:
        print(f"  {LABEL_BY_KEY[key]}  {key:34s} {counts[key]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(HERE, "sft-dataset.jsonl"))
    ap.add_argument("--heldout", default=os.path.join(HERE, "sft-dataset-heldout2.jsonl"),
                    help="clean held-out source (different generator seed than --train)")
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "sp-systemone"))
    ap.add_argument("--template", default=os.path.join(HERE, "systemone-template.txt"),
                    help="systemone chat template injected into the GGUF")
    ap.add_argument("--tokenizer", default="togethercomputer/Tev1-4B-experimental")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with open(args.template, encoding="utf-8") as fh:
        template = fh.read()
    render = make_renderer(tokenizer, template)

    convert(args.train, args.out_prefix + "-train.jsonl", render)
    convert(args.heldout, args.out_prefix + "-heldout.jsonl", render)


if __name__ == "__main__":
    main()
