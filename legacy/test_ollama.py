"""Verify tev1-neely through Ollama /api/generate against held-out samples."""
import json
import os
import urllib.request

HOST = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_TEST_MODEL", "tev1-neely:latest")
DATA = "/home/wnf/tev1-finetune/sft-dataset-heldout.jsonl"
N = int(os.environ.get("OLLAMA_TEST_N", "20"))


def build_prompt(s):
    return (
        "State:\n" + json.dumps(s["state"], ensure_ascii=False)
        + "\n\nQuestion: " + s["question"] + "\nAnswer:\n"
    )


def gen(prompt):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 16, "stop": ["\n", "<|im_end|>"]},
    }).encode()
    req = urllib.request.Request(HOST + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]


samples = [json.loads(l) for l in open(DATA)][:N]
correct = 0
for i, s in enumerate(samples, 1):
    out = gen(build_prompt(s)).strip()
    ok = out == s["answer"]
    correct += ok
    mark = "OK " if ok else "XX "
    print(f"{mark}[{i:2}] exp={s['answer']:<30} got={out!r}")

print(f"\nAccuracy: {correct}/{N} ({100*correct//N}%)")
