import json, time, urllib.request
from collections import Counter, defaultdict

PATH = "/home/wnf/tev1-finetune/sp-systemone-heldout.jsonl"
rows = [json.loads(l) for l in open(PATH) if l.strip()]
print(f"rows: {len(rows)}")

def ask(row):
    criteria = {o["key"]: o["description"] for o in row["options"]}
    choices = [o["key"] for o in row["options"]]
    payload = {
        "model": "tev1-neely",
        "state": row["state"],
        "questions": {"sp": {"type": "choice", "instructions": row["question"],
                             "criteria": criteria, "choices": choices}},
    }
    req = urllib.request.Request("http://localhost:11434/v1/systemone",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return out["answers"]["sp"]["choice"]

correct = 0
total = 0
per_class = defaultdict(lambda: [0, 0])  # key -> [correct, total]
confusion = Counter()
t0 = time.time()
for i, row in enumerate(rows):
    gold = row["answer_key"]
    try:
        pred = ask(row)
    except Exception as e:
        pred = f"<error:{e}>"
    total += 1
    per_class[gold][1] += 1
    if pred == gold:
        correct += 1
        per_class[gold][0] += 1
    else:
        confusion[(gold, pred)] += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(rows)} acc={correct/total:.3f} elapsed={time.time()-t0:.0f}s")

print(f"\nOVERALL accuracy: {correct}/{total} = {correct/total:.4f}  ({time.time()-t0:.0f}s)")
print("\nPer-class (gold): correct/total  acc")
for k in sorted(per_class):
    c, t = per_class[k]
    print(f"  {k:35s} {c:4d}/{t:<4d} {c/t:.3f}")
print("\nTop confusions (gold -> pred): count")
for (g, p), n in confusion.most_common(20):
    print(f"  {g:35s} -> {p:35s} {n}")
