#!/usr/bin/env python3
# Save as Plivo/scripts/analyze_errors.py

import json
import os
from collections import defaultdict

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def spans_set(spans):
    # convert list of dict spans to set of tuples (s,e,label)
    return set((s["start"], s["end"], s["label"]) for s in spans)

def classify_example(gold_spans, pred_spans):
    gset = spans_set(gold_spans)
    pset = spans_set(pred_spans)
    tp = gset & pset
    fp = pset - gset
    fn = gset - pset
    return list(tp), list(fp), list(fn)

def main(gold_path="data/stress.jsonl", pred_path="out_aug/stress_pred_final.json", out_errors="data/stress_errors.jsonl", top_n=50):
    gold_items = {obj["id"]: obj for obj in load_jsonl(gold_path)}
    pred_obj = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_obj = json.load(f)

    stats = defaultdict(int)
    error_examples = []

    for uid, gold in gold_items.items():
        preds = pred_obj.get(uid, [])
        tp, fp, fn = classify_example(gold.get("entities", []), preds)
        for s in tp:
            stats[("TP", s[2])] += 1
        for s in fp:
            stats[("FP", s[2])] += 1
        for s in fn:
            stats[("FN", s[2])] += 1

        if fp or fn:
            # collect example for inspection
            error_examples.append({
                "id": uid,
                "text": gold["text"],
                "gold": gold.get("entities", []),
                "pred": preds,
                "tp": [{"start":a,"end":b,"label":c} for (a,b,c) in tp],
                "fp": [{"start":a,"end":b,"label":c} for (a,b,c) in fp],
                "fn": [{"start":a,"end":b,"label":c} for (a,b,c) in fn],
            })

    # sort error_examples by number of errors descending
    error_examples.sort(key=lambda x: (len(x["fp"]) + len(x["fn"])), reverse=True)

    # print summary
    print("Summary counts (TP/FP/FN by label):")
    for (kind, lab), cnt in sorted(stats.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{lab:15s} {kind:3s} {cnt}")

    # write top failures to output jsonl for manual inspection
    with open(out_errors, "w", encoding="utf-8") as f:
        for ex in error_examples[:top_n]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {min(len(error_examples), top_n)} error examples to {out_errors}")
    print("Example (first):")
    if error_examples:
        ex = error_examples[0]
        print("ID:", ex["id"])
        print("Text:", ex["text"])
        print("Gold:", ex["gold"])
        print("Pred:", ex["pred"])
        print("FP:", ex["fp"])
        print("FN:", ex["fn"])
    else:
        print("No errors found (unexpected)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/stress.jsonl")
    ap.add_argument("--pred", default="out_aug/stress_pred_final.json")
    ap.add_argument("--out", default="data/stress_errors.jsonl")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    main(gold_path=args.gold, pred_path=args.pred, out_errors=args.out, top_n=args.top)
