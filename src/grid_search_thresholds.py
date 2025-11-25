#!/usr/bin/env python3
import json
import argparse
import itertools
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import math

# copy of small validation helpers from predict.py
def bio_to_spans_with_scores(text, offsets, label_ids, label_scores):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_scores = []
    for (start, end), lid, score in zip(offsets, label_ids, label_scores):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                span_score = min(current_scores) if current_scores else 0.0
                spans.append((current_start, current_end, current_label, float(span_score)))
                current_label = None
                current_scores = []
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                span_score = min(current_scores) if current_scores else 0.0
                spans.append((current_start, current_end, current_label, float(span_score)))
            current_label = ent_type
            current_start = start
            current_end = end
            current_scores = [score]
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                current_scores.append(score)
            else:
                if current_label is not None:
                    span_score = min(current_scores) if current_scores else 0.0
                    spans.append((current_start, current_end, current_label, float(span_score)))
                current_label = ent_type
                current_start = start
                current_end = end
                current_scores = [score]
    if current_label is not None:
        span_score = min(current_scores) if current_scores else 0.0
        spans.append((current_start, current_end, current_label, float(span_score)))
    return spans

def load_dev(dev_path):
    items = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append(obj)
    return items

def compute_p_r_f(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    return prec, rec, f1

def eval_predictions(gold_items, pred_map):
    # gold_items: list of dicts with id, entities
    # pred_map: uid -> [(s,e,label)]
    labels = set()
    for obj in gold_items:
        for s,e,lab in [(e["start"], e["end"], e["label"]) for e in obj.get("entities", [])]:
            labels.add(lab)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gold_map = {}
    for obj in gold_items:
        gold_map[obj["id"]] = [(e["start"], e["end"], e["label"]) for e in obj.get("entities", [])]
    for uid in gold_map:
        g = set(gold_map.get(uid, []))
        p = set(pred_map.get(uid, []))
        for span in p:
            if span in g:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in g:
            if span not in p:
                fn[span[2]] += 1
    # per-entity metrics printed by caller if desired
    # PII metric
    pii_tp=pii_fp=pii_fn=0
    non_tp=non_fp=non_fn=0
    for uid in gold_map:
        g_spans = gold_map.get(uid, [])
        p_spans = pred_map.get(uid, [])
        g_pii = set((s,e,"PII") for s,e,lab in g_spans if label_is_pii(lab))
        p_pii = set((s,e,"PII") for s,e,lab in p_spans if label_is_pii(lab))
        g_non = set((s,e,"NON") for s,e,lab in g_spans if not label_is_pii(lab))
        p_non = set((s,e,"NON") for s,e,lab in p_spans if not label_is_pii(lab))
        for span in p_pii:
            if span in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for span in g_pii:
            if span not in p_pii:
                pii_fn += 1
    p,r,f = compute_p_r_f(pii_tp, pii_fp, pii_fn)
    return {"pii_prec":p, "pii_rec":r, "pii_f1":f, "tp":tp, "fp":fp, "fn":fn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out_aug")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    dev_items = load_dev(args.dev)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    # thresholds to try (coarse grid)
    credit_ts = [0.5, 0.6, 0.7, 0.8, 0.9]
    email_ts = [0.4, 0.5, 0.6, 0.7]
    phone_ts = [0.2, 0.35, 0.5, 0.6]
    name_ts  = [0.3, 0.4, 0.5, 0.6]
    global_scale = [0.8, 1.0, 1.2]

    results = []
    combos = list(itertools.product(credit_ts, email_ts, phone_ts, name_ts, global_scale))
    print(f"Testing {len(combos)} threshold combinations...")
    for c_t, e_t, p_t, n_t, g_s in combos:
        # run through dev and predict in-memory
        preds = {}
        for obj in dev_items:
            uid = obj["id"]
            text = obj["text"]
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                probs = F.softmax(logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)
                pred_ids = pred_ids.cpu().tolist()
                max_probs = max_probs.cpu().tolist()
            spans = bio_to_spans_with_scores(text, offsets, pred_ids, max_probs)
            ents = []
            for s,e,lab,span_score in spans:
                # pick threshold
                thr = 0.5
                if lab == "CREDIT_CARD":
                    thr = c_t
                elif lab == "EMAIL":
                    thr = e_t
                elif lab == "PHONE":
                    thr = p_t
                elif lab == "PERSON_NAME":
                    thr = n_t
                thr *= g_s
                if span_score >= thr:
                    ents.append((int(s), int(e), lab))
            preds[uid] = ents
        metrics = eval_predictions(dev_items, preds)
        results.append((metrics["pii_prec"], metrics["pii_rec"], metrics["pii_f1"], c_t, e_t, p_t, n_t, g_s))

    # sort by pii precision desc, then f1 desc
    results.sort(key=lambda x: (x[0], x[2]), reverse=True)
    print("Top 10 threshold settings (pii_prec, pii_rec, pii_f1, credit_t, email_t, phone_t, name_t, scale):")
    for row in results[:10]:
        print(row)

if __name__ == "__main__":
    main()