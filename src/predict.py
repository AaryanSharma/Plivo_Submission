# src/predict.py
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import re

EMAIL_THRESH = 0.4
CREDIT_CARD_THRESH = 0.5
PHONE_THRESH = 0.2
PERSON_NAME_THRESH = 0.3
DEFAULT_THRESH = 0.5

# loose regexes that operate on the predicted character span (works on noisy STT forms too)
EMAIL_RE = re.compile(r"(@| at | dot | \. )", flags=re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
CREDIT_CARD_DIGITS_RE = re.compile(r"\d(?:[^0-9]*\d){11,19}")  # look for 12-19 digits with potential separators
PHONE_DIGITS_RE = re.compile(r"\d(?:[^0-9]*\d){6,20}")  # 7-20 digits possibly separated

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0", "null": "0",
    "one": "1", "first": "1",
    "two": "2", "second": "2", "to": "2", "too": "2",
    "three": "3", "third": "3",
    "four": "4", "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}
# ----------------- POST-PROCESSING HELPERS (insert into src/predict.py) -----------------

FILLER_STOPWORDS = {
    "haan","ha","so","my","naam","is","main","rehte","in","we","will","meet","on",
    "uh","actually","i","am","please","also","send","call","number","new","old"
}

EMAIL_INNER_RE = re.compile(
    r"(?:[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)\s*(?:@|at)\s*[A-Za-z0-9-]+(?:\s*(?:\.|dot)\s*[A-Za-z0-9-]+)+",
    flags=re.IGNORECASE
)

PHONE_CUE_RE = re.compile(r"(phone|mobile|number|call)\b.*", flags=re.IGNORECASE)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def span_text(full_text: str, span: tuple) -> str:
    s,e = span
    return full_text[s:e]

def is_filler_name(s: str) -> bool:
    s = s.strip().lower()
    if len(s) <= 2:
        return True
    # if any token is a filler stopword and the span is short, treat as filler
    toks = s.split()
    if all(t in FILLER_STOPWORDS for t in toks):
        return True
    # otherwise not filler
    return False

def shrink_email_span(full_text: str, start:int, end:int):
    """Within full_text[start:end], find the best email-like substring and return new (s,e).
       If none found, return original (start,end)."""
    sub = full_text[start:end]
    # try direct @ pattern first
    m = re.search(r"[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", sub)
    if m:
        s_off = start + m.start()
        e_off = start + m.end()
        return s_off, e_off
    # else try spoken 'at'/'dot' pattern
    m2 = EMAIL_INNER_RE.search(sub)
    if m2:
        s_off = start + m2.start()
        e_off = start + m2.end()
        return s_off, e_off
    return start, end

def find_phone_after_cue(full_text: str):
    """Return (start,end) of digit-like sequence following 'phone' cues, or None."""
    # Locate phone cues
    for m in PHONE_CUE_RE.finditer(full_text):
        # consider substring after cue
        tail = full_text[m.end(): m.end() + 200]  # enough lookahead
        # Convert tail to digits using existing text_to_digits helper (must be available)
        digits = text_to_digits(tail)
        if len(digits) >= 7:
            # find the character-level span of the token sequence in tail that produced the digits
            # greedy approach: expand from m.end() until digits count >=7
            acc = ""
            pos = m.end()
            # split tail into tokens with their char indices
            for mo in re.finditer(r"\S+", tail):
                tok = mo.group(0)
                tok_digits = text_to_digits(tok)
                if tok_digits:
                    acc += tok_digits
                pos_end = m.end() + mo.end()
                if len(acc) >= 7:
                    # compute start index: first char of first digit-producing token
                    # naive start: find first digit-producing token in tail
                    first_m = re.search(r"\S+", tail)
                    # more precise: find span start by searching back to first token with digits
                    # but simplest: return the slice covering mo.start() back to where the digits began
                    # locate the beginning index in full_text for the window we have consumed
                    # compute start as m.end() + index of first digit-char-containing token
                    # implement simple search for first digit-producing token:
                    running = ""
                    start_char = None
                    for mo2 in re.finditer(r"\S+", tail):
                        tok2 = mo2.group(0)
                        if text_to_digits(tok2):
                            start_char = m.end() + mo2.start()
                            break
                    if start_char is None:
                        start_char = m.end()
                    return start_char, m.end() + mo.end()
    return None

# ----------------- END helpers -----------------

def text_to_digits(text: str) -> str:
    """
    Convert a span like "four two 4 oh" or "one two three" into "4240" or "123".
    Keeps only digits and known spoken-digit words; strips punctuation.
    """
    # Normalize and remove punctuation that can interrupt tokens
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = clean_text.split()
    digits = []
    for t in tokens:
        # if the token itself is numeric (e.g., "4242" or "4"), keep numeric characters
        if t.isdigit():
            digits.append(t)
        else:
            # handle composite tokens that contain digits mixed with letters (rare)
            # extract any run of digits
            m = re.findall(r"\d+", t)
            if m:
                digits.extend(m)
            elif t in WORD_TO_DIGIT:
                digits.append(WORD_TO_DIGIT[t])
            # else ignore filler tokens (e.g., "please", "call", etc.)
    return "".join(digits)

def span_threshold_for_label(label: str):
    if label == "EMAIL":
        return EMAIL_THRESH
    if label == "CREDIT_CARD":
        return CREDIT_CARD_THRESH
    if label == "PHONE":
        return PHONE_THRESH
    if label == "PERSON_NAME":
        return PERSON_NAME_THRESH
    return DEFAULT_THRESH

def bio_to_spans_with_scores(text, offsets, label_ids, label_scores):
    """
    offsets: list of (start, end) per token
    label_ids: list of predicted label ids per token
    label_scores: list of max-prob per token (float)
    Returns: list of (start, end, label, span_score)
    """
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
                # broken I-tag sequence: start new span
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

def validate_span(text_span: str, label: str) -> bool:
    s = text_span.strip().lower()

    if label == "EMAIL":
        # require at least 'at' or '.' pattern or an @ sign in the span
        return bool(EMAIL_RE.search(s))

    # Normalize spoken numbers -> digits for numeric PII checks
    if label == "CREDIT_CARD":
        digits = text_to_digits(s)
        # credit card: allow 12-19 digits (many cards are 16; accept 12 as partial)
        return len(digits) >= 12

    if label == "PHONE":
        digits = text_to_digits(s)
        # phone numbers (spoken or numeric) usually have 7-15 digits; adjust as needed
        return 7 <= len(digits) <= 20

    # default: allow the span (rely on model score / thresholds for precision)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--span_threshold_scale", type=float, default=1.0,
                    help="Multiply all span thresholds by this factor (use <1 to be lenient).")
    args = ap.parse_args()

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # [seq_len, num_labels]
                probs = F.softmax(logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)
                pred_ids = pred_ids.cpu().tolist()
                max_probs = max_probs.cpu().tolist()

            # decode spans with per-token score
            spans = bio_to_spans_with_scores(text, offsets, pred_ids, max_probs)

            # first pass: filter by threshold + validate_span (but do NOT yet apply email shrinking / filler removal)
            candidate_spans = []
            for s, e, lab, span_score in spans:
                # apply label-specific threshold
                thr = span_threshold_for_label(lab) * args.span_threshold_scale
                if span_score < thr:
                    continue
                # preliminary accept; final validation / shrinking done below
                candidate_spans.append((int(s), int(e), lab, float(span_score)))

            # post-processing pass: shrink emails, filter filler person names, validate spans
            final_spans = []
            for s, e, lab, span_score in candidate_spans:
                # pull the raw span text
                span_text = text[s:e]

                # 1) shrink email spans to inner email-like substring (if found)
                if lab == "EMAIL":
                    new_s, new_e = shrink_email_span(text, s, e)
                    s, e = int(new_s), int(new_e)
                    span_text = text[s:e]

                # 2) filter PERSON_NAME filler short spans
                if lab == "PERSON_NAME":
                    if is_filler_name(span_text):
                        # drop obvious filler names
                        continue
                    if len(span_text.strip()) < 3:
                        continue

                # 3) final validate (numeric checks, email pattern, etc.)
                if not validate_span(span_text, lab):
                    continue

                # accepted span
                final_spans.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                        "score": float(span_score),
                    }
                )

            # 4) PHONE fallback: if no PHONE predicted, try to find a phone after a cue
            has_phone = any(ent["label"] == "PHONE" for ent in final_spans)
            if not has_phone:
                ph_span = find_phone_after_cue(text)  # returns (start,end) or None
                if ph_span is not None:
                    s2, e2 = ph_span
                    candidate_text = text[int(s2):int(e2)]
                    # validate before adding
                    if validate_span(candidate_text, "PHONE"):
                        final_spans.append(
                            {
                                "start": int(s2),
                                "end": int(e2),
                                "label": "PHONE",
                                "pii": True,
                                "score": 0.0,  # synthetic / fallback score (optional)
                            }
                        )

            results[uid] = final_spans

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()
