#!/usr/bin/env python3
# /mnt/data/scripts/generate_synthetic_stt.py
import json
import random
import os

OUTPUT_DIR = "data"
NUM_TRAIN = 850
NUM_DEV = 175
NUM_STRESS = 100

LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

NAMES = ["Ramesh", "Suresh", "Aditi", "Priya", "John", "Alice", "Bob", "Smith", "Kumar", "Sharma"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "New York", "London", "Pune", "Hyderabad"]
LOCATIONS = ["Starbucks", "Central Park", "Terminal 2", "Main Street", "Apollo Hospital"]
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]
DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "example"]

# STT spoken alternatives
DIGIT_MAP = {
    "0": ["zero", "oh", "o"],
    "1": ["one", "one"],
    "2": ["two", "to", "too"],
    "3": ["three"],
    "4": ["four", "for"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight"],
    "9": ["nine"]
}

def choose_spoken_or_digits(s, prob_spoken=0.6):
    """Given a digits string s, return spoken tokens, numeric tokens, or mixed."""
    if random.random() < prob_spoken:
        # spoken full
        return " ".join(random.choice(DIGIT_MAP[d]) for d in s)
    else:
        # numeric with possible separators or grouped
        if len(s) == 16:
            # group into 4s
            sep = random.choice([" ", "-", ""])
            grouped = " ".join(s[i:i+4] for i in range(0, 16, 4))
            if random.random() < 0.5:
                return grouped
            else:
                # mixed spoken + digits (first group spoken)
                first_group = " ".join(random.choice(DIGIT_MAP[d]) for d in s[:4])
                return f"{first_group} {s[4:8]} {s[8:12]} {s[12:]}"
        else:
            # phone: sometimes grouped (3 3 4), sometimes plain, sometimes spoken
            if random.random() < 0.5:
                return " ".join(s[i:i+3] for i in range(0, len(s), 3))
            else:
                return " ".join(s[i:i+2] for i in range(0, len(s), 2))

def gen_phone():
    n = random.choice([10,10,10,7,8])
    s = "".join(str(random.randint(0,9)) for _ in range(n))
    return choose_spoken_or_digits(s), s

def gen_card():
    s = "".join(str(random.randint(0,9)) for _ in range(16))
    return choose_spoken_or_digits(s), s

def gen_email():
    # yield both spoken and @ forms
    local = random.choice(NAMES).lower()
    if random.random() < 0.4:
        # spoken form: "first dot last at gmail dot com" or "first last at gmail dot com"
        if random.random() < 0.5:
            local_spoken = local
        else:
            local_spoken = local + " dot " + random.choice(NAMES).lower()
        return f"{local_spoken} at {random.choice(DOMAINS)} dot com"
    else:
        # normal form
        sep = random.choice(["", ".", "_"])
        local2 = local + (sep + random.choice(NAMES).lower() if random.random() < 0.5 else "")
        return f"{local2}@{random.choice(DOMAINS)}.com"

def gen_date():
    style = random.choice(["spoken", "mdy", "dmy", "yearspoken"])
    day = random.randint(1,28)
    month = random.choice(MONTHS)
    year = random.randint(1980,2025)
    if style == "spoken":
        # "twenty third of february two thousand twenty"
        day_spoken = str(day)
        return f"{day_spoken} {month} {year}"
    if style == "mdy":
        return f"{month} {day} {year}"
    if style == "dmy":
        return f"{day} {month} {year}"
    if style == "yearspoken":
        return f"{day} {month} {str(year)}"

TEMPLATES = [
    "my name is {PERSON_NAME} and i live in {CITY}",
    "contact me at {EMAIL} or {PHONE}",
    "payment using card {CREDIT_CARD} confirmed",
    "meeting on {DATE} at {LOCATION}",
    "is {PERSON_NAME} available in {CITY}",
    "send details to {EMAIL}",
    "my number is {PHONE}",
    "reservation at {LOCATION} for {DATE}",
    "call {PERSON_NAME} at {PHONE}",
    "the card ending with {CREDIT_CARD} failed"  # some variations
]

def build_example(uid, kind=None):
    # create one example; if kind is None pick a random template
    if kind is None:
        template = random.choice(TEMPLATES)
    else:
        # map simple templates
        mapping = {
            "CREDIT_CARD": "payment using card {CREDIT_CARD} confirmed",
            "PHONE": "my number is {PHONE}",
            "EMAIL": "send details to {EMAIL}",
            "PERSON_NAME": "this is {PERSON_NAME} speaking",
            "DATE": "i booked it on {DATE}",
            "CITY": "i am in {CITY}",
            "LOCATION": "meeting at {LOCATION}"
        }
        template = mapping.get(kind, random.choice(TEMPLATES))

    tokens = template.split()
    current_text = ""
    entities = []

    for token in tokens:
        content = token
        label = None
        if token.startswith("{") and token.endswith("}"):
            label = token[1:-1]
            if label == "PERSON_NAME":
                content = random.choice(NAMES)
            elif label == "CITY":
                content = random.choice(CITIES)
            elif label == "LOCATION":
                content = random.choice(LOCATIONS)
            elif label == "EMAIL":
                content = gen_email()
            elif label == "PHONE":
                content, _ = gen_phone()
            elif label == "CREDIT_CARD":
                content, _ = gen_card()
            elif label == "DATE":
                content = gen_date()

        # add space if needed
        if len(current_text) > 0:
            current_text += " "
        start_idx = len(current_text)
        current_text += content
        end_idx = len(current_text)

        if label:
            entities.append({"start": start_idx, "end": end_idx, "label": label})

    # lowercase to mimic STT
    return {"id": uid, "text": current_text.lower(), "entities": entities}

def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    synth_train = [build_example(f"synth_train_{i}") for i in range(NUM_TRAIN)]
    synth_dev = [build_example(f"synth_dev_{i}") for i in range(NUM_DEV)]
    synth_stress = [build_example(f"synth_stress_{i}") for i in range(NUM_STRESS)]

    save_jsonl(synth_train, os.path.join(OUTPUT_DIR, "synth_train.jsonl"))
    save_jsonl(synth_dev, os.path.join(OUTPUT_DIR, "synth_dev.jsonl"))
    save_jsonl(synth_stress, os.path.join(OUTPUT_DIR, "synth_stress.jsonl"))

    print("Wrote synthetic files:")
    print("  ", os.path.join(OUTPUT_DIR, "synth_train.jsonl"))
    print("  ", os.path.join(OUTPUT_DIR, "synth_dev.jsonl"))
    print("  ", os.path.join(OUTPUT_DIR, "synth_stress.jsonl"))

if __name__ == "__main__":
    main()