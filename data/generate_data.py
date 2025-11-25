import json
import random
import os

# Configuration
OUTPUT_DIR = "data"
NUM_TRAIN = 850
NUM_DEV = 175
NUM_STRESS = 100

LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

# Vocabulary for generation
NAMES = ["Ramesh", "Suresh", "Aditi", "Priya", "John", "Alice", "Bob", "Smith", "Kumar", "Sharma"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "New York", "London", "Pune", "Hyderabad"]
LOCATIONS = ["Starbucks", "Central Park", "Terminal 2", "Main Street", "Apollo Hospital"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "example"]

# Digit mapping for STT noise
DIGIT_MAP = {
    "0": ["zero", "oh", "null"],
    "1": ["one", "first"],
    "2": ["two", "second"],
    "3": ["three"],
    "4": ["four"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight"],
    "9": ["nine"]
}

def noisy_digits(digits_str):
    """Converts '1234' to 'one two three four' randomly."""
    words = []
    for d in digits_str:
        if d in DIGIT_MAP:
            words.append(random.choice(DIGIT_MAP[d]))
        else:
            words.append(d)
    return " ".join(words)

def generate_phone():
    # Generate 10 digit number
    num = "".join([str(random.randint(0, 9)) for _ in range(10)])
    # Return noisy version and original (we use noisy for text)
    return noisy_digits(num)

def generate_card():
    # Generate 16 digit number
    num = "".join([str(random.randint(0, 9)) for _ in range(16)])
    return noisy_digits(num)

def generate_email():
    name = random.choice(NAMES).lower()
    domain = random.choice(DOMAINS)
    # STT often says "at" and "dot"
    return f"{name} at {domain} dot com"

def generate_date():
    day = str(random.randint(1, 30))
    month = random.choice(MONTHS)
    year = str(random.randint(1980, 2025))
    
    fmt = random.choice([1, 2, 3])
    if fmt == 1:
        return f"{day} of {month} {noisy_digits(year)}"
    elif fmt == 2:
        return f"{month} {noisy_digits(day)} {year}"
    else:
        return f"{day} {month}"

def generate_sentence(uid):
    text_parts = []
    entities = []
    
    # Templates for sentence structure
    templates = [
        "my name is {PERSON_NAME} and i live in {CITY}",
        "contact me at {EMAIL} or {PHONE}",
        "payment using card {CREDIT_CARD} confirmed",
        "meeting on {DATE} at {LOCATION}",
        "is {PERSON_NAME} available in {CITY}",
        "send details to {EMAIL}",
        "my number is {PHONE}",
        "reservation at {LOCATION} for {DATE}"
    ]
    
    template = random.choice(templates)
    
    # Split by spaces to reconstruct and track offsets
    tokens = template.split()
    current_text = ""
    
    for token in tokens:
        content = ""
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
                content = generate_email()
            elif label == "PHONE":
                content = generate_phone()
            elif label == "CREDIT_CARD":
                content = generate_card()
            elif label == "DATE":
                content = generate_date()
        else:
            content = token

        # Add space if not start
        if len(current_text) > 0:
            current_text += " "
        
        start_idx = len(current_text)
        current_text += content
        end_idx = len(current_text)
        
        if label:
            entities.append({
                "start": start_idx,
                "end": end_idx,
                "label": label
            })

    # Lowercase everything as per STT noise commonality
    return {
        "id": uid,
        "text": current_text.lower(),
        "entities": entities
    }

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Generating {NUM_TRAIN} train examples...")
    train_data = [generate_sentence(f"train_{i}") for i in range(NUM_TRAIN)]
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))

    print(f"Generating {NUM_DEV} dev examples...")
    dev_data = [generate_sentence(f"dev_{i}") for i in range(NUM_DEV)]
    save_jsonl(dev_data, os.path.join(OUTPUT_DIR, "dev.jsonl"))

    print(f"Generating {NUM_STRESS} stress examples...")
    stress_data = [generate_sentence(f"stress_{i}") for i in range(NUM_STRESS)]
    save_jsonl(stress_data, os.path.join(OUTPUT_DIR, "stress.jsonl"))
    
    # Create an empty test.jsonl as placeholder
    with open(os.path.join(OUTPUT_DIR, "test.jsonl"), "w") as f:
        pass

    print("Data generation complete.")

if __name__ == "__main__":
    main()