import argparse
import csv
import random
import re

# -------------------------------------------
#  SYNONYM DICTIONARIES (LARGE)
# -------------------------------------------

EN_SYNONYMS = {
    "move forward": [
        "go forward", "walk forward", "move ahead", "advance", "step forward",
        "proceed forward", "keep going forward"
    ],
    "move backward": [
        "go backward", "move back", "step back", "reverse", "go in reverse",
        "walk backwards", "back up"
    ],
    "turn left": [
        "go left", "move left", "turn to the left", "rotate left",
        "shift left", "head left"
    ],
    "turn right": [
        "go right", "move right", "turn to the right", "rotate right",
        "shift right", "head right"
    ],
    "sit down": [
        "sit", "take a seat", "have a seat", "sit properly",
        "sit now", "lower yourself and sit"
    ],
    "stand up": [
        "stand", "get up", "rise", "stand straight", "stand now"
    ],
    "do a trick": [
        "perform a trick", "show a trick", "do something cool",
        "show me something", "do something impressive"
    ]
}

AR_SYNONYMS = {
    "تحرك لقدام": [
        "تقدم", "تحرك للأمام", "الى الأمام", "امش لقدام", "سر للأمام"
    ],
    "تحرك لورا": [
        "ارجع", "تراجع", "تحرك للخلف", "الى الخلف", "امش لورا"
    ],
    "لف شمال": [
        "انعطف يسار", "اتجه يسارا", "روح شمال", "خذ يسار", "اترك يمين وروح يسار"
    ],
    "لف يمين": [
        "انعطف يمين", "اتجه يمينا", "روح يمين", "خذ يمين", "اترك يسار وروح يمين"
    ],
    "اجلس": [
        "اقعد", "تفضل بالجلوس", "اجلس الآن", "اقعد لو سمحت"
    ],
    "قف": [
        "انهض", "قف واقف", "وقف", "وقف حالاً"
    ],
    "قم بحركة": [
        "اعمل حركة", "قم بخدعة", "فرجيني حركة", "سوي حركة حلوة"
    ]
}

# -------------------------------------------
#  AUGMENTATION FUNCTION
# -------------------------------------------

def replace_with_synonyms(sentence, lang):
    """Replace a phrase with a synonym if found."""
    syn_dict = EN_SYNONYMS if lang == "en" else AR_SYNONYMS

    # Try long keys first (important)
    keys_sorted = sorted(syn_dict.keys(), key=len, reverse=True)

    for key in keys_sorted:
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, sentence.lower()):
            return re.sub(pattern,
                          random.choice(syn_dict[key]),
                          sentence,
                          flags=re.IGNORECASE)

    return sentence


def detect_lang(text):
    return "ar" if any("\u0600" <= c <= "\u06FF" for c in text) else "en"


def augment_text(text, num_variants=10):
    variants = set()
    lang = detect_lang(text)

    for _ in range(num_variants):
        new_text = text

        # Apply synonyms with probability
        if random.random() < 0.8:
            new_text = replace_with_synonyms(new_text, lang)

        # Random shuffle small additions
        if random.random() < 0.3:
            additions_en = ["please", "now", "right away", "quickly"]
            additions_ar = ["من فضلك", "يلا", "هلق", "بسرعة"]

            if lang == "en":
                new_text += " " + random.choice(additions_en)
            else:
                new_text += " " + random.choice(additions_ar)

        variants.add(new_text.strip())

    return list(variants)


# -------------------------------------------
#  MAIN SCRIPT
# -------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multiplier", type=int, default=10,
                        help="How many augmented samples per row")
    args = parser.parse_args()

    rows_out = []

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            text = row["text"]
            label = row["label"]

            # original row
            rows_out.append(row)

            # augmented rows
            aug_list = augment_text(text, num_variants=args.multiplier)
            for aug in aug_list:
                rows_out.append({"text": aug, "label": label})

    # Write output
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done! Generated {len(rows_out)} rows.")


if __name__ == "__main__":
    main()

