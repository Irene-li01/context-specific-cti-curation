"""
prepare_cyner.py
================
Converts CyNER CoNLL files (train.txt, valid.txt, test.txt) into the BIO JSON
format used by train_secbert.py, then combines with the existing
secbert_train_data.json to produce a unified training set.

CyNER label -> our model label mapping:
    Malware      -> TECH  (malware is a type of technology)
    System       -> TECH  (OS, software, platforms)
    Organization -> ACTOR (threat actors, APT groups, companies)
    Indicator    -> O     (skip -- handled by regex in ner_cti.py)
    Vulnerability-> O     (skip -- handled by regex / CVE pattern)

Output:
    Machine learning/combined_train_data.json   (used by train_secbert.py)

Usage:
    python "Machine learning/prepare_cyner.py"
"""

from __future__ import annotations
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent
TRAIN_DATA    = HERE / "train_data"
EXISTING_DATA = HERE / "secbert_train_data.json"
OUTPUT_FILE   = HERE / "combined_train_data.json"

CYNER_FILES = [
    TRAIN_DATA / "train.txt",
    TRAIN_DATA / "valid.txt",
    TRAIN_DATA / "test.txt",
]

# ── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {
    "B-Malware":      "B-TECH",
    "I-Malware":      "I-TECH",
    "B-System":       "B-TECH",
    "I-System":       "I-TECH",
    "B-Organization": "B-ACTOR",
    "I-Organization": "I-ACTOR",
    # Indicators and Vulnerabilities are handled by regex — map to O
    "B-Indicator":    "O",
    "I-Indicator":    "O",
    "B-Vulnerability":"O",
    "I-Vulnerability":"O",
    "O":              "O",
}


def parse_conll(filepath: Path) -> list[dict]:
    """Parse a CoNLL file into a list of {tokens, ner_tags} dicts."""
    records = []
    tokens, tags = [], []

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                # Blank line = sentence boundary
                if tokens:
                    records.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    token, label = parts
                    tokens.append(token)
                    tags.append(LABEL_MAP.get(label, "O"))

    # Flush any trailing sentence
    if tokens:
        records.append({"tokens": tokens, "ner_tags": tags})

    return records


def main():
    # 1. Convert CyNER files
    cyner_records = []
    for filepath in CYNER_FILES:
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found, skipping.")
            continue
        batch = parse_conll(filepath)
        cyner_records.extend(batch)
        print(f"  Parsed {len(batch):,} sentences from {filepath.name}")

    print(f"\n  CyNER total: {len(cyner_records):,} sentences")

    # 2. Load existing training data (has IND and TTP coverage)
    existing_records = []
    if EXISTING_DATA.exists():
        with EXISTING_DATA.open("r", encoding="utf-8") as f:
            existing_records = json.load(f)
        print(f"  Existing data: {len(existing_records):,} sentences")
    else:
        print("  WARNING: secbert_train_data.json not found — using CyNER only.")

    # 3. Combine and shuffle
    import random
    combined = cyner_records + existing_records
    random.seed(42)
    random.shuffle(combined)

    print(f"\n  Combined total: {len(combined):,} sentences")

    # 4. Label distribution summary
    from collections import Counter
    all_tags = [tag for rec in combined for tag in rec["ner_tags"]]
    tag_counts = Counter(all_tags)
    print("\n  Label distribution:")
    for label, count in sorted(tag_counts.items()):
        pct = count / len(all_tags) * 100
        print(f"    {label:<12} {count:>8,}  ({pct:.1f}%)")

    # 5. Save
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved combined dataset -> {OUTPUT_FILE.name}")
    print(f"\n  Next step: update data_file in train_secbert.py to point to:")
    print(f"    {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
