import pandas as pd
import re
import json
from pathlib import Path

FILTERED_DIR = Path("data/cleaned")
FALLBACK_XLSX = "data/misp/misp_categorized.xlsx"
FALLBACK_SHEET = "Attribution"
OUTPUT_CSV = "ner/clean_cti_dataset.csv"
OUTPUT_JSON = "ner/cti_ner_input.json"
OUTPUT_STATS = "ner/cti_ner_input_stats.json"

TEXT_RICH_TYPES = {
    "campaign-id",
    "campaign-name",
    "comment",
    "text",
    "threat-actor",
}

STRUCTURED_IOC_TYPES = {
    "dns-soa-email",
    "x509-fingerprint-md5",
    "x509-fingerprint-sha1",
    "x509-fingerprint-sha256",
}

CONTEXTUAL_TYPES = {
    "malware-sample",
    "snort",
    "sigma",
    "stix2-pattern",
    "target-external",
    "target-location",
    "target-machine",
    "target-org",
    "target-user",
    "vulnerability",
    "yara",
}

ALLOWED_TYPES = TEXT_RICH_TYPES | STRUCTURED_IOC_TYPES | CONTEXTUAL_TYPES


def latest_filtered_path():
    candidates = sorted(FILTERED_DIR.glob("misp_filtered_*.json"))
    return candidates[-1] if candidates else None


def tags_to_text(tags):
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags if str(tag).strip())
    return str(tags or "")


def load_filtered_json(path):
    try:
        from crypto_utils import decrypt_json
        data = decrypt_json(path)
    except ImportError:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    rows = []
    for attr in data:
        rows.append({
            "event_id": attr.get("event_id", ""),
            "attr_id": attr.get("id", ""),
            "category": attr.get("category", ""),
            "type": attr.get("type", ""),
            "indicator": attr.get("value", ""),
            "description": attr.get("comment", ""),
            "event_info": attr.get("event_info", ""),
            "tags": tags_to_text(attr.get("tags", [])),
        })
    return pd.DataFrame(rows)


def load_fallback_xlsx():
    df = pd.read_excel(FALLBACK_XLSX, sheet_name=FALLBACK_SHEET)
    print("Columns:", df.columns.tolist())
    df = df.rename(columns={
        "ID": "attr_id",
        "Type": "type",
        "Value": "indicator",
        "Comment": "description",
        "Event ID": "event_id",
        "Event Info": "event_info",
        "Tags": "tags"
    })
    if "category" not in df.columns:
        df["category"] = FALLBACK_SHEET
    return df


filtered_path = latest_filtered_path()
if filtered_path:
    print(f"Using filtered CTI input: {filtered_path}")
    df = load_filtered_json(filtered_path)
    source_file = str(filtered_path)
else:
    print(f"No filtered CTI found under {FILTERED_DIR}; falling back to {FALLBACK_XLSX}")
    df = load_fallback_xlsx()
    source_file = FALLBACK_XLSX

input_rows = len(df)

# Make sure all required columns exist
for col in ["event_id", "attr_id", "category", "indicator", "type", "description", "tags", "event_info"]:
    if col not in df.columns:
        df[col] = ""

# Clean key columns before filtering. This avoids silently dropping rows because
# of mixed case or whitespace in the MISP export.
df["indicator"] = df["indicator"].astype(str).str.strip()
df["type"] = df["type"].astype(str).str.strip().str.lower()

# Remove rows where key fields are missing or empty
df = df.dropna(subset=["indicator", "type"])
df = df[df["indicator"].astype(str).str.strip() != ""]
df = df[df["type"].astype(str).str.strip() != ""]
df = df[df["indicator"].str.lower() != "nan"]
df = df[df["type"].str.lower() != "nan"]

candidate_rows = len(df)

# Keep semantic fields plus contextual objects that help NER.
# The filtered dataset may contain hundreds of thousands of raw URL/hash/IP rows;
# those are useful for IoC display, but too noisy and expensive for entity extraction.
excluded_by_type_df = df[~df["type"].isin(ALLOWED_TYPES)]
top_excluded_types = {
    str(type_name): int(count)
    for type_name, count in excluded_by_type_df["type"].value_counts().head(20).items()
}
df = df[df["type"].isin(ALLOWED_TYPES)]

# Fill missing values for text fields
df["description"] = df["description"].fillna("not available")
df["tags"] = df["tags"].fillna("none")
df["event_info"] = df["event_info"].fillna("")


def repair_mojibake(text):
    text = str(text)
    replacements = {
        "â€œ": '"',
        "â€�": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
        "â€TM": "'",
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã¡": "á",
        "Ã¢": "â",
        "Ã±": "ñ",
        "Ã¶": "ö",
        "Ã¼": "ü",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    if "â" not in text and "Ã" not in text:
        return text
    try:
        repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
    except UnicodeError:
        repaired = text
    bad_before = text.count("â") + text.count("Ã")
    bad_after = repaired.count("â") + repaired.count("Ã")
    return repaired if bad_after < bad_before else text


def clean_text(text):
    text = repair_mojibake(text).strip()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    if text.lower() == "nan":
        return ""
    return text


def shorten_text(text, max_length=140):
    text = clean_text(text)
    if len(text) <= max_length:
        return text

    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if 20 <= len(sentence) <= max_length:
        return sentence

    return text[: max_length - 3].rstrip() + "..."


def build_title(row):
    event_info = clean_text(row.get("event_info", ""))
    if event_info:
        return shorten_text(event_info)

    category = clean_text(row.get("category", "")) or "CTI"
    indicator = shorten_text(row.get("indicator", ""), max_length=90)
    if indicator:
        return f"{category} indicator: {indicator}"

    attr_type = clean_text(row.get("type", "")) or "record"
    return f"{category} {attr_type} record"


# Apply cleaning to text columns
df["indicator"] = df["indicator"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)
df["tags"] = df["tags"].apply(clean_text)
df["event_info"] = df["event_info"].apply(clean_text)
df["category"] = df["category"].apply(clean_text)

# Remove duplicate rows
df = df.drop_duplicates(subset=["indicator", "type"])

# Save cleaned dataset
df.to_csv(OUTPUT_CSV, index=False)

# Convert cleaned data into NER input format
ner_input = []

for _, row in df.iterrows():
    text_parts = []

    if row["indicator"]:
        text_parts.append(f"Category: {row['category']}")
        text_parts.append(f"Type: {row['type']}")
        text_parts.append(f"Indicator: {row['indicator']}")

    if row["description"] and row["description"].lower() != "not available":
        text_parts.append(f"Description: {row['description']}")

    if row["event_info"]:
        text_parts.append(f"Context: {row['event_info']}")

    if row["tags"] and row["tags"].lower() != "none":
        text_parts.append(f"Tags: {row['tags']}")

    raw_text = " | ".join(text_parts)
    title = build_title(row)

    ner_input.append({
        "event_id": str(row["event_id"]),
        "attr_id": str(row["attr_id"]),
        "category": str(row["category"]),
        "type": str(row["type"]),
        "indicator": str(row["indicator"]),
        "title": title,
        "description": str(row["description"]),
        "embedding_text": raw_text,
        "raw_text": raw_text
    })

# Save JSON file for NER module
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(ner_input, f, indent=2, ensure_ascii=False)

ner_stats = {
    "source_file": source_file,
    "input_rows": int(input_rows),
    "candidate_rows": int(candidate_rows),
    "ner_rows": len(ner_input),
    "excluded_missing_required": int(input_rows - candidate_rows),
    "excluded_by_type": int(len(excluded_by_type_df)),
    "top_excluded_types": top_excluded_types,
    "allowed_types": sorted(ALLOWED_TYPES),
}
with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    json.dump(ner_stats, f, indent=2, ensure_ascii=False)

print("Cleaning finished.")
print("clean_cti_dataset.csv has been created.")
print("NER input created: ner/cti_ner_input.json")
print("NER input stats created: ner/cti_ner_input_stats.json")
print("Number of rows:", len(ner_input))
