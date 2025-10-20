import os
import pandas as pd
from collections import defaultdict
import json
import re

INPUT_TSV = "./data/tsv/ru/bionnel_ru_train.tsv"   # Path to your TSV
RAW_TEXT_DIR = "./data/texts/ru/train"             # Raw .txt files for each doc
OUTPUT_DIR = "./binder_data/ru/train"              # Output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load TSV
df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str)

# Ensure required columns
required_cols = {"document_id", "text", "entity_type", "spans"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# Group annotations by document
annotations_by_doc = defaultdict(list)

for idx, row in df.iterrows():
    try:
        doc_id = row["document_id"]
        mention = row["text"]
        label = row["entity_type"]
        span_str = str(row["spans"]).strip()

        for span in span_str.split(","):
            if "-" in span:
                start, end = map(int, span.strip().split("-"))
                annotations_by_doc[doc_id].append({
                    "start": start,
                    "end": end,
                    "mention": mention,
                    "label": label
                })
            else:
                print(f"⚠️ Bad span format in row {idx}: {span}")
    except Exception as e:
        print(f"❌ Error parsing row {idx}: {row.to_dict()}")
        print(f"   → {e}")

def tokenize_with_char_offsets(text):
    """Simple whitespace tokenizer with char-level word offsets."""
    words = []
    offsets = []
    for match in re.finditer(r'\S+', text):
        start, end = match.start(), match.end()
        words.append(text[start:end])
        offsets.append((start, end))
    return words, offsets

converted = 0

for doc_id, entities in annotations_by_doc.items():
    raw_path = os.path.join(RAW_TEXT_DIR, f"{doc_id}.txt")
    if not os.path.exists(raw_path):
        print(f"⚠️ Missing text for {doc_id}")
        continue

    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    words, offset_mapping = tokenize_with_char_offsets(text)
    word_start_chars, word_end_chars = zip(*offset_mapping) if offset_mapping else ([], [])

    # Extract entity fields
    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    for ent in entities:
        entity_types.append(ent["label"])
        entity_start_chars.append(ent["start"])
        entity_end_chars.append(ent["end"])

    out_data = {
        "id": doc_id,
        "text": text,
        "entity_types": entity_types,
        "entity_start_chars": entity_start_chars,
        "entity_end_chars": entity_end_chars,
        "word_start_chars": list(word_start_chars),
        "word_end_chars": list(word_end_chars)
    }

    with open(os.path.join(OUTPUT_DIR, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    converted += 1

print(f"✅ Done! Converted {converted} documents to {OUTPUT_DIR}")
