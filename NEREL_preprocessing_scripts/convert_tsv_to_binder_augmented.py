import os
import pandas as pd
from collections import defaultdict
import json
import re

# === Config ===
LANG = "ru"
SPLIT = "train"
AUGMENT_TYPE = "synonym"
PERCENT = 0.5

# === Derived Paths ===
# BASE_TSV = f"./data/tsv/{LANG}/bionnel_{LANG}_{SPLIT}.tsv"
# BASE_TEXT_DIR = f"./data/texts/{LANG}/{SPLIT}"

# AUG_TSV = f"./data/tsv_{AUGMENT_TYPE}_p{int(PERCENT*100)}/{LANG}/bionnel_{LANG}_{SPLIT}_augmented_{AUGMENT_TYPE}_p{int(PERCENT*100)}.tsv"
# AUG_TEXT_DIR = f"./data/texts_augmented_{AUGMENT_TYPE}_p{int(PERCENT*100)}/{LANG}/{SPLIT}"

# OUTPUT_DIR = f"./binder_data_augmented_{AUGMENT_TYPE}_p{int(PERCENT*100)}/{LANG}/{SPLIT}"


BASE_TSV = f"./data/tsv/{LANG}/bionnel_{LANG}_{SPLIT}.tsv"
BASE_TEXT_DIR = f"./data/texts/{LANG}/{SPLIT}"

AUG_TSV = f"./data/tsv_augmented_synonym_pdoc100_pword50/{LANG}/bionnel_{LANG}_{SPLIT}_augmented_synonym_pdoc200_pword50.tsv"
AUG_TEXT_DIR = f"./data/texts_augmented_synonym_pdoc100_pword50/{LANG}/{SPLIT}"

OUTPUT_DIR = f"./binder_data_augmented_synonym_pdoc100_pword50/{LANG}/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotations(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    required_cols = {"document_id", "text", "entity_type", "spans"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    annotations = defaultdict(list)
    for idx, row in df.iterrows():
        doc_id = row["document_id"]
        full_mention = row["text"]
        label = row["entity_type"]
        span_str = str(row["spans"]).strip()

        try:
            for span in span_str.split(","):
                if "-" in span:
                    start, end = map(int, span.strip().split("-"))
                    annotations[doc_id].append({
                        "start": start,
                        "end": end,
                        "mention": full_mention,  # optional: text[start:end] for precision
                        "label": label
                    })
                else:
                    print(f"⚠️ Bad span format in row {idx}: {span}")
        except Exception as e:
            print(f"❌ Error parsing row {idx}: {row.to_dict()}")
            print(f"   → {e}")
    return annotations


def tokenize_with_char_offsets(text):
    words = []
    offsets = []
    for match in re.finditer(r'\S+', text):
        start, end = match.start(), match.end()
        words.append(text[start:end])
        offsets.append((start, end))
    return words, offsets

def convert_and_write_base(annotations_by_doc, raw_dir, output_dir, counter_start=0):
    raw_files = os.listdir(raw_dir)
    raw_lookup = {fname.replace(".txt", ""): fname for fname in raw_files}
    missing = 0
    converted = counter_start

    for doc_id, entities in annotations_by_doc.items():
        file_name = f"{doc_id}.txt"
        if file_name not in raw_files:
            print(f"⚠️ Missing base text for {doc_id}")
            missing += 1
            continue

        raw_path = os.path.join(raw_dir, file_name)
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read()

        converted += _write_json(doc_id, text, entities, output_dir)

    return converted, missing


def convert_and_write_augmented(annotations_by_doc, raw_dir, output_dir, counter_start=0):
    raw_files = os.listdir(raw_dir)
    raw_lookup = {fname.replace(".txt", ""): fname for fname in raw_files}
    missing = 0
    converted = counter_start

    for doc_id, entities in annotations_by_doc.items():
        # Only process if there's a matching file that is augmented
        matching_files = [fname for fname in raw_files if fname == f"{doc_id}.txt"]
        if not matching_files:
            continue  # Skip non-augmented (i.e., base) doc_ids

        raw_path = os.path.join(raw_dir, matching_files[0])
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read()

        converted += _write_json(doc_id, text, entities, output_dir)

    return converted, missing


def _write_json(doc_id, text, entities, output_dir):
    words, offset_mapping = tokenize_with_char_offsets(text)
    word_start_chars, word_end_chars = zip(*offset_mapping) if offset_mapping else ([], [])

    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    used_spans = set()

    for ent in entities:
        start, end = ent["start"], ent["end"]
        mention = ent["mention"]

        # Primary alignment check
        if text[start:end] == mention and (start, end) not in used_spans:
            entity_types.append(ent["label"])
            entity_start_chars.append(start)
            entity_end_chars.append(end)
            used_spans.add((start, end))
        else:
            # Fallback regex search
            found = False
            for match in re.finditer(re.escape(mention), text):
                s, e = match.start(), match.end()
                if (s, e) == (start, end) and (s, e) not in used_spans:
                    entity_types.append(ent["label"])
                    entity_start_chars.append(s)
                    entity_end_chars.append(e)
                    used_spans.add((s, e))
                    found = True
                    break

    out_data = {
        "id": doc_id,
        "text": text,
        "entity_types": entity_types,
        "entity_start_chars": entity_start_chars,
        "entity_end_chars": entity_end_chars,
        "word_start_chars": list(word_start_chars),
        "word_end_chars": list(word_end_chars)
    }

    out_path = os.path.join(output_dir, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    return 1


# === Load and convert both sets ===
#print("🔹 Processing base data")
#base_annotations = load_annotations(BASE_TSV)
#converted, base_missing = convert_and_write_base(base_annotations, BASE_TEXT_DIR, OUTPUT_DIR)

print("🔹 Processing augmented data")
aug_annotations = load_annotations(AUG_TSV)
converted, aug_missing = convert_and_write_augmented(aug_annotations, AUG_TEXT_DIR, OUTPUT_DIR, counter_start=0)

# === Summary ===
print(f"\n✅ Done! Converted {converted} documents: {converted - len(aug_annotations)} base + {len(aug_annotations)} augmented")
#if base_missing or aug_missing:
#    print(f"⚠️ Skipped {base_missing} original and {aug_missing} augmented docs due to missing raw text files.")
