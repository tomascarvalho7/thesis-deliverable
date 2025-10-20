import os
import random
import pandas as pd
import nlpaug.augmenter.word as naw
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import nltk
import difflib

# === Configuration ===
ROOT_INPUT_DIR = "./data/parquet"
RAW_TEXT_DIR = "./data/texts"
AUGMENT_PERCENT = 0.5           # Percentage of words to augment in each sentence
NUM_AUG_PER_DOC = 1             # Number of augmentations per doc
AUGMENTATION_TYPE = "synonym"  # or "backtranslation", "insertion", etc.
AUGMENT_PERCENT_DATASET = 0.1  # (ignored in debug mode)

DEBUG_DOC_ID = "25726786_en" \
""  # 👈 Hardcoded for debugging

# === Output Directories ===
AUG_STR = f"pdoc{int(AUGMENT_PERCENT_DATASET * 100)}_pword{int(AUGMENT_PERCENT * 100)}"
OUTPUT_TEXT_DIR = f"./data/texts_augmented_{AUGMENTATION_TYPE}_{AUG_STR}"
OUTPUT_PARQUET_DIR = f"./data/parquet_{AUGMENTATION_TYPE}_{AUG_STR}"

# === Initialize augmenter ===
aug = naw.SynonymAug(aug_src='wordnet', aug_p=AUGMENT_PERCENT, verbose=1)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def get_lang_and_split(doc_id, lang_dir):
    for split in ['train', 'dev']:
        path = os.path.join(RAW_TEXT_DIR, lang_dir, split, f"{doc_id}.txt")
        if os.path.exists(path):
            return split, path
    return None, None


def mask_entities(text, span_rows):
    masked_text = text
    entity_map = {}
    offset = 0

    for i, row in enumerate(span_rows):
        ent_text = row['text']
        placeholder = f"ENT{i}"

        start = masked_text.find(ent_text, offset)
        if start == -1:
            # entity not found, skip
            continue
        end = start + len(ent_text)

        masked_text = masked_text[:start] + placeholder + masked_text[end:]

        entity_map[placeholder] = (ent_text, start)

        offset = start + len(placeholder)

    return masked_text, entity_map


def unmask_entities(aug_text, entity_map):
    # Sort placeholders by length descending to avoid partial replacements if needed
    for placeholder, (original, _) in sorted(entity_map.items(), key=lambda x: len(x[0]), reverse=True):
        aug_text = aug_text.replace(placeholder, original)
    return aug_text


def augment_text_and_spans(text, span_rows):
    print("Original text:", text)
    masked_text, entity_map = mask_entities(text, span_rows)
    print("Masked text for augmentation:", masked_text)
    print("Entity map:", entity_map)
    sentences = sent_tokenize(masked_text)
    print("Sentences for augmentation:", sentences)
    augmented_sentences = []

    for sent in sentences:
        augmented = aug.augment(sent)
        if isinstance(augmented, list):
            augmented = augmented[0]
        augmented_sentences.append(augmented)

    print("Augmented sentences:", augmented_sentences)

    augmented_masked = " ".join(augmented_sentences)

    if isinstance(augmented_masked, list):
        augmented_masked = augmented_masked[0]

    augmented_text = unmask_entities(augmented_masked, entity_map)

    # Recalculate spans using original entity text
    new_spans = []
    for row in span_rows:
        original_text = row['text']
        ent_type = row['entity_type']
        new_start = augmented_text.find(original_text)
        if new_start == -1:
            continue
        new_end = new_start + len(original_text)
        new_spans.append({
            'text': original_text,
            'entity_type': ent_type,
            'spans': f"{new_start}-{new_end}"
        })

    return augmented_text, new_spans


def augment_file(input_path, lang_dir):
    df = pd.read_parquet(input_path)

    if "document_id" not in df or "text" not in df:
        print(f"⚠️ Skipping {input_path}: missing required columns.")
        return

    grouped = df.groupby("document_id")
    augmented_rows = []

    for doc_id, group in grouped:
        if doc_id != DEBUG_DOC_ID:
            continue

        split, text_path = get_lang_and_split(doc_id, lang_dir)
        if text_path is None:
            print(f"⚠️ Raw text file not found for {doc_id}")
            return

        with open(text_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        span_rows = group.to_dict(orient='records')

        for i in range(NUM_AUG_PER_DOC):
            aug_text, aug_span_rows = augment_text_and_spans(original_text, span_rows)
            if not aug_span_rows:
                continue

            new_doc_id = f"{doc_id}_aug_{AUGMENTATION_TYPE}_pdoc{int(AUGMENT_PERCENT_DATASET * 100)}_pword{int(AUGMENT_PERCENT * 100)}_{i+1}"

            output_txt_path = os.path.join(OUTPUT_TEXT_DIR, lang_dir, split, f"{new_doc_id}.txt")
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            with open(output_txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write(aug_text)

            for span in aug_span_rows:
                augmented_rows.append({
                    'document_id': new_doc_id,
                    'text': span['text'],
                    'entity_type': span['entity_type'],
                    'spans': span['spans'],
                    'UMLS_CUI': None
                })

    if not augmented_rows:
        print(f"⚠️ No augmentations produced for {input_path}")
        return

    percent_suffix = f"_augmented_{AUGMENTATION_TYPE}_pdoc{int(AUGMENT_PERCENT_DATASET * 100)}_pword{int(AUGMENT_PERCENT * 100)}"
    base_filename = os.path.basename(input_path).replace(".parquet", f"{percent_suffix}.parquet")

    os.makedirs(os.path.join(OUTPUT_PARQUET_DIR, lang_dir), exist_ok=True)
    output_path = os.path.join(OUTPUT_PARQUET_DIR, lang_dir, base_filename)

    combined_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    combined_df.to_parquet(output_path, index=False)

    print(f"📊 Original docs: {len(df)} | Augmented docs: {len(augmented_rows)} | Total saved: {len(combined_df)}")
    print(f"✅ Saved: {output_path}")


def main():
    for lang_dir in ['en', 'ru', 'bilingual']:
        lang_path = os.path.join(ROOT_INPUT_DIR, lang_dir)
        if not os.path.isdir(lang_path):
            continue

        print(f"\n🌍 Processing language folder: {lang_path}")
        for file in os.listdir(lang_path):
            if file.endswith(".parquet") and DEBUG_DOC_ID.split("_")[1] in file:
                full_path = os.path.join(lang_path, file)
                augment_file(full_path, lang_dir)
                break  # Process only one file


if __name__ == "__main__":
    main()
