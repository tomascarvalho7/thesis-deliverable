import os
import random
import pandas as pd
import nlpaug.augmenter.word as naw
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import nltk
import difflib
import torch

# === Configuration ===
ROOT_INPUT_DIR = "./data/parquet"
RAW_TEXT_DIR = "./data/texts"
AUGMENT_PERCENT = 0.5              # Not used by backtranslation but kept for consistency
NUM_AUG_PER_DOC = 1                # Number of augmentations per doc
AUGMENTATION_TYPE = "backtranslation"

# === Output Directories (reflecting full doc augmentation) ===
AUG_STR = f"pdoc100_pword{int(AUGMENT_PERCENT * 100)}"
OUTPUT_TEXT_DIR = f"./data/texts_augmented_{AUGMENTATION_TYPE}_{AUG_STR}"
OUTPUT_PARQUET_DIR = f"./data/parquet_{AUGMENTATION_TYPE}_{AUG_STR}"
OUTPUT_CHANGES_DIR = f"./data/augmented_words_{AUGMENTATION_TYPE}_{AUG_STR}"

# === Initialize augmenter ===
aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-de',
    to_model_name='Helsinki-NLP/opus-mt-de-en',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=1
)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def extract_word_changes(original, augmented):
    original_tokens = word_tokenize(original)
    augmented_tokens = word_tokenize(augmented)

    changes = []
    matcher = difflib.SequenceMatcher(None, original_tokens, augmented_tokens)

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            for o, a in zip(original_tokens[i1:i2], augmented_tokens[j1:j2]):
                changes.append((o, a))
        elif opcode == 'insert':
            for a in augmented_tokens[j1:j2]:
                changes.append(("<inserted>", a))
        elif opcode == 'delete':
            for o in original_tokens[i1:i2]:
                changes.append((o, "<deleted>"))
    return changes


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
            continue
        end = start + len(ent_text)

        masked_text = masked_text[:start] + placeholder + masked_text[end:]
        entity_map[placeholder] = (ent_text, start)
        offset = start + len(placeholder)

    return masked_text, entity_map


def unmask_entities(aug_text, entity_map):
    for placeholder, (original, _) in sorted(entity_map.items(), key=lambda x: len(x[0]), reverse=True):
        aug_text = aug_text.replace(placeholder, original)
    return aug_text


def augment_text_and_spans(text, span_rows):
    masked_text, entity_map = mask_entities(text, span_rows)
    sentences = sent_tokenize(masked_text)
    augmented_sentences = []

    for sent in sentences:
        try:
            augmented = aug.augment(sent)
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            augmented = sent

        if isinstance(augmented, list):
            augmented = augmented[0]
        augmented_sentences.append(augmented)

    augmented_masked = " ".join(augmented_sentences)

    if isinstance(augmented_masked, list):
        augmented_masked = augmented_masked[0]

    augmented_text = unmask_entities(augmented_masked, entity_map)

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

    word_changes = extract_word_changes(masked_text, augmented_masked)
    return augmented_text, new_spans, word_changes


def augment_file(input_path, lang_dir):
    df = pd.read_parquet(input_path)

    if "document_id" not in df or "text" not in df:
        print(f"⚠️ Skipping {input_path}: missing required columns.")
        return

    all_doc_ids = df["document_id"].unique()
    sampled_doc_ids = set(all_doc_ids)  # AUGMENT ALL DOCS

    print(f"🔁 Augmenting {len(sampled_doc_ids)} of {len(all_doc_ids)} docs in {input_path}")
    augmented_rows = []
    grouped = df.groupby("document_id")

    for doc_id, group in tqdm(grouped, desc=f"→ {os.path.basename(input_path)}"):
        if doc_id not in sampled_doc_ids:
            continue

        split, text_path = get_lang_and_split(doc_id, lang_dir)
        if text_path is None:
            print(f"⚠️ Raw text file not found for {doc_id}")
            continue

        with open(text_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        span_rows = group.to_dict(orient='records')

        base_txt_path = os.path.join(OUTPUT_TEXT_DIR, lang_dir, split, f"{doc_id}_base.txt")
        os.makedirs(os.path.dirname(base_txt_path), exist_ok=True)
        if not os.path.exists(base_txt_path):  # avoid overwriting if already saved
            with open(base_txt_path, 'w', encoding='utf-8') as base_f:
                base_f.write(original_text)

        for i in range(NUM_AUG_PER_DOC):
            aug_text, aug_span_rows, word_changes = augment_text_and_spans(original_text, span_rows)
            if not aug_span_rows:
                continue

            new_doc_id = f"{doc_id}_aug_{AUGMENTATION_TYPE}_pdoc100_pword{int(AUGMENT_PERCENT * 100)}_{i+1}"

            # Save augmented raw text
            output_txt_path = os.path.join(OUTPUT_TEXT_DIR, lang_dir, split, f"{new_doc_id}.txt")
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            with open(output_txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write(aug_text)

            # Save word changes
            changes_output_path = os.path.join(OUTPUT_CHANGES_DIR, lang_dir, split, f"{new_doc_id}.txt")
            os.makedirs(os.path.dirname(changes_output_path), exist_ok=True)
            with open(changes_output_path, 'w', encoding='utf-8') as change_file:
                if word_changes:
                    for orig, aug in word_changes:
                        change_file.write(f"{orig} -> {aug}\n")
                else:
                    change_file.write("No changes detected\n")

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

    percent_suffix = f"_augmented_{AUGMENTATION_TYPE}_pdoc100_pword{int(AUGMENT_PERCENT * 100)}"
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
            print(f"📁 Skipping missing folder: {lang_path}")
            continue

        print(f"\n🌍 Processing language folder: {lang_path}")
        for file in os.listdir(lang_path):
            if file.endswith(".parquet") and "_augmented" not in file:
                full_path = os.path.join(lang_path, file)
                augment_file(full_path, lang_dir)


if __name__ == "__main__":
    main()
