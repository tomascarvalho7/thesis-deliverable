import os
import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# === Configuration ===
ROOT_INPUT_DIR = "./data/parquet"
RAW_TEXT_DIR = "./data/texts"
AUGMENTED_TEXT_DIR = "./data/texts_augmented"
AUGMENT_PERCENT = 0.1               # % of words to augment in each file
NUM_AUG_PER_DOC = 1                 # Number of augmentations per doc
AUGMENTATION_TYPE = "wordembs"      # Word embeddings type

# === Initialize Word Embeddings Augmenter ===
# Download GloVe from https://nlp.stanford.edu/projects/glove/
# e.g., use glove.6B.100d.txt and set its path below
GLOVE_PATH = "./embeddings/glove.6B.100d.txt"

aug = naw.WordEmbsAug(
    model_type='glove',
    model_path=GLOVE_PATH,
    action='substitute',
    aug_p=AUGMENT_PERCENT
)

nltk.download('averaged_perceptron_tagger_eng')


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
        start = text.find(ent_text, offset)
        if start == -1:
            continue
        end = start + len(ent_text)
        placeholder = f"ENT{i}"
        entity_map[placeholder] = (ent_text, start)
        masked_text = masked_text[:start] + placeholder + masked_text[end:]
        offset = start + len(placeholder)
    return masked_text, entity_map


def unmask_entities(aug_text, entity_map):
    for placeholder, (original, _) in entity_map.items():
        aug_text = aug_text.replace(placeholder, original)
    return aug_text


def augment_text_and_spans(text, span_rows):
    masked_text, entity_map = mask_entities(text, span_rows)

    sentences = sent_tokenize(masked_text)
    augmented_sentences = []

    for sent in sentences:
        try:
            aug_sent = aug.augment(sent)
            if isinstance(aug_sent, list):
                aug_sent = aug_sent[0]
            augmented_sentences.append(aug_sent)
        except Exception as e:
            print(f"⚠️ Error augmenting sentence: '{sent}' → {e}")
            augmented_sentences.append(sent)  # fallback to original

    augmented_masked = " ".join(augmented_sentences)

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

    return augmented_text, new_spans

def augment_file(input_path, lang_dir):
    df = pd.read_parquet(input_path)

    if "document_id" not in df or "text" not in df:
        print(f"⚠️ Skipping {input_path}: missing required columns.")
        return

    all_doc_ids = df["document_id"].unique()
    sampled_doc_ids = set(all_doc_ids)  # process all documents

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

        for i in range(NUM_AUG_PER_DOC):
            aug_text, aug_span_rows = augment_text_and_spans(original_text, span_rows)
            if not aug_span_rows:
                continue

            new_doc_id = f"{doc_id}_aug_{AUGMENTATION_TYPE}{i+1}"

            # Save augmented raw text
            output_txt_path = os.path.join(AUGMENTED_TEXT_DIR, lang_dir, split, f"{new_doc_id}.txt")
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

    # Save combined data
    percent_suffix = f"_augmented_{AUGMENTATION_TYPE}_p{int(AUGMENT_PERCENT * 100)}"
    output_path = input_path.replace(".parquet", f"{percent_suffix}.parquet")

    combined_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    combined_df.to_parquet(output_path, index=False)
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
