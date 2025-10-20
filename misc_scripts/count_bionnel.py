import os
import pandas as pd
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

# === Config ===
PARQUET_DIR = "./data/parquet_synonym_pdoc100_pword50"
TEXT_DIR = "./data/texts_augmented_synonym_pdoc100_pword50"

# Load tokenizer (multilingual for Russian & English)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def infer_split_from_filename(fname: str) -> str:
    """Infer split from parquet filename by checking keywords."""
    fname_lower = fname.lower()
    if "train" in fname_lower:
        return "train"
    elif "dev" in fname_lower:
        return "dev"
    else:
        raise ValueError(f"❌ Could not infer split from filename: {fname}")


def compute_all_stats(df, lang, split_name):
    stats = {}

    # --- Basic counts ---
    stats['Dataset'] = f"{lang.capitalize()} - {split_name.capitalize()}"
    stats['Language'] = lang.capitalize()
    stats['Split'] = split_name
    stats['Document Count'] = df['document_id'].nunique()
    stats['Annotations'] = len(df)

    # Token Count: using tokenizer on entity text
    stats['Token Count'] = df['text'].apply(lambda x: len(tokenizer.tokenize(str(x)))).sum()

    # --- Entity type frequencies ---
    entity_counts = Counter(df['entity_type'])
    stats['Entity Types'] = dict(entity_counts)

    # --- Tokens per entity ---
    token_lengths = []
    tokens_by_entity_type = defaultdict(list)
    skipped = 0

    for _, row in df.iterrows():
        entity_type = row['entity_type']
        try:
            entity_text = str(row['text']).strip()
            if not entity_text:
                raise ValueError("Empty entity text")

            tokens = tokenizer.tokenize(entity_text)
            num_tokens = len(tokens)

            token_lengths.append(num_tokens)
            tokens_by_entity_type[entity_type].append(num_tokens)

        except Exception:
            skipped += 1
            continue

    print(f"\n✅ Processed {len(token_lengths)} entities successfully, skipped {skipped}.")

    stats['Avg Tokens per Entity'] = (
        round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0.0
    )

    stats['Avg Tokens per Entity Type'] = {
        ent: round(sum(lengths) / len(lengths), 2)
        for ent, lengths in tokens_by_entity_type.items()
    }

    # --- Sentence count from raw texts ---
    text_split_dir = os.path.join(TEXT_DIR, lang, split_name)
    sentence_count = 0
    doc_count = 0

    if os.path.isdir(text_split_dir):
        for fname in os.listdir(text_split_dir):
            if fname.endswith(".txt"):
                doc_count += 1
                with open(os.path.join(text_split_dir, fname), "r", encoding="utf-8") as f:
                    raw_text = f.read()
                sentence_count += len(sent_tokenize(raw_text))

    stats['Sentence Count'] = sentence_count
    stats['Raw Text Documents'] = doc_count

    # --- Print summary ---
    print(f"\n📊 Stats for {lang.capitalize()} - {split_name.capitalize()}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def main():
    all_stats = []

    for lang in ['ru', 'en']:
        lang_dir = os.path.join(PARQUET_DIR, lang)
        if not os.path.isdir(lang_dir):
            print(f"⚠️ Missing folder: {lang_dir}")
            continue

        for fname in os.listdir(lang_dir):
            if fname.endswith(".parquet"):
                try:
                    split_name = infer_split_from_filename(fname)
                except ValueError as e:
                    print(e)
                    continue

                parquet_path = os.path.join(lang_dir, fname)
                print(f"\n🔍 Processing {parquet_path} (inferred split: {split_name})")

                df = pd.read_parquet(parquet_path)
                stats = compute_all_stats(df, lang, split_name)
                all_stats.append(stats)

    return all_stats


if __name__ == "__main__":
    stats = main()
