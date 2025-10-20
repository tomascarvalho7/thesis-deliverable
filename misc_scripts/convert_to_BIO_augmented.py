import os
import re
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def process_doc(doc_text, spans_and_types):
    char_labels = ['O'] * len(doc_text)

    for (start, end), ent_type in spans_and_types:
        if start >= len(char_labels) or end > len(char_labels):
            continue
        for i in range(start, end):
            char_labels[i] = f'I-{ent_type}'
        char_labels[start] = f'B-{ent_type}'

    tokenized = tokenizer(doc_text, return_offsets_mapping=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    offsets = tokenized['offset_mapping']

    output = []
    for token, (start, end) in zip(tokens, offsets):
        if start == end:
            continue
        label = char_labels[start] if start < len(char_labels) else 'O'

        if token.startswith("##") and output:
            prev_label = output[-1][1]
            if prev_label != 'O':
                label = f'I-{prev_label.split("-")[1]}'

        output.append((token, label))
    return output

def convert_parquet_to_bio(parquet_path, raw_texts_dir, output_path):
    df = pd.read_parquet(parquet_path)
    grouped = df.groupby("document_id")

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, group in grouped:
            doc_text = None

            raw_text_path = None

            # Search in texts/{lang}/{split} first
            for subdir in ['dev', 'train', 'test']:
                possible_path = os.path.join(raw_texts_dir, subdir, doc_id + ".txt")
                if os.path.exists(possible_path):
                    raw_text_path = possible_path
                    break

            # Fallback: search in texts_augmented/{lang}/{split}
            if raw_text_path is None:
                lang_dir = os.path.basename(raw_texts_dir)  # e.g., 'en'
                augmented_base = raw_texts_dir.replace("texts", "texts_augmented")
                for subdir in ['dev', 'train', 'test']:
                    possible_aug_path = os.path.join(augmented_base, subdir, doc_id + ".txt")
                    if os.path.exists(possible_aug_path):
                        raw_text_path = possible_aug_path
                        print(f"🔄 Fallback to augmented text: {possible_aug_path}")
                        break


            if raw_text_path:
                with open(raw_text_path, "r", encoding="utf-8") as text_file:
                    doc_text = text_file.read()
            elif "text" in group.columns:
                doc_text = group.iloc[0]["text"]
                print(f"🧪 Using in-parquet text for: {doc_id}")
            else:
                print(f"❌ No text found for: {doc_id}")
                continue

            spans_and_types = []
            for _, row in group.iterrows():
                try:
                    for span_str in row['spans'].split(','):
                        start, end = map(int, span_str.strip().split('-'))
                        spans_and_types.append(((start, end), row['entity_type']))
                except Exception as e:
                    print(f"⚠️ Skipping span '{span_str}' in {row['document_id']}: {e}")

            # Sentence splitting
            sentences = sent_tokenize(doc_text)
            sentence_spans = []
            start = 0
            for sent in sentences:
                sent_start = doc_text.find(sent, start)
                sent_end = sent_start + len(sent)
                sentence_spans.append((sent_start, sent_end))
                start = sent_end

            for sent_start, sent_end in sentence_spans:
                sentence_text = doc_text[sent_start:sent_end]
                if not sentence_text.strip():
                    continue

                local_spans = []
                for (g_start, g_end), ent_type in spans_and_types:
                    if g_start >= sent_start and g_end <= sent_end:
                        local_spans.append(((g_start - sent_start, g_end - sent_start), ent_type))

                token_label_pairs = process_doc(sentence_text, local_spans)
                for token, label in token_label_pairs:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")

def extract_aug_percent(filename):
    match = re.search(r'_augmented_p(\d+)', filename)
    return int(match.group(1)) if match else None

def detect_split_from_filename(filename):
    fname = os.path.basename(filename).lower()

    # Detect split
    split = "unknown"
    if re.search(r'(^|[_\-\.])train([_\-\.]|$)', fname):
        split = "train"
    elif re.search(r'(^|[_\-\.])dev([_\-\.]|$)', fname):
        split = "dev"
    elif re.search(r'(^|[_\-\.])test([_\-\.]|$)', fname):
        split = "test"

    # Detect augmentation type (e.g., 'synonym', 'swap', etc.) - assuming it comes after '_augmented_'
    aug_type_match = re.search(r'_augmented_([a-z]+)', fname)
    aug_type = aug_type_match.group(1) if aug_type_match else None

    # Detect augmentation percent (e.g., p10, p5)
    aug_percent_match = re.search(r'_p(\d+)', fname)
    aug_percent = aug_percent_match.group(1) if aug_percent_match else None

    # Build full split string
    if aug_type and aug_percent:
        return f"{split}_augmented_{aug_type}_p{aug_percent}"
    elif aug_type:
        return f"{split}_augmented_{aug_type}"
    elif aug_percent:
        return f"{split}_augmented_p{aug_percent}"
    else:
        return split

def process_all_parquet_files(
    root_dir="./data/parquet",
    raw_texts_base_dir="./data/texts",
    output_base="./data/bio_format_sentence"
):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.endswith(".parquet"):
                continue
            # No skipping of augmented files here!
            input_path = os.path.join(dirpath, file)
            
            rel_path = os.path.relpath(dirpath, root_dir)
            parts = rel_path.split(os.sep)
            if len(parts) < 1:
                print(f"⚠️ Skipping unexpected path format: {rel_path}")
                continue
            
            lang = parts[0]
            aug_percent = extract_aug_percent(file)
            split = detect_split_from_filename(file)
            if aug_percent:
                split = f"{split}_augmented_p{aug_percent}"
            
            raw_texts_dir = os.path.join(raw_texts_base_dir, lang)
            output_dir = os.path.join(output_base, lang)
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"bionnel_{lang}_{split}.tsv"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"📄 Converting {file} → {output_filename}")
            convert_parquet_to_bio(input_path, raw_texts_dir, output_path)


if __name__ == "__main__":
    process_all_parquet_files()
