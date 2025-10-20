import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load the vocabulary from the vocab.parquet file (optional, currently unused)
vocab_df = pd.read_parquet("./data/vocabular/bionnel_vocab_bilingual.parquet")

# Function to process each sentence and create token-level labels
def process_doc(doc_text, spans_and_types):
    char_labels = ['O'] * len(doc_text)

    # Apply entity labels based on the character spans
    for (start, end), ent_type in spans_and_types:
        if start >= len(char_labels) or end > len(char_labels):
            continue
        for i in range(start, end):
            char_labels[i] = f'I-{ent_type}'
        char_labels[start] = f'B-{ent_type}'

    # Tokenize the document text
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

# Function to convert the parquet file to BIO format sentence by sentence
def convert_parquet_to_bio(parquet_path, raw_texts_dir, output_path):
    df = pd.read_parquet(parquet_path)

    # Group data by document ID
    grouped = df.groupby("document_id")

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, group in grouped:
            raw_text_path = None
            for subdir in ['dev', 'train']:
                possible_path = os.path.join(raw_texts_dir, subdir, doc_id + ".txt")
                if os.path.exists(possible_path):
                    raw_text_path = possible_path
                    break
                
            if raw_text_path is None:
                print(f"⚠️ Raw text file missing for {doc_id}")
                continue

            with open(raw_text_path, "r", encoding="utf-8") as text_file:
                doc_text = text_file.read()

            spans_and_types = []
            for _, row in group.iterrows():
                try:
                    span_strings = row['spans'].split(',')
                    for span_str in span_strings:
                        try:
                            start, end = map(int, span_str.strip().split('-'))
                            spans_and_types.append(((start, end), row['entity_type']))
                        except Exception as e:
                            print(f"⚠️ Skipping span '{span_str}' in {row['document_id']}: {e}")
                except:
                    continue

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

                # Adjust spans to sentence-local offsets
                local_spans_and_types = []
                for (global_start, global_end), ent_type in spans_and_types:
                    if global_start >= sent_start and global_end <= sent_end:
                        local_start = global_start - sent_start
                        local_end = global_end - sent_start
                        local_spans_and_types.append(((local_start, local_end), ent_type))

                # Skip empty sentences
                if not sentence_text.strip():
                    continue

                token_label_pairs = process_doc(sentence_text, local_spans_and_types)
                for token, label in token_label_pairs:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")

# Function to process all parquet files in the specified directory
def process_all_parquet_files(root_dir="./data/parquet", raw_texts_base_dir="./data/texts", output_base="./data/bio_format_sentence"):
    raw_text_dirs = ['en/dev', 'en/train', 'ru/dev', 'ru/train']
    
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".parquet"):
                input_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(input_path, root_dir)
                output_path = os.path.join(output_base, relative_path.replace(".parquet", ".tsv"))

                lang = 'en' if 'en' in dirpath else 'ru'
                raw_texts_dir = os.path.join(raw_texts_base_dir, lang)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"📄 Converting: {input_path} → {output_path}")
                convert_parquet_to_bio(input_path, raw_texts_dir, output_path)

if __name__ == "__main__":
    process_all_parquet_files()
