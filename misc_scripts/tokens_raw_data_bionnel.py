import os
import csv
from pathlib import Path
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
from collections import defaultdict

nltk.download("punkt")
nltk.download("punkt_tab")


# Load multilingual tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Output directory for CSVs
output_dir = Path("./data/tokenized_count")
output_dir.mkdir(parents=True, exist_ok=True)

def tokenize_and_save(base_path="texts"):
    stats = defaultdict(lambda: {"total_tokens": 0, "total_sentences": 0})

    for lang in ["en", "ru"]:
        for split in ["train", "dev"]:
            dir_path = os.path.join(base_path, lang, split)
            if not os.path.exists(dir_path):
                continue

            print(f"📂 Processing: {dir_path}")

            for fname in os.listdir(dir_path):
                if not fname.endswith(".txt"):
                    continue

                file_path = os.path.join(dir_path, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Sentence segmentation
                sentences = sent_tokenize(content)

                # Token counting per sentence
                sentence_token_counts = []
                for sent in sentences:
                    tokens = tokenizer.tokenize(sent)
                    sentence_token_counts.append(len(tokens))

                # Save per-document token counts
                out_file = output_dir / f"{lang}_{split}_{fname.replace('.txt', '')}.csv"
                with open(out_file, "w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["sentence_index", "token_count"])
                    for idx, count in enumerate(sentence_token_counts):
                        writer.writerow([idx, count])

                # Update global stats
                stats[f"{lang}_{split}"]["total_tokens"] += sum(sentence_token_counts)
                stats[f"{lang}_{split}"]["total_sentences"] += len(sentence_token_counts)

    # Print summary
    print("\n📊 Average Tokens per Sentence:")
    for key, value in stats.items():
        total_tokens = value["total_tokens"]
        total_sentences = value["total_sentences"]
        avg = round(total_tokens / total_sentences, 2) if total_sentences else 0.0
        print(f"{key}: {avg} tokens/sentence")

# Run it
tokenize_and_save("./data/texts")
