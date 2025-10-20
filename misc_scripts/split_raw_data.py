import os
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

def prepare_ner_sentences(input_dir="texts", output_dir="ner_input"):
    for lang in ["en", "ru"]:
        for split in ["train", "dev"]:
            input_path = Path(input_dir) / lang / split
            output_path = Path(output_dir) / lang / split
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"📄 Processing {input_path}")

            for fname in os.listdir(input_path):
                if not fname.endswith(".txt"):
                    continue

                # Read the raw document
                with open(input_path / fname, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split into sentences
                sentences = sent_tokenize(content)

                # Save each sentence to a new line
                with open(output_path / fname, "w", encoding="utf-8") as out_f:
                    for sentence in sentences:
                        cleaned = sentence.strip()
                        if cleaned:
                            out_f.write(cleaned + "\n")

            print(f"✅ Saved split sentences to {output_path}")

# Run the function
prepare_ner_sentences("./data/texts_augmented", "./data/bionnel_split_raw_augmented")
