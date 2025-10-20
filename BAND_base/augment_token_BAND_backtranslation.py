import os
import json
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import torch

nltk.download("punkt")

# === Config ===
DATA_DIR = "./data/bandtokenstratified"   # folder with train.json, dev.json, test.json
OUTPUT_DIR = "./BAND_token_aug_backtranslation"
NUM_AUG_PER_SAMPLE = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Backtranslation Augmenter ===
AUGMENTER = naw.BackTranslationAug(
    from_model_name="facebook/wmt19-en-de",
    to_model_name="facebook/wmt19-de-en",
    device="cuda" if torch.cuda.is_available() else "cpu"
)


def augment_example(example):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    entities = []
    masked_tokens = []
    i, entity_id = 0, 0

    # === Step 1: Mask entities ===
    while i < len(tokens):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            entity_tokens = [tokens[i]]
            entity_tags = [ner_tags[i]]
            j = i + 1
            while j < len(tokens) and ner_tags[j].startswith("I-"):
                entity_tokens.append(tokens[j])
                entity_tags.append(ner_tags[j])
                j += 1
            mask_token = f"__ENT{entity_id}__"
            entities.append((mask_token, entity_tokens, entity_tags))
            masked_tokens.append(mask_token)
            i = j
            entity_id += 1
        else:
            masked_tokens.append(tokens[i])
            i += 1

    # === Step 2: Backtranslate ===
    sentence = " ".join(masked_tokens)
    try:
        augmented = AUGMENTER.augment(sentence, n=NUM_AUG_PER_SAMPLE)
    except Exception as e:
        print(f"⚠️ Augmentation failed for {example['id']}: {e}")
        return []

    if isinstance(augmented, str):
        augmented = [augmented]

    augmented_examples = []
    for aug_text in augmented:
        aug_tokens = word_tokenize(aug_text)
        new_tokens, new_tags = [], []
        for token in aug_tokens:
            ent = next((e for e in entities if e[0] == token), None)
            if ent:
                _, ent_tokens, ent_tags = ent
                new_tokens.extend(ent_tokens)
                new_tags.extend(ent_tags)
            else:
                new_tokens.append(token)
                new_tags.append("O")
        augmented_examples.append({
            "id": example["id"] + "_aug",
            "tokens": new_tokens,
            "ner_tags": new_tags
        })

    return augmented_examples


def augment_file(input_file, output_file, augment=True):
    out_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(input_file)}"):
            example = json.loads(line.strip())
            out_data.append(example)  # keep original
            if augment:  # only augment train
                out_data.extend(augment_example(example))

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in out_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(out_data)} examples to {output_file}")


# === Process splits ===
for split in ["train.json", "dev.json", "test.json"]:
    in_path = os.path.join(DATA_DIR, split)
    out_path = os.path.join(OUTPUT_DIR, split)
    augment_file(in_path, out_path, augment=(split == "train.json"))
