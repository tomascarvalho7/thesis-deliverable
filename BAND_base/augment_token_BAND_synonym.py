import os
import json
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk

nltk.download("punkt")

# === Config ===
DATA_DIR = "./data/bandtokenstratified"          # directory containing train.json, dev.json, test.json
OUTPUT_DIR = "./BAND_token_aug_syn"
NUM_AUG_PER_SAMPLE = 1
AUG_P = 0.5  # augment 50% of context words

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Synonym Augmenter ===
AUGMENTER = naw.SynonymAug(aug_src='wordnet', aug_p=AUG_P)


def augment_example(example):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    entities = []
    masked_tokens = []
    masked_tags = []
    entity_id = 0
    i = 0

    # Step 1: Mask entities
    while i < len(tokens):
        tag = ner_tags[i]
        token = tokens[i]

        is_entity_start = tag.startswith("B-")
        if is_entity_start:
            entity_tokens = [token]
            entity_tags = [tag]

            j = i + 1
            while j < len(tokens) and ner_tags[j].startswith("I-"):
                entity_tokens.append(tokens[j])
                entity_tags.append(ner_tags[j])
                j += 1

            mask_token = f"__ENT{entity_id}__"
            entities.append((mask_token, entity_tokens, entity_tags))
            masked_tokens.append(mask_token)
            masked_tags.append("O")
            i = j
            entity_id += 1
        else:
            masked_tokens.append(token)
            masked_tags.append(tag)
            i += 1

    # Step 2: Augment masked text
    sentence = " ".join(masked_tokens)
    augmented_sentences = AUGMENTER.augment(sentence, n=NUM_AUG_PER_SAMPLE)
    if isinstance(augmented_sentences, str):
        augmented_sentences = [augmented_sentences]

    augmented_examples = []

    # Step 3: Replace placeholders with original entities
    for aug_text in augmented_sentences:
        aug_tokens = word_tokenize(aug_text)
        new_tokens, new_tags = [], []

        for token in aug_tokens:
            mask_entity = next((ent for ent in entities if ent[0] == token), None)
            if mask_entity:
                _, entity_tokens, entity_tags = mask_entity
                new_tokens.extend(entity_tokens)
                new_tags.extend(entity_tags)
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
    augmented_dataset = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(input_file)}"):
            example = json.loads(line.strip())
            augmented_dataset.append(example)
            if augment:  # only augment train
                augmented_dataset.extend(augment_example(example))

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in augmented_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(augmented_dataset)} examples to {output_file}")


# === Run for splits ===
for split in ["train.json", "dev.json", "test.json"]:
    in_path = os.path.join(DATA_DIR, split)
    out_path = os.path.join(OUTPUT_DIR, split)
    augment_file(in_path, out_path, augment=(split == "train.json"))
