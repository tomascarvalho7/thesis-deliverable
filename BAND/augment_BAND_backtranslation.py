import os
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
import nltk
import torch
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.word as naw

nltk.download("punkt")

# === Config ===
DATA_DIR = "./band_dataset"
OUTPUT_DIR = "./BAND_merged_aug_backtranslation"
NUM_AUG_PER_SAMPLE = 1

# === Backtranslation Augmenter ===
AUGMENTER = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def augment_example(example):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    entities = []
    masked_tokens = []
    masked_tags = []

    entity_id = 0
    i = 0
    while i < len(tokens):
        tag = ner_tags[i]
        token = tokens[i]

        is_entity_start = False
        if isinstance(tag, str):
            is_entity_start = tag.startswith("B-")
        else:
            is_entity_start = (tag != 0 and (i == 0 or ner_tags[i - 1] == 0))

        if is_entity_start:
            entity_tokens = [token]
            j = i + 1
            while j < len(tokens):
                next_tag = ner_tags[j]
                if (isinstance(next_tag, str) and next_tag.startswith("I-")) or (not isinstance(next_tag, str) and next_tag != 0):
                    entity_tokens.append(tokens[j])
                    j += 1
                else:
                    break

            mask_token = f"__ENT{entity_id}__"
            entities.append((mask_token, entity_tokens, ner_tags[i:j]))
            masked_tokens.append(mask_token)
            masked_tags.append("O" if isinstance(tag, str) else 0)
            i = j
            entity_id += 1
        else:
            masked_tokens.append(token)
            masked_tags.append(tag)
            i += 1

    sentence = " ".join(masked_tokens)
    try:
        aug_sentence = AUGMENTER.augment(sentence)
    except Exception as e:
        print(f"⚠️ Augmentation failed: {e}")
        return []

    aug_tokens = word_tokenize(aug_sentence)
    new_tokens = []
    new_tags = []

    for token in aug_tokens:
        mask_entity = next((ent for ent in entities if ent[0] == token), None)
        if mask_entity:
            _, entity_tokens, entity_tags = mask_entity
            new_tokens.extend(entity_tokens)
            new_tags.extend(entity_tags)
        else:
            new_tokens.append(token)
            new_tags.append("O" if isinstance(ner_tags[0], str) else 0)

    return [{
        "tokens": new_tokens,
        "ner_tags": new_tags
    }]


def augment_dataset(dataset_split):
    new_examples = {"tokens": [], "ner_tags": []}
    skipped = 0

    for example in tqdm(dataset_split, desc="Backtranslating examples"):
        augmented = augment_example(example)
        if not augmented:
            skipped += 1
            continue
        for aug in augmented:
            new_examples["tokens"].append(aug["tokens"])
            new_examples["ner_tags"].append(aug["ner_tags"])

    print(f"⚠️ Skipped {skipped} examples due to errors.")
    return Dataset.from_dict(new_examples)

# === Process each split ===
for split in ["train", "validation", "test"]:
    print(f"\n🚀 Processing split: {split}")

    ds_path = os.path.join(DATA_DIR, split)
    dataset = load_dataset("arrow", data_files={"data": f"{ds_path}/data-00000-of-00001.arrow"})["data"]

    augmented = augment_dataset(dataset)
    merged = concatenate_datasets([dataset, augmented])
    print(f"📊 Original: {len(dataset)}, Augmented: {len(augmented)}, Total: {len(merged)}")

    out_split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_split_dir, exist_ok=True)
    merged.save_to_disk(out_split_dir)
    print(f"✅ Saved merged split to {out_split_dir}")
