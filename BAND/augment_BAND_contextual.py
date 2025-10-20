import os
import random
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nltk

nltk.download("punkt")

# === Config ===
DATA_DIR = "./band_dataset"
OUTPUT_DIR = "./BAND_merged_aug_bert"
NUM_AUG_PER_SAMPLE = 1
AUG_P = 0.5  # Augment 30% of the non-entity words

# === BERT-based Augmenter ===
AUGMENTER = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu',
    aug_p=AUG_P
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_example(example):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    # Only keep 'O' (non-entity) tokens for augmentation
    tokens_for_aug = [tok if tag == 0 else "[ENTITY]" for tok, tag in zip(tokens, ner_tags)]
    sentence = " ".join(tokens_for_aug)

    # Generate augmented sentence
    try:
        augmented_sentence = AUGMENTER.augment(sentence)
    except Exception:
        return []  # Fail gracefully

    if isinstance(augmented_sentence, str):
        augmented_sentence = augmented_sentence.split()

    # Restore original entities in place of placeholder
    if len(augmented_sentence) != len(tokens):
        return []  # Discard if lengths don't match

    new_tokens = []
    for orig_token, aug_token, tag in zip(tokens, augmented_sentence, ner_tags):
        if tag == 0 and aug_token != "[ENTITY]":
            new_tokens.append(aug_token)
        else:
            new_tokens.append(orig_token)

    if new_tokens == tokens:
        return []  # No actual change

    return [{
        "tokens": new_tokens,
        "ner_tags": ner_tags
    }]

def augment_dataset(dataset_split):
    new_examples = {"tokens": [], "ner_tags": []}
    skipped, augmented = 0, 0

    for example in tqdm(dataset_split, desc="Augmenting examples"):
        augmented_exs = augment_example(example)
        if not augmented_exs:
            skipped += 1
            continue
        augmented += 1
        for aug in augmented_exs:
            new_examples["tokens"].append(aug["tokens"])
            new_examples["ner_tags"].append(aug["ner_tags"])

    print(f"✅ Augmented: {augmented}, ⚠️ Skipped: {skipped}")
    return Dataset.from_dict(new_examples)

# === Process splits ===
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
