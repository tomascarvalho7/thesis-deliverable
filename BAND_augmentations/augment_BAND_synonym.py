import os
from datasets import Dataset, load_dataset, concatenate_datasets
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# === Config ===
DATA_DIR = "./band_dataset"                       # Original BAND dataset root
OUTPUT_DIR = "./BAND_merged_aug_syn"              # Where merged data will be saved
NUM_AUG_PER_SAMPLE = 1
AUG_P = 0.5  # Augment 30% of the words per sentence

# === Initialize Synonym Augmenter with aug_p ===
AUGMENTER = naw.SynonymAug(aug_src='wordnet', aug_p=AUG_P)

# === Ensure output structure ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Initialize SynonymAug with stopwords ===
AUGMENTER = naw.SynonymAug(aug_src='wordnet', aug_p=AUG_P)

from nltk.tokenize import word_tokenize

def augment_example(example):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    # Step 1: Identify entities and mask them
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
            # If integer tags, 0 = O, others = entity (simple heuristic)
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

            mask_token = f"__ENT{entity_id}__"  # No spaces inside mask token
            entities.append((mask_token, entity_tokens, ner_tags[i:j]))
            masked_tokens.append(mask_token)
            masked_tags.append("O" if isinstance(tag, str) else 0)  # Mask token gets 'O'
            i = j
            entity_id += 1
        else:
            masked_tokens.append(token)
            masked_tags.append(tag)
            i += 1

    # Step 2: Augment the masked sentence
    sentence = " ".join(masked_tokens)
    augmented_sentences = AUGMENTER.augment(sentence, n=NUM_AUG_PER_SAMPLE)
    if isinstance(augmented_sentences, str):
        augmented_sentences = [augmented_sentences]

    augmented_examples = []

    # Step 3 and 4: For each augmented sentence, replace mask tokens with original entity tokens and tags
    for aug_text in augmented_sentences:
        # Proper tokenization to separate punctuation from tokens
        aug_tokens = word_tokenize(aug_text)
        new_tokens = []
        new_tags = []

        for token in aug_tokens:
            # Exact match with mask token (no spaces)
            mask_entity = next((ent for ent in entities if ent[0] == token), None)

            if mask_entity:
                # Replace mask token with original entity tokens and tags
                _, entity_tokens, entity_tags = mask_entity
                new_tokens.extend(entity_tokens)
                new_tags.extend(entity_tags)
            else:
                # New or changed tokens get 'O' label
                new_tokens.append(token)
                new_tags.append("O" if isinstance(ner_tags[0], str) else 0)

        augmented_examples.append({
            "tokens": new_tokens,
            "ner_tags": new_tags
        })

    return augmented_examples


def augment_dataset(dataset_split):
    new_examples = {"tokens": [], "ner_tags": []}
    skipped = 0

    for example in tqdm(dataset_split, desc="Augmenting examples"):
        augmented = augment_example(example)
        if not augmented:
            skipped += 1
        for aug in augmented:
            new_examples["tokens"].append(aug["tokens"])
            new_examples["ner_tags"].append(aug["ner_tags"])

    print(f"⚠️ Skipped {skipped} examples due to token length mismatch or augmentation issues.")
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
