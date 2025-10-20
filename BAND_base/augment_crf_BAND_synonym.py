import os
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

nltk.download("punkt")

# === Config ===
DATA_DIR = "./data/bandcrfstratified"
OUTPUT_DIR = "./BAND_crf_stratified_aug_syn"
NUM_AUG_PER_SAMPLE = 1
AUG_P = 0.5  # 50% of context words
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Synonym Augmenter
AUGMENTER = naw.SynonymAug(aug_src='wordnet', aug_p=AUG_P)

def read_conll_file(path):
    """Read CoNLL-style file into list of sentences: each = (tokens, tags)."""
    sentences, tokens, tags = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # sentence boundary
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            token, tag = line.split("\t")
            tokens.append(token)
            tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences

def write_conll_file(sentences, path):
    """Write sentences back to CoNLL format."""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, tags in sentences:
            for t, l in zip(tokens, tags):
                f.write(f"{t}\t{l}\n")
            f.write("\n")

def augment_sentence(tokens, tags):
    """Augment a sentence while preserving entity spans."""
    entities = []
    masked_tokens = []
    entity_id, i = 0, 0

    # === Step 1: Mask entities ===
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):  # entity start
            entity_tokens = [tokens[i]]
            entity_tags = [tags[i]]
            j = i + 1
            while j < len(tokens) and tags[j].startswith("I-"):
                entity_tokens.append(tokens[j])
                entity_tags.append(tags[j])
                j += 1
            mask_token = f"__ENT{entity_id}__"
            entities.append((mask_token, entity_tokens, entity_tags))
            masked_tokens.append(mask_token)
            i = j
            entity_id += 1
        else:
            masked_tokens.append(tokens[i])
            i += 1

    # === Step 2: Apply augmentation to masked sentence ===
    sentence = " ".join(masked_tokens)
    augmented_sentences = AUGMENTER.augment(sentence, n=NUM_AUG_PER_SAMPLE)
    if isinstance(augmented_sentences, str):
        augmented_sentences = [augmented_sentences]

    augmented_out = []
    for aug_text in augmented_sentences:
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
        augmented_out.append((new_tokens, new_tags))

    return augmented_out

def augment_file(input_path, output_path):
    sentences = read_conll_file(input_path)
    augmented_sentences = []
    for tokens, tags in tqdm(sentences, desc=f"Augmenting {os.path.basename(input_path)}"):
        augmented_sentences.append((tokens, tags))  # keep original
        augmented_sentences.extend(augment_sentence(tokens, tags))  # add augmented
    write_conll_file(augmented_sentences, output_path)
    print(f"✅ Saved {len(augmented_sentences)} sentences to {output_path}")

# === Process train/dev/test ===
for split in ["train.txt", "dev.txt", "test.txt"]:
    in_path = os.path.join(DATA_DIR, split)
    out_path = os.path.join(OUTPUT_DIR, split)
    augment_file(in_path, out_path)
