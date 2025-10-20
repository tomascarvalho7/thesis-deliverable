import os
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

nltk.download("punkt")

# === Config ===
DATA_DIR = "./data/bandcrfstratified"
OUTPUT_DIR = "./BAND_crf_stratified_aug_bert"
NUM_AUG_PER_SAMPLE = 1
AUG_P = 0.5  # 50% of non-entity words

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === BERT-based Augmenter ===
AUGMENTER = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    aug_p=AUG_P,
    device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'
)

def read_conll_file(path):
    """Read BIO .txt file into list of sentences."""
    sentences, tokens, tags = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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
    """Write list of sentences back to BIO .txt format."""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, tags in sentences:
            for t, l in zip(tokens, tags):
                f.write(f"{t}\t{l}\n")
            f.write("\n")

def augment_sentence(tokens, tags):
    """Augment only non-entity tokens with BERT-based substitution."""
    # Mask entities with placeholder
    masked_tokens = [tok if tag == "O" else "__ENT__" for tok, tag in zip(tokens, tags)]
    sentence = " ".join(masked_tokens)

    try:
        augmented = AUGMENTER.augment(sentence, n=NUM_AUG_PER_SAMPLE)
    except Exception as e:
        print(f"⚠️ Augmentation failed: {e}")
        return []

    if isinstance(augmented, str):
        augmented = [augmented]

    augmented_out = []
    for aug_text in augmented:
        aug_tokens = word_tokenize(aug_text)

        if len(aug_tokens) != len(masked_tokens):
            continue  # Skip if token lengths mismatch

        new_tokens = []
        new_tags = []
        for orig_tok, aug_tok, tag in zip(masked_tokens, aug_tokens, tags):
            if orig_tok == "__ENT__":
                # Restore original entity tokens
                new_tokens.append(tokens[len(new_tokens)])
                new_tags.append(tag)
            else:
                new_tokens.append(aug_tok)
                new_tags.append(tag)

        if new_tokens != tokens:  # Only keep if actual change
            augmented_out.append((new_tokens, new_tags))

    return augmented_out

def augment_file(input_path, output_path, augment=True):
    sentences = read_conll_file(input_path)
    final_sentences = []
    for tokens, tags in tqdm(sentences, desc=f"Processing {os.path.basename(input_path)}"):
        final_sentences.append((tokens, tags))  # keep original
        if augment:  # only augment train
            final_sentences.extend(augment_sentence(tokens, tags))
    write_conll_file(final_sentences, output_path)
    print(f"✅ Saved {len(final_sentences)} sentences to {output_path}")

# === Process splits ===
for split in ["train.txt", "dev.txt", "test.txt"]:
    in_path = os.path.join(DATA_DIR, split)
    out_path = os.path.join(OUTPUT_DIR, split)
    augment_file(in_path, out_path, augment=(split == "train.txt"))
