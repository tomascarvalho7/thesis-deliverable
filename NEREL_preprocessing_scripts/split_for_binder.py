import os
import shutil
import random

SOURCE_ROOT = "./binder_data_augmented_synonym_pdoc100_pword50"
DEST_ROOT = "./binder/data_augmented_synonym_pdoc100_pword50"
languages = ["en", "ru"]
splits = ["train", "dev"]

# Collect all files for each language
lang_data = {}

for lang in languages:
    lang_data[lang] = {"train": [], "dev": []}
    for split in splits:
        src_dir = os.path.join(SOURCE_ROOT, lang, split)
        if os.path.exists(src_dir):
            lang_data[lang][split] = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".json")]
        else:
            print(f"⚠️ Missing {split} directory for {lang} at {src_dir}")

# Split dev into dev/test (50/50)
bilingual = {"train": [], "dev": [], "test": []}

for lang in languages:
    print(f"📦 Processing {lang}")

    # Shuffle dev to create test
    dev_files = lang_data[lang]["dev"]
    random.seed(42)
    random.shuffle(dev_files)
    half = len(dev_files) // 2
    dev_split = dev_files[:half]
    test_split = dev_files[half:]

    lang_splits = {
        "train": lang_data[lang]["train"],
        "dev": dev_split,
        "test": test_split,
    }

    for split, files in lang_splits.items():
        # Save per-language
        out_dir = os.path.join(DEST_ROOT, lang, "bioNNEL_dataset", split)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            doc_id = os.path.basename(f).replace(f"_{lang}.json", ".json")
            shutil.copy(f, os.path.join(out_dir, doc_id))

        # Add to bilingual
        bilingual[split].extend(files)

# Save bilingual dataset
print("🌍 Creating bilingual dataset")
for split, files in bilingual.items():
    out_dir = os.path.join(DEST_ROOT, "bilingual", "bioNNEL_dataset", split)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        lang = "en" if "_en.json" in f else "ru"
        doc_id = os.path.basename(f).replace(f"_{lang}.json", ".json")
        shutil.copy(f, os.path.join(out_dir, doc_id))

print("✅ All datasets ready: en, ru, bilingual (train/dev/test)")
