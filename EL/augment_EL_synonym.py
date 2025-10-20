import pandas as pd
import random
import os
import nlpaug.augmenter.word as naw

# --- Config ---
LANGUAGES = ['en', 'ru']
SPLITS = ['train', 'dev']
INPUT_BASE_PATH = './data/parquet'
AUGMENT_PERCENTAGE = 1
PERCENT_LABEL = int(AUGMENT_PERCENTAGE * 100)
OUTPUT_BASE_PATH = f'./data/augmented_synonym_entity_linking_{PERCENT_LABEL}'

syn_aug = naw.SynonymAug(aug_src='wordnet')

def augment_entities(df, augment_percentage):
    eligible_indices = df.index[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip().split()) <= 5)].tolist()
    num_to_augment = int(len(eligible_indices) * augment_percentage)

    # Sample and sort to preserve original index order
    augment_indices = sorted(random.sample(eligible_indices, num_to_augment))

    augmented_rows = []

    for idx in augment_indices:
        original_row = df.loc[idx]
        original_text = original_row['text']
        augmented_text = syn_aug.augment(original_text)

        if not augmented_text or augmented_text[0].strip().lower() == original_text.strip().lower():
            continue

        new_row = original_row.copy()
        new_row['text'] = augmented_text[0]
        new_row['spans'] = ''  # Clear spans since offsets are now incorrect

        augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)

    return df, augmented_df  # Return separately


# --- Main Processing Loop ---
for lang in LANGUAGES:
    for split in SPLITS:
        input_path = os.path.join(INPUT_BASE_PATH, lang, f'bionnel_{lang}_{split}.parquet')
        output_folder = os.path.join(OUTPUT_BASE_PATH, lang)
        output_path = os.path.join(output_folder, f'bionnel_{lang}_{split}.parquet')

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path} — skipping.")
            continue

        print(f"🔄 Processing {input_path}...")

        # Load, augment, save
        df = pd.read_parquet(input_path)
        df, augmented_df = augment_entities(df, AUGMENT_PERCENTAGE)

        # Append augmented data (already in aligned order)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)

        combined_df.to_parquet(output_path, index=False)

        print(f"✅ Saved augmented data to: {output_path}")
