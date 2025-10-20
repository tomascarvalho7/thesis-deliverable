import pandas as pd
import os
import nlpaug.augmenter.word as naw
import torch

# --- Config ---
LANGUAGES = ['en', 'ru']
SPLITS = ['train', 'dev']
INPUT_BASE_PATH = './data/parquet'
OUTPUT_BASE_PATH = './data/augmented_backtranslation_entity_linking_full'  # new folder

# --- Helper to choose augmentation pipeline per language ---
def get_backtranslation_augmenter(lang):
    if lang == "en":
        return naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-de',
            to_model_name='Helsinki-NLP/opus-mt-de-en',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif lang == "ru":
        return naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-ru-en',
            to_model_name='Helsinki-NLP/opus-mt-en-ru',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        raise ValueError(f"Unsupported language: {lang}")


def augment_entities(df, bt_aug):
    """Augment ALL rows of the dataframe."""
    augmented_rows = []

    for idx, original_row in df.iterrows():
        original_text = original_row['text']

        try:
            augmented_text = bt_aug.augment(original_text)
        except Exception as e:
            print(f"⚠️ Skipped due to error: {e}")
            continue

        if not augmented_text or augmented_text[0].strip().lower() == original_text.strip().lower():
            continue

        new_row = original_row.copy()
        new_row['text'] = augmented_text[0]
        new_row['spans'] = ''  # Invalidate spans due to text change
        augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    return df, augmented_df


# --- Main Processing Loop ---
for lang in LANGUAGES:
    bt_aug = get_backtranslation_augmenter(lang)  # choose correct pipeline

    for split in SPLITS:
        input_path = os.path.join(INPUT_BASE_PATH, lang, f'bionnel_{lang}_{split}.parquet')
        output_folder = os.path.join(OUTPUT_BASE_PATH, lang)
        output_path = os.path.join(output_folder, f'bionnel_{lang}_{split}.parquet')

        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path} — skipping.")
            continue

        print(f"🔄 Processing {input_path}...")

        df = pd.read_parquet(input_path)
        df, augmented_df = augment_entities(df, bt_aug)

        # Append augmented data so size ≈ 2x original
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        combined_df.to_parquet(output_path, index=False)

        print(f"✅ Original: {len(df)}, Augmented: {len(augmented_df)}, Total: {len(combined_df)}")
        print(f"✅ Saved augmented data to: {output_path}")
