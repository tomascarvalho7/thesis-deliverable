import os
import pandas as pd

# === CONFIGURATION ===
INPUT_DIR = "./data/parquet_synonym_pdoc200_pword10"  # All Parquet files are here
OUTPUT_DIR = "./data/tsv_augmented_synonym_pdoc200_pword10"  # All converted TSVs will go here

def convert_parquet_to_tsv(parquet_path, output_tsv_path):
    try:
        df = pd.read_parquet(parquet_path)
        df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"✅ Converted: {parquet_path} → {output_tsv_path}")
    except Exception as e:
        print(f"❌ Failed to convert {parquet_path}: {e}")

def main():
    for dirpath, _, filenames in os.walk(INPUT_DIR):
        for file in filenames:
            if not file.endswith(".parquet"):
                continue

            parquet_path = os.path.join(dirpath, file)
            relative_path = os.path.relpath(parquet_path, INPUT_DIR)

            # Replace .parquet with .tsv
            output_rel_path = os.path.splitext(relative_path)[0] + ".tsv"
            output_tsv_path = os.path.join(OUTPUT_DIR, output_rel_path)

            # Create necessary output subdirectories
            os.makedirs(os.path.dirname(output_tsv_path), exist_ok=True)

            # Convert
            convert_parquet_to_tsv(parquet_path, output_tsv_path)

if __name__ == "__main__":
    main()
