import os
from datasets import load_from_disk
import json

# === Config ===
ROOT_DIR = "./BAND_merged_aug_syn"      # The root directory containing arrow datasets
EXPORT_FORMAT = "jsonl"                 # "jsonl" or "csv"
OUTPUT_SUFFIX = "_readable"             # Suffix for exported readable files

def convert_arrow_to_readable(dataset_path):
    try:
        ds = load_from_disk(dataset_path)
        output_path = f"{dataset_path}{OUTPUT_SUFFIX}.{EXPORT_FORMAT}"

        if EXPORT_FORMAT == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        elif EXPORT_FORMAT == "csv":
            ds.to_csv(output_path)
        else:
            print(f"❌ Unknown format: {EXPORT_FORMAT}")
            return

        print(f"✅ Converted: {dataset_path} → {output_path}")

    except Exception as e:
        print(f"⚠️ Failed to convert {dataset_path}: {e}")

def walk_and_convert(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if "data-00000-of-00001.arrow" in files and "state.json" in files:
            convert_arrow_to_readable(root)

if __name__ == "__main__":
    walk_and_convert(ROOT_DIR)
