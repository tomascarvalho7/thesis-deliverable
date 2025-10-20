import os
import json
import argparse

def merge_json_files(input_dir, output_file):
    merged_data = []

    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            path = os.path.join(input_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    merged_data.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"❌ Error reading {path}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged {len(merged_data)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder with JSON files")
    parser.add_argument("output_file", help="Path to save merged JSON")

    args = parser.parse_args()
    merge_json_files(args.input_dir, args.output_file)

#python merge_json_binder.py ./binder/data/en/bioNNEL_dataset/train ./binder/data/en/train.json
