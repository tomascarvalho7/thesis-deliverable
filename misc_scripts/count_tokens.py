import json
import argparse
from collections import Counter
import os

def process_file(file_path, token_counter, ner_counter):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            token_counter.update(data["tokens"])
            ner_tags = [tag.split('-')[-1] for tag in data["ner_tags"]]
            ner_counter.update(ner_tags)

def main():
    parser = argparse.ArgumentParser(description="Count unique tokens and NER tags in BAND dataset files.")
    parser.add_argument('files', nargs='+', help="Paths to the JSONL files.")
    args = parser.parse_args()
    
    token_counter = Counter()
    ner_counter = Counter()
    
    for file_path in args.files:
        if os.path.isfile(file_path):
            process_file(file_path, token_counter, ner_counter)
        else:
            print(f"Warning: File '{file_path}' not found. Skipping.")
    
    print("\nUnique token count:", len(token_counter))
    print("Unique NER tag count:", len(ner_counter))
    print("\nNER Tag Frequencies:")
    for tag, count in ner_counter.items():
        print(f"{tag}: {count}")

if __name__ == "__main__":
    main()
