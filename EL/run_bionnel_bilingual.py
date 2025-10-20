import os
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f" CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f" Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print(" No GPU detected. Running on CPU.")

# Function to load the raw BIO data
def load_bio_data(data_dir):
    def read_bio_file(file_path):
        sentences = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            sentence = []
            label = []
            for line in f:
                if line.strip():  # not empty line
                    token, entity = line.strip().split("\t")
                    sentence.append(token)
                    label.append(entity)
                else:  # empty line indicates end of sentence
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []
        return sentences, labels

    dataset_dict = {}
    for lang in ['en', 'ru']:
        train_file = os.path.join(data_dir, lang, f'bionnel_{lang}_train.tsv')
        dev_file = os.path.join(data_dir, lang, f'bionnel_{lang}_dev.tsv')

        train_sentences, train_labels = read_bio_file(train_file)
        val_sentences, val_labels = read_bio_file(dev_file)

        train_dataset = Dataset.from_dict({"tokens": train_sentences, "labels": train_labels})
        val_dataset = Dataset.from_dict({"tokens": val_sentences, "labels": val_labels})

        dataset_dict[lang] = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    return dataset_dict

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Define label mapping
entity_types = [
    "ACTIVITY", "ADMINISTRATION_ROUTE", "ANATOMY", "CHEM", "DEVICE", "DISO", "FINDING", "FOOD", "GENE",
    "INJURY_POISONING", "HEALTH_CARE_ACTIVITY", "LABPROC", "LIVB", "MEDPROC", "MENTALPROC", "PHYS",
    "SCIPROC", "AGE", "CITY", "COUNTRY", "DATE", "DISTRICT", "EVENT", "FAMILY", "FACILITY",
    "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PRODUCT",
    "PROFESSION", "STATE_OR_PROVINCE", "TIME"
]
label_list = ['O'] + [f'B-{ent}' for ent in entity_types] + [f'I-{ent}' for ent in entity_types]
label_map = {label: i for i, label in enumerate(label_list)}

print("Label map:", label_map)

# Global variable to track tokens during evaluation
eval_tokens_lookup = {}

def preprocess_data(dataset_dict, label_map):
    def tokenize_and_align_labels(example, example_index=None, lang=None):
        tokenized_inputs = tokenizer(
            example["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            is_split_into_words=True
        )
        word_ids = tokenized_inputs.word_ids()
        new_labels = []
        unknown_labels = set()

        for idx in word_ids:
            if idx is None:
                new_labels.append(-100)
            else:
                label_str = example["labels"][idx]
                label_id = label_map.get(label_str, -100)
                if label_id == -100:
                    unknown_labels.add(label_str)
                new_labels.append(label_id)

        if unknown_labels:
            print("⚠️ Unknown labels found during preprocessing:", unknown_labels)

        tokenized_inputs["labels"] = new_labels

        # Store token info for evaluation (optional for debugging)
        if lang == "en" and example_index is not None:
            eval_tokens_lookup[example_index] = example["tokens"]

        return tokenized_inputs

    for lang, dataset in dataset_dict.items():
        for split in ["train", "validation", "test"]:
            if split in dataset:
                dataset[split] = dataset[split].map(
                    lambda example, idx: tokenize_and_align_labels(example, idx, lang),
                    with_indices=True,
                    batched=False
                )

    return dataset_dict

# Load the model
model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", 
    num_labels=len(label_list)
)
model.config.id2label = {i: label for i, label in enumerate(label_list)}
model.config.label2id = label_map

# Define compute metrics function with debug logs
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.tensor(predictions)
    predictions = torch.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    print("\n🔍 Mismatched predictions (label ≠ prediction):")
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        current_true = []
        current_pred = []

        tokens = eval_tokens_lookup.get(i, [])

        for j, (p_val, l_val) in enumerate(zip(pred, label)):
            if l_val != -100:
                true_label = model.config.id2label[l_val.item()]
                pred_label = model.config.id2label[p_val.item()]
                current_true.append(true_label)
                current_pred.append(pred_label)

                if true_label != pred_label:
                    token_str = tokens[j] if j < len(tokens) else "<?>"
                    print(f"[Sample {i} | Token {j}] Token: '{token_str}'  TRUE: {true_label}  ≠  PRED: {pred_label}")

        true_labels.append(current_true)
        true_predictions.append(current_pred)

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    warmup_steps=500,
)

# Load the dataset
data_dir = "./data/bio_format_sentence"
dataset = load_bio_data(data_dir)

# Split validation into validation + test
for lang in dataset:
    val_split = dataset[lang]["validation"].train_test_split(test_size=0.5, seed=42)
    dataset[lang]["validation"] = val_split["train"]
    dataset[lang]["test"] = val_split["test"]

# Merge en + ru into a bilingual dataset
bilingual_dataset = DatasetDict({
    "train": concatenate_datasets([dataset["en"]["train"], dataset["ru"]["train"]]),
    "validation": concatenate_datasets([dataset["en"]["validation"], dataset["ru"]["validation"]]),
    "test": concatenate_datasets([dataset["en"]["test"], dataset["ru"]["test"]])
})

# Preprocess the merged bilingual dataset
processed_bilingual_data = preprocess_data({"bilingual": bilingual_dataset}, label_map)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_bilingual_data["bilingual"]["train"],
    eval_dataset=processed_bilingual_data["bilingual"]["validation"],
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate(processed_bilingual_data["bilingual"]["test"])

# Save model and tokenizer
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")
