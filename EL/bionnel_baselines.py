import logging
from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import pandas as pd
from typing import List, Optional
import numpy as np
from typing import List, Dict
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

# Function to obtain embeddings given textual names
# (either entities of vocabular concept names)
def encode_names(names, bert_encoder, tokenizer, max_length, device,
                 batch_size=256, show_progress=False):
    bert_encoder.eval()
    if isinstance(names, np.ndarray):
        names = names.tolist()

    name_encodings = tokenizer(names, padding="max_length",
                               max_length=max_length, truncation=True,
                               return_tensors="pt")
    input_ids = name_encodings["input_ids"]  # .to(device)
    attention_mask = name_encodings["attention_mask"]  # .to(device)

    embs = []

    num_samples = len(names)
    indices = range(0, num_samples, batch_size)
    if show_progress:
        indices = tqdm(indices, desc="Encoding names", unit="batch")

    with torch.no_grad():
        for i in indices:
            batch_input_ids = input_ids[i:i + batch_size].to(device)
            batch_attention_mask = attention_mask[i:i + batch_size].to(device)

            outputs = bert_encoder(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )

            batch_embeddings = outputs.last_hidden_state[:, 0]
            batch_embeddings = batch_embeddings.detach().cpu()

            embs.append(batch_embeddings)

        final_embeddings = torch.cat(embs, dim=0).detach()
        assert final_embeddings.size(0) == num_samples

    return final_embeddings

def create_repeated_tensor(range_bound, repeat_times):
    base_tensor = torch.arange(range_bound)
    repeated_tensor = torch.repeat_interleave(base_tensor, repeat_times)

    return repeated_tensor

# Calculating top-k candidates for each query entity based on
# its cosine similarity between vocabulary concept names
def get_torch_query_dict_score_matrix(query_names, tokenizer, bert_encoder, vocab_names, base_k, device,
                                      query_batch_size, max_length, show_progress, vocab_batch_size=256):
    bert_encoder.eval()
    num_queries = len(query_names)
    vocab_length = len(vocab_names)
    query_embs = encode_names(names=query_names, bert_encoder=bert_encoder,
                              tokenizer=tokenizer, max_length=max_length,
                              device=device, batch_size=query_batch_size,
                              show_progress=show_progress).unsqueeze(1).to(device).detach()
    assert num_queries == len(query_embs)

    overall_max = None
    overall_max_indices = None
    with torch.no_grad():
        for vocab_start_pos in tqdm(range(0, vocab_length, vocab_batch_size)):
            vocab_end_pos = min(vocab_start_pos + vocab_batch_size, vocab_length)
            batch_vocab_names = vocab_names[vocab_start_pos:vocab_end_pos]

            batch_vocab_embeddings = encode_names(names=batch_vocab_names, bert_encoder=bert_encoder,
                                                  tokenizer=tokenizer, max_length=max_length,
                                                  device=device, batch_size=vocab_batch_size, show_progress=False).to(device)

            # (num_queries, 1, emb_h) x (1, batch_size, emb_h) = (num_queries, batch_size)
            batch_score_matrix = F.cosine_similarity(query_embs,
                                                     batch_vocab_embeddings.unsqueeze(0),
                                                     dim=-1)  # .detach().cpu().numpy()
            assert batch_score_matrix.shape == (num_queries, vocab_end_pos - vocab_start_pos)
            k = min(base_k, vocab_end_pos - vocab_start_pos)
            # (num_queries, batch_size) -> num_queries, k
            b_max, b_indices = torch.topk(batch_score_matrix, k=k, dim=1)
            b_indices += vocab_start_pos
            if overall_max is None:
                overall_max = b_max
                overall_max_indices = b_indices

            assert b_max.size() == (num_queries, k)
            assert b_indices.size() == (num_queries, k)
            # (num_queries * 2, k)
            concat_max = torch.cat((overall_max, b_max), dim=1)
            concat_indices = torch.cat((overall_max_indices, b_indices), dim=1)
            assert concat_max.size() == (num_queries, base_k + k)
            assert concat_indices.size() == (num_queries, base_k + k)

            overall_max, local_indices = torch.topk(concat_max, k=base_k, dim=1)
            assert overall_max.size() == (num_queries, base_k)
            assert local_indices.size() == (num_queries, base_k)

            x_index = create_repeated_tensor(range_bound=num_queries, repeat_times=base_k)

            overall_max_indices = concat_indices[x_index, local_indices.view(-1)]
            overall_max_indices = overall_max_indices.view(size=overall_max.size())
    return {
        "best_scores": overall_max,
        "best_indices": overall_max_indices
    }

DOCUMENT_ID_COL = "document_id"
SPANS_COL = "spans"
RANK_COL = "rank"
# Sample primary key column
SAMPLE_PK_COL = "pk"
TRUE_CUI_COL = "UMLS_CUI"
RANK_COL = "rank"
PREDICTION_COL = "prediction"


def evaluate(df):
    pred_labels = df["prediction"].values
    true_labels = df["label"].values

    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    return p, r, f1, acc


def calculate_metrics(pred_df: pd.DataFrame, sample_pk2true_cui: Dict[int, str]):
    sample_id2min_true_predicted_rank: Dict[int, int] = {}
    for _, row in pred_df.iterrows():
        sample_pk = row[SAMPLE_PK_COL]
        rank = row[RANK_COL]
        pred_cui = row[PREDICTION_COL]

        true_cui = sample_pk2true_cui[sample_pk]
        if pred_cui == true_cui:
            if sample_id2min_true_predicted_rank.get(sample_pk) is None:
                sample_id2min_true_predicted_rank[sample_pk] = rank
            sample_id2min_true_predicted_rank[sample_pk] = min(sample_id2min_true_predicted_rank[sample_pk], rank)
    acc_1_sum = 0.
    acc_5_sum = 0.
    mrr_sum = 0.
    for sample_id in sample_pk2true_cui.keys():
        rank = sample_id2min_true_predicted_rank.get(sample_id, -1)
        assert rank != 0
        if rank == -1:
            sample_acc_1 = 0.
            sample_acc_5 = 0.
            sample_mrr = 0.
        else:
            sample_acc_1 = 0.
            sample_acc_5 = 0.
            if rank == 1:
                sample_acc_1 = 1.
            if rank <= 5:
                sample_acc_5 = 1.
            sample_mrr = 1. / rank
        acc_1_sum += sample_acc_1
        acc_5_sum += sample_acc_5
        mrr_sum += sample_mrr
    num_samples = len(sample_pk2true_cui)

    acc_1 = acc_1_sum / num_samples
    acc_5 = acc_5_sum / num_samples
    mrr = mrr_sum / num_samples

    return {
        "Acc@1": acc_1,
        "Acc@5": acc_5,
        "MRR": mrr
    }


def create_sample_pk2_true_cui_map(df: pd.DataFrame, true_cui_column) -> Dict[int, str]:
    sample_pk2cui = {}
    for _, row in df.iterrows():
        sample_pk = row[SAMPLE_PK_COL]
        true_cui = row[true_cui_column]
        assert '|' not in true_cui

        sample_pk2cui[sample_pk] = true_cui

    return sample_pk2cui


def create_row_primary_key(row):
    doc_id = row[DOCUMENT_ID_COL]
    spans = row[SPANS_COL]
    assert '|' not in str(doc_id)
    assert '|' not in str(spans)
    pk = f"{doc_id}|{spans}"

    return pk

def filter_vocab(vocab, lang):
    if lang == "ru":
        ru_cuis_set = set(vocab[vocab["lang"] == "RUS"]["CUI"].unique())
        vocab = vocab[vocab["CUI"].isin(ru_cuis_set)][["CUI", "semantic_type", "concept_name"]]
        print(f"Created Russian vocab: {vocab.shape}")
    elif lang == "en":
        vocab = vocab[vocab["lang"] == "ENG"][["CUI", "semantic_type", "concept_name"]]
        print(f"Created English vocab: {vocab.shape}")

    return vocab


# Loading monolingual data (Track 1: Russian/English)
ru_data_train = load_dataset("andorei/BioNNE-L", "Russian", split="train")
ru_data_dev = load_dataset("andorei/BioNNE-L", "Russian", split="dev")
en_data_train = load_dataset("andorei/BioNNE-L", "English", split="train")
en_data_dev = load_dataset("andorei/BioNNE-L", "English", split="dev")

ru_data_train = ru_data_train.to_pandas()
ru_data_dev = ru_data_dev.to_pandas()
en_data_train = en_data_train.to_pandas()
en_data_dev = en_data_dev.to_pandas()


# Loading normalization vocabulary
vocab = load_dataset("andorei/BioNNE-L", "Vocabulary", split="train")
vocab = vocab.to_pandas()


def filter_vocab(vocab, lang):
    if lang == "ru":
        ru_cuis_set = set(vocab[vocab["lang"] == "RUS"]["CUI"].unique())
        vocab = vocab[vocab["CUI"].isin(ru_cuis_set)][["CUI", "semantic_type", "concept_name"]]
        print(f"Created Russian vocab: {vocab.shape}")
    elif lang == "en":
        vocab = vocab[vocab["lang"] == "ENG"][["CUI", "semantic_type", "concept_name"]]
        print(f"Created English vocab: {vocab.shape}")

    return vocab

model_name = "andorei/BERGAMOT-multilingual-GAT"
query_batch_size = 128
vocab_batch_size = 800
max_length = 48

output_ru_path = "./predictions/ru_predictions.tsv"
k = 5

output_dir = os.path.dirname(output_ru_path)
if not os.path.exists(output_dir) and output_dir != '':
    os.makedirs(output_dir)


def make_predictions(entities_df, tokenizer, bert_encoder, vocab, max_length,
                     k, device, query_batch_size, vocab_batch_size):
    predictions_list = []

    for chem_type in ("DISO", "CHEM", "ANATOMY"):
        subset_df = entities_df[entities_df["entity_type"] == chem_type]

        document_ids = subset_df["document_id"].values
        query_names = subset_df["text"].values
        entity_types = subset_df["entity_type"].values
        ground_truth_cuis = subset_df["UMLS_CUI"].values
        spans = subset_df["spans"].values
        subset_vocab = vocab[vocab["semantic_type"] == chem_type]

        vocab_names = subset_vocab["concept_name"].values
        vocab_cuis = subset_vocab["CUI"].values

        pred_d = get_torch_query_dict_score_matrix(query_names, tokenizer, bert_encoder, vocab_names, k, device,
                                                   query_batch_size, vocab_batch_size=vocab_batch_size,
                                                   max_length=max_length, show_progress=True)
        # <queries, k>
        pred_indices = pred_d["best_indices"]
        assert len(pred_indices) == len(query_names) == len(spans) == len(document_ids)
        for doc_id, pred_idx, qn, sp in zip(document_ids, pred_indices, query_names, spans):
            pred_cuis = [vocab_cuis[x.item()] for x in pred_idx]
            for rank, cui in enumerate(pred_cuis):
                # (1) document_id, (2) spans, (3) rank, (4) prediction.
                d = {
                    "document_id": doc_id,
                    "spans": sp,
                    "rank": rank + 1,
                    "prediction": cui
                }
                predictions_list.append(d)

    return pd.DataFrame(predictions_list)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

bert_encoder = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ru_vocab = filter_vocab(vocab, "ru")
ru_vocab.shape

ru_predictions_df = make_predictions(entities_df=ru_data_dev, tokenizer=tokenizer,
                              bert_encoder=bert_encoder, vocab=ru_vocab,
                              query_batch_size=query_batch_size,
                              k=k, max_length=max_length, device=device,
                              vocab_batch_size=vocab_batch_size)

ru_predictions_df[SAMPLE_PK_COL] = ru_predictions_df.apply(create_row_primary_key, axis=1)
ru_data_dev[SAMPLE_PK_COL] = ru_data_dev.apply(create_row_primary_key, axis=1)

ru_data_dev = ru_data_dev.merge(ru_predictions_df[[SAMPLE_PK_COL, RANK_COL, PREDICTION_COL]], on=SAMPLE_PK_COL)
ru_data_dev = ru_data_dev[ru_data_dev[TRUE_CUI_COL] != "CUILESS"]

sample_pk2true_cui = create_sample_pk2_true_cui_map(df=ru_data_dev,
                                                        true_cui_column=TRUE_CUI_COL)

eval_dict = calculate_metrics(pred_df=ru_data_dev,
                                  sample_pk2true_cui=sample_pk2true_cui)

with open("ru_results.txt", "w") as f:
    for key, value in eval_dict.items():
        f.write(f"{key}: {value:.4f}\n")

output_ru_path = "./predictions/ru_predictions.tsv"

output_dir = os.path.dirname(output_ru_path)
if not os.path.exists(output_dir) and output_dir != '':
    os.makedirs(output_dir)
ru_predictions_df.to_csv(output_ru_path, sep='\t', index=False)

model_name = "andorei/gebert_eng_gat"
query_batch_size = 128
vocab_batch_size = 1600
max_length = 32
k = 5

bert_encoder = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

en_vocab = filter_vocab(vocab, "en")
en_vocab.shape

en_predictions_df = make_predictions(entities_df=en_data_dev, tokenizer=tokenizer,
                              bert_encoder=bert_encoder, vocab=en_vocab,
                              query_batch_size=query_batch_size,
                              k=k, max_length=max_length, device=device,
                              vocab_batch_size=vocab_batch_size)

en_predictions_df[SAMPLE_PK_COL] = en_predictions_df.apply(create_row_primary_key, axis=1)
en_data_dev[SAMPLE_PK_COL] = en_data_dev.apply(create_row_primary_key, axis=1)

en_data_dev = en_data_dev.merge(en_predictions_df[[SAMPLE_PK_COL, RANK_COL, PREDICTION_COL]], on=SAMPLE_PK_COL)
en_data_dev = en_data_dev[en_data_dev[TRUE_CUI_COL] != "CUILESS"]

sample_pk2true_cui = create_sample_pk2_true_cui_map(df=en_data_dev,
                                                        true_cui_column=TRUE_CUI_COL)

eval_dict = calculate_metrics(pred_df=en_data_dev,
                                  sample_pk2true_cui=sample_pk2true_cui)

with open("en_results.txt", "w") as f:
    for key, value in eval_dict.items():
        f.write(f"{key}: {value:.4f}\n")

output_en_path = "./predictions/en_predictions.tsv"

output_dir = os.path.dirname(output_en_path)
if not os.path.exists(output_dir) and output_dir != '':
    os.makedirs(output_dir)
en_predictions_df.to_csv(output_en_path, sep='\t', index=False)