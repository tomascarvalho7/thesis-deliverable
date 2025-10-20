import os
from typing import Dict
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ===================== Embedding & Utility Functions =====================

def encode_names(names, bert_encoder, tokenizer, max_length, device,
                 batch_size=256, show_progress=False):
    bert_encoder.eval()
    if isinstance(names, pd.Series):
        names = names.tolist()

    name_encodings = tokenizer(
        names, padding="max_length",
        max_length=max_length, truncation=True,
        return_tensors="pt"
    )
    input_ids = name_encodings["input_ids"]
    attention_mask = name_encodings["attention_mask"]

    embs = []
    indices = range(0, len(names), batch_size)
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
            embs.append(batch_embeddings.detach().cpu())

    final_embeddings = torch.cat(embs, dim=0).detach()
    assert final_embeddings.size(0) == len(names)
    return final_embeddings


def create_repeated_tensor(range_bound, repeat_times):
    return torch.repeat_interleave(torch.arange(range_bound), repeat_times)


def get_torch_query_dict_score_matrix(query_names, tokenizer, bert_encoder, vocab_names, base_k, device,
                                      query_batch_size, max_length, show_progress, vocab_batch_size=256):
    bert_encoder.eval()
    num_queries = len(query_names)
    vocab_length = len(vocab_names)

    query_embs = encode_names(
        names=query_names, bert_encoder=bert_encoder,
        tokenizer=tokenizer, max_length=max_length,
        device=device, batch_size=query_batch_size,
        show_progress=show_progress
    ).unsqueeze(1).to(device).detach()

    overall_max, overall_max_indices = None, None
    with torch.no_grad():
        for vocab_start_pos in tqdm(range(0, vocab_length, vocab_batch_size)):
            vocab_end_pos = min(vocab_start_pos + vocab_batch_size, vocab_length)
            batch_vocab_names = vocab_names[vocab_start_pos:vocab_end_pos]

            batch_vocab_embeddings = encode_names(
                names=batch_vocab_names,
                bert_encoder=bert_encoder,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
                batch_size=vocab_batch_size,
                show_progress=False
            ).to(device)

            batch_score_matrix = F.cosine_similarity(
                query_embs,
                batch_vocab_embeddings.unsqueeze(0),
                dim=-1
            )
            k = min(base_k, vocab_end_pos - vocab_start_pos)
            b_max, b_indices = torch.topk(batch_score_matrix, k=k, dim=1)
            b_indices += vocab_start_pos

            if overall_max is None:
                overall_max, overall_max_indices = b_max, b_indices
            else:
                concat_max = torch.cat((overall_max, b_max), dim=1)
                concat_indices = torch.cat((overall_max_indices, b_indices), dim=1)
                overall_max, local_indices = torch.topk(concat_max, k=base_k, dim=1)
                x_index = create_repeated_tensor(range_bound=num_queries, repeat_times=base_k)
                overall_max_indices = concat_indices[x_index, local_indices.view(-1)].view(size=overall_max.size())

    return {"best_scores": overall_max, "best_indices": overall_max_indices}


# ===================== Evaluation =====================

DOCUMENT_ID_COL = "document_id"
SPANS_COL = "spans"
SAMPLE_PK_COL = "pk"
TRUE_CUI_COL = "UMLS_CUI"
RANK_COL = "rank"
PREDICTION_COL = "prediction"


def calculate_retrieval_metrics(pred_df: pd.DataFrame, sample_pk2true_cui: Dict[int, str]):
    sample_id2min_true_predicted_rank: Dict[int, int] = {}
    for _, row in pred_df.iterrows():
        sample_pk, rank, pred_cui = row[SAMPLE_PK_COL], row[RANK_COL], row[PREDICTION_COL]
        true_cui = sample_pk2true_cui[sample_pk]
        if pred_cui == true_cui:
            if sample_pk not in sample_id2min_true_predicted_rank:
                sample_id2min_true_predicted_rank[sample_pk] = rank
            sample_id2min_true_predicted_rank[sample_pk] = min(sample_id2min_true_predicted_rank[sample_pk], rank)

    acc_1_sum = acc_5_sum = mrr_sum = 0.
    for sample_id in sample_pk2true_cui.keys():
        rank = sample_id2min_true_predicted_rank.get(sample_id, -1)
        if rank == -1:
            continue
        if rank == 1:
            acc_1_sum += 1.
        if rank <= 5:
            acc_5_sum += 1.
        mrr_sum += 1. / rank

    num_samples = len(sample_pk2true_cui)
    return {"Acc@1": acc_1_sum / num_samples, "Acc@5": acc_5_sum / num_samples, "MRR": mrr_sum / num_samples}


def calculate_classification_metrics(pred_df: pd.DataFrame, sample_pk2true_cui: Dict[int, str]):
    top1_df = pred_df[pred_df[RANK_COL] == 1]
    y_true = [sample_pk2true_cui[pk] for pk in top1_df[SAMPLE_PK_COL]]
    y_pred = top1_df[PREDICTION_COL].tolist()
    return {
        "Precision": precision_score(y_true, y_pred, average="micro"),
        "Recall": recall_score(y_true, y_pred, average="micro"),
        "F1": f1_score(y_true, y_pred, average="micro"),
        "Accuracy": accuracy_score(y_true, y_pred)
    }


def create_sample_pk2_true_cui_map(df: pd.DataFrame, true_cui_column) -> Dict[int, str]:
    return {row[SAMPLE_PK_COL]: row[true_cui_column] for _, row in df.iterrows()}


def create_row_primary_key(row):
    return f"{row[DOCUMENT_ID_COL]}|{row[SPANS_COL]}"


def filter_vocab(vocab, lang):
    if lang == "ru":
        ru_cuis_set = set(vocab[vocab["lang"] == "RUS"]["CUI"].unique())
        vocab = vocab[vocab["CUI"].isin(ru_cuis_set)][["CUI", "semantic_type", "concept_name"]]
        print(f"Created Russian vocab: {vocab.shape}")
    return vocab


# ===================== Prediction =====================

def make_predictions(entities_df, tokenizer, bert_encoder, vocab, max_length,
                     k, device, query_batch_size, vocab_batch_size):
    predictions_list = []
    for chem_type in ("DISO", "CHEM", "ANATOMY"):
        subset_df = entities_df[entities_df["entity_type"] == chem_type]
        document_ids = subset_df["document_id"].values
        query_names = subset_df["text"].astype(str).tolist()
        spans = subset_df["spans"].values
        subset_vocab = vocab[vocab["semantic_type"] == chem_type]
        vocab_names = subset_vocab["concept_name"].astype(str).tolist()
        vocab_cuis = subset_vocab["CUI"].values

        pred_d = get_torch_query_dict_score_matrix(
            query_names, tokenizer, bert_encoder, vocab_names, k, device,
            query_batch_size, vocab_batch_size=vocab_batch_size,
            max_length=max_length, show_progress=True
        )
        pred_indices = pred_d["best_indices"]
        for doc_id, pred_idx, sp in zip(document_ids, pred_indices, spans):
            pred_cuis = [vocab_cuis[x.item()] for x in pred_idx]
            for rank, cui in enumerate(pred_cuis):
                predictions_list.append({
                    "document_id": doc_id,
                    "spans": sp,
                    "rank": rank + 1,
                    "prediction": cui
                })
    return pd.DataFrame(predictions_list)


# ===================== Training Loop (Russian Only) =====================

def run_training(epochs=3):
    # ---- Load LOCAL parquet datasets ----
    ru_data_train = pd.read_parquet("./data/parquet/ru/bionnel_ru_train.parquet")
    ru_data_dev = pd.read_parquet("./data/parquet/ru/bionnel_ru_dev.parquet")
    vocab = pd.read_parquet("./data/vocabular/bionnel_vocab_bilingual.parquet")
    ru_vocab = filter_vocab(vocab, "ru")

    model_name = "andorei/BERGAMOT-multilingual-GAT"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_encoder = AutoModel.from_pretrained(model_name).to(device)

    query_batch_size, vocab_batch_size, max_length, k = 128, 800, 48, 5

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch} (Russian) =====")

        predictions_df = make_predictions(
            entities_df=ru_data_dev,
            tokenizer=tokenizer,
            bert_encoder=bert_encoder,
            vocab=ru_vocab,
            query_batch_size=query_batch_size,
            k=k,
            max_length=max_length,
            device=device,
            vocab_batch_size=vocab_batch_size
        )

        predictions_df[SAMPLE_PK_COL] = predictions_df.apply(create_row_primary_key, axis=1)
        ru_data_dev[SAMPLE_PK_COL] = ru_data_dev.apply(create_row_primary_key, axis=1)

        merged_df = ru_data_dev.merge(
            predictions_df[[SAMPLE_PK_COL, RANK_COL, PREDICTION_COL]], on=SAMPLE_PK_COL
        )
        merged_df = merged_df[merged_df[TRUE_CUI_COL] != "CUILESS"]

        sample_pk2true_cui = create_sample_pk2_true_cui_map(merged_df, TRUE_CUI_COL)

        retrieval_metrics = calculate_retrieval_metrics(merged_df, sample_pk2true_cui)
        classification_metrics = calculate_classification_metrics(merged_df, sample_pk2true_cui)

        print("Evaluation Metrics:")
        for k, v in {**retrieval_metrics, **classification_metrics}.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    run_training(epochs=3)
