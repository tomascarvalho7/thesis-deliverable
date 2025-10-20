#!/bin/bash

DATA_SPLIT=$1

# Validate DATA_SPLIT
if [[ "$DATA_SPLIT" != "rand" && "$DATA_SPLIT" != "stratified" ]]; then
  echo "Invalid DATA_SPLIT: $DATA_SPLIT. Must be 'rand' or 'stratified'."
  exit 1
fi

DATA=bandcrf${DATA_SPLIT}
DATA_DIR=data/$DATA
OUTPUT_DIR=output/"$DATA"

mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

BERT_BASE_DIR=bert-base-cased

python3 -u $DBG BERT-BiLSTM-CRF-NER-pytorch/ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR 2>&1 | tee output/log.txt

# Additional command for testing (commented out by default)
# python -u $DBG BERT-BiLSTM-CRF-NER-pytorch/ner.py --model_name_or_path ${BERT_BASE_DIR} --do_test True --max_seq_length 256 --test_file ${DATA_DIR}/test.txt --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 3 --do_lower_case --logging_steps 200 --need_birnn True --rnn_dim 256 --clean True --output_dir $OUTPUT_DIR

# Move the log file to the output directory
mv output/log.txt $OUTPUT_DIR/log.txt
