#!/bin/bash

DATA_SPLIT=$1

if [[ "$DATA_SPLIT" != "rand" && "$DATA_SPLIT" != "stratified" ]]; then
  echo "Invalid DATA_SPLIT: $DATA_SPLIT. Must be 'rand' or 'stratified'."
  exit 1
fi

# Combine TYPE and DATA_SPLIT to form the DATA variable
DATA=bandtoken${DATA_SPLIT}
EXP=$DATA
MODEL=bert-base-cased

# Set output directory
OUTPUT_DIR=output/$EXP
mkdir -p $OUTPUT_DIR

# Run the Python script with specified arguments
python3 src/tf_ner.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --save_steps 10000000000 \
  --train_file data/$DATA/train.json \
  --validation_file data/$DATA/dev.json \
  --test_file data/$DATA/test.json 2>&1 | tee $OUTPUT_DIR/log.txt \

#python3 src/tf_ner.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir output/tf_ner --do_train --do_eval --overwrite_output_dir --save_steps 10000000000