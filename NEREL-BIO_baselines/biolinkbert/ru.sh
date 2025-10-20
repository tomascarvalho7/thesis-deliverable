#!/bin/bash

# Activate virtual environment
source ~/myenv/bin/activate

cd ./binder

# Run your Python
python3 run_ner.py   --model_name_or_path michiyasunaga/BioLinkBERT-base  --dataset_name bioNNEL_ru --train_file ./data/ru/train.json   --validation_file ./data/ru/dev.json   --test_file ./data/ru/test.json   --entity_type_file ./data/ru/bioNNEL_dataset/entity_types.json   --output_dir ./outputs/run_ru_bert   --do_train --do_eval --do_predict   --overwrite_output_dir   --max_seq_length 256   --per_device_train_batch_size 4   --per_device_eval_batch_size 4   --learning_rate 3e-5   --load_best_model_at_end True   --num_train_epochs 100   --evaluation_strategy epoch  --save_strategy epoch  --save_total_limit 1   --metric_for_best_model f1   --greater_is_better True
