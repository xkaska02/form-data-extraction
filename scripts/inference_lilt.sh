#!/bin/bash

BASE_DIR="/home/kaska/BP/form-data-extraction"

source $BASE_DIR/venv/bin/activate

SCRIPT_DIR=$BASE_DIR/code
# MODEL_DIR=$BASE_DIR/models_tmp/czert_lr2e-05_bs4_train287_max_len8
DATAFILES_DIR=$BASE_DIR/data_files/forms_json_dataset
# MODEL_CHECKPOINT=checkpoint-288
MODEL_PATH="xkaska02/lilt_robeczech_lr3e-05_bs8_train287"

python3 $SCRIPT_DIR/inference_lilt_text_only.py \
    --model_path $MODEL_PATH \
    --train_file $DATAFILES_DIR/train_split.json \
    --test_file $DATAFILES_DIR/test_split.json \
    --validation_file $DATAFILES_DIR/validation_split.json \
    --sample_count 5