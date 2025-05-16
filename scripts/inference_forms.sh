#!/bin/bash

BASE_DIR="/home/kaska/BP/form-data-extraction"

source $BASE_DIR/venv/bin/activate

SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp/czert_lr2e-05_bs4_train287_max_len8
DATAFILES_DIR=$BASE_DIR/data_files/forms_json_dataset
MODEL_CHECKPOINT=checkpoint-288

    
    # --model_path "UWB-AIR/Czert-B-base-cased" \
python3 $SCRIPT_DIR/inference_forms.py \
    --model_path $SAVE_DIR/$MODEL_CHECKPOINT \
    --train_file $DATAFILES_DIR/train_split.json \
    --test_file $DATAFILES_DIR/test_split.json \
    --validation_file $DATAFILES_DIR/validation_split.json 

# python3 $SCRIPT_DIR/visualize_results.py