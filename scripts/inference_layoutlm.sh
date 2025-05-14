#!/bin/bash

BASE_DIR="/mnt/c/Users/kaska/Documents/vysoka_skola/form-data-extraction"
SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp/layoutlmv3
DATAFILES_DIR=$BASE_DIR/data_files
MODEL_CHECKPOINT=checkpoint-1000

    
    # --model_path "UWB-AIR/Czert-B-base-cased" \
python3 $SCRIPT_DIR/inference_layoutlm.py \
    --model_path $SAVE_DIR/$MODEL_CHECKPOINT \
    --train_file $DATAFILES_DIR/train.parquet \
    --test_file $DATAFILES_DIR/test.parquet