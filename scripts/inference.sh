#!/bin/bash

BASE_DIR="/mnt/c/Users/kaska/Documents/vysoka_skola/form-data-extraction"
SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp
DATAFILES_DIR=$BASE_DIR/data_files
MODEL_CHECKPOINT=checkpoint-900

    
python3 $SCRIPT_DIR/inference.py \
    --model_path $SAVE_DIR/$MODEL_CHECKPOINT \
    --train_file $DATAFILES_DIR/train.parquet \
    --test_file $DATAFILES_DIR/test.parquet