#!/bin/bash

BASE_DIR="/mnt/c/Users/kaska/Documents/vysoka_skola/form-data-extraction"
SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp
DATAFILES_DIR=$BASE_DIR/data_files
LABEL_LIST=(O NUMBER_IN_ADDR GEOGRAPHICAL_NAME INSTITUTION MEDIA NUMBER_EXPRESSION ARTIFACT_NAME PERSONAL_NAME TIME_EXPRESSION)

mkdir -p $SAVE_DIR

python3 $SCRIPT_DIR/train.py --save_path $SAVE_DIR \
    --train_file $DATAFILES_DIR/train.parquet \
    --test_file $DATAFILES_DIR/test.parquet \
    --lr 2e-5 \
    --batch_size 16 \
    --epochs 2 \
    --decay 0.01 \
    --eval_strat epoch \
    --save_strat epoch \
    --label_list ${LABEL_LIST[@]} \
    --file_type parquet