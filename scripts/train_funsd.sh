#!/bin/bash

BASE_DIR="/mnt/c/Users/kaska/Documents/vysoka_skola/form-data-extraction"
SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp/funsd_more_epochs
DATAFILES_DIR=$BASE_DIR/data_files/FUNSD/dataset
LABEL_LIST=(OTHER, QUESTION, ANSWER)

mkdir -p $SAVE_DIR

python3 $SCRIPT_DIR/train_funsd.py --save_path $SAVE_DIR \
    --train_folder $DATAFILES_DIR/training_data/annotations \
    --test_folder $DATAFILES_DIR/testing_data/annotations \
    --lr 2e-5 \
    --batch_size 2 \
    --epochs 10 \
    --decay 0.01 \
    --eval_strat epoch \
    --save_strat epoch \
    --label_list ${LABEL_LIST[@]} \
    --file_type json \
    --model_path google-bert/bert-base-uncased