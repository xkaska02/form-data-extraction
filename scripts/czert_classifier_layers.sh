#!/bin/bash

BASE_DIR="/home/kaska/BP/form-data-extraction"

source $BASE_DIR/venv/bin/activate

SCRIPT_DIR=$BASE_DIR/code
SAVE_DIR=$BASE_DIR/models_tmp/forms/model
DATAFILES_DIR=$BASE_DIR/data_files/forms_json_dataset
MODEL=UWB-AIR/Czert-B-base-cased

mkdir -p $SAVE_DIR

python3 $SCRIPT_DIR/train_bert_on_forms.py \
    --save_path $SAVE_DIR \
    --model_path $MODEL \
    --model_name czert \
    --train_file $DATAFILES_DIR/train_split.json \
    --test_file $DATAFILES_DIR/test_split.json \
    --validation_file $DATAFILES_DIR/validation_split.json \
    --lr 2e-5 \
    --batch_size 4 \
    --epochs 30 \
    --decay 0.01 \
    --eval_strat epoch \
    --save_strat epoch \
    --classifier_head_layers 1 \
    --experiment_name classifier_layers

rm -r $SAVE_DIR/*

python3 $SCRIPT_DIR/train_bert_on_forms.py \
    --save_path $SAVE_DIR \
    --model_path $MODEL \
    --model_name czert \
    --train_file $DATAFILES_DIR/train_split.json \
    --test_file $DATAFILES_DIR/test_split.json \
    --validation_file $DATAFILES_DIR/validation_split.json \
    --lr 2e-5 \
    --batch_size 4 \
    --epochs 30 \
    --decay 0.01 \
    --eval_strat epoch \
    --save_strat epoch \
    --classifier_head_layers 2 \
    --experiment_name classifier_layers

rm -r $SAVE_DIR/*

python3 $SCRIPT_DIR/train_bert_on_forms.py \
    --save_path $SAVE_DIR \
    --model_path $MODEL \
    --model_name czert \
    --train_file $DATAFILES_DIR/train_split.json \
    --test_file $DATAFILES_DIR/test_split.json \
    --validation_file $DATAFILES_DIR/validation_split.json \
    --lr 2e-5 \
    --batch_size 4 \
    --epochs 30 \
    --decay 0.01 \
    --eval_strat epoch \
    --save_strat epoch \
    --classifier_head_layers 3 \
    --experiment_name classifier_layers

rm -r $SAVE_DIR/*
