#!/bin/bash

BASE_DIR="/home/kaska/BP/form-data-extraction"

source $BASE_DIR/venv/bin/activate

SCRIPT_DIR=$BASE_DIR/code
DATAFILES_DIR=$BASE_DIR/data_files/forms_json_dataset
INPUT_DIR=$BASE_DIR/out
# INPUT_FILE=czert_lr2e-05_bs4_train287_max_len8.json
# INPUT_FILE=lilt-xlm-roberta-base.json
INPUT_FILE=lilt_robeczech_lr3e-05_bs8_train287.json

python3 $SCRIPT_DIR/visualize_results.py \
    --input_file $INPUT_DIR/$INPUT_FILE \
    --save_images False
    # --save_path out/test