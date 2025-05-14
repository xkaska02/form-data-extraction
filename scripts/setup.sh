#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

chmod +x $BASE_DIR/scripts/run_lilt_experiments.sh
chmod +x $BASE_DIR/scripts/lilt_train_dataset_size.sh
chmod +x $BASE_DIR/scripts/lilt_loss_on_subtokens.sh
chmod +x $BASE_DIR/scripts/lilt_classifier_layers.sh
chmod +x $BASE_DIR/scripts/different_lilt_backbones.sh
