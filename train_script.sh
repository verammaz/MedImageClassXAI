#!/bin/bash

# Get access to global variables
source ./config.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [options] -m <model_name> -t <train_hparams.json>

Required Arguments:
  -m <model_name>                  Name of model to train (SimpleCNN, ResNet, ViT).
  -t <train_hparams.json>          File with training hyperparameters.

Options:
  -h                               Display this message.
  -v                               Enable verbose mode.

EOF
    exit 1
}

# Variables
MODEL=
HPARAMS=
VERBOSE=

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h) usage ;;
        -v) VERBOSE=1 ;;
        -m) MODEL="$2"; shift ;;
        -t) HPARAMS="$2"; shift ;;
        *) echo "Error: Unkown argument/option: $1" ; usage ;;
    esac
    shift
done

if [ MODEL == SimpleCNN ]; then

    python $SCRIPTS/cnn.py
