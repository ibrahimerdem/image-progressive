#!/bin/bash

# Train VAE encoder-decoder
# Usage: ./run_vae_train.sh [epochs] [retrain] [checkpoint]
# Example: ./run_vae_train.sh 50 0
# Example (resume): ./run_vae_train.sh 100 1 checkpoints/vae_epoch_50.pth

EPOCHS=${1:-50}
RETRAIN=${2:-0}
CHECKPOINT=${3:-""}

echo "Training VAE for $EPOCHS epochs..."

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ $RETRAIN -eq 1 ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    python3 vae_trainer.py --epochs $EPOCHS --retrain $RETRAIN --checkpoint $CHECKPOINT
else
    echo "Starting fresh training"
    python3 vae_trainer.py --epochs $EPOCHS --retrain $RETRAIN
fi
