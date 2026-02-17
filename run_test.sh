#!/usr/bin/env bash
set -e

# Simple test evaluation script
# Runs evaluation on test dataset with default settings

CHECKPOINT=${1:-checkpoints/multimodal_basic_ddp_final_ddp.pth}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Running test evaluation..."
echo "Checkpoint: ${CHECKPOINT}"

python evaluation.py \
  --mode test \
  --checkpoint "${CHECKPOINT}" \
  --device cuda:0 \
  --batch_size 16

echo "Done!"

