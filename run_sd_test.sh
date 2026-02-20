#!/usr/bin/env bash
set -e

# Simple SD test evaluation script
# Runs evaluation on test dataset with default settings

SD_CHECKPOINT=${1:-checkpoints/sd/sd_ddp_epoch_0050.pth}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Running SD test evaluation..."
echo "Checkpoint: ${SD_CHECKPOINT}"

python sd_evaluation.py \
  --checkpoint "${SD_CHECKPOINT}" \
  --device cuda:0 \
  --batch_size 4 \
  --inference_steps 200

echo "Done!"
