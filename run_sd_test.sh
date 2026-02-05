#!/usr/bin/env bash
set -e

# Script to run SD model test evaluation
# Usage: ./run_sd_test.sh <sd_checkpoint_path> [vae_checkpoint_path] [device] [batch_size] [inference_steps]

SD_CHECKPOINT=${1:-checkpoints/sd/sd_ddp_epoch_0050.pth}
VAE_CHECKPOINT=${2:-""}
DEVICE=${3:-cuda:0}
BATCH_SIZE=${4:-4}
INFERENCE_STEPS=${5:-50}

if [ ! -f "${SD_CHECKPOINT}" ]; then
  echo "Error: SD checkpoint not found: ${SD_CHECKPOINT}"
  echo "Usage: ./run_sd_test.sh <sd_checkpoint> [vae_checkpoint] [device] [batch_size] [inference_steps]"
  echo "Note: VAE checkpoint is optional, defaults to config.SD_VAE_CKPT"
  exit 1
fi

if [ -n "${VAE_CHECKPOINT}" ] && [ ! -f "${VAE_CHECKPOINT}" ]; then
  echo "Error: VAE checkpoint not found: ${VAE_CHECKPOINT}"
  echo "Usage: ./run_sd_test.sh <sd_checkpoint> [vae_checkpoint] [device] [batch_size] [inference_steps]"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "============================================"
echo "SD Model Test Evaluation"
echo "============================================"
echo "SD Checkpoint:   ${SD_CHECKPOINT}"
if [ -n "${VAE_CHECKPOINT}" ]; then
  echo "VAE Checkpoint:  ${VAE_CHECKPOINT}"
else
  echo "VAE Checkpoint:  (from config.SD_VAE_CKPT)"
fi
echo "Device:          ${DEVICE}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Inference Steps: ${INFERENCE_STEPS}"
echo "============================================"
echo ""

if [ -n "${VAE_CHECKPOINT}" ]; then
  python sd_evaluation.py \
    --checkpoint "${SD_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --inference_steps "${INFERENCE_STEPS}"
else
  python sd_evaluation.py \
    --checkpoint "${SD_CHECKPOINT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --inference_steps "${INFERENCE_STEPS}"
fi

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: outputs/sd/sd_test_results.txt"
echo "Samples saved to: outputs/sd/sample_*"
echo "============================================"
