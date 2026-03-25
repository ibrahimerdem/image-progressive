#!/usr/bin/env bash
set -e

CHECKPOINT=${1:-""}
VAE_CHECKPOINT=${2:-""}
DEVICE=${3:-cuda:0}
BATCH_SIZE=${4:-4}
INFERENCE_STEPS=${5:-50}

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Error: SD checkpoint not found: ${CHECKPOINT}"
  echo "Usage: ./run_sd_test.sh <sd_checkpoint> [vae_checkpoint] [device] [batch_size] [inference_steps]"
  echo "Note: VAE checkpoint is optional, defaults to config.VAE_CKPT"
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
echo "Test Evaluation"
echo "============================================"
echo "Checkpoint:   ${CHECKPOINT}"
if [ -n "${VAE_CHECKPOINT}" ]; then
  echo "VAE Checkpoint:  ${VAE_CHECKPOINT}"
else
  echo "VAE Checkpoint:  (from config.VAE_CKPT)"
fi
echo "Device:          ${DEVICE}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Inference Steps: ${INFERENCE_STEPS}"
echo "============================================"
echo ""

if [ -n "${VAE_CHECKPOINT}" ]; then
  python ld_evaluation.py \
    --checkpoint "${CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --inference_steps "${INFERENCE_STEPS}"
else
  python ld_evaluation.py \
    --checkpoint "${CHECKPOINT}" \
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
