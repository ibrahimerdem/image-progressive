#!/usr/bin/env bash
set -e

LD_CHECKPOINT=${1:-""}
VAE_CHECKPOINT=${2:-""}

if [ ! -f "${LD_CHECKPOINT}" ]; then
  echo "Error: LD checkpoint not found: ${LD_CHECKPOINT}"
  echo "Usage: ./run_ld_test.sh <ld_checkpoint> [vae_checkpoint]"
  echo "Note: VAE checkpoint is optional, defaults to config.VAE_CKPT"
  exit 1
fi

if [ -n "${VAE_CHECKPOINT}" ] && [ ! -f "${VAE_CHECKPOINT}" ]; then
  echo "Error: VAE checkpoint not found: ${VAE_CHECKPOINT}"
  echo "Usage: ./run_ld_test.sh <ld_checkpoint> [vae_checkpoint]"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "============================================"
echo "Latent Diffusion Model Test Evaluation"
echo "============================================"
echo "LD Checkpoint:   ${LD_CHECKPOINT}"
if [ -n "${VAE_CHECKPOINT}" ]; then
  echo "VAE Checkpoint:  ${VAE_CHECKPOINT}"
else
  echo "VAE Checkpoint:  (from config.py)"
fi
echo "Device / Batch / Steps: (from config.py)"
echo "============================================"
echo ""

if [ -n "${VAE_CHECKPOINT}" ]; then
  python ld_evaluation.py \
    --checkpoint "${LD_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}"
else
  python ld_evaluation.py \
    --checkpoint "${LD_CHECKPOINT}"
fi