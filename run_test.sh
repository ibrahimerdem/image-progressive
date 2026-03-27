#!/usr/bin/env bash
set -e

CHECKPOINT=${1:-""}

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Error: Checkpoint not found: ${CHECKPOINT}"
  echo "Usage: ./run_ld_test.sh <checkpoint>"
  exit 1
fi
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "============================================"
echo "Reverse Model Test Evaluation"
echo "============================================"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Device / Batch / Steps: (from config.py)"
echo "============================================"
echo ""

python reverse_evaluation.py --checkpoint "${CHECKPOINT}"