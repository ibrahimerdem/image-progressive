#!/usr/bin/env bash
set -e

# Script to run test evaluation or inference
# Usage examples:
#   ./run_test.sh test <checkpoint_path>
#   ./run_test.sh inference <checkpoint_path> <input_image> <features>

MODE=${1:-test}
CHECKPOINT=${2:-checkpoints/multimodal_basic_ddp_epoch_10.pth}
INPUT_IMAGE=${3:-}
FEATURES=${4:-}
OUTPUT=${5:-output.png}
DEVICE=${6:-cuda:0}
BATCH_SIZE=${7:-16}
NUM_SAMPLES=${8:-1}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Running in ${MODE} mode..."
echo "Checkpoint: ${CHECKPOINT}"
echo "Device: ${DEVICE}"

if [ "${MODE}" == "test" ]; then
  echo "Evaluating on test dataset..."
  python evaluation.py \
    --mode test \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}"

elif [ "${MODE}" == "inference" ]; then
  if [ -z "${INPUT_IMAGE}" ]; then
    echo "Error: INPUT_IMAGE is required for inference mode"
    echo "Usage: ./run_test.sh inference <checkpoint> <input_image> <features> [output] [device] [batch_size] [num_samples]"
    echo "Example: ./run_test.sh inference checkpoints/model_epoch_10.pth data/initial/image.png \"20,30,50,25,0.5,-10,2,3,5000\""
    exit 1
  fi
  
  if [ -z "${FEATURES}" ]; then
    echo "Error: FEATURES is required for inference mode"
    echo "Features should be comma-separated values matching config.FEATURE_COLUMNS"
    echo "Expected features: yarn_number,frequency,fabric_elasticity,cielab_l_raw,cielab_a_raw,cielab_b_raw,bleaching,duration,concentration"
    exit 1
  fi
  
  echo "Performing inference..."
  echo "Input image: ${INPUT_IMAGE}"
  echo "Features: ${FEATURES}"
  echo "Output: ${OUTPUT}"
  echo "Number of samples: ${NUM_SAMPLES}"
  
  python evaluation.py \
    --mode inference \
    --checkpoint "${CHECKPOINT}" \
    --input_image "${INPUT_IMAGE}" \
    --features "${FEATURES}" \
    --output "${OUTPUT}" \
    --device "${DEVICE}" \
    --num_samples "${NUM_SAMPLES}"

else
  echo "Error: Invalid mode '${MODE}'"
  echo "Valid modes: test, inference"
  exit 1
fi

echo "Done!"
