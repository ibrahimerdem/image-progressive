#!/usr/bin/env bash
set -e

EPOCHS=${1:-100}
RETRAIN=${2:-0}
CHECKPOINT=${3:-}
NAME=${4:-conditional_vae}
VERSION=${5:-single}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DEVICE_IDS=$(python - <<'EOF'
import config
ids = getattr(config, 'DEVICE_IDS', [0])
print(','.join(str(i) for i in ids))
EOF
)

export CUDA_VISIBLE_DEVICES="${DEVICE_IDS}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "Starting VAE training for ${EPOCHS} epochs (retrain=${RETRAIN}, checkpoint=${CHECKPOINT})"

EXTRA_ARGS=("--name" "${NAME}" "--version" "${VERSION}")
if [ -n "${CHECKPOINT}" ]; then
  EXTRA_ARGS+=("--checkpoint" "${CHECKPOINT}")
fi

python feature_vae_trainer.py --epochs "${EPOCHS}" --retrain "${RETRAIN}" "${EXTRA_ARGS[@]}"
