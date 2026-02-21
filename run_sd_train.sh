#!/usr/bin/env bash
set -e

STEPS=${1:-100000}
RETRAIN=${2:-0}
CHECKPOINT=${3:-}

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

if [ -z "${DEVICE_IDS}" ]; then
  echo "No DEVICE_IDS found in config.py"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${DEVICE_IDS}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "Launching Stable Diffusion training for ${STEPS} steps (retrain=${RETRAIN}, checkpoint=${CHECKPOINT})"

EXTRA_ARGS=()
if [ -n "${CHECKPOINT}" ]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

python sd_trainer.py --steps "${STEPS}" --retrain "${RETRAIN}" "${EXTRA_ARGS[@]}"
