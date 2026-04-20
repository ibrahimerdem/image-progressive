#!/usr/bin/env bash
set -e

EPOCHS=${1:-100}
RESUME=${2:-}

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

echo "Launching VAE retraining for ${EPOCHS} epochs (resume=${RESUME})"

EXTRA_ARGS=()
if [ -n "${RESUME}" ]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

python vae_trainer.py --epochs "${EPOCHS}" "${EXTRA_ARGS[@]}"