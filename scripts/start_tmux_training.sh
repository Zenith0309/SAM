#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${1:-sam-train}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/train_sam_decoder_bdd100k.py"
LOG_DIR="${REPO_ROOT}/logs"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-SAM}

mkdir -p "${LOG_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please ensure conda is installed and on PATH." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
CONDA_ACTIVATE="source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate \"${CONDA_ENV_NAME}\""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please install tmux before running this script." >&2
  exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "Training script not found at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1; then
  echo "tmux session '${SESSION_NAME}' already exists. Attaching now..."
  tmux attach -t "${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -c "${REPO_ROOT}"

TRAIN_CMD="${CONDA_ACTIVATE} && cd \"${REPO_ROOT}\" && \\
python train_sam_decoder_bdd100k.py \\
  --distributed \\
  --world-size 4 \\
  --num-epochs 100 \\
  --debug-train-samples 128 \\
  --debug-val-samples 64 \\
  --max-classes-per-image 6 \\
  2>&1 | tee \"${LOG_FILE}\""
tmux send-keys -t "${SESSION_NAME}:0" \
  "${TRAIN_CMD}; echo '[tmux] Training finished. Logs saved to ${LOG_FILE}'; exec bash" C-m

tmux new-window -t "${SESSION_NAME}" -n "gpu"
GPU_CMD="${CONDA_ACTIVATE} && watch -n1 nvidia-smi"
tmux send-keys -t "${SESSION_NAME}:1" "${GPU_CMD}" C-m

echo "Started tmux session '${SESSION_NAME}'. Logs: ${LOG_FILE}"
echo "Window 0: training (logs tee'd). Window 1: GPU monitor."
echo "Use 'tmux attach -t ${SESSION_NAME}' to attach, or 'tmux kill-session -t ${SESSION_NAME}' to stop it."
