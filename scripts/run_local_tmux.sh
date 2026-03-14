#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_local_tmux.sh <session_name> <trainer> <config_path> [run_name] [-- extra args]

Launch a local PTDR training job inside a detached tmux session and make sure
the configured work_dir exists before stdout/stderr are redirected.

Arguments:
  session_name  tmux session name to create
  trainer       One of: dbnet, parseq
  config_path   Repo-relative config path, e.g. experiments/ptdr/configs/dbnet_r50.yaml
  run_name      Optional wandb run name override
  extra args    Optional trainer CLI overrides passed through after run_name

Environment variables:
  CONDA_ENV     Conda environment name. Default: ptdr
  REPO_DIR      Repo checkout path. Default: current working directory
  GPU_DEVICES   Optional comma-separated GPU list, e.g. 1,2
  NPROC_PER_NODE  Number of processes for torchrun. Default: 1
  MASTER_PORT   Rendezvous port for torchrun. Default: derived from session name
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 1
fi

SESSION_NAME="$1"
TRAINER="$2"
CONFIG_PATH="$3"
RUN_NAME="${4:-}"
if [[ $# -gt 4 ]]; then
  shift 4
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi
CONDA_ENV="${CONDA_ENV:-ptdr}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
GPU_DEVICES="${GPU_DEVICES:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SESSION_HASH="$(printf '%s' "$SESSION_NAME" | cksum | awk '{print $1}')"
MASTER_PORT_DEFAULT="$((20000 + SESSION_HASH % 20000))"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"

case "$TRAINER" in
  dbnet)
    ENTRYPOINT="experiments/ptdr/train_dbnetpp.py"
    ;;
  parseq)
    ENTRYPOINT="experiments/ptdr/train_parseq.py"
    ;;
  *)
    echo "Unsupported trainer: $TRAINER" >&2
    exit 1
    ;;
esac

cd "$REPO_DIR"
WORK_DIR="$(python - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
print(cfg["training"]["work_dir"])
PY
)"

mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/wandb/cache" "$WORK_DIR/wandb/config" "$WORK_DIR/wandb/data" "$WORK_DIR/wandb/artifacts" "$WORK_DIR/wandb/tmp"
LOG_PATH="$WORK_DIR/train.log"

LAUNCH_CMD=()
if [[ "$TRAINER" == "dbnet" && "$NPROC_PER_NODE" -gt 1 ]]; then
  LAUNCH_CMD=(
    torchrun
    --nnodes 1
    --nproc_per_node "$NPROC_PER_NODE"
    --rdzv_backend c10d
    --rdzv_endpoint "127.0.0.1:$MASTER_PORT"
    "$ENTRYPOINT"
    --config_path "$CONFIG_PATH"
  )
else
  LAUNCH_CMD=(
    python
    "$ENTRYPOINT"
    --config_path "$CONFIG_PATH"
  )
fi

if [[ -n "$RUN_NAME" ]]; then
  LAUNCH_CMD+=(--wandb.run_name "$RUN_NAME")
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

printf -v LAUNCH_CMD_STR '%q ' "${LAUNCH_CMD[@]}"

GPU_EXPORT_CMD=""
if [[ -n "$GPU_DEVICES" ]]; then
  GPU_EXPORT_CMD="export CUDA_VISIBLE_DEVICES='$GPU_DEVICES' && "
fi

tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true
tmux new-session -d -s "$SESSION_NAME" \
  "trap '' HUP && cd '$REPO_DIR' && export WANDB_DIR='$WORK_DIR/wandb' WANDB_CACHE_DIR='$WORK_DIR/wandb/cache' WANDB_CONFIG_DIR='$WORK_DIR/wandb/config' WANDB_DATA_DIR='$WORK_DIR/wandb/data' WANDB_ARTIFACT_DIR='$WORK_DIR/wandb/artifacts' TMPDIR='$WORK_DIR/wandb/tmp' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 && ${GPU_EXPORT_CMD}source \$(conda info --base)/etc/profile.d/conda.sh && set +u && conda activate '$CONDA_ENV' && set -u && ${LAUNCH_CMD_STR}>> '$LOG_PATH' 2>&1; STATUS=\$?; echo \"process exited with status \$STATUS\" >> '$LOG_PATH'; exec bash"
tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null

echo "Session: $SESSION_NAME"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_PATH"
echo "GPUs: ${GPU_DEVICES:-default}"
echo "Processes: $NPROC_PER_NODE"
if [[ "$TRAINER" == "dbnet" && "$NPROC_PER_NODE" -gt 1 ]]; then
  echo "Master port: $MASTER_PORT"
fi
