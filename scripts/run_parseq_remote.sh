#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_parseq_remote.sh <host> [run_name]

Launch PARSeq training on a remote PTDR runner inside a tmux session. The
script picks currently idle GPUs on the remote host, constrains the process to
those devices with CUDA_VISIBLE_DEVICES, and passes the selected device count to
the PARSeq trainer.

Environment variables:
  REMOTE_DIR         Remote checkout path. Default: ~/PTDR
  CONDA_ENV          Conda environment name. Default: ptdr
  CONFIG_PATH        Remote PARSeq config path. Default: experiments/ptdr/configs/parseq.yaml
  SESSION_NAME       tmux session name. Default: ptdr-parseq
  LOG_PATH           Remote log file path. Default: <REMOTE_DIR>/work_dirs/parseq_remote.log
  GPU_COUNT          Number of idle GPUs to claim. Default: 1
  MAX_USED_MEM_MB    Max memory already in use for a GPU to be considered idle. Default: 512
  MAX_UTILIZATION    Max GPU utilization percent for a GPU to be considered idle. Default: 10
  SSH_PASSWORD or SSHPASS
               Optional SSH password for non-interactive login via sshpass.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

HOST="$1"
RUN_NAME="${2:-}"
REMOTE_DIR="${REMOTE_DIR:-~/PTDR}"
CONDA_ENV="${CONDA_ENV:-ptdr}"
CONFIG_PATH="${CONFIG_PATH:-experiments/ptdr/configs/parseq.yaml}"
SESSION_NAME="${SESSION_NAME:-ptdr-parseq}"
LOG_PATH="${LOG_PATH:-$REMOTE_DIR/work_dirs/parseq_remote.log}"
GPU_COUNT="${GPU_COUNT:-1}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-512}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
SSH_PASSWORD_VALUE="${SSH_PASSWORD:-${SSHPASS:-}}"

RUN_NAME_ARG=""
if [[ -n "$RUN_NAME" ]]; then
  RUN_NAME_ARG="--wandb.run_name $RUN_NAME"
fi

SSH_CMD=(ssh)
if [[ -n "$SSH_PASSWORD_VALUE" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required when SSH_PASSWORD or SSHPASS is set." >&2
    exit 1
  fi
  export SSHPASS="$SSH_PASSWORD_VALUE"
  SSH_CMD=(sshpass -e ssh)
fi

"${SSH_CMD[@]}" "$HOST" "bash -lc '
set -euo pipefail
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
set +u
conda activate \"$CONDA_ENV\"
set -u
cd \"$REMOTE_DIR\"
mkdir -p \"\$(dirname \"$LOG_PATH\")\"

AVAILABLE_GPUS=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F\", *\" '\''(\$2 + 0) <= '"$MAX_USED_MEM_MB"' && (\$3 + 0) <= '"$MAX_UTILIZATION"' {print \$1}'\'' | head -n \"$GPU_COUNT\" | paste -sd, -)

if [[ -z \"\$AVAILABLE_GPUS\" ]]; then
  echo \"No idle GPUs found on $HOST (thresholds: mem<='"$MAX_USED_MEM_MB"'MB, util<='"$MAX_UTILIZATION"'%).\" >&2
  exit 1
fi

SELECTED_COUNT=\$(awk -F, '\''{print NF}'\'' <<<\"\$AVAILABLE_GPUS\")
echo \"Using GPU(s) on $HOST: \$AVAILABLE_GPUS\"
echo \"Logging to: $LOG_PATH\"

tmux kill-session -t \"$SESSION_NAME\" >/dev/null 2>&1 || true
tmux new-session -d -s \"$SESSION_NAME\" \"cd $REMOTE_DIR && source \\\$(conda info --base)/etc/profile.d/conda.sh && set +u && conda activate $CONDA_ENV && set -u && export CUDA_VISIBLE_DEVICES=\$AVAILABLE_GPUS && python experiments/ptdr/train_parseq.py --config_path $CONFIG_PATH --training.accelerator gpu --training.devices \$SELECTED_COUNT $RUN_NAME_ARG >> $LOG_PATH 2>&1\"
tmux list-sessions
'"

echo "Attached command:"
echo "  ssh $HOST"
echo "  tmux attach -t $SESSION_NAME"
echo "Log file:"
echo "  ssh $HOST"
echo "  tail -n 200 $LOG_PATH"
