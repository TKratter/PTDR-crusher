#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_dbnetpp_remote.sh <host> [run_name]

Launch DBNet++ training on a remote PTDR runner inside a tmux session named
"ptdr". The remote checkout is assumed to exist at ~/PTDR unless REMOTE_DIR is
overridden.

Environment variables:
  REMOTE_DIR   Remote checkout path. Default: ~/PTDR
  CONDA_ENV    Conda environment name. Default: ptdr
  CONFIG_PATH  Remote DBNet++ config path. Default: experiments/ptdr/configs/dbnetpp.yaml
  LOG_PATH     Remote log file path. Default: <REMOTE_DIR>/work_dirs/dbnetpp_remote.log
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
CONFIG_PATH="${CONFIG_PATH:-experiments/ptdr/configs/dbnetpp.yaml}"
LOG_PATH="${LOG_PATH:-$REMOTE_DIR/work_dirs/dbnetpp_remote.log}"
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

tmux kill-session -t ptdr >/dev/null 2>&1 || true
echo \"Logging to: $LOG_PATH\"
tmux new-session -d -s ptdr \"cd $REMOTE_DIR && source \\\$(conda info --base)/etc/profile.d/conda.sh && set +u && conda activate $CONDA_ENV && set -u && python experiments/ptdr/train_dbnetpp.py --config_path $CONFIG_PATH $RUN_NAME_ARG >> $LOG_PATH 2>&1\"
tmux list-sessions
'"

echo "Attached command:"
echo "  ssh $HOST"
echo "  tmux attach -t ptdr"
echo "Log file:"
echo "  ssh $HOST"
echo "  tail -n 200 $LOG_PATH"
