#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_remote.sh <host>

Bootstrap a remote PTDR runner over SSH. This script assumes the remote host
already has SSH access, a usable shell, and conda installed.

Environment variables:
  REMOTE_DIR   Remote checkout path. Default: ~/PTDR
  CONDA_ENV    Conda environment name. Default: ptdr
  PYTHON_VER   Python version for conda. Default: 3.10
  REPO_URL     Git URL to clone if REMOTE_DIR does not exist.
               Default: local remote.origin.url when available, otherwise
               https://github.com/zobeirraisi/PTDR.git
  SSH_PASSWORD or SSHPASS
               Optional SSH password for non-interactive login via sshpass.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

HOST="$1"
REMOTE_DIR="${REMOTE_DIR:-~/PTDR}"
CONDA_ENV="${CONDA_ENV:-ptdr}"
PYTHON_VER="${PYTHON_VER:-3.10}"
DEFAULT_REPO_URL="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" config --get remote.origin.url 2>/dev/null || true)"
REPO_URL="${REPO_URL:-${DEFAULT_REPO_URL:-https://github.com/zobeirraisi/PTDR.git}}"
SSH_PASSWORD_VALUE="${SSH_PASSWORD:-${SSHPASS:-}}"

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

if ! command -v git >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y git tmux rsync
fi

if [[ ! -d \"$REMOTE_DIR/.git\" ]]; then
  git clone \"$REPO_URL\" \"$REMOTE_DIR\"
fi

cd \"$REMOTE_DIR\"

if ! command -v conda >/dev/null 2>&1; then
  echo \"conda is not installed on $HOST\" >&2
  exit 1
fi

source \"\$(conda info --base)/etc/profile.d/conda.sh\"

if ! conda env list | awk '\''{print \$1}'\'' | grep -qx \"$CONDA_ENV\"; then
  conda create -n \"$CONDA_ENV\" python=\"$PYTHON_VER\" -y
fi

set +u
conda activate \"$CONDA_ENV\"
set -u
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

python -c \"import numpy, torch, mmcv, mmengine, mmdet, mmocr; print(numpy.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(mmcv.__version__); print(mmdet.__version__); print(mmocr.__version__)\"
'"
