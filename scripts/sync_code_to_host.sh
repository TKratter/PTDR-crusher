#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/sync_code_to_host.sh <host>

Sync code and small config files to a remote PTDR checkout without copying the
dataset or local training artifacts. The destination defaults to ~/PTDR/.

Environment variables:
  REMOTE_DIR   Remote checkout path. Default: ~/PTDR
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
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_PASSWORD_VALUE="${SSH_PASSWORD:-${SSHPASS:-}}"

RSYNC_CMD=(rsync -az --delete)
if [[ -n "$SSH_PASSWORD_VALUE" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required when SSH_PASSWORD or SSHPASS is set." >&2
    exit 1
  fi
  export SSHPASS="$SSH_PASSWORD_VALUE"
  RSYNC_CMD+=( -e "sshpass -e ssh" )
fi

"${RSYNC_CMD[@]}" \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '**/__pycache__' \
  --exclude 'work_dirs' \
  --exclude 'work_dirs/**' \
  --exclude '.dist_test' \
  --exclude '.dist_test/**' \
  --exclude 'dataset' \
  --exclude 'dataset/**' \
  --exclude 'experiments/ptdr/manifests' \
  --exclude 'experiments/ptdr/manifests/**' \
  "$REPO_ROOT/" \
  "$HOST:$REMOTE_DIR/"
