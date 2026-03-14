#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE_RUNNER="${REPO_DIR}/scripts/run_parseq_curriculum_resume_safe.sh"
LOG_PATH="${LOG_PATH:-${REPO_DIR}/work_dirs/parseq_curriculum_watchdog.log}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
RESTART_COUNT=0

cd "${REPO_DIR}"

phase_completed() {
  rg -q "phase3 exited with status 0" "${REPO_DIR}/work_dirs/parseq_curriculum_resume_2gpu_safe.log" 2>/dev/null
}

parseq_running() {
  pgrep -f "run_parseq_curriculum_resume_safe.sh|train_parseq.py --config_path experiments/ptdr/configs/parseq_curriculum_p2.yaml|train_parseq.py --config_path experiments/ptdr/configs/parseq_curriculum_p3.yaml" >/dev/null 2>&1
}

echo "[watchdog] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"

while true; do
  if phase_completed; then
    echo "[watchdog] phase 3 already completed, stopping at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
    exit 0
  fi

  if parseq_running; then
    echo "[watchdog] parseq alive at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  if [[ ${RESTART_COUNT} -ge ${MAX_RESTARTS} ]]; then
    echo "[watchdog] reached restart limit ${MAX_RESTARTS}, giving up at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
    exit 1
  fi

  RESTART_COUNT=$((RESTART_COUNT + 1))
  echo "[watchdog] restart ${RESTART_COUNT}/${MAX_RESTARTS} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
  bash "${SAFE_RUNNER}" >> "${LOG_PATH}" 2>&1 || true
  sleep "${SLEEP_SECONDS}"
done
