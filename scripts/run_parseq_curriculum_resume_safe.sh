#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LOG_PATH:-${REPO_DIR}/work_dirs/parseq_curriculum_resume_2gpu_safe.log}"
LOCK_PATH="${LOCK_PATH:-${REPO_DIR}/work_dirs/parseq_curriculum_resume_2gpu_safe.lock}"
PHASE2_CONFIG="${PHASE2_CONFIG:-experiments/ptdr/configs/parseq_curriculum_p2.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-experiments/ptdr/configs/parseq_curriculum_p3.yaml}"
PHASE2_CKPT="${PHASE2_CKPT:-${REPO_DIR}/work_dirs/pqcur_p2/checkpoints/last.ckpt}"
PHASE2_RUN_NAME="${PHASE2_RUN_NAME:-parseq-curriculum-resume-p2-2gpu-safe}"
PHASE3_RUN_NAME="${PHASE3_RUN_NAME:-parseq-curriculum-resume-p3-2gpu-safe}"

cd "${REPO_DIR}"

mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[safe-resume] another PARSeq resume process already holds ${LOCK_PATH}, exiting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
CONDA_ENV="${CONDA_ENV:-ptdr}"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

find_phase2_best_checkpoint() {
  python - <<'PY' "${REPO_DIR}"
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
ckpt_dir = repo_dir / "work_dirs" / "pqcur_p2" / "checkpoints"
preferred = sorted(ckpt_dir.glob("best-val-*.ckpt"), key=lambda path: path.stat().st_mtime)
print(preferred[-1] if preferred else (ckpt_dir / "last.ckpt"))
PY
}

run_phase2() {
  python experiments/ptdr/train_parseq.py \
    --config_path "${PHASE2_CONFIG}" \
    --training.devices 2 \
    --training.resume_from_checkpoint "${PHASE2_CKPT}" \
    --wandb.run_name "${PHASE2_RUN_NAME}" \
    >> "${LOG_PATH}" 2>&1
}

run_phase3() {
  local phase2_best_ckpt
  phase2_best_ckpt="$(find_phase2_best_checkpoint)"
  python experiments/ptdr/train_parseq.py \
    --config_path "${PHASE3_CONFIG}" \
    --training.devices 2 \
    --model.init_from_checkpoint "${phase2_best_ckpt}" \
    --wandb.run_name "${PHASE3_RUN_NAME}" \
    >> "${LOG_PATH}" 2>&1
}

echo "[safe-resume] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
run_phase2
PHASE2_STATUS=$?
echo "phase2 exited with status ${PHASE2_STATUS}" >> "${LOG_PATH}"
if [[ ${PHASE2_STATUS} -ne 0 ]]; then
  exit "${PHASE2_STATUS}"
fi

run_phase3
PHASE3_STATUS=$?
echo "phase3 exited with status ${PHASE3_STATUS}" >> "${LOG_PATH}"
exit "${PHASE3_STATUS}"
