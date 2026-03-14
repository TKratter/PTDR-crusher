#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_dbnetpp_two_stage.sh [stage1_config] [stage2_config] [run_prefix]

Runs DBNet++ in two stages:
1. Clean PTDR-only training from the SynthText-pretrained DBNet++ detector
2. PTDR-only hard-condition fine-tuning from the best stage-1 checkpoint

Defaults:
  stage1_config = experiments/ptdr/configs/dbnetpp_multidata_clean_stage.yaml
  stage2_config = experiments/ptdr/configs/dbnetpp_multidata_hard_finetune_stage.yaml
  run_prefix    = dbnetpp-two-stage-<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_CONFIG="${1:-experiments/ptdr/configs/dbnetpp_multidata_clean_stage.yaml}"
STAGE2_CONFIG="${2:-experiments/ptdr/configs/dbnetpp_multidata_hard_finetune_stage.yaml}"
RUN_PREFIX="${3:-dbnetpp-two-stage-$(date +%Y%m%d-%H%M%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29561}"

cd "${REPO_DIR}"

stage_work_dir() {
  python - <<'PY' "$1"
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(cfg["training"]["work_dir"])
PY
}

find_stage_best_checkpoint() {
  python - <<'PY' "$1"
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
best_candidates = sorted(work_dir.glob("best_icdar_hmean_epoch_*.pth"), key=lambda path: path.stat().st_mtime)
if best_candidates:
    print(best_candidates[-1])
    raise SystemExit(0)

latest = work_dir / "latest.pth"
if latest.exists():
    print(latest)
    raise SystemExit(0)

epoch_candidates = sorted(work_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
if epoch_candidates:
    print(epoch_candidates[-1])
    raise SystemExit(0)

raise SystemExit(f"No DBNet checkpoint found under {work_dir}")
PY
}

find_stage_resume_checkpoint() {
  python - <<'PY' "$1"
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
last_checkpoint = work_dir / "last_checkpoint"
if last_checkpoint.exists():
    checkpoint_path = Path(last_checkpoint.read_text(encoding="utf-8").strip())
    if checkpoint_path.exists():
        print(checkpoint_path)
        raise SystemExit(0)

epoch_candidates = sorted(
    work_dir.glob("epoch_*.pth"),
    key=lambda path: path.stat().st_mtime,
)
if epoch_candidates:
    print(epoch_candidates[-1])
    raise SystemExit(0)

raise SystemExit(1)
PY
}

run_stage() {
  local config_path="$1"
  shift
  local extra_args=("$@")
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun \
      --nnodes 1 \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --rdzv_backend c10d \
      --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
      experiments/ptdr/train_dbnetpp.py \
      --config_path "${config_path}" \
      "${extra_args[@]}"
  else
    python experiments/ptdr/train_dbnetpp.py \
      --config_path "${config_path}" \
      "${extra_args[@]}"
  fi
}

STAGE1_WORK_DIR="$(stage_work_dir "${STAGE1_CONFIG}")"
STAGE2_WORK_DIR="$(stage_work_dir "${STAGE2_CONFIG}")"
mkdir -p "${STAGE1_WORK_DIR}" "${STAGE2_WORK_DIR}"

echo "[two-stage] stage 1 config: ${STAGE1_CONFIG}"
if STAGE1_RESUME_CKPT="$(find_stage_resume_checkpoint "${STAGE1_WORK_DIR}" 2>/dev/null)"; then
  echo "[two-stage] stage 1 resume checkpoint: ${STAGE1_RESUME_CKPT}"
  run_stage "${STAGE1_CONFIG}" \
    --training.resume true \
    --mmocr.load_from "${STAGE1_RESUME_CKPT}" \
    --wandb.run_name "${RUN_PREFIX}-stage1"
else
  run_stage "${STAGE1_CONFIG}" \
    --training.resume false \
    --wandb.run_name "${RUN_PREFIX}-stage1"
fi

STAGE1_CKPT="$(find_stage_best_checkpoint "${STAGE1_WORK_DIR}")"
echo "[two-stage] stage 1 checkpoint: ${STAGE1_CKPT}"

echo "[two-stage] stage 2 config: ${STAGE2_CONFIG}"
if STAGE2_RESUME_CKPT="$(find_stage_resume_checkpoint "${STAGE2_WORK_DIR}" 2>/dev/null)"; then
  echo "[two-stage] stage 2 resume checkpoint: ${STAGE2_RESUME_CKPT}"
  run_stage "${STAGE2_CONFIG}" \
    --training.resume true \
    --mmocr.load_from "${STAGE2_RESUME_CKPT}" \
    --wandb.run_name "${RUN_PREFIX}-stage2"
else
  run_stage "${STAGE2_CONFIG}" \
    --training.resume false \
    --mmocr.load_from "${STAGE1_CKPT}" \
    --wandb.run_name "${RUN_PREFIX}-stage2"
fi
