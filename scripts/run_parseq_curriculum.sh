#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_parseq_curriculum.sh [phase1_config] [phase2_config] [phase3_config] [run_prefix]

Runs PARSeq in three stages:
1. PTDR-heavier clean multidataset training
2. PTDR-dominant in-domain adaptation with detector-style crop jitter
3. Mild robustness fine-tuning

Defaults:
  phase1_config = experiments/ptdr/configs/parseq_curriculum_p1.yaml
  phase2_config = experiments/ptdr/configs/parseq_curriculum_p2.yaml
  phase3_config = experiments/ptdr/configs/parseq_curriculum_p3.yaml
  run_prefix    = parseq-curriculum-<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE1_CONFIG="${1:-experiments/ptdr/configs/parseq_curriculum_p1.yaml}"
PHASE2_CONFIG="${2:-experiments/ptdr/configs/parseq_curriculum_p2.yaml}"
PHASE3_CONFIG="${3:-experiments/ptdr/configs/parseq_curriculum_p3.yaml}"
RUN_PREFIX="${4:-parseq-curriculum-$(date +%Y%m%d-%H%M%S)}"

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

find_stage_best_val_checkpoint() {
  python - <<'PY' "$1"
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
ckpt_dir = work_dir / "checkpoints"
best_candidates = sorted(
    ckpt_dir.glob("best-val-*.ckpt"),
    key=lambda path: path.stat().st_mtime,
)
if best_candidates:
    print(best_candidates[-1])
    raise SystemExit(0)
last_ckpt = ckpt_dir / "last.ckpt"
if last_ckpt.exists():
    print(last_ckpt)
    raise SystemExit(0)
raise SystemExit(f"No checkpoint found under {ckpt_dir}")
PY
}

PHASE1_WORK_DIR="$(stage_work_dir "${PHASE1_CONFIG}")"
PHASE2_WORK_DIR="$(stage_work_dir "${PHASE2_CONFIG}")"
PHASE3_WORK_DIR="$(stage_work_dir "${PHASE3_CONFIG}")"
mkdir -p "${PHASE1_WORK_DIR}" "${PHASE2_WORK_DIR}" "${PHASE3_WORK_DIR}"

echo "[curriculum] phase 1 config: ${PHASE1_CONFIG}"
python experiments/ptdr/train_parseq.py \
  --config_path "${PHASE1_CONFIG}" \
  --wandb.run_name "${RUN_PREFIX}-p1"

PHASE1_CKPT="$(find_stage_best_val_checkpoint "${PHASE1_WORK_DIR}")"
echo "[curriculum] phase 1 checkpoint: ${PHASE1_CKPT}"

echo "[curriculum] phase 2 config: ${PHASE2_CONFIG}"
python experiments/ptdr/train_parseq.py \
  --config_path "${PHASE2_CONFIG}" \
  --model.init_from_checkpoint "${PHASE1_CKPT}" \
  --wandb.run_name "${RUN_PREFIX}-p2"

PHASE2_CKPT="$(find_stage_best_val_checkpoint "${PHASE2_WORK_DIR}")"
echo "[curriculum] phase 2 checkpoint: ${PHASE2_CKPT}"

echo "[curriculum] phase 3 config: ${PHASE3_CONFIG}"
python experiments/ptdr/train_parseq.py \
  --config_path "${PHASE3_CONFIG}" \
  --model.init_from_checkpoint "${PHASE2_CKPT}" \
  --wandb.run_name "${RUN_PREFIX}-p3"
