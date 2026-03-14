#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_parseq_two_stage.sh [stage1_config] [stage2_config] [run_prefix]

Runs PARSeq in two stages:
1. PTDR-Synth pretraining
2. Fine-tuning on the mixed all-data config, initialized from the best stage-1 checkpoint

Defaults:
  stage1_config = experiments/ptdr/configs/parseq_ptdr_synth_pretrain.yaml
  stage2_config = experiments/ptdr/configs/parseq_multidata_all_norm_v2_hard_from_ptdr_synth.yaml
  run_prefix    = parseq-two-stage-<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_CONFIG="${1:-experiments/ptdr/configs/parseq_ptdr_synth_pretrain.yaml}"
STAGE2_CONFIG="${2:-experiments/ptdr/configs/parseq_multidata_all_norm_v2_hard_from_ptdr_synth.yaml}"
RUN_PREFIX="${3:-parseq-two-stage-$(date +%Y%m%d-%H%M%S)}"

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
ckpt_dir = work_dir / "checkpoints"
best_candidates = sorted(
    [path for path in ckpt_dir.glob("epoch=*.ckpt") if path.name != "last.ckpt"],
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

STAGE1_WORK_DIR="$(stage_work_dir "${STAGE1_CONFIG}")"
STAGE2_WORK_DIR="$(stage_work_dir "${STAGE2_CONFIG}")"
mkdir -p "${STAGE1_WORK_DIR}" "${STAGE2_WORK_DIR}"

echo "[two-stage] stage 1 config: ${STAGE1_CONFIG}"
python experiments/ptdr/train_parseq.py \
  --config_path "${STAGE1_CONFIG}" \
  --wandb.run_name "${RUN_PREFIX}-stage1"

STAGE1_CKPT="$(find_stage_best_checkpoint "${STAGE1_WORK_DIR}")"
echo "[two-stage] stage 1 checkpoint: ${STAGE1_CKPT}"

echo "[two-stage] stage 2 config: ${STAGE2_CONFIG}"
python experiments/ptdr/train_parseq.py \
  --config_path "${STAGE2_CONFIG}" \
  --model.init_from_checkpoint "${STAGE1_CKPT}" \
  --wandb.run_name "${RUN_PREFIX}-stage2"
