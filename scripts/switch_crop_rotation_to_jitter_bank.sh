#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="experiments/ptdr/configs/crop_rotation_classifier_128_jitter_exhaustive_fast_bank.yaml"
BANK_SUMMARY="$ROOT/experiments/ptdr/manifests/crop_rotation_jitter_bank_exhaustive_v1/summary.json"
WORK_DIR="$ROOT/work_dirs/crx4"
CHECKPOINT="$WORK_DIR/checkpoints/best-00.ckpt"
SESSION="crop-rot-fast"
LOG_FILE="$WORK_DIR/train_bank_resume_2gpu.log"
RUN_NAME="crop-rotation-128-jitter-exhaustive-fast-bank-resume-2gpu"
CONDA_ENV="${CONDA_ENV:-ptdr}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$ROOT"

if [[ ! -f "$BANK_SUMMARY" ]]; then
  echo "bank summary not found: $BANK_SUMMARY" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "train_crop_rotation_classifier.py.*crop_rotation_classifier_128_jitter_exhaustive_fast.yaml" 2>/dev/null || true
pkill -f "train_crop_rotation_classifier.py.*crop_rotation_classifier_128_jitter_exhaustive_fast_bank.yaml" 2>/dev/null || true
sleep 2

tmux new-session -d -s "$SESSION" \
  "trap '' HUP && cd $ROOT && export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 && \
   source \$(conda info --base)/etc/profile.d/conda.sh && set +u && conda activate $CONDA_ENV && set -u && \
   python experiments/ptdr/train_crop_rotation_classifier.py \
     --config_path $CONFIG \
     --training.devices 2 \
     --training.resume_from_checkpoint $CHECKPOINT \
     --wandb.run_name $RUN_NAME \
     >> $LOG_FILE 2>&1; \
   STATUS=\$?; echo \"process exited with status \$STATUS\" >> $LOG_FILE; exec bash"

echo "switched crop rotation training to jitter bank using $CHECKPOINT"
