#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_SUMMARY="$ROOT/experiments/ptdr/manifests/crop_rotation_jitter_bank_exhaustive_v1/summary.json"
LOG_FILE="$ROOT/work_dirs/crx4/jitter_bank_switch.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

echo "watching for jitter bank at $(date -u +%FT%TZ)" >> "$LOG_FILE"
while [[ ! -f "$BANK_SUMMARY" ]]; do
  sleep 30
done

echo "bank ready at $(date -u +%FT%TZ)" >> "$LOG_FILE"
bash "$ROOT/scripts/switch_crop_rotation_to_jitter_bank.sh" >> "$LOG_FILE" 2>&1
