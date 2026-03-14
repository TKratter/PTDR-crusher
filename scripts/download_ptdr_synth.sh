#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_ROOT="${REPO_ROOT}/dataset/synth"
ARCHIVE_PATH="${OUT_ROOT}/PTDR-SYNTH.zip"
EXTRACT_ROOT="${OUT_ROOT}"
CANONICAL_LINK="${OUT_ROOT}/ptdr_synth"
URL="https://www.dropbox.com/scl/fi/i48a447j6d9dqgwso9qyz/PTDR-SYNTH.zip?rlkey=uggcfeghsae0gcugdx99vvuot&st=krm7n0dp&dl=1"

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  wget -O "${ARCHIVE_PATH}" "${URL}"
fi

if [[ ! -f "${OUT_ROOT}/PTDR-SYNTH/gt.txt" && ! -f "${OUT_ROOT}/PTDR-SYNTH/PTDR-SYNTH/gt.txt" ]]; then
  unzip -q -o "${ARCHIVE_PATH}" -d "${EXTRACT_ROOT}"
fi

TARGET_ROOT="${OUT_ROOT}/PTDR-SYNTH"
if [[ -f "${OUT_ROOT}/PTDR-SYNTH/PTDR-SYNTH/gt.txt" ]]; then
  TARGET_ROOT="${OUT_ROOT}/PTDR-SYNTH/PTDR-SYNTH"
fi

ln -sfn "${TARGET_ROOT}" "${CANONICAL_LINK}"

echo "PTDR-Synth ready at ${CANONICAL_LINK}"
