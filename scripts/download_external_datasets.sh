#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/dataset/external"
CACHE_ROOT="${DATA_ROOT}/_downloads"
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ptdr
set -u

mkdir -p "${DATA_ROOT}" "${CACHE_ROOT}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

download_file() {
  local url="$1"
  local out="$2"
  local tmp="${out}.part"
  if [[ -f "${out}" ]]; then
    if [[ "${out}" == *.zip ]]; then
      if unzip -tqq "${out}" >/dev/null 2>&1; then
        echo "Using cached $(basename "${out}")"
        return
      fi
      echo "Discarding invalid cached zip $(basename "${out}")"
      rm -f "${out}"
    elif [[ -s "${out}" ]]; then
      echo "Using cached $(basename "${out}")"
      return
    else
      rm -f "${out}"
    fi
  fi
  rm -f "${tmp}"
  if ! wget -c -O "${tmp}" "${url}"; then
    rm -f "${tmp}"
    return 1
  fi
  mv "${tmp}" "${out}"
}

download_gdrive() {
  local file_id="$1"
  local out="$2"
  local tmp="${out}.part"
  if [[ -f "${out}" ]]; then
    if [[ "${out}" == *.zip ]] && unzip -tqq "${out}" >/dev/null 2>&1; then
      echo "Using cached $(basename "${out}")"
      return
    fi
    if [[ "${out}" != *.zip && -s "${out}" ]]; then
      echo "Using cached $(basename "${out}")"
      return
    fi
    echo "Discarding invalid cached download $(basename "${out}")"
    rm -f "${out}"
  fi
  rm -f "${tmp}"
  if ! python -m gdown --fuzzy "https://drive.google.com/uc?id=${file_id}" -O "${tmp}"; then
    rm -f "${tmp}"
    return 1
  fi
  mv "${tmp}" "${out}"
}

extract_zip() {
  local archive="$1"
  local dst="$2"
  mkdir -p "${dst}"
  unzip -oq "${archive}" -d "${dst}"
}

normalize_evarest_recognition() {
  local root="$1"
  local merged="${root}/merged"
  mkdir -p "${merged}"
  find "${root}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 | while IFS= read -r -d '' path; do
    local base
    base="$(basename "${path}")"
    cp -n "${path}" "${merged}/${base}" || true
  done
}

require_cmd wget
require_cmd unzip
python - <<'PY'
import importlib.util
import subprocess
import sys
missing = []
for package in ("gdown", "libtorrent"):
    if importlib.util.find_spec(package) is None:
        missing.append(package)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

download_ctw1500_test_labels() {
  local out="$1"
  local tmp="${out}.part"
  local eval_bundle="${CACHE_ROOT}/ctw1500_eval_bundle.zip"
  local torrent_path="${CACHE_ROOT}/ctw1500.torrent"

  if [[ -f "${out}" ]] && unzip -tqq "${out}" >/dev/null 2>&1; then
    echo "Using cached $(basename "${out}")"
    return
  fi
  rm -f "${out}" "${tmp}"

  if download_file "https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download" "${out}"; then
    return
  fi

  rm -f "${out}" "${tmp}"
  download_file "https://adelaideuniversity.box.com/shared/static/ys234cg1rtgke051hu33lbm5ri0bvxr0.zip" "${eval_bundle}"
  unzip -p "${eval_bundle}" datasets/evaluation/gt_ctw1500.zip > "${tmp}"
  if unzip -tqq "${tmp}" >/dev/null 2>&1; then
    mv "${tmp}" "${out}"
    return
  fi

  rm -f "${out}" "${tmp}"
  download_file "https://orion.hyper.ai/tracker/download?torrent=18392" "${torrent_path}"
  python - <<'PY' "${torrent_path}" "${tmp}"
from __future__ import annotations

import sys
import time
from pathlib import Path

import libtorrent as lt

torrent_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
target_name = "uoeFl0pCN9BOCN5"

info = lt.torrent_info(str(torrent_path))
save_path = output_path.parent / "_ctw1500_torrent"
save_path.mkdir(parents=True, exist_ok=True)

target_index = None
offset = 0
target_start = 0
target_end = 0
for index, file_entry in enumerate(info.files()):
    file_path = Path(file_entry.path)
    file_size = file_entry.size
    if file_path.name == target_name:
        target_index = index
        target_start = offset
        target_end = offset + file_size - 1
        break
    offset += file_size

if target_index is None:
    raise RuntimeError(f"Failed to find {target_name!r} in {torrent_path}")

piece_length = info.piece_length()
first_piece = target_start // piece_length
last_piece = target_end // piece_length
piece_offset = target_start - first_piece * piece_length

session = lt.session(
    {
        "listen_interfaces": "0.0.0.0:6881",
        "enable_dht": True,
        "enable_lsd": True,
        "enable_upnp": False,
        "enable_natpmp": False,
    }
)
handle = session.add_torrent({"save_path": str(save_path), "ti": info})
for file_index in range(info.num_files()):
    handle.file_priority(file_index, 0)
for piece_index in range(info.num_pieces()):
    handle.piece_priority(piece_index, 0)
handle.file_priority(target_index, 7)
for piece_index in range(first_piece, last_piece + 1):
    handle.piece_priority(piece_index, 7)
    handle.set_piece_deadline(piece_index, 0)

deadline = time.time() + 900
while time.time() < deadline:
    if all(handle.have_piece(piece_index) for piece_index in range(first_piece, last_piece + 1)):
        break
    time.sleep(5)
else:
    raise TimeoutError("Timed out downloading CTW1500 test labels from the torrent fallback.")

buffers: list[bytes] = []
for piece_index in range(first_piece, last_piece + 1):
    handle.read_piece(piece_index)

read_deadline = time.time() + 60
while time.time() < read_deadline and len(buffers) < (last_piece - first_piece + 1):
    for alert in session.pop_alerts():
        if isinstance(alert, lt.read_piece_alert) and first_piece <= alert.piece <= last_piece:
            buffers.append(bytes(alert.buffer))
    time.sleep(1)

if len(buffers) < (last_piece - first_piece + 1):
    raise TimeoutError("Timed out reading the downloaded CTW1500 label piece.")

piece_blob = b"".join(buffers)
payload = piece_blob[piece_offset : piece_offset + (target_end - target_start + 1)]
if len(payload) != (target_end - target_start + 1):
    raise RuntimeError("Downloaded CTW1500 label payload has an unexpected size.")
output_path.write_bytes(payload)
PY
  unzip -tqq "${tmp}" >/dev/null 2>&1
  mv "${tmp}" "${out}"
}

echo "Downloading Total-Text"
TOTALTEXT_ROOT="${DATA_ROOT}/totaltext"
mkdir -p "${TOTALTEXT_ROOT}"
download_file "https://adelaideuniversity.box.com/shared/static/8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip" "${CACHE_ROOT}/totaltext_images.zip"
download_file "https://adelaideuniversity.box.com/shared/static/2vmpvjb48pcrszeegx2eznzc4izan4zf.zip" "${CACHE_ROOT}/totaltext_txt_format.zip"
extract_zip "${CACHE_ROOT}/totaltext_images.zip" "${TOTALTEXT_ROOT}"
extract_zip "${CACHE_ROOT}/totaltext_txt_format.zip" "${TOTALTEXT_ROOT}"

echo "Downloading CTW1500"
CTW_ROOT="${DATA_ROOT}/ctw1500"
mkdir -p "${CTW_ROOT}"
download_file "https://adelaideuniversity.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip" "${CACHE_ROOT}/ctw1500_train_images.zip"
download_file "https://adelaideuniversity.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip" "${CACHE_ROOT}/ctw1500_train_labels.zip"
download_file "https://adelaideuniversity.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip" "${CACHE_ROOT}/ctw1500_test_images.zip"
download_ctw1500_test_labels "${CACHE_ROOT}/ctw1500_test_labels.zip"
extract_zip "${CACHE_ROOT}/ctw1500_train_images.zip" "${CTW_ROOT}"
extract_zip "${CACHE_ROOT}/ctw1500_train_labels.zip" "${CTW_ROOT}"
extract_zip "${CACHE_ROOT}/ctw1500_test_images.zip" "${CTW_ROOT}"
extract_zip "${CACHE_ROOT}/ctw1500_test_labels.zip" "${CTW_ROOT}"

echo "Downloading TextOCR"
TEXTOCR_ROOT="${DATA_ROOT}/textocr"
mkdir -p "${TEXTOCR_ROOT}"
download_file "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" "${CACHE_ROOT}/textocr_train_val_images.zip"
download_file "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip" "${CACHE_ROOT}/textocr_test_images.zip"
download_file "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json" "${TEXTOCR_ROOT}/TextOCR_0.1_train.json"
download_file "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json" "${TEXTOCR_ROOT}/TextOCR_0.1_val.json"
download_file "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_test.json" "${TEXTOCR_ROOT}/TextOCR_0.1_test.json"
extract_zip "${CACHE_ROOT}/textocr_train_val_images.zip" "${TEXTOCR_ROOT}"
extract_zip "${CACHE_ROOT}/textocr_test_images.zip" "${TEXTOCR_ROOT}"

echo "Downloading EvArEST"
EVAREST_DET_ROOT="${DATA_ROOT}/evarest_detection"
EVAREST_REC_ROOT="${DATA_ROOT}/evarest_recognition"
mkdir -p "${EVAREST_DET_ROOT}" "${EVAREST_REC_ROOT}"
download_gdrive "1a1Jf12nyIDswunky5kLM4JishRj2_4Jy" "${CACHE_ROOT}/evarest_detection_train.zip"
download_gdrive "15jWxmZb9zoKHys40Cuz-57kV2PTO-cvH" "${CACHE_ROOT}/evarest_detection_test.zip"
download_gdrive "1ADdCb66VvndcBnRo38IIymL9HZsE4FNx" "${CACHE_ROOT}/evarest_rec_ar_train.zip"
download_gdrive "1vyypcLpX6DTuogNxTeueRRl_PLvcbRsz" "${CACHE_ROOT}/evarest_rec_en_train.zip"
download_gdrive "1P1SnF4ZKOA1PBC6HRAYLxZa82eOGFR2F" "${CACHE_ROOT}/evarest_rec_ar_test.zip"
download_gdrive "1HCPSAeJGNP5LtdIjAuu7ZbeFDTubMYDx" "${CACHE_ROOT}/evarest_rec_en_test.zip"
extract_zip "${CACHE_ROOT}/evarest_detection_train.zip" "${EVAREST_DET_ROOT}"
extract_zip "${CACHE_ROOT}/evarest_detection_test.zip" "${EVAREST_DET_ROOT}"
extract_zip "${CACHE_ROOT}/evarest_rec_ar_train.zip" "${EVAREST_REC_ROOT}"
extract_zip "${CACHE_ROOT}/evarest_rec_en_train.zip" "${EVAREST_REC_ROOT}"
extract_zip "${CACHE_ROOT}/evarest_rec_ar_test.zip" "${EVAREST_REC_ROOT}"
extract_zip "${CACHE_ROOT}/evarest_rec_en_test.zip" "${EVAREST_REC_ROOT}"
normalize_evarest_recognition "${EVAREST_REC_ROOT}"

echo "Downloading IR-LPR"
IRLPR_DET_ROOT="${DATA_ROOT}/ir_lpr_detection"
IRLPR_REC_ROOT="${DATA_ROOT}/ir_lpr_recognition"
mkdir -p "${IRLPR_DET_ROOT}" "${IRLPR_REC_ROOT}"
download_gdrive "1XtZ-XQ8ImNFf40D-bFqTm0UVFqNKhbLi" "${CACHE_ROOT}/ir_lpr_car_train.zip"
download_gdrive "1hwz6X-Zp7JpJL35K6P3z7k6O_PTXhUcT" "${CACHE_ROOT}/ir_lpr_car_val.zip"
download_gdrive "1pe4_HgXb9dctFGJXVNlyNcKSXZeht0lX" "${CACHE_ROOT}/ir_lpr_car_test.zip"
download_gdrive "1ubkg7E2vGEOqS4K_quwf9Vl-i8IVpklM" "${CACHE_ROOT}/ir_lpr_plate_train.zip"
download_gdrive "1AL5Zsg2hDqcwF8ZmR0MJTbjgXIoE5W-I" "${CACHE_ROOT}/ir_lpr_plate_val.zip"
download_gdrive "1lLh_kxrHy1teUB2NguHVuOZwA5rjL5kx" "${CACHE_ROOT}/ir_lpr_plate_test.zip"
extract_zip "${CACHE_ROOT}/ir_lpr_car_train.zip" "${IRLPR_DET_ROOT}"
extract_zip "${CACHE_ROOT}/ir_lpr_car_val.zip" "${IRLPR_DET_ROOT}"
extract_zip "${CACHE_ROOT}/ir_lpr_car_test.zip" "${IRLPR_DET_ROOT}"
extract_zip "${CACHE_ROOT}/ir_lpr_plate_train.zip" "${IRLPR_REC_ROOT}"
extract_zip "${CACHE_ROOT}/ir_lpr_plate_val.zip" "${IRLPR_REC_ROOT}"
extract_zip "${CACHE_ROOT}/ir_lpr_plate_test.zip" "${IRLPR_REC_ROOT}"

echo "Preparing MLT placeholder"
mkdir -p "${DATA_ROOT}/icdar2019_mlt"

cat <<EOF
Downloaded public datasets under:
  ${DATA_ROOT}

ICDAR2019 MLT should be placed under:
  ${DATA_ROOT}/icdar2019_mlt
EOF
