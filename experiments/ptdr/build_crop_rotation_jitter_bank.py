#!/usr/bin/env python3

import json
import math
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import lmdb
import pyrallis

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_schema import DEFAULT_REPO_ROOT, resolve_repo_relative
from rotation_solution_utils import LmdbRecognitionDataset, encode_png_bytes, jitter_quad_detector_style, load_rgb_image, perspective_crop_from_quad
from train_crop_rotation_classifier import CropRotationExperimentConfig, build_detection_crop_records


def compute_default_bank_size(settings: CropRotationExperimentConfig, train_root: Path) -> int:
    base_length = len(LmdbRecognitionDataset(train_root))
    effective_length = base_length * 4 if settings.data.exhaustive_right_angle_classes else base_length
    return max(1, int(math.ceil(effective_length * float(settings.data.detector_jitter_probability))))


def _remove_existing_lmdb(root: Path) -> None:
    for child_name in ("data.mdb", "lock.mdb", "summary.json"):
        child_path = root / child_name
        if child_path.exists() or child_path.is_symlink():
            child_path.unlink()
    for child in root.iterdir() if root.exists() else ():
        if child.is_dir():
            shutil.rmtree(child)


def _load_detection_image_cached(
    cache: OrderedDict[str, object],
    image_path: Path,
    cache_size: int,
):
    cache_key = str(image_path)
    cached = cache.get(cache_key)
    if cached is not None:
        cache.move_to_end(cache_key)
        return cached
    image_rgb = load_rgb_image(image_path)
    if cache_size > 0:
        cache[cache_key] = image_rgb
        while len(cache) > cache_size:
            cache.popitem(last=False)
    return image_rgb


def sample_detector_jitter_crop(
    detection_records: list[dict],
    rng: random.Random,
    detector_jitter_attempts: int,
    detection_image_cache_size: int,
    image_cache: OrderedDict[str, object],
) -> tuple[object, dict]:
    while True:
        record = rng.choice(detection_records)
        instance = rng.choice(record["instances"])
        image_rgb = _load_detection_image_cached(
            cache=image_cache,
            image_path=record["image_path"],
            cache_size=detection_image_cache_size,
        )
        image_height, image_width = image_rgb.shape[:2]
        for _ in range(max(1, int(detector_jitter_attempts))):
            jittered_quad = jitter_quad_detector_style(
                polygon=instance["polygon"],
                image_size=(image_height, image_width),
                rng=rng,
            )
            if jittered_quad is None:
                continue
            crop = perspective_crop_from_quad(image_rgb, jittered_quad)
            if crop is None:
                continue
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            metadata = {
                "domain": record["domain"],
                "repo_relative_path": record["repo_relative_path"],
                "text": instance["text"],
                "height": int(crop.shape[0]),
                "width": int(crop.shape[1]),
            }
            return crop, metadata


@pyrallis.wrap(config_path=str(SCRIPT_DIR / "configs" / "crop_rotation_classifier_128_jitter_exhaustive_fast.yaml"))
def main(settings: CropRotationExperimentConfig) -> None:
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    train_root = resolve_repo_relative(repo_root, settings.data.train_lmdb_root)
    output_root = settings.data.detector_jitter_bank_root
    if output_root is None:
        raise ValueError("data.detector_jitter_bank_root must be set to build a jitter bank.")
    output_root = resolve_repo_relative(repo_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bank_size = (
        int(settings.data.detector_jitter_bank_size)
        if settings.data.detector_jitter_bank_size is not None
        else compute_default_bank_size(settings, train_root)
    )
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("count", -1)) == bank_size and (output_root / "data.mdb").exists():
            print(
                json.dumps(
                    {
                        "status": "already_exists",
                        "output_root": str(output_root),
                        "count": bank_size,
                    }
                )
            )
            return
    _remove_existing_lmdb(output_root)

    detection_records = build_detection_crop_records(
        repo_root=repo_root,
        dataset_root=resolve_repo_relative(repo_root, settings.data.detection_dataset_root),
        include_domains=settings.data.include_domains,
    )
    rng = random.Random(int(settings.data.detector_jitter_bank_seed))
    image_cache: OrderedDict[str, object] = OrderedDict()
    map_size_bytes = 16 * (1024**3)
    env = lmdb.open(str(output_root), map_size=map_size_bytes, subdir=True, lock=True)
    start_time = time.time()
    commit_interval = 512
    next_progress = 2000

    print(
        json.dumps(
            {
                "status": "building",
                "output_root": str(output_root),
                "count": bank_size,
                "seed": int(settings.data.detector_jitter_bank_seed),
            }
        )
    )
    txn = env.begin(write=True)
    try:
        for sample_index in range(1, bank_size + 1):
            crop, metadata = sample_detector_jitter_crop(
                detection_records=detection_records,
                rng=rng,
                detector_jitter_attempts=settings.data.detector_jitter_attempts,
                detection_image_cache_size=settings.data.detection_image_cache_size,
                image_cache=image_cache,
            )
            txn.put(f"image-{sample_index:09d}".encode("ascii"), encode_png_bytes(crop))
            txn.put(f"label-{sample_index:09d}".encode("ascii"), metadata["text"].encode("utf-8"))
            txn.put(f"meta-{sample_index:09d}".encode("ascii"), json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
            if sample_index % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
            if sample_index >= next_progress or sample_index == bank_size:
                elapsed = time.time() - start_time
                samples_per_second = sample_index / max(elapsed, 1e-6)
                eta_seconds = max(0.0, (bank_size - sample_index) / max(samples_per_second, 1e-6))
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "done": sample_index,
                            "total": bank_size,
                            "samples_per_second": round(samples_per_second, 3),
                            "eta_seconds": round(eta_seconds, 1),
                        }
                    ),
                    flush=True,
                )
                next_progress += 2000
        txn.put(b"num-samples", str(bank_size).encode("ascii"))
        txn.commit()
    finally:
        env.sync()
        env.close()

    elapsed = time.time() - start_time
    summary = {
        "count": bank_size,
        "elapsed_seconds": elapsed,
        "samples_per_second": bank_size / max(elapsed, 1e-6),
        "seed": int(settings.data.detector_jitter_bank_seed),
        "detector_jitter_attempts": int(settings.data.detector_jitter_attempts),
        "detector_jitter_probability": float(settings.data.detector_jitter_probability),
        "exhaustive_right_angle_classes": bool(settings.data.exhaustive_right_angle_classes),
        "bank_size_policy": (
            "config"
            if settings.data.detector_jitter_bank_size is not None
            else "expected_jitter_uses_per_epoch"
        ),
        "include_domains": list(settings.data.include_domains),
        "train_lmdb_root": str(train_root),
        "detection_dataset_root": str(resolve_repo_relative(repo_root, settings.data.detection_dataset_root)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "done", "output_root": str(output_root), "elapsed_seconds": round(elapsed, 1)}))


if __name__ == "__main__":
    main()
