#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_dataset import iter_detection_annotation_files, parse_detection_line
from external_datasets import detection_mix_summary, mix_train_items, path_for_manifest


def build_detection_record(txt_path: Path, repo_root: Path) -> tuple[dict | None, list[dict]]:
    errors: list[dict] = []
    image_path = txt_path.with_suffix(".jpg")
    if not image_path.exists():
        errors.append({"file": str(txt_path), "line": 0, "reason": "missing_image"})
        return None, errors

    instances: list[dict] = []
    for line_number, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            polygon, transcription = parse_detection_line(raw_line)
        except ValueError as exc:
            errors.append({"file": str(txt_path), "line": line_number, "reason": str(exc)})
            continue
        xs = polygon[0::2]
        ys = polygon[1::2]
        instances.append(
            {
                "polygon": polygon,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "bbox_label": 0,
                "ignore": False,
                "text": transcription,
            }
        )

    if not instances:
        errors.append({"file": str(txt_path), "line": 0, "reason": "no_valid_instances"})
        return None, errors

    with Image.open(image_path) as image:
        width, height = image.size

    record = {
        "img_path": path_for_manifest(image_path, repo_root),
        "height": height,
        "width": width,
        "instances": instances,
    }
    return record, errors


def make_manifest(records: list[dict]) -> dict:
    return {
        "metainfo": {
            "dataset_type": "TextDetDataset",
            "task_name": "textdet",
            "category": [{"id": 0, "name": "text"}],
        },
        "data_list": records,
    }


def split_records(
    grouped_records: dict[str, list[dict]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []
    for domain, records in sorted(grouped_records.items()):
        bucket = list(records)
        rng.shuffle(bucket)
        if val_ratio <= 0 or len(bucket) <= 1:
            train_records.extend(bucket)
            continue
        val_count = max(1, round(len(bucket) * val_ratio))
        val_count = min(val_count, len(bucket) - 1)
        val_records.extend(bucket[:val_count])
        train_records.extend(bucket[val_count:])
    return train_records, val_records


def build_detection_manifests(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    include_domains: Sequence[str] | None,
    val_ratio: float,
    seed: int,
    external_train_records: dict[str, list] | None = None,
    train_mix: dict[str, float] | None = None,
    min_ptdr_fraction: float = 0.3,
    extra_errors: Sequence[dict] | None = None,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    grouped_train_records: dict[str, list[dict]] = defaultdict(list)
    test_records: list[dict] = []
    errors: list[dict] = []

    for split in ("train", "test"):
        for txt_path, domain in iter_detection_annotation_files(dataset_root, split, include_domains):
            record, record_errors = build_detection_record(txt_path, repo_root)
            errors.extend(record_errors)
            if record is None:
                continue
            if split == "train":
                grouped_train_records[domain].append(record)
            else:
                test_records.append(record)

    primary_train_records, val_records = split_records(grouped_train_records, val_ratio=val_ratio, seed=seed)
    train_records = list(primary_train_records)
    mix_summary = None
    if external_train_records:
        mixed_items, mix_details = mix_train_items(
            {"ptdr": list(primary_train_records), **external_train_records},
            train_mix=train_mix or {"ptdr": 1.0},
            seed=seed,
            min_ptdr_fraction=min_ptdr_fraction,
        )
        # DetectionRecord instances expose `.record`; plain PTDR entries are dicts already.
        train_records = [getattr(item, "record", item) for item in mixed_items]
        mix_summary = detection_mix_summary(mix_details)

    train_path = output_root / "textdet_train.json"
    val_path = output_root / "textdet_val.json"
    test_path = output_root / "textdet_test.json"
    summary_path = output_root / "summary.json"

    train_path.write_text(json.dumps(make_manifest(train_records), indent=2, ensure_ascii=False), encoding="utf-8")
    val_path.write_text(json.dumps(make_manifest(val_records), indent=2, ensure_ascii=False), encoding="utf-8")
    test_path.write_text(json.dumps(make_manifest(test_records), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "include_domains": list(include_domains or []),
        "seed": seed,
        "val_ratio": val_ratio,
        "train_images": len(train_records),
        "train_primary_images": len(primary_train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "external_train_dataset_count": sum(1 for records in (external_train_records or {}).values() if records),
        "train_mix": mix_summary,
        "errors": errors + list(extra_errors or []),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "train_ann": train_path,
        "val_ann": val_path,
        "test_ann": test_path,
        "summary": summary_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MMOCR detection manifests for PTDR.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains the dataset directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional detection dataset root override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" / "dbnetpp",
        help="Output directory for the generated manifest files.",
    )
    parser.add_argument(
        "--include-domain",
        action="append",
        default=[],
        help="Domain prefix to include, for example outdoor_text or outdoor_text/retail_exteriors.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction sampled from train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_root = (args.dataset_root or repo_root / "dataset" / "detection").resolve()
    result = build_detection_manifests(
        repo_root=repo_root,
        dataset_root=dataset_root,
        output_root=args.output_root.resolve(),
        include_domains=args.include_domain or None,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
