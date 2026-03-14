from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from .detection_augmentations import (
        DEFAULT_PRESETS,
        apply_preset_to_image_instances,
        choose_preset_for_key,
        stable_seed,
    )
except ImportError:
    from detection_augmentations import DEFAULT_PRESETS, apply_preset_to_image_instances, choose_preset_for_key, stable_seed


def path_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_manifest_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data_list"]


def _count_manifest_records(path: Path) -> int:
    return len(_load_manifest_records(path))


def _save_manifest(records: list[dict], output_path: Path) -> None:
    payload = {
        "metainfo": {
            "dataset_type": "TextDetDataset",
            "task_name": "textdet",
            "category": [{"id": 0, "name": "text"}],
        },
        "data_list": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _build_variant_record(repo_root: Path, original_record: dict, image_rel_path: str, image: np.ndarray, instances: list[dict]) -> dict:
    height, width = image.shape[:2]
    record = copy.deepcopy(original_record)
    record["img_path"] = image_rel_path
    record["height"] = int(height)
    record["width"] = int(width)
    record["instances"] = instances
    return record


def build_detection_eval_variants(
    repo_root: Path,
    val_ann_path: Path,
    test_ann_path: Path,
    output_root: Path,
    seed: int = 42,
    rotation_angles: tuple[int, ...] = (90, 180, 270),
    include_test_hard: bool = True,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    images_root = output_root / "images"
    expected_outputs = {
        **{f"val_rot{int(angle)}": output_root / f"textdet_val_rot{int(angle)}.json" for angle in rotation_angles},
        "val_hard": output_root / "textdet_val_hard.json",
    }
    if include_test_hard:
        expected_outputs["test_hard"] = output_root / "textdet_test_hard.json"
    expected_outputs["summary"] = output_root / "summary.json"
    if all(path.exists() for path in expected_outputs.values()):
        return expected_outputs

    val_records = _load_manifest_records(val_ann_path)
    test_records = _load_manifest_records(test_ann_path)

    outputs: dict[str, Path] = {}
    summary: dict[str, object] = {
        "seed": seed,
        "rotation_angles": list(rotation_angles),
        "preset_names": [preset["name"] for preset in DEFAULT_PRESETS],
        "variants": {},
    }

    for angle in rotation_angles:
        variant_name = f"val_rot{int(angle)}"
        ann_path = output_root / f"textdet_{variant_name}.json"
        if ann_path.exists():
            outputs[variant_name] = ann_path
            summary["variants"][variant_name] = {"images": _count_manifest_records(ann_path)}
        else:
            variant_records: list[dict] = []
            for index, record in enumerate(val_records):
                source_image = repo_root / record["img_path"]
                image = _load_rgb_image(source_image)
                preset = {"name": variant_name, "rotation_deg": float(angle)}
                image_seed = stable_seed(seed, variant_name, record["img_path"], index)
                augmented_image, augmented_instances = apply_preset_to_image_instances(
                    image=image,
                    instances=record["instances"],
                    preset=preset,
                    seed=image_seed,
                )
                out_image = images_root / variant_name / f"{index:05d}_{Path(record['img_path']).stem}.png"
                _save_rgb_image(out_image, augmented_image)
                variant_records.append(
                    _build_variant_record(
                        repo_root=repo_root,
                        original_record=record,
                        image_rel_path=path_for_manifest(out_image, repo_root),
                        image=augmented_image,
                        instances=augmented_instances,
                    )
                )
            _save_manifest(variant_records, ann_path)
            outputs[variant_name] = ann_path
            summary["variants"][variant_name] = {"images": len(variant_records)}

    split_records = [("val_hard", val_records)]
    if include_test_hard:
        split_records.append(("test_hard", test_records))
    for split_name, records in split_records:
        preset_counter: Counter[str] = Counter()
        for index, record in enumerate(records):
            preset, _ = choose_preset_for_key(record["img_path"], seed=seed, presets=DEFAULT_PRESETS)
            preset_counter[preset["name"]] += 1
        ann_path = output_root / f"textdet_{split_name}.json"
        if ann_path.exists():
            outputs[split_name] = ann_path
            summary["variants"][split_name] = {
                "images": _count_manifest_records(ann_path),
                "preset_counts": dict(sorted(preset_counter.items())),
            }
            continue

        variant_records = []
        for index, record in enumerate(records):
            source_image = repo_root / record["img_path"]
            image = _load_rgb_image(source_image)
            preset, preset_index = choose_preset_for_key(record["img_path"], seed=seed, presets=DEFAULT_PRESETS)
            image_seed = stable_seed(seed, split_name, record["img_path"], preset_index, index)
            augmented_image, augmented_instances = apply_preset_to_image_instances(
                image=image,
                instances=record["instances"],
                preset=preset,
                seed=image_seed,
            )
            out_image = images_root / split_name / f"{index:05d}_{Path(record['img_path']).stem}.png"
            _save_rgb_image(out_image, augmented_image)
            variant_records.append(
                _build_variant_record(
                    repo_root=repo_root,
                    original_record=record,
                    image_rel_path=path_for_manifest(out_image, repo_root),
                    image=augmented_image,
                    instances=augmented_instances,
                )
            )
        _save_manifest(variant_records, ann_path)
        outputs[split_name] = ann_path
        summary["variants"][split_name] = {
            "images": len(variant_records),
            "preset_counts": dict(sorted(preset_counter.items())),
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs
