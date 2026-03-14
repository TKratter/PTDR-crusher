#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from mmocr.datasets.preparers.parsers.coco_parser import COCOTextDetAnnParser
from mmocr.datasets.preparers.parsers.ctw1500_parser import CTW1500AnnParser
from mmocr.datasets.preparers.parsers.totaltext_parser import TotaltextTextDetAnnParser
from mmocr.utils import poly2bbox


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_TEXT_VALUES = {"", "#", "###", "."}
IR_LPR_PLATE_TOKENS = ("plate", "license", "lp", "پلاک")
KNOWN_SCRIPT_LABELS = {
    "arabic",
    "bengali",
    "chinese",
    "devanagari",
    "english",
    "farsi",
    "french",
    "german",
    "hindi",
    "italian",
    "japanese",
    "korean",
    "latin",
    "persian",
    "symbols",
}
KNOWN_LANGUAGE_LABELS = {"arabic", "english", "latin", "persian", "farsi"}
TRAIN_MIX_DATASETS = (
    "ptdr",
    "icdar2019_mlt",
    "evarest",
    "totaltext",
    "ctw1500",
    "textocr",
    "ir_lpr",
)


@dataclass(frozen=True)
class DetectionRecord:
    source: str
    image_path: Path
    record: dict


@dataclass(frozen=True)
class RecognitionSample:
    source: str
    domain: str
    label: str
    repo_relative_path: str
    image_path: Path | None = None
    image_bytes: bytes | None = None
    crop_box: tuple[float, float, float, float] | None = None


def path_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def normalized_text(text: str | None, fallback: str = "text") -> str:
    value = (text or "").strip().lstrip("\ufeff")
    return value or fallback


def text_is_ignored(text: str | None) -> bool:
    value = (text or "").strip().lstrip("\ufeff")
    return value in IGNORE_TEXT_VALUES


def build_detection_instance(polygon: Sequence[float], text: str, ignore: bool = False) -> dict:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return {
        "polygon": [float(value) for value in polygon],
        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
        "bbox_label": 0,
        "ignore": bool(ignore),
        "text": normalized_text(text),
    }


def build_detection_record(repo_root: Path, image_path: Path, instances: list[dict]) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "img_path": path_for_manifest(image_path, repo_root),
        "height": height,
        "width": width,
        "instances": instances,
    }


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_image_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in iter_image_files(root):
        by_name[image_path.name.lower()].append(image_path)
        by_stem[image_path.stem.lower()].append(image_path)
    return dict(by_name), dict(by_stem)


def resolve_image_path(
    root: Path,
    image_lookup: tuple[dict[str, list[Path]], dict[str, list[Path]]],
    image_ref: str | Path,
) -> Path | None:
    by_name, by_stem = image_lookup
    image_path = Path(image_ref)
    candidates: list[Path] = []
    if image_path.is_absolute() and image_path.exists():
        return image_path
    if not image_path.is_absolute():
        direct = (root / image_path).resolve()
        if direct.exists():
            return direct
        candidates.extend(by_name.get(image_path.name.lower(), []))
        candidates.extend(by_stem.get(image_path.stem.lower(), []))
    for candidate in candidates:
        return candidate
    return None


def is_ir_lpr_plate_label(label: str | None) -> bool:
    value = normalized_text(label, fallback="").casefold()
    if not value:
        return False
    if value in {"lp", "license"}:
        return True
    return any(token in value for token in IR_LPR_PLATE_TOKENS if token)


def sample_to_target(items: list, target_count: int, seed: int) -> list:
    if target_count <= 0 or not items:
        return []
    rng = random.Random(seed)
    if target_count <= len(items):
        return rng.sample(items, target_count)
    repeats, remainder = divmod(target_count, len(items))
    sampled = list(items) * repeats
    if remainder:
        sampled.extend(rng.sample(items, remainder))
    rng.shuffle(sampled)
    return sampled


def normalize_train_mix(train_mix: dict[str, float], available_counts: dict[str, int], min_ptdr_fraction: float) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for dataset_name in TRAIN_MIX_DATASETS:
        raw_value = float(train_mix.get(dataset_name, 0.0))
        if raw_value < 0:
            raise ValueError(f"Train mix for {dataset_name!r} must be non-negative.")
        if available_counts.get(dataset_name, 0) > 0 and raw_value > 0:
            normalized[dataset_name] = raw_value
    if available_counts.get("ptdr", 0) > 0 and "ptdr" not in normalized:
        normalized["ptdr"] = 1.0 if not normalized else 0.0
    normalized = {name: weight for name, weight in normalized.items() if weight > 0}
    if not normalized:
        raise ValueError("No training data is available after applying the configured dataset roots and train mix.")
    total = sum(normalized.values())
    fractions = {name: weight / total for name, weight in normalized.items()}
    if "ptdr" not in fractions:
        raise ValueError("PTDR must remain part of the effective train mix.")
    if fractions["ptdr"] < min_ptdr_fraction and len(fractions) > 1:
        raise ValueError(
            f"Configured PTDR fraction {fractions['ptdr']:.3f} is below the required minimum {min_ptdr_fraction:.3f}."
        )
    return fractions


def mix_train_items(
    dataset_items: dict[str, list],
    train_mix: dict[str, float],
    seed: int,
    min_ptdr_fraction: float,
) -> tuple[list, dict]:
    available_counts = {name: len(items) for name, items in dataset_items.items()}
    fractions = normalize_train_mix(train_mix, available_counts, min_ptdr_fraction)
    ptdr_items = dataset_items.get("ptdr", [])
    if not ptdr_items:
        raise ValueError("PTDR training data is empty; cannot build a mixed train split.")
    if len(fractions) == 1:
        return list(ptdr_items), {
            "normalized_fractions": fractions,
            "raw_counts": available_counts,
            "effective_counts": {"ptdr": len(ptdr_items)},
        }

    total_target = max(len(ptdr_items), math.floor(len(ptdr_items) / fractions["ptdr"]))
    mixed = list(ptdr_items)
    effective_counts = {"ptdr": len(ptdr_items)}
    remaining_budget = max(0, total_target - len(ptdr_items))
    target_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for index, dataset_name in enumerate(TRAIN_MIX_DATASETS, start=1):
        if dataset_name == "ptdr" or dataset_name not in fractions:
            continue
        ideal = total_target * fractions[dataset_name]
        target_count = min(remaining_budget, math.floor(ideal))
        target_counts[dataset_name] = target_count
        remainders.append((ideal - target_count, dataset_name))
        remaining_budget -= target_count
    for _, dataset_name in sorted(remainders, reverse=True):
        if remaining_budget <= 0:
            break
        target_counts[dataset_name] += 1
        remaining_budget -= 1
    for index, dataset_name in enumerate(TRAIN_MIX_DATASETS, start=1):
        if dataset_name == "ptdr" or dataset_name not in fractions:
            continue
        sampled = sample_to_target(dataset_items.get(dataset_name, []), target_counts.get(dataset_name, 0), seed + index * 9973)
        mixed.extend(sampled)
        effective_counts[dataset_name] = len(sampled)
    return mixed, {
        "normalized_fractions": fractions,
        "raw_counts": available_counts,
        "effective_counts": effective_counts,
    }


def detection_mix_summary(mix_details: dict) -> dict:
    raw_counts = mix_details["raw_counts"]
    effective_counts = mix_details["effective_counts"]
    total_effective = sum(effective_counts.values()) or 1
    normalized = mix_details["normalized_fractions"]
    summary: dict[str, dict] = {}
    for dataset_name in TRAIN_MIX_DATASETS:
        if dataset_name not in raw_counts and dataset_name not in effective_counts:
            continue
        effective_count = effective_counts.get(dataset_name, 0)
        summary[dataset_name] = {
            "raw_count": raw_counts.get(dataset_name, 0),
            "effective_count": effective_count,
            "configured_fraction": normalized.get(dataset_name, 0.0),
            "effective_fraction": effective_count / total_effective if total_effective else 0.0,
        }
    return summary


def train_mix_summary(mix_details: dict) -> dict:
    return detection_mix_summary(mix_details)


def build_recognition_crop_samples(repo_root: Path, records: Sequence[DetectionRecord], source_name: str) -> list[RecognitionSample]:
    grouped: dict[Path, list[dict]] = defaultdict(list)
    for wrapped in records:
        grouped[wrapped.image_path].extend(wrapped.record["instances"])

    samples: list[RecognitionSample] = []
    for image_path, instances in grouped.items():
        stem = image_path.stem
        suffix = image_path.suffix or ".jpg"
        for instance_index, instance in enumerate(instances):
            text = instance.get("text", "")
            if instance.get("ignore", False) or text_is_ignored(text):
                continue
            bbox = poly2bbox(instance["polygon"])
            samples.append(
                RecognitionSample(
                    source=source_name,
                    domain=source_name,
                    label=normalized_text(text),
                    repo_relative_path=f"{source_name}/{stem}_{instance_index}{suffix}",
                    image_path=image_path,
                    crop_box=tuple(float(value) for value in bbox),
                )
            )
    return samples


def parse_csv_fields(line: str) -> list[str]:
    return [part.strip() for part in next(csv.reader([line], skipinitialspace=True))]


def first_nonempty_lines(path: Path, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cleaned = raw_line.strip().lstrip("\ufeff")
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= limit:
            break
    return lines


def find_sidecar_image(annotation_path: Path, image_lookup: tuple[dict[str, list[Path]], dict[str, list[Path]]]) -> Path | None:
    stem = annotation_path.stem
    candidates = [stem]
    if stem.lower().startswith("gt_"):
        candidates.append(stem[3:])
    if stem.lower().startswith("poly_gt_"):
        candidates.append(stem[8:])
    if re.fullmatch(r"0+\d+", stem):
        candidates.append(str(int(stem)))
    for candidate in candidates:
        image_path = resolve_image_path(annotation_path.parent.parent, image_lookup, candidate)
        if image_path is not None:
            return image_path
    return None


def load_totaltext_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    parser = TotaltextTextDetAnnParser(split="train")
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for ann_path in sorted(dataset_root.rglob("poly_gt_img*.txt")):
        image_path = find_sidecar_image(ann_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(ann_path), "line": 0, "reason": "missing_image"})
            continue
        _, instances = parser.parse_file(str(image_path), str(ann_path))
        converted = [
            build_detection_instance(instance["poly"], instance.get("text", "text"), instance.get("ignore", False))
            for instance in instances
            if instance.get("poly")
        ]
        if converted:
            records.append(DetectionRecord("totaltext", image_path, build_detection_record(repo_root, image_path, converted)))
    return records, errors


def load_ctw1500_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    train_parser = CTW1500AnnParser(split="train")
    test_parser = CTW1500AnnParser(split="test")
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []

    for ann_path in sorted(dataset_root.rglob("*.xml")):
        image_path = find_sidecar_image(ann_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(ann_path), "line": 0, "reason": "missing_image"})
            continue
        _, instances = train_parser.parse_file(str(image_path), str(ann_path))
        converted = [
            build_detection_instance(np.asarray(instance["poly"]).reshape(-1).tolist(), instance.get("text", "text"), False)
            for instance in instances
        ]
        if converted:
            records.append(DetectionRecord("ctw1500", image_path, build_detection_record(repo_root, image_path, converted)))

    for ann_path in sorted(dataset_root.rglob("*.txt")):
        if not re.fullmatch(r"0+\d+\.txt", ann_path.name):
            continue
        image_path = find_sidecar_image(ann_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(ann_path), "line": 0, "reason": "missing_image"})
            continue
        _, instances = test_parser.parse_file(str(image_path), str(ann_path))
        converted = [
            build_detection_instance(np.asarray(instance["poly"]).reshape(-1).tolist(), instance.get("text", "text"), instance.get("ignore", False))
            for instance in instances
        ]
        if converted:
            records.append(DetectionRecord("ctw1500", image_path, build_detection_record(repo_root, image_path, converted)))
    return records, errors


def load_textocr_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    json_paths = sorted(
        path
        for path in dataset_root.rglob("*.json")
        if "textocr" in path.name.lower() or path.name.lower() in {"train.json", "val.json"}
    )
    image_root_candidates = [
        dataset_root / "textdet_imgs" / "images",
        dataset_root / "train_images",
        dataset_root / "images",
        dataset_root,
    ]
    image_root = next((path for path in image_root_candidates if path.exists()), dataset_root)

    for ann_path in json_paths:
        try:
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append({"file": str(ann_path), "line": 0, "reason": f"textocr_json_parse_failed: {exc}"})
            continue
        if not any(key in payload for key in ("anns", "annotations", "imgToAnns")):
            continue
        split = "val" if "val" in ann_path.stem.lower() else "train"
        parser = COCOTextDetAnnParser(split=split, variant="textocr")
        try:
            parsed = parser.parse_files(str(image_root), str(ann_path))
        except Exception as exc:  # pragma: no cover - best effort on external format
            errors.append({"file": str(ann_path), "line": 0, "reason": f"textocr_parse_failed: {exc}"})
            continue
        for img_path_string, instances in parsed:
            image_path = Path(img_path_string)
            converted = [
                build_detection_instance(instance["poly"], instance.get("text", "text"), instance.get("ignore", False))
                for instance in instances
                if instance.get("poly")
            ]
            if converted:
                records.append(DetectionRecord("textocr", image_path, build_detection_record(repo_root, image_path, converted)))
    return records, errors


def parse_mlt_line(line: str, allowed_scripts: set[str]) -> tuple[list[float], str, bool] | None:
    parts = parse_csv_fields(line)
    if len(parts) < 9:
        raise ValueError(f"expected at least 9 comma-separated fields, found {len(parts)}")
    try:
        polygon = [float(value) for value in parts[:8]]
    except ValueError as exc:
        raise ValueError("non-numeric polygon coordinate") from exc
    remainder = [part.strip() for part in parts[8:]]
    script = remainder[0] if remainder else ""
    text = ",".join(remainder[1:]).strip() if len(remainder) > 1 else ""
    if script and allowed_scripts and script.casefold() not in allowed_scripts:
        return None
    ignore = text_is_ignored(text)
    return polygon, normalized_text(text, fallback="###" if ignore else script or "text"), ignore


def load_mlt_detection_records(
    repo_root: Path,
    dataset_root: Path,
    source_name: str,
    allowed_scripts: Sequence[str],
) -> tuple[list[DetectionRecord], list[dict]]:
    allowed = {script.casefold() for script in allowed_scripts}
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for ann_path in sorted(dataset_root.rglob("*.txt")):
        sample_lines = first_nonempty_lines(ann_path, limit=2)
        if not sample_lines:
            continue
        try:
            if all(parse_mlt_line(line, allowed) is None for line in sample_lines):
                continue
        except ValueError:
            continue
        image_path = find_sidecar_image(ann_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(ann_path), "line": 0, "reason": "missing_image"})
            continue
        instances: list[dict] = []
        for line_number, raw_line in enumerate(ann_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            cleaned = raw_line.strip().lstrip("\ufeff")
            if not cleaned:
                continue
            try:
                parsed = parse_mlt_line(cleaned, allowed)
            except ValueError as exc:
                errors.append({"file": str(ann_path), "line": line_number, "reason": str(exc)})
                continue
            if parsed is None:
                continue
            polygon, text, ignore = parsed
            instances.append(build_detection_instance(polygon, text, ignore))
        if instances:
            records.append(DetectionRecord(source_name, image_path, build_detection_record(repo_root, image_path, instances)))
    return records, errors


def parse_evarest_line(line: str) -> tuple[list[float], str, bool]:
    parts = parse_csv_fields(line)
    if len(parts) < 8:
        raise ValueError(f"expected at least 8 comma-separated fields, found {len(parts)}")
    try:
        polygon = [float(value) for value in parts[:8]]
    except ValueError as exc:
        raise ValueError("non-numeric polygon coordinate") from exc
    tail = [part for part in parts[8:] if part]
    if not tail:
        return polygon, "text", False
    if len(tail) == 1:
        text = tail[0]
        if text.casefold() in KNOWN_LANGUAGE_LABELS:
            text = "text"
        return polygon, normalized_text(text), False
    if tail[0].casefold() in KNOWN_LANGUAGE_LABELS and len(tail) >= 2:
        return polygon, normalized_text(",".join(tail[1:])), False
    if tail[-1].casefold() in KNOWN_LANGUAGE_LABELS:
        return polygon, normalized_text(",".join(tail[:-1])), False
    text = normalized_text(",".join(tail))
    return polygon, text, text_is_ignored(text)


def load_evarest_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for ann_path in sorted(dataset_root.rglob("*.txt")):
        sample_lines = first_nonempty_lines(ann_path, limit=2)
        if not sample_lines:
            continue
        try:
            parse_evarest_line(sample_lines[0])
        except ValueError:
            continue
        image_path = find_sidecar_image(ann_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(ann_path), "line": 0, "reason": "missing_image"})
            continue
        instances: list[dict] = []
        for line_number, raw_line in enumerate(ann_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            cleaned = raw_line.strip().lstrip("\ufeff")
            if not cleaned:
                continue
            try:
                polygon, text, ignore = parse_evarest_line(cleaned)
            except ValueError as exc:
                errors.append({"file": str(ann_path), "line": line_number, "reason": str(exc)})
                continue
            instances.append(build_detection_instance(polygon, text, ignore))
        if instances:
            records.append(DetectionRecord("evarest", image_path, build_detection_record(repo_root, image_path, instances)))
    return records, errors


def parse_coco_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for json_path in sorted(dataset_root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
            continue
        categories = {item.get("id"): item.get("name", "") for item in payload.get("categories", []) if isinstance(item, dict)}
        plate_category_ids = {
            category_id
            for category_id, name in categories.items()
            if "plate" in str(name).casefold()
        }
        images = {item.get("id"): item for item in payload.get("images", []) if isinstance(item, dict)}
        grouped: dict[int, list[dict]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            category_id = ann.get("category_id")
            if plate_category_ids and category_id not in plate_category_ids:
                continue
            grouped[ann.get("image_id")].append(ann)
        image_lookup = build_image_index(dataset_root)
        for image_id, anns in grouped.items():
            image_info = images.get(image_id)
            if not image_info:
                continue
            image_path = resolve_image_path(dataset_root, image_lookup, image_info.get("file_name", ""))
            if image_path is None:
                errors.append({"file": str(json_path), "line": 0, "reason": "missing_image", "image_id": image_id})
                continue
            instances: list[dict] = []
            for ann in anns:
                if ann.get("segmentation"):
                    polygon = ann["segmentation"][0]
                elif ann.get("bbox"):
                    x, y, w, h = ann["bbox"]
                    polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
                else:
                    continue
                text = ann.get("text") or ann.get("transcription") or "plate"
                instances.append(build_detection_instance(polygon, text, bool(ann.get("ignore", False))))
            if instances:
                records.append(DetectionRecord("ir_lpr", image_path, build_detection_record(repo_root, image_path, instances)))
    return records, errors


def parse_voc_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for xml_path in sorted(dataset_root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        image_path = find_sidecar_image(xml_path, image_lookup)
        if image_path is None:
            filename_node = tree.find(".//filename")
            if filename_node is not None and filename_node.text:
                image_path = resolve_image_path(dataset_root, image_lookup, filename_node.text.strip())
        if image_path is None:
            errors.append({"file": str(xml_path), "line": 0, "reason": "missing_image"})
            continue
        instances: list[dict] = []
        for obj in tree.findall(".//object"):
            label = normalized_text(obj.findtext("name"), fallback="plate")
            if not is_ir_lpr_plate_label(label):
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            polygon = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            instances.append(build_detection_instance(polygon, "plate"))
        if instances:
            records.append(DetectionRecord("ir_lpr", image_path, build_detection_record(repo_root, image_path, instances)))
    return records, errors


def parse_yolo_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    image_lookup = build_image_index(dataset_root)
    records: list[DetectionRecord] = []
    errors: list[dict] = []
    for txt_path in sorted(dataset_root.rglob("*.txt")):
        image_path = find_sidecar_image(txt_path, image_lookup)
        if image_path is None:
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            continue
        instances: list[dict] = []
        parsed_line = False
        for line_number, raw_line in enumerate(txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            cleaned = raw_line.strip()
            if not cleaned:
                continue
            parts = cleaned.split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 5:
                _, xc, yc, bw, bh = values
                xmin = (xc - bw / 2.0) * width
                ymin = (yc - bh / 2.0) * height
                xmax = (xc + bw / 2.0) * width
                ymax = (yc + bh / 2.0) * height
                polygon = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                instances.append(build_detection_instance(polygon, "plate"))
                parsed_line = True
            elif len(values) >= 9:
                polygon = values[1:9]
                instances.append(build_detection_instance(polygon, "plate"))
                parsed_line = True
        if parsed_line and instances:
            records.append(DetectionRecord("ir_lpr", image_path, build_detection_record(repo_root, image_path, instances)))
        elif parsed_line is False and txt_path.parent != image_path.parent:
            errors.append({"file": str(txt_path), "line": 0, "reason": "unsupported_yolo_label"})
    return records, errors


def load_ir_lpr_detection_records(repo_root: Path, dataset_root: Path) -> tuple[list[DetectionRecord], list[dict]]:
    for loader in (parse_coco_detection_records, parse_voc_detection_records, parse_yolo_detection_records):
        records, errors = loader(repo_root, dataset_root)
        if records:
            return records, errors
    return [], [{"file": str(dataset_root), "line": 0, "reason": "no_supported_ir_lpr_detection_annotations"}]


def compose_ir_lpr_label(xml_path: Path) -> str | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    characters: list[tuple[float, str]] = []
    for obj in root.findall(".//object"):
        label = normalized_text(obj.findtext("name"), fallback="")
        if not label or is_ir_lpr_plate_label(label):
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin"))
        except (TypeError, ValueError):
            continue
        characters.append((xmin, label))

    if not characters:
        return None
    characters.sort(key=lambda item: item[0])
    return normalized_text("".join(label for _, label in characters), fallback="")


def load_ir_lpr_recognition_samples(
    repo_root: Path,
    dataset_root: Path,
    source_name: str,
) -> tuple[list[RecognitionSample], list[dict]]:
    image_lookup = build_image_index(dataset_root)
    samples: list[RecognitionSample] = []
    errors: list[dict] = []

    for xml_path in sorted(dataset_root.rglob("*.xml")):
        image_path = find_sidecar_image(xml_path, image_lookup)
        if image_path is None:
            errors.append({"file": str(xml_path), "line": 0, "reason": "missing_image"})
            continue
        label = compose_ir_lpr_label(xml_path)
        if not label:
            errors.append({"file": str(xml_path), "line": 0, "reason": "missing_ir_lpr_label"})
            continue
        samples.append(
            RecognitionSample(
                source=source_name,
                domain=source_name,
                label=label,
                image_path=image_path,
                repo_relative_path=path_for_manifest(image_path, repo_root),
            )
        )

    if samples:
        return samples, errors
    return load_generic_recognition_samples(repo_root, dataset_root, source_name)


def parse_line_recognition_file(path: Path) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cleaned = raw_line.strip().lstrip("\ufeff")
        if not cleaned:
            continue
        if "\t" in cleaned:
            image_name, label = cleaned.split("\t", 1)
        elif "," in cleaned:
            image_name, label = cleaned.split(",", 1)
        elif ";" in cleaned:
            image_name, label = cleaned.split(";", 1)
        else:
            continue
        parsed.append((image_name.strip(), label.strip()))
    return parsed


def parse_json_recognition_file(path: Path) -> list[tuple[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    parsed: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        if "images" in payload and "annotations" in payload:
            return []
        for key, value in payload.items():
            if isinstance(value, str):
                parsed.append((key, value))
            elif isinstance(value, dict):
                image_name = value.get("img") or value.get("image") or value.get("file_name") or value.get("filename")
                label = value.get("text") or value.get("label") or value.get("transcription")
                if image_name and label:
                    parsed.append((str(image_name), str(label)))
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            image_name = item.get("img") or item.get("image") or item.get("file_name") or item.get("filename")
            label = item.get("text") or item.get("label") or item.get("transcription")
            if image_name and label:
                parsed.append((str(image_name), str(label)))
    return parsed


def load_generic_recognition_samples(repo_root: Path, dataset_root: Path, source_name: str) -> tuple[list[RecognitionSample], list[dict]]:
    image_lookup = build_image_index(dataset_root)
    samples: list[RecognitionSample] = []
    errors: list[dict] = []
    for ann_path in sorted(dataset_root.rglob("*")):
        if not ann_path.is_file() or ann_path.suffix.lower() not in {".txt", ".csv", ".tsv", ".json"}:
            continue
        if ann_path.suffix.lower() == ".json":
            pairs = parse_json_recognition_file(ann_path)
        else:
            pairs = parse_line_recognition_file(ann_path)
        if not pairs:
            continue
        for line_number, (image_ref, label) in enumerate(pairs, start=1):
            image_path = resolve_image_path(dataset_root, image_lookup, image_ref)
            if image_path is None:
                errors.append({"file": str(ann_path), "line": line_number, "reason": "missing_image", "image_ref": image_ref})
                continue
            normalized_label = normalized_text(label)
            if text_is_ignored(normalized_label):
                continue
            samples.append(
                RecognitionSample(
                    source=source_name,
                    domain=source_name,
                    label=normalized_label,
                    image_path=image_path,
                    repo_relative_path=path_for_manifest(image_path, repo_root),
                )
            )
    return samples, errors


def resolve_external_root(repo_root: Path, root: Path | None) -> Path | None:
    if root is None:
        return None
    return root if root.is_absolute() else (repo_root / root).resolve()


def build_external_detection_train_records(
    repo_root: Path,
    external_roots: dict[str, Path | None],
    mlt_scripts: Sequence[str],
) -> tuple[dict[str, list[DetectionRecord]], list[dict]]:
    records_by_dataset: dict[str, list[DetectionRecord]] = {}
    errors: list[dict] = []

    if external_roots.get("icdar2019_mlt") is not None:
        records_by_dataset["icdar2019_mlt"], dataset_errors = load_mlt_detection_records(
            repo_root, external_roots["icdar2019_mlt"], "icdar2019_mlt", mlt_scripts
        )
        errors.extend(dataset_errors)
    if external_roots.get("evarest_detection") is not None:
        records_by_dataset["evarest"], dataset_errors = load_evarest_detection_records(
            repo_root, external_roots["evarest_detection"]
        )
        errors.extend(dataset_errors)
    if external_roots.get("totaltext") is not None:
        records_by_dataset["totaltext"], dataset_errors = load_totaltext_detection_records(
            repo_root, external_roots["totaltext"]
        )
        errors.extend(dataset_errors)
    if external_roots.get("ctw1500") is not None:
        records_by_dataset["ctw1500"], dataset_errors = load_ctw1500_detection_records(
            repo_root, external_roots["ctw1500"]
        )
        errors.extend(dataset_errors)
    if external_roots.get("textocr") is not None:
        records_by_dataset["textocr"], dataset_errors = load_textocr_detection_records(
            repo_root, external_roots["textocr"]
        )
        errors.extend(dataset_errors)
    if external_roots.get("ir_lpr_detection") is not None:
        records_by_dataset["ir_lpr"], dataset_errors = load_ir_lpr_detection_records(
            repo_root, external_roots["ir_lpr_detection"]
        )
        errors.extend(dataset_errors)
    return records_by_dataset, errors


def build_external_recognition_train_samples(
    repo_root: Path,
    external_roots: dict[str, Path | None],
    mlt_scripts: Sequence[str],
) -> tuple[dict[str, list[RecognitionSample]], list[dict]]:
    samples_by_dataset: dict[str, list[RecognitionSample]] = {}
    errors: list[dict] = []

    if external_roots.get("icdar2019_mlt") is not None:
        records, dataset_errors = load_mlt_detection_records(
            repo_root, external_roots["icdar2019_mlt"], "icdar2019_mlt", mlt_scripts
        )
        samples_by_dataset["icdar2019_mlt"] = build_recognition_crop_samples(repo_root, records, "icdar2019_mlt")
        errors.extend(dataset_errors)
    if external_roots.get("evarest_recognition") is not None:
        samples_by_dataset["evarest"], dataset_errors = load_generic_recognition_samples(
            repo_root, external_roots["evarest_recognition"], "evarest"
        )
        errors.extend(dataset_errors)
    elif external_roots.get("evarest_detection") is not None:
        records, dataset_errors = load_evarest_detection_records(repo_root, external_roots["evarest_detection"])
        samples_by_dataset["evarest"] = build_recognition_crop_samples(repo_root, records, "evarest")
        errors.extend(dataset_errors)
    if external_roots.get("totaltext") is not None:
        records, dataset_errors = load_totaltext_detection_records(repo_root, external_roots["totaltext"])
        samples_by_dataset["totaltext"] = build_recognition_crop_samples(repo_root, records, "totaltext")
        errors.extend(dataset_errors)
    if external_roots.get("ctw1500") is not None:
        records, dataset_errors = load_ctw1500_detection_records(repo_root, external_roots["ctw1500"])
        samples_by_dataset["ctw1500"] = build_recognition_crop_samples(repo_root, records, "ctw1500")
        errors.extend(dataset_errors)
    if external_roots.get("textocr") is not None:
        records, dataset_errors = load_textocr_detection_records(repo_root, external_roots["textocr"])
        samples_by_dataset["textocr"] = build_recognition_crop_samples(repo_root, records, "textocr")
        errors.extend(dataset_errors)
    if external_roots.get("ir_lpr_recognition") is not None:
        samples_by_dataset["ir_lpr"], dataset_errors = load_ir_lpr_recognition_samples(
            repo_root, external_roots["ir_lpr_recognition"], "ir_lpr"
        )
        errors.extend(dataset_errors)
    elif external_roots.get("ir_lpr_detection") is not None:
        records, dataset_errors = load_ir_lpr_detection_records(repo_root, external_roots["ir_lpr_detection"])
        samples_by_dataset["ir_lpr"] = build_recognition_crop_samples(repo_root, records, "ir_lpr")
        errors.extend(dataset_errors)
    return samples_by_dataset, errors
