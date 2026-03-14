#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lmdb
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_dataset import iter_recognition_ground_truth_files, parse_recognition_line
from external_datasets import mix_train_items, train_mix_summary
from text_normalization import canonicalize_arabic_persian_text


CONTROL_CHARS = {"\n", "\r", "\t", "\x0b", "\x0c"}


@dataclass(frozen=True)
class ExtraRecognitionTrainSource:
    dataset_root: Path
    include_domains: Sequence[str] | None = None
    name: str | None = None
    layout: str = "auto"


def recognition_manifest_outputs(output_root: Path) -> dict[str, Path | str]:
    return {
        "root_dir": output_root,
        "train_dir": "real",
        "charset_train": output_root / "charset_train.txt",
        "charset_eval": output_root / "charset_eval.txt",
        "summary": output_root / "summary.json",
    }


def charset_policy_summary(resolved_charset_policy: dict | None) -> dict | None:
    if resolved_charset_policy is None:
        return None
    return {
        "base_charset_path": resolved_charset_policy["base_charset_path"],
        "allow_arabic_extras_only": resolved_charset_policy["allow_arabic_extras_only"],
        "normalize_arabic_indic_digits": resolved_charset_policy["normalize_arabic_indic_digits"],
        "arabic_indic_digit_target": resolved_charset_policy["arabic_indic_digit_target"],
        "normalize_equivalent_arabic_persian_letters": resolved_charset_policy["normalize_equivalent_arabic_persian_letters"],
        "arabic_persian_letter_target": resolved_charset_policy["arabic_persian_letter_target"],
        "drop_unsupported_labels": resolved_charset_policy["drop_unsupported_labels"],
    }


def recognition_manifests_exist(output_root: Path, repo_root: Path | None = None, charset_policy=None) -> bool:
    outputs = recognition_manifest_outputs(output_root)
    required = [
        output_root / "train" / "real" / "ptdr" / "data.mdb",
        output_root / "val" / "PTDR" / "data.mdb",
        output_root / "test" / "PTDR" / "data.mdb",
        outputs["charset_train"],
        outputs["charset_eval"],
        outputs["summary"],
    ]
    if not all(Path(path).exists() for path in required):
        return False
    if repo_root is None:
        return True
    try:
        summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    except Exception:
        return False
    resolved_charset_policy = resolve_charset_policy(repo_root, charset_policy)
    return summary.get("charset_policy") == charset_policy_summary(resolved_charset_policy)


def path_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def build_recognition_samples(
    repo_root: Path,
    dataset_root: Path,
    split: str,
    include_domains: Sequence[str] | None,
    domain_prefix: str | None = None,
) -> tuple[list[dict], list[dict]]:
    samples: list[dict] = []
    errors: list[dict] = []
    for gt_path, domain in iter_recognition_ground_truth_files(dataset_root, split, include_domains):
        cropped_dir = gt_path.parent / "cropped"
        for line_number, raw_line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                image_name, label = parse_recognition_line(raw_line)
            except ValueError as exc:
                errors.append({"file": str(gt_path), "line": line_number, "reason": str(exc)})
                continue
            image_path = cropped_dir / image_name
            if not image_path.exists():
                errors.append(
                    {"file": str(gt_path), "line": line_number, "reason": "missing_image", "image_name": image_name}
                )
                continue
            sample_domain = domain if domain_prefix is None else f"{domain_prefix}/{domain}"
            samples.append(
                {
                    "domain": sample_domain,
                    "image_path": image_path,
                    "repo_relative_path": path_for_manifest(image_path, repo_root),
                    "label": label,
                }
            )
    return samples, errors


def parse_flat_recognition_line(line: str) -> tuple[str, str]:
    cleaned = line.rstrip("\n\r").lstrip("\ufeff")
    if not cleaned.strip():
        raise ValueError("empty line")
    if "," not in cleaned:
        raise ValueError("missing comma separator")
    image_name, label = cleaned.split(",", 1)
    image_name = image_name.strip()
    label = label.strip()
    if not image_name:
        raise ValueError("missing image name")
    image_path = Path(image_name)
    if image_path.name != image_name:
        raise ValueError("image name must not contain path separators")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise ValueError(f"invalid image extension: {image_path.suffix}")
    if not label:
        raise ValueError("missing label")
    return image_name, label


def build_flat_recognition_samples(
    repo_root: Path,
    dataset_root: Path,
    gt_filename: str,
    image_dirname: str,
    domain_name: str,
) -> tuple[list[dict], list[dict]]:
    gt_path = dataset_root / gt_filename
    image_dir = dataset_root / image_dirname
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing flat recognition ground-truth file: {gt_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing flat recognition image directory: {image_dir}")

    samples: list[dict] = []
    errors: list[dict] = []
    for line_number, raw_line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            image_name, label = parse_flat_recognition_line(raw_line)
        except ValueError as exc:
            errors.append({"file": str(gt_path), "line": line_number, "reason": str(exc)})
            continue
        image_path = image_dir / image_name
        if not image_path.exists():
            errors.append(
                {"file": str(gt_path), "line": line_number, "reason": "missing_image", "image_name": image_name}
            )
            continue
        samples.append(
            {
                "domain": domain_name,
                "image_path": image_path,
                "repo_relative_path": path_for_manifest(image_path, repo_root),
                "label": label,
            }
        )
    return samples, errors


def detect_extra_recognition_layout(source: ExtraRecognitionTrainSource) -> str:
    if source.layout != "auto":
        return source.layout
    if (source.dataset_root / "train").exists():
        return "standard"
    if (source.dataset_root / "gt.txt").exists() and (source.dataset_root / "images").exists():
        return "flat"
    raise FileNotFoundError(
        "Could not detect recognition dataset layout for "
        f"{source.dataset_root}. Expected either train/.../gt.* + cropped/ or flat gt.txt + images/."
    )


def build_samples_for_extra_recognition_source(
    repo_root: Path,
    source: ExtraRecognitionTrainSource,
) -> tuple[list[dict], list[dict], str]:
    source_name = source.name or source.dataset_root.name
    layout = detect_extra_recognition_layout(source)
    if layout == "standard":
        samples, errors = build_recognition_samples(
            repo_root=repo_root,
            dataset_root=source.dataset_root,
            split="train",
            include_domains=source.include_domains,
            domain_prefix=source_name,
        )
        return samples, errors, layout
    if layout == "flat":
        samples, errors = build_flat_recognition_samples(
            repo_root=repo_root,
            dataset_root=source.dataset_root,
            gt_filename="gt.txt",
            image_dirname="images",
            domain_name=f"{source_name}/synthetic",
        )
        return samples, errors, layout
    raise ValueError(f"Unsupported extra recognition dataset layout: {layout}")


def split_samples(
    samples: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["domain"]].append(sample)
    rng = random.Random(seed)
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    for domain, bucket in sorted(grouped.items()):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        if val_ratio <= 0 or len(shuffled) <= 1:
            train_samples.extend(shuffled)
            continue
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_count = min(val_count, len(shuffled) - 1)
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])
    return train_samples, val_samples


def materialize_sample(sample) -> dict:
    if isinstance(sample, dict):
        return sample
    return {
        "domain": getattr(sample, "domain"),
        "image_path": getattr(sample, "image_path", None),
        "image_bytes": getattr(sample, "image_bytes", None),
        "crop_box": getattr(sample, "crop_box", None),
        "repo_relative_path": getattr(sample, "repo_relative_path", ""),
        "label": getattr(sample, "label"),
    }


def encode_crop_from_box(image_path: Path, crop_box: Sequence[float]) -> bytes:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"L", "LA", "RGB", "RGBA"}:
            image = image.convert("RGB")
        width, height = image.size
        left = max(0, min(width, int(crop_box[0])))
        upper = max(0, min(height, int(crop_box[1])))
        right = max(left + 1, min(width, int(crop_box[2])))
        lower = max(upper + 1, min(height, int(crop_box[3])))
        patch = image.crop((left, upper, right, lower))
        if patch.mode not in {"L", "LA", "RGB", "RGBA"}:
            patch = patch.convert("RGB")
        buffer = io.BytesIO()
        try:
            patch.save(buffer, format="PNG")
        except OSError:
            buffer = io.BytesIO()
            patch.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def write_lmdb(samples: list[dict], lmdb_path: Path, map_size_bytes: int) -> None:
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size_bytes)
    cache: dict[bytes, bytes] = {}
    txn = env.begin(write=True)
    for index, sample in enumerate(samples, start=1):
        cache[f"label-{index:09d}".encode("ascii")] = sample["label"].encode("utf-8")
        if sample.get("image_bytes") is not None:
            cache[f"image-{index:09d}".encode("ascii")] = sample["image_bytes"]
        elif sample.get("crop_box") is not None:
            cache[f"image-{index:09d}".encode("ascii")] = encode_crop_from_box(sample["image_path"], sample["crop_box"])
        else:
            cache[f"image-{index:09d}".encode("ascii")] = sample["image_path"].read_bytes()
        if len(cache) >= 2000:
            for key, value in cache.items():
                txn.put(key, value)
            txn.commit()
            txn = env.begin(write=True)
            cache.clear()
    cache[b"num-samples"] = str(len(samples)).encode("ascii")
    for key, value in cache.items():
        txn.put(key, value)
    txn.commit()
    env.close()


def build_charset(samples: list[dict]) -> str:
    charset = sorted(
        {
            character
            for sample in samples
            for character in sample["label"]
            if character not in CONTROL_CHARS
        }
    )
    if not charset:
        raise ValueError(
            "Generated an empty charset from the selected recognition samples. "
            "Check include_domains, label parsing, and whether the selected splits contain valid labels."
        )
    return "".join(charset)


def resolve_charset_policy(repo_root: Path, charset_policy) -> dict | None:
    if charset_policy is None:
        return None
    def cfg_get(name: str, default=None):
        if isinstance(charset_policy, dict):
            return charset_policy.get(name, default)
        return getattr(charset_policy, name, default)

    base_charset_path = cfg_get("base_charset_path", None)
    if base_charset_path is not None:
        base_charset_path = Path(base_charset_path)
        if not base_charset_path.is_absolute():
            base_charset_path = (repo_root / base_charset_path).resolve()

    allow_arabic_extras_only = bool(cfg_get("allow_arabic_extras_only", False))
    normalize_arabic_indic_digits = bool(cfg_get("normalize_arabic_indic_digits", False))
    normalize_equivalent_arabic_persian_letters = bool(cfg_get("normalize_equivalent_arabic_persian_letters", False))
    drop_unsupported_labels = bool(cfg_get("drop_unsupported_labels", False))
    digit_target = cfg_get("arabic_indic_digit_target", "persian")
    letter_target = cfg_get("arabic_persian_letter_target", "persian")

    if not any(
        (
            base_charset_path is not None,
            allow_arabic_extras_only,
            normalize_arabic_indic_digits,
            normalize_equivalent_arabic_persian_letters,
            drop_unsupported_labels,
        )
    ):
        return None

    base_charset = set()
    if base_charset_path is not None:
        if not base_charset_path.exists():
            raise FileNotFoundError(f"Missing base charset file for PARSeq charset policy: {base_charset_path}")
        base_charset = set(base_charset_path.read_text(encoding="utf-8"))
    if allow_arabic_extras_only and not base_charset:
        raise ValueError("allow_arabic_extras_only requires a non-empty base_charset_path.")

    if digit_target not in {"ascii", "persian"}:
        raise ValueError("arabic_indic_digit_target must be either 'ascii' or 'persian'.")
    if letter_target not in {"persian"}:
        raise ValueError("arabic_persian_letter_target must currently be 'persian'.")

    return {
        "base_charset_path": str(base_charset_path) if base_charset_path is not None else None,
        "base_charset": base_charset,
        "allow_arabic_extras_only": allow_arabic_extras_only,
        "normalize_arabic_indic_digits": normalize_arabic_indic_digits,
        "arabic_indic_digit_target": digit_target,
        "normalize_equivalent_arabic_persian_letters": normalize_equivalent_arabic_persian_letters,
        "arabic_persian_letter_target": letter_target,
        "drop_unsupported_labels": drop_unsupported_labels,
    }


def normalize_label_for_charset_policy(label: str, charset_policy: dict | None) -> str:
    if charset_policy is None:
        return label
    return canonicalize_arabic_persian_text(
        label,
        normalize_unicode=False,
        digit_target=charset_policy["arabic_indic_digit_target"],
        normalize_digits=charset_policy["normalize_arabic_indic_digits"],
        canonical_letter_target=charset_policy["arabic_persian_letter_target"],
        normalize_equivalent_letters=charset_policy["normalize_equivalent_arabic_persian_letters"],
    )


def is_allowed_charset_character(character: str, charset_policy: dict | None) -> bool:
    if character in CONTROL_CHARS:
        return False
    if charset_policy is None:
        return True
    if character in charset_policy["base_charset"]:
        return True
    if charset_policy["allow_arabic_extras_only"]:
        return "ARABIC" in unicodedata.name(character, "")
    return True


def normalize_and_filter_samples(
    samples: list[dict],
    charset_policy: dict | None,
    split_name: str,
) -> tuple[list[dict], list[dict]]:
    if charset_policy is None:
        return [materialize_sample(sample) for sample in samples], []

    filtered_samples: list[dict] = []
    filtered_errors: list[dict] = []
    for sample in samples:
        materialized = materialize_sample(sample)
        normalized_label = normalize_label_for_charset_policy(materialized["label"], charset_policy)
        unsupported_chars = sorted(
            {
                character
                for character in normalized_label
                if character not in CONTROL_CHARS and not is_allowed_charset_character(character, charset_policy)
            }
        )
        if unsupported_chars and charset_policy["drop_unsupported_labels"]:
            filtered_errors.append(
                {
                    "split": split_name,
                    "reason": "unsupported_charset",
                    "label": normalized_label,
                    "unsupported_characters": unsupported_chars,
                    "repo_relative_path": materialized.get("repo_relative_path", ""),
                }
            )
            continue
        materialized["label"] = normalized_label
        filtered_samples.append(materialized)
    return filtered_samples, filtered_errors


def build_summary(samples: list[dict]) -> dict:
    domain_counts: dict[str, int] = defaultdict(int)
    max_label_length = 0
    for sample in samples:
        domain_counts[sample["domain"]] += 1
        max_label_length = max(max_label_length, len(sample["label"]))
    return {
        "count": len(samples),
        "max_label_length": max_label_length,
        "domains": dict(sorted(domain_counts.items())),
    }


def partition_samples_by_label_overlap(samples: list[dict], forbidden_labels: set[str]) -> tuple[list[dict], list[dict]]:
    if not forbidden_labels:
        return list(samples), []
    kept_samples: list[dict] = []
    overlapping_samples: list[dict] = []
    for sample in samples:
        if sample["label"] in forbidden_labels:
            overlapping_samples.append(sample)
        else:
            kept_samples.append(sample)
    return kept_samples, overlapping_samples


def build_recognition_manifests(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    include_domains: Sequence[str] | None,
    val_ratio: float,
    seed: int,
    map_size_bytes: int,
    extra_train_sources: Sequence[ExtraRecognitionTrainSource] | None = None,
    external_train_samples: dict[str, list[dict]] | None = None,
    train_mix: dict[str, float] | None = None,
    min_ptdr_fraction: float = 0.3,
    extra_errors: Sequence[dict] | None = None,
    charset_policy=None,
    include_ptdr_train_in_train_split: bool = True,
    exclude_extra_train_label_overlap_from_eval: bool = False,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    if recognition_manifests_exist(output_root, repo_root=repo_root, charset_policy=charset_policy):
        return recognition_manifest_outputs(output_root)

    resolved_charset_policy = resolve_charset_policy(repo_root, charset_policy)
    if output_root.exists():
        shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    raw_train_samples, train_errors = build_recognition_samples(
        repo_root=repo_root, dataset_root=dataset_root, split="train", include_domains=include_domains
    )
    test_samples, test_errors = build_recognition_samples(
        repo_root=repo_root, dataset_root=dataset_root, split="test", include_domains=include_domains
    )
    raw_train_samples, raw_train_filter_errors = normalize_and_filter_samples(
        raw_train_samples,
        resolved_charset_policy,
        split_name="train_source",
    )
    test_samples, test_filter_errors = normalize_and_filter_samples(
        test_samples,
        resolved_charset_policy,
        split_name="test",
    )

    extra_train_samples: list[dict] = []
    extra_train_errors: list[dict] = []
    extra_train_summary: dict[str, dict] = {}
    overlap_filtered_train_samples: list[dict] = []
    for source in extra_train_sources or []:
        source_name = source.name or source.dataset_root.name
        source_samples, source_errors, source_layout = build_samples_for_extra_recognition_source(
            repo_root=repo_root,
            source=source,
        )
        normalized_source_samples, source_filter_errors = normalize_and_filter_samples(
            source_samples,
            resolved_charset_policy,
            split_name=f"extra_train/{source_name}",
        )
        extra_train_samples.extend(normalized_source_samples)
        extra_train_errors.extend(source_errors)
        extra_train_errors.extend(source_filter_errors)
        extra_train_summary[source_name] = {
            "dataset_root": str(source.dataset_root),
            "layout": source_layout,
            "include_domains": list(source.include_domains or []),
            **build_summary(normalized_source_samples),
        }

    overlap_filter_summary = None
    if exclude_extra_train_label_overlap_from_eval and extra_train_samples:
        extra_train_labels = {sample["label"] for sample in extra_train_samples}
        filtered_train_source_samples, overlap_filtered_train_samples = partition_samples_by_label_overlap(
            raw_train_samples, extra_train_labels
        )
        filtered_test_samples, overlap_filtered_test_samples = partition_samples_by_label_overlap(
            test_samples, extra_train_labels
        )
        overlap_filter_summary = {
            "enabled": True,
            "extra_train_unique_labels": len(extra_train_labels),
            "ptdr_train_overlap_samples": len(overlap_filtered_train_samples),
            "ptdr_train_overlap_unique_labels": len({sample["label"] for sample in overlap_filtered_train_samples}),
            "ptdr_test_overlap_samples": len(overlap_filtered_test_samples),
            "ptdr_test_overlap_unique_labels": len({sample["label"] for sample in overlap_filtered_test_samples}),
        }
        raw_train_samples = filtered_train_source_samples
        test_samples = filtered_test_samples

    ptdr_train_candidates, val_samples = split_samples(raw_train_samples, val_ratio=val_ratio, seed=seed)
    primary_train_samples = list(ptdr_train_candidates) if include_ptdr_train_in_train_split else []
    if include_ptdr_train_in_train_split and overlap_filtered_train_samples:
        primary_train_samples.extend(overlap_filtered_train_samples)

    mixed_train_samples = list(primary_train_samples)
    mixed_train_summary = None
    if external_train_samples:
        normalized_external_train_samples: dict[str, list[dict]] = {}
        external_filter_errors: list[dict] = []
        for dataset_name, samples in external_train_samples.items():
            normalized_samples, sample_errors = normalize_and_filter_samples(
                samples,
                resolved_charset_policy,
                split_name=f"external_train/{dataset_name}",
            )
            normalized_external_train_samples[dataset_name] = normalized_samples
            external_filter_errors.extend(sample_errors)
        mixed_items, mix_details = mix_train_items(
            {"ptdr": list(primary_train_samples), **normalized_external_train_samples},
            train_mix=train_mix or {"ptdr": 1.0},
            seed=seed,
            min_ptdr_fraction=min_ptdr_fraction,
        )
        mixed_train_samples = [materialize_sample(sample) for sample in mixed_items]
        mixed_train_summary = train_mix_summary(mix_details)
        extra_train_errors.extend(external_filter_errors)

    train_samples = mixed_train_samples + extra_train_samples
    train_charset = build_charset(train_samples)
    eval_charset = build_charset(train_samples + val_samples + test_samples)

    train_lmdb = output_root / "train" / "real" / "ptdr"
    val_lmdb = output_root / "val" / "PTDR"
    test_lmdb = output_root / "test" / "PTDR"
    write_lmdb(train_samples, train_lmdb, map_size_bytes)
    write_lmdb(val_samples, val_lmdb, map_size_bytes)
    write_lmdb(test_samples, test_lmdb, map_size_bytes)

    train_charset_path = output_root / "charset_train.txt"
    eval_charset_path = output_root / "charset_eval.txt"
    train_charset_path.write_text(train_charset, encoding="utf-8")
    eval_charset_path.write_text(eval_charset, encoding="utf-8")

    summary = {
        "include_domains": list(include_domains or []),
        "seed": seed,
        "val_ratio": val_ratio,
        "charset_train_size": len(train_charset),
        "charset_eval_size": len(eval_charset),
        "extra_train_dataset_count": len(extra_train_summary),
        "extra_train_datasets": extra_train_summary,
        "exclude_extra_train_label_overlap_from_eval": exclude_extra_train_label_overlap_from_eval,
        "include_ptdr_train_in_train_split": include_ptdr_train_in_train_split,
        "extra_train_label_overlap_filter": overlap_filter_summary,
        "train": build_summary(train_samples),
        "train_primary": build_summary(primary_train_samples),
        "train_mixed": build_summary(mixed_train_samples),
        "val": build_summary(val_samples),
        "test": build_summary(test_samples),
        "external_train_dataset_count": sum(1 for samples in (external_train_samples or {}).values() if samples),
        "train_mix": mixed_train_summary,
        "charset_policy": {
            **charset_policy_summary(resolved_charset_policy),
        }
        if resolved_charset_policy is not None
        else None,
        "errors": train_errors
        + test_errors
        + raw_train_filter_errors
        + test_filter_errors
        + extra_train_errors
        + list(extra_errors or []),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "root_dir": output_root,
        "train_dir": train_lmdb.parent.name,
        "charset_train": train_charset_path,
        "charset_eval": eval_charset_path,
        "summary": summary_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PARSeq LMDB manifests for PTDR.")
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
        help="Optional recognition dataset root override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" / "parseq",
        help="Output directory for the generated LMDB tree.",
    )
    parser.add_argument(
        "--include-domain",
        action="append",
        default=[],
        help="Domain prefix to include, for example outdoor_text or document/printed.",
    )
    parser.add_argument(
        "--extra-train-dataset-root",
        action="append",
        default=[],
        help=(
            "Optional extra recognition dataset root to append to the training split only. "
            "Supported layouts are train/.../gt.* + cropped/ and flat gt.txt + images/."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction sampled from train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=8.0,
        help="LMDB map size in GiB. Increase if LMDB reports map_full.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_root = (args.dataset_root or repo_root / "dataset" / "recognition").resolve()
    extra_train_sources = [
        ExtraRecognitionTrainSource(dataset_root=Path(path).resolve(), include_domains=args.include_domain or None)
        for path in args.extra_train_dataset_root
    ]
    result = build_recognition_manifests(
        repo_root=repo_root,
        dataset_root=dataset_root,
        output_root=args.output_root.resolve(),
        include_domains=args.include_domain or None,
        val_ratio=args.val_ratio,
        seed=args.seed,
        map_size_bytes=int(args.map_size_gb * (1024**3)),
        extra_train_sources=extra_train_sources,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
