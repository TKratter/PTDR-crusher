#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


RECOGNITION_FILENAME_RE = re.compile(r"img_\d+\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
EMBEDDED_FILENAME_RE = re.compile(r"img_\d+\.(jpg|jpeg|png|bmp)", re.IGNORECASE)


def domain_is_included(domain: str, include_domains: Sequence[str] | None) -> bool:
    if not include_domains:
        return True
    for candidate in include_domains:
        candidate = candidate.strip().strip("/")
        if not candidate:
            continue
        if domain == candidate or domain.startswith(candidate + "/"):
            return True
    return False


def iter_detection_annotation_files(
    dataset_root: Path,
    split: str,
    include_domains: Sequence[str] | None = None,
) -> Iterable[tuple[Path, str]]:
    split_root = dataset_root / split
    for txt_path in sorted(split_root.glob("*/*/*.txt")):
        domain = txt_path.parent.relative_to(split_root).as_posix()
        if domain_is_included(domain, include_domains):
            yield txt_path, domain


def iter_recognition_ground_truth_files(
    dataset_root: Path,
    split: str,
    include_domains: Sequence[str] | None = None,
) -> Iterable[tuple[Path, str]]:
    split_root = dataset_root / split
    for gt_path in sorted(split_root.glob("*/*/gt.*")):
        domain = gt_path.parent.relative_to(split_root).as_posix()
        if domain_is_included(domain, include_domains):
            yield gt_path, domain


def parse_detection_line(line: str) -> tuple[list[float], str]:
    cleaned = line.strip().lstrip("\ufeff")
    if not cleaned:
        raise ValueError("empty line")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) < 9:
        raise ValueError(f"expected at least 9 comma-separated fields, found {len(parts)}")
    try:
        polygon = [float(value) for value in parts[:8]]
    except ValueError as exc:
        raise ValueError("non-numeric polygon coordinate") from exc
    transcription = ",".join(parts[8:]).strip()
    if not transcription:
        raise ValueError("missing transcription")
    return polygon, transcription


def parse_recognition_line(line: str) -> tuple[str, str]:
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
    if not RECOGNITION_FILENAME_RE.match(image_name):
        raise ValueError(f"invalid image name: {image_name}")
    if not label:
        raise ValueError("missing label")
    if EMBEDDED_FILENAME_RE.search(label):
        raise ValueError("label appears to contain a concatenated filename")
    return image_name, label


def validate_detection_dataset(
    dataset_root: Path,
    include_domains: Sequence[str] | None = None,
) -> dict:
    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "error_count": 0,
    }
    total_errors = 0
    for split in ("train", "test"):
        split_summary: dict[str, object] = {
            "files": 0,
            "lines": 0,
            "valid_lines": 0,
            "errors": [],
            "errors_by_type": {},
            "domains": {},
        }
        error_counter: Counter[str] = Counter()
        for txt_path, domain in iter_detection_annotation_files(dataset_root, split, include_domains):
            split_summary["files"] += 1
            domain_summary = split_summary["domains"].setdefault(domain, {"files": 0, "valid_lines": 0})
            domain_summary["files"] += 1
            image_path = txt_path.with_suffix(".jpg")
            if not image_path.exists():
                error_counter["missing_image"] += 1
                split_summary["errors"].append(
                    {"file": str(txt_path), "line": 0, "reason": "missing_image"}
                )
                continue
            for line_number, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
                split_summary["lines"] += 1
                try:
                    parse_detection_line(raw_line)
                except ValueError as exc:
                    error_counter[str(exc)] += 1
                    split_summary["errors"].append(
                        {"file": str(txt_path), "line": line_number, "reason": str(exc)}
                    )
                    continue
                split_summary["valid_lines"] += 1
                domain_summary["valid_lines"] += 1
        split_summary["errors_by_type"] = dict(error_counter)
        total_errors += sum(error_counter.values())
        summary["splits"][split] = split_summary
    summary["error_count"] = total_errors
    return summary


def validate_recognition_dataset(
    dataset_root: Path,
    include_domains: Sequence[str] | None = None,
) -> dict:
    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "error_count": 0,
    }
    total_errors = 0
    for split in ("train", "test"):
        split_summary: dict[str, object] = {
            "files": 0,
            "lines": 0,
            "valid_lines": 0,
            "errors": [],
            "errors_by_type": {},
            "domains": {},
        }
        error_counter: Counter[str] = Counter()
        for gt_path, domain in iter_recognition_ground_truth_files(dataset_root, split, include_domains):
            split_summary["files"] += 1
            domain_summary = split_summary["domains"].setdefault(domain, {"files": 0, "valid_lines": 0})
            domain_summary["files"] += 1
            cropped_dir = gt_path.parent / "cropped"
            if not cropped_dir.exists():
                error_counter["missing_cropped_dir"] += 1
                split_summary["errors"].append(
                    {"file": str(gt_path), "line": 0, "reason": "missing_cropped_dir"}
                )
                continue
            for line_number, raw_line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
                split_summary["lines"] += 1
                try:
                    image_name, _ = parse_recognition_line(raw_line)
                except ValueError as exc:
                    error_counter[str(exc)] += 1
                    split_summary["errors"].append(
                        {"file": str(gt_path), "line": line_number, "reason": str(exc)}
                    )
                    continue
                if not (cropped_dir / image_name).exists():
                    error_counter["missing_image"] += 1
                    split_summary["errors"].append(
                        {
                            "file": str(gt_path),
                            "line": line_number,
                            "reason": "missing_image",
                            "image_name": image_name,
                        }
                    )
                    continue
                split_summary["valid_lines"] += 1
                domain_summary["valid_lines"] += 1
        split_summary["errors_by_type"] = dict(error_counter)
        total_errors += sum(error_counter.values())
        summary["splits"][split] = split_summary
    summary["error_count"] = total_errors
    return summary


def print_summary(summary: dict) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PTDR detection and recognition annotations.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains the dataset directory.",
    )
    parser.add_argument(
        "--include-domain",
        action="append",
        default=[],
        help="Domain prefix to include, for example outdoor_text or outdoor_text/retail_exteriors.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the combined validation report.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with a non-zero status if malformed annotations are found.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    include_domains = args.include_domain or None
    report = {
        "detection": validate_detection_dataset(repo_root / "dataset" / "detection", include_domains),
        "recognition": validate_recognition_dataset(repo_root / "dataset" / "recognition", include_domains),
    }
    print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    has_errors = report["detection"]["error_count"] or report["recognition"]["error_count"]
    return 1 if args.fail_on_error and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
