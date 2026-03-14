from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import lmdb

try:
    from .build_recognition_manifest import (
        charset_policy_summary,
        build_recognition_samples,
        normalize_and_filter_samples,
        resolve_charset_policy,
        split_samples,
        write_lmdb,
    )
    from .recognition_augmentations import (
        build_hard_variant_samples,
        build_rotated_variant_samples,
    )
except ImportError:
    from build_recognition_manifest import (
        charset_policy_summary,
        build_recognition_samples,
        normalize_and_filter_samples,
        resolve_charset_policy,
        split_samples,
        write_lmdb,
    )
    from recognition_augmentations import build_hard_variant_samples, build_rotated_variant_samples


def _lmdb_ready(root: Path) -> bool:
    return (root / "data.mdb").exists()


def _variant_summary(samples: Sequence[dict]) -> dict:
    domains: dict[str, int] = {}
    max_label_length = 0
    for sample in samples:
        domain = str(sample["domain"])
        domains[domain] = domains.get(domain, 0) + 1
        max_label_length = max(max_label_length, len(sample["label"]))
    return {
        "count": len(samples),
        "max_label_length": max_label_length,
        "domains": dict(sorted(domains.items())),
    }


def _read_lmdb_variant_summary(lmdb_root: Path) -> dict | None:
    if not _lmdb_ready(lmdb_root):
        return None
    env = lmdb.open(str(lmdb_root), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin() as txn:
            raw_count = txn.get(b"num-samples")
            if raw_count is None:
                return None
            count = int(raw_count.decode("ascii"))
            max_label_length = 0
            for index in range(1, count + 1):
                label_key = f"label-{index:09d}".encode("ascii")
                raw_label = txn.get(label_key)
                if raw_label is None:
                    continue
                max_label_length = max(max_label_length, len(raw_label.decode("utf-8")))
    finally:
        env.close()
    return {
        "count": count,
        "max_label_length": max_label_length,
        "domains": {"PTDR": count},
    }


def _reuse_existing_cache_if_possible(
    output_root: Path,
    rotation_angles: Sequence[int],
    resolved_charset_policy: dict | None,
    include_domains: Sequence[str] | None,
    val_ratio: float,
    split_seed: int,
    hard_seed: int,
) -> dict[str, Path] | None:
    variant_roots = {
        "val": output_root / "val" / "PTDR",
        "test": output_root / "test" / "PTDR",
        **{f"val_rot{int(angle)}": output_root / f"val_rot{int(angle)}" / "PTDR" for angle in rotation_angles},
        "val_hard": output_root / "val_hard" / "PTDR",
        "test_hard": output_root / "test_hard" / "PTDR",
    }
    if not all(_lmdb_ready(root) for root in variant_roots.values()):
        return None

    variant_summaries: dict[str, dict] = {}
    for name, root in variant_roots.items():
        summary = _read_lmdb_variant_summary(root)
        if summary is None:
            return None
        variant_summaries[name] = summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "seed": hard_seed,
                "split_seed": split_seed,
                "val_ratio": val_ratio,
                "include_domains": list(include_domains or []),
                "rotation_angles": [int(angle) for angle in rotation_angles],
                "variants": variant_summaries,
                "charset_policy": None
                if resolved_charset_policy is None
                else charset_policy_summary(resolved_charset_policy),
                "errors": [],
                "reused_existing_cache": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "val_root": output_root / "val",
        "test_root": output_root / "test",
        **{f"val_rot{int(angle)}_root": output_root / f"val_rot{int(angle)}" for angle in rotation_angles},
        "val_hard_root": output_root / "val_hard",
        "test_hard_root": output_root / "test_hard",
        "summary": summary_path,
    }


def build_recognition_eval_variants(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    include_domains: Sequence[str] | None,
    val_ratio: float,
    split_seed: int,
    map_size_bytes: int,
    charset_policy=None,
    hard_seed: int = 42,
    rotation_angles: Sequence[int] = (90, 180, 270),
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_charset_policy = resolve_charset_policy(repo_root, charset_policy)
    expected = {
        "val_root": output_root / "val" / "PTDR" / "data.mdb",
        "test_root": output_root / "test" / "PTDR" / "data.mdb",
        **{
            f"val_rot{int(angle)}_root": output_root / f"val_rot{int(angle)}" / "PTDR" / "data.mdb"
            for angle in rotation_angles
        },
        "val_hard_root": output_root / "val_hard" / "PTDR" / "data.mdb",
        "test_hard_root": output_root / "test_hard" / "PTDR" / "data.mdb",
        "summary": output_root / "summary.json",
    }
    if all(path.exists() for path in expected.values()):
        try:
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            summary = None
        if summary is not None and summary.get("charset_policy") == charset_policy_summary(resolved_charset_policy):
            return {
                "val_root": output_root / "val",
                "test_root": output_root / "test",
                **{f"val_rot{int(angle)}_root": output_root / f"val_rot{int(angle)}" for angle in rotation_angles},
                "val_hard_root": output_root / "val_hard",
                "test_hard_root": output_root / "test_hard",
                "summary": output_root / "summary.json",
            }
        shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    reused_outputs = _reuse_existing_cache_if_possible(
        output_root=output_root,
        rotation_angles=rotation_angles,
        resolved_charset_policy=resolved_charset_policy,
        include_domains=include_domains,
        val_ratio=val_ratio,
        split_seed=split_seed,
        hard_seed=hard_seed,
    )
    if reused_outputs is not None:
        return reused_outputs
    raw_train_samples, train_errors = build_recognition_samples(
        repo_root=repo_root,
        dataset_root=dataset_root,
        split="train",
        include_domains=include_domains,
    )
    raw_test_samples, test_errors = build_recognition_samples(
        repo_root=repo_root,
        dataset_root=dataset_root,
        split="test",
        include_domains=include_domains,
    )
    train_samples, train_filter_errors = normalize_and_filter_samples(
        raw_train_samples,
        resolved_charset_policy,
        split_name="shared_eval_train_source",
    )
    test_samples, test_filter_errors = normalize_and_filter_samples(
        raw_test_samples,
        resolved_charset_policy,
        split_name="shared_eval_test",
    )
    _, val_samples = split_samples(train_samples, val_ratio=val_ratio, seed=split_seed)

    val_root = output_root / "val" / "PTDR"
    test_root = output_root / "test" / "PTDR"
    write_lmdb(val_samples, val_root, map_size_bytes)
    write_lmdb(test_samples, test_root, map_size_bytes)

    outputs = {
        "val_root": val_root.parent,
        "test_root": test_root.parent,
    }
    summary: dict[str, object] = {
        "seed": hard_seed,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "include_domains": list(include_domains or []),
        "rotation_angles": [int(angle) for angle in rotation_angles],
        "variants": {
            "val": _variant_summary(val_samples),
            "test": _variant_summary(test_samples),
        },
        "charset_policy": None
        if resolved_charset_policy is None
        else charset_policy_summary(resolved_charset_policy),
        "errors": train_errors + test_errors + train_filter_errors + test_filter_errors,
    }

    for angle in rotation_angles:
        variant_name = f"val_rot{int(angle)}"
        variant_samples = build_rotated_variant_samples(val_samples, angle=int(angle), seed=hard_seed)
        variant_root = output_root / variant_name / "PTDR"
        write_lmdb(variant_samples, variant_root, map_size_bytes)
        outputs[f"{variant_name}_root"] = variant_root.parent
        summary["variants"][variant_name] = _variant_summary(variant_samples)

    val_hard_samples, val_hard_preset_counts = build_hard_variant_samples(val_samples, split_name="val_hard", seed=hard_seed)
    val_hard_root = output_root / "val_hard" / "PTDR"
    write_lmdb(val_hard_samples, val_hard_root, map_size_bytes)
    outputs["val_hard_root"] = val_hard_root.parent
    summary["variants"]["val_hard"] = {
        **_variant_summary(val_hard_samples),
        "preset_counts": val_hard_preset_counts,
    }

    test_hard_samples, test_hard_preset_counts = build_hard_variant_samples(
        test_samples,
        split_name="test_hard",
        seed=hard_seed,
    )
    test_hard_root = output_root / "test_hard" / "PTDR"
    write_lmdb(test_hard_samples, test_hard_root, map_size_bytes)
    outputs["test_hard_root"] = test_hard_root.parent
    summary["variants"]["test_hard"] = {
        **_variant_summary(test_hard_samples),
        "preset_counts": test_hard_preset_counts,
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs
