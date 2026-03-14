#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """# DBNet Train Sample Preview

This notebook shows **actual DBNet training samples after the real train pipeline** used by the current config.

Each comparison draws:

- **green polygons**: GT polygons
- **orange dashed boxes**: GT axis-aligned boxes derived from the polygons

The left panel is the original source image from the mixed train manifest.
The right panel is the transformed image after the configured DBNet training pipeline, including:

- manifest-level dataset mixing
- train-only size limiting for giant source images
- the custom hard augmentation presets
- the base MMOCR DBNet train augmentation stack
"""
        ),
        code_cell(
            """from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from IPython.display import display

REPO_ROOT = Path.cwd().resolve()
repo_root_hint = os.environ.get("PTDR_REPO_ROOT")
candidate_roots = [REPO_ROOT, *REPO_ROOT.parents]
if repo_root_hint:
    candidate_roots.append(Path(repo_root_hint).expanduser())
for candidate in candidate_roots:
    if (candidate / "experiments/ptdr/dbnet_training_preview.py").exists():
        REPO_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate the PTDR repo root from the current notebook working directory.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ptdr.dbnet_training_preview import (
    build_train_preview_pipeline,
    choose_preview_records,
    load_train_preview_records,
    preview_dbnet_train_samples,
    resolve_training_preview_context,
)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
print(f"Repo root: {REPO_ROOT}")
"""
        ),
        code_cell(
            """CONFIG_PATH = REPO_ROOT / "experiments/ptdr/configs/dbnet_multidata_all_r18_hard.yaml"
SAMPLE_COUNT = 6
RANDOM_SEED = 42
SHOW_BBOXES = True

# Set to a list like ["ptdr", "icdar2019_mlt"] to inspect only specific sources.
INCLUDE_SOURCES = None
"""
        ),
        code_cell(
            """context = resolve_training_preview_context(CONFIG_PATH)
repo_root = context["repo_root"]
summary = context["summary"]
pipeline = build_train_preview_pipeline(context["pipeline_cfg"])
records = load_train_preview_records(Path(context["manifests"]["train_ann"]))

print(f"Config: {CONFIG_PATH}")
print(f"Train manifest: {context['manifests']['train_ann']}")
print(f"Train images: {summary['train_images']:,}")
print(f"Val images: {summary['val_images']:,}")
print(f"Test images: {summary['test_images']:,}")
print("\\nConfigured mix summary:")
for name, stats in summary["train_mix"].items():
    print(f"  {name:14s} raw={stats['raw_count']:>6} effective={stats['effective_count']:>6} frac={stats['effective_fraction']:.3f}")

source_counts = Counter(record.source for record in records)
print("\\nActual manifest source counts:")
for name, count in sorted(source_counts.items()):
    print(f"  {name:14s} {count:>6}")

print("\\nTrain pipeline:")
for idx, step in enumerate(context["pipeline_cfg"]):
    print(f"  [{idx}] {step['type']}")
"""
        ),
        code_cell(
            """selected = choose_preview_records(
    records,
    count=SAMPLE_COUNT,
    seed=RANDOM_SEED,
    include_sources=INCLUDE_SOURCES,
)

print(f"Selected {len(selected)} records.")
for item in selected:
    print(f"[{item.index:04d}] {item.source:14s} {item.record['img_path']}")
"""
        ),
        code_cell(
            """for fig in preview_dbnet_train_samples(
    repo_root=repo_root,
    preview_records=selected,
    pipeline=pipeline,
    seed=RANDOM_SEED,
    show_bboxes=SHOW_BBOXES,
):
    display(fig)
    plt.close(fig)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook()
    output_path = Path(__file__).resolve().parent / "dbnet_training_preview.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
