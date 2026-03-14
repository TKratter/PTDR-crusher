from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

try:
    from .validate_dataset import iter_detection_annotation_files, parse_detection_line
    from .detection_augmentations import (
        DEFAULT_PRESETS,
        apply_preset_to_image_instances,
        bbox_from_polygon,
        polygon_to_array,
    )
except ImportError:
    from validate_dataset import iter_detection_annotation_files, parse_detection_line
    from detection_augmentations import DEFAULT_PRESETS, apply_preset_to_image_instances, bbox_from_polygon, polygon_to_array


@dataclass(frozen=True)
class DetectionSample:
    split: str
    domain: str
    annotation_path: Path
    image_path: Path
    instances: list[dict]


def load_ptdr_detection_samples(
    repo_root: Path,
    split: str = "train",
    include_domains: Sequence[str] | None = None,
) -> list[DetectionSample]:
    dataset_root = repo_root / "dataset" / "detection"
    samples: list[DetectionSample] = []
    for txt_path, domain in iter_detection_annotation_files(dataset_root, split, include_domains):
        image_path = txt_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        instances: list[dict] = []
        for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
            try:
                polygon, text = parse_detection_line(raw_line)
            except ValueError:
                continue
            instances.append({"polygon": polygon, "text": text})
        if not instances:
            continue
        samples.append(
            DetectionSample(
                split=split,
                domain=domain,
                annotation_path=txt_path,
                image_path=image_path,
                instances=instances,
            )
        )
    return samples


def choose_samples(samples: Sequence[DetectionSample], count: int, seed: int) -> list[DetectionSample]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    items = list(samples)
    if count >= len(items):
        return items
    return rng.sample(items, count)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def apply_preset(sample: DetectionSample, preset: dict, seed: int) -> tuple[np.ndarray, list[dict]]:
    image = load_rgb_image(sample.image_path)
    return apply_preset_to_image_instances(image=image, instances=sample.instances, preset=preset, seed=seed)


def build_variant_bundle(
    sample: DetectionSample,
    presets: Sequence[dict],
    seed: int,
) -> list[tuple[str, np.ndarray, list[dict]]]:
    bundle: list[tuple[str, np.ndarray, list[dict]]] = [("original", load_rgb_image(sample.image_path), copy.deepcopy(sample.instances))]
    for index, preset in enumerate(presets, start=1):
        name = str(preset.get("name", f"preset_{index}"))
        image, instances = apply_preset(sample, preset=preset, seed=seed + (index * 1009))
        bundle.append((name, image, instances))
    return bundle


def _label_text(text: str, max_chars: int = 24) -> str:
    value = text.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "..."


def plot_detection_view(
    ax,
    image: np.ndarray,
    instances: Sequence[dict],
    title: str,
    show_bbox: bool = True,
    show_text: bool = False,
    max_labels: int = 20,
) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    for index, instance in enumerate(instances):
        points = polygon_to_array(instance["polygon"])
        ax.add_patch(
            PolygonPatch(
                points,
                closed=True,
                fill=False,
                edgecolor="#00ff99",
                linewidth=1.6,
            )
        )
        if show_bbox:
            x0, y0, x1, y1 = bbox_from_polygon(points)
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#ffb000",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        if show_text and index < max_labels:
            text = _label_text(str(instance.get("text", "")))
            if text:
                x0, y0, _, _ = bbox_from_polygon(points)
                ax.text(
                    x0,
                    max(0.0, y0 - 4.0),
                    text,
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 1.5},
                )


def preview_sample_variants(
    sample: DetectionSample,
    presets: Sequence[dict],
    seed: int = 42,
    columns: int = 3,
    show_bbox: bool = True,
    show_text: bool = False,
    max_labels: int = 20,
    figsize_scale: float = 5.0,
):
    bundle = build_variant_bundle(sample, presets=presets, seed=seed)
    columns = max(1, columns)
    rows = int(np.ceil(len(bundle) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * figsize_scale, rows * figsize_scale))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image, instances) in zip(axes.ravel(), bundle):
        plot_detection_view(
            ax,
            image=image,
            instances=instances,
            title=title,
            show_bbox=show_bbox,
            show_text=show_text,
            max_labels=max_labels,
        )
    figure.suptitle(f"{sample.split} / {sample.domain} / {sample.image_path.name}", fontsize=14)
    figure.tight_layout()
    return figure


def preview_preset_across_samples(
    samples: Sequence[DetectionSample],
    preset: dict,
    seed: int = 42,
    show_bbox: bool = True,
    show_text: bool = False,
    max_labels: int = 20,
    figsize_scale: float = 4.6,
):
    rows = len(samples)
    if rows <= 0:
        raise ValueError("At least one sample is required.")
    figure, axes = plt.subplots(rows, 2, figsize=(2 * figsize_scale, rows * figsize_scale))
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(samples):
        original = load_rgb_image(sample.image_path)
        augmented, transformed_instances = apply_preset(sample, preset=preset, seed=seed + row * 997)
        plot_detection_view(
            axes[row, 0],
            image=original,
            instances=sample.instances,
            title=f"original: {sample.image_path.name}",
            show_bbox=show_bbox,
            show_text=show_text,
            max_labels=max_labels,
        )
        plot_detection_view(
            axes[row, 1],
            image=augmented,
            instances=transformed_instances,
            title=f"augmented: {preset.get('name', 'preset')}",
            show_bbox=show_bbox,
            show_text=show_text,
            max_labels=max_labels,
        )
    figure.tight_layout()
    return figure
