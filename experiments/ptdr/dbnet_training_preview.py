from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
from mmcv.transforms import Compose
from mmengine.registry import init_default_scope
from PIL import Image, ImageOps

try:
    from .config_schema import DBNetPPExperimentConfig, DEFAULT_REPO_ROOT, resolve_repo_relative
    from .train_dbnetpp import (
        configure_mmocr,
        detection_train_mix,
        infer_launcher,
        prepare_detection_manifests,
        resolve_detection_external_roots,
        validate_detection_external_config,
    )
except ImportError:
    from config_schema import DBNetPPExperimentConfig, DEFAULT_REPO_ROOT, resolve_repo_relative
    from train_dbnetpp import (
        configure_mmocr,
        detection_train_mix,
        infer_launcher,
        prepare_detection_manifests,
        resolve_detection_external_roots,
        validate_detection_external_config,
    )


@dataclass(frozen=True)
class PreviewRecord:
    index: int
    record: dict
    source: str


def load_dbnet_settings(config_path: Path) -> DBNetPPExperimentConfig:
    return pyrallis.parse(config_class=DBNetPPExperimentConfig, config_path=str(config_path), args=[])


def infer_source_from_img_path(img_path: str) -> str:
    value = img_path.replace("\\", "/")
    if "dataset/detection/" in value:
        return "ptdr"
    if "dataset/external/icdar2019_mlt/" in value:
        return "icdar2019_mlt"
    if "dataset/external/evarest_detection/" in value:
        return "evarest"
    if "dataset/external/totaltext/" in value:
        return "totaltext"
    if "dataset/external/ctw1500/" in value:
        return "ctw1500"
    if "dataset/external/textocr/" in value:
        return "textocr"
    if "dataset/external/ir_lpr_detection/" in value:
        return "ir_lpr"
    return "unknown"


def load_train_preview_records(manifest_path: Path) -> list[PreviewRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for index, record in enumerate(payload["data_list"]):
        records.append(
            PreviewRecord(
                index=index,
                record=record,
                source=infer_source_from_img_path(str(record["img_path"])),
            )
        )
    return records


def choose_preview_records(
    records: Sequence[PreviewRecord],
    count: int,
    seed: int,
    include_sources: Sequence[str] | None = None,
) -> list[PreviewRecord]:
    items = list(records)
    if include_sources:
        wanted = set(include_sources)
        items = [record for record in items if record.source in wanted]
    rng = random.Random(seed)
    if count >= len(items):
        return items
    return rng.sample(items, count)


def resolve_training_preview_context(config_path: Path) -> dict:
    settings = load_dbnet_settings(config_path)
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    dataset_root = resolve_repo_relative(repo_root, settings.dataset_root)
    manifests_dir = resolve_repo_relative(repo_root, settings.manifests_dir)
    launcher = infer_launcher(settings.training.launcher)
    external_roots = resolve_detection_external_roots(repo_root, settings)
    validate_detection_external_config(settings, external_roots)
    manifests = prepare_detection_manifests(
        repo_root=repo_root,
        dataset_root=dataset_root,
        output_root=manifests_dir,
        include_domains=settings.include_domains,
        val_ratio=settings.val_ratio,
        seed=settings.split_seed,
        launcher=launcher,
        external_roots=external_roots,
        train_mix=detection_train_mix(settings),
        min_ptdr_fraction=settings.train_mix.min_ptdr_fraction,
        mlt_scripts=settings.mlt.scripts,
    )
    cfg = configure_mmocr(settings=settings, manifests=manifests, repo_root=repo_root)
    pipeline_cfg = [
        copy.deepcopy(step)
        for step in cfg.train_dataloader.dataset.pipeline
        if step.get("type") != "PackTextDetInputs"
    ]
    summary = json.loads(Path(manifests["summary"]).read_text(encoding="utf-8"))
    return {
        "settings": settings,
        "repo_root": repo_root,
        "manifests": manifests,
        "summary": summary,
        "pipeline_cfg": pipeline_cfg,
    }


def build_train_preview_pipeline(pipeline_cfg: Sequence[dict]) -> Compose:
    init_default_scope("mmocr")
    return Compose(list(pipeline_cfg))


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def run_preview_pipeline(
    repo_root: Path,
    preview_record: PreviewRecord,
    pipeline: Compose,
    seed: int,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    sample = copy.deepcopy(preview_record.record)
    sample["img_path"] = str(repo_root / sample["img_path"])
    return pipeline(sample)


def _as_points(polygon) -> np.ndarray:
    array = np.asarray(polygon, dtype=np.float32)
    if array.ndim == 2:
        return array
    return array.reshape(-1, 2)


def _bbox_from_points(points: np.ndarray) -> tuple[float, float, float, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    return x0, y0, x1, y1


def draw_detection_annotations(
    ax,
    image_rgb: np.ndarray,
    polygons: Sequence,
    title: str,
    show_bboxes: bool = True,
) -> None:
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    for polygon in polygons:
        points = _as_points(polygon)
        ax.add_patch(
            PolygonPatch(
                points,
                closed=True,
                fill=False,
                edgecolor="#00ff99",
                linewidth=1.6,
            )
        )
        if show_bboxes:
            x0, y0, x1, y1 = _bbox_from_points(points)
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


def preview_dbnet_train_sample(
    repo_root: Path,
    preview_record: PreviewRecord,
    pipeline: Compose,
    seed: int,
    show_bboxes: bool = True,
):
    original_image = load_rgb_image(repo_root / preview_record.record["img_path"])
    original_polygons = [instance["polygon"] for instance in preview_record.record["instances"]]
    transformed = run_preview_pipeline(repo_root, preview_record, pipeline, seed=seed)
    transformed_image = transformed["img"][:, :, ::-1]
    transformed_polygons = transformed["gt_polygons"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_detection_annotations(
        axes[0],
        original_image,
        original_polygons,
        title=f"original | {preview_record.source}",
        show_bboxes=show_bboxes,
    )
    draw_detection_annotations(
        axes[1],
        transformed_image,
        transformed_polygons,
        title="after DBNet train pipeline",
        show_bboxes=show_bboxes,
    )
    figure.suptitle(
        f"{Path(preview_record.record['img_path']).name} | index={preview_record.index}",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def preview_dbnet_train_samples(
    repo_root: Path,
    preview_records: Sequence[PreviewRecord],
    pipeline: Compose,
    seed: int,
    show_bboxes: bool = True,
):
    for row, preview_record in enumerate(preview_records):
        figure = preview_dbnet_train_sample(
            repo_root=repo_root,
            preview_record=preview_record,
            pipeline=pipeline,
            seed=seed + row * 997,
            show_bboxes=show_bboxes,
        )
        yield figure
