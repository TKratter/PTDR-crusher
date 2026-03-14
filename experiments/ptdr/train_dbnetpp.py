#!/usr/bin/env python3

import copy
import json
import os
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import mmcv
import mmengine.fileio as fileio
import numpy as np
import pyrallis
import torch
from mmengine.hooks import Hook
from mmengine.dist import is_main_process
from mmcv.transforms import BaseTransform
from mmocr.registry import HOOKS, TRANSFORMS
from mmocr.utils import poly_intersection, poly_iou, polys2shapely

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_detection_manifest import build_detection_manifests
from build_detection_eval_variants import build_detection_eval_variants
from config_schema import (
    DEFAULT_REPO_ROOT,
    DBNetPPExperimentConfig,
    resolve_repo_relative,
    serialize_config,
)
from detection_augmentations import DEFAULT_PRESETS, apply_preset_to_image_instances
from external_datasets import build_external_detection_train_records, resolve_external_root


def path_for_config(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def find_mmocr_base_config(filename: str) -> Path:
    import mmocr

    package_root = Path(mmocr.__file__).resolve().parent
    config_roots = [
        package_root / ".mim" / "configs" / "textdet",
        package_root / "configs" / "textdet",
        package_root.parent / ".mim" / "configs" / "textdet",
        package_root.parent / "configs" / "textdet",
    ]
    candidates = []
    for config_root in config_roots:
        candidates.append(config_root / filename)
        candidates.extend(config_root.glob(f"*/{filename}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find the MMOCR text detection base config {filename!r}. "
        "Install MMOCR with its bundled configs before running this script."
    )


def build_dataset_cfg(repo_root: Path, ann_file: Path) -> dict:
    return {
        "type": "OCRDataset",
        "data_root": str(repo_root),
        "ann_file": path_for_config(ann_file, repo_root),
        "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
        "pipeline": None,
    }


def attach_dataset_to_dataloader(dataloader_cfg: dict, dataset_cfg: dict) -> None:
    original_dataset = dataloader_cfg["dataset"]
    pipeline = original_dataset.get("pipeline")
    dataloader_cfg["dataset"] = {
        "type": "ConcatDataset",
        "datasets": [dataset_cfg],
        "pipeline": pipeline,
    }


def configure_dataloader_runtime(dataloader_cfg: dict, batch_size: int, training_cfg) -> None:
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = training_cfg.num_workers
    dataloader_cfg.pin_memory = training_cfg.pin_memory
    dataloader_cfg.persistent_workers = training_cfg.num_workers > 0
    if training_cfg.num_workers > 0:
        dataloader_cfg.prefetch_factor = training_cfg.prefetch_factor
    elif "prefetch_factor" in dataloader_cfg:
        dataloader_cfg.pop("prefetch_factor")


def infer_launcher(explicit_launcher: str | None) -> str:
    if explicit_launcher is not None:
        return explicit_launcher
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 or "LOCAL_RANK" in os.environ:
        return "pytorch"
    return "none"


def manifest_paths(output_root: Path) -> dict[str, Path]:
    return {
        "train_ann": output_root / "textdet_train.json",
        "val_ann": output_root / "textdet_val.json",
        "test_ann": output_root / "textdet_test.json",
        "summary": output_root / "summary.json",
    }


def prepare_detection_manifests(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    include_domains,
    val_ratio: float,
    seed: int,
    launcher: str,
    external_roots: dict[str, Path | None] | None = None,
    train_mix: dict[str, float] | None = None,
    min_ptdr_fraction: float = 0.3,
    mlt_scripts=None,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = manifest_paths(output_root)
    ready_file = output_root / ".manifests.ready"

    if all(path.exists() for path in manifests.values()):
        if launcher == "pytorch":
            ready_file.write_text("ready\n", encoding="utf-8")
        return manifests

    def build_all_manifests() -> dict[str, Path]:
        external_train_records, external_errors = build_external_detection_train_records(
            repo_root=repo_root,
            external_roots=external_roots or {},
            mlt_scripts=mlt_scripts or ["Arabic", "Latin"],
        )
        return build_detection_manifests(
            repo_root=repo_root,
            dataset_root=dataset_root,
            output_root=output_root,
            include_domains=include_domains,
            val_ratio=val_ratio,
            seed=seed,
            external_train_records=external_train_records,
            train_mix=train_mix,
            min_ptdr_fraction=min_ptdr_fraction,
            extra_errors=external_errors,
        )

    if launcher != "pytorch":
        return build_all_manifests()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        ready_file.unlink(missing_ok=True)
        built_manifests = build_all_manifests()
        ready_file.write_text("ready\n", encoding="utf-8")
        return built_manifests

    deadline = time.time() + 7200
    while time.time() < deadline:
        if ready_file.exists() and all(path.exists() for path in manifests.values()):
            return manifests
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for rank 0 to prepare manifests under {output_root}.")


def build_vis_backends(settings: DBNetPPExperimentConfig) -> list[dict]:
    vis_backends = [{"type": "LocalVisBackend"}]
    wandb_cfg = settings.wandb
    if wandb_cfg.enabled:
        init_kwargs = build_wandb_init_kwargs(settings)
        vis_backends.append(
            {
                "type": "WandbVisBackend",
                "init_kwargs": {key: value for key, value in init_kwargs.items() if value is not None},
            }
        )
    return vis_backends


def prepare_wandb_env(work_dir: Path) -> None:
    wandb_root = work_dir / "wandb"
    cache_dir = wandb_root / "cache"
    config_dir = wandb_root / "config"
    data_dir = wandb_root / "data"
    artifact_dir = wandb_root / "artifacts"
    tmp_dir = wandb_root / "tmp"
    for directory in (wandb_root, cache_dir, config_dir, data_dir, artifact_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(data_dir))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(artifact_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    tempfile.tempdir = os.environ["TMPDIR"]


def build_wandb_init_kwargs(settings: DBNetPPExperimentConfig) -> dict:
    wandb_cfg = settings.wandb
    run_id = os.environ.setdefault("WANDB_RUN_ID", uuid.uuid4().hex[:8])
    init_kwargs = {
        "project": wandb_cfg.project,
        "entity": wandb_cfg.entity,
        "group": wandb_cfg.group,
        "job_type": wandb_cfg.job_type,
        "name": wandb_cfg.run_name or settings.experiment_name,
        "tags": wandb_cfg.tags,
        "config": serialize_config(settings),
        "id": run_id,
        "resume": "allow",
        "dir": os.environ.get("WANDB_DIR"),
    }
    return {key: value for key, value in init_kwargs.items() if value is not None}


def maybe_initialize_wandb_early(settings: DBNetPPExperimentConfig) -> None:
    if not settings.wandb.enabled:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return

    import wandb

    init_kwargs = build_wandb_init_kwargs(settings)
    run = wandb.init(**init_kwargs)
    print(
        json.dumps(
            {
                "wandb_early_init": True,
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_run_url": getattr(run, "url", None),
            },
            ensure_ascii=False,
        )
    )


@TRANSFORMS.register_module()
class PTDRLimitImageSize(BaseTransform):
    def __init__(self, max_side: int | None = 4096, max_pixels: int | None = 16000000) -> None:
        self.max_side = int(max_side) if max_side else None
        self.max_pixels = int(max_pixels) if max_pixels else None

    def _target_scale(self, height: int, width: int) -> float:
        scale = 1.0
        if self.max_side and max(height, width) > self.max_side:
            scale = min(scale, self.max_side / float(max(height, width)))
        if self.max_pixels and (height * width) > self.max_pixels:
            scale = min(scale, float(np.sqrt(self.max_pixels / float(height * width))))
        return scale

    def _scale_instances(self, instances: list[dict], scale: float) -> list[dict]:
        scaled_instances: list[dict] = []
        for instance in instances:
            updated = copy.deepcopy(instance)
            polygon = np.asarray(instance["polygon"], dtype=np.float32).reshape(-1, 2) * scale
            updated["polygon"] = polygon.reshape(-1).tolist()
            xs = polygon[:, 0]
            ys = polygon[:, 1]
            updated["bbox"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            scaled_instances.append(updated)
        return scaled_instances

    def transform(self, results: dict) -> dict:
        image = results.get("img")
        if image is None:
            return results

        height, width = image.shape[:2]
        scale = self._target_scale(height, width)
        if scale >= 0.999:
            return results

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        results["img"] = resized
        results["img_shape"] = (new_height, new_width)
        results["ori_shape"] = (new_height, new_width)
        results["height"] = new_height
        results["width"] = new_width
        results["scale_factor"] = (scale, scale)

        if "gt_polygons" in results:
            results["gt_polygons"] = [
                np.asarray(polygon, dtype=np.float32).reshape(-1, 2) * scale for polygon in results["gt_polygons"]
            ]
        if "gt_bboxes" in results:
            results["gt_bboxes"] = np.asarray(results["gt_bboxes"], dtype=np.float32) * scale
        if "instances" in results:
            results["instances"] = self._scale_instances(list(results["instances"]), scale)
        return results


@TRANSFORMS.register_module()
class PTDRTextDetHardAug(BaseTransform):
    def __init__(self, probability: float = 0.75, seed: int = 42, presets: list[dict] | None = None) -> None:
        self.probability = float(probability)
        self.seed = int(seed)
        self.presets = copy.deepcopy(list(presets or DEFAULT_PRESETS))

    def set_probability(self, probability: float) -> None:
        self.probability = float(probability)

    def get_probability(self) -> float:
        return float(self.probability)

    def _instances_from_results(self, results: dict) -> list[dict]:
        polygons = results.get("gt_polygons", [])
        ignored = np.asarray(results.get("gt_ignored", np.zeros(len(polygons), dtype=np.bool_)), dtype=np.bool_)
        labels = np.asarray(results.get("gt_bboxes_labels", np.zeros(len(polygons), dtype=np.int64)), dtype=np.int64)
        texts = list(results.get("gt_texts", [""] * len(polygons)))
        instances = []
        for index, polygon in enumerate(polygons):
            poly_list = polygon.tolist() if hasattr(polygon, "tolist") else list(polygon)
            instances.append(
                {
                    "polygon": poly_list,
                    "text": texts[index] if index < len(texts) else "",
                    "bbox_label": int(labels[index]) if index < len(labels) else 0,
                    "ignore": bool(ignored[index]) if index < len(ignored) else False,
                }
            )
        return instances

    def transform(self, results: dict) -> dict:
        if self.probability <= 0 or not self.presets or np.random.rand() > self.probability:
            return results
        image_bgr = results.get("img")
        polygons = results.get("gt_polygons")
        if image_bgr is None or polygons is None or len(polygons) == 0:
            return results

        preset = copy.deepcopy(self.presets[int(np.random.randint(len(self.presets)))])
        aug_seed = int(np.random.randint(0, 2**31 - 1))
        image_rgb = image_bgr[:, :, ::-1]
        instances = self._instances_from_results(results)
        augmented_rgb, augmented_instances = apply_preset_to_image_instances(
            image=image_rgb,
            instances=instances,
            preset=preset,
            seed=self.seed + aug_seed,
        )
        augmented_bgr = augmented_rgb[:, :, ::-1].copy()

        height, width = augmented_bgr.shape[:2]
        results["img"] = augmented_bgr
        results["img_shape"] = (height, width)
        results["ori_shape"] = (height, width)
        results["height"] = height
        results["width"] = width
        results["gt_polygons"] = [np.asarray(instance["polygon"], dtype=np.float32) for instance in augmented_instances]
        results["gt_bboxes"] = np.asarray([instance["bbox"] for instance in augmented_instances], dtype=np.float32)
        results["gt_bboxes_labels"] = np.asarray(
            [int(instance.get("bbox_label", 0)) for instance in augmented_instances],
            dtype=np.int64,
        )
        results["gt_ignored"] = np.asarray(
            [bool(instance.get("ignore", False)) for instance in augmented_instances],
            dtype=np.bool_,
        )
        if "gt_texts" in results:
            results["gt_texts"] = [str(instance.get("text", "")) for instance in augmented_instances]
        if "instances" in results:
            results["instances"] = []
            for instance in augmented_instances:
                results["instances"].append(
                    {
                        "polygon": list(instance["polygon"]),
                        "bbox": list(instance["bbox"]),
                        "bbox_label": int(instance.get("bbox_label", 0)),
                        "ignore": bool(instance.get("ignore", False)),
                        "text": str(instance.get("text", "")),
                    }
                )
        return results


def _iter_dataset_pipelines(dataset) -> list:
    pipelines = []
    if dataset is None:
        return pipelines
    pipeline = getattr(dataset, "pipeline", None)
    if pipeline is not None:
        transforms = getattr(pipeline, "transforms", None)
        if transforms is not None:
            pipelines.append(transforms)
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        for child in nested:
            pipelines.extend(_iter_dataset_pipelines(child))
    child = getattr(dataset, "dataset", None)
    if child is not None:
        pipelines.extend(_iter_dataset_pipelines(child))
    return pipelines


def _find_hard_aug_transform(dataset) -> PTDRTextDetHardAug | None:
    for pipeline in _iter_dataset_pipelines(dataset):
        for transform in pipeline:
            if isinstance(transform, PTDRTextDetHardAug):
                return transform
    return None


@HOOKS.register_module()
class PTDRDBNetHardAugScheduleHook(Hook):
    def __init__(self, epochs: list[int], probabilities: list[float]) -> None:
        self.epochs = [int(epoch) for epoch in epochs]
        self.probabilities = [float(probability) for probability in probabilities]
        self._last_probability: float | None = None

    def _probability_for_epoch(self, epoch: int) -> float:
        active_probability = self.probabilities[0]
        for start_epoch, probability in zip(self.epochs, self.probabilities):
            if epoch >= start_epoch:
                active_probability = probability
            else:
                break
        return float(active_probability)

    def _apply_probability(self, runner, epoch: int) -> None:
        train_loop = getattr(runner, "train_loop", None)
        dataloader = getattr(train_loop, "dataloader", None)
        dataset = getattr(dataloader, "dataset", None)
        transform = _find_hard_aug_transform(dataset)
        if transform is None:
            return

        probability = self._probability_for_epoch(epoch)
        transform.set_probability(probability)
        if self._last_probability == probability:
            return
        self._last_probability = probability

        if is_main_process():
            step = epoch + 1
            runner.visualizer.add_scalars({"train/hard_aug_probability": probability}, step=step)
            print(
                json.dumps(
                    {
                        "epoch": step,
                        "hard_aug_probability": probability,
                        "hard_aug_schedule_epochs": self.epochs,
                        "hard_aug_schedule_probabilities": self.probabilities,
                    },
                    ensure_ascii=False,
                )
            )

    def before_train(self, runner) -> None:
        self._apply_probability(runner, getattr(runner, "epoch", 0))

    def before_train_epoch(self, runner) -> None:
        self._apply_probability(runner, getattr(runner, "epoch", 0))


@HOOKS.register_module()
class PTDRDBNetValSampleLoggerHook(Hook):
    def __init__(
        self,
        seed: int = 42,
        score_thr: float = 0.3,
        match_iou_thr: float = 0.5,
        ignore_precision_thr: float = 0.5,
    ) -> None:
        self.seed = seed
        self.score_thr = score_thr
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self._rng = random.Random(seed)
        self._candidate_counts = {"true_positive": 0, "false": 0}
        self._candidates = {"true_positive": None, "false": None}

    def before_val_epoch(self, runner) -> None:
        self._rng = random.Random(self.seed + runner.epoch)
        self._candidate_counts = {"true_positive": 0, "false": 0}
        self._candidates = {"true_positive": None, "false": None}

    def after_val_iter(self, runner, batch_idx: int, data_batch, outputs) -> None:
        if not is_main_process():
            return
        for output in outputs:
            cpu_sample = output.cpu()
            summary = self.summarize_sample(cpu_sample)
            if summary["is_true_positive"]:
                self.update_candidate("true_positive", cpu_sample, summary)
            elif summary["is_false"]:
                self.update_candidate("false", cpu_sample, summary)

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not is_main_process():
            return
        for sample_kind in ("true_positive", "false"):
            candidate = self._candidates[sample_kind]
            if candidate is not None:
                self.log_candidate(runner, sample_kind, candidate, metrics or {})

    def update_candidate(self, sample_kind: str, sample, summary: dict) -> None:
        self._candidate_counts[sample_kind] += 1
        if self._rng.randrange(self._candidate_counts[sample_kind]) == 0:
            self._candidates[sample_kind] = {"sample": sample, "summary": summary}

    def summarize_sample(self, data_sample) -> dict:
        gt_instances = data_sample.get("gt_instances", None)
        pred_instances = data_sample.get("pred_instances", None)
        if gt_instances is None or pred_instances is None:
            return {"is_true_positive": False, "is_false": False}

        gt_polygons = gt_instances.get("polygons", [])
        pred_polygons = pred_instances.get("polygons", [])
        pred_scores = pred_instances.get("scores", [])
        gt_ignore_flags = gt_instances.get("ignored", None)

        if hasattr(pred_scores, "cpu"):
            pred_scores = pred_scores.cpu().numpy()
        pred_scores = np.asarray(pred_scores, dtype=np.float32)

        if gt_ignore_flags is None:
            gt_ignore_flags = np.zeros(len(gt_polygons), dtype=bool)
        elif hasattr(gt_ignore_flags, "cpu"):
            gt_ignore_flags = gt_ignore_flags.cpu().numpy()
        gt_ignore_flags = np.asarray(gt_ignore_flags, dtype=bool)

        gt_polys = polys2shapely(gt_polygons)
        pred_polys = polys2shapely(pred_polygons)
        pred_ignore_flags = pred_scores < self.score_thr

        for pred_idx in np.where(~pred_ignore_flags)[0]:
            for gt_idx in np.where(gt_ignore_flags)[0]:
                precision = poly_intersection(gt_polys[gt_idx], pred_polys[pred_idx]) / (
                    pred_polys[pred_idx].area + 1e-5
                )
                if precision > self.ignore_precision_thr:
                    pred_ignore_flags[pred_idx] = True
                    break

        valid_gt_indices = np.where(~gt_ignore_flags)[0]
        valid_pred_indices = np.where(~pred_ignore_flags)[0]

        matched_gt_indexes = set()
        matched_pred_indexes = set()
        for gt_offset, gt_idx in enumerate(valid_gt_indices):
            for pred_offset, pred_idx in enumerate(valid_pred_indices):
                if poly_iou(gt_polys[gt_idx], pred_polys[pred_idx]) > self.match_iou_thr:
                    if gt_offset in matched_gt_indexes or pred_offset in matched_pred_indexes:
                        continue
                    matched_gt_indexes.add(gt_offset)
                    matched_pred_indexes.add(pred_offset)

        total_gt = len(valid_gt_indices)
        total_pred = len(valid_pred_indices)
        matched_gt = len(matched_gt_indexes)
        matched_pred = len(matched_pred_indexes)
        is_true_positive = total_gt > 0 and matched_gt == total_gt and matched_pred == total_pred
        is_false = (total_gt > 0 or total_pred > 0) and not is_true_positive
        return {
            "is_true_positive": is_true_positive,
            "is_false": is_false,
            "gt_count": total_gt,
            "pred_count": total_pred,
            "matched_gt": matched_gt,
            "matched_pred": matched_pred,
        }

    def log_candidate(self, runner, sample_kind: str, candidate: dict, metrics: dict) -> None:
        data_sample = candidate["sample"]
        summary = dict(candidate["summary"])
        img_path = data_sample.img_path
        image = mmcv.imfrombytes(fileio.get(img_path), channel_order="rgb")

        output_dir = Path(runner.work_dir) / "val_samples" / "dbnetpp"
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch = runner.epoch + 1
        stem = f"epoch_{epoch:03d}_{sample_kind}_{Path(img_path).stem}"
        out_file = output_dir / f"{stem}.png"
        meta_file = output_dir / f"{stem}.json"

        runner.visualizer.add_datasample(
            name=stem,
            image=image,
            data_sample=data_sample,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=self.score_thr,
            step=epoch,
            out_file=str(out_file),
        )

        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                serializable_metrics[key] = float(value)
            elif hasattr(value, "item"):
                serializable_metrics[key] = float(value.item())

        summary.update(
            {
                "epoch": epoch,
                "sample_kind": sample_kind,
                "img_path": img_path,
                "logged_image": str(out_file),
                "val_metrics": serializable_metrics,
            }
        )
        meta_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


@HOOKS.register_module()
class PTDRDBNetExtraValEvalHook(Hook):
    def __init__(
        self,
        repo_root: str,
        eval_sets: dict[str, str],
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.eval_sets = dict(eval_sets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self._dataloaders: dict[str, object] = {}

    def before_train(self, runner) -> None:
        self._dataloaders = {}
        for eval_name, ann_file in self.eval_sets.items():
            dataloader_cfg = copy.deepcopy(runner.cfg.val_dataloader)
            dataset_cfg = build_dataset_cfg(self.repo_root, Path(ann_file))
            attach_dataset_to_dataloader(dataloader_cfg, dataset_cfg)
            dataloader_cfg.batch_size = self.batch_size
            dataloader_cfg.num_workers = self.num_workers
            dataloader_cfg.pin_memory = self.pin_memory
            dataloader_cfg.persistent_workers = self.num_workers > 0
            if self.num_workers > 0:
                dataloader_cfg.prefetch_factor = self.prefetch_factor
            elif "prefetch_factor" in dataloader_cfg:
                dataloader_cfg.pop("prefetch_factor")
            if isinstance(dataloader_cfg.get("sampler"), dict):
                dataloader_cfg["sampler"]["shuffle"] = False
            self._dataloaders[eval_name] = runner.build_dataloader(dataloader_cfg, seed=getattr(runner, "seed", None))

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._dataloaders:
            return

        model = runner.model
        model.eval()
        extra_metrics: dict[str, float] = {}
        for eval_name, dataloader in self._dataloaders.items():
            evaluator = runner.build_evaluator({"type": "HmeanIOUMetric", "prefix": f"{eval_name}/icdar"})
            for data_batch in dataloader:
                with torch.no_grad():
                    outputs = model.val_step(data_batch)
                evaluator.process(data_samples=outputs, data_batch=data_batch)
            metrics_for_eval = evaluator.evaluate(len(dataloader.dataset))
            extra_metrics.update(metrics_for_eval)

        if is_main_process() and extra_metrics:
            output_dir = Path(runner.work_dir) / "extra_val_metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            epoch = runner.epoch + 1
            output_path = output_dir / f"epoch_{epoch:03d}.json"
            output_path.write_text(json.dumps(extra_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            runner.visualizer.add_scalars(extra_metrics, step=epoch, file_path=str(output_dir / "metrics.json"))
            print(json.dumps({"epoch": epoch, "extra_val_metrics": extra_metrics}, ensure_ascii=False))


def maybe_insert_hard_aug_train_transform(cfg, settings: DBNetPPExperimentConfig) -> None:
    if not settings.hard_aug.train.enabled:
        return
    pipeline = list(cfg.train_dataloader.dataset.pipeline)
    insert_at = len(pipeline)
    for index, step in enumerate(pipeline):
        if step.get("type") == "RandomCrop":
            insert_at = index
            break
    pipeline.insert(
        insert_at,
        {
            "type": "PTDRTextDetHardAug",
            "probability": settings.hard_aug.train.probability,
            "seed": settings.training.seed,
        },
    )
    cfg.train_dataloader.dataset.pipeline = pipeline
    if cfg.get("train_pipeline") is not None:
        cfg.train_pipeline = pipeline


def hard_aug_schedule_payload(settings: DBNetPPExperimentConfig) -> dict[str, list]:
    train_cfg = settings.hard_aug.train
    if not train_cfg.schedule_epochs:
        return {}
    return {
        "epochs": [int(epoch) for epoch in train_cfg.schedule_epochs],
        "probabilities": [float(probability) for probability in train_cfg.schedule_probabilities],
    }


def validate_hard_aug_schedule(settings: DBNetPPExperimentConfig) -> None:
    train_cfg = settings.hard_aug.train
    epochs = list(train_cfg.schedule_epochs)
    probabilities = list(train_cfg.schedule_probabilities)
    if not epochs and not probabilities:
        return
    if len(epochs) != len(probabilities):
        raise ValueError("hard_aug.train.schedule_epochs and schedule_probabilities must have the same length.")
    if not epochs:
        raise ValueError("hard_aug.train schedule must include at least one epoch when probabilities are provided.")
    if epochs[0] != 0:
        raise ValueError("hard_aug.train schedule must start at epoch 0.")
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("hard_aug.train schedule probabilities must be between 0 and 1.")
    if any(current >= next_value for current, next_value in zip(epochs, epochs[1:])):
        raise ValueError("hard_aug.train schedule_epochs must be strictly increasing.")


def maybe_insert_limit_image_transform(cfg, settings: DBNetPPExperimentConfig) -> None:
    max_side = settings.training.train_max_image_side
    max_pixels = settings.training.train_max_pixels
    if not max_side and not max_pixels:
        return

    pipeline = list(cfg.train_dataloader.dataset.pipeline)
    insert_at = 0
    for index, step in enumerate(pipeline):
        if step.get("type") == "LoadOCRAnnotations":
            insert_at = index + 1
            break
    pipeline.insert(
        insert_at,
        {
            "type": "PTDRLimitImageSize",
            "max_side": max_side,
            "max_pixels": max_pixels,
        },
    )
    cfg.train_dataloader.dataset.pipeline = pipeline
    if cfg.get("train_pipeline") is not None:
        cfg.train_pipeline = pipeline


def prepare_detection_eval_manifests(
    repo_root: Path,
    manifests: dict[str, Path],
    output_root: Path,
    settings: DBNetPPExperimentConfig,
    launcher: str,
) -> dict[str, Path]:
    if not settings.hard_aug.eval.enabled:
        return {}
    output_root.mkdir(parents=True, exist_ok=True)
    rotation_angles = tuple(settings.hard_aug.eval.rotation_angles)
    expected = {
        **{f"val_rot{int(angle)}": output_root / f"textdet_val_rot{int(angle)}.json" for angle in rotation_angles},
        "val_hard": output_root / "textdet_val_hard.json",
    }
    ready_file = output_root / ".eval_variants.ready"

    if all(path.exists() for path in expected.values()):
        if launcher == "pytorch":
            ready_file.write_text("ready\n", encoding="utf-8")
        return expected

    def build_variants() -> dict[str, Path]:
        return build_detection_eval_variants(
            repo_root=repo_root,
            val_ann_path=Path(manifests["val_ann"]),
            test_ann_path=Path(manifests["test_ann"]),
            output_root=output_root,
            seed=settings.hard_aug.eval.seed,
            rotation_angles=rotation_angles,
            include_test_hard=False,
        )

    if launcher != "pytorch":
        return build_variants()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        ready_file.unlink(missing_ok=True)
        outputs = build_variants()
        ready_file.write_text("ready\n", encoding="utf-8")
        return outputs

    deadline = time.time() + 7200
    while time.time() < deadline:
        if ready_file.exists() and all(path.exists() for path in expected.values()):
            return expected
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for rank 0 to prepare detection eval variants under {output_root}.")


def shared_detection_eval_root(repo_root: Path) -> Path:
    return repo_root / "experiments" / "ptdr" / "manifests" / "shared_textdet_eval_variants"


def configure_mmocr(settings: DBNetPPExperimentConfig, manifests: dict, repo_root: Path, extra_eval_manifests: dict[str, Path] | None = None):
    from mmengine.config import Config

    training_cfg = settings.training
    mmocr_cfg = settings.mmocr
    base_cfg = Config.fromfile(str(find_mmocr_base_config(mmocr_cfg.base_config)))

    train_dataset = build_dataset_cfg(repo_root, manifests["train_ann"])
    val_dataset = build_dataset_cfg(repo_root, manifests["val_ann"])
    test_dataset = build_dataset_cfg(repo_root, manifests["test_ann"])

    base_cfg.train_list = [train_dataset]
    base_cfg.test_list = [val_dataset]
    attach_dataset_to_dataloader(base_cfg.train_dataloader, train_dataset)
    attach_dataset_to_dataloader(base_cfg.val_dataloader, val_dataset)
    attach_dataset_to_dataloader(base_cfg.test_dataloader, test_dataset)

    configure_dataloader_runtime(base_cfg.train_dataloader, training_cfg.batch_size, training_cfg)
    configure_dataloader_runtime(base_cfg.val_dataloader, training_cfg.eval_batch_size, training_cfg)
    configure_dataloader_runtime(base_cfg.test_dataloader, training_cfg.eval_batch_size, training_cfg)
    maybe_insert_limit_image_transform(base_cfg, settings)
    maybe_insert_hard_aug_train_transform(base_cfg, settings)

    base_cfg.work_dir = str(resolve_repo_relative(repo_root, training_cfg.work_dir))
    base_cfg.launcher = infer_launcher(training_cfg.launcher)
    base_cfg.resume = training_cfg.resume
    base_cfg.randomness = {"seed": training_cfg.seed}
    base_cfg.train_cfg.max_epochs = training_cfg.max_epochs
    base_cfg.train_cfg.val_interval = training_cfg.val_interval
    base_cfg.default_hooks.logger.interval = training_cfg.logger_interval
    base_cfg.env_cfg["cudnn_benchmark"] = training_cfg.cudnn_benchmark
    base_cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "interval": training_cfg.checkpoint_interval,
        "by_epoch": True,
        "save_last": True,
        "save_best": "icdar/hmean",
        "rule": "greater",
        "max_keep_ckpts": training_cfg.max_keep_ckpts,
    }

    if base_cfg.get("auto_scale_lr") is not None:
        base_cfg.auto_scale_lr["enable"] = training_cfg.enable_auto_scale_lr
    if training_cfg.optimizer_lr is not None:
        base_cfg.optim_wrapper["optimizer"]["lr"] = float(training_cfg.optimizer_lr)

    if mmocr_cfg.load_from:
        base_cfg.load_from = mmocr_cfg.load_from

    if training_cfg.amp:
        base_cfg.optim_wrapper["type"] = "AmpOptimWrapper"
        base_cfg.optim_wrapper["loss_scale"] = "dynamic"

    if base_cfg.get("visualizer") is None:
        base_cfg.visualizer = {"type": "TextDetLocalVisualizer", "name": "visualizer"}
    base_cfg.visualizer["vis_backends"] = build_vis_backends(settings)
    custom_hooks = list(base_cfg.get("custom_hooks", []))
    hard_aug_schedule = hard_aug_schedule_payload(settings)
    if hard_aug_schedule:
        custom_hooks.append(
            {
                "type": "PTDRDBNetHardAugScheduleHook",
                "epochs": hard_aug_schedule["epochs"],
                "probabilities": hard_aug_schedule["probabilities"],
            }
        )
    custom_hooks.append(
        {
            "type": "PTDRDBNetValSampleLoggerHook",
            "seed": training_cfg.seed,
        }
    )
    if extra_eval_manifests:
        extra_eval_sets = {
            key: str(value)
            for key, value in extra_eval_manifests.items()
            if key in {"val_rot90", "val_rot180", "val_rot270", "val_hard"}
        }
        custom_hooks.append(
            {
                "type": "PTDRDBNetExtraValEvalHook",
                "repo_root": str(repo_root),
                "eval_sets": extra_eval_sets,
                "batch_size": training_cfg.eval_batch_size,
                "num_workers": training_cfg.num_workers,
                "pin_memory": training_cfg.pin_memory,
                "prefetch_factor": training_cfg.prefetch_factor,
            }
        )
    base_cfg.custom_hooks = custom_hooks

    return base_cfg


def preflight_check(cfg, manifests: dict) -> None:
    required_ann_files = {
        "train": manifests["train_ann"],
        "val": manifests["val_ann"],
        "test": manifests["test_ann"],
    }
    for split_name, ann_path in required_ann_files.items():
        if not Path(ann_path).exists():
            raise FileNotFoundError(f"Missing generated {split_name} annotation file: {ann_path}")

    for dataloader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dataloader = getattr(cfg, dataloader_name, None)
        if dataloader is None:
            raise RuntimeError(f"MMOCR config is missing {dataloader_name}.")
        dataset = dataloader.get("dataset")
        if dataset is None or "datasets" not in dataset:
            raise RuntimeError(f"{dataloader_name} does not expose the expected ConcatDataset structure.")


def resolve_detection_external_roots(repo_root: Path, settings: DBNetPPExperimentConfig) -> dict[str, Path | None]:
    roots_cfg = settings.external_datasets
    return {
        "icdar2019_mlt": resolve_external_root(repo_root, roots_cfg.icdar2019_mlt),
        "evarest_detection": resolve_external_root(repo_root, roots_cfg.evarest_detection),
        "totaltext": resolve_external_root(repo_root, roots_cfg.totaltext),
        "ctw1500": resolve_external_root(repo_root, roots_cfg.ctw1500),
        "textocr": resolve_external_root(repo_root, roots_cfg.textocr),
        "ir_lpr_detection": resolve_external_root(repo_root, roots_cfg.ir_lpr_detection),
    }


def detection_train_mix(settings: DBNetPPExperimentConfig) -> dict[str, float]:
    mix = settings.train_mix
    return {
        "ptdr": mix.ptdr,
        "icdar2019_mlt": mix.icdar2019_mlt,
        "evarest": mix.evarest,
        "totaltext": mix.totaltext,
        "ctw1500": mix.ctw1500,
        "textocr": mix.textocr,
        "ir_lpr": mix.ir_lpr,
    }


def validate_detection_external_config(settings: DBNetPPExperimentConfig, external_roots: dict[str, Path | None]) -> None:
    mix = detection_train_mix(settings)
    required_roots = {
        "icdar2019_mlt": "icdar2019_mlt",
        "evarest": "evarest_detection",
        "totaltext": "totaltext",
        "ctw1500": "ctw1500",
        "textocr": "textocr",
        "ir_lpr": "ir_lpr_detection",
    }
    for dataset_name, weight in mix.items():
        if dataset_name == "ptdr" or weight <= 0:
            continue
        root_key = required_roots[dataset_name]
        root_path = external_roots.get(root_key)
        if root_path is None:
            raise FileNotFoundError(f"Train mix enables {dataset_name} but no dataset root was configured for {root_key}.")
        if not root_path.exists():
            raise FileNotFoundError(f"Configured dataset root for {dataset_name} does not exist: {root_path}")


@pyrallis.wrap(config_path=str(SCRIPT_DIR / "configs" / "dbnetpp.yaml"))
def main(settings: DBNetPPExperimentConfig) -> None:
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    dataset_root = resolve_repo_relative(repo_root, settings.dataset_root)
    manifests_dir = resolve_repo_relative(repo_root, settings.manifests_dir)
    shared_eval_root = shared_detection_eval_root(repo_root)
    work_dir = resolve_repo_relative(repo_root, settings.training.work_dir)
    launcher = infer_launcher(settings.training.launcher)
    external_roots = resolve_detection_external_roots(repo_root, settings)
    validate_hard_aug_schedule(settings)
    validate_detection_external_config(settings, external_roots)

    work_dir.mkdir(parents=True, exist_ok=True)
    if settings.wandb.enabled:
        os.environ.setdefault("WANDB_MODE", settings.wandb.mode)
        prepare_wandb_env(work_dir)
        maybe_initialize_wandb_early(settings)

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
    extra_eval_manifests = prepare_detection_eval_manifests(
        repo_root=repo_root,
        manifests=manifests,
        output_root=shared_eval_root,
        settings=settings,
        launcher=launcher,
    )
    summary = json.loads(Path(manifests["summary"]).read_text(encoding="utf-8"))
    error_count = len(summary["errors"])
    if settings.fail_on_bad_annotations and error_count:
        raise SystemExit(f"Found {error_count} malformed detection annotations. Inspect {manifests['summary']}.")

    cfg = configure_mmocr(
        settings=settings,
        manifests=manifests,
        repo_root=repo_root,
        extra_eval_manifests=extra_eval_manifests,
    )
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    preflight_check(cfg, manifests)

    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    init_default_scope("mmocr")
    print(
        json.dumps(
            {
                "experiment_name": settings.experiment_name,
                "train_ann": str(manifests["train_ann"]),
                "val_ann": str(manifests["val_ann"]),
                "test_ann": str(manifests["test_ann"]),
                "work_dir": cfg.work_dir,
                "launcher": cfg.launcher,
                "resume": cfg.get("resume", False),
                "batch_size_per_device": cfg.train_dataloader.batch_size,
                "num_workers_per_rank": cfg.train_dataloader.num_workers,
                "pin_memory": cfg.train_dataloader.get("pin_memory", False),
                "prefetch_factor": cfg.train_dataloader.get("prefetch_factor", None),
                "optimizer_lr": cfg.optim_wrapper["optimizer"]["lr"],
                "train_max_image_side": settings.training.train_max_image_side,
                "train_max_pixels": settings.training.train_max_pixels,
                "hard_train_aug_enabled": settings.hard_aug.train.enabled,
                "hard_train_aug_probability": settings.hard_aug.train.probability,
                "hard_train_aug_schedule_epochs": settings.hard_aug.train.schedule_epochs,
                "hard_train_aug_schedule_probabilities": settings.hard_aug.train.schedule_probabilities,
                "hard_eval_enabled": settings.hard_aug.eval.enabled,
                "hard_eval_rotation_angles": settings.hard_aug.eval.rotation_angles,
                "shared_eval_root": str(shared_eval_root),
                "extra_eval_manifests": {key: str(value) for key, value in extra_eval_manifests.items()},
                "train_mix": detection_train_mix(settings),
                "min_ptdr_fraction": settings.train_mix.min_ptdr_fraction,
                "mlt_scripts": settings.mlt.scripts,
                "skipped_annotations": error_count,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
