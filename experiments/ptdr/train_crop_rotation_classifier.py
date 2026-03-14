#!/usr/bin/env python3

import json
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_recognition_eval_variants import build_recognition_eval_variants
from build_recognition_manifest import (
    build_recognition_samples,
    normalize_and_filter_samples,
    resolve_charset_policy,
    split_samples,
)
from config_schema import DEFAULT_REPO_ROOT, PARSeqCharsetPolicyConfig, WandbConfig, default_scene_domains, resolve_repo_relative, serialize_config
from detection_augmentations import DEFAULT_PRESETS, choose_preset_for_key
from rotation_solution_utils import (
    LmdbRecognitionDataset,
    RIGHT_ANGLE_ROTATIONS,
    angle_to_rotation_class,
    apply_reflect_preset,
    build_wandb_init_kwargs,
    correction_for_applied_rotation,
    jitter_quad_detector_style,
    load_rgb_image,
    perspective_crop_from_quad,
    prepare_wandb_env,
    rotate_image_reflect,
)
from validate_dataset import iter_detection_annotation_files, parse_detection_line


@dataclass
class CropRotationModelConfig:
    backbone: str = "mobilenet_v2"
    pretrained: bool = True
    image_size: List[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1


@dataclass
class CropRotationTrainingConfig:
    work_dir: Path = Path("work_dirs/crop_rotation_classifier")
    seed: int = 42
    batch_size: int = 256
    num_workers: int = 8
    max_epochs: int = 12
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    lr: float = 0.0005
    weight_decay: float = 0.0001
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 25
    resume_from_checkpoint: Optional[Path] = None


@dataclass
class CropRotationDataConfig:
    train_lmdb_root: Path = Path("experiments/ptdr/manifests/parseq_multidata_all_norm_v2/train/real/ptdr")
    detection_dataset_root: Path = Path("dataset/detection")
    detector_jitter_bank_root: Optional[Path] = None
    detector_jitter_bank_size: Optional[int] = None
    detector_jitter_bank_seed: int = 42
    shared_eval_root: Path = Path("experiments/ptdr/manifests/shared_recognition_eval_variants")
    shared_eval_label_root: Path = Path("experiments/ptdr/manifests/shared_crop_rotation_eval")
    dataset_root: Path = Path("dataset/recognition")
    include_domains: List[str] = field(default_factory=default_scene_domains)
    val_ratio: float = 0.1
    split_seed: int = 42
    hard_seed: int = 42
    hard_probability: float = 0.65
    small_rotation_probability: float = 0.35
    small_rotation_max_degrees: float = 10.0
    detector_jitter_probability: float = 0.35
    detector_jitter_attempts: int = 8
    detection_image_cache_size: int = 256
    exhaustive_right_angle_classes: bool = False
    charset_policy: PARSeqCharsetPolicyConfig = field(default_factory=PARSeqCharsetPolicyConfig)


@dataclass
class CropRotationExperimentConfig:
    experiment_name: str = "ptdr-crop-rotation-classifier"
    repo_root: Path = Path(".")
    model: CropRotationModelConfig = field(default_factory=CropRotationModelConfig)
    training: CropRotationTrainingConfig = field(default_factory=CropRotationTrainingConfig)
    data: CropRotationDataConfig = field(default_factory=CropRotationDataConfig)
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            group="crop-rotation",
            tags=["ptdr", "rotation", "recognition-crop", "hard-train"],
        )
    )


def build_logger(settings: CropRotationExperimentConfig, work_dir: Path):
    csv_logger = CSVLogger(save_dir=str(work_dir / "logs"), name=settings.experiment_name)
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    if not settings.wandb.enabled or local_rank != "0":
        return csv_logger
    os.environ.setdefault("WANDB_MODE", settings.wandb.mode)
    logger = WandbLogger(
        save_dir=str(work_dir),
        **{
            key: value
            for key, value in build_wandb_init_kwargs(
                run_prefix="crop_rotation",
                project=settings.wandb.project,
                entity=settings.wandb.entity,
                group=settings.wandb.group,
                job_type=settings.wandb.job_type,
                run_name=settings.wandb.run_name or settings.experiment_name,
            ).items()
            if value is not None
        },
    )
    if logger.experiment is not None:
        run_url = getattr(logger.experiment, "url", None)
        if callable(run_url):
            try:
                run_url = run_url()
            except TypeError:
                run_url = None
        logger.experiment.define_metric("trainer/global_step")
        logger.experiment.define_metric("epoch")
        for metric_name in (
            "train/loss",
            "train/accuracy",
            "val/accuracy",
            "val/loss",
            "val_rot90/accuracy",
            "val_rot180/accuracy",
            "val_rot270/accuracy",
            "val_hard/accuracy",
        ):
            logger.experiment.define_metric(metric_name, step_metric="trainer/global_step")
        print(
            f'{{"wandb_early_init": true, "wandb_run_id": "{logger.experiment.id}", '
            f'"wandb_run_name": "{logger.experiment.name}", "wandb_run_url": "{run_url}"}}'
        )
    logger.log_hyperparams(serialize_config(settings))
    return [logger, csv_logger]


def create_backbone(model_cfg: CropRotationModelConfig) -> torch.nn.Module:
    backbone = model_cfg.backbone.lower()
    if backbone != "mobilenet_v2":
        raise ValueError(f"Unsupported crop rotation backbone: {model_cfg.backbone}")
    weights = models.MobileNet_V2_Weights.DEFAULT if model_cfg.pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=float(model_cfg.dropout)),
        torch.nn.Linear(in_features, 4),
    )
    return model


class AspectPreservingSquareTransform:
    def __init__(self, image_size: tuple[int, int], fill_color: tuple[int, int, int] = (18, 18, 18)) -> None:
        self.image_size = tuple(int(value) for value in image_size)
        if self.image_size[0] != self.image_size[1]:
            raise ValueError(f"Crop rotation classifier expects a square image size, got {self.image_size}.")
        self.canvas_size = self.image_size[0]
        self.fill_color = tuple(int(value) for value in fill_color)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = self.canvas_size / max(width, height, 1)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize((resized_width, resized_height), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (self.canvas_size, self.canvas_size), color=self.fill_color)
        offset_x = (self.canvas_size - resized_width) // 2
        offset_y = (self.canvas_size - resized_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return self.normalize(self.to_tensor(canvas))


def build_detection_crop_records(
    repo_root: Path,
    dataset_root: Path,
    include_domains: list[str] | None,
) -> list[dict]:
    records: list[dict] = []
    for txt_path, domain in iter_detection_annotation_files(dataset_root, "train", include_domains):
        image_path = txt_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        instances: list[dict] = []
        for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
            try:
                polygon, transcription = parse_detection_line(raw_line)
            except ValueError:
                continue
            if not transcription.strip():
                continue
            instances.append({"polygon": polygon, "text": transcription})
        if instances:
            records.append(
                {
                    "domain": domain,
                    "image_path": image_path,
                    "repo_relative_path": str(image_path.relative_to(repo_root)),
                    "instances": instances,
                }
            )
    if not records:
        raise RuntimeError(f"No PTDR detection crop records found under {dataset_root}.")
    return records


def render_confusion_matrix_image(matrix: np.ndarray, title: str, class_names: list[str]) -> Image.Image:
    matrix = np.asarray(matrix, dtype=np.int64)
    cell_size = 88
    margin_left = 120
    margin_top = 88
    width = margin_left + cell_size * len(class_names) + 24
    height = margin_top + cell_size * len(class_names) + 36
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((16, 16), title, fill=(20, 20, 20))
    max_value = max(int(matrix.max()), 1)
    for col, class_name in enumerate(class_names):
        x = margin_left + col * cell_size + 24
        draw.text((x, 56), class_name, fill=(20, 20, 20))
    for row, class_name in enumerate(class_names):
        y = margin_top + row * cell_size + 32
        draw.text((24, y), class_name, fill=(20, 20, 20))
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            value = int(matrix[row, col])
            intensity = int(round(255 - (value / max_value) * 180))
            x0 = margin_left + col * cell_size
            y0 = margin_top + row * cell_size
            x1 = x0 + cell_size - 8
            y1 = y0 + cell_size - 8
            draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(intensity, intensity, 255), outline=(80, 80, 120))
            draw.text((x0 + 24, y0 + 28), str(value), fill=(10, 10, 10))
    return image


def ensure_recognition_eval_variants(repo_root: Path, settings: CropRotationExperimentConfig) -> Path:
    output_root = resolve_repo_relative(repo_root, settings.data.shared_eval_root)
    build_recognition_eval_variants(
        repo_root=repo_root,
        dataset_root=resolve_repo_relative(repo_root, settings.data.dataset_root),
        output_root=output_root,
        include_domains=settings.data.include_domains,
        val_ratio=settings.data.val_ratio,
        split_seed=settings.data.split_seed,
        map_size_bytes=8 * (1024**3),
        charset_policy=settings.data.charset_policy,
        hard_seed=settings.data.hard_seed,
        rotation_angles=(90, 180, 270),
    )
    return output_root


def ensure_crop_rotation_eval_labels(repo_root: Path, settings: CropRotationExperimentConfig) -> Path:
    output_root = resolve_repo_relative(repo_root, settings.data.shared_eval_label_root)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        return summary_path

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = resolve_repo_relative(repo_root, settings.data.dataset_root)
    raw_train_samples, _ = build_recognition_samples(
        repo_root=repo_root,
        dataset_root=dataset_root,
        split="train",
        include_domains=settings.data.include_domains,
    )
    charset_policy = resolve_charset_policy(repo_root, settings.data.charset_policy)
    train_samples, filter_errors = normalize_and_filter_samples(
        raw_train_samples,
        charset_policy,
        split_name="crop_rotation_eval_source",
    )
    _, val_samples = split_samples(train_samples, val_ratio=settings.data.val_ratio, seed=settings.data.split_seed)

    labels = {
        "val": [0 for _ in val_samples],
        "val_rot90": [angle_to_rotation_class(270) for _ in val_samples],
        "val_rot180": [angle_to_rotation_class(180) for _ in val_samples],
        "val_rot270": [angle_to_rotation_class(90) for _ in val_samples],
        "val_hard": [],
    }
    hard_preset_counts: dict[str, int] = {}
    for sample in val_samples:
        preset, _ = choose_preset_for_key(sample.get("repo_relative_path", ""), seed=settings.data.hard_seed, presets=DEFAULT_PRESETS)
        hard_preset_counts[preset["name"]] = hard_preset_counts.get(preset["name"], 0) + 1
        applied_rotation = int(round(float(preset.get("rotation_deg", 0.0)))) % 360
        if applied_rotation not in (0, 180):
            applied_rotation = 0
        labels["val_hard"].append(angle_to_rotation_class(correction_for_applied_rotation(applied_rotation)))

    summary = {
        "count": len(val_samples),
        "split_seed": settings.data.split_seed,
        "hard_seed": settings.data.hard_seed,
        "include_domains": list(settings.data.include_domains),
        "hard_preset_counts": dict(sorted(hard_preset_counts.items())),
        "filter_error_count": len(filter_errors),
        "labels": labels,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


class CropRotationTrainDataset(Dataset):
    def __init__(
        self,
        repo_root: Path,
        lmdb_root: Path,
        detection_dataset_root: Path,
        include_domains: list[str] | None,
        image_size: tuple[int, int],
        hard_probability: float,
        small_rotation_probability: float,
        small_rotation_max_degrees: float,
        detector_jitter_probability: float,
        detector_jitter_attempts: int,
        detection_image_cache_size: int,
        exhaustive_right_angle_classes: bool,
        detector_jitter_bank_root: Path | None,
    ) -> None:
        self.base = LmdbRecognitionDataset(lmdb_root)
        self.detection_records = build_detection_crop_records(
            repo_root=repo_root,
            dataset_root=detection_dataset_root,
            include_domains=include_domains,
        )
        self.image_size = tuple(int(v) for v in image_size)
        self.hard_probability = float(hard_probability)
        self.small_rotation_probability = float(small_rotation_probability)
        self.small_rotation_max_degrees = float(small_rotation_max_degrees)
        self.detector_jitter_probability = float(detector_jitter_probability)
        self.detector_jitter_attempts = max(1, int(detector_jitter_attempts))
        self.detection_image_cache_size = max(0, int(detection_image_cache_size))
        self.exhaustive_right_angle_classes = bool(exhaustive_right_angle_classes)
        self.transform = AspectPreservingSquareTransform(self.image_size)
        self._detection_image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.detector_jitter_bank = None
        if detector_jitter_bank_root is not None:
            bank_root = Path(detector_jitter_bank_root)
            if (bank_root / "data.mdb").exists():
                self.detector_jitter_bank = LmdbRecognitionDataset(bank_root)

    def __len__(self) -> int:
        if self.exhaustive_right_angle_classes:
            return len(self.base) * len(RIGHT_ANGLE_ROTATIONS)
        return len(self.base)

    def _sample_precomputed_detector_jitter_crop(self) -> np.ndarray | None:
        if self.detector_jitter_bank is None or len(self.detector_jitter_bank) == 0:
            return None
        image, _ = self.detector_jitter_bank[random.randrange(len(self.detector_jitter_bank))]
        image_rgb = np.asarray(image)
        if image_rgb.shape[0] < 2 or image_rgb.shape[1] < 2:
            return None
        return image_rgb

    def _sample_detector_jitter_crop(self) -> np.ndarray | None:
        record = random.choice(self.detection_records)
        instance = random.choice(record["instances"])
        image_rgb = self._load_detection_image(record["image_path"])
        image_height, image_width = image_rgb.shape[:2]
        for _ in range(self.detector_jitter_attempts):
            jittered_quad = jitter_quad_detector_style(
                polygon=instance["polygon"],
                image_size=(image_height, image_width),
                rng=random,
            )
            if jittered_quad is None:
                continue
            crop = perspective_crop_from_quad(image_rgb, jittered_quad)
            if crop is None:
                continue
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            return crop
        return None

    def _load_detection_image(self, image_path: Path) -> np.ndarray:
        cache_key = str(image_path)
        if self.detection_image_cache_size > 0:
            cached = self._detection_image_cache.get(cache_key)
            if cached is not None:
                self._detection_image_cache.move_to_end(cache_key)
                return cached
        image_rgb = load_rgb_image(image_path)
        if self.detection_image_cache_size > 0:
            self._detection_image_cache[cache_key] = image_rgb
            while len(self._detection_image_cache) > self.detection_image_cache_size:
                self._detection_image_cache.popitem(last=False)
        return image_rgb

    def __getitem__(self, index: int):
        if self.exhaustive_right_angle_classes:
            base_index = index // len(RIGHT_ANGLE_ROTATIONS)
            applied_right_angle = int(RIGHT_ANGLE_ROTATIONS[index % len(RIGHT_ANGLE_ROTATIONS)])
        else:
            base_index = index
            applied_right_angle = random.choice(RIGHT_ANGLE_ROTATIONS)
        image_rgb: np.ndarray | None = None
        if random.random() < self.detector_jitter_probability:
            image_rgb = self._sample_precomputed_detector_jitter_crop()
            if image_rgb is None:
                image_rgb = self._sample_detector_jitter_crop()
        if image_rgb is None:
            image, _ = self.base[base_index]
            image_rgb = np.asarray(image)
        if applied_right_angle:
            image_rgb = rotate_image_reflect(image_rgb, applied_right_angle)
        applied_total = applied_right_angle
        if random.random() < self.hard_probability:
            preset = random.choice(DEFAULT_PRESETS)
            image_rgb = apply_reflect_preset(image_rgb, preset=preset, seed=random.randint(0, 2**31 - 1))
            preset_rotation = int(round(float(preset.get("rotation_deg", 0.0)))) % 360
            if preset_rotation == 180:
                applied_total = (applied_total + 180) % 360
        if self.small_rotation_max_degrees > 0 and random.random() < self.small_rotation_probability:
            small_rotation = random.uniform(-self.small_rotation_max_degrees, self.small_rotation_max_degrees)
            image_rgb = rotate_image_reflect(image_rgb, small_rotation)
        correction_angle = correction_for_applied_rotation(applied_total)
        return self.transform(Image.fromarray(image_rgb, mode="RGB")), angle_to_rotation_class(correction_angle)


class CropRotationEvalDataset(Dataset):
    def __init__(self, lmdb_root: Path, labels: list[int], image_size: tuple[int, int]) -> None:
        self.base = LmdbRecognitionDataset(lmdb_root)
        self.labels = [int(label) for label in labels]
        if len(self.base) != len(self.labels):
            raise ValueError(f"Eval label count mismatch for {lmdb_root}: {len(self.base)} images vs {len(self.labels)} labels.")
        self.transform = AspectPreservingSquareTransform(image_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, _ = self.base[index]
        return self.transform(image), self.labels[index]


class CropRotationDataModule(pl.LightningDataModule):
    def __init__(self, repo_root: Path, settings: CropRotationExperimentConfig) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.settings = settings
        self.train_root = resolve_repo_relative(repo_root, settings.data.train_lmdb_root)
        self.eval_root = ensure_recognition_eval_variants(repo_root, settings)
        self.label_summary_path = ensure_crop_rotation_eval_labels(repo_root, settings)
        self.eval_label_summary = json.loads(self.label_summary_path.read_text(encoding="utf-8"))
        self.image_size = tuple(int(v) for v in settings.model.image_size)
        self.eval_names = ["val", "val_rot90", "val_rot180", "val_rot270", "val_hard"]

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = CropRotationTrainDataset(
            repo_root=self.repo_root,
            lmdb_root=self.train_root,
            detection_dataset_root=resolve_repo_relative(self.repo_root, self.settings.data.detection_dataset_root),
            include_domains=self.settings.data.include_domains,
            image_size=self.image_size,
            hard_probability=self.settings.data.hard_probability,
            small_rotation_probability=self.settings.data.small_rotation_probability,
            small_rotation_max_degrees=self.settings.data.small_rotation_max_degrees,
            detector_jitter_probability=self.settings.data.detector_jitter_probability,
            detector_jitter_attempts=self.settings.data.detector_jitter_attempts,
            detection_image_cache_size=self.settings.data.detection_image_cache_size,
            exhaustive_right_angle_classes=self.settings.data.exhaustive_right_angle_classes,
            detector_jitter_bank_root=resolve_repo_relative(self.repo_root, self.settings.data.detector_jitter_bank_root)
            if self.settings.data.detector_jitter_bank_root
            else None,
        )
        self.eval_datasets = {
            name: CropRotationEvalDataset(
                lmdb_root=self.eval_root / name / "PTDR",
                labels=self.eval_label_summary["labels"][name],
                image_size=self.image_size,
            )
            for name in self.eval_names
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.settings.training.batch_size,
            shuffle=True,
            num_workers=self.settings.training.num_workers,
            pin_memory=True,
            persistent_workers=self.settings.training.num_workers > 0,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.eval_datasets[name],
                batch_size=self.settings.training.batch_size,
                shuffle=False,
                num_workers=self.settings.training.num_workers,
                pin_memory=True,
                persistent_workers=self.settings.training.num_workers > 0,
            )
            for name in self.eval_names
        ]


class CropRotationLightningModule(pl.LightningModule):
    def __init__(self, settings: CropRotationExperimentConfig, eval_names: list[str]) -> None:
        super().__init__()
        self.save_hyperparameters(serialize_config(settings))
        self.settings = settings
        self.eval_names = list(eval_names)
        self.model = create_backbone(settings.model)
        self.class_names = ["0", "90", "180", "270"]
        self.validation_outputs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {name: [] for name in self.eval_names}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(self, batch, split_name: str):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(
            f"{split_name}/loss",
            loss,
            on_step=split_name == "train",
            on_epoch=True,
            prog_bar=split_name != "train",
            add_dataloader_idx=False,
            sync_dist=split_name != "train",
        )
        self.log(
            f"{split_name}/accuracy",
            accuracy,
            on_step=split_name == "train",
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=split_name != "train",
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        eval_name = self.eval_names[dataloader_idx]
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        self.log(f"{eval_name}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{eval_name}/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        self.validation_outputs[eval_name].append((predictions.detach().cpu(), labels.detach().cpu()))

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs = {name: [] for name in self.eval_names}

    def on_validation_epoch_end(self) -> None:
        work_dir = resolve_repo_relative(DEFAULT_REPO_ROOT, self.settings.training.work_dir)
        output_dir = work_dir / "confusion_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)
        rank_zero = getattr(self.trainer, "is_global_zero", True)
        wandb_logger = None
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        for eval_name in self.eval_names:
            items = self.validation_outputs.get(eval_name, [])
            if not items:
                continue
            local_payload = (
                torch.cat([item[0] for item in items], dim=0).numpy().tolist(),
                torch.cat([item[1] for item in items], dim=0).numpy().tolist(),
            )
            gathered_payloads = [local_payload]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if rank_zero:
                    gathered_payloads = [None for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.gather_object(local_payload, gathered_payloads, dst=0)
                else:
                    torch.distributed.gather_object(local_payload, None, dst=0)
                    continue
            if not rank_zero:
                continue
            predictions = np.asarray([value for payload in gathered_payloads for value in payload[0]], dtype=np.int64)
            labels = np.asarray([value for payload in gathered_payloads for value in payload[1]], dtype=np.int64)
            matrix = np.zeros((4, 4), dtype=np.int64)
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                matrix[int(label), int(prediction)] += 1
            matrix_path = output_dir / f"epoch_{int(self.current_epoch):02d}_{eval_name}.json"
            matrix_path.write_text(json.dumps(matrix.tolist(), indent=2), encoding="utf-8")
            image = render_confusion_matrix_image(matrix, title=f"{eval_name} confusion matrix", class_names=self.class_names)
            image_path = output_dir / f"epoch_{int(self.current_epoch):02d}_{eval_name}.png"
            image.save(image_path)
            if wandb_logger is not None and getattr(wandb_logger, "experiment", None) is not None:
                try:
                    import wandb

                    wandb_logger.experiment.log(
                        {
                            f"{eval_name}/confusion_matrix": wandb.Image(str(image_path)),
                            "trainer/global_step": int(self.global_step),
                        }
                    )
                except Exception:
                    pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.settings.training.lr,
            weight_decay=self.settings.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.settings.training.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


@pyrallis.wrap(config_path=str(SCRIPT_DIR / "configs" / "crop_rotation_classifier.yaml"))
def main(settings: CropRotationExperimentConfig) -> None:
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    work_dir = resolve_repo_relative(repo_root, settings.training.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(settings.training.seed, workers=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if settings.wandb.enabled:
        prepare_wandb_env(work_dir)
    logger = build_logger(settings, work_dir)

    data_module = CropRotationDataModule(repo_root=repo_root, settings=settings)
    data_module.setup("fit")
    model = CropRotationLightningModule(settings=settings, eval_names=data_module.eval_names)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=work_dir / "checkpoints",
            filename="best-{epoch:02d}",
            monitor="val_hard/accuracy",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=work_dir / "checkpoints",
            filename="epoch-{epoch:02d}",
            every_n_epochs=1,
            save_top_k=-1,
            auto_insert_metric_name=False,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        precision=settings.training.precision,
        max_epochs=settings.training.max_epochs,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=settings.training.log_every_n_steps,
        gradient_clip_val=settings.training.gradient_clip_val,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=str(settings.training.resume_from_checkpoint) if settings.training.resume_from_checkpoint else None)


if __name__ == "__main__":
    main()
