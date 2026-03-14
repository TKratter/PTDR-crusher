#!/usr/bin/env python3

import json
import os
import random
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import pyrallis
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_recognition_eval_variants import build_recognition_eval_variants
from build_recognition_manifest import (
    ExtraRecognitionTrainSource,
    build_recognition_manifests,
    recognition_manifest_outputs,
    recognition_manifests_exist,
)
from config_schema import (
    DEFAULT_REPO_ROOT,
    PARSeqExperimentConfig,
    resolve_repo_relative,
    serialize_config,
)
from external_datasets import build_external_recognition_train_samples, resolve_external_root
from recognition_augmentations import RandomHardRecognitionAugment, recognition_eval_variant_root


def source_name_for_dataset_root(dataset_root: Path) -> str:
    return dataset_root.name.strip().replace(" ", "_") or "extra_train"


def resolve_recognition_external_roots(repo_root: Path, settings: PARSeqExperimentConfig) -> dict[str, Path | None]:
    roots_cfg = settings.external_datasets
    return {
        "icdar2019_mlt": resolve_external_root(repo_root, roots_cfg.icdar2019_mlt),
        "evarest_detection": resolve_external_root(repo_root, roots_cfg.evarest_detection),
        "evarest_recognition": resolve_external_root(repo_root, roots_cfg.evarest_recognition),
        "totaltext": resolve_external_root(repo_root, roots_cfg.totaltext),
        "ctw1500": resolve_external_root(repo_root, roots_cfg.ctw1500),
        "textocr": resolve_external_root(repo_root, roots_cfg.textocr),
        "ir_lpr_detection": resolve_external_root(repo_root, roots_cfg.ir_lpr_detection),
        "ir_lpr_recognition": resolve_external_root(repo_root, roots_cfg.ir_lpr_recognition),
    }


def recognition_train_mix(settings: PARSeqExperimentConfig) -> dict[str, float]:
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


def validate_recognition_external_config(settings: PARSeqExperimentConfig, external_roots: dict[str, Path | None]) -> None:
    mix = recognition_train_mix(settings)
    for dataset_name, weight in mix.items():
        if dataset_name == "ptdr" or weight <= 0:
            continue
        if dataset_name == "evarest":
            candidates = [external_roots.get("evarest_recognition"), external_roots.get("evarest_detection")]
        elif dataset_name == "ir_lpr":
            candidates = [external_roots.get("ir_lpr_recognition"), external_roots.get("ir_lpr_detection")]
        else:
            root_key = dataset_name
            candidates = [external_roots.get(root_key)]
        if not any(path is not None and path.exists() for path in candidates):
            raise FileNotFoundError(f"Train mix enables {dataset_name} but no usable dataset root is configured.")


def build_wandb_init_kwargs(settings: PARSeqExperimentConfig) -> dict:
    wandb_cfg = settings.wandb
    run_id = os.environ.get("_PTDR_PARSEQ_WANDB_RUN_ID")
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
        os.environ["_PTDR_PARSEQ_WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RUN_ID"] = run_id
    return {
        "project": wandb_cfg.project,
        "entity": wandb_cfg.entity,
        "group": wandb_cfg.group,
        "job_type": wandb_cfg.job_type,
        "name": wandb_cfg.run_name or settings.experiment_name,
        "tags": wandb_cfg.tags,
        "save_dir": None,
        "id": run_id,
        "resume": "allow",
        "log_model": False,
    }


def build_logger(settings: PARSeqExperimentConfig, work_dir: Path):
    wandb_cfg = settings.wandb
    csv_logger = CSVLogger(save_dir=str(work_dir / "logs"), name=settings.experiment_name)
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    if not wandb_cfg.enabled or local_rank != "0":
        return csv_logger
    os.environ.setdefault("WANDB_MODE", wandb_cfg.mode)
    logger = WandbLogger(save_dir=str(work_dir), **{key: value for key, value in build_wandb_init_kwargs(settings).items() if value is not None and key != "save_dir"})
    if logger.experiment is not None:
        run_url = getattr(logger.experiment, "url", None)
        if callable(run_url):
            try:
                run_url = run_url()
            except TypeError:
                run_url = None
        logger.experiment.define_metric("trainer/global_step")
        for metric_name in (
            "loss",
            "val_loss",
            "val_accuracy",
            "val_NED",
            "lr-AdamW",
            "val_rot90/accuracy",
            "val_rot90/NED",
            "val_rot90/loss",
            "val_rot180/accuracy",
            "val_rot180/NED",
            "val_rot180/loss",
            "val_rot270/accuracy",
            "val_rot270/NED",
            "val_rot270/loss",
            "val_hard/accuracy",
            "val_hard/NED",
            "val_hard/loss",
        ):
            logger.experiment.define_metric(metric_name, step_metric="trainer/global_step")
        logger.experiment.define_metric("epoch")
        print(
            f'{{"wandb_early_init": true, "wandb_run_id": "{logger.experiment.id}", '
            f'"wandb_run_name": "{logger.experiment.name}", "wandb_run_url": "{run_url}"}}'
        )
    logger.log_hyperparams(serialize_config(settings))
    return [logger, csv_logger]


def prepare_wandb_env(work_dir: Path) -> None:
    wandb_root = work_dir / "wandb"
    cache_dir = wandb_root / "cache"
    config_dir = wandb_root / "config"
    data_dir = wandb_root / "data"
    artifact_dir = wandb_root / "artifacts"
    tmp_root = Path("/tmp") / "ptdr"
    tmp_dir = tmp_root / work_dir.name
    for directory in (wandb_root, cache_dir, config_dir, data_dir, artifact_dir, tmp_root, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_CONFIG_DIR"] = str(config_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ["WANDB_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["TMPDIR"] = str(tmp_dir)
    tempfile.tempdir = os.environ["TMPDIR"]


def determine_max_label_length(settings: PARSeqExperimentConfig, summary: dict) -> int:
    configured = settings.model.max_label_length
    if configured is None:
        return max(
            summary["train"]["max_label_length"],
            summary["val"]["max_label_length"],
            summary["test"]["max_label_length"],
        )
    return int(configured)


def resolve_collate_fn(collate_fn: str | None) -> str | None:
    # Treat the legacy "default" sentinel as "use the framework default".
    if collate_fn in (None, "", "default"):
        return None
    return collate_fn


def configure_torch_runtime(settings: PARSeqExperimentConfig) -> None:
    training_cfg = settings.training
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(training_cfg.matmul_precision)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = training_cfg.cudnn_benchmark


class PTDRSceneTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root_dir: str,
        train_dir: str,
        eval_root_dir: str,
        img_size: tuple[int, int],
        max_label_length: int,
        charset_train: str,
        charset_test: str,
        batch_size: int,
        num_workers: int,
        eval_num_workers: int,
        augment: bool,
        remove_whitespace: bool,
        normalize_unicode: bool,
        min_image_dim: int,
        rotation: int,
        collate_fn,
        hard_aug_train,
    ) -> None:
        super().__init__()
        self.train_root_dir = train_root_dir
        self.train_dir = train_dir
        self.eval_root_dir = Path(eval_root_dir)
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self.hard_aug_train = hard_aug_train
        self._train_dataset = None
        self._val_dataset = None
        self._extra_eval_loaders: dict[str, DataLoader] | None = None
        self._extra_eval_errors: dict[str, str] = {}

    @staticmethod
    def get_transform(
        img_size: tuple[int, int],
        augment: bool = False,
        rotation: int = 0,
        hard_aug_train=None,
    ):
        from torchvision import transforms as T
        from strhub.data.augment import rand_augment_transform

        transforms = []
        if augment:
            transforms.append(rand_augment_transform())
        if hard_aug_train is not None and getattr(hard_aug_train, "enabled", False):
            transforms.append(
                RandomHardRecognitionAugment(
                    probability=hard_aug_train.probability,
                    right_angle_rotation_probability=hard_aug_train.right_angle_rotation_probability,
                    rotation_angles=hard_aug_train.rotation_angles,
                    small_rotation_probability=hard_aug_train.small_rotation_probability,
                    small_rotation_max_degrees=hard_aug_train.small_rotation_max_degrees,
                    detector_jitter_probability=hard_aug_train.detector_jitter_probability,
                    detector_jitter_translate_ratio=hard_aug_train.detector_jitter_translate_ratio,
                    detector_jitter_scale_ratio=hard_aug_train.detector_jitter_scale_ratio,
                    detector_jitter_perspective_ratio=hard_aug_train.detector_jitter_perspective_ratio,
                )
            )
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True, fillcolor=(18, 18, 18)))
        transforms.extend(
            [
                T.Resize(img_size, T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )
        return T.Compose(transforms)

    def _build_dataset(self, root: Path | str, charset: str, augment: bool, rotation: int = 0):
        from torch.utils.data import ConcatDataset
        from strhub.data.dataset import LmdbDataset, build_tree_dataset

        transform = self.get_transform(
            self.img_size,
            augment=augment,
            rotation=rotation,
            hard_aug_train=self.hard_aug_train if augment else None,
        )
        try:
            return build_tree_dataset(
                root,
                charset,
                self.max_label_length,
                self.min_image_dim,
                self.remove_whitespace,
                self.normalize_unicode,
                transform=transform,
            )
        except AssertionError as exc:
            if "datasets should not be an empty iterable" not in str(exc):
                raise
            root_path = Path(root)
            lmdb_roots = sorted({mdb.parent for mdb in root_path.rglob("data.mdb")})
            if not lmdb_roots:
                raise
            datasets = [
                LmdbDataset(
                    str(lmdb_root),
                    charset,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
                for lmdb_root in lmdb_roots
            ]
            return ConcatDataset(datasets)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            root = Path(self.train_root_dir) / "train" / self.train_dir
            self._train_dataset = self._build_dataset(root, self.charset_train, augment=self.augment)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            root = self.eval_root_dir / "val"
            self._val_dataset = self._build_dataset(root, self.charset_test, augment=False)
        return self._val_dataset

    def build_eval_dataloader(self, split_name: str) -> DataLoader:
        dataset = self._build_dataset(self.eval_root_dir / split_name, self.charset_test, augment=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.eval_num_workers,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def extra_val_dataloaders(self) -> dict[str, DataLoader]:
        if self._extra_eval_loaders is None:
            self._extra_eval_loaders = {}
            self._extra_eval_errors = {}
            for split_name in ("val_rot90", "val_rot180", "val_rot270", "val_hard"):
                try:
                    self._extra_eval_loaders[split_name] = self.build_eval_dataloader(split_name)
                except Exception as exc:
                    self._extra_eval_errors[split_name] = f"{type(exc).__name__}: {exc}"
                    print(
                        json.dumps(
                            {
                                "warning": "parseq_extra_val_split_unavailable",
                                "split": split_name,
                                "error": self._extra_eval_errors[split_name],
                            },
                            ensure_ascii=False,
                        )
                    )
        return self._extra_eval_loaders

    @property
    def extra_eval_errors(self) -> dict[str, str]:
        return dict(self._extra_eval_errors)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.eval_num_workers,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


def load_pretrained_recognizer_weights(model, experiment: str) -> None:
    from strhub.models.utils import get_pretrained_weights

    target_model = model.model if "parseq" in experiment else model
    pretrained_state = get_pretrained_weights(experiment)
    if "parseq" not in experiment:
        target_model.load_state_dict(pretrained_state)
        return

    load_parseq_state_dict_compatibly(
        target_model=target_model,
        source_state=pretrained_state,
        source_label=f"pretrained::{experiment}",
    )


def load_parseq_state_dict_compatibly(target_model, source_state: dict, source_label: str) -> None:
    normalized_state = {}
    for key, value in source_state.items():
        normalized_key = key[6:] if key.startswith("model.") else key
        normalized_state[normalized_key] = value

    charset_dependent_prefixes = ("head.", "text_embed.embedding.")
    target_state = target_model.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in normalized_state.items():
        if key.startswith(charset_dependent_prefixes):
            skipped_keys.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None or target_value.shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    incompatible = target_model.load_state_dict(compatible_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected PARSeq checkpoint keys for {source_label}: {sorted(incompatible.unexpected_keys)}"
        )

    print(
        json.dumps(
            {
                "checkpoint_source": source_label,
                "loaded_tensors": len(compatible_state),
                "reinitialized_tensors": sorted(skipped_keys),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def load_checkpoint_initializer(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at {checkpoint_path} does not contain a Lightning state_dict.")
    target_model = model.model if hasattr(model, "model") else model
    load_parseq_state_dict_compatibly(
        target_model=target_model,
        source_state=state_dict,
        source_label=f"checkpoint::{checkpoint_path}",
    )


def prepare_safe_resume(
    model,
    repo_root: Path,
    resume_from_checkpoint: Path | None,
    configured_max_epochs: int,
) -> tuple[Path | None, int]:
    checkpoint_path = resolve_repo_relative(repo_root, resume_from_checkpoint)
    if checkpoint_path is None:
        return None, int(configured_max_epochs)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing PARSeq resume checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    completed_epochs = max(0, int(checkpoint.get("epoch", -1)) + 1)
    remaining_epochs = int(configured_max_epochs) - completed_epochs
    if remaining_epochs <= 0:
        raise SystemExit(
            f"Resume checkpoint {checkpoint_path} already completed {completed_epochs} epochs, "
            f"which meets/exceeds configured max_epochs={configured_max_epochs}."
        )

    load_checkpoint_initializer(model, checkpoint_path)

    print(
        json.dumps(
            {
                "resume_strategy": "weights_only_remaining_epochs",
                "resume_checkpoint": str(checkpoint_path),
                "completed_epochs": completed_epochs,
                "configured_max_epochs": int(configured_max_epochs),
                "effective_max_epochs": remaining_epochs,
                "global_step_at_checkpoint": int(checkpoint.get("global_step", 0)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return checkpoint_path, remaining_epochs


def iter_trainer_loggers(trainer) -> list:
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return [logger for logger in loggers if logger is not None]
    logger = getattr(trainer, "logger", None)
    return [logger] if logger is not None else []


def load_annotation_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().clamp(-1, 1)
    image = ((image * 0.5) + 0.5).permute(1, 2, 0).numpy()
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def wrap_caption_line(prefix: str, text: str, width: int = 56) -> list[str]:
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return [prefix]
    lines = [f"{prefix}{wrapped[0]}"]
    indent = " " * len(prefix)
    lines.extend(f"{indent}{line}" for line in wrapped[1:])
    return lines


def render_parseq_sample_image(image_tensor: torch.Tensor, gt_text: str, pred_text: str, is_match: bool) -> np.ndarray:
    image = Image.fromarray(tensor_to_rgb_image(image_tensor))
    font = load_annotation_font(18)
    small_font = load_annotation_font(16)
    status = "TRUE POSITIVE" if is_match else "FALSE"
    lines = [status]
    lines.extend(wrap_caption_line("GT: ", gt_text))
    lines.extend(wrap_caption_line("PRED: ", pred_text))

    drawer = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    line_heights = []
    line_widths = []
    for index, line in enumerate(lines):
        current_font = font if index == 0 else small_font
        bbox = drawer.textbbox((0, 0), line, font=current_font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    padding = 12
    spacing = 6
    text_height = sum(line_heights) + spacing * (len(lines) - 1)
    canvas_width = max(image.width, max(line_widths, default=0) + padding * 2)
    canvas = Image.new("RGB", (canvas_width, image.height + text_height + padding * 2), "white")
    canvas.paste(image, (0, 0))
    drawer = ImageDraw.Draw(canvas)

    y = image.height + padding
    for index, line in enumerate(lines):
        current_font = font if index == 0 else small_font
        fill = (0, 128, 0) if index == 0 and is_match else (196, 0, 0) if index == 0 else (0, 0, 0)
        drawer.text((padding, y), line, fill=fill, font=current_font)
        y += line_heights[index] + spacing

    return np.array(canvas)


class PARSeqExtraValEval(pl.Callback):
    def __init__(self, work_dir: Path, interval: int = 1) -> None:
        self.output_dir = work_dir / "extra_val_metrics"
        self.expected_splits = ("val_rot90", "val_rot180", "val_rot270", "val_hard")
        self.interval = max(1, int(interval))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if (trainer.current_epoch + 1) % self.interval != 0:
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "extra_val_dataloaders"):
            return

        extra_loaders = datamodule.extra_val_dataloaders()

        metric_keys = [
            f"{split_name}/{metric_name}"
            for split_name in self.expected_splits
            for metric_name in ("accuracy", "NED", "loss")
        ]
        was_training = pl_module.training
        metrics: dict[str, float] = {}
        try:
            if getattr(trainer, "is_global_zero", True):
                pl_module.eval()
                for split_name, dataloader in extra_loaders.items():
                    outputs = []
                    for images, labels in dataloader:
                        images = images.to(pl_module.device, non_blocking=True)
                        with torch.inference_mode():
                            result = pl_module._eval_step((images, list(labels)), True)
                        outputs.append(result)
                    acc, ned, loss = pl_module._aggregate_results(outputs)
                    metrics[f"{split_name}/accuracy"] = 100 * float(acc)
                    metrics[f"{split_name}/NED"] = 100 * float(ned)
                    metrics[f"{split_name}/loss"] = float(loss.detach().cpu()) if hasattr(loss, "detach") else float(loss)
                for split_name in self.expected_splits:
                    if split_name in extra_loaders:
                        continue
                    metrics.setdefault(f"{split_name}/accuracy", 0.0)
                    metrics.setdefault(f"{split_name}/NED", 0.0)
                    metrics.setdefault(f"{split_name}/loss", 0.0)
                extra_eval_errors = getattr(datamodule, "extra_eval_errors", {})
                for split_name, error in extra_eval_errors.items():
                    print(
                        json.dumps(
                            {
                                "warning": "parseq_extra_val_split_skipped",
                                "split": split_name,
                                "error": error,
                                "epoch": trainer.current_epoch + 1,
                            },
                            ensure_ascii=False,
                        )
                    )

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                for key in metric_keys:
                    value = metrics.get(key, 0.0)
                    buffer = torch.tensor([value], device=pl_module.device, dtype=torch.float32)
                    torch.distributed.broadcast(buffer, src=0)
                    metrics[key] = float(buffer.item())
        finally:
            if was_training:
                pl_module.train()

        callback_metrics = getattr(trainer, "callback_metrics", None)
        if callback_metrics is not None:
            for key, value in metrics.items():
                callback_metrics[key] = torch.tensor(value, device=pl_module.device)

        epoch = trainer.current_epoch + 1
        if getattr(trainer, "is_global_zero", True):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"epoch_{epoch:03d}.json"
            output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            for logger in iter_trainer_loggers(trainer):
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(metrics, step=trainer.global_step)


class PARSeqValSampleLogger(pl.Callback):
    def __init__(self, work_dir: Path, seed: int) -> None:
        self.output_dir = work_dir / "val_samples" / "parseq"
        self.seed = seed

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val_dataset"):
            return

        val_dataset = datamodule.val_dataset
        if len(val_dataset) == 0:
            return

        rng = random.Random(self.seed + trainer.current_epoch)
        indices = list(range(len(val_dataset)))
        rng.shuffle(indices)
        search_batch_size = max(1, min(32, getattr(datamodule, "batch_size", 32)))

        positive_sample = None
        false_sample = None
        was_training = pl_module.training
        pl_module.eval()
        try:
            for start in range(0, len(indices), search_batch_size):
                batch_indices = indices[start:start + search_batch_size]
                batch_items = [val_dataset[index] for index in batch_indices]
                images = torch.stack([item[0] for item in batch_items]).to(pl_module.device)
                with torch.inference_mode():
                    logits = pl_module(images)
                predictions, _ = pl_module.tokenizer.decode(logits.softmax(-1))
                for dataset_index, (image_tensor, label), prediction in zip(batch_indices, batch_items, predictions):
                    pred_text = pl_module.charset_adapter(prediction)
                    sample = {
                        "dataset_index": dataset_index,
                        "image_tensor": image_tensor,
                        "gt_text": label,
                        "pred_text": pred_text,
                        "is_match": pred_text == label,
                    }
                    if sample["is_match"] and positive_sample is None:
                        positive_sample = sample
                    elif not sample["is_match"] and false_sample is None:
                        false_sample = sample
                if positive_sample is not None and false_sample is not None:
                    break
        finally:
            if was_training:
                pl_module.train()

        epoch = trainer.current_epoch + 1
        if positive_sample is not None:
            self.log_sample(trainer, epoch, "true_positive", positive_sample)
        if false_sample is not None:
            self.log_sample(trainer, epoch, "false", false_sample)

    def log_sample(self, trainer: pl.Trainer, epoch: int, sample_kind: str, sample: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        annotated = render_parseq_sample_image(
            sample["image_tensor"],
            sample["gt_text"],
            sample["pred_text"],
            sample["is_match"],
        )
        stem = f"epoch_{epoch:03d}_{sample_kind}"
        image_path = self.output_dir / f"{stem}.png"
        metadata_path = self.output_dir / f"{stem}.json"
        Image.fromarray(annotated).save(image_path)
        metadata = {
            "epoch": epoch,
            "sample_kind": sample_kind,
            "dataset_index": sample["dataset_index"],
            "gt_text": sample["gt_text"],
            "pred_text": sample["pred_text"],
            "is_match": sample["is_match"],
            "logged_image": str(image_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        caption = (
            f"epoch={epoch} | type={sample_kind} | gt={sample['gt_text']} | pred={sample['pred_text']}"
        )
        for logger in iter_trainer_loggers(trainer):
            if hasattr(logger, "log_image"):
                logger.log_image(
                    key=f"val_samples/{sample_kind}",
                    images=[annotated],
                    caption=[caption],
                    step=trainer.global_step,
                )


@pyrallis.wrap(config_path=str(SCRIPT_DIR / "configs" / "parseq.yaml"))
def main(settings: PARSeqExperimentConfig) -> None:
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    dataset_root = resolve_repo_relative(repo_root, settings.dataset_root)
    manifests_dir = resolve_repo_relative(repo_root, settings.manifests_dir)
    shared_eval_root = recognition_eval_variant_root(repo_root)
    work_dir = resolve_repo_relative(repo_root, settings.training.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    external_roots = resolve_recognition_external_roots(repo_root, settings)
    validate_recognition_external_config(settings, external_roots)
    if settings.wandb.enabled:
        prepare_wandb_env(work_dir)
    logger = build_logger(settings, work_dir)
    extra_train_sources = []
    if settings.ptdr_synth_root is not None:
        ptdr_synth_root = resolve_repo_relative(repo_root, settings.ptdr_synth_root)
        if ptdr_synth_root is None or not ptdr_synth_root.exists():
            raise FileNotFoundError(
                f"Missing PTDR-Synth root: {ptdr_synth_root}. "
                "Download it first or point ptdr_synth_root at the extracted gt.txt + images directory."
            )
        extra_train_sources.append(
            ExtraRecognitionTrainSource(
                dataset_root=ptdr_synth_root,
                include_domains=None,
                name="ptdr_synth",
                layout="flat",
            )
        )
    for extra_dataset_root in settings.extra_train_dataset_roots:
        extra_root = resolve_repo_relative(repo_root, extra_dataset_root)
        if not extra_root.exists():
            raise FileNotFoundError(
                f"Missing extra PARSeq training dataset root: {extra_root}. "
                "Download or place the external dataset there before training."
            )
        extra_train_sources.append(
            ExtraRecognitionTrainSource(
                dataset_root=extra_root,
                include_domains=settings.include_domains,
                name=source_name_for_dataset_root(extra_root),
            )
        )
    if recognition_manifests_exist(manifests_dir, repo_root=repo_root, charset_policy=settings.charset_policy):
        manifests = recognition_manifest_outputs(manifests_dir)
    else:
        external_train_samples, external_train_errors = build_external_recognition_train_samples(
            repo_root=repo_root,
            external_roots=external_roots,
            mlt_scripts=settings.mlt.scripts,
        )
        manifests = build_recognition_manifests(
            repo_root=repo_root,
            dataset_root=dataset_root,
            output_root=manifests_dir,
            include_domains=settings.include_domains,
            val_ratio=settings.val_ratio,
            seed=settings.split_seed,
            map_size_bytes=int(settings.lmdb_map_size_gb * (1024**3)),
            extra_train_sources=extra_train_sources,
            external_train_samples=external_train_samples,
            train_mix=recognition_train_mix(settings),
            min_ptdr_fraction=settings.train_mix.min_ptdr_fraction,
            extra_errors=external_train_errors,
            charset_policy=settings.charset_policy,
            include_ptdr_train_in_train_split=settings.include_ptdr_train_in_train_split,
            exclude_extra_train_label_overlap_from_eval=settings.exclude_extra_train_label_overlap_from_eval,
        )
    shared_eval_manifests = build_recognition_eval_variants(
        repo_root=repo_root,
        dataset_root=dataset_root,
        output_root=shared_eval_root,
        include_domains=settings.include_domains,
        val_ratio=settings.val_ratio,
        split_seed=settings.split_seed,
        map_size_bytes=int(settings.lmdb_map_size_gb * (1024**3)),
        charset_policy=settings.charset_policy,
        hard_seed=settings.hard_aug.eval.seed,
        rotation_angles=settings.hard_aug.eval.rotation_angles,
    )
    summary = json.loads(Path(manifests["summary"]).read_text(encoding="utf-8"))
    error_count = len(summary["errors"])
    if settings.fail_on_bad_annotations and error_count:
        raise SystemExit(f"Found {error_count} malformed recognition annotations. Inspect {manifests['summary']}.")
    if summary["val"]["count"] == 0:
        raise SystemExit(
            "Recognition validation split is empty after manifest filtering. "
            "This usually means strict extra-train label-overlap exclusion removed all PTDR val samples. "
            "Disable overlap exclusion or use a different extra training dataset."
        )
    if summary["test"]["count"] == 0:
        raise SystemExit(
            "Recognition test split is empty after manifest filtering. "
            "Adjust the overlap exclusion policy or use a different extra training dataset."
        )

    charset_train = Path(manifests["charset_train"]).read_text(encoding="utf-8")
    charset_eval = Path(manifests["charset_eval"]).read_text(encoding="utf-8")
    if not charset_train or not charset_eval:
        raise SystemExit(
            "Generated an empty PARSeq charset. Inspect the manifest summary and selected include_domains "
            f"under {manifests['summary']}."
        )
    model_charset_train = charset_eval if settings.model.use_eval_charset_for_train_charset else charset_train
    max_label_length = determine_max_label_length(settings, summary)

    pl.seed_everything(settings.training.seed, workers=True)
    configure_torch_runtime(settings)

    from strhub.models.utils import create_model

    model = create_model(
        settings.model.name,
        pretrained=False,
        img_size=tuple(settings.model.img_size),
        max_label_length=max_label_length,
        charset_train=model_charset_train,
        charset_test=charset_eval,
        batch_size=settings.training.batch_size,
        lr=settings.training.lr,
        warmup_pct=settings.training.warmup_pct,
        weight_decay=settings.training.weight_decay,
    )
    if settings.model.pretrained:
        load_pretrained_recognizer_weights(model, settings.model.name)
    if settings.model.init_from_checkpoint is not None:
        checkpoint_path = resolve_repo_relative(repo_root, settings.model.init_from_checkpoint)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing PARSeq initialization checkpoint: {checkpoint_path}")
        load_checkpoint_initializer(model, checkpoint_path)
    resume_checkpoint_path, effective_max_epochs = prepare_safe_resume(
        model=model,
        repo_root=repo_root,
        resume_from_checkpoint=settings.training.resume_from_checkpoint,
        configured_max_epochs=settings.training.max_epochs,
    )

    datamodule = PTDRSceneTextDataModule(
        train_root_dir=str(Path(manifests["root_dir"]).resolve()),
        train_dir=manifests["train_dir"],
        eval_root_dir=str(shared_eval_manifests["val_root"].parent.resolve()),
        img_size=tuple(settings.model.img_size),
        max_label_length=max_label_length,
        charset_train=model_charset_train,
        charset_test=charset_eval,
        batch_size=settings.training.batch_size,
        num_workers=settings.training.num_workers,
        eval_num_workers=settings.training.eval_num_workers,
        augment=settings.data.augment,
        remove_whitespace=settings.data.remove_whitespace,
        normalize_unicode=settings.data.normalize_unicode,
        min_image_dim=settings.data.min_image_dim,
        rotation=settings.data.rotation,
        collate_fn=resolve_collate_fn(settings.data.collate_fn),
        hard_aug_train=settings.hard_aug.train,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(work_dir / "checkpoints"),
            filename="best-val-{epoch:03d}-{step:07d}",
            monitor="val_accuracy",
            mode="max",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=str(work_dir / "checkpoints"),
            filename="best-valhard-{epoch:03d}-{step:07d}",
            monitor="val_hard/accuracy",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False,
        ),
        PARSeqExtraValEval(work_dir=work_dir, interval=settings.training.extra_val_interval),
    ]
    if settings.training.enable_val_sample_logging:
        callbacks.append(PARSeqValSampleLogger(work_dir=work_dir, seed=settings.training.seed))

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        precision=settings.training.precision,
        max_epochs=effective_max_epochs,
        log_every_n_steps=settings.training.log_every_n_steps,
        accumulate_grad_batches=settings.training.accumulate_grad_batches,
        gradient_clip_val=settings.training.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
    )

    print(
        json.dumps(
            {
                "experiment_name": settings.experiment_name,
                "manifest_root": str(manifests["root_dir"]),
                "shared_eval_root": str(shared_eval_root),
                "work_dir": str(work_dir),
                "charset_train_size": len(charset_train),
                "model_charset_train_size": len(model_charset_train),
                "charset_eval_size": len(charset_eval),
                "max_label_length": max_label_length,
                "include_ptdr_train_in_train_split": settings.include_ptdr_train_in_train_split,
                "extra_train_dataset_count": summary.get("extra_train_dataset_count", 0),
                "external_train_dataset_count": summary.get("external_train_dataset_count", 0),
                "ptdr_synth_root": str(resolve_repo_relative(repo_root, settings.ptdr_synth_root))
                if settings.ptdr_synth_root is not None
                else None,
                "init_from_checkpoint": str(resolve_repo_relative(repo_root, settings.model.init_from_checkpoint))
                if settings.model.init_from_checkpoint is not None
                else None,
                "use_eval_charset_for_train_charset": settings.model.use_eval_charset_for_train_charset,
                "exclude_extra_train_label_overlap_from_eval": settings.exclude_extra_train_label_overlap_from_eval,
                "train_mix": recognition_train_mix(settings),
                "min_ptdr_fraction": settings.train_mix.min_ptdr_fraction,
                "mlt_scripts": settings.mlt.scripts,
                "hard_aug": serialize_config(settings.hard_aug),
                "charset_policy": serialize_config(settings.charset_policy),
                "num_workers": settings.training.num_workers,
                "eval_num_workers": settings.training.eval_num_workers,
                "batch_size": settings.training.batch_size,
                "cudnn_benchmark": settings.training.cudnn_benchmark,
                "matmul_precision": settings.training.matmul_precision,
                "skipped_annotations": error_count,
                "effective_max_epochs": effective_max_epochs,
                "resume_checkpoint_path": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    main()
