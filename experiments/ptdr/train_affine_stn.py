#!/usr/bin/env python3

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import pyrallis
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_detection_eval_variants import build_detection_eval_variants
from config_schema import DEFAULT_REPO_ROOT, WandbConfig, resolve_repo_relative, serialize_config
from mmocr.utils import poly_iou, polys2shapely
from rotation_solution_utils import (
    build_synthetic_affine_example,
    build_wandb_init_kwargs,
    load_rgb_image,
    normalized_affine_from_pixel_matrix,
    prepare_wandb_env,
    resize_image_and_polygons,
    transform_polygon_with_theta_inverse,
)


@dataclass
class AffineSTNModelConfig:
    input_size: List[int] = field(default_factory=lambda: [512, 512])
    hidden_dim: int = 128


@dataclass
class AffineSTNTrainingConfig:
    work_dir: Path = Path("work_dirs/affine_stn")
    seed: int = 42
    batch_size: int = 12
    num_workers: int = 8
    max_epochs: int = 20
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    lr: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 25
    recon_loss_weight: float = 1.0
    theta_loss_weight: float = 0.25
    resume_from_checkpoint: Optional[Path] = None


@dataclass
class AffineSTNDataConfig:
    train_manifest_path: Path = Path("experiments/ptdr/manifests/dbnet_multidata_all_r18_hard_faststable/textdet_train.json")
    val_manifest_path: Path = Path("experiments/ptdr/manifests/dbnet_multidata_all_r18_hard_faststable/textdet_val.json")
    test_manifest_path: Path = Path("experiments/ptdr/manifests/dbnet_multidata_all_r18_hard_faststable/textdet_test.json")
    shared_eval_root: Path = Path("experiments/ptdr/manifests/shared_textdet_eval_variants")
    right_angle_probability: float = 0.45
    max_small_rotation_deg: float = 18.0
    min_scale: float = 0.92
    max_scale: float = 1.08
    max_translation_ratio: float = 0.08
    max_shear_deg: float = 10.0
    photometric_probability: float = 0.8
    hard_seed: int = 42


@dataclass
class AffineSTNExperimentConfig:
    experiment_name: str = "ptdr-affine-stn"
    repo_root: Path = Path(".")
    model: AffineSTNModelConfig = field(default_factory=AffineSTNModelConfig)
    training: AffineSTNTrainingConfig = field(default_factory=AffineSTNTrainingConfig)
    data: AffineSTNDataConfig = field(default_factory=AffineSTNDataConfig)
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            group="affine-stn",
            tags=["ptdr", "stn", "rotation", "detection-preproc", "hard-train"],
        )
    )


def load_manifest_records(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload["data_list"])


def build_logger(settings: AffineSTNExperimentConfig, work_dir: Path):
    csv_logger = CSVLogger(save_dir=str(work_dir / "logs"), name=settings.experiment_name)
    if not settings.wandb.enabled:
        return csv_logger
    os.environ.setdefault("WANDB_MODE", settings.wandb.mode)
    logger = WandbLogger(
        save_dir=str(work_dir),
        **{
            key: value
            for key, value in build_wandb_init_kwargs(
                run_prefix="affine_stn",
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
        logger.experiment.define_metric("trainer/global_step")
        logger.experiment.define_metric("epoch")
        for metric_name in (
            "train/loss",
            "train/recon_l1",
            "train/theta_l1",
            "val/polygon_iou",
            "val_hard/polygon_iou",
            "val_rot90/polygon_iou",
            "val_rot180/polygon_iou",
            "val_rot270/polygon_iou",
        ):
            logger.experiment.define_metric(metric_name, step_metric="trainer/global_step")
        print(
            json.dumps(
                {
                    "wandb_early_init": True,
                    "wandb_run_id": logger.experiment.id,
                    "wandb_run_name": logger.experiment.name,
                    "wandb_run_url": getattr(logger.experiment, "url", None),
                },
                ensure_ascii=False,
            )
        )
    logger.log_hyperparams(serialize_config(settings))
    return [logger, csv_logger]


def ensure_shared_detection_eval_root(repo_root: Path, settings: AffineSTNExperimentConfig) -> Path:
    output_root = resolve_repo_relative(repo_root, settings.data.shared_eval_root)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        return output_root
    build_detection_eval_variants(
        repo_root=repo_root,
        val_ann_path=resolve_repo_relative(repo_root, settings.data.val_manifest_path),
        test_ann_path=resolve_repo_relative(repo_root, settings.data.test_manifest_path),
        output_root=output_root,
        seed=settings.data.hard_seed,
        rotation_angles=(90, 180, 270),
        include_test_hard=True,
    )
    return output_root


def tensor_from_rgb_image(image_rgb) -> torch.Tensor:
    return torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0


def mean_polygon_iou(clean_polygons: list[list[float]], predicted_polygons: list[list[float]]) -> float:
    if not clean_polygons or not predicted_polygons:
        return 0.0
    count = min(len(clean_polygons), len(predicted_polygons))
    clean_shapes = polys2shapely(clean_polygons[:count])
    predicted_shapes = polys2shapely(predicted_polygons[:count])
    ious: list[float] = []
    for clean_shape, predicted_shape in zip(clean_shapes, predicted_shapes):
        try:
            ious.append(float(poly_iou(clean_shape, predicted_shape)))
        except Exception:
            ious.append(0.0)
    return float(sum(ious) / max(len(ious), 1))


class SyntheticAffineTrainDataset(Dataset):
    def __init__(self, repo_root: Path, manifest_path: Path, input_size: tuple[int, int], settings: AffineSTNDataConfig, seed: int) -> None:
        self.repo_root = repo_root
        self.records = load_manifest_records(manifest_path)
        self.input_size = tuple(int(v) for v in input_size)
        self.settings = settings
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        clean_image = load_rgb_image(self.repo_root / record["img_path"])
        polygons = [instance["polygon"] for instance in record.get("instances", []) if not instance.get("ignore", False)]
        clean_resized, _ = resize_image_and_polygons(clean_image, polygons, self.input_size)
        worker_seed = (torch.initial_seed() % (2**31)) + self.seed + index * 1009
        rng = random.Random(worker_seed)
        hard_image, forward_matrix, _ = build_synthetic_affine_example(
            clean_image_rgb=clean_resized,
            rng=rng,
            max_small_rotation_deg=self.settings.max_small_rotation_deg,
            right_angle_probability=self.settings.right_angle_probability,
            min_scale=self.settings.min_scale,
            max_scale=self.settings.max_scale,
            max_translation_ratio=self.settings.max_translation_ratio,
            max_shear_deg=self.settings.max_shear_deg,
            photometric_probability=self.settings.photometric_probability,
        )
        theta = normalized_affine_from_pixel_matrix(forward_matrix, input_size=self.input_size)
        return {
            "hard": tensor_from_rgb_image(hard_image),
            "clean": tensor_from_rgb_image(clean_resized),
            "theta": torch.from_numpy(theta),
        }


class PairedAffineEvalDataset(Dataset):
    def __init__(self, repo_root: Path, clean_manifest_path: Path, hard_manifest_path: Path, input_size: tuple[int, int]) -> None:
        self.repo_root = repo_root
        self.clean_records = load_manifest_records(clean_manifest_path)
        self.hard_records = load_manifest_records(hard_manifest_path)
        if len(self.clean_records) != len(self.hard_records):
            raise ValueError(f"Mismatched clean/hard eval sizes: {clean_manifest_path} vs {hard_manifest_path}")
        self.input_size = tuple(int(v) for v in input_size)

    def __len__(self) -> int:
        return len(self.clean_records)

    def __getitem__(self, index: int):
        clean_record = self.clean_records[index]
        hard_record = self.hard_records[index]
        clean_image = load_rgb_image(self.repo_root / clean_record["img_path"])
        hard_image = load_rgb_image(self.repo_root / hard_record["img_path"])
        clean_polygons = [instance["polygon"] for instance in clean_record.get("instances", []) if not instance.get("ignore", False)]
        hard_polygons = [instance["polygon"] for instance in hard_record.get("instances", []) if not instance.get("ignore", False)]
        clean_resized, clean_polygons = resize_image_and_polygons(clean_image, clean_polygons, self.input_size)
        hard_resized, hard_polygons = resize_image_and_polygons(hard_image, hard_polygons, self.input_size)
        return {
            "hard": tensor_from_rgb_image(hard_resized),
            "clean": tensor_from_rgb_image(clean_resized),
            "hard_polygons": hard_polygons,
            "clean_polygons": clean_polygons,
        }


class AffineSTNDataModule(pl.LightningDataModule):
    def __init__(self, repo_root: Path, settings: AffineSTNExperimentConfig) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.settings = settings
        self.train_manifest = resolve_repo_relative(repo_root, settings.data.train_manifest_path)
        self.clean_val_manifest = resolve_repo_relative(repo_root, settings.data.val_manifest_path)
        self.shared_eval_root = ensure_shared_detection_eval_root(repo_root, settings)
        self.input_size = tuple(int(v) for v in settings.model.input_size)
        self.eval_names = ["val", "val_rot90", "val_rot180", "val_rot270", "val_hard"]

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = SyntheticAffineTrainDataset(
            repo_root=self.repo_root,
            manifest_path=self.train_manifest,
            input_size=self.input_size,
            settings=self.settings.data,
            seed=self.settings.training.seed,
        )
        variant_paths = {
            "val": self.clean_val_manifest,
            "val_rot90": self.shared_eval_root / "textdet_val_rot90.json",
            "val_rot180": self.shared_eval_root / "textdet_val_rot180.json",
            "val_rot270": self.shared_eval_root / "textdet_val_rot270.json",
            "val_hard": self.shared_eval_root / "textdet_val_hard.json",
        }
        self.eval_datasets = {
            name: PairedAffineEvalDataset(
                repo_root=self.repo_root,
                clean_manifest_path=self.clean_val_manifest,
                hard_manifest_path=variant_paths[name],
                input_size=self.input_size,
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
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda batch: batch[0],
            )
            for name in self.eval_names
        ]


class AffineSTNBackbone(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 6),
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.regressor(self.localization(images)).view(-1, 2, 3)
        grid = F.affine_grid(theta, images.size(), align_corners=True)
        normalized = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return normalized, theta


class AffineSTNLightningModule(pl.LightningModule):
    def __init__(self, settings: AffineSTNExperimentConfig, eval_names: list[str]) -> None:
        super().__init__()
        self.save_hyperparameters(serialize_config(settings))
        self.settings = settings
        self.eval_names = list(eval_names)
        self.model = AffineSTNBackbone(hidden_dim=settings.model.hidden_dim)
        self.input_size = tuple(int(v) for v in settings.model.input_size)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        normalized, theta = self(batch["hard"])
        recon_l1 = F.l1_loss(normalized, batch["clean"])
        theta_l1 = F.smooth_l1_loss(theta, batch["theta"])
        loss = (
            self.settings.training.recon_loss_weight * recon_l1
            + self.settings.training.theta_loss_weight * theta_l1
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_l1", recon_l1, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/theta_l1", theta_l1, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        eval_name = self.eval_names[dataloader_idx]
        hard = batch["hard"].unsqueeze(0)
        clean = batch["clean"].unsqueeze(0)
        normalized, theta = self(hard)
        recon_l1 = F.l1_loss(normalized, clean)
        theta_np = theta[0].detach().cpu().numpy()
        predicted_polygons = [
            transform_polygon_with_theta_inverse(polygon, theta_np, input_size=self.input_size)
            for polygon in batch["hard_polygons"]
        ]
        raw_iou = mean_polygon_iou(batch["clean_polygons"], batch["hard_polygons"])
        pred_iou = mean_polygon_iou(batch["clean_polygons"], predicted_polygons)
        iou_gain = pred_iou - raw_iou
        self.log(f"{eval_name}/recon_l1", recon_l1, on_step=False, on_epoch=True, prog_bar=eval_name == "val", add_dataloader_idx=False)
        self.log(f"{eval_name}/polygon_iou", pred_iou, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{eval_name}/polygon_iou_gain", iou_gain, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)

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


@pyrallis.wrap(config_path=str(SCRIPT_DIR / "configs" / "affine_stn.yaml"))
def main(settings: AffineSTNExperimentConfig) -> None:
    repo_root = resolve_repo_relative(DEFAULT_REPO_ROOT, settings.repo_root)
    work_dir = resolve_repo_relative(repo_root, settings.training.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(settings.training.seed, workers=True)
    if settings.wandb.enabled:
        prepare_wandb_env(work_dir)

    data_module = AffineSTNDataModule(repo_root=repo_root, settings=settings)
    data_module.setup("fit")
    model = AffineSTNLightningModule(settings=settings, eval_names=data_module.eval_names)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=work_dir / "checkpoints",
            filename="best-{epoch:02d}",
            monitor="val_hard/polygon_iou",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        precision=settings.training.precision,
        max_epochs=settings.training.max_epochs,
        logger=build_logger(settings, work_dir),
        callbacks=callbacks,
        log_every_n_steps=settings.training.log_every_n_steps,
        gradient_clip_val=settings.training.gradient_clip_val,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=str(settings.training.resume_from_checkpoint) if settings.training.resume_from_checkpoint else None)


if __name__ == "__main__":
    main()
