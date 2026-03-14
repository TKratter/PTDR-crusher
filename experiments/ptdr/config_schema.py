#!/usr/bin/env python3

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_relative(repo_root: Path, value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    return value if value.is_absolute() else (repo_root / value).resolve()


def serialize_config(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: serialize_config(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialize_config(item) for item in value]
    if isinstance(value, list):
        return [serialize_config(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_config(item) for key, item in value.items()}
    return value


def default_scene_domains() -> list[str]:
    return ["indoor_text", "outdoor_text"]


@dataclass
class WandbConfig:
    enabled: bool = True
    mode: str = "online"
    project: str = "ptdr-ocr"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class MMOCRConfig:
    base_config: str = "dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
    load_from: Optional[str] = None


@dataclass
class ExternalDatasetRootsConfig:
    icdar2019_mlt: Optional[Path] = None
    evarest_detection: Optional[Path] = None
    evarest_recognition: Optional[Path] = None
    totaltext: Optional[Path] = None
    ctw1500: Optional[Path] = None
    textocr: Optional[Path] = None
    ir_lpr_detection: Optional[Path] = None
    ir_lpr_recognition: Optional[Path] = None


@dataclass
class TrainMixConfig:
    ptdr: float = 1.0
    icdar2019_mlt: float = 0.0
    evarest: float = 0.0
    totaltext: float = 0.0
    ctw1500: float = 0.0
    textocr: float = 0.0
    ir_lpr: float = 0.0
    min_ptdr_fraction: float = 0.3


@dataclass
class MLTConfig:
    scripts: List[str] = field(default_factory=lambda: ["Arabic", "Latin"])


@dataclass
class DBNetPPTrainingConfig:
    work_dir: Path = Path("work_dirs/dbnetpp_ptdr_scene")
    seed: int = 42
    batch_size: int = 8
    eval_batch_size: int = 1
    num_workers: int = 10
    pin_memory: bool = True
    prefetch_factor: int = 3
    cudnn_benchmark: bool = True
    max_epochs: int = 100
    val_interval: int = 5
    amp: bool = True
    logger_interval: int = 50
    checkpoint_interval: int = 1
    max_keep_ckpts: int = 3
    resume: bool = False
    launcher: Optional[str] = None
    enable_auto_scale_lr: bool = True
    optimizer_lr: Optional[float] = None
    train_max_image_side: Optional[int] = 4096
    train_max_pixels: Optional[int] = 16000000


@dataclass
class DBNetPTHardAugTrainConfig:
    enabled: bool = True
    probability: float = 0.75
    schedule_epochs: List[int] = field(default_factory=list)
    schedule_probabilities: List[float] = field(default_factory=list)


@dataclass
class DBNetPTHardEvalConfig:
    enabled: bool = True
    seed: int = 42
    rotation_angles: List[int] = field(default_factory=lambda: [90, 180, 270])


@dataclass
class DBNetPTHardAugConfig:
    train: DBNetPTHardAugTrainConfig = field(default_factory=DBNetPTHardAugTrainConfig)
    eval: DBNetPTHardEvalConfig = field(default_factory=DBNetPTHardEvalConfig)


@dataclass
class DBNetPPExperimentConfig:
    experiment_name: str = "ptdr-dbnetpp-scene"
    repo_root: Path = Path(".")
    dataset_root: Path = Path("dataset/detection")
    include_domains: List[str] = field(default_factory=default_scene_domains)
    manifests_dir: Path = Path("experiments/ptdr/manifests/dbnetpp_scene")
    val_ratio: float = 0.1
    split_seed: int = 42
    fail_on_bad_annotations: bool = False
    external_datasets: ExternalDatasetRootsConfig = field(default_factory=ExternalDatasetRootsConfig)
    train_mix: TrainMixConfig = field(default_factory=TrainMixConfig)
    mlt: MLTConfig = field(default_factory=MLTConfig)
    mmocr: MMOCRConfig = field(default_factory=MMOCRConfig)
    training: DBNetPPTrainingConfig = field(default_factory=DBNetPPTrainingConfig)
    hard_aug: DBNetPTHardAugConfig = field(default_factory=DBNetPTHardAugConfig)
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            group="dbnetpp",
            tags=["ptdr", "dbnetpp", "detection"],
        )
    )


@dataclass
class PARSeqModelConfig:
    name: str = "parseq"
    pretrained: bool = True
    init_from_checkpoint: Optional[Path] = None
    use_eval_charset_for_train_charset: bool = False
    img_size: List[int] = field(default_factory=lambda: [32, 128])
    max_label_length: Optional[int] = None


@dataclass
class PARSeqTrainingConfig:
    work_dir: Path = Path("work_dirs/parseq_ptdr_scene")
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 8
    eval_num_workers: int = 0
    max_epochs: int = 20
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    log_every_n_steps: int = 25
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    lr: float = 0.001
    warmup_pct: float = 0.075
    weight_decay: float = 0.0
    cudnn_benchmark: bool = True
    matmul_precision: str = "high"
    resume_from_checkpoint: Optional[Path] = None
    extra_val_interval: int = 1
    enable_val_sample_logging: bool = True


@dataclass
class PARSeqDataConfig:
    augment: bool = True
    remove_whitespace: bool = False
    normalize_unicode: bool = False
    min_image_dim: int = 0
    rotation: int = 0
    collate_fn: Optional[str] = None


@dataclass
class PARSeqHardAugTrainConfig:
    enabled: bool = True
    probability: float = 0.65
    right_angle_rotation_probability: float = 0.35
    rotation_angles: List[int] = field(default_factory=lambda: [90, 180, 270])
    small_rotation_probability: float = 0.0
    small_rotation_max_degrees: float = 0.0
    detector_jitter_probability: float = 0.0
    detector_jitter_translate_ratio: float = 0.08
    detector_jitter_scale_ratio: float = 0.12
    detector_jitter_perspective_ratio: float = 0.04


@dataclass
class PARSeqHardEvalConfig:
    enabled: bool = True
    seed: int = 42
    rotation_angles: List[int] = field(default_factory=lambda: [90, 180, 270])


@dataclass
class PARSeqHardAugConfig:
    train: PARSeqHardAugTrainConfig = field(default_factory=PARSeqHardAugTrainConfig)
    eval: PARSeqHardEvalConfig = field(default_factory=PARSeqHardEvalConfig)


@dataclass
class PARSeqCharsetPolicyConfig:
    base_charset_path: Optional[Path] = None
    allow_arabic_extras_only: bool = False
    normalize_arabic_indic_digits: bool = True
    arabic_indic_digit_target: str = "persian"
    normalize_equivalent_arabic_persian_letters: bool = True
    arabic_persian_letter_target: str = "persian"
    drop_unsupported_labels: bool = False


@dataclass
class PARSeqExperimentConfig:
    experiment_name: str = "ptdr-parseq-scene"
    repo_root: Path = Path(".")
    dataset_root: Path = Path("dataset/recognition")
    include_domains: List[str] = field(default_factory=default_scene_domains)
    manifests_dir: Path = Path("experiments/ptdr/manifests/parseq_scene")
    val_ratio: float = 0.1
    split_seed: int = 42
    fail_on_bad_annotations: bool = False
    include_ptdr_train_in_train_split: bool = True
    exclude_extra_train_label_overlap_from_eval: bool = False
    lmdb_map_size_gb: float = 8.0
    ptdr_synth_root: Optional[Path] = None
    extra_train_dataset_roots: List[Path] = field(default_factory=list)
    external_datasets: ExternalDatasetRootsConfig = field(default_factory=ExternalDatasetRootsConfig)
    train_mix: TrainMixConfig = field(default_factory=TrainMixConfig)
    mlt: MLTConfig = field(default_factory=MLTConfig)
    model: PARSeqModelConfig = field(default_factory=PARSeqModelConfig)
    training: PARSeqTrainingConfig = field(default_factory=PARSeqTrainingConfig)
    data: PARSeqDataConfig = field(default_factory=PARSeqDataConfig)
    hard_aug: PARSeqHardAugConfig = field(default_factory=PARSeqHardAugConfig)
    charset_policy: PARSeqCharsetPolicyConfig = field(default_factory=PARSeqCharsetPolicyConfig)
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            group="parseq",
            tags=["ptdr", "parseq", "recognition"],
        )
    )
