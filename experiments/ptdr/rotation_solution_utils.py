from __future__ import annotations

import io
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

try:
    from .detection_augmentations import apply_photometric_ops, choose_preset_for_key
except ImportError:
    from detection_augmentations import apply_photometric_ops, choose_preset_for_key


RIGHT_ANGLE_ROTATIONS = (0, 90, 180, 270)


def prepare_wandb_env(work_dir: Path) -> None:
    wandb_root = work_dir / "wandb"
    cache_dir = wandb_root / "cache"
    config_dir = wandb_root / "config"
    data_dir = wandb_root / "data"
    artifact_dir = wandb_root / "artifacts"
    tmp_dir = Path("/tmp") / "ptdr"
    short_tmp_dir = tmp_dir / work_dir.name
    for directory in (wandb_root, cache_dir, config_dir, data_dir, artifact_dir, tmp_dir, short_tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(data_dir))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(artifact_dir))
    os.environ.setdefault("TMPDIR", str(short_tmp_dir))
    tempfile.tempdir = os.environ["TMPDIR"]


def build_wandb_init_kwargs(run_prefix: str, project: str, entity: str | None, group: str | None, job_type: str, run_name: str | None) -> dict:
    env_key = f"_PTDR_{run_prefix.upper()}_WANDB_RUN_ID"
    run_id = os.environ.get(env_key)
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
        os.environ[env_key] = run_id
    os.environ.setdefault("WANDB_RUN_ID", run_id)
    return {
        "project": project,
        "entity": entity,
        "group": group,
        "job_type": job_type,
        "name": run_name,
        "id": run_id,
        "resume": "allow",
        "log_model": False,
    }


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def pil_from_rgb_array(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def decode_rgb_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def encode_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    pil_from_rgb_array(image).save(buffer, format="PNG")
    return buffer.getvalue()


def rotation_class_to_angle(class_index: int) -> int:
    return int(RIGHT_ANGLE_ROTATIONS[int(class_index)])


def angle_to_rotation_class(angle: int) -> int:
    normalized = int(angle) % 360
    if normalized not in RIGHT_ANGLE_ROTATIONS:
        raise ValueError(f"Unsupported right-angle rotation: {angle}")
    return RIGHT_ANGLE_ROTATIONS.index(normalized)


def correction_for_applied_rotation(angle: int) -> int:
    return (-int(angle)) % 360


def polygon_to_quad(polygon: Sequence[float]) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) == 4:
        return order_points_clockwise(points)
    rect = cv2.minAreaRect(points)
    return order_points_clockwise(cv2.boxPoints(rect))


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    top_left = np.argmin(points[:, 0] + points[:, 1])
    points = np.roll(points, -int(top_left), axis=0)
    if np.cross(points[1] - points[0], points[2] - points[0]) < 0:
        points[[1, 3]] = points[[3, 1]]
    return points.astype(np.float32)


def perspective_crop(image_rgb: np.ndarray, polygon: Sequence[float]) -> np.ndarray | None:
    quad = polygon_to_quad(polygon)
    return perspective_crop_from_quad(image_rgb, quad)


def perspective_crop_from_quad(image_rgb: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    quad = order_points_clockwise(np.asarray(quad, dtype=np.float32).reshape(4, 2))
    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_right = np.linalg.norm(quad[2] - quad[1])
    height_left = np.linalg.norm(quad[3] - quad[0])
    crop_width = max(1, int(round(max(width_top, width_bottom))))
    crop_height = max(1, int(round(max(height_left, height_right))))
    if crop_width < 2 or crop_height < 2:
        return None
    destination = np.array(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(
        image_rgb,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def jitter_quad_detector_style(
    polygon: Sequence[float],
    image_size: tuple[int, int],
    rng: random.Random,
    center_jitter_ratio: float = 0.06,
    scale_jitter_ratio: float = 0.12,
    corner_jitter_ratio: float = 0.05,
    min_bbox_intersection_pixels: float = 4.0,
    min_crop_side: float = 2.0,
) -> np.ndarray | None:
    image_height, image_width = int(image_size[0]), int(image_size[1])
    quad = polygon_to_quad(polygon).astype(np.float32)

    center = quad.mean(axis=0)
    centered = quad - center
    widths = [np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])]
    heights = [np.linalg.norm(quad[2] - quad[1]), np.linalg.norm(quad[3] - quad[0])]
    crop_width = max(widths)
    crop_height = max(heights)
    if crop_width < min_crop_side or crop_height < min_crop_side:
        return None

    translate_x = rng.uniform(-center_jitter_ratio, center_jitter_ratio) * crop_width
    translate_y = rng.uniform(-center_jitter_ratio, center_jitter_ratio) * crop_height
    scale_x = rng.uniform(1.0 - scale_jitter_ratio, 1.0 + scale_jitter_ratio)
    scale_y = rng.uniform(1.0 - scale_jitter_ratio, 1.0 + scale_jitter_ratio)
    jittered = centered.copy()
    jittered[:, 0] *= scale_x
    jittered[:, 1] *= scale_y
    jittered += np.array([center[0] + translate_x, center[1] + translate_y], dtype=np.float32)

    corner_scale = np.array([crop_width * corner_jitter_ratio, crop_height * corner_jitter_ratio], dtype=np.float32)
    corner_noise = np.asarray(
        [[rng.uniform(-corner_scale[0], corner_scale[0]), rng.uniform(-corner_scale[1], corner_scale[1])] for _ in range(4)],
        dtype=np.float32,
    )
    jittered += corner_noise
    jittered = order_points_clockwise(jittered)

    bbox_left = float(jittered[:, 0].min())
    bbox_top = float(jittered[:, 1].min())
    bbox_right = float(jittered[:, 0].max())
    bbox_bottom = float(jittered[:, 1].max())
    inside_left = max(0.0, bbox_left)
    inside_top = max(0.0, bbox_top)
    inside_right = min(float(image_width), bbox_right)
    inside_bottom = min(float(image_height), bbox_bottom)
    intersection_width = max(0.0, inside_right - inside_left)
    intersection_height = max(0.0, inside_bottom - inside_top)
    if intersection_width * intersection_height < min_bbox_intersection_pixels:
        return None
    return jittered.astype(np.float32)


def rotate_image_reflect(image_rgb: np.ndarray, angle: float) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    out_width = int(np.ceil((height * sin_v) + (width * cos_v)))
    out_height = int(np.ceil((height * cos_v) + (width * sin_v)))
    matrix[0, 2] += (out_width / 2.0) - center[0]
    matrix[1, 2] += (out_height / 2.0) - center[1]
    return cv2.warpAffine(
        image_rgb,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def random_perspective_reflect(image_rgb: np.ndarray, strength: float, seed: int) -> np.ndarray:
    if strength <= 0:
        return image_rgb
    height, width = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    max_x = width * strength
    max_y = height * strength
    jitter = rng.uniform(low=[-max_x, -max_y], high=[max_x, max_y], size=(4, 2)).astype(np.float32)
    target = source + jitter
    target[:, 0] = np.clip(target[:, 0], 0, max(width - 1, 1))
    target[:, 1] = np.clip(target[:, 1], 0, max(height - 1, 1))
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_reflect_preset(image_rgb: np.ndarray, preset: dict, seed: int) -> np.ndarray:
    result = image_rgb
    rotation_deg = float(preset.get("rotation_deg", 0.0))
    if rotation_deg:
        result = rotate_image_reflect(result, rotation_deg)
    perspective_strength = float(preset.get("perspective_strength", 0.0))
    if perspective_strength:
        result = random_perspective_reflect(result, perspective_strength, seed=seed)
    return apply_photometric_ops(result, preset=preset, seed=seed)


def resize_image_and_polygons(image_rgb: np.ndarray, polygons: Sequence[Sequence[float]], size: tuple[int, int]) -> tuple[np.ndarray, list[list[float]]]:
    target_height, target_width = int(size[0]), int(size[1])
    source_height, source_width = image_rgb.shape[:2]
    scale_x = target_width / max(float(source_width), 1.0)
    scale_y = target_height / max(float(source_height), 1.0)
    resized_image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    resized_polygons: list[list[float]] = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        resized_polygons.append(points.reshape(-1).tolist())
    return resized_image, resized_polygons


def affine_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_2x3, dtype=np.float32).reshape(2, 3)
    return np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def invert_affine_2x3(matrix_2x3: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(matrix_2x3, dtype=np.float32).reshape(2, 3)).astype(np.float32)


def _pix_to_norm(width: int, height: int) -> np.ndarray:
    width_denom = max(width - 1, 1)
    height_denom = max(height - 1, 1)
    return np.array(
        [
            [2.0 / width_denom, 0.0, -1.0],
            [0.0, 2.0 / height_denom, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _norm_to_pix(width: int, height: int) -> np.ndarray:
    return np.linalg.inv(_pix_to_norm(width, height)).astype(np.float32)


def normalized_affine_from_pixel_matrix(
    matrix_2x3: np.ndarray,
    input_size: tuple[int, int],
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if output_size is None:
        output_size = input_size
    input_height, input_width = int(input_size[0]), int(input_size[1])
    output_height, output_width = int(output_size[0]), int(output_size[1])
    theta = (
        _pix_to_norm(input_width, input_height)
        @ affine_homogeneous(matrix_2x3)
        @ _norm_to_pix(output_width, output_height)
    )
    return theta[:2].astype(np.float32)


def inverse_theta(theta: np.ndarray) -> np.ndarray:
    return np.linalg.inv(affine_homogeneous(theta)).astype(np.float32)


def transform_polygon_with_theta_inverse(
    polygon: Sequence[float],
    theta: np.ndarray,
    input_size: tuple[int, int],
    output_size: tuple[int, int] | None = None,
) -> list[float]:
    if output_size is None:
        output_size = input_size
    input_height, input_width = int(input_size[0]), int(input_size[1])
    output_height, output_width = int(output_size[0]), int(output_size[1])
    inv_theta = inverse_theta(theta)
    input_norm = _pix_to_norm(input_width, input_height)
    output_pix = _norm_to_pix(output_width, output_height)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    input_points_norm = hom @ input_norm.T
    output_points_norm = input_points_norm @ inv_theta.T
    output_points_pix = output_points_norm @ output_pix.T
    return output_points_pix[:, :2].reshape(-1).tolist()


def forward_affine_matrix(
    width: int,
    height: int,
    rotation_deg: float,
    scale: float,
    shear_deg: float,
    translate_x_px: float,
    translate_y_px: float,
) -> np.ndarray:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    theta = np.deg2rad(rotation_deg)
    shear = np.deg2rad(shear_deg)
    translate_to_origin = np.array([[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]], dtype=np.float32)
    rotate_scale = np.array(
        [
            [scale * np.cos(theta), -scale * np.sin(theta), 0.0],
            [scale * np.sin(theta), scale * np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shear_matrix = np.array([[1.0, np.tan(shear), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    translate_back = np.array(
        [[1.0, 0.0, center_x + translate_x_px], [0.0, 1.0, center_y + translate_y_px], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    forward = translate_back @ shear_matrix @ rotate_scale @ translate_to_origin
    return forward[:2].astype(np.float32)


def apply_forward_affine(image_rgb: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    return cv2.warpAffine(
        image_rgb,
        np.asarray(matrix_2x3, dtype=np.float32).reshape(2, 3),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def build_synthetic_affine_example(
    clean_image_rgb: np.ndarray,
    rng: random.Random,
    max_small_rotation_deg: float,
    right_angle_probability: float,
    min_scale: float,
    max_scale: float,
    max_translation_ratio: float,
    max_shear_deg: float,
    photometric_probability: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = clean_image_rgb.shape[:2]
    right_angle = rng.choice(RIGHT_ANGLE_ROTATIONS[1:]) if rng.random() < right_angle_probability else 0
    small_rotation = rng.uniform(-max_small_rotation_deg, max_small_rotation_deg)
    scale = rng.uniform(min_scale, max_scale)
    shear_deg = rng.uniform(-max_shear_deg, max_shear_deg)
    translate_x_px = rng.uniform(-max_translation_ratio, max_translation_ratio) * width
    translate_y_px = rng.uniform(-max_translation_ratio, max_translation_ratio) * height
    forward = forward_affine_matrix(
        width=width,
        height=height,
        rotation_deg=right_angle + small_rotation,
        scale=scale,
        shear_deg=shear_deg,
        translate_x_px=translate_x_px,
        translate_y_px=translate_y_px,
    )
    hard_image = apply_forward_affine(clean_image_rgb, forward)
    preset_name = None
    if rng.random() < photometric_probability:
        preset_names = ("mild_tilt", "oblique_capture", "low_light_blur", "harsh_glare")
        preset_name = rng.choice(preset_names)
        preset, _ = choose_preset_for_key(f"synthetic::{preset_name}", seed=rng.randint(0, 2**31 - 1))
        preset["rotation_deg"] = 0.0
        preset["perspective_strength"] = 0.0
        hard_image = apply_photometric_ops(hard_image, preset=preset, seed=rng.randint(0, 2**31 - 1))
    metadata = {
        "right_angle": int(right_angle),
        "small_rotation_deg": float(small_rotation),
        "scale": float(scale),
        "shear_deg": float(shear_deg),
        "translate_x_px": float(translate_x_px),
        "translate_y_px": float(translate_y_px),
        "photometric_preset": preset_name,
    }
    return hard_image, forward, metadata


class LmdbRecognitionDataset(Dataset):
    def __init__(self, lmdb_root: str | Path) -> None:
        self.lmdb_root = Path(lmdb_root)
        if not (self.lmdb_root / "data.mdb").exists():
            raise FileNotFoundError(f"Missing LMDB dataset at {self.lmdb_root}")
        self._env = None
        with lmdb.open(
            str(self.lmdb_root),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        ) as env:
            with env.begin(write=False) as txn:
                raw_count = txn.get(b"num-samples")
                if raw_count is None:
                    raise KeyError(f"Missing num-samples key in {self.lmdb_root}")
                self.length = int(raw_count.decode("ascii"))

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
        return self._env

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        real_index = int(index) + 1
        with self._get_env().begin(write=False) as txn:
            image_bytes = txn.get(f"image-{real_index:09d}".encode("ascii"))
            label_bytes = txn.get(f"label-{real_index:09d}".encode("ascii"))
        if image_bytes is None or label_bytes is None:
            raise KeyError(f"Missing image/label for index {real_index} in {self.lmdb_root}")
        return decode_rgb_bytes(image_bytes), label_bytes.decode("utf-8")
