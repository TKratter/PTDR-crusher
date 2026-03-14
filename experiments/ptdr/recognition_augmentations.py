from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from .detection_augmentations import DEFAULT_PRESETS, apply_preset_to_image_instances, choose_preset_for_key, stable_seed
except ImportError:
    from detection_augmentations import DEFAULT_PRESETS, apply_preset_to_image_instances, choose_preset_for_key, stable_seed


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def rgb_array_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def encode_rgb_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    rgb_array_to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


def apply_preset_to_image_array(image: np.ndarray, preset: dict, seed: int) -> np.ndarray:
    augmented, _ = apply_preset_to_image_instances(image=image, instances=[], preset=preset, seed=seed)
    return augmented


def apply_preset_to_pil_image(image: Image.Image, preset: dict, seed: int) -> Image.Image:
    return rgb_array_to_pil(apply_preset_to_image_array(pil_to_rgb_array(image), preset=preset, seed=seed))


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
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    top_left = np.argmin(points[:, 0] + points[:, 1])
    points = np.roll(points, -int(top_left), axis=0)
    if np.cross(points[1] - points[0], points[2] - points[0]) < 0:
        points[[1, 3]] = points[[3, 1]]
    return points.astype(np.float32)


def apply_detector_style_crop_jitter(
    image_rgb: np.ndarray,
    translate_ratio: float,
    scale_ratio: float,
    perspective_ratio: float,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height < 2 or width < 2:
        return image_rgb

    pad_x = max(4, int(round(width * (translate_ratio + scale_ratio + perspective_ratio + 0.15))))
    pad_y = max(4, int(round(height * (translate_ratio + scale_ratio + perspective_ratio + 0.15))))
    padded = cv2.copyMakeBorder(image_rgb, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101)

    base_quad = np.array(
        [
            [pad_x, pad_y],
            [pad_x + width - 1, pad_y],
            [pad_x + width - 1, pad_y + height - 1],
            [pad_x, pad_y + height - 1],
        ],
        dtype=np.float32,
    )
    center = base_quad.mean(axis=0)
    jittered = base_quad - center
    jittered[:, 0] *= random.uniform(1.0 - scale_ratio, 1.0 + scale_ratio)
    jittered[:, 1] *= random.uniform(1.0 - scale_ratio, 1.0 + scale_ratio)
    jittered += center + np.array(
        [
            random.uniform(-translate_ratio, translate_ratio) * width,
            random.uniform(-translate_ratio, translate_ratio) * height,
        ],
        dtype=np.float32,
    )
    corner_noise = np.asarray(
        [
            [
                random.uniform(-perspective_ratio, perspective_ratio) * width,
                random.uniform(-perspective_ratio, perspective_ratio) * height,
            ]
            for _ in range(4)
        ],
        dtype=np.float32,
    )
    jittered += corner_noise
    jittered = order_points_clockwise(jittered)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(jittered, destination)
    return cv2.warpPerspective(
        padded,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


class RandomHardRecognitionAugment:
    def __init__(
        self,
        probability: float = 0.65,
        right_angle_rotation_probability: float = 0.35,
        rotation_angles: Sequence[int] = (90, 180, 270),
        small_rotation_probability: float = 0.0,
        small_rotation_max_degrees: float = 0.0,
        detector_jitter_probability: float = 0.0,
        detector_jitter_translate_ratio: float = 0.08,
        detector_jitter_scale_ratio: float = 0.12,
        detector_jitter_perspective_ratio: float = 0.04,
        presets: Sequence[dict] | None = None,
    ) -> None:
        self.probability = float(probability)
        self.right_angle_rotation_probability = float(right_angle_rotation_probability)
        self.rotation_angles = tuple(int(angle) for angle in rotation_angles)
        self.small_rotation_probability = float(small_rotation_probability)
        self.small_rotation_max_degrees = float(small_rotation_max_degrees)
        self.detector_jitter_probability = float(detector_jitter_probability)
        self.detector_jitter_translate_ratio = float(detector_jitter_translate_ratio)
        self.detector_jitter_scale_ratio = float(detector_jitter_scale_ratio)
        self.detector_jitter_perspective_ratio = float(detector_jitter_perspective_ratio)
        self.presets = [dict(preset) for preset in (presets or DEFAULT_PRESETS)]

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image_rgb = np.asarray(image)
        if self.detector_jitter_probability > 0 and random.random() < self.detector_jitter_probability:
            image_rgb = apply_detector_style_crop_jitter(
                image_rgb,
                translate_ratio=self.detector_jitter_translate_ratio,
                scale_ratio=self.detector_jitter_scale_ratio,
                perspective_ratio=self.detector_jitter_perspective_ratio,
            )
        if self.rotation_angles and random.random() < self.right_angle_rotation_probability:
            angle = random.choice(self.rotation_angles)
            image_rgb = rotate_image_reflect(image_rgb, angle)
        if self.small_rotation_max_degrees > 0 and random.random() < self.small_rotation_probability:
            angle = random.uniform(-self.small_rotation_max_degrees, self.small_rotation_max_degrees)
            image_rgb = rotate_image_reflect(image_rgb, angle)
        image = rgb_array_to_pil(image_rgb)
        if self.presets and random.random() < self.probability:
            preset = random.choice(self.presets)
            image = apply_preset_to_pil_image(image, preset=preset, seed=random.randint(0, 2**31 - 1))
        return image


def build_variant_sample(sample: dict, image_bytes: bytes) -> dict:
    variant = dict(sample)
    variant["image_bytes"] = image_bytes
    return variant


def build_rotated_variant_samples(samples: Sequence[dict], angle: int, seed: int) -> list[dict]:
    rotated: list[dict] = []
    for index, sample in enumerate(samples):
        with Image.open(sample["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rotated_image = image.rotate(angle, expand=True, fillcolor=(18, 18, 18))
        rotated.append(
            build_variant_sample(
                sample,
                encode_rgb_png_bytes(np.asarray(rotated_image)),
            )
        )
    return rotated


def build_hard_variant_samples(samples: Sequence[dict], split_name: str, seed: int) -> tuple[list[dict], dict[str, int]]:
    hard_samples: list[dict] = []
    preset_counts: dict[str, int] = {}
    for index, sample in enumerate(samples):
        preset, preset_index = choose_preset_for_key(sample.get("repo_relative_path", f"{split_name}:{index}"), seed=seed)
        preset_counts[preset["name"]] = preset_counts.get(preset["name"], 0) + 1
        image_seed = stable_seed(seed, split_name, sample.get("repo_relative_path", ""), preset_index, index)
        with Image.open(sample["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            augmented = apply_preset_to_pil_image(image, preset=preset, seed=image_seed)
        hard_samples.append(build_variant_sample(sample, encode_rgb_png_bytes(np.asarray(augmented))))
    return hard_samples, dict(sorted(preset_counts.items()))


def recognition_eval_variant_root(repo_root: Path) -> Path:
    return repo_root / "experiments" / "ptdr" / "manifests" / "shared_recognition_eval_variants"
