from __future__ import annotations

import copy
import hashlib
import io
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_PRESETS = [
    {
        "name": "mild_tilt",
        "rotation_deg": 18,
        "perspective_strength": 0.08,
        "brightness": 0.95,
        "contrast": 1.05,
    },
    {
        "name": "upside_down",
        "rotation_deg": 180,
        "perspective_strength": 0.04,
        "brightness": 0.9,
        "contrast": 1.1,
    },
    {
        "name": "oblique_capture",
        "rotation_deg": 28,
        "perspective_strength": 0.18,
        "shadow_strength": 0.24,
        "gamma": 1.15,
    },
    {
        "name": "low_light_blur",
        "rotation_deg": -12,
        "brightness": 0.7,
        "contrast": 0.9,
        "gamma": 1.35,
        "motion_blur_kernel": 13,
        "motion_blur_angle": -18,
        "jpeg_quality": 42,
    },
    {
        "name": "harsh_glare",
        "rotation_deg": 9,
        "perspective_strength": 0.12,
        "glare_strength": 0.35,
        "shadow_strength": 0.15,
        "contrast": 0.88,
    },
]


def stable_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:8], 16)


def polygon_to_array(polygon: Sequence[float]) -> np.ndarray:
    return np.asarray(polygon, dtype=np.float32).reshape(-1, 2)


def bbox_from_polygon(points: np.ndarray) -> list[float]:
    xs = points[:, 0]
    ys = points[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (2, 3):
        hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        transformed = hom @ matrix.T
        return transformed.astype(np.float32)
    if matrix.shape == (3, 3):
        hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        transformed = hom @ matrix.T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        return transformed.astype(np.float32)
    raise ValueError(f"Unsupported transform shape: {matrix.shape}")


def transform_instances(instances: Sequence[dict], matrix: np.ndarray) -> list[dict]:
    transformed_instances: list[dict] = []
    for instance in instances:
        points = polygon_to_array(instance["polygon"])
        transformed = _transform_points(points, matrix)
        new_instance = copy.deepcopy(instance)
        new_instance["polygon"] = transformed.reshape(-1).tolist()
        new_instance["bbox"] = bbox_from_polygon(transformed)
        transformed_instances.append(new_instance)
    return transformed_instances


def rotate_image_and_instances(
    image: np.ndarray,
    instances: Sequence[dict],
    angle_deg: float,
    expand: bool = True,
    fill: tuple[int, int, int] = (18, 18, 18),
) -> tuple[np.ndarray, list[dict]]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    out_width, out_height = width, height
    if expand:
        cos_v = abs(matrix[0, 0])
        sin_v = abs(matrix[0, 1])
        out_width = int(np.ceil((height * sin_v) + (width * cos_v)))
        out_height = int(np.ceil((height * cos_v) + (width * sin_v)))
        matrix[0, 2] += (out_width / 2.0) - center[0]
        matrix[1, 2] += (out_height / 2.0) - center[1]
    warped = cv2.warpAffine(
        image,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return warped, transform_instances(instances, matrix)


def random_perspective_transform(
    image: np.ndarray,
    instances: Sequence[dict],
    strength: float,
    seed: int,
    fill: tuple[int, int, int] = (18, 18, 18),
) -> tuple[np.ndarray, list[dict]]:
    if strength <= 0:
        return image, copy.deepcopy(list(instances))
    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    max_x = width * strength
    max_y = height * strength
    jitter = rng.uniform(low=[-max_x, -max_y], high=[max_x, max_y], size=(4, 2)).astype(np.float32)
    target = source + jitter
    target[:, 0] = np.clip(target[:, 0], 0, max(width - 1, 1))
    target[:, 1] = np.clip(target[:, 1], 0, max(height - 1, 1))
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return warped, transform_instances(instances, matrix)


def _pil_from_array(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def _array_from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return image
    base = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(base, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_shadow(image: np.ndarray, strength: float, angle_deg: float) -> np.ndarray:
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(angle_deg)
    direction = np.cos(theta) * xs + np.sin(theta) * ys
    direction = (direction - direction.min()) / max(float(np.ptp(direction)), 1e-6)
    mask = 1.0 - (strength * direction)
    shaded = image.astype(np.float32) * mask[..., None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def apply_glare(image: np.ndarray, strength: float, seed: int) -> np.ndarray:
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)
    center_x = rng.uniform(0.25 * width, 0.75 * width)
    center_y = rng.uniform(0.2 * height, 0.7 * height)
    sigma_x = max(width * rng.uniform(0.08, 0.2), 1.0)
    sigma_y = max(height * rng.uniform(0.08, 0.2), 1.0)
    ys, xs = np.mgrid[0:height, 0:width]
    gaussian = np.exp(-(((xs - center_x) ** 2) / (2 * sigma_x**2) + ((ys - center_y) ** 2) / (2 * sigma_y**2)))
    lifted = image.astype(np.float32) + (255.0 * strength * gaussian[..., None])
    return np.clip(lifted, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
    if kernel_size <= 1:
        return image
    kernel_size = int(max(3, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2.0, kernel_size / 2.0), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return image
    kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)


def apply_jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 5, 95))
    buffer = io.BytesIO()
    pil_image = _pil_from_array(image)
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return _array_from_pil(compressed)


def apply_photometric_ops(image: np.ndarray, preset: dict, seed: int) -> np.ndarray:
    result = image
    pil_image = _pil_from_array(result)
    brightness = float(preset.get("brightness", 1.0))
    contrast = float(preset.get("contrast", 1.0))
    color = float(preset.get("color", 1.0))
    if brightness != 1.0:
        pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    if contrast != 1.0:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    if color != 1.0:
        pil_image = ImageEnhance.Color(pil_image).enhance(color)
    blur_radius = float(preset.get("gaussian_blur_radius", 0.0))
    if blur_radius > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    result = _array_from_pil(pil_image)
    result = apply_gamma(result, gamma=float(preset.get("gamma", 1.0)))
    result = apply_shadow(
        result,
        strength=float(preset.get("shadow_strength", 0.0)),
        angle_deg=float(preset.get("shadow_angle", 35.0)),
    )
    result = apply_glare(result, strength=float(preset.get("glare_strength", 0.0)), seed=seed + 17)
    result = apply_motion_blur(
        result,
        kernel_size=int(preset.get("motion_blur_kernel", 0)),
        angle_deg=float(preset.get("motion_blur_angle", 0.0)),
    )
    jpeg_quality = preset.get("jpeg_quality")
    if jpeg_quality is not None:
        result = apply_jpeg_roundtrip(result, quality=int(jpeg_quality))
    return result


def apply_preset_to_image_instances(
    image: np.ndarray,
    instances: Sequence[dict],
    preset: dict,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    augmented_image = image
    augmented_instances = copy.deepcopy(list(instances))
    rotation_deg = float(preset.get("rotation_deg", 0.0))
    if rotation_deg:
        augmented_image, augmented_instances = rotate_image_and_instances(
            augmented_image,
            augmented_instances,
            angle_deg=rotation_deg,
        )
    perspective_strength = float(preset.get("perspective_strength", 0.0))
    if perspective_strength:
        augmented_image, augmented_instances = random_perspective_transform(
            augmented_image,
            augmented_instances,
            strength=perspective_strength,
            seed=seed,
        )
    augmented_image = apply_photometric_ops(augmented_image, preset=preset, seed=seed)
    return augmented_image, augmented_instances


def choose_preset_for_key(key: str, seed: int, presets: Sequence[dict] | None = None) -> tuple[dict, int]:
    preset_list = list(presets or DEFAULT_PRESETS)
    if not preset_list:
        raise ValueError("At least one preset is required.")
    index = stable_seed(key, seed, "preset_index") % len(preset_list)
    return copy.deepcopy(preset_list[index]), index
