#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmocr.apis.inferencers import TextDetInferencer
from mmocr.utils import poly_iou, polys2shapely
from torchvision import transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_schema import DEFAULT_REPO_ROOT
from text_normalization import canonicalize_arabic_persian_text


@dataclass
class GroundTruthWord:
    polygon: list[float]
    text: str
    normalized_text: str


@dataclass
class PredictedWord:
    polygon: list[float]
    score: float
    text: str
    normalized_text: str


@dataclass
class WordMatch:
    gt_index: int
    pred_index: int
    iou: float


@dataclass
class ImageEvaluation:
    image_path: str
    split: str
    ground_truth: list[GroundTruthWord]
    predictions: list[PredictedWord]
    matches: list[WordMatch]
    end_to_end_precision: float
    end_to_end_recall: float
    keyword_recall: float
    true_positives: int
    gt_count: int
    pred_count: int
    keyword_hits: int
    keyword_total: int
    gt_keywords: list[str]
    pred_keywords: list[str]


@dataclass
class DetectorBundle:
    inferencer: TextDetInferencer
    checkpoint_path: Path
    config: Config
    manifest_paths: dict[str, Path]


@dataclass
class RecognizerBundle:
    model: torch.nn.Module
    transform: T.Compose
    device: torch.device
    checkpoint_path: Path
    image_size: tuple[int, int]


def resolve_device(device: str | None = None) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_annotation_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_eval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = canonicalize_arabic_persian_text(
        normalized,
        normalize_unicode=False,
        digit_target="persian",
        normalize_digits=True,
        canonical_letter_target="persian",
        normalize_equivalent_letters=True,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def keyword_tokens(text: str) -> list[str]:
    return [token for token in normalize_eval_text(text).split(" ") if token]


def polygon_to_list(polygon: Any) -> list[float]:
    if hasattr(polygon, "detach"):
        polygon = polygon.detach().cpu().numpy()
    elif hasattr(polygon, "cpu"):
        polygon = polygon.cpu().numpy()
    polygon_array = np.asarray(polygon, dtype=np.float32).reshape(-1)
    return [float(value) for value in polygon_array.tolist()]


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]

    top_left_index = np.argmin(points[:, 0] + points[:, 1])
    points = np.roll(points, -int(top_left_index), axis=0)

    if np.cross(points[1] - points[0], points[2] - points[0]) < 0:
        points[[1, 3]] = points[[3, 1]]
    return points.astype(np.float32)


def polygon_to_quad(polygon: Sequence[float]) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        rect = cv2.minAreaRect(points)
        points = cv2.boxPoints(rect)
    return order_points_clockwise(points)


def perspective_crop(image_rgb: np.ndarray, polygon: Sequence[float]) -> np.ndarray | None:
    quad = polygon_to_quad(polygon)
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
    return cv2.warpPerspective(image_rgb, transform, (crop_width, crop_height), flags=cv2.INTER_CUBIC)


def extract_dataset_cfg(dataset_cfg: dict) -> dict:
    if dataset_cfg.get("type") == "ConcatDataset":
        datasets = dataset_cfg.get("datasets", [])
        if not datasets:
            raise KeyError("ConcatDataset does not contain any datasets.")
        return extract_dataset_cfg(datasets[0])
    return dataset_cfg


def resolve_manifest_path_from_cfg(cfg: Config, split: str, repo_root: Path) -> Path:
    dataloader_name = {"val": "val_dataloader", "test": "test_dataloader"}[split]
    dataloader_cfg = getattr(cfg, dataloader_name)
    dataset_cfg = extract_dataset_cfg(dataloader_cfg["dataset"])
    ann_file = Path(dataset_cfg["ann_file"])
    if ann_file.is_absolute():
        return ann_file

    data_root = dataset_cfg.get("data_root")
    if data_root:
        data_root_path = Path(data_root)
        if not data_root_path.is_absolute():
            data_root_path = (repo_root / data_root_path).resolve()
        return (data_root_path / ann_file).resolve()
    return (repo_root / ann_file).resolve()


def load_detection_records(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(data["data_list"])


def build_dbnet_inference_config(cfg_text: str) -> Config:
    cfg = Config.fromstring(cfg_text, ".py")
    if cfg.get("visualizer") is None:
        cfg.visualizer = {"type": "TextDetLocalVisualizer", "name": "visualizer", "vis_backends": []}
    else:
        cfg.visualizer["vis_backends"] = []
        cfg.visualizer.setdefault("name", "visualizer")
    return cfg


def load_dbnet_detector(
    checkpoint_path: str | Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    device: str | None = None,
) -> DetectorBundle:
    checkpoint_path = Path(checkpoint_path).resolve()
    repo_root = repo_root.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_text = checkpoint["meta"]["cfg"]
    cfg = build_dbnet_inference_config(cfg_text)
    init_default_scope("mmocr")
    inferencer = TextDetInferencer(model=cfg, weights=str(checkpoint_path), device=str(resolve_device(device)))
    manifest_paths = {
        split: resolve_manifest_path_from_cfg(cfg, split, repo_root)
        for split in ("val", "test")
    }
    return DetectorBundle(
        inferencer=inferencer,
        checkpoint_path=checkpoint_path,
        config=cfg,
        manifest_paths=manifest_paths,
    )


def load_parseq_recognizer(checkpoint_path: str | Path, device: str | None = None) -> RecognizerBundle:
    from strhub.models.utils import create_model

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = dict(checkpoint["hyper_parameters"])
    experiment_name = hparams.pop("name")
    hparams.pop("_target_", None)
    hparams.pop("_convert_", None)
    hparams["img_size"] = tuple(hparams["img_size"])

    model = create_model(experiment_name, pretrained=False, **hparams)
    model_state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    load_result = model.model.load_state_dict(model_state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Failed to restore the PARSeq checkpoint cleanly. "
            f"Missing keys: {sorted(load_result.missing_keys)} | "
            f"Unexpected keys: {sorted(load_result.unexpected_keys)}"
        )

    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)
    transform = T.Compose(
        [
            T.Resize(hparams["img_size"], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    return RecognizerBundle(
        model=model,
        transform=transform,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        image_size=hparams["img_size"],
    )


def collect_detector_predictions(
    detector: DetectorBundle,
    image_path: str | Path,
    score_thr: float,
) -> tuple[list[list[float]], list[float]]:
    init_default_scope("mmocr")
    inference = detector.inferencer(
        str(image_path),
        return_datasamples=True,
        progress_bar=False,
        draw_pred=False,
        out_dir="",
        save_vis=False,
        save_pred=False,
    )
    data_sample = inference["predictions"][0]
    pred_instances = data_sample.pred_instances
    polygons = [polygon_to_list(polygon) for polygon in pred_instances.polygons]
    scores = pred_instances.scores
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores, dtype=np.float32)

    filtered_polygons: list[list[float]] = []
    filtered_scores: list[float] = []
    for polygon, score in zip(polygons, scores.tolist()):
        if float(score) < score_thr:
            continue
        filtered_polygons.append(polygon)
        filtered_scores.append(float(score))
    return filtered_polygons, filtered_scores


def recognize_crops(
    recognizer: RecognizerBundle,
    image_rgb: np.ndarray,
    polygons: Sequence[Sequence[float]],
    batch_size: int = 32,
) -> tuple[list[str], list[np.ndarray | None]]:
    crops = [perspective_crop(image_rgb, polygon) for polygon in polygons]
    predictions = [""] * len(crops)

    valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
    for start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[start:start + batch_size]
        batch_tensors = torch.stack(
            [recognizer.transform(Image.fromarray(crops[index])) for index in batch_indices]
        ).to(recognizer.device)
        with torch.inference_mode():
            logits = recognizer.model(batch_tensors)
        decoded, _ = recognizer.model.tokenizer.decode(logits.softmax(-1))
        for index, prediction in zip(batch_indices, decoded):
            predictions[index] = recognizer.model.charset_adapter(prediction)
    return predictions, crops


def build_ground_truth_words(record: dict) -> list[GroundTruthWord]:
    words = []
    for instance in record.get("instances", []):
        text = instance.get("text", "")
        words.append(
            GroundTruthWord(
                polygon=polygon_to_list(instance["polygon"]),
                text=text,
                normalized_text=normalize_eval_text(text),
            )
        )
    return words


def build_predicted_words(
    polygons: Sequence[Sequence[float]],
    scores: Sequence[float],
    texts: Sequence[str],
) -> list[PredictedWord]:
    predictions = []
    for polygon, score, text in zip(polygons, scores, texts):
        predictions.append(
            PredictedWord(
                polygon=polygon_to_list(polygon),
                score=float(score),
                text=text,
                normalized_text=normalize_eval_text(text),
            )
        )
    return predictions


def match_predictions(
    ground_truth: Sequence[GroundTruthWord],
    predictions: Sequence[PredictedWord],
    iou_thr: float,
) -> list[WordMatch]:
    if not ground_truth or not predictions:
        return []

    gt_polygons = polys2shapely([word.polygon for word in ground_truth])
    pred_polygons = polys2shapely([word.polygon for word in predictions])
    candidates: list[tuple[float, float, int, int]] = []
    for gt_index, gt_word in enumerate(ground_truth):
        if not gt_word.normalized_text:
            continue
        for pred_index, prediction in enumerate(predictions):
            if gt_word.normalized_text != prediction.normalized_text:
                continue
            iou = float(poly_iou(gt_polygons[gt_index], pred_polygons[pred_index]))
            if iou >= iou_thr:
                candidates.append((iou, prediction.score, gt_index, pred_index))

    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[WordMatch] = []
    for iou, _score, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append(WordMatch(gt_index=gt_index, pred_index=pred_index, iou=iou))
    return matches


def compute_keyword_recall(
    ground_truth: Sequence[GroundTruthWord],
    predictions: Sequence[PredictedWord],
) -> tuple[float, int, int, list[str], list[str]]:
    gt_keywords = sorted({token for word in ground_truth for token in keyword_tokens(word.text)})
    pred_keywords = sorted({token for word in predictions for token in keyword_tokens(word.text)})
    if not gt_keywords:
        return 0.0, 0, 0, gt_keywords, pred_keywords

    pred_keyword_set = set(pred_keywords)
    hits = sum(1 for token in gt_keywords if token in pred_keyword_set)
    return hits / len(gt_keywords), hits, len(gt_keywords), gt_keywords, pred_keywords


def run_end_to_end_inference(
    record: dict,
    detector: DetectorBundle,
    recognizer: RecognizerBundle,
    repo_root: Path = DEFAULT_REPO_ROOT,
    split: str = "test",
    det_score_thr: float = 0.3,
    match_iou_thr: float = 0.5,
    recognition_batch_size: int = 32,
) -> tuple[ImageEvaluation, np.ndarray, list[np.ndarray | None]]:
    repo_root = repo_root.resolve()
    image_path = (repo_root / record["img_path"]).resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    polygons, scores = collect_detector_predictions(detector, image_path, det_score_thr)
    recognized_texts, crops = recognize_crops(
        recognizer,
        image_rgb=image_rgb,
        polygons=polygons,
        batch_size=recognition_batch_size,
    )

    ground_truth = build_ground_truth_words(record)
    predictions = build_predicted_words(polygons, scores, recognized_texts)
    matches = match_predictions(ground_truth, predictions, iou_thr=match_iou_thr)

    true_positives = len(matches)
    pred_count = len(predictions)
    gt_count = len(ground_truth)
    precision = true_positives / pred_count if pred_count else 0.0
    recall = true_positives / gt_count if gt_count else 0.0
    keyword_recall, keyword_hits, keyword_total, gt_keywords, pred_keywords = compute_keyword_recall(
        ground_truth,
        predictions,
    )
    result = ImageEvaluation(
        image_path=record["img_path"],
        split=split,
        ground_truth=ground_truth,
        predictions=predictions,
        matches=matches,
        end_to_end_precision=precision,
        end_to_end_recall=recall,
        keyword_recall=keyword_recall,
        true_positives=true_positives,
        gt_count=gt_count,
        pred_count=pred_count,
        keyword_hits=keyword_hits,
        keyword_total=keyword_total,
        gt_keywords=gt_keywords,
        pred_keywords=pred_keywords,
    )
    return result, image_rgb, crops


def evaluate_records(
    records: Sequence[dict],
    detector: DetectorBundle,
    recognizer: RecognizerBundle,
    repo_root: Path = DEFAULT_REPO_ROOT,
    split: str = "test",
    det_score_thr: float = 0.3,
    match_iou_thr: float = 0.5,
    recognition_batch_size: int = 32,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[ImageEvaluation]]:
    selected_records = list(records[:limit] if limit is not None else records)
    results: list[ImageEvaluation] = []
    matched_total = 0
    gt_total = 0
    pred_total = 0
    keyword_hits = 0
    keyword_total = 0

    iterator: Iterable[dict]
    try:
        from tqdm import tqdm

        iterator = tqdm(selected_records, desc=f"Evaluating {split}", unit="image")
    except Exception:
        iterator = selected_records

    for record in iterator:
        result, _image_rgb, _crops = run_end_to_end_inference(
            record=record,
            detector=detector,
            recognizer=recognizer,
            repo_root=repo_root,
            split=split,
            det_score_thr=det_score_thr,
            match_iou_thr=match_iou_thr,
            recognition_batch_size=recognition_batch_size,
        )
        results.append(result)
        matched_total += result.true_positives
        gt_total += result.gt_count
        pred_total += result.pred_count
        keyword_hits += result.keyword_hits
        keyword_total += result.keyword_total

    summary = {
        "split": split,
        "images_evaluated": len(results),
        "detector_checkpoint": str(detector.checkpoint_path),
        "recognizer_checkpoint": str(recognizer.checkpoint_path),
        "match_iou_thr": match_iou_thr,
        "det_score_thr": det_score_thr,
        "recognition_batch_size": recognition_batch_size,
        "true_positives": matched_total,
        "ground_truth_instances": gt_total,
        "predicted_instances": pred_total,
        "end_to_end_precision": matched_total / pred_total if pred_total else 0.0,
        "end_to_end_recall": matched_total / gt_total if gt_total else 0.0,
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
        "keyword_recall": keyword_hits / keyword_total if keyword_total else 0.0,
    }
    return summary, results


def result_to_dict(result: ImageEvaluation) -> dict[str, Any]:
    return asdict(result)


def save_evaluation_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: Sequence[ImageEvaluation],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_image_path = output_dir / "per_image.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with per_image_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")


def draw_polygon(drawer: ImageDraw.ImageDraw, polygon: Sequence[float], color: tuple[int, int, int], width: int) -> None:
    points = [(float(polygon[idx]), float(polygon[idx + 1])) for idx in range(0, len(polygon), 2)]
    if len(points) < 2:
        return
    drawer.line(points + [points[0]], fill=color, width=width)


def polygon_anchor(polygon: Sequence[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return float(min(xs)), float(min(ys))


def render_detection_overlay(
    image_rgb: np.ndarray,
    ground_truth: Sequence[GroundTruthWord],
    predictions: Sequence[PredictedWord],
    matches: Sequence[WordMatch] | None = None,
) -> np.ndarray:
    canvas = Image.fromarray(image_rgb.copy())
    drawer = ImageDraw.Draw(canvas)
    font = load_annotation_font(14)
    line_width = max(2, int(round(max(canvas.size) / 500)))

    matched_gt = {match.gt_index for match in matches or []}
    matched_pred = {match.pred_index for match in matches or []}

    for gt_index, word in enumerate(ground_truth):
        color = (34, 139, 34) if gt_index in matched_gt else (220, 20, 60)
        draw_polygon(drawer, word.polygon, color=color, width=line_width)
        x, y = polygon_anchor(word.polygon)
        drawer.text((x, max(0.0, y - 16.0)), f"GT: {word.text}", fill=color, font=font)

    for pred_index, prediction in enumerate(predictions):
        color = (30, 144, 255) if pred_index in matched_pred else (255, 140, 0)
        draw_polygon(drawer, prediction.polygon, color=color, width=line_width)
        x, y = polygon_anchor(prediction.polygon)
        label = f"P: {prediction.text or '<empty>'} ({prediction.score:.2f})"
        drawer.text((x, y + 2.0), label, fill=color, font=font)
    return np.array(canvas)


def render_recognition_gallery(
    crops: Sequence[np.ndarray | None],
    predictions: Sequence[PredictedWord],
    columns: int = 4,
) -> np.ndarray:
    valid_items = [(crop, prediction) for crop, prediction in zip(crops, predictions) if crop is not None]
    if not valid_items:
        canvas = Image.new("RGB", (640, 120), "white")
        drawer = ImageDraw.Draw(canvas)
        drawer.text((16, 48), "No predicted text crops above the current score threshold.", fill="black")
        return np.array(canvas)

    font = load_annotation_font(14)
    thumb_width = 220
    thumb_height = 96
    padding = 12
    caption_height = 40
    columns = max(1, columns)
    rows = int(math.ceil(len(valid_items) / columns))
    canvas = Image.new(
        "RGB",
        (
            columns * (thumb_width + padding) + padding,
            rows * (thumb_height + caption_height + padding) + padding,
        ),
        "white",
    )
    drawer = ImageDraw.Draw(canvas)

    for index, (crop, prediction) in enumerate(valid_items):
        row = index // columns
        col = index % columns
        x = padding + col * (thumb_width + padding)
        y = padding + row * (thumb_height + caption_height + padding)
        thumb = Image.fromarray(crop).resize((thumb_width, thumb_height))
        canvas.paste(thumb, (x, y))
        caption = f"{prediction.text or '<empty>'} | score={prediction.score:.2f}"
        drawer.text((x, y + thumb_height + 6), caption, fill="black", font=font)
    return np.array(canvas)


def save_visualizations(
    output_dir: Path,
    image_result: ImageEvaluation,
    image_rgb: np.ndarray,
    crops: Sequence[np.ndarray | None],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_result.image_path).stem
    overlay_path = output_dir / f"{stem}_overlay.png"
    gallery_path = output_dir / f"{stem}_crops.png"
    overlay = render_detection_overlay(
        image_rgb=image_rgb,
        ground_truth=image_result.ground_truth,
        predictions=image_result.predictions,
        matches=image_result.matches,
    )
    gallery = render_recognition_gallery(crops=crops, predictions=image_result.predictions)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(gallery).save(gallery_path)
    return {
        "overlay": str(overlay_path),
        "crops": str(gallery_path),
    }
