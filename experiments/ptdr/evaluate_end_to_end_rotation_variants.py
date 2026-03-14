#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pyrallis
import torch
from PIL import Image

from config_schema import DEFAULT_REPO_ROOT
from end_to_end_utils import (
    GroundTruthWord,
    ImageEvaluation,
    PredictedWord,
    WordMatch,
    build_ground_truth_words,
    build_predicted_words,
    compute_keyword_recall,
    load_dbnet_detector,
    load_detection_records,
    load_parseq_recognizer,
    match_predictions,
    perspective_crop,
    resolve_device,
    save_evaluation_outputs,
)
from rotation_solution_utils import rotate_image_reflect, rotation_class_to_angle
from train_affine_stn import AffineSTNBackbone
from train_crop_rotation_classifier import (
    AspectPreservingSquareTransform,
    CropRotationModelConfig,
    create_backbone,
)


DEFAULT_DBNET_CKPT = DEFAULT_REPO_ROOT / "work_dirs" / "dbnet_multidata_all_r18_hard_faststable" / "best_icdar_hmean_epoch_80.pth"
DEFAULT_PARSEQ_CKPT = DEFAULT_REPO_ROOT / "work_dirs" / "parseq_multidata_all_norm_v2_hard" / "checkpoints" / "epoch=epoch=017-step=step=0003636.ckpt"
DEFAULT_CROP_ROT_CKPT = DEFAULT_REPO_ROOT / "work_dirs" / "crop_rotation_classifier" / "checkpoints" / "best-03.ckpt"
DEFAULT_STN_CKPT = DEFAULT_REPO_ROOT / "work_dirs" / "affine_stn" / "checkpoints" / "best-08.ckpt"


@dataclass
class CropRotationBundle:
    model: torch.nn.Module
    transform: AspectPreservingSquareTransform
    device: torch.device
    checkpoint_path: Path


@dataclass
class STNBundle:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    input_size: tuple[int, int]


def extract_prefixed_state_dict(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_crop_rotation_bundle(checkpoint_path: str | Path, device: str | None = None) -> CropRotationBundle:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    settings_payload = checkpoint["hyper_parameters"]
    model_cfg = CropRotationModelConfig(**settings_payload["model"])
    model = create_backbone(model_cfg)
    model.load_state_dict(extract_prefixed_state_dict(checkpoint["state_dict"], "model."), strict=True)
    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)
    transform = AspectPreservingSquareTransform(tuple(int(v) for v in model_cfg.image_size))
    return CropRotationBundle(
        model=model,
        transform=transform,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
    )


def load_stn_bundle(checkpoint_path: str | Path, device: str | None = None) -> STNBundle:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    settings_payload = checkpoint["hyper_parameters"]
    model_cfg = settings_payload["model"]
    model = AffineSTNBackbone(hidden_dim=int(model_cfg["hidden_dim"]))
    model.load_state_dict(extract_prefixed_state_dict(checkpoint["state_dict"], "model."), strict=True)
    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)
    input_size = tuple(int(v) for v in model_cfg["input_size"])
    return STNBundle(
        model=model,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        input_size=input_size,
    )


def infer_detector_polygons(detector, image_source: str | Path, score_thr: float) -> tuple[list[list[float]], list[float]]:
    inference = detector.inferencer(
        str(image_source),
        return_datasamples=True,
        progress_bar=False,
        draw_pred=False,
        out_dir="",
        save_vis=False,
        save_pred=False,
    )
    data_sample = inference["predictions"][0]
    pred_instances = data_sample.pred_instances
    polygons = []
    scores = pred_instances.scores
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores, dtype=np.float32)
    for polygon, score in zip(pred_instances.polygons, scores.tolist()):
        if float(score) < score_thr:
            continue
        polygon = np.asarray(polygon, dtype=np.float32).reshape(-1)
        polygons.append([float(value) for value in polygon.tolist()])
    return polygons, [float(score) for score in scores.tolist() if float(score) >= score_thr]


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


def transform_polygon_with_theta_forward(
    polygon: Sequence[float],
    theta: np.ndarray,
    output_size: tuple[int, int],
    input_size: tuple[int, int],
) -> list[float]:
    output_height, output_width = int(output_size[0]), int(output_size[1])
    input_height, input_width = int(input_size[0]), int(input_size[1])
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    output_points_norm = hom @ _pix_to_norm(output_width, output_height).T
    input_points_norm = output_points_norm @ np.asarray(theta, dtype=np.float32).reshape(2, 3).T
    input_points_norm_h = np.concatenate(
        [input_points_norm, np.ones((len(input_points_norm), 1), dtype=np.float32)],
        axis=1,
    )
    input_points_pix = input_points_norm_h @ _norm_to_pix(input_width, input_height).T
    return input_points_pix[:, :2].reshape(-1).tolist()


def apply_stn_and_map_predictions(
    stn: STNBundle,
    detector,
    image_rgb: np.ndarray,
    score_thr: float,
) -> tuple[list[list[float]], list[float]]:
    original_height, original_width = image_rgb.shape[:2]
    input_height, input_width = stn.input_size
    resized_rgb = cv2.resize(image_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(stn.device)
    with torch.inference_mode():
        normalized, theta = stn.model(tensor)
    normalized_rgb = (
        normalized[0]
        .detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    theta_np = theta[0].detach().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
        Image.fromarray(normalized_rgb).save(handle.name)
        polygons, scores = infer_detector_polygons(detector, handle.name, score_thr)

    mapped_polygons = []
    scale_x = original_width / max(float(input_width), 1.0)
    scale_y = original_height / max(float(input_height), 1.0)
    for polygon in polygons:
        polygon_in_resized_input = transform_polygon_with_theta_forward(
            polygon=polygon,
            theta=theta_np,
            output_size=stn.input_size,
            input_size=stn.input_size,
        )
        points = np.asarray(polygon_in_resized_input, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        mapped_polygons.append(points.reshape(-1).tolist())
    return mapped_polygons, scores


def correct_crops_with_classifier(
    crop_bundle: CropRotationBundle,
    crops: Sequence[np.ndarray | None],
    batch_size: int = 64,
) -> tuple[list[np.ndarray | None], list[int]]:
    corrected = list(crops)
    predicted_angles = [0] * len(crops)
    valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
    for start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[start:start + batch_size]
        batch = torch.stack(
            [crop_bundle.transform(Image.fromarray(crops[index], mode="RGB")) for index in batch_indices]
        ).to(crop_bundle.device)
        with torch.inference_mode():
            logits = crop_bundle.model(batch)
        classes = logits.argmax(dim=-1).detach().cpu().tolist()
        for index, class_index in zip(batch_indices, classes):
            angle = rotation_class_to_angle(int(class_index))
            predicted_angles[index] = int(angle)
            crop = crops[index]
            corrected[index] = rotate_image_reflect(crop, angle) if angle else crop
    return corrected, predicted_angles


def recognize_crop_arrays(
    recognizer,
    crops: Sequence[np.ndarray | None],
    batch_size: int = 32,
) -> list[str]:
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
    return predictions


def run_variant_inference(
    record: dict,
    detector,
    recognizer,
    repo_root: Path,
    split: str,
    det_score_thr: float,
    match_iou_thr: float,
    recognition_batch_size: int,
    crop_bundle: CropRotationBundle | None = None,
    stn_bundle: STNBundle | None = None,
) -> ImageEvaluation:
    image_path = (repo_root / record["img_path"]).resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if stn_bundle is not None:
        polygons, scores = apply_stn_and_map_predictions(stn_bundle, detector, image_rgb=image_rgb, score_thr=det_score_thr)
    else:
        polygons, scores = infer_detector_polygons(detector, image_path, det_score_thr)

    crops = [perspective_crop(image_rgb, polygon) for polygon in polygons]
    if crop_bundle is not None:
        crops, _angles = correct_crops_with_classifier(crop_bundle, crops)
    recognized_texts = recognize_crop_arrays(recognizer, crops=crops, batch_size=recognition_batch_size)

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
    return ImageEvaluation(
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


def evaluate_variant_on_records(
    variant_name: str,
    split: str,
    records: Sequence[dict],
    detector,
    recognizer,
    repo_root: Path,
    det_score_thr: float,
    match_iou_thr: float,
    recognition_batch_size: int,
    crop_bundle: CropRotationBundle | None = None,
    stn_bundle: STNBundle | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[ImageEvaluation]]:
    selected_records = list(records[:limit] if limit is not None else records)
    results: list[ImageEvaluation] = []
    matched_total = 0
    gt_total = 0
    pred_total = 0
    keyword_hits = 0
    keyword_total = 0

    try:
        from tqdm import tqdm

        iterator = tqdm(selected_records, desc=f"{variant_name}:{split}", unit="image")
    except Exception:
        iterator = selected_records

    for record in iterator:
        result = run_variant_inference(
            record=record,
            detector=detector,
            recognizer=recognizer,
            repo_root=repo_root,
            split=split,
            det_score_thr=det_score_thr,
            match_iou_thr=match_iou_thr,
            recognition_batch_size=recognition_batch_size,
            crop_bundle=crop_bundle,
            stn_bundle=stn_bundle,
        )
        results.append(result)
        matched_total += result.true_positives
        gt_total += result.gt_count
        pred_total += result.pred_count
        keyword_hits += result.keyword_hits
        keyword_total += result.keyword_total

    summary = {
        "variant": variant_name,
        "split": split,
        "images_evaluated": len(results),
        "detector_checkpoint": str(detector.checkpoint_path),
        "recognizer_checkpoint": str(recognizer.checkpoint_path),
        "crop_rotation_checkpoint": str(crop_bundle.checkpoint_path) if crop_bundle is not None else None,
        "stn_checkpoint": str(stn_bundle.checkpoint_path) if stn_bundle is not None else None,
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


def default_val_manifest_paths(detector, repo_root: Path) -> dict[str, Path]:
    shared_eval_root = repo_root / "experiments" / "ptdr" / "manifests" / "shared_textdet_eval_variants"
    return {
        "val": detector.manifest_paths["val"],
        "val_rot90": shared_eval_root / "textdet_val_rot90.json",
        "val_rot180": shared_eval_root / "textdet_val_rot180.json",
        "val_rot270": shared_eval_root / "textdet_val_rot270.json",
        "val_hard": shared_eval_root / "textdet_val_hard.json",
    }


def build_output_dir(repo_root: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (repo_root / "work_dirs" / "end_to_end_eval" / "rotation_variant_report").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate end-to-end OCR on PTDR val variants with four options: baseline, "
            "crop rotation classifier, affine STN, and both together."
        )
    )
    parser.add_argument("--dbnet-checkpoint", type=Path, default=DEFAULT_DBNET_CKPT)
    parser.add_argument("--parseq-checkpoint", type=Path, default=DEFAULT_PARSEQ_CKPT)
    parser.add_argument("--crop-rotation-checkpoint", type=Path, default=DEFAULT_CROP_ROT_CKPT)
    parser.add_argument("--stn-checkpoint", type=Path, default=DEFAULT_STN_CKPT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--det-score-thr", type=float, default=0.3)
    parser.add_argument("--match-iou-thr", type=float, default=0.5)
    parser.add_argument("--recognition-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "crop", "stn", "both"],
        default=["baseline", "crop", "stn", "both"],
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    detector = load_dbnet_detector(args.dbnet_checkpoint.resolve(), repo_root=repo_root, device=args.device)
    recognizer = load_parseq_recognizer(args.parseq_checkpoint.resolve(), device=args.device)
    need_crop = any(variant in {"crop", "both"} for variant in args.variants)
    need_stn = any(variant in {"stn", "both"} for variant in args.variants)
    crop_bundle = (
        load_crop_rotation_bundle(args.crop_rotation_checkpoint.resolve(), device=args.device)
        if need_crop
        else None
    )
    stn_bundle = (
        load_stn_bundle(args.stn_checkpoint.resolve(), device=args.device)
        if need_stn
        else None
    )

    manifest_paths = default_val_manifest_paths(detector, repo_root)
    output_dir = build_output_dir(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_variants = {
        "baseline": {"crop_bundle": None, "stn_bundle": None},
        "crop": {"crop_bundle": crop_bundle, "stn_bundle": None},
        "stn": {"crop_bundle": None, "stn_bundle": stn_bundle},
        "both": {"crop_bundle": crop_bundle, "stn_bundle": stn_bundle},
    }
    variants = {name: all_variants[name] for name in args.variants}

    combined_summary: dict[str, Any] = {
        "dbnet_checkpoint": str(detector.checkpoint_path),
        "parseq_checkpoint": str(recognizer.checkpoint_path),
        "crop_rotation_checkpoint": str(crop_bundle.checkpoint_path) if crop_bundle is not None else None,
        "stn_checkpoint": str(stn_bundle.checkpoint_path) if stn_bundle is not None else None,
        "det_score_thr": args.det_score_thr,
        "match_iou_thr": args.match_iou_thr,
        "recognition_batch_size": args.recognition_batch_size,
        "requested_variants": args.variants,
        "variants": {},
    }

    for split_name, manifest_path in manifest_paths.items():
        records = load_detection_records(manifest_path)
        for variant_name, variant_kwargs in variants.items():
            variant_output_dir = output_dir / variant_name / split_name
            summary_path = variant_output_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                combined_summary.setdefault("variants", {}).setdefault(variant_name, {})[split_name] = summary
                print(json.dumps({"skipped_existing": True, "variant": variant_name, "split": split_name}, ensure_ascii=False))
                continue
            summary, results = evaluate_variant_on_records(
                variant_name=variant_name,
                split=split_name,
                records=records,
                detector=detector,
                recognizer=recognizer,
                repo_root=repo_root,
                det_score_thr=args.det_score_thr,
                match_iou_thr=args.match_iou_thr,
                recognition_batch_size=args.recognition_batch_size,
                crop_bundle=variant_kwargs["crop_bundle"],
                stn_bundle=variant_kwargs["stn_bundle"],
                limit=args.limit,
            )
            save_evaluation_outputs(variant_output_dir, summary, results)
            combined_summary.setdefault("variants", {}).setdefault(variant_name, {})[split_name] = summary
            print(json.dumps(summary, ensure_ascii=False))

    (output_dir / "combined_summary.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(combined_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
