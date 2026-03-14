#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_schema import DEFAULT_REPO_ROOT
from end_to_end_utils import (
    evaluate_records,
    load_dbnet_detector,
    load_detection_records,
    load_parseq_recognizer,
    run_end_to_end_inference,
    save_evaluation_outputs,
    save_visualizations,
)


def build_output_dir(
    repo_root: Path,
    dbnet_checkpoint: Path,
    parseq_checkpoint: Path,
    split: str,
    explicit_output_dir: Path | None,
) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dbnet_name = dbnet_checkpoint.stem
    parseq_name = parseq_checkpoint.stem
    return (repo_root / "work_dirs" / "end_to_end_eval" / f"{timestamp}_{split}_{dbnet_name}_{parseq_name}").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PTDR end-to-end OCR evaluation by combining a DBNet text detector checkpoint "
            "with a PARSeq recognition checkpoint."
        )
    )
    parser.add_argument("--dbnet-checkpoint", type=Path, required=True, help="Path to a DBNet/MMOCR checkpoint (.pth).")
    parser.add_argument(
        "--parseq-checkpoint",
        type=Path,
        required=True,
        help="Path to a PARSeq Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Which detection manifest split to evaluate. Default: test.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root used to resolve manifest and image paths.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest override. If omitted, the script uses the manifest path embedded in the DBNet checkpoint config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for inference, for example cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        default=0.3,
        help="Minimum DBNet polygon score kept for end-to-end evaluation.",
    )
    parser.add_argument(
        "--match-iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for matching a predicted polygon to a ground-truth polygon.",
    )
    parser.add_argument(
        "--recognition-batch-size",
        type=int,
        default=32,
        help="Batch size used when PARSeq recognizes the detected crops from one image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional image cap for smoke tests or debugging.",
    )
    parser.add_argument(
        "--save-vis",
        type=int,
        default=12,
        help="How many evaluated images to save with detailed overlay and crop visualizations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary metrics, per-image JSONL, and visualization PNGs.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dbnet_checkpoint = args.dbnet_checkpoint.resolve()
    parseq_checkpoint = args.parseq_checkpoint.resolve()

    detector = load_dbnet_detector(dbnet_checkpoint, repo_root=repo_root, device=args.device)
    recognizer = load_parseq_recognizer(parseq_checkpoint, device=args.device)

    manifest_path = args.manifest_path.resolve() if args.manifest_path is not None else detector.manifest_paths[args.split]
    records = load_detection_records(manifest_path)

    output_dir = build_output_dir(
        repo_root=repo_root,
        dbnet_checkpoint=dbnet_checkpoint,
        parseq_checkpoint=parseq_checkpoint,
        split=args.split,
        explicit_output_dir=args.output_dir,
    )
    vis_dir = output_dir / "visualizations"

    summary, results = evaluate_records(
        records=records,
        detector=detector,
        recognizer=recognizer,
        repo_root=repo_root,
        split=args.split,
        det_score_thr=args.det_score_thr,
        match_iou_thr=args.match_iou_thr,
        recognition_batch_size=args.recognition_batch_size,
        limit=args.limit,
    )
    summary["manifest_path"] = str(manifest_path)
    summary["output_dir"] = str(output_dir)
    save_evaluation_outputs(output_dir=output_dir, summary=summary, results=results)

    for result in results[: max(0, args.save_vis)]:
        record = next(record for record in records if record["img_path"] == result.image_path)
        fresh_result, image_rgb, crops = run_end_to_end_inference(
            record=record,
            detector=detector,
            recognizer=recognizer,
            repo_root=repo_root,
            split=args.split,
            det_score_thr=args.det_score_thr,
            match_iou_thr=args.match_iou_thr,
            recognition_batch_size=args.recognition_batch_size,
        )
        files = save_visualizations(vis_dir, fresh_result, image_rgb=image_rgb, crops=crops)
        print(json.dumps({"image": fresh_result.image_path, "visualizations": files}, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
