from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dinov3_deimv2_common import (  # noqa: E402
    ensure_output_dir,
    load_checkpoint,
    load_config,
    parse_image_size,
    resolve_data_paths,
    save_run_context,
)
from scripts.train_dino import validate_detection_split  # noqa: E402
from src.evaluation.baseline_aligned import (  # noqa: E402
    evaluate_normalized_predictions,
    sha256_file,
)
from src.models import build_detector_from_config  # noqa: E402
from src.evaluation.face_bbox_iou50 import write_evaluation as write_bbox_iou50_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct-ellipse DEIMv2-L on an ellipse COCO split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", default=None, help="INT or HEIGHTxWIDTH")
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, default=0.30)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--baseline-annotation", type=Path)
    parser.add_argument("--baseline-aligned-output-dir", type=Path)
    parser.add_argument("--architecture-id")
    parser.add_argument("--bbox-iou50-output-dir", type=Path)
    parser.add_argument("--bbox-iou50-alpha", type=float)
    parser.add_argument("--bbox-iou50-validation-metrics", type=Path)
    parser.add_argument("--bbox-iou50-threshold-floor-exclusive", type=float, default=0.33)
    parser.add_argument(
        "--comparison-stage",
        default="full_image_face_ellipse",
    )
    parser.add_argument(
        "--bbox-reference-map",
        type=float,
        default=None,
        help="Optional bbox mAP50:95 baseline used for the <=0.01 non-regression gate.",
    )
    return parser.parse_args()


def _baseline_predictions(
    records: list[dict[str, Any]],
    annotation_path: Path,
) -> dict[int, dict[str, Any]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    exact = {
        str(image["file_name"]).replace("\\", "/"): int(image["id"])
        for image in coco["images"]
    }
    basenames: dict[str, int | None] = {}
    for file_name, image_id in exact.items():
        name = Path(file_name).name
        basenames[name] = (
            image_id
            if name not in basenames
            else None
        )
    output = {
        int(image["id"]): {"heads": [], "faces": [], "poses": []}
        for image in coco["images"]
    }
    for record in records:
        file_name = str(record.get("file_name", "")).replace("\\", "/")
        image_id = exact.get(file_name)
        if image_id is None:
            image_id = basenames.get(Path(file_name).name)
        if image_id is None:
            raise KeyError(
                f"Detector prediction is absent from baseline annotations: {file_name}"
            )
        output[image_id]["faces"].append(
            {
                "bbox_xyxy": record["bbox_xyxy"],
                "ellipse": record.get("ellipse"),
                "score": float(record["score"]),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    baseline_values = (
        args.baseline_annotation,
        args.baseline_aligned_output_dir,
        args.architecture_id,
    )
    if any(value is not None for value in baseline_values) and not all(
        value is not None for value in baseline_values
    ):
        raise ValueError(
            "--baseline-annotation, --baseline-aligned-output-dir, and "
            "--architecture-id must be provided together"
        )
    output_dir = ensure_output_dir(args.output_dir)
    config = load_config(args.config)
    config, _ = resolve_data_paths(config, args.data_root, split=args.split)
    save_run_context(output_dir, config)
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    image_size = parse_image_size(
        args.image_size or config.get("data", {}).get("image_size", 960)
    )
    if image_size is None:
        raise ValueError("image_size must be configured")

    build_config = copy.deepcopy(config)
    build_config.get("model", {}).pop("init_checkpoint", None)
    model = build_detector_from_config(build_config).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    predictions_path = (
        output_dir / "predictions.json"
        if args.baseline_aligned_output_dir is not None
        or args.bbox_iou50_output_dir is not None
        else None
    )
    metrics = validate_detection_split(
        model,
        config,
        device=device,
        split=args.split,
        image_size=image_size,
        fixed_threshold=float(args.fixed_threshold),
        score_min=float(args.score_min),
        max_images=args.max_images,
        batch_size=int(args.batch_size),
        predictions_output=predictions_path,
    )
    acceptance_cfg = dict(config.get("acceptance", {}))
    configured_references = acceptance_cfg.get("bbox_reference_map", {})
    configured_reference = configured_references.get(args.split) if isinstance(configured_references, dict) else None
    bbox_reference = args.bbox_reference_map if args.bbox_reference_map is not None else configured_reference
    max_bbox_drop = float(acceptance_cfg.get("max_bbox_map_drop", 0.01))
    bbox_non_regression = (
        None
        if bbox_reference is None
        else float(metrics["mAP"]) >= float(bbox_reference) - max_bbox_drop
    )
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "acceptance": {
            "ellipse_beats_bbox_ellipse_baseline": metrics["ellipse_mAP"] > metrics["bbox_ellipse_mAP"],
            "bbox_reference_map": float(bbox_reference) if bbox_reference is not None else None,
            "max_bbox_map_drop": max_bbox_drop,
            "bbox_non_regression": bbox_non_regression,
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.bbox_iou50_output_dir is not None:
        if args.bbox_iou50_alpha is None:
            raise ValueError(
                "--bbox-iou50-alpha is required with --bbox-iou50-output-dir"
            )
        assert predictions_path is not None
        annotation_path = Path(str(config.get("data", {}).get("annotation_path", "")))
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"resolved FACE bbox annotation is missing: {annotation_path}"
            )
        category_ids = [
            int(value)
            for value in config.get("data", {}).get("category_ids", [2])
        ]
        bbox_metrics = write_bbox_iou50_evaluation(
            annotation_path=annotation_path,
            predictions_path=predictions_path,
            output_dir=args.bbox_iou50_output_dir.expanduser().resolve(),
            split=str(args.split),
            alpha=float(args.bbox_iou50_alpha),
            category_ids=category_ids,
            common_threshold=0.33,
            threshold_floor_exclusive=float(
                args.bbox_iou50_threshold_floor_exclusive
            ),
            validation_metrics_path=(
                args.bbox_iou50_validation_metrics.expanduser().resolve()
                if args.bbox_iou50_validation_metrics is not None
                else None
            ),
        )
        summary["bbox_iou50"] = {
            "status": "success",
            "output_dir": str(args.bbox_iou50_output_dir.expanduser().resolve()),
            "selected_threshold": bbox_metrics["validation_selected"]["threshold"],
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.baseline_aligned_output_dir is not None:
        assert args.baseline_annotation is not None
        assert args.architecture_id is not None
        assert predictions_path is not None
        annotation_path = args.baseline_annotation.expanduser().resolve()
        records = json.loads(predictions_path.read_text(encoding="utf-8"))
        baseline_metrics = evaluate_normalized_predictions(
            annotation_path=annotation_path,
            predictions_by_image=_baseline_predictions(records, annotation_path),
            split=str(args.split),
            architecture_id=str(args.architecture_id),
            comparison_stage=str(args.comparison_stage),
            output_dir=args.baseline_aligned_output_dir.expanduser().resolve(),
            provenance={
                "baseline_role": "candidate_local_face_detector",
                "config": str(args.config.expanduser().resolve()),
                "config_sha256": sha256_file(args.config.expanduser().resolve()),
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": sha256_file(
                    args.checkpoint.expanduser().resolve()
                ),
                "raw_predictions": str(predictions_path.resolve()),
                "raw_predictions_sha256": sha256_file(predictions_path.resolve()),
                "seed": int(config.get("train", {}).get("seed", 42)),
            },
            runtime={
                "end_to_end_elapsed_seconds": float(
                    metrics["elapsed_seconds"]
                ),
                "images_per_second": float(metrics["num_images"])
                / max(float(metrics["elapsed_seconds"]), 1.0e-9),
                "batch_size": int(args.batch_size),
                "input_size": list(image_size)
                if isinstance(image_size, tuple)
                else int(image_size),
            },
            face_score_candidate_floor=float(args.score_min),
        )
        summary["baseline_aligned"] = {
            "status": baseline_metrics["status"],
            "output_dir": str(
                args.baseline_aligned_output_dir.expanduser().resolve()
            ),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
