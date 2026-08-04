from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from .eval_dinov3_deimv2 import (
        AREA_RANGES,
        average_precision,
        load_coco,
        threshold_metrics,
        write_csv,
    )
except ImportError:
    from eval_dinov3_deimv2 import (  # type: ignore[no-redef]
        AREA_RANGES,
        average_precision,
        load_coco,
        threshold_metrics,
        write_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RTMDet with the same manual COCO-like metrics used by "
            "the face_pose_drive_v1 YOLO/DINO benchmark."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--category-id", type=int, default=2)
    parser.add_argument("--category-name", default="face")
    parser.add_argument(
        "--model-label-id",
        type=int,
        default=0,
        help="Zero-based MMDetection label to evaluate (RTMDet-HF uses head=0, face=1).",
    )
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, required=True)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def tensor_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def prediction_rows(
    image_info: dict[str, Any],
    result: Any,
    *,
    category_id: int,
    category_name: str,
    model_label_id: int,
    score_min: float,
    max_det: int,
) -> list[dict[str, Any]]:
    instances = result.pred_instances
    boxes = tensor_rows(instances.bboxes)
    scores = tensor_rows(instances.scores)
    labels = tensor_rows(instances.labels)
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    rows: list[dict[str, Any]] = []
    for index in order[:max_det]:
        if float(scores[index]) < score_min or int(labels[index]) != model_label_id:
            continue
        x1, y1, x2, y2 = [float(value) for value in boxes[index]]
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "image_id": int(image_info["id"]),
                "file_name": str(image_info.get("file_name", "")),
                "category_id": int(category_id),
                "category_name": category_name,
                "score": float(scores[index]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from mmdet.apis import inference_detector, init_detector  # type: ignore

    images, gt_by_image = load_coco(args.annotation, [args.category_id])
    model = init_detector(
        args.config.as_posix(), args.checkpoint.as_posix(), device=args.device
    )

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for image_info in images:
        image_path = args.image_root / str(image_info["file_name"])
        result = inference_detector(model, image_path.as_posix())
        predictions.extend(
            prediction_rows(
                image_info,
                result,
                category_id=args.category_id,
                category_name=args.category_name,
                model_label_id=args.model_label_id,
                score_min=args.score_min,
                max_det=args.max_det,
            )
        )
    elapsed = time.perf_counter() - started

    iou_thresholds = [
        round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)
    ]
    ap_values = {
        f"AP{int(threshold * 100):02d}": average_precision(
            gt_by_image,
            predictions,
            category_id=args.category_id,
            iou_threshold=threshold,
        )
        for threshold in iou_thresholds
    }
    ap_values["mAP50_95"] = float(np.mean(list(ap_values.values())))

    size_ap: dict[str, float] = {}
    for size_name, area_range in AREA_RANGES.items():
        size_ap[size_name] = float(
            np.mean(
                [
                    average_precision(
                        gt_by_image,
                        predictions,
                        category_id=args.category_id,
                        iou_threshold=threshold,
                        area_range=area_range,
                    )
                    for threshold in iou_thresholds
                ]
            )
        )

    thresholds = [round(float(value), 2) for value in np.arange(0.01, 0.96, 0.01)]
    threshold_rows = [
        threshold_metrics(
            gt_by_image,
            predictions,
            category_id=None,
            score_threshold=threshold,
        )
        for threshold in thresholds
    ]
    fixed_metrics = threshold_metrics(
        gt_by_image,
        predictions,
        category_id=None,
        score_threshold=args.fixed_threshold,
    )
    best_metrics = max(
        threshold_rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
        ),
    )

    metrics = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "annotation": str(args.annotation),
        "image_root": str(args.image_root),
        "model_label_id": args.model_label_id,
        "manual_coco_like_evaluator": True,
        "num_images": len(images),
        "num_gt": sum(len(rows) for rows in gt_by_image.values()),
        "num_predictions_score_min": len(predictions),
        "mAP": ap_values["mAP50_95"],
        "AP50": ap_values["AP50"],
        "AP75": ap_values["AP75"],
        "AP_small": size_ap["small"],
        "AP_medium": size_ap["medium"],
        "AP_large": size_ap["large"],
        "recall_fixed_threshold": fixed_metrics["recall"],
        "precision_fixed_threshold": fixed_metrics["precision"],
        "f1_fixed_threshold": fixed_metrics["f1"],
        "tp_fixed_threshold": fixed_metrics["tp"],
        "fp_fixed_threshold": fixed_metrics["fp"],
        "fn_fixed_threshold": fixed_metrics["fn"],
        "fixed_threshold": args.fixed_threshold,
        "best_threshold_by_f1": best_metrics,
        "class_ap": {args.category_name: ap_values},
        "size_ap": size_ap,
        "inference_elapsed_sec": elapsed,
        "inference_images_per_sec": len(images) / elapsed if elapsed > 0 else 0.0,
        "score_min": args.score_min,
        "max_det": args.max_det,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "threshold_metrics.csv",
        threshold_rows,
        ["category_id", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
