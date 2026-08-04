from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_dinov3_deimv2 import AREA_RANGES, average_precision, load_coco, threshold_metrics, write_csv
from scripts_YOLO.yolo26_pose.inference import iter_batched_image_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO detector with the same manual COCO-like metrics used by eval_dinov3_deimv2.py."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=2)
    parser.add_argument("--category-name", default="face")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, required=True)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    """Resolve a COCO file name without duplicating shared path components.

    COCO files in this project store paths such as ``images/test/frame.jpg``.
    Callers may pass either the dataset root or the more specific
    ``.../images/test`` directory as ``image_root``.  Prefer the conventional
    direct join, then remove only an exact overlap between the root suffix and
    the file-name prefix.  This keeps path handling deterministic and avoids a
    filesystem-wide search.
    """

    relative = Path(file_name)
    if relative.is_absolute():
        if relative.is_file():
            return relative
        raise FileNotFoundError(f"COCO image does not exist: {relative}")

    direct = image_root / relative
    if direct.is_file():
        return direct

    root_parts = image_root.parts
    file_parts = relative.parts
    max_overlap = min(len(root_parts), len(file_parts))
    for overlap in range(max_overlap, 0, -1):
        if root_parts[-overlap:] != file_parts[:overlap]:
            continue
        candidate = image_root.joinpath(*file_parts[overlap:])
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Unable to resolve COCO image path: "
        f"image_root={image_root}, file_name={file_name}, direct={direct}"
    )


def resolve_image_sources(image_root: Path, images: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    sources: list[str] = []
    for image in images:
        file_name = str(image.get("file_name", ""))
        try:
            sources.append(str(resolve_image_path(image_root, file_name)))
        except FileNotFoundError as error:
            missing.append(str(error))
    if missing:
        preview = "\n".join(missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Failed to resolve {len(missing)}/{len(images)} COCO image paths:\n{preview}{suffix}"
        )
    return sources


def prediction_rows(
    image_info: dict[str, Any],
    result: Any,
    *,
    category_id: int,
    category_name: str,
) -> list[dict[str, Any]]:
    boxes = tensor_to_list(result.boxes.xyxy) if result.boxes is not None else []
    scores = tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    rows = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
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

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt before evaluation.") from error

    images, gt_by_image = load_coco(args.annotation, [args.category_id])
    sources = [Path(source) for source in resolve_image_sources(args.image_root, images)]
    model = YOLO(str(args.weights))

    started = time.perf_counter()
    results = iter_batched_image_predictions(
        model,
        sources,
        imgsz=args.image_size,
        conf=args.score_min,
        device=args.device,
        batch_size=args.batch_size,
        max_det=args.max_det,
    )
    predictions: list[dict[str, Any]] = []
    for image_info, (_, result) in zip(images, results, strict=True):
        predictions.extend(
            prediction_rows(
                image_info,
                result,
                category_id=args.category_id,
                category_name=args.category_name,
            )
        )
    elapsed = time.perf_counter() - started

    iou_thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
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

    size_ap = {}
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

    threshold_rows = [
        threshold_metrics(gt_by_image, predictions, category_id=None, score_threshold=threshold)
        for threshold in [round(float(value), 2) for value in np.arange(0.01, 0.96, 0.01)]
    ]
    fixed_metrics = threshold_metrics(
        gt_by_image,
        predictions,
        category_id=None,
        score_threshold=args.fixed_threshold,
    )
    best_metrics = max(
        threshold_rows,
        key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])),
    )

    metrics = {
        "weights": str(args.weights),
        "annotation": str(args.annotation),
        "image_root": str(args.image_root),
        "manual_coco_like_evaluator": True,
        "image_size": args.image_size,
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
        "fixed_threshold": args.fixed_threshold,
        "best_threshold_by_f1": best_metrics,
        "class_ap": {args.category_name: ap_values},
        "size_ap": size_ap,
        "inference_elapsed_sec": elapsed,
        "inference_images_per_sec": len(images) / elapsed if elapsed > 0 else 0.0,
        "score_min": args.score_min,
        "max_det": args.max_det,
        "batch_size": args.batch_size,
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
