from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dinov3_deimv2_common import (  # noqa: E402
    category_ids,
    class_names,
    ensure_output_dir,
    image_to_tensor,
    load_checkpoint,
    load_config,
    resolve_data_paths,
    save_run_context,
    scale_detections_to_original,
)
from src.models import build_detector_from_config  # noqa: E402


AREA_RANGES = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}
COLORS = {"gt": "lime", "head": "yellow", "face": "red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 + official-style DEIMv2 detector on a COCO split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, default=0.30)
    parser.add_argument("--max-visualizations", type=int, default=30)
    parser.add_argument(
        "--visualization-random-seed",
        type=int,
        default=None,
        help="Randomly choose visualization frames with this seed while still evaluating the full split.",
    )
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def box_area_xyxy(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / max(union, 1e-9)


def load_coco(annotation_path: Path, selected_category_ids: list[int]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = list(data.get("images", []))
    allowed = set(int(value) for value in selected_category_ids)
    gt_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in images}
    for annotation in data.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))
        if category_id not in allowed or int(annotation.get("iscrowd", 0)):
            continue
        xyxy = xywh_to_xyxy(annotation.get("bbox", [0, 0, 0, 0]))
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            {
                "image_id": int(annotation["image_id"]),
                "category_id": category_id,
                "bbox_xyxy": xyxy,
                "area": float(annotation.get("area", box_area_xyxy(xyxy))),
            }
        )
    return images, gt_by_image


def detections_to_records(
    image_info: dict[str, Any],
    detections: dict[str, torch.Tensor],
    names: list[str],
    coco_ids: list[int],
) -> list[dict[str, Any]]:
    image_id = int(image_info["id"])
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    labels = detections.get("labels")
    labels = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.zeros((boxes.shape[0],), dtype=torch.long)
    rows = []
    for index in range(boxes.shape[0]):
        label_index = int(labels[index])
        category_id = coco_ids[label_index] if 0 <= label_index < len(coco_ids) else label_index + 1
        category_name = names[label_index] if 0 <= label_index < len(names) else str(category_id)
        x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "image_id": image_id,
                "file_name": image_info.get("file_name", ""),
                "category_id": int(category_id),
                "category_name": category_name,
                "score": float(scores[index]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return rows


def filter_gt_area(gt: dict[int, list[dict[str, Any]]], area_range: tuple[float, float] | None) -> dict[int, list[dict[str, Any]]]:
    if area_range is None:
        return gt
    low, high = area_range
    return {
        image_id: [item for item in rows if low <= float(item["area"]) < high]
        for image_id, rows in gt.items()
    }


def average_precision(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float,
    area_range: tuple[float, float] | None = None,
) -> float:
    gt_filtered = filter_gt_area(gt_by_image, area_range)
    gt_category = {
        image_id: [item for item in rows if int(item["category_id"]) == int(category_id)]
        for image_id, rows in gt_filtered.items()
    }
    total_gt = sum(len(rows) for rows in gt_category.values())
    if total_gt == 0:
        return 0.0
    preds = [row for row in predictions if int(row["category_id"]) == int(category_id)]
    preds.sort(key=lambda row: float(row["score"]), reverse=True)
    if not preds:
        return 0.0
    matched: dict[int, set[int]] = defaultdict(set)
    tp: list[float] = []
    fp: list[float] = []
    for pred in preds:
        image_id = int(pred["image_id"])
        gt_rows = gt_category.get(image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_rows):
            if index in matched[image_id]:
                continue
            overlap = iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(1e-9, tp_cum + fp_cum)
    ap = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        keep = recalls >= recall_threshold
        ap += float(precisions[keep].max()) if keep.any() else 0.0
    return ap / 101.0


def threshold_metrics(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    category_id: int | None,
    score_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    gt_rows_by_image = {}
    for image_id, rows in gt_by_image.items():
        gt_rows_by_image[image_id] = [row for row in rows if category_id is None or int(row["category_id"]) == int(category_id)]
    pred_rows = [row for row in predictions if float(row["score"]) >= score_threshold and (category_id is None or int(row["category_id"]) == int(category_id))]
    pred_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    tp = fp = 0
    for pred in pred_rows:
        image_id = int(pred["image_id"])
        gt_rows = gt_rows_by_image.get(image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_rows):
            if index in matched[image_id]:
                continue
            overlap = iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            tp += 1
        else:
            fp += 1
    total_gt = sum(len(rows) for rows in gt_rows_by_image.values())
    fn = max(0, total_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "category_id": "all" if category_id is None else int(category_id),
        "threshold": float(score_threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def draw_overlay(path: Path, image: Image.Image, gt: list[dict[str, Any]], predictions: list[dict[str, Any]], names_by_id: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for row in gt:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        name = names_by_id.get(int(row["category_id"]), str(row["category_id"]))
        draw.rectangle((x1, y1, x2, y2), outline=COLORS["gt"], width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), f"GT {name}", fill=COLORS["gt"])
    for row in predictions:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        color = COLORS.get(str(row["category_name"]), "yellow")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), f"{row['category_name']} {row['score']:.2f}", fill=color)
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    config = load_config(args.config)
    config, _force_synthetic = resolve_data_paths(config, args.data_root, split=args.split)
    save_run_context(output_dir, config)

    data_cfg = config.get("data", {})
    annotation_path = Path(data_cfg.get("annotation_path", ""))
    image_root = Path(data_cfg.get("image_root", data_cfg.get("data_root", "")))
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found for split={args.split}: {annotation_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    names = class_names(config)
    coco_ids = category_ids(config)
    names_by_id = {category_id: names[index] if index < len(names) else str(category_id) for index, category_id in enumerate(coco_ids)}
    image_size = int(args.image_size or data_cfg.get("image_size") or config.get("train", {}).get("imgsz", 960))
    images, gt_by_image = load_coco(annotation_path, coco_ids)

    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    model = build_detector_from_config(config).to(device)
    missing: list[str] = []
    unexpected: list[str] = []
    if args.checkpoint.exists():
        missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    elif not args.allow_random_init:
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if hasattr(model, "score_threshold"):
        model.score_threshold = float(args.score_min)
    if hasattr(model, "postprocessor") and hasattr(model.postprocessor, "score_threshold"):
        model.postprocessor.score_threshold = float(args.score_min)
    model.eval()

    predictions: list[dict[str, Any]] = []
    overlay_dir = output_dir / "visualizations"
    failure_dir = output_dir / "failure_samples"
    visualization_count = min(args.max_visualizations, len(images))
    if args.visualization_random_seed is None:
        visualization_indices = set(range(visualization_count))
    else:
        visualization_indices = set(
            random.Random(args.visualization_random_seed).sample(range(len(images)), visualization_count)
        )
    started = time.perf_counter()
    for index, image_info in enumerate(images):
        image_path = image_root / str(image_info.get("file_name", ""))
        image = Image.open(image_path).convert("RGB")
        tensor, input_hw, original_hw = image_to_tensor(image, image_size)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            detections = model(tensor.to(device))[0]
        detections = scale_detections_to_original(detections, input_hw=input_hw, original_hw=original_hw)
        rows = detections_to_records(image_info, detections, names, coco_ids)
        predictions.extend(rows)
        fixed_rows = [row for row in rows if float(row["score"]) >= float(args.fixed_threshold)]
        if index in visualization_indices:
            draw_overlay(overlay_dir / Path(str(image_info.get("file_name", image_info["id"]))).name, image.copy(), gt_by_image.get(int(image_info["id"]), []), fixed_rows, names_by_id)
        metrics_for_image = threshold_metrics(
            {int(image_info["id"]): gt_by_image.get(int(image_info["id"]), [])},
            fixed_rows,
            category_id=None,
            score_threshold=args.fixed_threshold,
        )
        if index in visualization_indices and (metrics_for_image["fp"] > 0 or metrics_for_image["fn"] > 0):
            draw_overlay(failure_dir / Path(str(image_info.get("file_name", image_info["id"]))).name, image.copy(), gt_by_image.get(int(image_info["id"]), []), fixed_rows, names_by_id)
    elapsed = time.perf_counter() - started

    iou_thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
    class_ap: dict[str, dict[str, float]] = {}
    for category_id in coco_ids:
        ap_values = {
            f"AP{int(threshold * 100):02d}": average_precision(gt_by_image, predictions, category_id=category_id, iou_threshold=threshold)
            for threshold in iou_thresholds
        }
        ap_values["mAP50_95"] = float(np.mean(list(ap_values.values()))) if ap_values else 0.0
        class_ap[names_by_id.get(category_id, str(category_id))] = ap_values

    ap50_values = [values.get("AP50", 0.0) for values in class_ap.values()]
    ap75_values = [values.get("AP75", 0.0) for values in class_ap.values()]
    map_values = [values.get("mAP50_95", 0.0) for values in class_ap.values()]
    size_ap = {}
    for size_name, area_range in AREA_RANGES.items():
        per_class = [
            float(
                np.mean(
                    [
                        average_precision(gt_by_image, predictions, category_id=category_id, iou_threshold=threshold, area_range=area_range)
                        for threshold in iou_thresholds
                    ]
                )
            )
            for category_id in coco_ids
        ]
        size_ap[size_name] = float(np.mean(per_class)) if per_class else 0.0

    threshold_rows: list[dict[str, Any]] = []
    for threshold in [round(float(value), 2) for value in np.arange(0.01, 0.96, 0.01)]:
        threshold_rows.append(threshold_metrics(gt_by_image, predictions, category_id=None, score_threshold=threshold))
        for category_id in coco_ids:
            row = threshold_metrics(gt_by_image, predictions, category_id=category_id, score_threshold=threshold)
            row["category_name"] = names_by_id.get(category_id, str(category_id))
            threshold_rows.append(row)
    fixed_metrics = threshold_metrics(gt_by_image, predictions, category_id=None, score_threshold=args.fixed_threshold)
    best_metrics = max((row for row in threshold_rows if row["category_id"] == "all"), key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])), default=fixed_metrics)

    metrics = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "annotation": str(annotation_path),
        "image_root": str(image_root),
        "split": args.split,
        "device": device,
        "manual_coco_like_evaluator": True,
        "image_size": image_size,
        "num_images": len(images),
        "num_gt": sum(len(rows) for rows in gt_by_image.values()),
        "num_predictions_score_min": len(predictions),
        "mAP": float(np.mean(map_values)) if map_values else 0.0,
        "AP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "AP75": float(np.mean(ap75_values)) if ap75_values else 0.0,
        "AP_small": size_ap["small"],
        "AP_medium": size_ap["medium"],
        "AP_large": size_ap["large"],
        "recall_fixed_threshold": fixed_metrics["recall"],
        "precision_fixed_threshold": fixed_metrics["precision"],
        "f1_fixed_threshold": fixed_metrics["f1"],
        "fixed_threshold": args.fixed_threshold,
        "best_threshold_by_f1": best_metrics,
        "class_ap": class_ap,
        "size_ap": size_ap,
        "inference_elapsed_sec": elapsed,
        "inference_images_per_sec": len(images) / elapsed if elapsed > 0 else None,
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "visualization_dir": str(overlay_dir),
        "visualization_random_seed": args.visualization_random_seed,
        "failure_samples_dir": str(failure_dir),
    }
    metrics_rows = [
        {
            "mAP": metrics["mAP"],
            "AP50": metrics["AP50"],
            "AP75": metrics["AP75"],
            "AP_small": metrics["AP_small"],
            "AP_medium": metrics["AP_medium"],
            "AP_large": metrics["AP_large"],
            "precision": metrics["precision_fixed_threshold"],
            "recall": metrics["recall_fixed_threshold"],
            "f1": metrics["f1_fixed_threshold"],
            **{f"{name}_mAP50_95": values["mAP50_95"] for name, values in class_ap.items()},
            **{f"{name}_AP50": values["AP50"] for name, values in class_ap.items()},
        }
    ]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "metrics.csv", metrics_rows, list(metrics_rows[0].keys()))
    write_csv(
        output_dir / "threshold_metrics.csv",
        threshold_rows,
        ["category_id", "threshold", "tp", "fp", "fn", "precision", "recall", "f1", "category_name"],
    )
    (output_dir / "eval.log").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_lines = [
        "# DINOv3 Official-Style DEIMv2 Evaluation Summary",
        "",
        "## Purpose",
        "Compare bbox detection quality for head/face detection.",
        "",
        "## Result",
        f"- mAP50:95: {metrics['mAP']:.4f}",
        f"- AP50: {metrics['AP50']:.4f}",
        f"- AP75: {metrics['AP75']:.4f}",
        f"- fixed threshold F1: {metrics['f1_fixed_threshold']:.4f}",
        "",
        "## Next Improvements",
        "- Compare against YOLO11s and DINOv3 CenterFace-style on the same split.",
        "- Inspect failure_samples for class confusion and missed small faces.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
