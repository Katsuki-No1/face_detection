from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402
from src.data import FaceKeypointDataset, collate_face_keypoint_batch  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINO CenterFace ellipse detector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--thresholds", default="0.01:0.95:0.01")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-pck-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-visualizations", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit evaluated samples for smoke/debug runs.")
    return parser.parse_args()


def parse_thresholds(value: str, fixed_threshold: float | None) -> list[float]:
    if fixed_threshold is not None:
        return [fixed_threshold]
    if ":" in value:
        start, end, step = [float(item) for item in value.split(":", 2)]
        thresholds = []
        current = start
        while current <= end + 1e-12:
            thresholds.append(round(current, 6))
            current += step
        return thresholds
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = bbox_area([x1, y1, x2, y2])
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / union if union > 0 else 0.0


def greedy_matches(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], iou_threshold: float) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            iou = bbox_iou(gt["bbox"], prediction["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def ellipse_polygon(ellipse: list[float], *, steps: int = 96) -> list[tuple[float, float]]:
    cx, cy, rx, ry, theta = ellipse
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points = []
    for index in range(steps):
        angle = 2.0 * math.pi * index / steps
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append((cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t))
    return points


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _inside(point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> bool:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) >= (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])


def _intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = edge_start
    x4, y4 = edge_end
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def convex_intersection(subject: list[tuple[float, float]], clipper: list[tuple[float, float]]) -> list[tuple[float, float]]:
    output = subject
    for index, edge_start in enumerate(clipper):
        edge_end = clipper[(index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, edge_start, edge_end):
                if not _inside(previous, edge_start, edge_end):
                    output.append(_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif _inside(previous, edge_start, edge_end):
                output.append(_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def ellipse_iou(first: list[float], second: list[float]) -> float:
    first_poly = ellipse_polygon(first)
    second_poly = ellipse_polygon(second)
    inter = polygon_area(convex_intersection(first_poly, second_poly))
    union = polygon_area(first_poly) + polygon_area(second_poly) - inter
    return inter / union if union > 0 else 0.0


def object_scale(gt: dict[str, Any]) -> float:
    return math.sqrt(max(1.0, bbox_area(gt["bbox"])))


def load_model(config: dict[str, Any], checkpoint: Path, device: str, prediction_conf_min: float):
    model = build_detector_from_config(config).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    if hasattr(model, "postprocessor"):
        model.postprocessor.score_threshold = prediction_conf_min
    model.eval()
    return model


def target_to_ground_truth(target: dict[str, Any], image_tensor: torch.Tensor) -> list[dict[str, Any]]:
    boxes = target["boxes"].detach().cpu()
    ellipses = target["ellipses"].detach().cpu()
    keypoints = target["keypoints"].detach().cpu()
    masks = target["keypoint_mask"].detach().cpu()
    objects = target.get("metadata", {}).get("objects", []) if isinstance(target.get("metadata"), dict) else []
    image_h, image_w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    image_mean = float(image_tensor.mean().item())
    rows = []
    for index in range(boxes.shape[0]):
        ellipse = [float(value) for value in ellipses[index].tolist()]
        points = []
        for point_index, name in enumerate(KEYPOINT_NAMES[: keypoints.shape[1]]):
            points.append(
                {
                    "name": name,
                    "x": float(keypoints[index, point_index, 0]),
                    "y": float(keypoints[index, point_index, 1]),
                    "visibility": int(bool(masks[index, point_index].item())),
                }
            )
        tags = []
        area_ratio = bbox_area([float(value) for value in boxes[index].tolist()]) / max(1.0, float(image_h * image_w))
        if area_ratio < 0.01:
            tags.append("small_face")
        if abs(math.degrees(ellipse[4])) >= 20.0:
            tags.append("angled_face")
        if image_mean < 0.25:
            tags.append("dark_low_quality_face")
        if index < len(objects) and isinstance(objects[index], dict):
            obj = objects[index]
            face = obj.get("face", {}) if isinstance(obj.get("face"), dict) else {}
            if bool(face.get("occluded", False)) or any(bool(point.get("occluded", False)) for point in obj.get("keypoints", {}).values() if isinstance(point, dict)):
                tags.append("occluded_face")
            if obj.get("warnings"):
                tags.append("low_quality_face")
        rows.append(
            {
                "bbox": [float(value) for value in boxes[index].tolist()],
                "ellipse": ellipse,
                "keypoints": points,
                "tags": tags,
            }
        )
    return rows


def detections_to_predictions(detections: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    boxes = detections.get("boxes", detections.get("bboxes", torch.zeros((0, 4)))).detach().cpu()
    scores = detections.get("scores", torch.zeros((boxes.shape[0],))).detach().cpu()
    ellipses = detections.get("ellipses", torch.zeros((boxes.shape[0], 5))).detach().cpu()
    keypoints = detections.get("keypoints", torch.zeros((boxes.shape[0], len(KEYPOINT_NAMES), 2))).detach().cpu()
    predictions = []
    for index in range(boxes.shape[0]):
        predictions.append(
            {
                "bbox": [float(value) for value in boxes[index].tolist()],
                "ellipse": [float(value) for value in ellipses[index].tolist()] if index < ellipses.shape[0] else None,
                "score": float(scores[index]) if index < scores.shape[0] else 0.0,
                "keypoints": [
                    {"name": KEYPOINT_NAMES[point_index], "x": float(keypoints[index, point_index, 0]), "y": float(keypoints[index, point_index, 1])}
                    for point_index in range(min(len(KEYPOINT_NAMES), keypoints.shape[1]))
                ],
            }
        )
    return predictions


def average_precision(rows: list[dict[str, Any]], iou_threshold: float) -> float:
    total_gt = sum(len(row["ground_truth"]) for row in rows)
    if total_gt == 0:
        return 0.0
    detections = []
    for image_index, row in enumerate(rows):
        for pred_index, prediction in enumerate(row["predictions"]):
            detections.append((prediction["score"], image_index, pred_index))
    detections.sort(reverse=True)
    matched: dict[int, set[int]] = {}
    tp = []
    fp = []
    for _score, image_index, pred_index in detections:
        row = rows[image_index]
        prediction = row["predictions"][pred_index]
        best_iou = 0.0
        best_gt = -1
        for gt_index, gt in enumerate(row["ground_truth"]):
            if gt_index in matched.get(image_index, set()):
                continue
            iou = bbox_iou(gt["bbox"], prediction["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_iou >= iou_threshold and best_gt >= 0:
            matched.setdefault(image_index, set()).add(best_gt)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        precisions = precision[recall >= threshold]
        ap += float(precisions.max()) if precisions.size else 0.0
    return ap / 101.0


def evaluate_threshold(rows: list[dict[str, Any]], threshold: float, iou_threshold: float, pck_threshold: float) -> dict[str, Any]:
    tp = fp = fn = 0
    ious = []
    ellipse_center_errors = []
    ellipse_rx_errors = []
    ellipse_ry_errors = []
    ellipse_ious = []
    keypoint_errors = []
    pck_hits = 0
    pck_total = 0
    condition_totals: dict[str, int] = {}
    condition_matched: dict[str, int] = {}
    for row in rows:
        predictions = [prediction for prediction in row["predictions"] if prediction["score"] >= threshold]
        matches = greedy_matches(row["ground_truth"], predictions, iou_threshold)
        tp += len(matches)
        fp += len(predictions) - len(matches)
        fn += len(row["ground_truth"]) - len(matches)
        matched_gt = {gt_index for gt_index, _, _ in matches}
        for gt_index, gt in enumerate(row["ground_truth"]):
            for tag in gt.get("tags", []):
                condition_totals[tag] = condition_totals.get(tag, 0) + 1
                if gt_index in matched_gt:
                    condition_matched[tag] = condition_matched.get(tag, 0) + 1
        for gt_index, pred_index, iou in matches:
            gt = row["ground_truth"][gt_index]
            prediction = predictions[pred_index]
            ious.append(iou)
            if prediction.get("ellipse") is not None:
                gt_ellipse = gt["ellipse"]
                pred_ellipse = prediction["ellipse"]
                scale = max(1.0, math.sqrt(max(1.0, gt_ellipse[2] * 2.0 * gt_ellipse[3] * 2.0)))
                ellipse_center_errors.append(math.hypot(pred_ellipse[0] - gt_ellipse[0], pred_ellipse[1] - gt_ellipse[1]) / scale)
                ellipse_rx_errors.append(abs(pred_ellipse[2] - gt_ellipse[2]) / max(1.0, gt_ellipse[2]))
                ellipse_ry_errors.append(abs(pred_ellipse[3] - gt_ellipse[3]) / max(1.0, gt_ellipse[3]))
                ellipse_ious.append(ellipse_iou(gt_ellipse, pred_ellipse))
            pred_points = {point["name"]: point for point in prediction.get("keypoints", [])}
            scale = object_scale(gt)
            for gt_point in gt.get("keypoints", []):
                if int(gt_point.get("visibility", 0)) <= 0:
                    continue
                pred_point = pred_points.get(gt_point["name"])
                if pred_point is None:
                    continue
                error = math.hypot(pred_point["x"] - gt_point["x"], pred_point["y"] - gt_point["y"]) / scale
                keypoint_errors.append(error)
                pck_hits += int(error <= pck_threshold)
                pck_total += 1
    metrics = metrics_from_counts(tp, fp, fn)
    metrics.update(
        {
            "threshold": threshold,
            "iou_threshold": iou_threshold,
            "mean_matched_iou": sum(ious) / len(ious) if ious else None,
            "ellipse_center_error": sum(ellipse_center_errors) / len(ellipse_center_errors) if ellipse_center_errors else None,
            "ellipse_rx_error": sum(ellipse_rx_errors) / len(ellipse_rx_errors) if ellipse_rx_errors else None,
            "ellipse_ry_error": sum(ellipse_ry_errors) / len(ellipse_ry_errors) if ellipse_ry_errors else None,
            "ellipse_iou": sum(ellipse_ious) / len(ellipse_ious) if ellipse_ious else None,
            "keypoint_nme": sum(keypoint_errors) / len(keypoint_errors) if keypoint_errors else None,
            "keypoint_pck": pck_hits / pck_total if pck_total else None,
            "visible_keypoints": pck_total,
            "condition_recall": {
                tag: condition_matched.get(tag, 0) / total if total else None
                for tag, total in sorted(condition_totals.items())
            },
        }
    )
    return metrics


def draw_overlay(image: Image.Image, row: dict[str, Any], threshold: float) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for gt_index, gt in enumerate(row["ground_truth"]):
        draw.line(ellipse_polygon(gt["ellipse"]) + [ellipse_polygon(gt["ellipse"])[0]], fill=(0, 220, 0), width=2)
        x1, y1, x2, y2 = gt["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 170, 0), width=1)
        draw.text((x1, max(0, y1 - 12)), f"GT {gt_index}", fill=(0, 220, 0))
    for pred_index, prediction in enumerate([item for item in row["predictions"] if item["score"] >= threshold]):
        color = (255, 0, 0)
        if prediction.get("ellipse"):
            polygon = ellipse_polygon(prediction["ellipse"])
            draw.line(polygon + [polygon[0]], fill=color, width=2)
        x1, y1, x2, y2 = prediction["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 180, 0), width=1)
        for point in prediction.get("keypoints", []):
            x = float(point["x"])
            y = float(point["y"])
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 0))
        draw.text((x1, y2 + 2), f"P {pred_index} {prediction['score']:.2f}", fill=color)
    return image


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: Any, limit: int = 96) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return text[:limit].strip("._") or "sample"


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    config.setdefault("data", {})["split"] = args.split
    config["data"]["image_size"] = args.imgsz
    dataset = FaceKeypointDataset(
        data_root=args.data_root,
        split=args.split,
        metadata_path=config.get("data", {}).get("metadata_path"),
        image_size=args.imgsz,
        include_negative=bool(config.get("data", {}).get("include_negative", True)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_face_keypoint_batch)
    model = load_model(config, args.checkpoint, args.device, args.prediction_conf_min)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for image_tensor, targets in loader:
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
        image_tensor = image_tensor.to(args.device)
        with torch.no_grad():
            detections = model(image_tensor)[0]
        target = targets[0]
        row = {
            "image_id": str(target.get("image_id", "")),
            "ground_truth": target_to_ground_truth(target, image_tensor[0].detach().cpu()),
            "predictions": detections_to_predictions(detections),
            "image_tensor": image_tensor[0].detach().cpu(),
        }
        rows.append(row)
        for prediction in row["predictions"]:
            prediction_rows.append(
                {
                    "image_id": row["image_id"],
                    "score": prediction["score"],
                    "bbox": json.dumps(prediction["bbox"]),
                    "ellipse": json.dumps(prediction.get("ellipse")),
                    "keypoints": json.dumps(prediction.get("keypoints", [])),
                }
            )

    thresholds = parse_thresholds(args.thresholds, args.fixed_threshold)
    threshold_rows = [evaluate_threshold(rows, threshold, args.iou_threshold, args.keypoint_pck_threshold) for threshold in thresholds]
    best = max(threshold_rows, key=lambda item: (float(item["f1"]), float(item["recall"]), float(item["precision"])))
    ap_thresholds = [round(0.50 + index * 0.05, 2) for index in range(10)]
    ap_by_threshold = {f"AP{int(threshold * 100)}": average_precision(rows, threshold) for threshold in ap_thresholds}
    bbox_map = sum(ap_by_threshold.values()) / len(ap_by_threshold) if ap_by_threshold else 0.0
    summary = {
        "model": "dino_centerface_ellipse",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "images": len(rows),
        "gt_instances": sum(len(row["ground_truth"]) for row in rows),
        "prediction_conf_min": args.prediction_conf_min,
        "best_threshold": best,
        "bbox_map": bbox_map,
        "bbox_ap": ap_by_threshold,
        "output_dir": str(args.output_dir),
    }
    serializable_rows = [
        {key: value for key, value in row.items() if key != "image_tensor"}
        for row in rows
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(json.dumps(serializable_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "threshold_metrics.csv", threshold_rows)
    write_csv(args.output_dir / "prediction_rows.csv", prediction_rows)
    condition_rows = [
        {"condition": key, "recall": value}
        for key, value in dict(best.get("condition_recall", {})).items()
    ]
    write_csv(args.output_dir / "condition_recall.csv", condition_rows)

    overlay_rows = []
    for index, row in enumerate(rows[: args.max_visualizations]):
        overlay_path = args.output_dir / "overlays" / f"{index:03d}_{safe_name(row['image_id'])}.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_overlay(tensor_to_pil(row["image_tensor"]), row, float(best["threshold"])).save(overlay_path)
        overlay_rows.append({"image_id": row["image_id"], "overlay": str(overlay_path), "gt": len(row["ground_truth"])})
    write_csv(args.output_dir / "visualization_index.csv", overlay_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


if __name__ == "__main__":
    raise SystemExit(main())
