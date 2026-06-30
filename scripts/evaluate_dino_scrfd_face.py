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
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_YOLO.evaluate_yolo_pose_metrics import (  # noqa: E402
    bbox_iou,
    greedy_matches,
    image_paths,
    label_path_for_image,
    load_ground_truth,
    object_scale,
    visible_keypoints,
)
from src.config_utils import load_yaml_config  # noqa: E402
from src.models import build_dino_scrfd_face_detector  # noqa: E402


KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize DINO-SCRFD face/keypoint predictions.")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-visualizations", type=int, default=30)
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for overlays.") from error
    return cv2


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


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_model(config: dict[str, Any], checkpoint_path: Path, device: str, prediction_conf_min: float):
    model = build_dino_scrfd_face_detector(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.postprocessor.score_threshold = prediction_conf_min
    model.eval()
    return model


def scale_predictions(
    detections: dict[str, torch.Tensor],
    *,
    input_size: int,
    original_width: int,
    original_height: int,
) -> list[dict[str, Any]]:
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    keypoints = detections["keypoints"].detach().cpu()
    scale_x = original_width / float(input_size)
    scale_y = original_height / float(input_size)
    predictions = []
    for index in range(boxes.shape[0]):
        box = boxes[index].clone()
        box[0::2] *= scale_x
        box[1::2] *= scale_y
        points = []
        for point_index, name in enumerate(KEYPOINT_NAMES):
            x = float(keypoints[index, point_index, 0]) * scale_x
            y = float(keypoints[index, point_index, 1]) * scale_y
            points.append({"name": name, "x": x, "y": y, "score": None})
        predictions.append(
            {
                "bbox": [float(value) for value in box.tolist()],
                "score": float(scores[index]),
                "keypoints": points,
            }
        )
    return predictions


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_threshold(
    rows: list[dict[str, Any]],
    threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    ious = []
    keypoint_errors = []
    visible_count = 0
    for row in rows:
        ground_truth = row["ground_truth"]
        predictions = [prediction for prediction in row["predictions"] if prediction["score"] >= threshold]
        matches = greedy_matches(ground_truth, predictions, iou_threshold)
        tp += len(matches)
        fp += len(predictions) - len(matches)
        fn += len(ground_truth) - len(matches)
        for gt_index, pred_index, iou in matches:
            ious.append(iou)
            scale = object_scale(ground_truth[gt_index])
            pred_points = {point["name"]: point for point in predictions[pred_index].get("keypoints", [])}
            for gt_point in visible_keypoints(ground_truth[gt_index]):
                pred_point = pred_points.get(gt_point["name"])
                if pred_point is None:
                    continue
                error = math.hypot(pred_point["x"] - gt_point["x"], pred_point["y"] - gt_point["y"]) / scale
                keypoint_errors.append(error)
                visible_count += 1
    metrics = metrics_from_counts(tp, fp, fn)
    metrics.update(
        {
            "threshold": threshold,
            "mean_iou": sum(ious) / len(ious) if ious else 0.0,
            "matched_instances": len(ious),
            "mean_keypoint_error_norm": sum(keypoint_errors) / len(keypoint_errors) if keypoint_errors else None,
            "visible_keypoints_matched": visible_count,
        }
    )
    return metrics


def draw_label(cv2: Any, image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, (origin[0], max(14, origin[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_box(cv2: Any, image: Any, box: list[float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    draw_label(cv2, image, label, (x1, y1 - 5), color)


def draw_points(cv2: Any, image: Any, points: list[dict[str, Any]], color: tuple[int, int, int]) -> None:
    for point in points:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        if x <= 0.0 and y <= 0.0:
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)
        draw_label(cv2, image, str(point.get("name", "")), (center[0] + 4, center[1] + 12), color)


def render_overlay(cv2: Any, row: dict[str, Any], output_path: Path, threshold: float) -> dict[str, Any]:
    image = cv2.imread(row["image_path"])
    if image is None:
        raise RuntimeError(f"Failed to read image: {row['image_path']}")
    ground_truth = row["ground_truth"]
    predictions = [prediction for prediction in row["predictions"] if prediction["score"] >= threshold]
    for gt_index, gt in enumerate(ground_truth):
        draw_box(cv2, image, gt["bbox"], (0, 220, 0), f"GT {gt_index}")
        draw_points(cv2, image, visible_keypoints(gt), (0, 220, 0))
    for pred_index, prediction in enumerate(predictions):
        best_iou = max([bbox_iou(prediction["bbox"], gt["bbox"]) for gt in ground_truth] or [0.0])
        color = (0, 0, 255) if best_iou >= 0.5 else (180, 0, 180)
        draw_box(cv2, image, prediction["bbox"], color, f"D {pred_index} {prediction['score']:.2f} IoU {best_iou:.2f}")
        draw_points(cv2, image, prediction.get("keypoints", []), color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {
        "image": row["image_path"],
        "visualization": str(output_path),
        "gt_count": len(ground_truth),
        "prediction_count": len(predictions),
    }


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


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    cards = []
    for row in rows:
        image_name = Path(row["visualization"]).name
        cards.append(
            f'<article class="card"><img src="overlays/{image_name}"><div>{Path(row["image"]).name}</div>'
            f'<div>GT {row["gt_count"]} / Pred {row["prediction_count"]}</div></article>'
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #171717; }}
h1 {{ font-size: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
.card {{ background: #fff; border: 1px solid #d7d7d2; border-radius: 6px; padding: 8px; }}
img {{ width: 100%; height: auto; display: block; }}
div {{ font-size: 12px; word-break: break-all; }}
</style></head><body><h1>{title}</h1><div class="grid">{''.join(cards)}</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(args.config)
    cv2 = import_cv2()
    model = load_model(config, args.checkpoint, args.device, args.prediction_conf_min)
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for image_path in image_paths(args.data_root, args.split):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        tensor = image_to_tensor(image, args.imgsz).to(args.device)
        with torch.no_grad():
            detections = model(tensor)[0]
        predictions = scale_predictions(
            detections,
            input_size=args.imgsz,
            original_width=original_width,
            original_height=original_height,
        )
        ground_truth = load_ground_truth(label_path_for_image(args.data_root, args.split, image_path), original_width, original_height)
        rows.append({"image_path": str(image_path), "ground_truth": ground_truth, "predictions": predictions})
        for prediction in predictions:
            prediction_records.append(
                {
                    "image": str(image_path),
                    "score": prediction["score"],
                    "bbox": json.dumps(prediction["bbox"]),
                    "keypoints": json.dumps(prediction["keypoints"]),
                }
            )

    thresholds = parse_thresholds(args.thresholds, args.fixed_threshold)
    threshold_rows = [evaluate_threshold(rows, threshold, args.iou_threshold) for threshold in thresholds]
    best = max(threshold_rows, key=lambda item: (float(item["f1"]), float(item["recall"]), float(item["precision"])))
    summary = {
        "model": "dino_scrfd",
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "images": len(rows),
        "gt_instances": sum(len(row["ground_truth"]) for row in rows),
        "prediction_conf_min": args.prediction_conf_min,
        "best_threshold": best,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "threshold_metrics.csv", threshold_rows)
    write_csv(args.output_dir / "prediction_rows.csv", prediction_records)
    overlay_rows = []
    for row in rows[: args.max_visualizations]:
        image_name = Path(row["image_path"]).name
        overlay_rows.append(render_overlay(cv2, row, args.output_dir / "overlays" / image_name, float(best["threshold"])))
    write_csv(args.output_dir / "visualization_index.csv", overlay_rows)
    write_html(args.output_dir / "index.html", overlay_rows, f"DINO-SCRFD {args.split} overlays")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
