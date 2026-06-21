import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO Pose mAP, fixed-threshold detection metrics, and keypoint quality."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-error-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-error-images", action="store_true")
    parser.add_argument("--max-error-images", type=int, default=80)
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for image loading and overlays.") from error
    return cv2


def tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


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


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    objects = []
    if not label_path.exists():
        return objects
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 5 + len(KEYPOINT_NAMES) * 3:
            raise ValueError(f"Invalid YOLO pose label at {label_path}:{line_number}")
        xc, yc, bw, bh = values[1:5]
        box_width = bw * width
        box_height = bh * height
        cx = xc * width
        cy = yc * height
        keypoints = []
        raw_keypoints = values[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            x_value = raw_keypoints[index * 3]
            y_value = raw_keypoints[index * 3 + 1]
            visibility = int(raw_keypoints[index * 3 + 2])
            keypoints.append(
                {
                    "name": name,
                    "x": x_value * width if visibility else 0.0,
                    "y": y_value * height if visibility else 0.0,
                    "visibility": visibility,
                }
            )
        objects.append(
            {
                "bbox": [
                    cx - box_width / 2.0,
                    cy - box_height / 2.0,
                    cx + box_width / 2.0,
                    cy + box_height / 2.0,
                ],
                "keypoints": keypoints,
            }
        )
    return objects


def result_predictions(result: Any) -> list[dict[str, Any]]:
    boxes = tensor_to_list(result.boxes.xyxy) if result.boxes is not None else []
    scores = tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    keypoint_xy = tensor_to_list(result.keypoints.xy) if result.keypoints is not None else []
    keypoint_conf = tensor_to_list(getattr(result.keypoints, "conf", None)) if result.keypoints is not None else []
    predictions = []
    for index, box in enumerate(boxes):
        points = []
        xy_values = keypoint_xy[index] if index < len(keypoint_xy) else []
        conf_values = keypoint_conf[index] if index < len(keypoint_conf) else []
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            xy = xy_values[keypoint_index] if keypoint_index < len(xy_values) else [0.0, 0.0]
            confidence = conf_values[keypoint_index] if keypoint_index < len(conf_values) else None
            points.append(
                {
                    "name": name,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "score": None if confidence is None else float(confidence),
                }
            )
        predictions.append(
            {
                "bbox": [float(value) for value in box],
                "score": float(scores[index]) if index < len(scores) else 0.0,
                "keypoints": points,
            }
        )
    return predictions


def greedy_matches(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
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


def object_scale(gt: dict[str, Any]) -> float:
    box = gt["bbox"]
    return math.sqrt(max(1.0, bbox_area(box)))


def visible_keypoints(gt: dict[str, Any]) -> list[dict[str, Any]]:
    return [point for point in gt["keypoints"] if int(point["visibility"]) > 0]


def keypoint_errors(gt: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    scale = object_scale(gt)
    pred_by_name = {point["name"]: point for point in prediction.get("keypoints", [])}
    errors = []
    for gt_point in visible_keypoints(gt):
        pred_point = pred_by_name.get(gt_point["name"])
        if pred_point is None:
            continue
        distance = math.hypot(pred_point["x"] - gt_point["x"], pred_point["y"] - gt_point["y"])
        errors.append(
            {
                "name": gt_point["name"],
                "distance_px": distance,
                "normalized_error": distance / scale,
            }
        )
    return errors


def gt_condition_tags(gt: dict[str, Any], width: int, height: int, image_gt_count: int) -> list[str]:
    box = gt["bbox"]
    area_ratio = bbox_area(box) / max(1.0, float(width * height))
    tags = []
    if area_ratio < 0.01:
        tags.append("small_face")
    elif area_ratio < 0.05:
        tags.append("medium_face")
    else:
        tags.append("large_face")
    margin_x = width * 0.02
    margin_y = height * 0.02
    if box[0] <= margin_x or box[1] <= margin_y or box[2] >= width - margin_x or box[3] >= height - margin_y:
        tags.append("edge_face")
    visible_count = len(visible_keypoints(gt))
    if visible_count < len(KEYPOINT_NAMES):
        tags.append("missing_gt_keypoints")
    if visible_count <= 2:
        tags.append("very_sparse_keypoints")
    if image_gt_count >= 3:
        tags.append("crowded_image")
    return tags


def best_iou_for_gt(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return max(bbox_iou(gt["bbox"], prediction["bbox"]) for prediction in predictions)


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


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


def draw_error_image(
    cv2: Any,
    image_path: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for gt in ground_truth:
        box = [int(round(value)) for value in gt["bbox"]]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        for point in visible_keypoints(gt):
            cv2.circle(image, (int(round(point["x"])), int(round(point["y"]))), 3, (0, 255, 0), -1)
    for prediction in predictions:
        box = [int(round(value)) for value in prediction["bbox"]]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(
            image,
            f'{prediction["score"]:.2f}',
            (box[0], max(0, box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        for point in prediction.get("keypoints", []):
            cv2.circle(image, (int(round(point["x"])), int(round(point["y"]))), 3, (0, 0, 255), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def flatten_results_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): flatten_results_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [flatten_results_dict(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def metric_attr(obj: Any, name: str) -> float | None:
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_map_validation(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.prediction_conf_min,
        verbose=False,
        plots=False,
    )
    box = getattr(metrics, "box", None)
    pose = getattr(metrics, "pose", None)
    return {
        "box": {
            "map50_95": metric_attr(box, "map"),
            "map50": metric_attr(box, "map50"),
            "precision": metric_attr(box, "mp"),
            "recall": metric_attr(box, "mr"),
        },
        "pose": {
            "map50_95": metric_attr(pose, "map"),
            "map50": metric_attr(pose, "map50"),
            "precision": metric_attr(pose, "mp"),
            "recall": metric_attr(pose, "mr"),
        },
        "results_dict": flatten_results_dict(getattr(metrics, "results_dict", {})),
    }


def summarize_keypoints(
    keypoint_rows: list[dict[str, Any]],
    per_match_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not keypoint_rows:
        return {
            "visible_keypoints": 0,
            "nme": None,
            "pck_0.05": None,
            "pck_0.10": None,
            "mean_matched_object_nme": None,
            "per_keypoint": {},
        }
    errors = [float(row["normalized_error"]) for row in keypoint_rows]
    per_keypoint = {}
    for name in KEYPOINT_NAMES:
        subset = [float(row["normalized_error"]) for row in keypoint_rows if row["keypoint"] == name]
        if subset:
            per_keypoint[name] = {
                "count": len(subset),
                "nme": sum(subset) / len(subset),
                "pck_0.05": sum(error <= 0.05 for error in subset) / len(subset),
                "pck_0.10": sum(error <= 0.10 for error in subset) / len(subset),
            }
    object_errors = [
        float(row["mean_keypoint_error"])
        for row in per_match_rows
        if row.get("mean_keypoint_error") not in (None, "")
    ]
    return {
        "visible_keypoints": len(errors),
        "nme": sum(errors) / len(errors),
        "pck_0.05": sum(error <= 0.05 for error in errors) / len(errors),
        "pck_0.10": sum(error <= 0.10 for error in errors) / len(errors),
        "mean_matched_object_nme": sum(object_errors) / len(object_errors) if object_errors else None,
        "per_keypoint": per_keypoint,
    }


def category_summary(failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_failure_type = Counter(row["failure_type"] for row in failure_rows)
    by_tag: Counter[str] = Counter()
    by_failure_type_and_tag: Counter[str] = Counter()
    for row in failure_rows:
        tags = [tag for tag in str(row.get("tags", "")).split("|") if tag]
        for tag in tags:
            by_tag[tag] += 1
            by_failure_type_and_tag[f'{row["failure_type"]}:{tag}'] += 1
    return {
        "total_failures": len(failure_rows),
        "by_failure_type": dict(sorted(by_failure_type.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "by_failure_type_and_tag": dict(sorted(by_failure_type_and_tag.items())),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2 = import_cv2()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt first.") from error

    model = YOLO(str(args.weights))
    map_metrics = run_map_validation(model, args)
    images = image_paths(args.data_root, args.split)
    filtered_predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    ground_truth_by_image: dict[str, list[dict[str, Any]]] = {}
    image_size_by_image: dict[str, tuple[int, int]] = {}
    match_rows: list[dict[str, Any]] = []
    keypoint_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    mean_ious: list[float] = []
    error_images_written = 0

    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.prediction_conf_min,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(images, results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        predictions = [
            prediction
            for prediction in result_predictions(result)
            if prediction["score"] >= args.conf_threshold
        ]
        ground_truth = load_ground_truth(
            label_path_for_image(args.data_root, args.split, image_path),
            width,
            height,
        )
        matches = greedy_matches(ground_truth, predictions, args.iou_threshold)
        matched_gt = {gt_index for gt_index, _, _ in matches}
        matched_pred = {pred_index for _, pred_index, _ in matches}
        image_has_failure = False

        for gt_index, pred_index, iou in matches:
            gt = ground_truth[gt_index]
            prediction = predictions[pred_index]
            errors = keypoint_errors(gt, prediction)
            mean_error = sum(error["normalized_error"] for error in errors) / len(errors) if errors else None
            mean_ious.append(iou)
            match_rows.append(
                {
                    "image": str(image_path),
                    "gt_index": gt_index,
                    "prediction_index": pred_index,
                    "score": prediction["score"],
                    "iou": iou,
                    "visible_keypoints": len(errors),
                    "mean_keypoint_error": mean_error if mean_error is not None else "",
                    "tags": "|".join(gt_condition_tags(gt, width, height, len(ground_truth))),
                }
            )
            for error in errors:
                keypoint_rows.append(
                    {
                        "image": str(image_path),
                        "gt_index": gt_index,
                        "prediction_index": pred_index,
                        "keypoint": error["name"],
                        "distance_px": error["distance_px"],
                        "normalized_error": error["normalized_error"],
                    }
                )
            if mean_error is not None and mean_error > args.keypoint_error_threshold:
                image_has_failure = True
                failure_rows.append(
                    {
                        "image": str(image_path),
                        "failure_type": "keypoint_error",
                        "gt_index": gt_index,
                        "prediction_index": pred_index,
                        "score": prediction["score"],
                        "iou": iou,
                        "mean_keypoint_error": mean_error,
                        "tags": "|".join(gt_condition_tags(gt, width, height, len(ground_truth))),
                    }
                )

        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            image_has_failure = True
            best_iou = best_iou_for_gt(gt, predictions)
            failure_type = "localization_error" if best_iou >= 0.1 else "missed_detection"
            failure_rows.append(
                {
                    "image": str(image_path),
                    "failure_type": failure_type,
                    "gt_index": gt_index,
                    "prediction_index": "",
                    "score": "",
                    "iou": best_iou,
                    "mean_keypoint_error": "",
                    "tags": "|".join(gt_condition_tags(gt, width, height, len(ground_truth))),
                }
            )

        for pred_index, prediction in enumerate(predictions):
            if pred_index in matched_pred:
                continue
            image_has_failure = True
            failure_rows.append(
                {
                    "image": str(image_path),
                    "failure_type": "false_positive",
                    "gt_index": "",
                    "prediction_index": pred_index,
                    "score": prediction["score"],
                    "iou": "",
                    "mean_keypoint_error": "",
                    "tags": "negative_image" if not ground_truth else "",
                }
            )

        if args.save_error_images and image_has_failure and error_images_written < args.max_error_images:
            draw_error_image(
                cv2,
                image_path,
                args.output_dir / "error_images" / image_path.name,
                ground_truth,
                predictions,
            )
            error_images_written += 1

        filtered_predictions_by_image[image_path.name] = predictions
        ground_truth_by_image[image_path.name] = ground_truth
        image_size_by_image[image_path.name] = (width, height)

    tp = len(match_rows)
    fp = sum(
        max(0, len(filtered_predictions_by_image[image_name]) - len(
            [row for row in match_rows if Path(str(row["image"])).name == image_name]
        ))
        for image_name in filtered_predictions_by_image
    )
    fn = sum(
        max(0, len(ground_truth_by_image[image_name]) - len(
            [row for row in match_rows if Path(str(row["image"])).name == image_name]
        ))
        for image_name in ground_truth_by_image
    )
    fixed_threshold_metrics = metrics_from_counts(tp, fp, fn)
    fixed_threshold_metrics.update(
        {
            "threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "mean_matched_iou": sum(mean_ious) / len(mean_ious) if mean_ious else None,
        }
    )
    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "data_root": str(args.data_root),
        "split": args.split,
        "images": len(images),
        "ground_truth_objects": sum(len(objects) for objects in ground_truth_by_image.values()),
        "predicted_objects": sum(len(objects) for objects in filtered_predictions_by_image.values()),
        "map": map_metrics,
        "fixed_threshold_detection": fixed_threshold_metrics,
        "keypoints": summarize_keypoints(keypoint_rows, match_rows),
        "failure_categories": category_summary(failure_rows),
        "error_images_written": error_images_written,
    }
    (args.output_dir / f"{args.split}_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / f"{args.split}_matches.csv", match_rows)
    write_csv(args.output_dir / f"{args.split}_keypoint_errors.csv", keypoint_rows)
    write_csv(args.output_dir / f"{args.split}_failure_analysis.csv", failure_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
