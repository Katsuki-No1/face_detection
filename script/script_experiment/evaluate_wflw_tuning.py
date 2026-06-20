import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from evaluate_face_detections import bbox_iou, scale_bbox


ATTRIBUTE_NAMES = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")
WFLW_LEFT_EYE = tuple(range(60, 68))
WFLW_RIGHT_EYE = tuple(range(68, 76))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize WFLW detection, attribute, and eye-landmark tuning results."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/wflw_ground_truth_test.json"),
        help="WFLW ground truth JSON.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/predictions"),
        help="Directory containing wflw tuning prediction JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="wflw_tuning_*.json",
        help="Prediction filename glob. Default: wflw_tuning_*.json.",
    )
    parser.add_argument(
        "--bbox-scales",
        nargs="+",
        type=float,
        default=[0.9, 1.0, 1.1, 1.2],
        help="Prediction bbox scales to evaluate.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for face matching. Default: 0.5.",
    )
    parser.add_argument(
        "--detection-output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/wflw_tuning_detection_summary.csv"),
        help="Output detection summary CSV.",
    )
    parser.add_argument(
        "--attribute-output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/wflw_tuning_attribute_summary.csv"),
        help="Output attribute summary CSV.",
    )
    parser.add_argument(
        "--eye-output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/wflw_tuning_eye_landmark_summary.csv"),
        help="Output eye landmark summary CSV.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def prediction_config(predictions: dict[str, Any], path: Path) -> dict[str, Any]:
    model = predictions.get("model", {})
    config = {
        "prediction_file": str(path),
        "input_size": model.get("input_size"),
        "score_threshold": model.get("score_threshold"),
        "nms_threshold": model.get("nms_threshold"),
        "mediapipe_crop_margin": model.get("mediapipe_crop_margin"),
    }
    if all(value is not None for key, value in config.items() if key != "prediction_file"):
        return config

    match = re.search(r"i(\d+)_s([0-9.]+)_n([0-9.]+)_m([0-9.]+)", path.stem)
    if match:
        config.update(
            {
                "input_size": int(match.group(1)),
                "score_threshold": float(match.group(2)),
                "nms_threshold": float(match.group(3)),
                "mediapipe_crop_margin": float(match.group(4)),
            }
        )
    return config


def group_ground_truth(ground_truth: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in ground_truth.get("records", []):
        key = (record["image_stem"], record["image_name"])
        image = grouped.setdefault(
            key,
            {
                "image_stem": record["image_stem"],
                "image_name": record["image_name"],
                "width": record["width"],
                "height": record["height"],
                "faces": [],
            },
        )
        landmarks = []
        for shape in record.get("simple_landmarks", []):
            if shape.get("label") == "wflw_98pt":
                landmarks = shape.get("points", [])
                break
        for head in record.get("heads", []):
            image["faces"].append(
                {
                    "bbox": head["bbox"],
                    "attributes": record.get("attributes", {}),
                    "landmarks": landmarks,
                }
            )
    return grouped


def group_predictions(predictions: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in predictions.get("records", []):
        key = (record["image_stem"], record["image_name"])
        grouped.setdefault(key, record.get("detections", []))
    return grouped


def match_faces(
    faces: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    bbox_scale: float,
    iou_threshold: float,
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates = []
    for face_index, face in enumerate(faces):
        for detection_index, detection in enumerate(detections):
            pred_bbox = scale_bbox(detection["bbox"], bbox_scale, image_width, image_height)
            iou = bbox_iou(face["bbox"], pred_bbox)
            if iou >= iou_threshold:
                candidates.append((iou, face_index, detection_index))

    candidates.sort(reverse=True)
    matched_faces = set()
    matched_detections = set()
    matches = []
    for iou, face_index, detection_index in candidates:
        if face_index in matched_faces or detection_index in matched_detections:
            continue
        matched_faces.add(face_index)
        matched_detections.add(detection_index)
        matches.append(
            {
                "face_index": face_index,
                "detection_index": detection_index,
                "iou": iou,
            }
        )
    return matches, matched_faces, matched_detections


def point_center(points: list[dict[str, float]], indices: tuple[int, ...]) -> dict[str, float] | None:
    selected = [points[index] for index in indices if index < len(points)]
    if not selected:
        return None
    return {
        "x": sum(point["x"] for point in selected) / len(selected),
        "y": sum(point["y"] for point in selected) / len(selected),
    }


def wflw_eye_points(face: dict[str, Any]) -> list[dict[str, float]]:
    landmarks = face.get("landmarks", [])
    points = []
    left = point_center(landmarks, WFLW_LEFT_EYE)
    right = point_center(landmarks, WFLW_RIGHT_EYE)
    if left is not None:
        points.append({"label": "left_eye", **left})
    if right is not None:
        points.append({"label": "right_eye", **right})
    return points


def predicted_eye_points(detection: dict[str, Any], source: str) -> list[dict[str, float]]:
    if source == "mediapipe":
        return detection.get("mediapipe_eye_landmarks", [])
    if source == "scrfd":
        return [
            point
            for point in detection.get("scrfd_landmarks_5pt", [])
            if point.get("label") in {"left_eye", "right_eye"}
        ]
    if source == "fallback":
        mediapipe = detection.get("mediapipe_eye_landmarks", [])
        if mediapipe:
            return mediapipe
        return predicted_eye_points(detection, "scrfd")
    raise ValueError(f"Unsupported eye source: {source}")


def distance(first: dict[str, float], second: dict[str, float]) -> float:
    return math.hypot(first["x"] - second["x"], first["y"] - second["y"])


def match_points(gt_points: list[dict[str, float]], pred_points: list[dict[str, float]]) -> list[float]:
    candidates = []
    for gt_index, gt_point in enumerate(gt_points):
        for pred_index, pred_point in enumerate(pred_points):
            candidates.append((distance(gt_point, pred_point), gt_index, pred_index))
    candidates.sort()
    matched_gt = set()
    matched_pred = set()
    errors = []
    for error, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        errors.append(error)
    return errors


def face_diagonal(face: dict[str, Any]) -> float:
    bbox = face["bbox"]
    return max(1.0, math.hypot(bbox["width"], bbox["height"]))


def evaluate_prediction(
    ground_truth_by_image: dict[tuple[str, str], dict[str, Any]],
    predictions_by_image: dict[tuple[str, str], list[dict[str, Any]]],
    config: dict[str, Any],
    bbox_scale: float,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = {
        "images": 0,
        "ground_truth_regions": 0,
        "detections": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "matched_iou_sum": 0.0,
        "images_without_detection": 0,
    }
    attr_totals = {
        name: {"positive": 0, "matched_positive": 0, "negative": 0, "matched_negative": 0}
        for name in ATTRIBUTE_NAMES
    }
    eye_totals = {
        source: {
            "matched_faces": 0,
            "gt_eye_points": 0,
            "pred_eye_points": 0,
            "matched_eye_points": 0,
            "pixel_error_sum": 0.0,
            "normalized_error_sum": 0.0,
        }
        for source in ("mediapipe", "scrfd", "fallback")
    }

    for key, image in ground_truth_by_image.items():
        faces = image["faces"]
        detections = predictions_by_image.get(key, [])
        matches, matched_faces, matched_detections = match_faces(
            faces,
            detections,
            bbox_scale,
            iou_threshold,
            image["width"],
            image["height"],
        )

        totals["images"] += 1
        totals["ground_truth_regions"] += len(faces)
        totals["detections"] += len(detections)
        totals["true_positive"] += len(matches)
        totals["false_positive"] += len(detections) - len(matched_detections)
        totals["false_negative"] += len(faces) - len(matched_faces)
        totals["matched_iou_sum"] += sum(match["iou"] for match in matches)
        if not detections:
            totals["images_without_detection"] += 1

        for face_index, face in enumerate(faces):
            matched = face_index in matched_faces
            for name in ATTRIBUTE_NAMES:
                if face.get("attributes", {}).get(name, 0):
                    attr_totals[name]["positive"] += 1
                    attr_totals[name]["matched_positive"] += int(matched)
                else:
                    attr_totals[name]["negative"] += 1
                    attr_totals[name]["matched_negative"] += int(matched)

        for match in matches:
            face = faces[match["face_index"]]
            detection = detections[match["detection_index"]]
            gt_eye_points = wflw_eye_points(face)
            normalizer = face_diagonal(face)
            for source, source_totals in eye_totals.items():
                pred_points = predicted_eye_points(detection, source)
                errors = match_points(gt_eye_points, pred_points)
                source_totals["matched_faces"] += 1
                source_totals["gt_eye_points"] += len(gt_eye_points)
                source_totals["pred_eye_points"] += len(pred_points)
                source_totals["matched_eye_points"] += len(errors)
                source_totals["pixel_error_sum"] += sum(errors)
                source_totals["normalized_error_sum"] += sum(error / normalizer for error in errors)

    tp = totals["true_positive"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    detection_row = {
        **config,
        "bbox_scale": bbox_scale,
        "iou_threshold": iou_threshold,
        "images": totals["images"],
        "ground_truth_regions": totals["ground_truth_regions"],
        "detections": totals["detections"],
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": totals["matched_iou_sum"] / tp if tp else 0.0,
        "images_without_detection": totals["images_without_detection"],
    }

    attribute_rows = []
    for name, values in attr_totals.items():
        positive = values["positive"]
        negative = values["negative"]
        attribute_rows.append(
            {
                **config,
                "bbox_scale": bbox_scale,
                "attribute": name,
                "positive_count": positive,
                "positive_recall": values["matched_positive"] / positive if positive else 0.0,
                "negative_count": negative,
                "negative_recall": values["matched_negative"] / negative if negative else 0.0,
                "recall_gap_positive_minus_negative": (
                    (values["matched_positive"] / positive if positive else 0.0)
                    - (values["matched_negative"] / negative if negative else 0.0)
                ),
            }
        )

    eye_rows = []
    for source, values in eye_totals.items():
        matched_points = values["matched_eye_points"]
        gt_points = values["gt_eye_points"]
        eye_rows.append(
            {
                **config,
                "bbox_scale": bbox_scale,
                "eye_source": source,
                "matched_faces": values["matched_faces"],
                "gt_eye_points": gt_points,
                "pred_eye_points": values["pred_eye_points"],
                "matched_eye_points": matched_points,
                "eye_point_recall_on_matched_faces": matched_points / gt_points if gt_points else 0.0,
                "mean_pixel_error": values["pixel_error_sum"] / matched_points if matched_points else 0.0,
                "mean_normalized_error": values["normalized_error_sum"] / matched_points
                if matched_points
                else 0.0,
            }
        )

    return detection_row, attribute_rows, eye_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    ground_truth_by_image = group_ground_truth(load_json(args.ground_truth))
    detection_rows = []
    attribute_rows = []
    eye_rows = []

    prediction_paths = sorted(args.predictions_dir.glob(args.pattern))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files matched: {args.predictions_dir / args.pattern}")

    for prediction_path in prediction_paths:
        predictions = load_json(prediction_path)
        config = prediction_config(predictions, prediction_path)
        predictions_by_image = group_predictions(predictions)
        for bbox_scale in args.bbox_scales:
            detection_row, attr, eye = evaluate_prediction(
                ground_truth_by_image,
                predictions_by_image,
                config,
                bbox_scale,
                args.iou_threshold,
            )
            detection_rows.append(detection_row)
            attribute_rows.extend(attr)
            eye_rows.extend(eye)

    write_csv(args.detection_output, detection_rows)
    write_csv(args.attribute_output, attribute_rows)
    write_csv(args.eye_output, eye_rows)
    print(f"detection: {args.detection_output}")
    print(f"attribute: {args.attribute_output}")
    print(f"eye: {args.eye_output}")
    best = max(detection_rows, key=lambda row: (row["recall"], row["f1"]))
    print(
        "best_recall:",
        f"input={best['input_size']}",
        f"score={best['score_threshold']}",
        f"nms={best['nms_threshold']}",
        f"margin={best['mediapipe_crop_margin']}",
        f"scale={best['bbox_scale']}",
        f"recall={best['recall']}",
        f"precision={best['precision']}",
        f"f1={best['f1']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
