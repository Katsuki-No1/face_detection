import argparse
import csv
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from evaluate_face_detections import (
    bbox_iou,
    evaluate,
    find_prediction,
    load_json,
    merge_records_by_image,
    prediction_records_by_key,
    select_ground_truth_regions,
    select_records_for_scope,
)


MIN_BOX_SIZES = (0.0, 30.0, 40.0, 50.0, 60.0, 80.0)
POST_NMS_IOUS = (None, 0.3, 0.4)
MAX_DETECTIONS = (None, 3, 5)

TARGET_REGION = "face_bbox"
BBOX_SCALE = 1.0
IOU_THRESHOLD = 0.3
COVERAGE_THRESHOLD = 0.5
MATCH_POLICY = "iou_or_coverage"
COMPARISON_SCOPE = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate face-label detections without bbox expansion and optimize with "
            "recall plus frame-level face-count metrics."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/ground_truth.json"),
        help="CVAT ground truth JSON.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/predictions"),
        help="Directory containing prediction JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="cvat_recall_constrained_i*.json",
        help="Prediction glob. Default: cvat_recall_constrained_i*.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/face_bbox_count_tuning_summary.csv"),
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--recommendations-output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/face_bbox_count_tuning_recommendations.json"),
        help="Output recommendations JSON.",
    )
    return parser.parse_args()


def parse_config(prediction_path: Path, predictions: dict[str, Any]) -> dict[str, Any]:
    model = predictions.get("model", {})
    return {
        "prediction_file": str(prediction_path),
        "input_size": model.get("input_size"),
        "score_threshold": model.get("score_threshold"),
        "nms_threshold": model.get("nms_threshold"),
        "mediapipe_crop_margin": model.get("mediapipe_crop_margin"),
    }


def box_size(detection: dict[str, Any]) -> float:
    bbox = detection["bbox"]
    width = max(0.0, float(bbox.get("width", float(bbox["xbr"]) - float(bbox["xtl"]))))
    height = max(0.0, float(bbox.get("height", float(bbox["ybr"]) - float(bbox["ytl"]))))
    return (width * height) ** 0.5


def post_nms(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(detections, key=lambda item: item.get("score", 0.0), reverse=True):
        if all(bbox_iou(detection["bbox"], existing["bbox"]) < threshold for existing in kept):
            kept.append(detection)
    return kept


def filtered_predictions(
    predictions: dict[str, Any],
    min_box_size: float,
    post_nms_iou: float | None,
    max_detections: int | None,
) -> dict[str, Any]:
    output = deepcopy(predictions)
    for record in output.get("records", []):
        detections = [
            detection
            for detection in record.get("detections", [])
            if box_size(detection) >= min_box_size
        ]
        if post_nms_iou is not None:
            detections = post_nms(detections, post_nms_iou)
        detections = sorted(detections, key=lambda item: item.get("score", 0.0), reverse=True)
        if max_detections is not None:
            detections = detections[:max_detections]
        record["detections"] = detections
    return output


def count_metrics(
    ground_truth: dict[str, Any],
    predictions: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_key = prediction_records_by_key(predictions)
    total_abs_error = 0
    total_over = 0
    total_under = 0
    exact = 0
    frames_with_over = 0
    frames_with_under = 0
    total_gt = 0
    total_pred = 0
    frames = 0

    for record in records:
        prediction = find_prediction(predictions_by_key, record)
        gt_count = len(select_ground_truth_regions(record, TARGET_REGION))
        pred_count = len([] if prediction is None else prediction.get("detections", []))
        diff = pred_count - gt_count
        frames += 1
        total_gt += gt_count
        total_pred += pred_count
        total_abs_error += abs(diff)
        if diff == 0:
            exact += 1
        if diff > 0:
            total_over += diff
            frames_with_over += 1
        elif diff < 0:
            total_under += -diff
            frames_with_under += 1

    return {
        "count_exact_frame_accuracy": exact / frames if frames else 0.0,
        "count_mae_per_frame": total_abs_error / frames if frames else 0.0,
        "count_over_per_gt": total_over / total_gt if total_gt else 0.0,
        "count_under_per_gt": total_under / total_gt if total_gt else 0.0,
        "frames_with_over_count": frames_with_over,
        "frames_with_under_count": frames_with_under,
        "predicted_faces": total_pred,
    }


def recall_count_score(row: dict[str, Any]) -> float:
    return (
        row["recall"]
        - 0.20 * row["count_over_per_gt"]
        - 0.10 * row["count_under_per_gt"]
        - 0.05 * row["false_positive"] / max(1, row["ground_truth_regions"])
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    records = merge_records_by_image(select_records_for_scope(ground_truth.get("records", []), COMPARISON_SCOPE))
    prediction_paths = sorted(
        path
        for path in args.predictions_dir.glob(args.pattern)
        if "_filtered_" not in path.stem
    )
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files matched: {args.predictions_dir / args.pattern}")

    rows = []
    for prediction_path in prediction_paths:
        predictions = load_json(prediction_path)
        config = parse_config(prediction_path, predictions)
        for min_box_size, post_nms_iou, max_detections in product(
            MIN_BOX_SIZES, POST_NMS_IOUS, MAX_DETECTIONS
        ):
            candidate = filtered_predictions(predictions, min_box_size, post_nms_iou, max_detections)
            summary, _ = evaluate(
                ground_truth,
                candidate,
                TARGET_REGION,
                IOU_THRESHOLD,
                COMPARISON_SCOPE,
                BBOX_SCALE,
                MATCH_POLICY,
                COVERAGE_THRESHOLD,
            )
            counts = count_metrics(ground_truth, candidate, records)
            row = {
                **config,
                "target_region": TARGET_REGION,
                "bbox_scale": BBOX_SCALE,
                "iou_threshold": IOU_THRESHOLD,
                "match_policy": f"iou>={IOU_THRESHOLD} OR face_coverage>={COVERAGE_THRESHOLD}",
                "min_box_size": min_box_size,
                "post_nms_iou": "" if post_nms_iou is None else post_nms_iou,
                "max_detections": "" if max_detections is None else max_detections,
                "images": summary["images"],
                "ground_truth_regions": summary["ground_truth_regions"],
                "detections": summary["detections"],
                "true_positive": summary["true_positive"],
                "false_positive": summary["false_positive"],
                "false_negative": summary["false_negative"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "mean_iou": summary["mean_iou"],
                "frame_macro_recall": summary["frame_macro_recall"],
                "frame_macro_precision": summary["frame_macro_precision"],
                "frames_with_false_positive": summary["frames_with_false_positive"],
                "frames_with_false_negative": summary["frames_with_false_negative"],
                **counts,
            }
            row["recall_count_score"] = recall_count_score(row)
            rows.append(row)

    write_csv(args.output, rows)

    best_recall = max(rows, key=lambda row: (row["recall"], row["recall_count_score"], row["precision"]))
    practical = [
        row
        for row in rows
        if row["recall"] >= 0.70
        and row["count_over_per_gt"] <= 0.75
        and row["precision"] >= 0.45
    ]
    best_practical = max(practical, key=lambda row: (row["recall_count_score"], row["recall"], row["precision"])) if practical else None
    best_score = max(rows, key=lambda row: (row["recall_count_score"], row["recall"], row["precision"]))
    recommendations = {
        "policy": (
            "Use face_bbox annotations, bbox_scale=1.0, and optimize recall while penalizing "
            "over-counted faces and false positives."
        ),
        "best_recall": best_recall,
        "best_recall_count_score": best_score,
        "best_practical_recall_count": best_practical,
        "outputs": {"summary": str(args.output)},
    }
    write_json(args.recommendations_output, recommendations)
    print(f"summary: {args.output}")
    print(f"recommendations: {args.recommendations_output}")
    best = best_practical or best_score
    print(
        "best:",
        f"input={best['input_size']}",
        f"score={best['score_threshold']}",
        f"nms={best['nms_threshold']}",
        f"min_size={best['min_box_size']}",
        f"post_nms={best['post_nms_iou']}",
        f"max_det={best['max_detections']}",
        f"recall={best['recall']}",
        f"precision={best['precision']}",
        f"count_over_per_gt={best['count_over_per_gt']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
