import argparse
import csv
import json
import random
import subprocess
import sys
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluate_eye_landmarks import (
    bbox_diagonal,
    ground_truth_eye_points,
    match_points,
    predicted_eye_points,
)
from evaluate_face_detections import (
    bbox_iou,
    evaluate,
    find_prediction,
    greedy_match,
    load_json,
    merge_records_by_image,
    prediction_records_by_key,
    select_ground_truth_regions,
    select_records_for_scope,
)
from visualize_detections import safe_name, visualize_record


INPUT_SIZES = (640, 960)
SCORE_THRESHOLDS = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
NMS_THRESHOLDS = (0.3, 0.4, 0.5)
MEDIAPIPE_CROP_MARGINS = (0.15, 0.3, 0.5)
BBOX_SCALES = (1.5, 1.8, 2.0, 2.2)

IOU_THRESHOLD = 0.3
COVERAGE_THRESHOLD = 0.5
MATCH_POLICY = "iou_or_coverage"
TARGET_REGION = "head"
COMPARISON_SCOPE = "all"
DUPLICATE_IOU_THRESHOLD = 0.5
EYE_ERROR_LIMIT = 0.0206
PRECISION_FLOOR = 0.4
DETECTIONS_PER_GT_LIMIT = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and summarize CVAT recall-constrained SCRFD/MediaPipe tuning, "
            "including detection metrics, eye metrics, duplicate analysis, and overlays."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/ground_truth.json"),
        help="CVAT ground truth JSON.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/scrfd/det_10g.onnx"),
        help="SCRFD ONNX model path.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/predictions"),
        help="Directory for generated prediction JSON files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports"),
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/overlays"),
        help="Directory for generated overlay images.",
    )
    parser.add_argument(
        "--prefix",
        default="cvat_recall_constrained",
        help="Output filename/directory prefix.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit passed to detection and evaluation.",
    )
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Run SCRFD/MediaPipe for missing prediction files.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Regenerate prediction files even when they already exist.",
    )
    parser.add_argument(
        "--skip-mediapipe",
        action="store_true",
        help="Pass --skip-mediapipe to prediction generation.",
    )
    parser.add_argument(
        "--overlay-random-count",
        type=int,
        default=20,
        help="Representative random overlays per selected condition. Default: 20.",
    )
    parser.add_argument(
        "--overlay-seed",
        type=int,
        default=20260613,
        help="Random seed for representative overlays.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Progress interval passed to run_scrfd_mediapipe.py. Default: 0.",
    )
    parser.add_argument(
        "--grid-preset",
        choices=["full", "smoke"],
        default="full",
        help="Use the full fixed grid, or a two-condition smoke grid. Default: full.",
    )
    return parser.parse_args()


def config_name(prefix: str, input_size: int, score: float, nms: float, margin: float) -> str:
    return f"{prefix}_i{input_size}_s{score:g}_n{nms:g}_m{margin:g}"


def prediction_path(args: argparse.Namespace, input_size: int, score: float, nms: float, margin: float) -> Path:
    return args.predictions_dir / f"{config_name(args.prefix, input_size, score, nms, margin)}.json"


def run_prediction(
    args: argparse.Namespace,
    input_size: int,
    score: float,
    nms: float,
    margin: float,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        "script/script_model/run_scrfd_mediapipe.py",
        "--ground-truth",
        str(args.ground_truth),
        "--model",
        str(args.model),
        "--output",
        str(output_path),
        "--input-size",
        str(input_size),
        "--score-threshold",
        str(score),
        "--nms-threshold",
        str(nms),
        "--mediapipe-crop-margin",
        str(margin),
        "--print-every",
        str(args.print_every),
    ]
    if args.max_images is not None:
        command.extend(["--max-images", str(args.max_images)])
    if args.skip_mediapipe:
        command.append("--skip-mediapipe")
    subprocess.run(command, check=True)


def load_or_generate_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    if args.grid_preset == "smoke":
        grid = (
            (640, 0.05, 0.4, 0.15),
            (640, 0.1, 0.4, 0.15),
        )
    else:
        grid = product(INPUT_SIZES, SCORE_THRESHOLDS, NMS_THRESHOLDS, MEDIAPIPE_CROP_MARGINS)

    for input_size, score, nms, margin in grid:
        path = prediction_path(args, input_size, score, nms, margin)
        if args.force_run or (args.run_missing and not path.is_file()):
            print(f"generating: {path}")
            run_prediction(args, input_size, score, nms, margin, path)
        if path.is_file():
            configs.append(
                {
                    "condition_name": config_name(args.prefix, input_size, score, nms, margin),
                    "prediction_file": str(path),
                    "input_size": input_size,
                    "score_threshold": score,
                    "nms_threshold": nms,
                    "mediapipe_crop_margin": margin,
                }
            )
    if not configs:
        raise FileNotFoundError(
            "No prediction files found. Re-run with --run-missing to generate the tuning grid."
        )
    return configs


def duplicate_pairs(detections: list[dict[str, Any]], threshold: float = DUPLICATE_IOU_THRESHOLD) -> int:
    count = 0
    for first, second in combinations(detections, 2):
        if bbox_iou(first["bbox"], second["bbox"]) >= threshold:
            count += 1
    return count


def duplicate_analysis(
    records: list[dict[str, Any]],
    predictions_by_key: dict[tuple[str, str, str], dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        prediction = find_prediction(predictions_by_key, record)
        detections = [] if prediction is None else prediction.get("detections", [])
        dupes = duplicate_pairs(detections)
        rows.append(
            {
                **config,
                "annotation_kind": record.get("annotation_kind", ""),
                "annotation_mode": record.get("annotation_mode", ""),
                "image_stem": record.get("image_stem", ""),
                "image_name": record.get("image_name", ""),
                "detection_count": len(detections),
                "duplicate_pairs": dupes,
            }
        )
    return rows


def eye_summary_for_matches(
    records: list[dict[str, Any]],
    predictions_by_key: dict[tuple[str, str, str], dict[str, Any]],
    bbox_scale: float,
) -> dict[str, Any]:
    totals = {
        "matched_faces": 0,
        "gt_eye_points": 0,
        "pred_eye_points": 0,
        "matched_eye_points": 0,
        "pixel_error_sum": 0.0,
        "normalized_error_sum": 0.0,
    }

    for record in records:
        prediction = find_prediction(predictions_by_key, record)
        detections = [] if prediction is None else prediction.get("detections", [])
        heads = select_ground_truth_regions(record, TARGET_REGION)
        matches, _, _, _ = greedy_match(
            heads,
            detections,
            IOU_THRESHOLD,
            bbox_scale,
            match_policy=MATCH_POLICY,
            coverage_threshold=COVERAGE_THRESHOLD,
            image_width=record.get("width"),
            image_height=record.get("height"),
        )
        for match in matches:
            head = heads[match["ground_truth_index"]]
            detection = detections[match["prediction_index"]]
            gt_points = ground_truth_eye_points(record, head["bbox"])
            pred_points = predicted_eye_points(detection)
            point_matches = match_points(gt_points, pred_points)
            normalizer = max(bbox_diagonal(head["bbox"]), 1.0)
            totals["matched_faces"] += 1
            totals["gt_eye_points"] += len(gt_points)
            totals["pred_eye_points"] += len(pred_points)
            totals["matched_eye_points"] += len(point_matches)
            totals["pixel_error_sum"] += sum(point["pixel_error"] for point in point_matches)
            totals["normalized_error_sum"] += sum(
                point["pixel_error"] / normalizer for point in point_matches
            )

    matched = totals["matched_eye_points"]
    gt_points = totals["gt_eye_points"]
    return {
        **totals,
        "eye_point_recall_on_matched_faces": matched / gt_points if gt_points else 0.0,
        "mean_pixel_error": totals["pixel_error_sum"] / matched if matched else 0.0,
        "mean_normalized_error": totals["normalized_error_sum"] / matched if matched else 0.0,
    }


def f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denom if denom else 0.0


def build_rows(
    ground_truth: dict[str, Any],
    records: list[dict[str, Any]],
    configs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detection_rows = []
    eye_rows = []
    duplicate_rows = []
    predictions_cache = {}

    for config in configs:
        predictions = load_json(Path(config["prediction_file"]))
        predictions_cache[config["prediction_file"]] = predictions
        predictions_by_key = prediction_records_by_key(predictions)
        duplicate_rows.extend(duplicate_analysis(records, predictions_by_key, config))

        for bbox_scale in BBOX_SCALES:
            summary, _ = evaluate(
                ground_truth,
                predictions,
                TARGET_REGION,
                IOU_THRESHOLD,
                COMPARISON_SCOPE,
                bbox_scale,
                MATCH_POLICY,
                COVERAGE_THRESHOLD,
            )
            gt_count = summary["ground_truth_regions"]
            detections_per_gt = summary["detections"] / gt_count if gt_count else 0.0
            row = {
                **config,
                "bbox_scale": bbox_scale,
                "match_policy": f"iou>={IOU_THRESHOLD} OR head_coverage>={COVERAGE_THRESHOLD}",
                "images": summary["images"],
                "ground_truth_regions": gt_count,
                "detections": summary["detections"],
                "true_positive": summary["true_positive"],
                "false_positive": summary["false_positive"],
                "false_negative": summary["false_negative"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "f2": f_score(summary["precision"], summary["recall"], beta=2.0),
                "mean_iou": summary["mean_iou"],
                "frame_macro_recall": summary["frame_macro_recall"],
                "frame_macro_precision": summary["frame_macro_precision"],
                "frames_with_false_positive": summary["frames_with_false_positive"],
                "frames_with_false_negative": summary["frames_with_false_negative"],
                "detections_per_gt": detections_per_gt,
            }
            detection_rows.append(row)

            eye = eye_summary_for_matches(records, predictions_by_key, bbox_scale)
            eye_rows.append(
                {
                    **config,
                    "bbox_scale": bbox_scale,
                    **eye,
                }
            )

    return detection_rows, eye_rows, duplicate_rows, predictions_cache


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["prediction_file"],
        row["input_size"],
        row["score_threshold"],
        row["nms_threshold"],
        row["mediapipe_crop_margin"],
        row["bbox_scale"],
    )


def attach_eye_metrics(rows: list[dict[str, Any]], eye_rows: list[dict[str, Any]]) -> None:
    eye_by_key = {row_key(row): row for row in eye_rows}
    for row in rows:
        eye = eye_by_key.get(row_key(row), {})
        row["eye_point_recall_on_matched_faces"] = eye.get("eye_point_recall_on_matched_faces", 0.0)
        row["mean_eye_pixel_error"] = eye.get("mean_pixel_error", 0.0)
        row["mean_eye_normalized_error"] = eye.get("mean_normalized_error", 0.0)


def select_recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No detection rows available.")
    best_recall_raw = max(rows, key=lambda row: (row["recall"], row["f2"], row["precision"]))
    constrained = [
        row
        for row in rows
        if row["detections_per_gt"] <= DETECTIONS_PER_GT_LIMIT
        and row["precision"] >= PRECISION_FLOOR
    ]
    if not constrained:
        constrained = rows

    max_recall = max(row["recall"] for row in constrained)
    near_best = [row for row in constrained if row["recall"] >= max_recall - 0.02]
    best_constrained = max(
        near_best,
        key=lambda row: (
            row["recall"],
            row["f2"],
            -row["frames_with_false_positive"],
            row["precision"],
            -row["detections_per_gt"],
        ),
    )
    best_f2 = max(rows, key=lambda row: (row["f2"], row["recall"], row["precision"]))
    eye_safe = [
        row
        for row in rows
        if row["detections_per_gt"] <= DETECTIONS_PER_GT_LIMIT
        and row["precision"] >= PRECISION_FLOOR
        and row["mean_eye_normalized_error"] <= EYE_ERROR_LIMIT
    ]
    best_eye_safe = max(eye_safe, key=lambda row: (row["recall"], row["f2"], row["precision"])) if eye_safe else None

    return {
        "selection_policy": (
            "Recall constrained by detections_per_gt<=2.0 and precision>=0.4. "
            "Eye error is reported separately because making it a hard constraint can "
            "select very low-recall conditions."
        ),
        "best_constrained": best_constrained,
        "best_recall_raw": best_recall_raw,
        "best_f2": best_f2,
        "best_eye_safe": best_eye_safe,
        "acceptance": {
            "target_recall": 0.705,
            "precision_floor": PRECISION_FLOOR,
            "detections_per_gt_limit": DETECTIONS_PER_GT_LIMIT,
            "eye_mean_normalized_error_limit": EYE_ERROR_LIMIT,
            "best_constrained_passes": (
                best_constrained["recall"] >= 0.705
                and best_constrained["precision"] >= PRECISION_FLOOR
                and best_constrained["detections_per_gt"] <= DETECTIONS_PER_GT_LIMIT
            ),
            "best_eye_safe_available": best_eye_safe is not None,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for overlay generation.") from error
    return cv2


def select_overlay_records(
    records: list[dict[str, Any]],
    predictions_by_key: dict[tuple[str, str, str], dict[str, Any]],
    bbox_scale: float,
    category: str,
    random_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = []
    rng = random.Random(seed)
    for record in records:
        prediction = find_prediction(predictions_by_key, record)
        detections = [] if prediction is None else prediction.get("detections", [])
        heads = select_ground_truth_regions(record, TARGET_REGION)
        matches, tp, fp, fn = greedy_match(
            heads,
            detections,
            IOU_THRESHOLD,
            bbox_scale,
            match_policy=MATCH_POLICY,
            coverage_threshold=COVERAGE_THRESHOLD,
            image_width=record.get("width"),
            image_height=record.get("height"),
        )
        dupes = duplicate_pairs(detections)
        mean_iou = sum(match["iou"] for match in matches) / len(matches) if matches else 0.0
        metric = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "duplicate_pairs": dupes,
            "mean_iou": mean_iou,
        }
        keep = False
        if category == "all_errors":
            keep = fp > 0 or fn > 0
        elif category == "random":
            keep = True
        elif category == "false_negative_cases":
            keep = fn > 0
        elif category == "false_positive_cases":
            keep = fp > 0
        elif category == "duplicate_cases":
            keep = dupes > 0
        if keep:
            candidates.append((record, metric))

    if category == "random":
        rng.shuffle(candidates)
        candidates = candidates[:random_count]
    elif category == "duplicate_cases":
        candidates.sort(key=lambda item: (item[1]["duplicate_pairs"], item[1]["fp"]), reverse=True)
        candidates = candidates[: max(random_count, 20)]
    elif category in {"false_negative_cases", "false_positive_cases"}:
        candidates.sort(key=lambda item: (item[1]["fn"], item[1]["fp"]), reverse=True)
        candidates = candidates[: max(random_count, 20)]

    for record, metric in candidates:
        record["_overlay_metric"] = metric
    return [record for record, _ in candidates]


def write_overlays_for_condition(
    args: argparse.Namespace,
    cv2: Any,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    condition_label: str,
    predictions: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions_by_key = prediction_records_by_key(predictions)
    bbox_scale = condition["bbox_scale"]
    categories = ["all_errors", "random"]
    if condition_label == "best_constrained":
        categories.extend(["false_negative_cases", "false_positive_cases", "duplicate_cases"])
    elif condition_label == "best_recall_raw":
        categories.extend(["false_positive_cases", "duplicate_cases"])

    overlay_rows = []
    for category in categories:
        selected = select_overlay_records(
            records,
            predictions_by_key,
            bbox_scale,
            category,
            args.overlay_random_count,
            args.overlay_seed,
        )
        output_dir = args.overlays_dir / f"{args.prefix}_{condition_label}" / category
        viz_args = SimpleNamespace(
            output_dir=output_dir,
            bbox_scale=bbox_scale,
            iou_threshold=IOU_THRESHOLD,
            match_policy=MATCH_POLICY,
            coverage_threshold=COVERAGE_THRESHOLD,
            only_errors=False,
            show_bbox_values=True,
        )
        for record in selected:
            key = (record.get("annotation_kind", ""), record.get("image_stem", ""), record.get("image_name", ""))
            row = visualize_record(record, predictions_by_key.get(key), viz_args, cv2)
            if row is None:
                continue
            metric = record.pop("_overlay_metric", {})
            overlay_rows.append(
                {
                    "output_path": str(output_dir / safe_name(record)),
                    "condition_name": condition.get("condition_name", ""),
                    "condition_label": condition_label,
                    "category": category,
                    "score_threshold": condition["score_threshold"],
                    "nms_threshold": condition["nms_threshold"],
                    "mediapipe_crop_margin": condition["mediapipe_crop_margin"],
                    "bbox_scale": bbox_scale,
                    "tp": metric.get("tp", row["true_positive"]),
                    "fp": metric.get("fp", row["false_positive"]),
                    "fn": metric.get("fn", row["false_negative"]),
                    "duplicate_pairs": metric.get("duplicate_pairs", 0),
                    "mean_iou": metric.get("mean_iou", row["mean_iou"]),
                }
            )
    return overlay_rows


def main() -> int:
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    records = merge_records_by_image(select_records_for_scope(ground_truth.get("records", []), COMPARISON_SCOPE))
    if args.max_images is not None:
        records = records[: args.max_images]
        ground_truth = {**ground_truth, "records": ground_truth.get("records", [])[: args.max_images]}

    configs = load_or_generate_predictions(args)
    detection_rows, eye_rows, duplicate_rows, predictions_cache = build_rows(ground_truth, records, configs)
    attach_eye_metrics(detection_rows, eye_rows)
    recommendations = select_recommendations(detection_rows)

    report_prefix = args.reports_dir / args.prefix
    detection_output = report_prefix.with_name(f"{report_prefix.name}_detection_summary.csv")
    eye_output = report_prefix.with_name(f"{report_prefix.name}_eye_summary.csv")
    duplicate_output = report_prefix.with_name(f"{report_prefix.name}_duplicate_analysis.csv")
    recommendations_output = report_prefix.with_name(f"{report_prefix.name}_recommendations.json")
    overlay_index_output = report_prefix.with_name(f"{report_prefix.name}_overlay_index.csv")

    write_csv(detection_output, detection_rows)
    write_csv(eye_output, eye_rows)
    write_csv(duplicate_output, duplicate_rows)

    cv2 = import_cv2()
    overlay_rows = []
    for label in ("best_constrained", "best_recall_raw"):
        condition = recommendations[label]
        predictions = predictions_cache.get(condition["prediction_file"])
        if predictions is None:
            predictions = load_json(Path(condition["prediction_file"]))
        overlay_rows.extend(write_overlays_for_condition(args, cv2, records, condition, label, predictions))
    write_csv(overlay_index_output, overlay_rows)

    recommendations["outputs"] = {
        "detection_summary": str(detection_output),
        "eye_summary": str(eye_output),
        "duplicate_analysis": str(duplicate_output),
        "overlay_index": str(overlay_index_output),
        "overlays_dir": str(args.overlays_dir),
    }
    write_json(recommendations_output, recommendations)

    print(f"detection: {detection_output}")
    print(f"eye: {eye_output}")
    print(f"duplicates: {duplicate_output}")
    print(f"overlays: {overlay_index_output}")
    print(f"recommendations: {recommendations_output}")
    best = recommendations["best_constrained"]
    print(
        "best_constrained:",
        f"score={best['score_threshold']}",
        f"nms={best['nms_threshold']}",
        f"margin={best['mediapipe_crop_margin']}",
        f"scale={best['bbox_scale']}",
        f"recall={best['recall']}",
        f"precision={best['precision']}",
        f"detections_per_gt={best['detections_per_gt']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
