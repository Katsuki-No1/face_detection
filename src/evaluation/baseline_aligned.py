"""Shared Face-DINO-V3 evaluation contract for scheduled architectures.

The module deliberately evaluates normalized prediction records instead of a
particular model class.  Model-specific evaluators only need to export boxes,
ellipses, keypoints, and state probabilities in original-image coordinates.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.ellipse_detection import ellipse_iou


SCHEMA_VERSION = "face-dino-baseline-aligned-v2"
FACE_PRECISION_CONSTRAINT = 0.90
FACE_THRESHOLD_SELECTION_FILENAME = "face_threshold_selection.json"
THRESHOLDS = {
    "head_score": 0.30,
    "head_max_detections": 100,
    "head_nms": False,
    "head_matching_iou": 0.50,
    "head_roi_scale": 1.20,
    "face_score": 0.33,
    "point_presence": 0.70,
    "point_nme": [0.05, 0.10],
    "angle_axis_ratio": 1.05,
}
EXPECTED = {
    "val": {
        "groups": 6,
        "images": 530,
        "heads": 882,
        "face_positive": 834,
        "no_face": 46,
        "ignored_ambiguous": 2,
        "annotation_sha256": "0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0",
    },
    "test": {
        "groups": 6,
        "images": 528,
        "heads": 806,
        "face_positive": 766,
        "no_face": 39,
        "ignored_ambiguous": 1,
        "annotation_sha256": "bd32bd1591686947bfcc79cbbcd79e1059bba1569e8bd06127d3327f2b776487",
    },
}
# These are the unmatched Heads whose nearby rejected Face makes no-face
# supervision ambiguous.  The current V3 JSON predates serialization of
# face_supervision_valid, so the immutable IDs are part of the evaluation
# contract.  They reproduce the report inventory exactly.
IGNORED_HEAD_IDS = {"val": {1219, 4717}, "test": {4096}}
SEMANTIC_SLOTS = {"eye": (0, 1), "nose": (2,), "mouth": (3, 4)}
OCCLUSION_SEVERITY_CASES = (
    "occlusion_proxy_0_percent",
    "occlusion_proxy_25_percent",
    "occlusion_proxy_50_percent",
    "occlusion_proxy_75_percent",
    "occlusion_proxy_100_percent",
)

# The designated report values are intentionally kept separate from computed
# candidate values.  Test never writes back into this mapping.
BASELINE_TEST = {
    "head.ap": 0.755,
    "head.ap50": 0.963,
    "head.ap75": 0.856,
    "head.ap_small": 0.217,
    "head.ap_medium": 0.402,
    "head.ap_large": 0.776,
    "head.ar1": 0.503,
    "head.ar10": 0.806,
    "head.ar100": 0.821,
    "head.ar_small": 0.250,
    "head.ar_medium": 0.535,
    "head.ar_large": 0.841,
    "gt_head.face.macro_f1": 0.812,
    "gt_head.face.face_present_recall": 0.969,
    "gt_head.face.no_face_recall": 0.769,
    "gt_head.face.false_face_rate": 0.231,
    "gt_head.face.true_positive": 742,
    "gt_head.face.false_positive": 9,
    "gt_head.face.false_negative": 24,
    "gt_head.face.true_negative": 30,
    "gt_head.ellipse.mean_iou": 0.723,
    "gt_head.ellipse.median_iou": 0.787,
    "gt_head.ellipse.p10_iou": 0.450,
    "gt_head.ellipse.iou_ge_0.50": 0.871,
    "gt_head.ellipse.iou_ge_0.75": 0.595,
    "gt_head.ellipse.center_nme": 0.0527,
    "gt_head.ellipse.major_relative_error": 0.127,
    "gt_head.ellipse.minor_relative_error": 0.176,
    "gt_head.ellipse.angle_mae_deg": 9.32,
    "gt_head.keypoint.macro_f1_0.05": 0.835,
    "gt_head.keypoint.macro_f1_0.10": 0.896,
    "gt_head.keypoint.micro_recall_0.05": 0.843,
    "gt_head.keypoint.micro_recall_0.10": 0.908,
    "gt_head.keypoint.matched_point_nme_0.10": 0.0190,
    "gt_head.keypoint.visible_point_nme_0.10": 0.0174,
    "gt_head.keypoint.occluded_point_nme_0.10": 0.0312,
    "gt_head.state.state_accuracy": 0.903,
    "gt_head.state.state_macro_f1": 0.801,
    "gt_head.state.visible_f1": 0.944,
    "gt_head.state.occluded_conditional_f1": 0.658,
    "gt_head.state.occluded_e2e_precision": 0.400,
    "gt_head.state.occluded_e2e_recall": 0.588,
    "gt_head.state.occluded_e2e_f1": 0.476,
    "matched_head.face.macro_f1": 0.809,
    "matched_head.counts.matched_face_positive": 734,
    "matched_head.counts.matched_supervised_heads": 760,
    "matched_head.ellipse.mean_iou": 0.73,
    "matched_head.ellipse.median_iou": 0.79,
    "matched_head.ellipse.p10_iou": 0.48,
    "matched_head.ellipse.iou_ge_0.50": 0.89,
    "matched_head.ellipse.iou_ge_0.75": 0.61,
    "matched_head.ellipse.center_nme": 0.05,
    "matched_head.ellipse.major_relative_error": 0.12,
    "matched_head.ellipse.minor_relative_error": 0.17,
    "matched_head.ellipse.angle_mae_deg": 8.97,
    "matched_head.keypoint.macro_f1_0.05": 0.85,
    "matched_head.keypoint.macro_f1_0.10": 0.91,
    "matched_head.keypoint.matched_point_nme_0.10": 0.02,
    "hard_cases.all_five_visible.instances": 298,
    "hard_cases.all_five_visible.ellipse.mean_iou": 0.828,
    "hard_cases.all_five_visible.ellipse.p10_iou": 0.730,
    "hard_cases.one_to_four_visible.instances": 400,
    "hard_cases.one_to_four_visible.ellipse.mean_iou": 0.700,
    "hard_cases.one_to_four_visible.ellipse.p10_iou": 0.461,
    "hard_cases.all_missing.instances": 46,
    "hard_cases.all_missing.ellipse.mean_iou": 0.376,
    "hard_cases.all_missing.ellipse.p10_iou": 0.119,
    "hard_cases.occluded_only.instances": 22,
    "hard_cases.occluded_only.ellipse.mean_iou": 0.379,
    "hard_cases.occluded_only.ellipse.p10_iou": 0.128,
    "hard_cases.high_eccentricity.instances": 135,
    "hard_cases.high_eccentricity.ellipse.mean_iou": 0.577,
    "hard_cases.high_eccentricity.ellipse.p10_iou": 0.313,
    "hard_cases.low_eccentricity.instances": 631,
    "hard_cases.low_eccentricity.ellipse.mean_iou": 0.752,
    "hard_cases.low_eccentricity.ellipse.p10_iou": 0.521,
    "hard_cases.small_head_tercile.instances": 256,
    "hard_cases.small_head_tercile.ellipse.mean_iou": 0.676,
    "hard_cases.small_head_tercile.ellipse.p10_iou": 0.313,
}

BASELINE_VAL = {
    "head.ap": 0.736,
    "head.ap50": 0.941,
    "head.ap75": 0.847,
    "head.ap_small": 0.000,
    "head.ap_medium": 0.555,
    "head.ap_large": 0.753,
    "head.ar1": 0.459,
    "head.ar10": 0.784,
    "head.ar100": 0.816,
    "head.ar_small": 0.033,
    "head.ar_medium": 0.722,
    "head.ar_large": 0.828,
    "gt_head.face.macro_f1": 0.79,
    "gt_head.face.face_present_recall": 0.97,
    "gt_head.face.no_face_recall": 0.67,
    "gt_head.face.false_face_rate": 0.33,
    "gt_head.ellipse.mean_iou": 0.70,
    "gt_head.ellipse.median_iou": 0.76,
    "gt_head.ellipse.p10_iou": 0.39,
    "gt_head.ellipse.iou_ge_0.50": 0.85,
    "gt_head.ellipse.iou_ge_0.75": 0.52,
    "gt_head.ellipse.center_nme": 0.06,
    "gt_head.ellipse.major_relative_error": 0.13,
    "gt_head.ellipse.minor_relative_error": 0.20,
    "gt_head.ellipse.angle_mae_deg": 11.15,
    "gt_head.keypoint.macro_f1_0.05": 0.80,
    "gt_head.keypoint.macro_f1_0.10": 0.88,
    "gt_head.keypoint.micro_recall_0.05": 0.80,
    "gt_head.keypoint.micro_recall_0.10": 0.88,
    "gt_head.keypoint.matched_point_nme_0.10": 0.02,
    "gt_head.keypoint.visible_point_nme_0.10": 0.02,
    "gt_head.keypoint.occluded_point_nme_0.10": 0.04,
    "gt_head.state.state_accuracy": 0.896,
    "gt_head.state.state_macro_f1": 0.768,
    "gt_head.state.visible_f1": 0.940,
    "gt_head.state.occluded_conditional_f1": 0.595,
    "gt_head.state.occluded_e2e_precision": 0.343,
    "gt_head.state.occluded_e2e_recall": 0.495,
    "gt_head.state.occluded_e2e_f1": 0.405,
    "matched_head.counts.matched_face_positive": 787,
    "matched_head.counts.matched_supervised_heads": 814,
    "matched_head.face.macro_f1": 0.75,
    "matched_head.ellipse.mean_iou": 0.71,
    "matched_head.ellipse.median_iou": 0.76,
    "matched_head.ellipse.p10_iou": 0.45,
    "matched_head.ellipse.iou_ge_0.50": 0.88,
    "matched_head.ellipse.iou_ge_0.75": 0.54,
    "matched_head.ellipse.center_nme": 0.05,
    "matched_head.ellipse.major_relative_error": 0.13,
    "matched_head.ellipse.minor_relative_error": 0.18,
    "matched_head.ellipse.angle_mae_deg": 10.19,
    "matched_head.keypoint.macro_f1_0.05": 0.81,
    "matched_head.keypoint.macro_f1_0.10": 0.89,
    "matched_head.keypoint.matched_point_nme_0.10": 0.02,
}

TABLE_METRICS = (
    "cascade.total_face_positive_heads",
    "cascade.head_detected_heads",
    "cascade.head_recall",
    "cascade.face_detected_after_head_heads",
    "cascade.face_given_head_rate",
    "cascade.face_end_to_end_recall",
    "cascade.keypoint_positive_heads",
    "cascade.keypoint_eligible_after_face_heads",
    "cascade.keypoint_detected_after_face_heads",
    "cascade.keypoint_given_face_rate",
    "cascade.keypoint_detected_end_to_end_heads",
    "cascade.keypoint_head_end_to_end_recall",
    "cascade.gt_keypoint_points_total",
    "cascade.gt_keypoint_points_after_head",
    "cascade.gt_keypoint_points_after_face",
    "cascade.predicted_keypoint_points_after_face",
    "cascade.correct_keypoint_points_0.05",
    "cascade.correct_keypoint_points_0.10",
    "cascade.correct_keypoint_point_recall_0.05",
    "cascade.correct_keypoint_point_recall_0.10",
    "cascade.correct_keypoint_point_recall_given_face_0.05",
    "cascade.correct_keypoint_point_recall_given_face_0.10",
    "gt_head.face.supervised_heads",
    "gt_head.face.positive_heads",
    "gt_head.face.negative_heads",
    "gt_head.face.face_present_precision",
    "gt_head.keypoint.micro_recall_0.05",
    "gt_head.keypoint.micro_recall_0.10",
    "gt_head.keypoint.matched_points_0.10",
    "gt_head.keypoint.visible_matched_points_0.10",
    "gt_head.keypoint.occluded_matched_points_0.10",
    "gt_head.state.evaluated_present_points",
    "matched_head.keypoint.micro_recall_0.05",
    "matched_head.keypoint.micro_recall_0.10",
    "matched_head.face.face_present_precision",
    "runtime.total_seconds",
    "runtime.images_per_second",
    "runtime.model_inference_milliseconds_per_image",
    "runtime.peak_vram_mib",
    "runtime.batch_size",
    "runtime.input_size",
) + tuple(
    f"{scope}.keypoint.per_semantic.{semantic}.{threshold}.{metric}"
    for scope in ("gt_head", "matched_head")
    for semantic in ("eye", "nose", "mouth")
    for threshold in ("0.05", "0.10")
    for metric in ("pred", "gt", "precision", "recall", "f1")
) + tuple(
    f"hard_cases.{case}.{section}.{metric}"
    for case in (
        "all_five_visible",
        "one_to_four_visible",
        "all_missing",
        "occluded_only",
        "high_eccentricity",
        "low_eccentricity",
        "image_edge",
        "occlusion_yes",
        "small_head_tercile",
        *OCCLUSION_SEVERITY_CASES,
    )
    for section, metric in (
        ("ellipse", "mean_iou"),
        ("ellipse", "p10_iou"),
        ("keypoint", "macro_f1_0.05"),
        ("keypoint", "macro_f1_0.10"),
    )
) + tuple(
    f"hard_cases.{case}.{scope}.{metric}"
    for case in OCCLUSION_SEVERITY_CASES
    for scope, metric in (
        ("face_detection.gt_head", "detected_faces"),
        ("face_detection.gt_head", "recall"),
        ("face_detection.gt_head", "mean_iou"),
        ("face_detection.gt_head", "median_iou"),
        ("face_detection.gt_head", "p10_iou"),
        ("face_detection.gt_head", "iou_recall_0.50"),
        ("face_detection.gt_head", "iou_recall_0.75"),
        ("face_detection.deployment", "matched_heads"),
        ("face_detection.deployment", "head_recall"),
        ("face_detection.deployment", "detected_faces"),
        ("face_detection.deployment", "face_end_to_end_recall"),
        ("face_detection.deployment", "mean_iou"),
        ("face_detection.deployment", "iou_recall_0.50"),
        ("face_detection.deployment", "iou_recall_0.75"),
    )
) + tuple(
    f"{scope}.state.{metric}"
    for scope in ("gt_head", "matched_head")
    for metric in (
        "state_accuracy",
        "state_macro_f1",
        "visible_f1",
        "occluded_conditional_f1",
        "occluded_e2e_precision",
        "occluded_e2e_recall",
        "occluded_e2e_f1",
    )
) + tuple(
    f"hard_cases.{case}.instances"
    for case in (
        "all_five_visible",
        "one_to_four_visible",
        "all_missing",
        "occluded_only",
        "high_eccentricity",
        "low_eccentricity",
        "image_edge",
        "occlusion_yes",
        "small_head_tercile",
        *OCCLUSION_SEVERITY_CASES,
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unsupported(reason: str) -> dict[str, Any]:
    return {"supported": False, "unsupported_reason": reason}


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0e-12, area_a + area_b - intersection)


def _head_square(annotation: Mapping[str, Any]) -> tuple[list[float], float]:
    x, y, width, height = map(float, annotation["bbox"])
    side = THRESHOLDS["head_roi_scale"] * max(width, height)
    cx, cy = x + width * 0.5, y + height * 0.5
    return [cx - side * 0.5, cy - side * 0.5, cx + side * 0.5, cy + side * 0.5], side


def _prediction_center(prediction: Mapping[str, Any]) -> tuple[float, float] | None:
    ellipse = prediction.get("ellipse")
    if isinstance(ellipse, Sequence) and len(ellipse) == 5:
        return float(ellipse[0]), float(ellipse[1])
    box = prediction.get("bbox_xyxy")
    return _center(box) if isinstance(box, Sequence) and len(box) == 4 else None


def _assign_to_heads(
    heads: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    candidates: list[tuple[float, float, int, int]] = []
    for head_index, head in enumerate(heads):
        roi, side = _head_square(head)
        head_center = _center(roi)
        for prediction_index, prediction in enumerate(predictions):
            point = _prediction_center(prediction)
            if point is None or not (roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]):
                continue
            distance = math.hypot(point[0] - head_center[0], point[1] - head_center[1]) / max(side, 1e-9)
            candidates.append((float(prediction.get("score", 0.0)), -distance, head_index, prediction_index))
    assigned_heads: set[int] = set()
    assigned_predictions: set[int] = set()
    result: dict[int, int] = {}
    for _score, _distance, head_index, prediction_index in sorted(candidates, reverse=True):
        if head_index in assigned_heads or prediction_index in assigned_predictions:
            continue
        result[head_index] = prediction_index
        assigned_heads.add(head_index)
        assigned_predictions.add(prediction_index)
    return result


def _assign_linked_to_heads(
    head_predictions: Sequence[Mapping[str, Any]],
    head_matches: Mapping[int, int],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    """Resolve explicit query parent links through matched predicted HEADs."""

    query_to_gt = {
        int(head_predictions[prediction_index]["query_index"]): int(gt_index)
        for gt_index, prediction_index in head_matches.items()
        if head_predictions[prediction_index].get("query_index") is not None
    }
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        parent = prediction.get("parent_head_query_index")
        if parent is None or int(parent) < 0 or int(parent) not in query_to_gt:
            continue
        candidates.append(
            (
                float(prediction.get("score", 0.0)),
                query_to_gt[int(parent)],
                prediction_index,
            )
        )
    assigned: dict[int, int] = {}
    for _score, gt_index, prediction_index in sorted(
        candidates, reverse=True
    ):
        assigned.setdefault(gt_index, prediction_index)
    return assigned


def _match_boxes(
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            overlap = _iou(gt, prediction["bbox_xyxy"])
            if overlap >= THRESHOLDS["head_matching_iou"]:
                candidates.append((overlap, gt_index, prediction_index))
    matched_gt: set[int] = set()
    matched_prediction: set[int] = set()
    result = {}
    for _overlap, gt_index, prediction_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or prediction_index in matched_prediction:
            continue
        result[gt_index] = prediction_index
        matched_gt.add(gt_index)
        matched_prediction.add(prediction_index)
    return result


def _binary(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _face_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row["face_target"] != "ignored"]
    tp = sum(row["face_target"] == "positive" and row["face_predicted"] for row in rows)
    fp = sum(row["face_target"] == "negative" and row["face_predicted"] for row in rows)
    fn = sum(row["face_target"] == "positive" and not row["face_predicted"] for row in rows)
    tn = sum(row["face_target"] == "negative" and not row["face_predicted"] for row in rows)
    positive, negative = _binary(tp, fp, fn), _binary(tn, fn, fp)
    return {
        "supported": True,
        "supervised_heads": len(rows),
        "positive_heads": tp + fn,
        "negative_heads": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "macro_f1": 0.5 * (positive["f1"] + negative["f1"]),
        "face_present_precision": positive["precision"],
        "face_present_recall": positive["recall"],
        "no_face_recall": negative["recall"],
        "false_face_rate": fp / max(1, fp + tn),
    }


def select_face_threshold_precision_constrained(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_threshold: float = 0.33,
    precision_constraint: float = FACE_PRECISION_CONSTRAINT,
    precision_metric_path: str = "gt_head.face.face_present_precision",
    population: str = "supervised_gt_heads",
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    """Select the validation FACE threshold with maximum constrained recall.

    Precision must be strictly greater than ``precision_constraint`` and at
    least one FACE prediction must be emitted.  Recall is maximized first;
    ties prefer higher precision and then the lower threshold.
    """

    supervised = [row for row in rows if row["face_target"] != "ignored"]
    candidate_thresholds = sorted(
        {
            float(minimum_threshold),
            *(
                float(row.get("face_score", 0.0))
                for row in supervised
                if float(row.get("face_score", 0.0)) >= minimum_threshold
            ),
        }
    )
    candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        threshold_rows = []
        for source in supervised:
            row = dict(source)
            score = float(row.get("face_score", 0.0))
            has_assignment = bool(
                score > 0.0 or row.get("_assigned_ellipse") is not None
            )
            row["face_predicted"] = bool(
                has_assignment and score >= threshold
            )
            threshold_rows.append(row)
        face = _face_metrics(threshold_rows)
        predicted = int(face["true_positive"]) + int(face["false_positive"])
        precision = float(face["face_present_precision"])
        recall = float(face["face_present_recall"])
        if predicted > 0 and precision > precision_constraint:
            candidates.append(
                {
                    "face_threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "predicted_heads": predicted,
                    "true_positive": int(face["true_positive"]),
                    "false_positive": int(face["false_positive"]),
                    "false_negative": int(face["false_negative"]),
                    "true_negative": int(face["true_negative"]),
                }
            )
    if not candidates:
        if allow_unavailable:
            return {
                "status": "unsupported",
                "unsupported_reason": (
                    "no_validation_face_threshold_satisfies_precision_gt_0.90"
                ),
                "selection_source": "validation_precision_gt_0.90_max_recall",
                "objective": "maximize_recall",
                "precision_constraint": {
                    "metric": precision_metric_path,
                    "operator": ">",
                    "value": float(precision_constraint),
                },
                "selection_population": population,
                "selection_population_heads": len(supervised),
                "minimum_threshold": float(minimum_threshold),
                "candidate_threshold_count": len(candidate_thresholds),
                "eligible_threshold_count": 0,
                "face_threshold": None,
                "tie_break": ["higher_precision", "lower_threshold"],
            }
        raise RuntimeError(
            "No validation FACE threshold emits a prediction while satisfying "
            f"precision > {precision_constraint:.2f}"
        )
    selected = max(
        candidates,
        key=lambda item: (
            float(item["recall"]),
            float(item["precision"]),
            -float(item["face_threshold"]),
        ),
    )
    return {
        **selected,
        "selection_source": "validation_precision_gt_0.90_max_recall",
        "objective": "maximize_recall",
        "precision_constraint": {
            "metric": precision_metric_path,
            "operator": ">",
            "value": float(precision_constraint),
        },
        "selection_population": population,
        "selection_population_heads": len(supervised),
        "minimum_threshold": float(minimum_threshold),
        "candidate_threshold_count": len(candidate_thresholds),
        "eligible_threshold_count": len(candidates),
        "tie_break": ["higher_precision", "lower_threshold"],
    }


def _ellipse_metrics(rows: Sequence[Mapping[str, Any]], supported: bool) -> dict[str, Any]:
    if not supported:
        return _unsupported("architecture_does_not_output_face_ellipse")
    overlaps, centers, majors, minors, angles = [], [], [], [], []
    for row in rows:
        if row["face_target"] != "positive":
            continue
        predicted, target = row.get("predicted_ellipse"), row["gt_ellipse"]
        if predicted is None:
            overlaps.append(0.0)
            continue
        overlaps.append(float(ellipse_iou(predicted, target)))
        centers.append(math.hypot(predicted[0] - target[0], predicted[1] - target[1]) / row["head_roi_side"])
        majors.append(abs(predicted[2] - target[2]) / max(1e-9, target[2]))
        minors.append(abs(predicted[3] - target[3]) / max(1e-9, target[3]))
        if target[2] / max(1e-9, target[3]) >= THRESHOLDS["angle_axis_ratio"]:
            difference = abs((predicted[4] - target[4]) % math.pi)
            angles.append(math.degrees(min(difference, math.pi - difference)))
    return {
        "supported": True,
        "positive_heads": len(overlaps),
        "mean_iou": float(np.mean(overlaps)) if overlaps else None,
        "median_iou": float(np.median(overlaps)) if overlaps else None,
        "p10_iou": float(np.quantile(overlaps, 0.10)) if overlaps else None,
        "iou_ge_0.50": sum(x >= 0.50 for x in overlaps) / len(overlaps) if overlaps else None,
        "iou_ge_0.75": sum(x >= 0.75 for x in overlaps) / len(overlaps) if overlaps else None,
        "center_nme": float(np.mean(centers)) if centers else None,
        "major_relative_error": float(np.mean(majors)) if majors else None,
        "minor_relative_error": float(np.mean(minors)) if minors else None,
        "angle_samples": len(angles),
        "angle_mae_deg": float(np.mean(angles)) if angles else None,
        "iou_implementation": "repository_96_point_polygon",
    }


def _keypoint_metrics(
    rows: Sequence[Mapping[str, Any]],
    supported: bool,
    *,
    point_threshold: float = 0.70,
) -> dict[str, Any]:
    if not supported:
        return _unsupported("architecture_does_not_output_fixed_five_keypoints")
    counts = {name: {threshold: {"pred": 0, "gt": 0, "tp": 0} for threshold in (0.05, 0.10)} for name in SEMANTIC_SLOTS}
    matched: list[float] = []
    visible: list[float] = []
    occluded: list[float] = []
    for row in rows:
        if row["face_target"] == "ignored":
            continue
        states = row["gt_keypoint_states"]
        points, scores = row.get("predicted_keypoints"), row.get("point_scores")
        for semantic, slots in SEMANTIC_SLOTS.items():
            for slot in slots:
                gt_present = states[slot] > 0
                predicted = bool(
                    row["face_predicted"]
                    and points is not None
                    and scores is not None
                    and float(scores[slot]) >= point_threshold
                )
                error = None
                if gt_present and points is not None:
                    error = math.hypot(
                        float(points[slot][0]) - row["gt_keypoints"][slot][0],
                        float(points[slot][1]) - row["gt_keypoints"][slot][1],
                    ) / row["head_roi_side"]
                for threshold in (0.05, 0.10):
                    item = counts[semantic][threshold]
                    item["pred"] += int(predicted)
                    item["gt"] += int(gt_present)
                    item["tp"] += int(gt_present and predicted and error is not None and error <= threshold)
                if gt_present and predicted and error is not None and error <= 0.10:
                    matched.append(error)
                    (visible if states[slot] == 2 else occluded).append(error)
    per_semantic: dict[str, Any] = {}
    macro, total_tp, total_gt = {}, {}, {}
    for threshold in (0.05, 0.10):
        f1s, tps, gts = [], 0, 0
        for semantic in SEMANTIC_SLOTS:
            item = counts[semantic][threshold]
            measure = _binary(item["tp"], item["pred"] - item["tp"], item["gt"] - item["tp"])
            per_semantic.setdefault(semantic, {})[f"{threshold:.2f}"] = {**item, **measure}
            f1s.append(measure["f1"])
            tps += item["tp"]
            gts += item["gt"]
        macro[threshold], total_tp[threshold], total_gt[threshold] = float(np.mean(f1s)), tps, gts
    return {
        "supported": True,
        "point_threshold": point_threshold,
        "macro_f1_0.05": macro[0.05],
        "macro_f1_0.10": macro[0.10],
        "micro_recall_0.05": total_tp[0.05] / max(1, total_gt[0.05]),
        "micro_recall_0.10": total_tp[0.10] / max(1, total_gt[0.10]),
        "matched_point_nme_0.10": float(np.mean(matched)) if matched else None,
        "visible_point_nme_0.10": float(np.mean(visible)) if visible else None,
        "occluded_point_nme_0.10": float(np.mean(occluded)) if occluded else None,
        "matched_points_0.10": len(matched),
        "visible_matched_points_0.10": len(visible),
        "occluded_matched_points_0.10": len(occluded),
        "per_semantic": per_semantic,
    }


def _cascade_waterfall(
    rows: Sequence[Mapping[str, Any]],
    *,
    head_supported: bool,
    keypoint_supported: bool,
    point_threshold: float = 0.70,
) -> dict[str, Any]:
    """Count the same GT Head instances through HEAD -> FACE -> keypoint.

    The denominator is every attribute-valid (face-positive) GT Head. HEAD
    success requires a predicted Head matched at the locked IoU. FACE and
    keypoint success are then gated by all preceding stages, so downstream
    counts cannot hide upstream misses.
    """

    if not head_supported:
        return _unsupported("architecture_does_not_output_head_boxes")

    positive = [row for row in rows if row["face_target"] == "positive"]
    head_detected = [row for row in positive if bool(row.get("head_detected"))]
    face_detected = [row for row in head_detected if bool(row["face_predicted"])]
    keypoint_positive = [
        row
        for row in positive
        if any(int(state) > 0 for state in row.get("gt_keypoint_states", ()))
    ]
    keypoint_eligible = [
        row
        for row in face_detected
        if any(int(state) > 0 for state in row.get("gt_keypoint_states", ()))
    ]
    keypoint_eligible_ids = {
        int(row["annotation_id"])
        for row in keypoint_eligible
        if row.get("annotation_id") is not None
    }

    result: dict[str, Any] = {
        "supported": True,
        "denominator": "attribute_valid_face_positive_gt_heads",
        "head_match": "predicted_head_to_gt_head_iou_at_least_0.50",
        "face_gate": "assigned_face_score_at_least_locked_face_threshold",
        "keypoint_gate": (
            "preceding_head_and_face_success_and_at_least_one_point_presence_"
            "score_at_or_above_threshold"
        ),
        "point_threshold": float(point_threshold),
        "total_face_positive_heads": len(positive),
        "head_detected_heads": len(head_detected),
        "head_recall": len(head_detected) / max(1, len(positive)),
        "face_detected_after_head_heads": len(face_detected),
        "face_given_head_rate": len(face_detected) / max(1, len(head_detected)),
        "face_end_to_end_recall": len(face_detected) / max(1, len(positive)),
        "keypoint_supported": bool(keypoint_supported),
        "keypoint_positive_heads": len(keypoint_positive),
    }
    if not keypoint_supported:
        result["keypoint_unsupported_reason"] = (
            "architecture_does_not_output_fixed_five_keypoints"
        )
        return result

    keypoint_detected: list[Mapping[str, Any]] = []
    gt_points_total = 0
    gt_points_after_head = 0
    gt_points_after_face = 0
    predicted_points_after_face = 0
    correct = {0.05: 0, 0.10: 0}
    correct_heads = {0.05: 0, 0.10: 0}

    for row in positive:
        states = [int(state) for state in row.get("gt_keypoint_states", ())]
        gt_present = [index for index, state in enumerate(states) if state > 0]
        gt_points_total += len(gt_present)
        if bool(row.get("head_detected")):
            gt_points_after_head += len(gt_present)
        if int(row.get("annotation_id", -1)) not in keypoint_eligible_ids:
            continue
        gt_points_after_face += len(gt_present)
        points = row.get("predicted_keypoints")
        scores = row.get("point_scores")
        predicted_slots = []
        if points is not None and scores is not None:
            predicted_slots = [
                slot
                for slot in range(min(len(points), len(scores), len(states)))
                if float(scores[slot]) >= point_threshold
            ]
        predicted_points_after_face += len(predicted_slots)
        if predicted_slots:
            keypoint_detected.append(row)

        correct_in_head = {0.05: False, 0.10: False}
        if points is None or scores is None:
            continue
        for slot in gt_present:
            if slot >= len(points) or slot >= len(scores):
                continue
            if float(scores[slot]) < point_threshold:
                continue
            error = math.hypot(
                float(points[slot][0]) - float(row["gt_keypoints"][slot][0]),
                float(points[slot][1]) - float(row["gt_keypoints"][slot][1]),
            ) / max(float(row["head_roi_side"]), 1.0e-12)
            for threshold in (0.05, 0.10):
                if error <= threshold:
                    correct[threshold] += 1
                    correct_in_head[threshold] = True
        for threshold in (0.05, 0.10):
            correct_heads[threshold] += int(correct_in_head[threshold])

    result.update(
        {
            "keypoint_eligible_after_face_heads": len(keypoint_eligible),
            "keypoint_detected_after_face_heads": len(keypoint_detected),
            "keypoint_given_face_rate": len(keypoint_detected)
            / max(1, len(keypoint_eligible)),
            "keypoint_detected_end_to_end_heads": len(keypoint_detected),
            "keypoint_head_end_to_end_recall": len(keypoint_detected)
            / max(1, len(keypoint_positive)),
            "keypoint_correct_heads_0.05": correct_heads[0.05],
            "keypoint_correct_heads_0.10": correct_heads[0.10],
            "gt_keypoint_points_total": gt_points_total,
            "gt_keypoint_points_after_head": gt_points_after_head,
            "gt_keypoint_points_after_face": gt_points_after_face,
            "predicted_keypoint_points_after_face": predicted_points_after_face,
            "correct_keypoint_points_0.05": correct[0.05],
            "correct_keypoint_points_0.10": correct[0.10],
            "correct_keypoint_point_recall_0.05": correct[0.05]
            / max(1, gt_points_total),
            "correct_keypoint_point_recall_0.10": correct[0.10]
            / max(1, gt_points_total),
            "correct_keypoint_point_recall_given_face_0.05": correct[0.05]
            / max(1, gt_points_after_face),
            "correct_keypoint_point_recall_given_face_0.10": correct[0.10]
            / max(1, gt_points_after_face),
        }
    )
    return result


def _state_metrics(
    rows: Sequence[Mapping[str, Any]],
    supported: bool,
    *,
    point_threshold: float = 0.70,
) -> dict[str, Any]:
    if not supported:
        return _unsupported("architecture_does_not_output_missing_occluded_visible_state")
    confusion = np.zeros((2, 2), dtype=np.int64)  # rows/cols: occluded, visible
    e2e_tp = e2e_pred = e2e_gt = 0
    samples = 0
    for row in rows:
        if row["face_target"] != "positive":
            continue
        probabilities = row.get("state_probabilities")
        points, scores = row.get("predicted_keypoints"), row.get("point_scores")
        if probabilities is None:
            continue
        for slot, gt_state in enumerate(row["gt_keypoint_states"]):
            if gt_state not in (1, 2):
                continue
            state_row = np.asarray(probabilities[slot])
            predicted_state = 1 if state_row[1] >= state_row[2] else 2
            confusion[gt_state - 1, predicted_state - 1] += 1
            samples += 1
            is_occluded_prediction = bool(
                predicted_state == 1
                and row["face_predicted"]
                and points is not None
                and scores is not None
                and scores[slot] >= point_threshold
            )
            e2e_pred += int(is_occluded_prediction)
            e2e_gt += int(gt_state == 1)
            if is_occluded_prediction and gt_state == 1:
                error = math.hypot(points[slot][0] - row["gt_keypoints"][slot][0], points[slot][1] - row["gt_keypoints"][slot][1]) / row["head_roi_side"]
                e2e_tp += int(error <= 0.10)
    if not samples:
        return _unsupported("state_probabilities_were_not_present_in_prediction_records")
    occluded = _binary(int(confusion[0, 0]), int(confusion[1, 0]), int(confusion[0, 1]))
    visible = _binary(int(confusion[1, 1]), int(confusion[0, 1]), int(confusion[1, 0]))
    e2e = _binary(e2e_tp, e2e_pred - e2e_tp, e2e_gt - e2e_tp)
    return {
        "supported": True,
        "point_threshold": point_threshold,
        "evaluated_present_points": samples,
        "state_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "state_macro_f1": 0.5 * (occluded["f1"] + visible["f1"]),
        "visible_f1": visible["f1"],
        "occluded_conditional_f1": occluded["f1"],
        "occluded_e2e_precision": e2e["precision"],
        "occluded_e2e_recall": e2e["recall"],
        "occluded_e2e_f1": e2e["f1"],
        "confusion": {
            "labels": ["occluded", "visible"],
            "rows": "ground_truth",
            "columns": "prediction",
            "values": confusion.tolist(),
        },
    }


def _head_coco_metrics(
    coco: Mapping[str, Any],
    predictions_by_image: Mapping[int, Mapping[str, Any]],
    supported: bool,
) -> dict[str, Any]:
    if not supported:
        return _unsupported("architecture_does_not_output_head_boxes")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return _unsupported("pycocotools_is_not_installed")
    detections = []
    for image in coco["images"]:
        rows = [
            row for row in predictions_by_image.get(int(image["id"]), {}).get("heads", [])
            if float(row.get("score", 0.0)) >= THRESHOLDS["head_score"]
        ]
        for row in sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:100]:
            x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
            detections.append({"image_id": int(image["id"]), "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(row["score"])})
    dataset = dict(coco)
    dataset.setdefault("info", {})
    dataset.setdefault("licenses", [])
    gt = COCO()
    gt.dataset = dataset
    gt.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        result = gt.loadRes(detections)
        evaluator = COCOeval(gt, result, "bbox")
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = evaluator.stats.tolist()
    keys = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large")
    return {"supported": True, "detections_at_locked_threshold": len(detections), **dict(zip(keys, map(float, values), strict=True))}


def _occlusion_severity_case(states: Sequence[int]) -> str | None:
    """Map five semantic states to the documented coarse occlusion proxy.

    The source data has point states, not an occluder mask or visible-face-area
    annotation.  A non-visible slot (missing=0 or occluded=1) therefore acts as
    one fifth of the proxy.  The 40% and 60% cases share the approximate 50%
    bucket.  Zero is retained as a control population.
    """

    if len(states) != 5:
        return None
    non_visible = sum(int(state) != 2 for state in states)
    return {
        0: "occlusion_proxy_0_percent",
        1: "occlusion_proxy_25_percent",
        2: "occlusion_proxy_50_percent",
        3: "occlusion_proxy_50_percent",
        4: "occlusion_proxy_75_percent",
        5: "occlusion_proxy_100_percent",
    }[non_visible]


def _face_detection_slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    ellipse_supported: bool,
    head_supported: bool,
) -> dict[str, Any]:
    """Measure thresholded FACE localization on one positive-GT stratum."""

    positive = [row for row in rows if row["face_target"] == "positive"]
    instance_count = len(positive)

    def localization(
        selected: Sequence[Mapping[str, Any]],
        *,
        detected_key: str,
    ) -> dict[str, Any]:
        detected_count = sum(bool(row[detected_key]) for row in selected)
        if not ellipse_supported:
            return {
                "detected_faces": detected_count,
                "recall": detected_count / instance_count if instance_count else None,
                **_unsupported("architecture_does_not_output_face_ellipse"),
            }
        overlaps = []
        for row in selected:
            predicted = row.get("predicted_ellipse")
            overlap = (
                float(ellipse_iou(predicted, row["gt_ellipse"]))
                if bool(row[detected_key]) and predicted is not None
                else 0.0
            )
            overlaps.append(overlap)
        return {
            "supported": True,
            "detected_faces": detected_count,
            "recall": detected_count / instance_count if instance_count else None,
            "mean_iou": float(np.mean(overlaps)) if overlaps else None,
            "median_iou": float(np.median(overlaps)) if overlaps else None,
            "p10_iou": float(np.quantile(overlaps, 0.10)) if overlaps else None,
            "iou_recall_0.50": (
                sum(value >= 0.50 for value in overlaps) / instance_count
                if instance_count
                else None
            ),
            "iou_recall_0.75": (
                sum(value >= 0.75 for value in overlaps) / instance_count
                if instance_count
                else None
            ),
            "miss_iou_policy": "zero",
            "iou_implementation": "repository_96_point_polygon",
        }

    gt_rows = []
    deployment_rows = []
    for source in positive:
        gt_row = dict(source)
        gt_row["_slice_detected"] = bool(source.get("face_predicted"))
        gt_rows.append(gt_row)
        deployment_row = dict(source)
        deployment_row["_slice_detected"] = bool(
            source.get("head_detected") and source.get("face_predicted")
        )
        deployment_rows.append(deployment_row)

    gt_head = localization(gt_rows, detected_key="_slice_detected")
    if head_supported:
        deployment = localization(
            deployment_rows,
            detected_key="_slice_detected",
        )
        deployment.update(
            {
                "matched_heads": sum(
                    bool(row.get("head_detected")) for row in positive
                ),
                "head_recall": (
                    sum(bool(row.get("head_detected")) for row in positive)
                    / instance_count
                    if instance_count
                    else None
                ),
                "face_end_to_end_recall": deployment.pop("recall"),
            }
        )
    else:
        deployment = _unsupported("architecture_does_not_output_head_boxes")
    return {
        "instances": instance_count,
        "denominator": "attribute_valid_face_positive_gt_heads_in_slice",
        "gt_head": gt_head,
        "deployment": deployment,
    }


def _slice_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    positive_rows = [row for row in rows if row["face_target"] == "positive"]
    sides = [row["head_roi_side"] for row in positive_rows]
    small_cut = float(np.quantile(sides, 1 / 3)) if sides else 0.0
    # The reference report froze a high-eccentricity tail of 132 Val and 135
    # Test instances.  Preserve that exact descriptive stratum inventory;
    # this rank cut is never used for model or threshold selection.
    high_count = {834: 132, 766: 135}.get(
        len(positive_rows),
        int(round(0.17 * len(positive_rows))),
    )
    high_eccentricity_ids = {
        int(row["annotation_id"])
        for row in sorted(
            positive_rows,
            key=lambda item: (
                item["gt_ellipse"][2] / max(1e-9, item["gt_ellipse"][3]),
                int(item["annotation_id"]),
            ),
            reverse=True,
        )[:high_count]
    }
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in positive_rows:
        states = row["gt_keypoint_states"]
        visible = sum(state == 2 for state in states)
        present = sum(state > 0 for state in states)
        tags = []
        if visible == 5:
            tags.append("all_five_visible")
        if 1 <= visible <= 4:
            tags.append("one_to_four_visible")
        if present == 0:
            tags.append("all_missing")
        if present and all(state != 2 for state in states):
            tags.append("occluded_only")
        tags.append(
            "high_eccentricity"
            if int(row["annotation_id"]) in high_eccentricity_ids
            else "low_eccentricity"
        )
        x1, y1, x2, y2 = row["head_bbox_xyxy"]
        if x1 <= 0 or y1 <= 0 or x2 >= row["image_width"] or y2 >= row["image_height"]:
            tags.append("image_edge")
        if any(state == 1 for state in states):
            tags.append("occlusion_yes")
        severity_case = _occlusion_severity_case(states)
        if severity_case is not None:
            tags.append(severity_case)
        if row["head_roi_side"] <= small_cut:
            tags.append("small_head_tercile")
        for tag in tags:
            result[tag].append(row)
    return result


def _flatten_supported(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key not in {"supported", "unsupported_reason", "confusion"}:
                _flatten_supported(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def evaluate_normalized_predictions(
    *,
    annotation_path: Path,
    predictions_by_image: Mapping[int, Mapping[str, Any]],
    split: str,
    architecture_id: str,
    comparison_stage: str,
    output_dir: Path,
    provenance: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    allow_partial_inventory: bool = False,
    validation_selected_face_threshold: float | None = None,
    validation_selected_point_threshold: float | None = None,
    face_score_candidate_floor: float | None = None,
    local_face_bbox_metrics: Mapping[str, Any] | None = None,
    local_face_bbox_markdown: Sequence[str] | None = None,
    inventory_contract: Mapping[str, Any] | None = None,
    validation_face_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate and write the canonical JSON, Markdown table, and marker."""

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    expected = EXPECTED[split] if inventory_contract is None else inventory_contract
    if inventory_contract is not None and allow_partial_inventory:
        raise ValueError("An explicit split contract requires full inventory verification")
    annotation_hash = sha256_file(annotation_path)
    if annotation_hash != expected["annotation_sha256"]:
        raise RuntimeError(f"{split} annotation SHA-256 mismatch: {annotation_hash}")
    image_by_id = {int(row["id"]): row for row in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    ignored = IGNORED_HEAD_IDS[split] if inventory_contract is None else set(inventory_contract["ignored_head_ids"])
    if set(map(int, predictions_by_image)) != set(image_by_id) and not allow_partial_inventory:
        raise RuntimeError("Prediction image IDs differ from the declared split")
    rows: list[dict[str, Any]] = []
    linked_matched_rows: list[dict[str, Any]] = []
    explicit_parent_links = False
    matched_head_annotation_ids: set[int] = set()
    has_head = any(value.get("heads") for value in predictions_by_image.values())
    has_face = any(value.get("faces") for value in predictions_by_image.values())
    has_ellipse = any(any(row.get("ellipse") is not None for row in value.get("faces", [])) for value in predictions_by_image.values())
    has_keypoint = any(value.get("poses") for value in predictions_by_image.values())
    has_state = any(any(row.get("state_probabilities") is not None for row in value.get("poses", [])) for value in predictions_by_image.values())
    for image_id, image in image_by_id.items():
        heads = annotations_by_image.get(image_id, [])
        prediction = predictions_by_image.get(image_id, {})
        faces, poses = prediction.get("faces", []), prediction.get("poses", [])
        face_assignment = _assign_to_heads(heads, faces)
        pose_assignment = _assign_to_heads(heads, poses)
        head_predictions = [
            row for row in prediction.get("heads", [])
            if float(row.get("score", 0.0)) >= THRESHOLDS["head_score"]
        ][:100]
        head_matches = _match_boxes([_xywh_to_xyxy(head["bbox"]) for head in heads], head_predictions) if has_head else {}
        image_rows: list[dict[str, Any]] = []
        for index, head in enumerate(heads):
            annotation_id = int(head["id"])
            if index in head_matches:
                matched_head_annotation_ids.add(annotation_id)
            target = "positive" if bool(head.get("attribute_valid", False)) else "ignored" if annotation_id in ignored else "negative"
            face = faces[face_assignment[index]] if index in face_assignment else None
            pose = poses[pose_assignment[index]] if index in pose_assignment else None
            face_score = float(face.get("score", 0.0)) if face is not None else 0.0
            face_predicted = face is not None and face_score >= THRESHOLDS["face_score"]
            _roi, roi_side = _head_square(head)
            image_rows.append({
                "image_id": image_id,
                "image_width": float(image["width"]),
                "image_height": float(image["height"]),
                "annotation_id": annotation_id,
                "face_target": target,
                "face_score": face_score,
                "face_predicted": face_predicted,
                "head_detected": index in head_matches,
                "head_bbox_xyxy": _xywh_to_xyxy(head["bbox"]),
                "head_roi_side": roi_side,
                "gt_ellipse": list(map(float, head["ellipse"])) if target == "positive" else None,
                "predicted_ellipse": list(map(float, face["ellipse"])) if face is not None and face.get("ellipse") is not None else None,
                "_assigned_ellipse": (
                    list(map(float, face["ellipse"]))
                    if face is not None and face.get("ellipse") is not None
                    else None
                ),
                "gt_keypoints": [[float(x), float(y)] for x, y in head.get("keypoints", [])],
                "gt_keypoint_states": list(map(int, head.get("keypoint_states", []))),
                "predicted_keypoints": pose.get("keypoints") if pose is not None else None,
                "point_scores": pose.get("point_scores") if pose is not None else None,
                "state_probabilities": pose.get("state_probabilities") if pose is not None else None,
            })
        rows.extend(image_rows)
        image_has_explicit_links = bool(
            prediction.get("uses_explicit_parent_links", False)
        ) or any(
            item.get("parent_head_query_index") is not None
            for item in (*faces, *poses)
        )
        if image_has_explicit_links:
            explicit_parent_links = True
            linked_faces = _assign_linked_to_heads(
                head_predictions, head_matches, faces
            )
            linked_poses = _assign_linked_to_heads(
                head_predictions, head_matches, poses
            )
            for gt_index in head_matches:
                linked = dict(image_rows[gt_index])
                face = (
                    faces[linked_faces[gt_index]]
                    if gt_index in linked_faces
                    else None
                )
                pose = (
                    poses[linked_poses[gt_index]]
                    if gt_index in linked_poses
                    else None
                )
                linked["face_score"] = (
                    float(face.get("score", 0.0)) if face is not None else 0.0
                )
                linked["face_predicted"] = bool(
                    face is not None
                    and linked["face_score"] >= THRESHOLDS["face_score"]
                )
                linked["predicted_ellipse"] = (
                    list(map(float, face["ellipse"]))
                    if face is not None and face.get("ellipse") is not None
                    else None
                )
                linked["_assigned_ellipse"] = linked["predicted_ellipse"]
                linked["predicted_keypoints"] = (
                    pose.get("keypoints") if pose is not None else None
                )
                linked["point_scores"] = (
                    pose.get("point_scores") if pose is not None else None
                )
                linked["state_probabilities"] = (
                    pose.get("state_probabilities") if pose is not None else None
                )
                linked_matched_rows.append(linked)
    inventory = {
        "groups": len({str(image.get("source_video_id", "")) for image in coco["images"]}),
        "images": len(coco["images"]),
        "heads": len(rows),
        "face_positive": sum(row["face_target"] == "positive" for row in rows),
        "no_face": sum(row["face_target"] == "negative" for row in rows),
        "ignored_ambiguous": sum(row["face_target"] == "ignored" for row in rows),
        "prediction_images": len(predictions_by_image),
    }
    for key in ("groups", "images", "heads", "face_positive", "no_face", "ignored_ambiguous"):
        if not allow_partial_inventory and inventory[key] != expected[key]:
            raise RuntimeError(f"{split} inventory mismatch for {key}: expected={expected[key]} actual={inventory[key]}")
    matched_rows = (
        linked_matched_rows
        if explicit_parent_links
        else [
            row
            for row in rows
            if row["annotation_id"] in matched_head_annotation_ids
        ]
    )

    def operating_point(
        source_rows: Sequence[Mapping[str, Any]],
        *,
        face_threshold: float,
        point_threshold: float,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        selected_rows = []
        for source in source_rows:
            row = dict(source)
            row["face_predicted"] = bool(
                row.get("_assigned_ellipse") is not None
                or row.get("face_score", 0.0) > 0.0
            ) and float(row.get("face_score", 0.0)) >= face_threshold
            # Geometry is evaluated for every face-positive Head independently
            # of the binary face operating point, matching the reference
            # report's GT-Head ellipse diagnostic.
            row["predicted_ellipse"] = row.get("_assigned_ellipse")
            selected_rows.append(row)
        gt = {
            "face": (
                _face_metrics(selected_rows)
                if has_face
                else _unsupported("architecture_does_not_output_face_predictions")
            ),
            "ellipse": _ellipse_metrics(selected_rows, has_ellipse),
            "keypoint": _keypoint_metrics(
                selected_rows,
                has_keypoint,
                point_threshold=point_threshold,
            ),
            "state": _state_metrics(
                selected_rows,
                has_state,
                point_threshold=point_threshold,
            ),
        }
        selected_by_id = {
            int(row["annotation_id"]): row for row in selected_rows
        }
        matched = [
            selected_by_id[int(row["annotation_id"])]
            for row in matched_rows
        ]
        matched_sections = {
            "counts": {
                "matched_supervised_heads": sum(row["face_target"] != "ignored" for row in matched),
                "matched_face_positive": sum(row["face_target"] == "positive" for row in matched),
            } if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "face": (
                _face_metrics(matched)
                if has_head and has_face
                else _unsupported(
                    "architecture_does_not_output_face_predictions"
                    if has_head
                    else "architecture_does_not_output_head_boxes"
                )
            ),
            "ellipse": _ellipse_metrics(matched, has_ellipse) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "keypoint": _keypoint_metrics(matched, has_keypoint, point_threshold=point_threshold) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "state": _state_metrics(matched, has_state, point_threshold=point_threshold) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
        }
        cases = {}
        for name, subset in _slice_rows(selected_rows).items():
            cases[name] = {
                "instances": len(subset),
                "face_detection": _face_detection_slice_metrics(
                    subset,
                    ellipse_supported=has_ellipse,
                    head_supported=has_head,
                ),
                "ellipse": _ellipse_metrics(subset, has_ellipse),
                "keypoint": _keypoint_metrics(
                    subset,
                    has_keypoint,
                    point_threshold=point_threshold,
                ),
            }
        cascade = _cascade_waterfall(
            selected_rows,
            head_supported=has_head,
            keypoint_supported=has_keypoint,
            point_threshold=point_threshold,
        )
        return selected_rows, gt, matched_sections, cases, cascade

    selected_point_threshold = (
        float(validation_selected_point_threshold)
        if validation_selected_point_threshold is not None
        else float(THRESHOLDS["point_presence"])
    )
    selection_candidate_floor = (
        float(face_score_candidate_floor)
        if face_score_candidate_floor is not None
        else float(THRESHOLDS["face_score"])
    )
    if not 0.0 <= selection_candidate_floor <= 1.0:
        raise ValueError(
            "FACE score candidate floor must be in [0, 1], got "
            f"{selection_candidate_floor}"
        )
    validation_selection_path = (
        output_dir.parent.parent
        / "val"
        / output_dir.name
        / FACE_THRESHOLD_SELECTION_FILENAME
    )
    if not has_face:
        face_threshold_selection = {
            "schema_version": SCHEMA_VERSION,
            "status": "unsupported",
            "unsupported_reason": "architecture_does_not_output_face_predictions",
            "selected_on_split": None,
            "architecture_id": architecture_id,
            "comparison_stage": comparison_stage,
            "face_threshold": float(THRESHOLDS["face_score"]),
            "minimum_threshold": selection_candidate_floor,
            "point_threshold": selected_point_threshold,
            "annotation_sha256": annotation_hash,
        }
        selection_application_source = "not_applicable_to_head_only_model"
    elif validation_face_selection is not None:
        face_threshold_selection = dict(validation_face_selection)
        if face_threshold_selection.get("selected_on_split") != "val" and face_threshold_selection.get("selection_attempted_on_split") != "val":
            raise ValueError("External FACE selection must be locked on validation")
        if face_threshold_selection.get("architecture_id") != architecture_id:
            raise ValueError("External FACE selection belongs to another architecture")
        selection_application_source = "explicit_validation_bbox_iou50_lock"
    elif split == "val":
        face_threshold_selection = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "selected_on_split": "val",
            "architecture_id": architecture_id,
            "comparison_stage": comparison_stage,
            **select_face_threshold_precision_constrained(
                matched_rows if has_head else rows,
                minimum_threshold=selection_candidate_floor,
                precision_metric_path=(
                    "matched_head.face.face_present_precision"
                    if has_head
                    else "gt_head.face.face_present_precision"
                ),
                population=(
                    "supervised_gt_heads_with_matched_predicted_head"
                    if has_head
                    else "supervised_gt_heads"
                ),
                allow_unavailable=True,
            ),
            "point_threshold": selected_point_threshold,
            "annotation_sha256": annotation_hash,
        }
        if face_threshold_selection.get("status") == "unsupported":
            face_threshold_selection["selected_on_split"] = None
            face_threshold_selection["selection_attempted_on_split"] = "val"
            selection_application_source = (
                "validation_had_no_precision_constrained_operating_point"
            )
        else:
            selection_application_source = (
                "selected_automatically_on_validation"
            )
    elif validation_selection_path.is_file():
        face_threshold_selection = json.loads(
            validation_selection_path.read_text(encoding="utf-8")
        )
        selection_status = face_threshold_selection.get("status")
        if selection_status == "success":
            if face_threshold_selection.get("selected_on_split") != "val":
                raise RuntimeError(
                    "Test FACE threshold must have been selected on validation: "
                    f"{validation_selection_path}"
                )
        elif selection_status == "unsupported":
            if (
                face_threshold_selection.get("selection_attempted_on_split")
                != "val"
                or face_threshold_selection.get("unsupported_reason")
                != "no_validation_face_threshold_satisfies_precision_gt_0.90"
            ):
                raise RuntimeError(
                    "Validation FACE threshold unavailability is not a valid "
                    f"precision-constrained Val result: {validation_selection_path}"
                )
        else:
            raise RuntimeError(
                "Validation FACE threshold selection has invalid status: "
                f"{validation_selection_path}"
            )
        if face_threshold_selection.get("architecture_id") != architecture_id:
            raise RuntimeError(
                "Validation FACE threshold belongs to another architecture: "
                f"{validation_selection_path}"
            )
        if face_threshold_selection.get("comparison_stage") != comparison_stage:
            raise RuntimeError(
                "Validation FACE threshold belongs to another comparison "
                f"stage: {validation_selection_path}"
            )
        constraint = face_threshold_selection.get("precision_constraint", {})
        if (
            constraint.get("operator") != ">"
            or float(constraint.get("value", -1.0))
            != FACE_PRECISION_CONSTRAINT
        ):
            raise RuntimeError(
                "Validation FACE threshold selection has the wrong precision "
                f"constraint: {validation_selection_path}"
            )
        selected_point_threshold = float(
            face_threshold_selection.get(
                "point_threshold",
                THRESHOLDS["point_presence"],
            )
        )
        selection_application_source = (
            "frozen_validation_artifact"
            if selection_status == "success"
            else "frozen_validation_artifact_no_feasible_operating_point"
        )
    elif validation_selected_face_threshold is not None:
        # Compatibility path for callers whose Val and Test directories are
        # not siblings.  The argument is explicitly validation-selected; Test
        # still never computes a threshold from its own labels.
        face_threshold_selection = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "selected_on_split": "val",
            "architecture_id": architecture_id,
            "comparison_stage": comparison_stage,
            "selection_source": "explicit_validation_selected_threshold",
            "objective": "maximize_recall",
            "precision_constraint": {
                "metric": "gt_head.face.face_present_precision",
                "operator": ">",
                "value": FACE_PRECISION_CONSTRAINT,
            },
            "face_threshold": float(validation_selected_face_threshold),
            "minimum_threshold": selection_candidate_floor,
            "point_threshold": selected_point_threshold,
            "validation_metrics_unavailable": True,
        }
        selection_application_source = "explicit_validation_threshold"
    else:
        raise FileNotFoundError(
            "Test evaluation requires the validation-selected FACE threshold "
            f"artifact: {validation_selection_path}"
        )

    rows, gt_sections, matched_sections, hard_cases, cascade = operating_point(
        rows,
        face_threshold=float(THRESHOLDS["face_score"]),
        point_threshold=float(THRESHOLDS["point_presence"]),
    )
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "architecture_id": architecture_id,
        "comparison_stage": comparison_stage,
        "split": split,
        "inventory": inventory,
        "thresholds": {
            **THRESHOLDS,
            "selection_source": (
                "common_face_0.33_and_validation_precision_gt_0.90_max_recall"
            ),
        },
        "coordinate_contract": {
            "boxes": "xyxy_pixels_in_original_image",
            "ellipse": "cx_cy_major_radius_minor_radius_theta_radians",
            "keypoint_order": ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"],
            "keypoint_state": {"missing": 0, "occluded": 1, "visible": 2},
            "nme_denominator": "1.20 * max(GT Head width, GT Head height)",
            "eccentricity_slice": (
                "frozen reference-report rank tail: Val 132 / Test 135; "
                "descriptive only and never used for selection"
            ),
            "occlusion_severity_proxy": {
                "source": "five_fixed_semantic_keypoint_states",
                "non_visible_definition": "state != visible (state != 2)",
                "limitations": (
                    "weakness proxy, not pixel occlusion area ground truth; "
                    "includes object occlusion, self-occlusion, off-screen "
                    "truncation, and missing annotation"
                ),
                "buckets": {
                    "occlusion_proxy_0_percent": "0/5 non-visible slots (control)",
                    "occlusion_proxy_25_percent": "1/5 non-visible slots (20%)",
                    "occlusion_proxy_50_percent": "2-3/5 non-visible slots (40-60%)",
                    "occlusion_proxy_75_percent": "4/5 non-visible slots (80%)",
                    "occlusion_proxy_100_percent": "5/5 non-visible slots (100%)",
                },
                "selection_use": "diagnostic_only_never_select_on_test",
            },
        },
        "head": _head_coco_metrics(coco, predictions_by_image, has_head),
        "gt_head": gt_sections,
        "matched_head": matched_sections,
        "cascade": cascade,
        "hard_cases": hard_cases,
        "runtime": dict(runtime or _unsupported("runtime_was_not_supplied_by_model_evaluator")),
        "provenance": {
            "annotation": str(annotation_path),
            "annotation_sha256": annotation_hash,
            "ignored_head_ids": sorted(ignored),
            "git": _git_state(Path(__file__).resolve().parents[2]),
            **dict(provenance or {}),
        },
    }
    if has_face and face_threshold_selection.get("status") == "success":
        (
            _selected_rows,
            selected_gt,
            selected_matched,
            selected_hard_cases,
            selected_cascade,
        ) = operating_point(
            rows,
            face_threshold=float(face_threshold_selection["face_threshold"]),
            point_threshold=selected_point_threshold,
        )
        metrics["validation_selected_operating_point"] = {
            "selection_source": selection_application_source,
            "selection_policy": "precision_gt_0.90_then_maximize_recall",
            "selected_on_split": "val",
            "test_selection_forbidden": True,
            "face_threshold": float(face_threshold_selection["face_threshold"]),
            "point_threshold": selected_point_threshold,
            "validation_selection": face_threshold_selection,
            "gt_head": selected_gt,
            "matched_head": selected_matched,
            "hard_cases": selected_hard_cases,
            "cascade": selected_cascade,
        }
    elif has_face:
        metrics["validation_selected_operating_point"] = {
            "supported": False,
            "unsupported_reason": face_threshold_selection.get(
                "unsupported_reason",
                "validation_face_threshold_selection_unavailable",
            ),
            "selection_policy": "precision_gt_0.90_then_maximize_recall",
            "selected_on_split": None,
            "test_selection_forbidden": True,
            "face_threshold": None,
            "point_threshold": selected_point_threshold,
            "validation_selection": face_threshold_selection,
        }
    else:
        metrics["validation_selected_operating_point"] = _unsupported(
            "architecture_does_not_output_face_predictions"
        )
    flattened: dict[str, Any] = {}
    for key in (
        "head",
        "gt_head",
        "matched_head",
        "cascade",
        "hard_cases",
        "runtime",
    ):
        _flatten_supported(key, metrics[key], flattened)
    validation_selected_flattened: dict[str, Any] = {}
    selected = metrics.get("validation_selected_operating_point")
    if isinstance(selected, Mapping) and selected.get("supported") is not False:
        for key in ("gt_head", "matched_head", "cascade", "hard_cases"):
            _flatten_supported(
                key,
                selected[key],
                validation_selected_flattened,
            )
    comparison = []
    baseline_reference = (BASELINE_VAL if split == "val" else BASELINE_TEST) if inventory_contract is None else {}
    comparison_paths = list(baseline_reference)
    comparison_paths.extend(
        name
        for name in TABLE_METRICS
        if name not in baseline_reference
    )
    for name in comparison_paths:
        baseline = baseline_reference.get(name)
        candidate = flattened.get(name)
        validation_selected_candidate = validation_selected_flattened.get(
            name,
            candidate if name.startswith("head.") else None,
        )
        comparison.append({
            "metric": name,
            "baseline": baseline,
            "candidate": candidate,
            "absolute_delta": (
                None
                if candidate is None or baseline is None
                else candidate - baseline
            ),
            "relative_delta": (
                None
                if candidate is None or baseline in {None, 0}
                else (candidate - baseline) / abs(baseline)
            ),
            "supported": candidate is not None,
            "validation_selected_candidate": validation_selected_candidate,
            "validation_selected_absolute_delta": (
                None
                if validation_selected_candidate is None or baseline is None
                else validation_selected_candidate - baseline
            ),
            "validation_selected_relative_delta": (
                None
                if validation_selected_candidate is None or baseline in {None, 0}
                else (validation_selected_candidate - baseline) / abs(baseline)
            ),
        })
    metrics["comparison_to_face_dino_report"] = comparison
    metrics["baseline_reference_split"] = split if inventory_contract is None else None
    if inventory_contract is not None:
        metrics["historical_reference_unsupported_reason"] = "Different split: locked historical report deltas are forbidden"
        metrics["inventory_contract"] = dict(inventory_contract)
        metrics["coordinate_contract"]["eccentricity_slice"] = "descriptive top 16 percent by GT ellipse axis ratio; no selection"
    if (
        not allow_partial_inventory
        and inventory["prediction_images"] != inventory["images"]
    ):
        raise RuntimeError(
            "Prediction inventory is incomplete: "
            f"{inventory['prediction_images']} != {inventory['images']}"
        )

    def assert_finite(value: Any, path: str = "metrics") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                assert_finite(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                assert_finite(item, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite common metric: {path}={value}")

    continuous_detector_identity = (provenance or {}).get("continuous_local_face_contract")
    if local_face_bbox_metrics is not None:
        if continuous_detector_identity is not None:
            raise ValueError("Cannot combine independent detector and shared-detector FACE contracts")
        metrics["primary_local_face_bbox"] = dict(local_face_bbox_metrics)
    if continuous_detector_identity is not None:
        from .continuous_face_contract import primary_face_metrics

        metrics["primary_local_face_bbox"] = primary_face_metrics(
            coco, predictions_by_image, split=split, output_dir=output_dir,
            candidate_floor=selection_candidate_floor, detector_identity=continuous_detector_identity,
        )
    assert_finite(metrics)
    assertions = {
        "inventory": True,
        "finite_values": True,
        "head_metrics": not has_head or metrics["head"].get("supported") is True,
        "ellipse_metrics": not has_ellipse or metrics["gt_head"]["ellipse"].get("supported") is True,
        "keypoint_metrics": not has_keypoint or metrics["gt_head"]["keypoint"].get("supported") is True,
        "state_metrics": not has_state or metrics["gt_head"]["state"].get("supported") is True,
        "cascade_waterfall": not has_head
        or metrics["cascade"].get("supported") is True,
        "face_threshold_selection": (
            not has_face
            or (
                face_threshold_selection.get("status") == "success"
                and
                face_threshold_selection.get("selected_on_split") == "val"
                and float(face_threshold_selection["face_threshold"])
                >= float(face_threshold_selection["minimum_threshold"])
            )
            or (
                face_threshold_selection.get("status") == "unsupported"
                and face_threshold_selection.get("selection_attempted_on_split")
                == "val"
                and face_threshold_selection.get("unsupported_reason")
                == "no_validation_face_threshold_satisfies_precision_gt_0.90"
                and int(face_threshold_selection.get("eligible_threshold_count", -1))
                == 0
            )
        ),
    }
    if not all(assertions.values()):
        raise RuntimeError(f"Baseline-aligned metric assertion failed: {assertions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    applied_face_threshold_selection = {
        **face_threshold_selection,
        "applied_on_split": split,
        "application_source": selection_application_source,
        "validation_selection_artifact": (
            str(validation_selection_path)
            if split == "test" and validation_selection_path.is_file()
            else None
        ),
    }
    (output_dir / FACE_THRESHOLD_SELECTION_FILENAME).write_text(
        json.dumps(
            applied_face_threshold_selection,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"# Baseline-aligned comparison: {architecture_id} / {split}",
        "",
    ]
    if local_face_bbox_markdown is not None:
        lines.extend(local_face_bbox_markdown)
    elif "primary_local_face_bbox" in metrics:
        from .continuous_face_contract import markdown_rows

        lines.extend(markdown_rows(metrics["primary_local_face_bbox"]))
    baseline_roles = dict((provenance or {}).get("baseline_roles", {}))
    if baseline_roles:
        lines.extend(
            [
                "Baseline roles:",
                "",
                *[
                    f"- `{role}`: {value}"
                    for role, value in baseline_roles.items()
                ],
                "",
            ]
        )
    lines.extend([
        "| Metric | Face DINO report | Candidate (common Face 0.33) | Candidate (Val-selected P>0.90 OP) | Common Δ abs | Common Δ rel | Val-selected Δ abs | Val-selected Δ rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison:
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| `{row['metric']}` | {fmt(row['baseline'])} | "
            f"{fmt(row['candidate'])} | "
            f"{fmt(row['validation_selected_candidate'])} | "
            f"{fmt(row['absolute_delta'])} | "
            f"{fmt(row['relative_delta'])} | "
            f"{fmt(row['validation_selected_absolute_delta'])} | "
            f"{fmt(row['validation_selected_relative_delta'])} |"
        )
    common_operating_point = (
        "Head 0.30, Face 0.33, and point presence 0.70"
        if has_head
        else "Face 0.33 and point presence 0.70"
    )
    operating_point_note = (
        (
            f"The common operating point is {common_operating_point}. No "
            "validation Face threshold emitted a prediction "
            "while satisfying strict precision > 0.90, so the Val-selected "
            "operating point is reported as N/A and Test does not select a "
            "replacement threshold."
        )
        if has_face and face_threshold_selection.get("status") == "unsupported"
        else (
        "This HEAD-only architecture does not output FACE predictions, so "
        "FACE threshold selection is not applicable."
        if not has_face
        else (
            f"The common operating point is {common_operating_point}. The "
            "second operating point uses Face threshold "
            f"{float(face_threshold_selection['face_threshold']):.8g}, selected "
            "on Val by maximizing recall subject to precision > 0.90 among "
            + (
                "matched predicted HEADs"
                if has_head
                else "all supervised GT Heads (the declared pipeline has no HEAD stage)"
            )
            + ", then frozen before Test. NME uses "
            "1.20 × max(GT Head width, height)."
        )
    )
    )
    lines.extend([
        "",
        operating_point_note,
        "",
        "The historical Face DINO report column used Face 0.80; FACE-sensitive "
        "deltas against that column are descriptive rather than "
        "threshold-aligned. Direct architecture comparison must use the "
        "common 0.33 column or each model's Val-selected column consistently.",
        "",
        "Unsupported outputs remain N/A in this table; see `unsupported_reason` in metrics.json.",
    ])
    (output_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "architecture_id": architecture_id,
        "split": split,
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "face_threshold_selection_sha256": sha256_file(
            output_dir / FACE_THRESHOLD_SELECTION_FILENAME
        ),
        "inventory_verified": not allow_partial_inventory,
        "supported_sections": {"head": has_head, "face": has_face, "ellipse": has_ellipse, "keypoint": has_keypoint, "state": has_state},
        "assertions": assertions,
    }
    (output_dir / "success.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if has_head and has_keypoint:
        (output_dir / "cascade_waterfall_success.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return metrics
