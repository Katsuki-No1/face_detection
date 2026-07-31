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


SCHEMA_VERSION = "face-dino-baseline-aligned-v1"
THRESHOLDS = {
    "head_score": 0.30,
    "head_max_detections": 100,
    "head_nms": False,
    "head_matching_iou": 0.50,
    "head_roi_scale": 1.20,
    "face_score": 0.80,
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
    "gt_head.face.supervised_heads",
    "gt_head.face.positive_heads",
    "gt_head.face.negative_heads",
    "gt_head.keypoint.micro_recall_0.05",
    "gt_head.keypoint.micro_recall_0.10",
    "gt_head.keypoint.matched_points_0.10",
    "gt_head.keypoint.visible_matched_points_0.10",
    "gt_head.keypoint.occluded_matched_points_0.10",
    "gt_head.state.evaluated_present_points",
    "matched_head.keypoint.micro_recall_0.05",
    "matched_head.keypoint.micro_recall_0.10",
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
    )
    for section, metric in (
        ("ellipse", "mean_iou"),
        ("ellipse", "p10_iou"),
        ("keypoint", "macro_f1_0.05"),
        ("keypoint", "macro_f1_0.10"),
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
        "face_present_recall": positive["recall"],
        "no_face_recall": negative["recall"],
        "false_face_rate": fp / max(1, fp + tn),
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
) -> dict[str, Any]:
    """Evaluate and write the canonical JSON, Markdown table, and marker."""

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    expected = EXPECTED[split]
    annotation_hash = sha256_file(annotation_path)
    if annotation_hash != expected["annotation_sha256"]:
        raise RuntimeError(f"{split} annotation SHA-256 mismatch: {annotation_hash}")
    image_by_id = {int(row["id"]): row for row in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    ignored = IGNORED_HEAD_IDS[split]
    rows: list[dict[str, Any]] = []
    matched_head_annotation_ids: set[int] = set()
    has_head = any(value.get("heads") for value in predictions_by_image.values())
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
            rows.append({
                "image_id": image_id,
                "image_width": float(image["width"]),
                "image_height": float(image["height"]),
                "annotation_id": annotation_id,
                "face_target": target,
                "face_score": face_score,
                "face_predicted": face_predicted,
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
    matched_rows = [row for row in rows if row["annotation_id"] in matched_head_annotation_ids]

    def operating_point(
        source_rows: Sequence[Mapping[str, Any]],
        *,
        face_threshold: float,
        point_threshold: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
            "face": _face_metrics(selected_rows),
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
            "face": _face_metrics(matched) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "ellipse": _ellipse_metrics(matched, has_ellipse) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "keypoint": _keypoint_metrics(matched, has_keypoint, point_threshold=point_threshold) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
            "state": _state_metrics(matched, has_state, point_threshold=point_threshold) if has_head else _unsupported("architecture_does_not_output_head_boxes"),
        }
        cases = {}
        for name, subset in _slice_rows(selected_rows).items():
            cases[name] = {
                "instances": len(subset),
                "ellipse": _ellipse_metrics(subset, has_ellipse),
                "keypoint": _keypoint_metrics(
                    subset,
                    has_keypoint,
                    point_threshold=point_threshold,
                ),
            }
        return selected_rows, gt, matched_sections, cases

    rows, gt_sections, matched_sections, hard_cases = operating_point(
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
        "thresholds": {**THRESHOLDS, "selection_source": "locked_baseline_report_and_validation_only"},
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
        },
        "head": _head_coco_metrics(coco, predictions_by_image, has_head),
        "gt_head": gt_sections,
        "matched_head": matched_sections,
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
    if validation_selected_face_threshold is not None:
        selected_point_threshold = (
            float(validation_selected_point_threshold)
            if validation_selected_point_threshold is not None
            else float(THRESHOLDS["point_presence"])
        )
        (
            _selected_rows,
            selected_gt,
            selected_matched,
            selected_hard_cases,
        ) = operating_point(
            rows,
            face_threshold=float(validation_selected_face_threshold),
            point_threshold=selected_point_threshold,
        )
        metrics["validation_selected_operating_point"] = {
            "selection_source": "validation_only",
            "test_selection_forbidden": True,
            "face_threshold": float(validation_selected_face_threshold),
            "point_threshold": selected_point_threshold,
            "gt_head": selected_gt,
            "matched_head": selected_matched,
            "hard_cases": selected_hard_cases,
        }
    flattened: dict[str, Any] = {}
    for key in ("head", "gt_head", "matched_head", "hard_cases", "runtime"):
        _flatten_supported(key, metrics[key], flattened)
    validation_selected_flattened: dict[str, Any] = {}
    selected = metrics.get("validation_selected_operating_point")
    if isinstance(selected, Mapping):
        for key in ("gt_head", "matched_head", "hard_cases"):
            _flatten_supported(
                key,
                selected[key],
                validation_selected_flattened,
            )
    comparison = []
    baseline_reference = BASELINE_VAL if split == "val" else BASELINE_TEST
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
        })
    metrics["comparison_to_face_dino_report"] = comparison
    metrics["baseline_reference_split"] = split
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

    assert_finite(metrics)
    assertions = {
        "inventory": True,
        "finite_values": True,
        "head_metrics": not has_head or metrics["head"].get("supported") is True,
        "ellipse_metrics": not has_ellipse or metrics["gt_head"]["ellipse"].get("supported") is True,
        "keypoint_metrics": not has_keypoint or metrics["gt_head"]["keypoint"].get("supported") is True,
        "state_metrics": not has_state or metrics["gt_head"]["state"].get("supported") is True,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"Baseline-aligned metric assertion failed: {assertions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"# Baseline-aligned comparison: {architecture_id} / {split}",
        "",
        "| Metric | Face DINO report | Candidate (same locked OP) | Candidate (Val-selected OP) | Locked delta | Val-selected delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| `{row['metric']}` | {fmt(row['baseline'])} | "
            f"{fmt(row['candidate'])} | "
            f"{fmt(row['validation_selected_candidate'])} | "
            f"{fmt(row['absolute_delta'])} | "
            f"{fmt(row['validation_selected_absolute_delta'])} |"
        )
    lines.extend([
        "",
        "The directly comparable operating point is locked at Head 0.30, "
        "Face 0.80, and point presence 0.70. A separately labeled "
        "validation-selected operating point is also shown when supplied. "
        "NME uses 1.20 × max(GT Head width, height); Test was never used for selection.",
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
        "inventory_verified": not allow_partial_inventory,
        "supported_sections": {"head": has_head, "ellipse": has_ellipse, "keypoint": has_keypoint, "state": has_state},
        "assertions": assertions,
    }
    (output_dir / "success.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
