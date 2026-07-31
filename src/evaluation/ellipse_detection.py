from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - pure-Python fallback remains available.
    cv2 = None


def ellipse_polygon(ellipse: list[float], *, steps: int = 96) -> list[tuple[float, float]]:
    cx, cy, major, minor, theta = [float(value) for value in ellipse]
    cosine = math.cos(theta)
    sine = math.sin(theta)
    points = []
    for index in range(steps):
        angle = 2.0 * math.pi * index / steps
        local_x = major * math.cos(angle)
        local_y = minor * math.sin(angle)
        points.append((cx + local_x * cosine - local_y * sine, cy + local_x * sine + local_y * cosine))
    return points


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) * 0.5


def _inside(point, edge_start, edge_end) -> bool:
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= -1.0e-9


def _intersection(first, second, edge_start, edge_end):
    dc = (edge_start[0] - edge_end[0], edge_start[1] - edge_end[1])
    dp = (first[0] - second[0], first[1] - second[1])
    n1 = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0]
    n2 = first[0] * second[1] - first[1] * second[0]
    denominator = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denominator) < 1.0e-12:
        return second
    return ((n1 * dp[0] - n2 * dc[0]) / denominator, (n1 * dp[1] - n2 * dc[1]) / denominator)


def convex_intersection(
    subject: list[tuple[float, float]],
    clipper: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = subject
    for edge_index in range(len(clipper)):
        edge_start = clipper[edge_index]
        edge_end = clipper[(edge_index + 1) % len(clipper)]
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


def ellipse_iou(first: list[float], second: list[float], *, steps: int = 96) -> float:
    first_polygon = ellipse_polygon(first, steps=steps)
    second_polygon = ellipse_polygon(second, steps=steps)
    if cv2 is not None:
        first_array = np.asarray(first_polygon, dtype=np.float32)
        second_array = np.asarray(second_polygon, dtype=np.float32)
        intersection = float(cv2.intersectConvexConvex(first_array, second_array)[0])
        union = float(cv2.contourArea(first_array)) + float(cv2.contourArea(second_array)) - intersection
        return min(1.0, max(0.0, intersection / union)) if union > 0.0 else 0.0
    intersection = polygon_area(convex_intersection(first_polygon, second_polygon))
    union = polygon_area(first_polygon) + polygon_area(second_polygon) - intersection
    return min(1.0, max(0.0, intersection / union)) if union > 0.0 else 0.0


def _average_precision(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float,
    ellipse_key: str,
    iou_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] | None = None,
) -> float:
    selected_gt = {
        image_id: [row for row in rows if int(row["category_id"]) == int(category_id) and row.get("ellipse") is not None]
        for image_id, rows in ground_truth.items()
    }
    total_gt = sum(len(rows) for rows in selected_gt.values())
    selected_predictions = [
        row
        for row in predictions
        if int(row["category_id"]) == int(category_id) and row.get(ellipse_key) is not None
    ]
    selected_predictions.sort(key=lambda row: float(row["score"]), reverse=True)
    if total_gt == 0 or not selected_predictions:
        return 0.0

    matched: dict[int, set[int]] = defaultdict(set)
    true_positives = []
    false_positives = []
    for prediction in selected_predictions:
        image_id = int(prediction["image_id"])
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(selected_gt.get(image_id, [])):
            if index in matched[image_id]:
                continue
            cache_key = (tuple(prediction[ellipse_key]), tuple(target["ellipse"]))
            if iou_cache is not None and cache_key in iou_cache:
                overlap = iou_cache[cache_key]
            else:
                overlap = ellipse_iou(prediction[ellipse_key], target["ellipse"])
                if iou_cache is not None:
                    iou_cache[cache_key] = overlap
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    true_positive_sum = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    false_positive_sum = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recall = true_positive_sum / max(1, total_gt)
    precision = true_positive_sum / np.maximum(1.0e-9, true_positive_sum + false_positive_sum)
    return float(
        np.mean(
            [
                float(precision[recall >= threshold].max()) if np.any(recall >= threshold) else 0.0
                for threshold in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _matched_geometry_metrics(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float,
    circle_ratio_threshold: float,
    iou_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] | None = None,
) -> dict[str, float]:
    selected_predictions = [row for row in predictions if float(row["score"]) >= score_threshold and row.get("ellipse") is not None]
    selected_predictions.sort(key=lambda row: float(row["score"]), reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    overlaps = []
    center_errors = []
    major_errors = []
    minor_errors = []
    angle_errors = []
    for prediction in selected_predictions:
        image_id = int(prediction["image_id"])
        candidates = ground_truth.get(image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(candidates):
            if index in matched[image_id] or target.get("ellipse") is None:
                continue
            if int(target["category_id"]) != int(prediction["category_id"]):
                continue
            cache_key = (tuple(prediction["ellipse"]), tuple(target["ellipse"]))
            if iou_cache is not None and cache_key in iou_cache:
                overlap = iou_cache[cache_key]
            else:
                overlap = ellipse_iou(prediction["ellipse"], target["ellipse"])
                if iou_cache is not None:
                    iou_cache[cache_key] = overlap
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index < 0 or best_iou < iou_threshold:
            continue
        matched[image_id].add(best_index)
        target = candidates[best_index]["ellipse"]
        predicted = prediction["ellipse"]
        overlaps.append(best_iou)
        normalization = max(1.0e-6, 2.0 * math.sqrt(max(1.0e-12, target[2] * target[3])))
        center_errors.append(math.hypot(predicted[0] - target[0], predicted[1] - target[1]) / normalization)
        major_errors.append(abs(predicted[2] - target[2]) / max(1.0e-6, target[2]))
        minor_errors.append(abs(predicted[3] - target[3]) / max(1.0e-6, target[3]))
        if target[2] / max(1.0e-6, target[3]) >= circle_ratio_threshold:
            difference = abs((predicted[4] - target[4]) % math.pi)
            difference = min(difference, math.pi - difference)
            angle_errors.append(math.degrees(difference))
    return {
        "mean_ellipse_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "ellipse_center_nme": float(np.mean(center_errors)) if center_errors else 0.0,
        "ellipse_major_relative_error": float(np.mean(major_errors)) if major_errors else 0.0,
        "ellipse_minor_relative_error": float(np.mean(minor_errors)) if minor_errors else 0.0,
        "ellipse_angle_mae_deg": float(np.mean(angle_errors)) if angle_errors else 0.0,
        "ellipse_matched": float(len(overlaps)),
        "ellipse_angle_samples": float(len(angle_errors)),
    }


def ellipse_detection_metrics(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    category_ids: list[int],
    fixed_threshold: float = 0.30,
    circle_ratio_threshold: float = 1.05,
) -> dict[str, float]:
    thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
    ellipse_iou_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}
    baseline_iou_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}
    ap_values = [
        _average_precision(
            ground_truth,
            predictions,
            category_id=category_id,
            iou_threshold=threshold,
            ellipse_key="ellipse",
            iou_cache=ellipse_iou_cache,
        )
        for category_id in category_ids
        for threshold in thresholds
    ]
    ap50 = [
        _average_precision(
            ground_truth,
            predictions,
            category_id=category_id,
            iou_threshold=0.50,
            ellipse_key="ellipse",
            iou_cache=ellipse_iou_cache,
        )
        for category_id in category_ids
    ]
    ap75 = [
        _average_precision(
            ground_truth,
            predictions,
            category_id=category_id,
            iou_threshold=0.75,
            ellipse_key="ellipse",
            iou_cache=ellipse_iou_cache,
        )
        for category_id in category_ids
    ]
    baseline_values = [
        _average_precision(
            ground_truth,
            predictions,
            category_id=category_id,
            iou_threshold=threshold,
            ellipse_key="bbox_ellipse",
            iou_cache=baseline_iou_cache,
        )
        for category_id in category_ids
        for threshold in thresholds
    ]
    geometry = _matched_geometry_metrics(
        ground_truth,
        predictions,
        score_threshold=fixed_threshold,
        iou_threshold=0.50,
        circle_ratio_threshold=circle_ratio_threshold,
        iou_cache=ellipse_iou_cache,
    )
    return {
        "ellipse_mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "ellipse_AP50": float(np.mean(ap50)) if ap50 else 0.0,
        "ellipse_AP75": float(np.mean(ap75)) if ap75 else 0.0,
        "bbox_ellipse_mAP": float(np.mean(baseline_values)) if baseline_values else 0.0,
        **geometry,
    }


__all__ = [
    "convex_intersection",
    "ellipse_detection_metrics",
    "ellipse_iou",
    "ellipse_polygon",
    "polygon_area",
]
