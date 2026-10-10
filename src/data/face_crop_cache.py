from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def canonicalize_ellipse(ellipse: Sequence[float]) -> list[float]:
    """Validate and canonicalize [cx, cy, major, minor, theta_rad]."""

    if len(ellipse) != 5:
        raise ValueError(f"Ellipse must contain [cx, cy, major, minor, theta]: {ellipse}")
    cx, cy, first_radius, second_radius, theta = (
        float(value) for value in ellipse
    )
    if not all(
        math.isfinite(value)
        for value in (cx, cy, first_radius, second_radius, theta)
    ):
        raise ValueError(f"Ellipse contains non-finite values: {ellipse}")
    if first_radius <= 0.0 or second_radius <= 0.0:
        raise ValueError(f"Ellipse radii must be positive: {ellipse}")
    if second_radius > first_radius:
        first_radius, second_radius = second_radius, first_radius
        theta += math.pi * 0.5
    theta = (theta + math.pi * 0.5) % math.pi - math.pi * 0.5
    return [cx, cy, first_radius, second_radius, theta]


def perturb_ellipse(
    ellipse: Sequence[float],
    *,
    center_offset: tuple[float, float],
    axis_scale: tuple[float, float],
    angle_offset_radians: float,
) -> list[float]:
    """Perturb center, both axes, and angle in the ellipse-local frame."""

    cx, cy, major, minor, theta = canonicalize_ellipse(ellipse)
    major_offset = float(center_offset[0]) * (2.0 * major)
    minor_offset = float(center_offset[1]) * (2.0 * minor)
    cosine, sine = math.cos(theta), math.sin(theta)
    cx += major_offset * cosine - minor_offset * sine
    cy += major_offset * sine + minor_offset * cosine
    return canonicalize_ellipse(
        (
            cx,
            cy,
            major * float(axis_scale[0]),
            minor * float(axis_scale[1]),
            theta + float(angle_offset_radians),
        )
    )


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0,
        min(ly2, ry2) - max(ly1, ry1),
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(1.0e-8, left_area + right_area - intersection)


def ellipse_aligned_rectangle(
    ellipse: Sequence[float],
) -> tuple[list[float], float]:
    """Return a tight axis-aware ROI and rotation aligned to an ellipse."""

    cx, cy, first_radius, second_radius, theta = canonicalize_ellipse(ellipse)
    return (
        [
            cx - first_radius,
            cy - second_radius,
            cx + first_radius,
            cy + second_radius,
        ],
        math.degrees(theta),
    )


def crop_geometry_from_detection(
    detection: Mapping[str, Any],
    *,
    require_ellipse: bool = True,
) -> tuple[list[float], float]:
    ellipse = detection.get("ellipse")
    if isinstance(ellipse, (list, tuple)) and len(ellipse) == 5:
        return ellipse_aligned_rectangle(ellipse)
    if require_ellipse:
        raise RuntimeError(
            "An ellipse-aligned crop was required, but the detector prediction "
            f"did not contain a valid ellipse: {detection}"
        )
    box = detection.get("crop_bbox_xyxy", detection.get("bbox_xyxy"))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RuntimeError(f"Detector prediction has no valid crop box: {detection}")
    return [float(value) for value in box], 0.0


def load_detector_prediction_cache(
    path: str | Path,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            "Required ellipse-DEIM crop cache is missing: "
            f"{cache_path}. Run keypoint_comparison2 preflight first."
        )
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    metadata = value.get("metadata")
    predictions = value.get("predictions")
    if not isinstance(metadata, dict) or not isinstance(predictions, dict):
        raise RuntimeError(f"Invalid detector crop cache schema: {cache_path}")
    if metadata.get("crop_geometry") not in {"ellipse", "ellipse_aligned"}:
        raise RuntimeError(
            f"Crop cache must provide ellipse geometry, got "
            f"{metadata.get('crop_geometry')!r}: {cache_path}"
        )
    return metadata, {
        int(image_id): [dict(row) for row in rows]
        for image_id, rows in predictions.items()
    }


def match_annotations_to_detector_crops(
    coco: Mapping[str, Any],
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    category_ids: Sequence[int],
    minimum_iou: float,
    require_ellipse: bool = True,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """One-to-one Hungarian assignment from GT instances to detector crops."""

    allowed = {int(value) for value in category_ids}
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    eligible = 0
    for annotation in coco.get("annotations", []):
        if int(annotation.get("category_id", -1)) not in allowed:
            continue
        bbox = annotation.get("bbox", ())
        if len(bbox) != 4 or float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)
        eligible += 1

    assignments: dict[int, dict[str, Any]] = {}
    matched_ious: list[float] = []
    for image_id, annotations in annotations_by_image.items():
        detections = list(predictions.get(image_id, ()))
        if not detections:
            continue
        gt_boxes = [
            [
                float(row["bbox"][0]),
                float(row["bbox"][1]),
                float(row["bbox"][0]) + float(row["bbox"][2]),
                float(row["bbox"][1]) + float(row["bbox"][3]),
            ]
            for row in annotations
        ]
        detection_boxes = [
            list(
                row.get(
                    "detector_bbox_xyxy",
                    row.get("bbox_xyxy", ()),
                )
            )
            for row in detections
        ]
        if any(len(box) != 4 for box in detection_boxes):
            raise RuntimeError(
                f"Detector cache contains an invalid matching box for image_id={image_id}"
            )
        overlaps = np.asarray(
            [
                [iou_xyxy(gt_box, detection_box) for detection_box in detection_boxes]
                for gt_box in gt_boxes
            ],
            dtype=np.float64,
        )
        gt_indices, detection_indices = linear_sum_assignment(1.0 - overlaps)
        for gt_index, detection_index in zip(
            gt_indices.tolist(),
            detection_indices.tolist(),
            strict=True,
        ):
            overlap = float(overlaps[gt_index, detection_index])
            if overlap < float(minimum_iou):
                continue
            detection = dict(detections[detection_index])
            crop_box, rotation = crop_geometry_from_detection(
                detection,
                require_ellipse=require_ellipse,
            )
            annotation_id = int(annotations[gt_index].get("id", -1))
            if annotation_id < 0:
                raise RuntimeError(
                    "Detector-crop training requires stable COCO annotation IDs"
                )
            assignments[annotation_id] = {
                **detection,
                "crop_bbox_xyxy": crop_box,
                "crop_rotation_degrees": rotation,
                "matching_iou": overlap,
            }
            matched_ious.append(overlap)
    return assignments, {
        "eligible_annotations": eligible,
        "matched_annotations": len(assignments),
        "unmatched_annotations": eligible - len(assignments),
        "coverage": len(assignments) / max(1, eligible),
        "mean_matching_iou": (
            float(np.mean(matched_ious)) if matched_ious else None
        ),
        "minimum_iou": float(minimum_iou),
        "crop_geometry": "ellipse_aligned" if require_ellipse else "detector",
    }
