"""Compatibility serialization for existing normalized five-point text labels.

This is a dataset format adapter, not a YOLO model dependency. Preserve the
historical 0/2 text encoding (including occluded points); source state flags
remain in metadata for the separate semantic 0/1/2 exports.
"""

from typing import Any

from src.data.face_annotations import (
    KEYPOINT_NAMES, assign_keypoints_with_audit, bboxes_differ, clip_bbox_to_image,
    normalize_bbox, point_inside_image,
)
from src.data.keypoint_side_assignment import POLICY

def normalized_pose_line_for_face(
    record: dict[str, Any],
    face: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
    *, side_policy: str = POLICY,
) -> tuple[str | None, dict[str, Any]]:
    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    metadata = {
        "warnings": [],
        "keypoints": {},
        "face": face,
        "cvat_ellipse": face.get("cvat_ellipse"),
    }
    if width <= 0 or height <= 0:
        metadata["warnings"].append("invalid_image_size")
        return None, metadata
    bbox = face["bbox"]
    clipped_bbox = clip_bbox_to_image(bbox, width, height)
    metadata["raw_bbox"] = bbox
    metadata["training_bbox"] = clipped_bbox
    if bboxes_differ(bbox, clipped_bbox):
        metadata["warnings"].append("face_bbox_clipped_to_image")
    bbox_values = normalize_bbox(clipped_bbox, width, height)
    if bbox_values[2] <= 0.0 or bbox_values[3] <= 0.0:
        metadata["warnings"].append("invalid_clipped_face_bbox")
        metadata["exclusion_reason"] = "invalid_clipped_face_bbox"
        return None, metadata

    keypoints, side_audit = assign_keypoints_with_audit(record, face_bboxes, face_index, side_policy=side_policy)
    metadata["keypoint_side_assignment"] = side_audit
    values: list[float | int] = [0, *bbox_values]
    for name in KEYPOINT_NAMES:
        point = keypoints[name]
        metadata["keypoints"][name] = point
        if point["v"] == 0:
            values.extend([0.0, 0.0, 0])
            continue
        if not point_inside_image(point, width, height):
            metadata["warnings"].append(f"{name}_outside_image")
            values.extend([0.0, 0.0, 0])
            continue
        nx = point["x"] / width
        ny = point["y"] / height
        values.extend([nx, ny, 2])

    return " ".join(format_label_value(value) for value in values), metadata


def format_label_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}".rstrip("0").rstrip(".")
