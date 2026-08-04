from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

DERIVED_LABEL_SCHEMA_VERSION = 1


def ellipse_contour(
    ellipse: Sequence[float],
    *,
    points: int = 720,
) -> np.ndarray:
    cx, cy, major, minor, theta = (float(value) for value in ellipse)
    if not all(math.isfinite(value) for value in (cx, cy, major, minor, theta)):
        raise ValueError(f"Non-finite ellipse: {ellipse}")
    if major <= 0.0 or minor <= 0.0:
        raise ValueError(f"Ellipse axes must be positive: {ellipse}")
    angles = np.arange(points, dtype=np.float64) * (2.0 * math.pi / points)
    cosine, sine = math.cos(theta), math.sin(theta)
    local_x = major * np.cos(angles)
    local_y = minor * np.sin(angles)
    return np.stack(
        (
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        ),
        axis=-1,
    )


def _clip_polygon_half_plane(
    polygon: np.ndarray,
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    if polygon.shape[0] == 0:
        return polygon
    output = []
    previous = polygon[-1]
    previous_inside = (
        previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
    )
    for current in polygon:
        current_inside = (
            current[axis] >= boundary if keep_greater else current[axis] <= boundary
        )
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            fraction = (
                0.0
                if abs(float(delta)) < 1.0e-12
                else (boundary - previous[axis]) / delta
            )
            intersection = previous + fraction * (current - previous)
            intersection[axis] = boundary
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def clip_polygon_to_image(
    polygon: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    for axis, boundary, keep_greater in (
        (0, 0.0, True),
        (0, float(width), False),
        (1, 0.0, True),
        (1, float(height), False),
    ):
        output = _clip_polygon_half_plane(
            output,
            axis=axis,
            boundary=boundary,
            keep_greater=keep_greater,
        )
    return output


def polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ellipse_visible_geometry(
    ellipse: Sequence[float],
    *,
    width: int,
    height: int,
    contour_points: int = 720,
) -> dict[str, Any]:
    contour = ellipse_contour(ellipse, points=contour_points)
    visible = clip_polygon_to_image(contour, width=width, height=height)
    full_area = math.pi * float(ellipse[2]) * float(ellipse[3])
    visible_area = polygon_area(visible)
    if visible.shape[0]:
        visible_aabb = [
            float(visible[:, 0].min()),
            float(visible[:, 1].min()),
            float(visible[:, 0].max()),
            float(visible[:, 1].max()),
        ]
    else:
        visible_aabb = [0.0, 0.0, 0.0, 0.0]
    full_min = contour.min(axis=0)
    full_max = contour.max(axis=0)
    return {
        "full_ellipse": [float(value) for value in ellipse],
        "full_aabb_xyxy": [
            float(full_min[0]),
            float(full_min[1]),
            float(full_max[0]),
            float(full_max[1]),
        ],
        "visible_polygon": visible.tolist(),
        "visible_aabb_xyxy": visible_aabb,
        "full_area": full_area,
        "visible_area": visible_area,
        "visible_area_ratio": min(
            1.0, max(0.0, visible_area / max(full_area, 1.0e-12))
        ),
        "truncation": {
            "left": bool(full_min[0] < 0.0),
            "right": bool(full_max[0] > width),
            "top": bool(full_min[1] < 0.0),
            "bottom": bool(full_max[1] > height),
        },
    }


def mask_to_rle(mask: torch.Tensor) -> dict[str, Any]:
    value = mask.detach().cpu().to(dtype=torch.uint8).reshape(-1).tolist()
    counts = []
    previous = 0
    run = 0
    for item in value:
        if int(item) == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = int(item)
    counts.append(run)
    return {
        "size": list(mask.shape[-2:]),
        "order": "row_major",
        "starts_with": 0,
        "counts": counts,
    }


def rle_to_mask(payload: dict[str, Any]) -> torch.Tensor:
    values: list[int] = []
    state = int(payload.get("starts_with", 0))
    for count in payload["counts"]:
        values.extend([state] * int(count))
        state = 1 - state
    height, width = (int(value) for value in payload["size"])
    if len(values) != height * width:
        raise ValueError("RLE count does not match mask size")
    return torch.tensor(values, dtype=torch.bool).reshape(height, width)


def derive_instance_labels(
    *,
    ellipse: Sequence[float],
    head_box_xyxy: Sequence[float],
    image_size: tuple[int, int],
    keypoints_xyv: Sequence[Sequence[float]],
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    height, width = (int(value) for value in image_size)
    visible_geometry = ellipse_visible_geometry(
        ellipse,
        width=width,
        height=height,
    )
    hx1, hy1, hx2, hy2 = (float(value) for value in head_box_xyxy)
    head_width = max(1.0e-6, hx2 - hx1)
    head_height = max(1.0e-6, hy2 - hy1)
    head_center = (0.5 * (hx1 + hx2), 0.5 * (hy1 + hy2))
    cx, cy, major, minor, _ = (float(value) for value in ellipse)
    states = [int(point[2]) for point in keypoints_xyv]
    labeled = [state > 0 for state in states]
    visible = [state > 1 for state in states]
    roll_evidence = None
    if len(keypoints_xyv) >= 2 and labeled[0] and labeled[1]:
        left, right = keypoints_xyv[0], keypoints_xyv[1]
        roll_evidence = math.atan2(
            float(right[1]) - float(left[1]),
            float(right[0]) - float(left[0]),
        )
    border_distances = [
        hx1 / max(1.0, width),
        hy1 / max(1.0, height),
        (width - hx2) / max(1.0, width),
        (height - hy2) / max(1.0, height),
    ]
    return {
        "schema_version": DERIVED_LABEL_SCHEMA_VERSION,
        **visible_geometry,
        "face_center_relative_to_head": [
            (cx - head_center[0]) / head_width,
            (cy - head_center[1]) / head_height,
        ],
        "face_axes_relative_to_head": [
            major / head_width,
            minor / head_height,
        ],
        "keypoint_states": states,
        "keypoint_labeled_mask": labeled,
        "keypoint_visible_mask": visible,
        "number_labeled_keypoints": sum(labeled),
        "number_visible_keypoints": sum(visible),
        "border_distances": border_distances,
        "weak_image_roll_radians": roll_evidence,
        "valid_pixel_mask_rle": (
            mask_to_rle(valid_mask) if valid_mask is not None else None
        ),
    }


__all__ = [
    "DERIVED_LABEL_SCHEMA_VERSION",
    "clip_polygon_to_image",
    "derive_instance_labels",
    "ellipse_contour",
    "ellipse_visible_geometry",
    "mask_to_rle",
    "polygon_area",
    "rle_to_mask",
]
