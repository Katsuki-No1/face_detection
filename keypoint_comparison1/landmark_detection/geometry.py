from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    """Affine mapping from a source ROI into fixed model-input coordinates."""

    matrix: np.ndarray
    inverse_matrix: np.ndarray
    center_xy: tuple[float, float]
    side: float
    rotation_deg: float
    output_size: int

    def original_to_crop(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.matrix)

    def crop_to_original(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.inverse_matrix)


def _rotated(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    return np.asarray(
        [vector[0] * cosine - vector[1] * sine, vector[0] * sine + vector[1] * cosine],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.size == 0:
        return values.reshape(-1, 2).copy()
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {values.shape}")
    affine = np.asarray(matrix, dtype=np.float32)
    if affine.shape != (2, 3):
        raise ValueError(f"matrix must have shape (2, 3), got {affine.shape}")
    homogeneous = np.concatenate([values, np.ones((values.shape[0], 1), dtype=np.float32)], axis=1)
    return homogeneous @ affine.T


def build_crop_transform(
    bbox_xyxy: Sequence[float],
    *,
    output_size: int,
    scale_factor: float = 1.25,
    center_offset_xy: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    minimum_side: float = 2.0,
    warp_bbox_aspect: bool = False,
) -> CropTransform:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    if output_size <= 1:
        raise ValueError("output_size must be greater than one")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox dimensions: {bbox_xyxy}")
    base_side = max(width, height)
    side = max(float(minimum_side), base_side * float(scale_factor))
    if warp_bbox_aspect:
        source_width = max(
            float(minimum_side),
            width * float(scale_factor),
        )
        source_height = max(
            float(minimum_side),
            height * float(scale_factor),
        )
        center_scale_x, center_scale_y = width, height
    else:
        source_width = source_height = side
        center_scale_x = center_scale_y = base_side
    center = np.asarray(
        [
            (x1 + x2) * 0.5 + float(center_offset_xy[0]) * center_scale_x,
            (y1 + y2) * 0.5 + float(center_offset_xy[1]) * center_scale_y,
        ],
        dtype=np.float32,
    )
    source_up = _rotated(
        np.asarray([0.0, -source_height * 0.5], dtype=np.float32),
        rotation_deg,
    )
    source_right = _rotated(
        np.asarray([source_width * 0.5, 0.0], dtype=np.float32),
        rotation_deg,
    )
    target_center = np.asarray([(output_size - 1) * 0.5, (output_size - 1) * 0.5], dtype=np.float32)
    target_half = (output_size - 1) * 0.5
    source = np.stack([center, center + source_up, center + source_right]).astype(np.float32)
    target = np.stack(
        [
            target_center,
            target_center + np.asarray([0.0, -target_half], dtype=np.float32),
            target_center + np.asarray([target_half, 0.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    matrix = cv2.getAffineTransform(source, target).astype(np.float32)
    inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
    return CropTransform(
        matrix=matrix,
        inverse_matrix=inverse,
        center_xy=(float(center[0]), float(center[1])),
        side=side,
        rotation_deg=float(rotation_deg),
        output_size=int(output_size),
    )


def warp_crop(image_rgb: np.ndarray, transform: CropTransform) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {image.shape}")
    return cv2.warpAffine(
        image,
        transform.matrix,
        (transform.output_size, transform.output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def normalized_crop_to_original(coordinates: np.ndarray, transform: CropTransform) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2).copy()
    points *= float(transform.output_size - 1)
    return transform.crop_to_original(points)


def original_to_normalized_crop(points: np.ndarray, transform: CropTransform) -> np.ndarray:
    crop_points = transform.original_to_crop(points)
    return crop_points / float(transform.output_size - 1)
