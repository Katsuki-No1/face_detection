from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class LandmarkOcclusionConfig:
    probability: float = 0.30
    regular_probability: float = 0.50
    landmark_anchor_probability: float = 0.70
    minimum_overlap: float = 0.10
    maximum_overlap: float = 0.40
    center_offset_fraction: float = 0.12
    maximum_attempts: int = 128

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "LandmarkOcclusionConfig":
        raw = dict(value or {})
        overlap = raw.get("face_overlap", (0.10, 0.40))
        if not isinstance(overlap, (list, tuple)) or len(overlap) != 2:
            raise ValueError("landmark occlusion face_overlap must contain [min, max]")
        config = cls(
            probability=float(raw.get("probability", 0.30)),
            regular_probability=float(raw.get("regular_probability", 0.50)),
            landmark_anchor_probability=float(
                raw.get("landmark_anchor_probability", 0.70)
            ),
            minimum_overlap=float(overlap[0]),
            maximum_overlap=float(overlap[1]),
            center_offset_fraction=float(raw.get("center_offset_fraction", 0.12)),
            maximum_attempts=int(raw.get("maximum_attempts", 128)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, value in (
            ("probability", self.probability),
            ("regular_probability", self.regular_probability),
            ("landmark_anchor_probability", self.landmark_anchor_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"landmark occlusion {name} must be in [0, 1]")
        if not 0.0 < self.minimum_overlap <= self.maximum_overlap < 1.0:
            raise ValueError(
                "landmark occlusion face_overlap must satisfy 0 < min <= max < 1"
            )
        if not 0.0 <= self.center_offset_fraction <= 0.50:
            raise ValueError(
                "landmark occlusion center_offset_fraction must be in [0, 0.5]"
            )
        if self.maximum_attempts < 1:
            raise ValueError("landmark occlusion maximum_attempts must be positive")


def _ellipse_polygon(ellipse: Sequence[float], *, steps: int = 192) -> np.ndarray:
    cx, cy, rx, ry, theta = (float(value) for value in ellipse)
    angles = np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    local_x = rx * np.cos(angles)
    local_y = ry * np.sin(angles)
    return np.stack(
        [
            cx + local_x * cos_theta - local_y * sin_theta,
            cy + local_x * sin_theta + local_y * cos_theta,
        ],
        axis=1,
    ).astype(np.float32)


def ellipse_region_mask(
    height: int,
    width: int,
    ellipse: Sequence[float],
    *,
    valid_pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize the on-canvas portion of an ellipse as the face denominator."""

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if height <= 0 or width <= 0 or len(ellipse) != 5:
        return mask.astype(np.bool_)
    values = [float(value) for value in ellipse]
    if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
        return mask.astype(np.bool_)
    polygon = np.rint(_ellipse_polygon(values)).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1, lineType=cv2.LINE_8)
    result = mask.astype(np.bool_)
    if valid_pixel_mask is not None:
        valid = np.asarray(valid_pixel_mask, dtype=np.bool_)
        if valid.shape != result.shape:
            raise ValueError(
                f"valid_pixel_mask shape must be {result.shape}, got {valid.shape}"
            )
        result &= valid
    return result


def transform_ellipse_affine(
    ellipse: Sequence[float], matrix: np.ndarray
) -> list[float]:
    """Transform an ellipse through a 2x3 affine using a contour fit."""

    points = _ellipse_polygon(ellipse, steps=192)
    affine = np.asarray(matrix, dtype=np.float32)
    if affine.shape != (2, 3):
        raise ValueError(f"affine matrix must be 2x3, got {affine.shape}")
    transformed = cv2.transform(points.reshape(1, -1, 2), affine).reshape(-1, 2)
    (cx, cy), (diameter_a, diameter_b), angle_degrees = cv2.fitEllipse(
        transformed.astype(np.float32)
    )
    return [
        float(cx),
        float(cy),
        max(0.5, float(diameter_a) * 0.5),
        max(0.5, float(diameter_b) * 0.5),
        math.radians(float(angle_degrees)),
    ]


def _random_point_in_ellipse(
    ellipse: Sequence[float], rng: random.Random
) -> tuple[float, float]:
    cx, cy, rx, ry, theta = (float(value) for value in ellipse)
    radius = math.sqrt(rng.random())
    angle = rng.uniform(0.0, 2.0 * math.pi)
    local_x = rx * radius * math.cos(angle)
    local_y = ry * radius * math.sin(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        cx + local_x * cos_theta - local_y * sin_theta,
        cy + local_x * sin_theta + local_y * cos_theta,
    )


def _offset_anchor(
    anchor: tuple[float, float],
    ellipse: Sequence[float],
    fraction: float,
    rng: random.Random,
) -> tuple[float, float]:
    _cx, _cy, rx, ry, theta = (float(value) for value in ellipse)
    local_x = rng.uniform(-fraction, fraction) * rx
    local_y = rng.uniform(-fraction, fraction) * ry
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        anchor[0] + local_x * cos_theta - local_y * sin_theta,
        anchor[1] + local_x * sin_theta + local_y * cos_theta,
    )


def _regular_candidate(
    height: int,
    width: int,
    *,
    anchor: tuple[float, float],
    desired_pixels: float,
    rng: random.Random,
    force_shape: str | None = None,
) -> tuple[np.ndarray, str]:
    mask = np.zeros((height, width), dtype=np.uint8)
    shape = force_shape or ("rectangle" if rng.random() < 0.5 else "ellipse")
    if shape not in {"rectangle", "ellipse"}:
        raise ValueError(f"Unsupported regular occlusion shape: {shape!r}")
    aspect = math.exp(rng.uniform(math.log(0.45), math.log(2.20)))
    area_scale = rng.uniform(0.85, 1.75)
    area = max(4.0, desired_pixels * area_scale)
    if shape == "ellipse":
        mask_width = math.sqrt(area * 4.0 * aspect / math.pi)
        mask_height = math.sqrt(area * 4.0 / (math.pi * aspect))
    else:
        mask_width = math.sqrt(area * aspect)
        mask_height = math.sqrt(area / aspect)
    angle = rng.uniform(0.0, 180.0)
    center = (float(anchor[0]), float(anchor[1]))
    if shape == "ellipse":
        cv2.ellipse(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            (
                max(1, int(round(mask_width * 0.5))),
                max(1, int(round(mask_height * 0.5))),
            ),
            angle,
            0.0,
            360.0,
            1,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
    else:
        rectangle = (center, (float(mask_width), float(mask_height)), float(angle))
        corners = np.rint(cv2.boxPoints(rectangle)).astype(np.int32)
        cv2.fillConvexPoly(mask, corners, 1, lineType=cv2.LINE_8)
    return mask.astype(np.bool_), shape


def _boundary_candidate(
    height: int,
    width: int,
    *,
    face_mask: np.ndarray,
    desired_pixels: float,
    rng: random.Random,
) -> tuple[np.ndarray, str]:
    """Mask a strip entering from one image edge, simulating truncation."""

    edge = rng.choice(("left", "right", "top", "bottom"))
    maximum = width if edge in {"left", "right"} else height
    best_mask = np.zeros((height, width), dtype=np.bool_)
    best_error = float("inf")
    for thickness in range(1, maximum + 1):
        candidate = np.zeros((height, width), dtype=np.bool_)
        if edge == "left":
            candidate[:, :thickness] = True
        elif edge == "right":
            candidate[:, width - thickness :] = True
        elif edge == "top":
            candidate[:thickness, :] = True
        else:
            candidate[height - thickness :, :] = True
        overlap_pixels = int(np.logical_and(candidate, face_mask).sum())
        error = abs(float(overlap_pixels) - desired_pixels)
        if error < best_error:
            best_error = error
            best_mask = candidate
        if overlap_pixels >= desired_pixels:
            break
    return best_mask, f"boundary_{edge}"


def _irregular_candidate(
    height: int,
    width: int,
    *,
    anchor: tuple[float, float],
    ellipse: Sequence[float],
    desired_pixels: float,
    rng: random.Random,
) -> tuple[np.ndarray, str]:
    _cx, _cy, rx, ry, _theta = (float(value) for value in ellipse)
    segments = rng.randint(3, 7)
    characteristic = max(2.0, math.sqrt(max(1.0, rx * ry)))
    step = rng.uniform(0.18, 0.48) * characteristic
    base_angle = rng.uniform(0.0, 2.0 * math.pi)
    points = [np.asarray(anchor, dtype=np.float32)]
    for _ in range(segments):
        base_angle += rng.uniform(-1.20, 1.20)
        distance = step * rng.uniform(0.60, 1.40)
        points.append(
            points[-1]
            + np.asarray(
                [math.cos(base_angle) * distance, math.sin(base_angle) * distance],
                dtype=np.float32,
            )
        )
    center = np.mean(points, axis=0)
    shift = np.asarray(anchor, dtype=np.float32) - center
    polyline = np.rint(np.stack(points) + shift).astype(np.int32)
    total_length = float(
        np.linalg.norm(np.diff(polyline.astype(np.float32), axis=0), axis=1).sum()
    )
    thickness = int(
        round(
            desired_pixels
            * rng.uniform(0.90, 1.55)
            / max(1.0, total_length)
        )
    )
    thickness = max(2, min(thickness, int(round(2.0 * max(rx, ry)))))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(
        mask,
        [polyline],
        isClosed=False,
        color=1,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    return mask.astype(np.bool_), "thick_polyline"


def generate_landmark_occlusion_mask(
    *,
    height: int,
    width: int,
    face_ellipse: Sequence[float],
    keypoints: np.ndarray,
    keypoint_visibility: np.ndarray,
    config: Mapping[str, Any] | LandmarkOcclusionConfig,
    rng: random.Random,
    valid_pixel_mask: np.ndarray | None = None,
    force_kind: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Generate one A1/A2 mask whose on-face area is in the configured range."""

    cfg = (
        config
        if isinstance(config, LandmarkOcclusionConfig)
        else LandmarkOcclusionConfig.from_mapping(config)
    )
    face_mask = ellipse_region_mask(
        height,
        width,
        face_ellipse,
        valid_pixel_mask=valid_pixel_mask,
    )
    face_pixels = int(face_mask.sum())
    if face_pixels < 16:
        return None
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    states = np.asarray(keypoint_visibility, dtype=np.int64).reshape(-1)
    labeled_indices = [
        index
        for index, (point, state) in enumerate(zip(points, states))
        if state > 0 and np.isfinite(point).all()
    ]
    use_landmark = bool(
        labeled_indices and rng.random() < cfg.landmark_anchor_probability
    )
    if use_landmark:
        landmark_index = rng.choice(labeled_indices)
        base_anchor = (float(points[landmark_index, 0]), float(points[landmark_index, 1]))
        anchor_source = "landmark"
    else:
        landmark_index = None
        base_anchor = _random_point_in_ellipse(face_ellipse, rng)
        anchor_source = "ellipse_random"
    anchor = _offset_anchor(
        base_anchor,
        face_ellipse,
        cfg.center_offset_fraction,
        rng,
    )
    desired_overlap = rng.uniform(cfg.minimum_overlap, cfg.maximum_overlap)
    desired_pixels = desired_overlap * face_pixels
    force_shape: str | None = None
    if force_kind is None:
        kind = "A1_regular" if rng.random() < cfg.regular_probability else "A2_irregular"
    else:
        normalized = force_kind.strip().lower()
        if normalized not in {
            "regular",
            "rectangle",
            "irregular",
            "boundary",
            "a1_regular",
            "a2_irregular",
        }:
            raise ValueError(f"Unsupported landmark occlusion kind: {force_kind!r}")
        if normalized == "rectangle":
            kind = "paired_rectangle"
            force_shape = "rectangle"
        elif normalized == "boundary":
            kind = "paired_boundary"
        else:
            kind = "A1_regular" if "regular" in normalized and "irregular" not in normalized else "A2_irregular"

    best: tuple[np.ndarray, str, float] | None = None
    for _attempt in range(1, cfg.maximum_attempts + 1):
        if kind in {"A1_regular", "paired_rectangle"}:
            candidate, shape = _regular_candidate(
                height,
                width,
                anchor=anchor,
                desired_pixels=desired_pixels,
                rng=rng,
                force_shape=force_shape,
            )
        elif kind == "paired_boundary":
            candidate, shape = _boundary_candidate(
                height,
                width,
                face_mask=face_mask,
                desired_pixels=desired_pixels,
                rng=rng,
            )
        else:
            candidate, shape = _irregular_candidate(
                height,
                width,
                anchor=anchor,
                ellipse=face_ellipse,
                desired_pixels=desired_pixels,
                rng=rng,
            )
        if valid_pixel_mask is not None:
            candidate &= np.asarray(valid_pixel_mask, dtype=np.bool_)
        overlap_pixels = int(np.logical_and(candidate, face_mask).sum())
        overlap = overlap_pixels / float(face_pixels)
        if best is None or abs(overlap - desired_overlap) < abs(best[2] - desired_overlap):
            best = (candidate, shape, overlap)
        if cfg.minimum_overlap <= overlap <= cfg.maximum_overlap:
            occluded_keypoints = []
            for point_index, (point, state) in enumerate(zip(points, states)):
                if state <= 0 or not np.isfinite(point).all():
                    continue
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                if 0 <= x < width and 0 <= y < height and bool(candidate[y, x]):
                    occluded_keypoints.append(point_index)
            return candidate, {
                "applied": True,
                "kind": kind,
                "shape": shape,
                "anchor_source": anchor_source,
                "anchor_keypoint_index": landmark_index,
                "anchor_xy": [float(anchor[0]), float(anchor[1])],
                "target_face_overlap": float(desired_overlap),
                "face_overlap": float(overlap),
                "face_pixels": face_pixels,
                "masked_face_pixels": overlap_pixels,
                "occluded_keypoint_indices": occluded_keypoints,
                "attempts": _attempt,
            }
    _ = best
    return None


def _random_fill(rng: random.Random, channels: int) -> np.ndarray:
    return np.asarray(
        [rng.randint(0, 255) for _ in range(channels)], dtype=np.float32
    ) / 255.0


def apply_landmark_occlusion_numpy(
    image: np.ndarray,
    *,
    face_ellipse: Sequence[float],
    keypoints: np.ndarray,
    keypoint_visibility: np.ndarray,
    config: Mapping[str, Any] | LandmarkOcclusionConfig,
    rng: random.Random,
    valid_pixel_mask: np.ndarray | None = None,
    force: bool = False,
    force_kind: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = (
        config
        if isinstance(config, LandmarkOcclusionConfig)
        else LandmarkOcclusionConfig.from_mapping(config)
    )
    empty = np.zeros(image.shape[:2], dtype=np.bool_)
    states = np.asarray(keypoint_visibility, dtype=np.int64).copy()
    if not force and (cfg.probability <= 0.0 or rng.random() >= cfg.probability):
        return image, empty, states, {"applied": False, "reason": "probability"}
    generated = generate_landmark_occlusion_mask(
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        face_ellipse=face_ellipse,
        keypoints=np.asarray(keypoints),
        keypoint_visibility=states,
        config=cfg,
        rng=rng,
        valid_pixel_mask=valid_pixel_mask,
        force_kind=force_kind,
    )
    if generated is None:
        return image, empty, states, {"applied": False, "reason": "no_valid_mask"}
    mask, metadata = generated
    result = np.asarray(image).copy()
    channels = 1 if result.ndim == 2 else int(result.shape[2])
    fill = _random_fill(rng, channels)
    if np.issubdtype(result.dtype, np.integer):
        fill_value = np.rint(fill * 255.0).astype(result.dtype)
    else:
        fill_value = fill.astype(result.dtype)
    result[mask] = fill_value[0] if result.ndim == 2 else fill_value
    for point_index in metadata["occluded_keypoint_indices"]:
        if states[point_index] > 0:
            states[point_index] = 1
    metadata = {**metadata, "fill_rgb": [float(value) for value in fill.tolist()]}
    return result, mask, states, metadata


def apply_landmark_occlusion_tensor(
    image: torch.Tensor,
    target: Mapping[str, Any],
    config: Mapping[str, Any] | LandmarkOcclusionConfig,
    *,
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply A1/A2 to exactly one selected face in a full-image sample."""

    cfg = (
        config
        if isinstance(config, LandmarkOcclusionConfig)
        else LandmarkOcclusionConfig.from_mapping(config)
    )
    updated = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }
    ellipses = updated.get("ellipses")
    keypoints = updated.get("keypoints")
    keypoint_mask = updated.get("keypoint_mask")
    if (
        cfg.probability <= 0.0
        or rng.random() >= cfg.probability
        or not isinstance(ellipses, torch.Tensor)
        or not isinstance(keypoints, torch.Tensor)
        or ellipses.numel() == 0
    ):
        return image, updated
    height, width = int(image.shape[-2]), int(image.shape[-1])
    padding_mask = updated.get("padding_mask")
    valid_pixel_mask = None
    if isinstance(padding_mask, torch.Tensor):
        valid_pixel_mask = ~padding_mask.detach().cpu().numpy().astype(np.bool_)
    candidates = []
    for index, ellipse in enumerate(ellipses.detach().cpu().tolist()):
        face_mask = ellipse_region_mask(
            height,
            width,
            ellipse,
            valid_pixel_mask=valid_pixel_mask,
        )
        if int(face_mask.sum()) >= 16:
            candidates.append(index)
    if not candidates:
        return image, updated
    selected = rng.choice(candidates)
    if isinstance(updated.get("keypoint_visibility"), torch.Tensor):
        visibility = updated["keypoint_visibility"].detach().cpu().numpy().astype(np.int64)
    else:
        if isinstance(keypoint_mask, torch.Tensor):
            visibility = keypoint_mask.detach().cpu().numpy().astype(np.int64) * 2
        else:
            visibility = np.full(keypoints.shape[:2], 2, dtype=np.int64)
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    result, mask, selected_states, metadata = apply_landmark_occlusion_numpy(
        array,
        face_ellipse=ellipses[selected].detach().cpu().tolist(),
        keypoints=keypoints[selected].detach().cpu().numpy(),
        keypoint_visibility=visibility[selected],
        config=cfg,
        rng=rng,
        valid_pixel_mask=valid_pixel_mask,
        force=True,
    )
    if not bool(metadata.get("applied", False)):
        return image, updated
    visibility[selected] = selected_states
    updated["keypoint_visibility"] = torch.as_tensor(
        visibility,
        dtype=torch.long,
        device=keypoints.device,
    )
    face_visibility = updated.get("face_visibility")
    if not isinstance(face_visibility, torch.Tensor):
        face_visibility = torch.full(
            (ellipses.shape[0],), 2, dtype=torch.long, device=ellipses.device
        )
    else:
        face_visibility = face_visibility.clone()
    if int(face_visibility[selected]) > 0:
        face_visibility[selected] = 1
    updated["face_visibility"] = face_visibility
    updated["landmark_occlusion_mask"] = torch.from_numpy(mask).to(
        device=image.device
    )
    target_metadata = dict(
        updated.get("metadata", {})
        if isinstance(updated.get("metadata"), Mapping)
        else {}
    )
    debug = dict(
        target_metadata.get("augmentation_debug", {})
        if isinstance(target_metadata.get("augmentation_debug"), Mapping)
        else {}
    )
    debug["landmark_guided_occlusion"] = {
        **metadata,
        "selected_face_index": int(selected),
        "augmented_faces": 1,
    }
    target_metadata["augmentation_debug"] = debug
    updated["metadata"] = target_metadata
    output = torch.from_numpy(np.ascontiguousarray(result)).permute(2, 0, 1)
    return output.to(device=image.device, dtype=image.dtype), updated


__all__ = [
    "LandmarkOcclusionConfig",
    "apply_landmark_occlusion_numpy",
    "apply_landmark_occlusion_tensor",
    "ellipse_region_mask",
    "generate_landmark_occlusion_mask",
    "transform_ellipse_affine",
]
