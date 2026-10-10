from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .landmark_occlusion import apply_landmark_occlusion_tensor


DEFAULT_KEYPOINT_FLIP_INDEX = (1, 0, 2, 4, 3)


def _enabled(value: Mapping[str, Any] | None) -> bool:
    return bool(value and value.get("enabled", False))


def has_active_augmentation(config: Mapping[str, Any] | None) -> bool:
    if not _enabled(config):
        return False
    numeric_keys = (
        "mosaic",
        "mixup",
        "ellipse_border_crop",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "brightness",
        "contrast",
        "gamma",
        "blur",
        "noise",
    )
    if any(float(config.get(key, 0.0) or 0.0) > 0.0 for key in numeric_keys):
        return True
    landmark_occlusion = config.get("landmark_guided_occlusion")
    return bool(
        isinstance(landmark_occlusion, Mapping)
        and float(landmark_occlusion.get("probability", 0.0) or 0.0) > 0.0
    )


def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _blur_radius_range(cfg: Mapping[str, Any]) -> tuple[float, float]:
    radius = cfg.get("blur_radius")
    if isinstance(radius, (list, tuple)) and len(radius) == 2:
        low = float(radius[0])
        high = float(radius[1])
    elif radius is not None:
        low = high = float(radius)
    else:
        low = float(cfg.get("blur_radius_min", 0.1) or 0.1)
        high = float(cfg.get("blur_radius_max", 1.5) or 1.5)
    low = max(0.0, low)
    high = max(low, high)
    return low, high


def _clone_target(target: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in target.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _padding_mask(
    target: Mapping[str, Any], *, height: int, width: int
) -> torch.Tensor:
    value = target.get("padding_mask")
    if isinstance(value, torch.Tensor):
        mask = value.to(dtype=torch.bool)
        if tuple(mask.shape) != (height, width):
            raise ValueError(
                f"padding_mask shape must match image {(height, width)}, got {tuple(mask.shape)}"
            )
        return mask
    return torch.zeros((height, width), dtype=torch.bool)


def _resize_padding_mask(
    mask: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    return (
        F.interpolate(
            mask.to(dtype=torch.float32).reshape(1, 1, *mask.shape),
            size=(height, width),
            mode="nearest",
        )
        .reshape(height, width)
        .to(dtype=torch.bool)
    )


def _set_padding_mask(target: dict[str, Any], mask: torch.Tensor) -> None:
    mask = mask.to(dtype=torch.bool)
    target["padding_mask"] = mask
    valid = torch.nonzero(~mask, as_tuple=False)
    if valid.numel() == 0:
        region = [0.0, 0.0, 0.0, 0.0]
    else:
        y1, x1 = valid.min(dim=0).values.tolist()
        y2, x2 = (valid.max(dim=0).values + 1).tolist()
        region = [float(x1), float(y1), float(x2), float(y2)]
    target["valid_image_region_xyxy"] = torch.tensor(region, dtype=torch.float32)


def _restore_padding_pixels(
    image: torch.Tensor,
    target: Mapping[str, Any],
    *,
    pad_color: Any,
) -> torch.Tensor:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    mask = _padding_mask(target, height=height, width=width)
    if not bool(mask.any()):
        return image
    if isinstance(pad_color, int):
        channels = (pad_color, pad_color, pad_color)
    elif isinstance(pad_color, (list, tuple)) and len(pad_color) == 3:
        channels = tuple(int(value) for value in pad_color)
    else:
        channels = (124, 116, 104)
    restored = image.clone()
    fill = restored.new_tensor(channels).div_(255.0).reshape(3, 1)
    restored[:, mask] = fill
    return restored


def _empty_like_target(target: Mapping[str, Any], *, image_id: str, height: int, width: int) -> dict[str, Any]:
    num_keypoints = int(target.get("keypoints", torch.zeros((0, 5, 2))).shape[1]) if isinstance(target.get("keypoints"), torch.Tensor) else 5
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "head_boxes": torch.zeros((0, 4), dtype=torch.float32),
        "ellipses": torch.zeros((0, 5), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "keypoints": torch.zeros((0, num_keypoints, 2), dtype=torch.float32),
        "keypoint_mask": torch.zeros((0, num_keypoints), dtype=torch.bool),
        "keypoint_visibility": torch.zeros((0, num_keypoints), dtype=torch.long),
        "face_visibility": torch.zeros((0,), dtype=torch.long),
        "image_id": image_id,
        "orig_size": (height, width),
        "size": (height, width),
        "metadata": {"augmentation_debug": {"ignored_instances": 0}},
    }


def _target_instance_count(target: Mapping[str, Any]) -> int:
    boxes = target.get("boxes")
    return int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0


def _scale_offset_target(
    target: Mapping[str, Any],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    height: int,
    width: int,
) -> dict[str, Any]:
    updated = _clone_target(target)
    boxes = updated.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale_x + offset_x
        boxes[:, 1::2] = boxes[:, 1::2] * scale_y + offset_y
        updated["boxes"] = boxes
    head_boxes = updated.get("head_boxes")
    if isinstance(head_boxes, torch.Tensor) and head_boxes.numel() > 0:
        head_boxes = head_boxes.clone()
        head_boxes[:, 0::2] = head_boxes[:, 0::2] * scale_x + offset_x
        head_boxes[:, 1::2] = head_boxes[:, 1::2] * scale_y + offset_y
        updated["head_boxes"] = head_boxes
    ellipses = updated.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        ellipses = ellipses.clone()
        ellipses[:, 0] = ellipses[:, 0] * scale_x + offset_x
        ellipses[:, 1] = ellipses[:, 1] * scale_y + offset_y
        ellipses[:, 2] *= abs(scale_x)
        ellipses[:, 3] *= abs(scale_y)
        updated["ellipses"] = ellipses
        boxes_from_ellipse = []
        for ellipse in ellipses.detach().cpu().tolist():
            box, _clipped = _ellipse_to_bbox(ellipse, width, height)
            boxes_from_ellipse.append(box if box is not None else [0.0, 0.0, 0.0, 0.0])
        updated["boxes"] = torch.as_tensor(boxes_from_ellipse, dtype=ellipses.dtype, device=ellipses.device)
    keypoints = updated.get("keypoints")
    if isinstance(keypoints, torch.Tensor) and keypoints.numel() > 0:
        keypoints = keypoints.clone()
        keypoints[..., 0] = keypoints[..., 0] * scale_x + offset_x
        keypoints[..., 1] = keypoints[..., 1] * scale_y + offset_y
        updated["keypoints"] = keypoints
    updated["orig_size"] = target.get("orig_size", (height, width))
    updated["size"] = (height, width)
    return updated


def _concat_targets(targets: list[Mapping[str, Any]], *, image_id: str, height: int, width: int) -> dict[str, Any]:
    if not targets:
        return _empty_like_target({}, image_id=image_id, height=height, width=width)
    keys = (
        "boxes",
        "head_boxes",
        "ellipses",
        "labels",
        "keypoints",
        "keypoint_mask",
    )
    output: dict[str, Any] = {}
    for key in keys:
        tensors = [target[key] for target in targets if isinstance(target.get(key), torch.Tensor)]
        output[key] = torch.cat(tensors, dim=0) if tensors else torch.empty((0,))
    if output["labels"].numel() == 0:
        output["labels"] = torch.ones((output["boxes"].shape[0],), dtype=torch.long)
    keypoint_visibility = []
    face_visibility = []
    for target in targets:
        count = _target_instance_count(target)
        masks = target.get("keypoint_mask")
        states = target.get("keypoint_visibility")
        if isinstance(states, torch.Tensor):
            keypoint_visibility.append(states.to(dtype=torch.long))
        elif isinstance(masks, torch.Tensor):
            keypoint_visibility.append(masks.to(dtype=torch.long) * 2)
        if isinstance(target.get("face_visibility"), torch.Tensor):
            face_visibility.append(target["face_visibility"].to(dtype=torch.long))
        else:
            face_visibility.append(torch.full((count,), 2, dtype=torch.long))
    output["keypoint_visibility"] = (
        torch.cat(keypoint_visibility, dim=0)
        if keypoint_visibility
        else torch.zeros((0, 5), dtype=torch.long)
    )
    output["face_visibility"] = (
        torch.cat(face_visibility, dim=0)
        if face_visibility
        else torch.zeros((0,), dtype=torch.long)
    )
    output["image_id"] = image_id
    output["orig_size"] = (height, width)
    output["size"] = (height, width)
    output["metadata"] = {
        "mosaic_sources": [str(target.get("image_id", "")) for target in targets],
        "augmentation_debug": {"ignored_instances": 0},
    }
    return output


def apply_mosaic(
    samples: list[tuple[torch.Tensor, Mapping[str, Any]]],
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create a deterministic 2x2 mosaic in the current resized training canvas."""

    _ = rng
    if len(samples) != 4:
        raise ValueError("mosaic requires exactly 4 samples")
    first_image = samples[0][0]
    height, width = int(first_image.shape[-2]), int(first_image.shape[-1])
    half_w = width // 2
    half_h = height // 2
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas_padding_mask = torch.ones((height, width), dtype=torch.bool)
    transformed_targets: list[Mapping[str, Any]] = []
    placements = ((0, 0), (half_w, 0), (0, half_h), (half_w, half_h))
    for (image_tensor, target), (offset_x, offset_y) in zip(samples, placements):
        image = _tensor_to_image(image_tensor).resize((half_w, half_h), Image.BILINEAR)
        canvas.paste(image, (offset_x, offset_y))
        source_mask = _padding_mask(target, height=height, width=width)
        resized_mask = _resize_padding_mask(
            source_mask, height=half_h, width=half_w
        )
        canvas_padding_mask[
            offset_y : offset_y + half_h, offset_x : offset_x + half_w
        ] = resized_mask
        transformed_targets.append(
            _scale_offset_target(
                target,
                scale_x=half_w / float(width),
                scale_y=half_h / float(height),
                offset_x=float(offset_x),
                offset_y=float(offset_y),
                height=height,
                width=width,
            )
        )
    image_id = "mosaic__" + "__".join(str(target.get("image_id", index)) for index, (_image, target) in enumerate(samples))
    output_target = _concat_targets(
        transformed_targets, image_id=image_id, height=height, width=width
    )
    _set_padding_mask(output_target, canvas_padding_mask)
    return _image_to_tensor(canvas), output_target


def apply_mixup(
    first: tuple[torch.Tensor, Mapping[str, Any]],
    second: tuple[torch.Tensor, Mapping[str, Any]],
    *,
    alpha: float = 0.2,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Blend two same-size training images and concatenate detection targets."""

    rng = rng or random.Random()
    image_a, target_a = first
    image_b, target_b = second
    if tuple(image_a.shape) != tuple(image_b.shape):
        raise ValueError(f"mixup images must have the same shape, got {tuple(image_a.shape)} and {tuple(image_b.shape)}")
    alpha = max(1e-3, float(alpha))
    lam = rng.betavariate(alpha, alpha)
    mixed_image = (image_a * lam + image_b * (1.0 - lam)).clamp(0.0, 1.0)
    height, width = int(image_a.shape[-2]), int(image_a.shape[-1])
    target = _concat_targets(
        [target_a, target_b],
        image_id=f"mixup__{target_a.get('image_id', 'a')}__{target_b.get('image_id', 'b')}",
        height=height,
        width=width,
    )
    _set_padding_mask(
        target,
        _padding_mask(target_a, height=height, width=width)
        & _padding_mask(target_b, height=height, width=width),
    )
    metadata = dict(target.get("metadata", {}) if isinstance(target.get("metadata"), Mapping) else {})
    metadata["mixup_lambda"] = float(lam)
    target["metadata"] = metadata
    return mixed_image, target


def _homography_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = vh[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def _random_geometry_matrix(
    width: int,
    height: int,
    cfg: Mapping[str, Any],
    rng: random.Random,
) -> np.ndarray:
    degrees = float(cfg.get("degrees", 0.0) or 0.0)
    translate = float(cfg.get("translate", 0.0) or 0.0)
    scale_cfg = float(cfg.get("scale", 0.0) or 0.0)
    shear_cfg = float(cfg.get("shear", 0.0) or 0.0)
    perspective = float(cfg.get("perspective", 0.0) or 0.0)

    angle = math.radians(rng.uniform(-degrees, degrees)) if degrees > 0 else 0.0
    scale = rng.uniform(max(0.05, 1.0 - scale_cfg), 1.0 + scale_cfg) if scale_cfg > 0 else 1.0
    shear_x = math.radians(rng.uniform(-shear_cfg, shear_cfg)) if shear_cfg > 0 else 0.0
    tx = rng.uniform(-translate, translate) * width if translate > 0 else 0.0
    ty = rng.uniform(-translate, translate) * height if translate > 0 else 0.0
    cx = width * 0.5
    cy = height * 0.5

    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    rotate_scale = np.array(
        [
            [math.cos(angle) * scale, -math.sin(angle) * scale, 0.0],
            [math.sin(angle) * scale, math.cos(angle) * scale, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shear = np.array([[1.0, math.tan(shear_x), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]])
    matrix = back @ rotate_scale @ shear @ to_origin

    if perspective > 0:
        source = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)
        jitter_x = perspective * width
        jitter_y = perspective * height
        destination = source + np.array(
            [[rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)] for _ in range(4)],
            dtype=np.float64,
        )
        matrix = _homography_from_points(source, destination) @ matrix
    return matrix


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points.astype(np.float64), ones], axis=1) @ matrix.T
    denom = homogeneous[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-9, np.sign(denom + 1e-12) * 1e-9, denom)
    return homogeneous[:, :2] / denom


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, theta: float, *, steps: int = 96) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xs = rx * np.cos(angles)
    ys = ry * np.sin(angles)
    return np.stack([cx + xs * cos_t - ys * sin_t, cy + xs * sin_t + ys * cos_t], axis=1)


def _ellipse_from_points(points: np.ndarray) -> tuple[float, float, float, float, float]:
    center = points.mean(axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(1, points.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 1e-9, None)
    vectors = vectors[:, order]
    radii = np.sqrt(values * 2.0)
    theta = math.atan2(float(vectors[1, 0]), float(vectors[0, 0]))
    return float(center[0]), float(center[1]), float(radii[0]), float(radii[1]), float(theta)


def _clip_box_from_points(points: np.ndarray, width: int, height: int) -> tuple[list[float] | None, bool]:
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    clipped = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, True
    return clipped, clipped != [x1, y1, x2, y2]


def _ellipse_to_bbox(ellipse: list[float], width: int, height: int) -> tuple[list[float] | None, bool]:
    cx, cy, rx, ry, theta = [float(value) for value in ellipse]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    half_width = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
    half_height = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
    raw = [cx - half_width, cy - half_height, cx + half_width, cy + half_height]
    clipped = [max(0.0, raw[0]), max(0.0, raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, True
    return clipped, clipped != raw


def _configured_range(
    cfg: Mapping[str, Any],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = cfg.get(key)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    elif value is not None:
        low = high = float(value)
    else:
        low, high = default
    low = max(0.0, low)
    high = max(low, high)
    return low, high


def _ellipse_border_crop_candidate(
    ellipse: list[float],
    *,
    instance_index: int,
    side: str,
    mode: str,
    margin_fraction: float,
    width: int,
    height: int,
    min_crop_scale: float,
) -> dict[str, Any] | None:
    cx, cy, rx, ry, theta = [float(value) for value in ellipse]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    extent_x = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
    extent_y = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
    if extent_x <= 0.0 or extent_y <= 0.0:
        return None

    if side in {"left", "right"}:
        center = cx
        extent = extent_x
        primary_size = width
        cross_center = cy
        cross_size = height
    else:
        center = cy
        extent = extent_y
        primary_size = height
        cross_center = cx
        cross_size = width

    if mode == "center":
        boundary = center + margin_fraction * extent if side in {"left", "top"} else center - margin_fraction * extent
    else:
        boundary = (
            center - extent + margin_fraction * extent
            if side in {"left", "top"}
            else center + extent - margin_fraction * extent
        )

    needs_greater_than_center = (mode == "center") == (side in {"left", "top"})
    boundary_px = int(round(boundary))
    if needs_greater_than_center:
        boundary_px = max(boundary_px, math.floor(center) + 1)
    else:
        boundary_px = min(boundary_px, math.ceil(center) - 1)
    if boundary_px <= 0 or boundary_px >= primary_size:
        return None

    if side in {"left", "top"}:
        primary_start = boundary_px
        primary_end = primary_size
    else:
        primary_start = 0
        primary_end = boundary_px
    primary_crop_size = primary_end - primary_start
    crop_scale = primary_crop_size / float(primary_size)
    if crop_scale < min_crop_scale:
        return None

    # Use the same relative crop in the cross-axis so resizing back to the
    # training canvas is isotropic even for non-square input tensors.
    cross_crop_size = max(2, min(cross_size, int(round(cross_size * crop_scale))))
    cross_start = int(round(cross_center - cross_crop_size * 0.5))
    cross_start = max(0, min(cross_size - cross_crop_size, cross_start))
    cross_end = cross_start + cross_crop_size

    if side in {"left", "right"}:
        crop_xyxy = (primary_start, cross_start, primary_end, cross_end)
    else:
        crop_xyxy = (cross_start, primary_start, cross_end, primary_end)
    x0, y0, x1, y1 = crop_xyxy
    selected_center_x = cx - x0
    selected_center_y = cy - y0
    selected_center_outside = not (0.0 <= selected_center_x < x1 - x0 and 0.0 <= selected_center_y < y1 - y0)
    raw_box = [cx - extent_x - x0, cy - extent_y - y0, cx + extent_x - x0, cy + extent_y - y0]
    selected_contour_outside = raw_box[0] < 0.0 or raw_box[1] < 0.0 or raw_box[2] > x1 - x0 or raw_box[3] > y1 - y0
    if not selected_contour_outside or selected_center_outside != (mode == "center"):
        return None
    if _ellipse_to_bbox([selected_center_x, selected_center_y, rx, ry, theta], x1 - x0, y1 - y0)[0] is None:
        return None
    return {
        "instance_index": int(instance_index),
        "mode": mode,
        "side": side,
        "margin_fraction": float(margin_fraction),
        "crop_xyxy": crop_xyxy,
        "crop_scale": float(crop_scale),
    }


def apply_ellipse_border_crop(
    image_tensor: torch.Tensor,
    target: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Crop through a selected ellipse contour or center and retain its amodal target."""

    count = _target_instance_count(target)
    ellipses_value = target.get("ellipses")
    if count == 0 or not isinstance(ellipses_value, torch.Tensor) or ellipses_value.numel() == 0:
        return image_tensor, _clone_target(target)

    height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    selected = select_ellipse_border_crop(
        ellipses_value,
        cfg,
        width=width,
        height=height,
        rng=rng,
    )
    if selected is None:
        return image_tensor, _clone_target(target)

    x0, y0, x1, y1 = selected["crop_xyxy"]
    crop_width = x1 - x0
    crop_height = y1 - y0
    scale_x = width / float(crop_width)
    scale_y = height / float(crop_height)
    matrix = np.array(
        [
            [scale_x, 0.0, -x0 * scale_x],
            [0.0, scale_y, -y0 * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image = _tensor_to_image(image_tensor)
    cropped_image = image.crop((x0, y0, x1, y1)).resize((width, height), Image.BILINEAR)
    transformed_target = transform_target_homography(
        target,
        matrix,
        width=width,
        height=height,
        min_ellipse_radius=float(cfg.get("min_ellipse_radius", 2.0) or 2.0),
        allow_outside_center=True,
    )
    source_padding_mask = _padding_mask(target, height=height, width=width)
    _set_padding_mask(
        transformed_target,
        _resize_padding_mask(
            source_padding_mask[y0:y1, x0:x1], height=height, width=width
        ),
    )

    source_ellipse = [float(value) for value in ellipses_value[selected["instance_index"]].detach().cpu().tolist()]
    transformed_points = _apply_homography(_ellipse_points(*source_ellipse), matrix)
    selected_ellipse = _ellipse_from_points(transformed_points)
    tcx, tcy, trx, try_, ttheta = selected_ellipse
    selected_box, _ = _ellipse_to_bbox([tcx, tcy, trx, try_, ttheta], width, height)
    if selected_box is None:
        return image_tensor, _clone_target(target)
    center_outside = tcx < 0.0 or tcx >= float(width) or tcy < 0.0 or tcy >= float(height)
    if center_outside != (selected["mode"] == "center"):
        return image_tensor, _clone_target(target)

    metadata = dict(
        transformed_target.get("metadata", {})
        if isinstance(transformed_target.get("metadata"), Mapping)
        else {}
    )
    debug = dict(
        metadata.get("augmentation_debug", {})
        if isinstance(metadata.get("augmentation_debug"), Mapping)
        else {}
    )
    debug["ellipse_border_crop"] = {
        **selected,
        "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "selected_ellipse": [float(value) for value in selected_ellipse],
        "center_outside": bool(center_outside),
        "contour_outside": True,
    }
    metadata["augmentation_debug"] = debug
    transformed_target["metadata"] = metadata
    return _image_to_tensor(cropped_image), transformed_target


def select_ellipse_border_crop(
    ellipses: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    width: int,
    height: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Select a crop guaranteed to put one ellipse contour or center outside."""

    if ellipses.numel() == 0:
        return None
    min_crop_scale = min(0.99, max(0.05, float(cfg.get("ellipse_border_crop_min_scale", 0.50) or 0.50)))
    center_probability = min(
        1.0,
        max(0.0, float(cfg.get("ellipse_border_crop_center_outside_probability", 0.50) or 0.0)),
    )
    mode = "center" if rng.random() < center_probability else "contour"
    margin_key = (
        "ellipse_border_crop_center_margin"
        if mode == "center"
        else "ellipse_border_crop_contour_margin"
    )
    default_margin = (0.02, 0.20) if mode == "center" else (0.10, 0.50)
    margin_low, margin_high = _configured_range(cfg, margin_key, default_margin)

    candidates: list[dict[str, Any]] = []
    for instance_index, ellipse_tensor in enumerate(ellipses.detach().cpu()):
        ellipse = [float(value) for value in ellipse_tensor.tolist()]
        if not all(math.isfinite(value) for value in ellipse) or ellipse[2] <= 0.0 or ellipse[3] <= 0.0:
            continue
        if _ellipse_to_bbox(ellipse, width, height)[0] is None:
            continue
        for side in ("left", "right", "top", "bottom"):
            candidate = _ellipse_border_crop_candidate(
                ellipse,
                instance_index=instance_index,
                side=side,
                mode=mode,
                margin_fraction=rng.uniform(margin_low, margin_high),
                width=width,
                height=height,
                min_crop_scale=min_crop_scale,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None
    return rng.choice(candidates)


def transform_target_homography(
    target: Mapping[str, Any],
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
    min_ellipse_radius: float = 2.0,
    allow_outside_center: bool = False,
) -> dict[str, Any]:
    count = _target_instance_count(target)
    if count == 0:
        updated = _empty_like_target(target, image_id=str(target.get("image_id", "")), height=height, width=width)
        updated["metadata"] = {"augmentation_debug": {"ignored_instances": 0, "clipped_instances": 0, "keypoints_outside": 0}}
        return updated

    ellipses = torch.as_tensor(target["ellipses"], dtype=torch.float32)
    keypoints = torch.as_tensor(target["keypoints"], dtype=torch.float32)
    keypoint_mask = torch.as_tensor(target["keypoint_mask"], dtype=torch.bool)
    keypoint_visibility = torch.as_tensor(
        target.get("keypoint_visibility", keypoint_mask.to(dtype=torch.long) * 2),
        dtype=torch.long,
    )
    face_visibility = torch.as_tensor(
        target.get("face_visibility", torch.full((count,), 2, dtype=torch.long)),
        dtype=torch.long,
    )
    labels = torch.as_tensor(target.get("labels", torch.ones((count,), dtype=torch.long)), dtype=torch.long)
    raw_head_boxes = torch.as_tensor(
        target.get("head_boxes", torch.empty((0, 4))),
        dtype=torch.float32,
    )
    # FACE-only datasets legitimately carry no HEAD boxes. Mosaic preserves
    # that optional field as an empty (0, 4) tensor even when FACE ellipses are
    # present, so treat it exactly like an omitted field rather than reshaping
    # zero elements to the FACE instance count.
    if raw_head_boxes.numel() == 0:
        head_boxes = torch.full((count, 4), float("nan"), dtype=torch.float32)
    elif raw_head_boxes.numel() == count * 4:
        head_boxes = raw_head_boxes.reshape(count, 4)
    else:
        raise ValueError(
            "head_boxes must be empty or contain one xyxy box per FACE "
            f"instance: faces={count}, shape={tuple(raw_head_boxes.shape)}"
        )

    kept_boxes: list[list[float]] = []
    kept_head_boxes: list[list[float]] = []
    kept_ellipses: list[list[float]] = []
    kept_keypoints: list[torch.Tensor] = []
    kept_masks: list[torch.Tensor] = []
    kept_keypoint_visibility: list[torch.Tensor] = []
    kept_face_visibility: list[int] = []
    kept_labels: list[int] = []
    ignored = 0
    clipped_instances = 0
    keypoints_outside = 0

    for index in range(count):
        cx, cy, rx, ry, theta = [float(value) for value in ellipses[index].tolist()]
        if rx <= 0.0 or ry <= 0.0:
            ignored += 1
            continue
        transformed_points = _apply_homography(_ellipse_points(cx, cy, rx, ry, theta), matrix)
        tcx, tcy, trx, try_, ttheta = _ellipse_from_points(transformed_points)
        box, clipped = _ellipse_to_bbox([tcx, tcy, trx, try_, ttheta], width, height)
        if (
            box is None
            or (
                not allow_outside_center
                and (tcx < 0.0 or tcx >= float(width) or tcy < 0.0 or tcy >= float(height))
            )
            or trx < min_ellipse_radius
            or try_ < min_ellipse_radius
        ):
            ignored += 1
            continue
        clipped_instances += int(clipped)

        source_head_box = head_boxes[index].detach().cpu().numpy()
        if np.isfinite(source_head_box).all():
            hx1, hy1, hx2, hy2 = [float(value) for value in source_head_box]
            head_corners = np.asarray(
                ((hx1, hy1), (hx2, hy1), (hx2, hy2), (hx1, hy2)),
                dtype=np.float64,
            )
            transformed_head = _apply_homography(head_corners, matrix)
            transformed_head_box, _ = _clip_box_from_points(
                transformed_head, width, height
            )
            kept_head_boxes.append(
                transformed_head_box
                if transformed_head_box is not None
                else [float("nan")] * 4
            )
        else:
            kept_head_boxes.append([float("nan")] * 4)

        points = keypoints[index].clone()
        mask = keypoint_mask[index].clone()
        visibility = keypoint_visibility[index].clone()
        if points.numel() > 0:
            visible_indices = torch.nonzero(mask, as_tuple=False).flatten()
            if visible_indices.numel() > 0:
                transformed_kpts = _apply_homography(points[visible_indices].numpy(), matrix)
                for local_index, point_index in enumerate(visible_indices.tolist()):
                    x, y = float(transformed_kpts[local_index, 0]), float(transformed_kpts[local_index, 1])
                    if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
                        mask[point_index] = False
                        visibility[point_index] = 0
                        points[point_index] = torch.tensor([0.0, 0.0])
                        keypoints_outside += 1
                    else:
                        points[point_index] = torch.tensor([x, y])

        kept_boxes.append(box)
        kept_ellipses.append([tcx, tcy, trx, try_, ttheta])
        kept_keypoints.append(points)
        kept_masks.append(mask)
        kept_keypoint_visibility.append(visibility)
        kept_face_visibility.append(int(face_visibility[index].item()))
        kept_labels.append(int(labels[index].item()))

    num_keypoints = keypoints.shape[1] if keypoints.ndim == 3 else 5
    metadata = dict(target.get("metadata", {}) if isinstance(target.get("metadata"), Mapping) else {})
    metadata["augmentation_debug"] = {
        **dict(metadata.get("augmentation_debug", {}) if isinstance(metadata.get("augmentation_debug"), Mapping) else {}),
        "ignored_instances": ignored,
        "clipped_instances": clipped_instances,
        "keypoints_outside": keypoints_outside,
        "kept_instances": len(kept_boxes),
    }
    if not kept_boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "head_boxes": torch.zeros((0, 4), dtype=torch.float32),
            "ellipses": torch.zeros((0, 5), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "keypoints": torch.zeros((0, num_keypoints, 2), dtype=torch.float32),
            "keypoint_mask": torch.zeros((0, num_keypoints), dtype=torch.bool),
            "keypoint_visibility": torch.zeros((0, num_keypoints), dtype=torch.long),
            "face_visibility": torch.zeros((0,), dtype=torch.long),
            "image_id": str(target.get("image_id", "")),
            "orig_size": target.get("orig_size", (height, width)),
            "size": (height, width),
            "metadata": metadata,
        }
    return {
        "boxes": torch.as_tensor(kept_boxes, dtype=torch.float32),
        "head_boxes": torch.as_tensor(kept_head_boxes, dtype=torch.float32),
        "ellipses": torch.as_tensor(kept_ellipses, dtype=torch.float32),
        "labels": torch.as_tensor(kept_labels, dtype=torch.long),
        "keypoints": torch.stack(kept_keypoints).to(torch.float32),
        "keypoint_mask": torch.stack(kept_masks).to(torch.bool),
        "keypoint_visibility": torch.stack(kept_keypoint_visibility).to(torch.long),
        "face_visibility": torch.as_tensor(kept_face_visibility, dtype=torch.long),
        "image_id": str(target.get("image_id", "")),
        "orig_size": target.get("orig_size", (height, width)),
        "size": (height, width),
        "metadata": metadata,
    }


def _transform_image(image: Image.Image, matrix: np.ndarray) -> Image.Image:
    width, height = image.size
    inverse = np.linalg.inv(matrix)
    inverse = inverse / inverse[2, 2]
    coeffs = tuple(float(value) for value in inverse.reshape(-1)[:8])
    return image.transform((width, height), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(114, 114, 114))


def _transform_padding_mask(mask: torch.Tensor, matrix: np.ndarray) -> torch.Tensor:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    inverse = np.linalg.inv(matrix)
    inverse = inverse / inverse[2, 2]
    coeffs = tuple(float(value) for value in inverse.reshape(-1)[:8])
    image = Image.fromarray(mask.detach().cpu().numpy().astype(np.uint8) * 255, mode="L")
    transformed = image.transform(
        (width, height),
        Image.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.NEAREST,
        fillcolor=255,
    )
    return torch.from_numpy(np.asarray(transformed, dtype=np.uint8) > 0)


def apply_geometry(
    image_tensor: torch.Tensor,
    target: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image = _tensor_to_image(image_tensor)
    width, height = image.size
    matrix = _random_geometry_matrix(width, height, cfg, rng)
    transformed_image = _transform_image(image, matrix)
    transformed_target = transform_target_homography(
        target,
        matrix,
        width=width,
        height=height,
        min_ellipse_radius=float(cfg.get("min_ellipse_radius", 2.0) or 2.0),
        allow_outside_center=bool(
            cfg.get("preserve_outside_ellipse_center", False)
            or float(cfg.get("ellipse_border_crop", 0.0) or 0.0) > 0.0
        ),
    )
    _set_padding_mask(
        transformed_target,
        _transform_padding_mask(
            _padding_mask(target, height=height, width=width), matrix
        ),
    )
    return _image_to_tensor(transformed_image), transformed_target


def apply_horizontal_flip(
    image_tensor: torch.Tensor,
    target: Mapping[str, Any],
    *,
    flip_index: tuple[int, ...] = DEFAULT_KEYPOINT_FLIP_INDEX,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image = torch.flip(image_tensor, dims=(2,))
    width = int(image_tensor.shape[-1])
    updated = _clone_target(target)
    _set_padding_mask(
        updated,
        torch.flip(
            _padding_mask(
                target,
                height=int(image_tensor.shape[-2]),
                width=int(image_tensor.shape[-1]),
            ),
            dims=(1,),
        ),
    )
    boxes = updated.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes = boxes.clone()
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = float(width) - old_x2
        boxes[:, 2] = float(width) - old_x1
        updated["boxes"] = boxes
    head_boxes = updated.get("head_boxes")
    if isinstance(head_boxes, torch.Tensor) and head_boxes.numel() > 0:
        head_boxes = head_boxes.clone()
        old_x1 = head_boxes[:, 0].clone()
        old_x2 = head_boxes[:, 2].clone()
        head_boxes[:, 0] = float(width) - old_x2
        head_boxes[:, 2] = float(width) - old_x1
        updated["head_boxes"] = head_boxes
    ellipses = updated.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        ellipses = ellipses.clone()
        ellipses[:, 0] = float(width) - ellipses[:, 0]
        ellipses[:, 4] = math.pi - ellipses[:, 4]
        updated["ellipses"] = ellipses
        boxes_from_ellipse = []
        for ellipse in ellipses.detach().cpu().tolist():
            box, _clipped = _ellipse_to_bbox(ellipse, width, int(image_tensor.shape[-2]))
            boxes_from_ellipse.append(box if box is not None else [0.0, 0.0, 0.0, 0.0])
        updated["boxes"] = torch.as_tensor(boxes_from_ellipse, dtype=ellipses.dtype, device=ellipses.device)
    keypoints = updated.get("keypoints")
    masks = updated.get("keypoint_mask")
    if isinstance(keypoints, torch.Tensor) and keypoints.numel() > 0:
        keypoints = keypoints.clone()
        keypoints[..., 0] = float(width) - keypoints[..., 0]
        order = list(flip_index[: keypoints.shape[1]])
        if len(order) == keypoints.shape[1]:
            keypoints = keypoints[:, order]
            if isinstance(masks, torch.Tensor):
                updated["keypoint_mask"] = masks.clone()[:, order]
            visibility = updated.get("keypoint_visibility")
            if isinstance(visibility, torch.Tensor):
                updated["keypoint_visibility"] = visibility.clone()[:, order]
        updated["keypoints"] = keypoints
    metadata = dict(updated.get("metadata", {}) if isinstance(updated.get("metadata"), Mapping) else {})
    metadata["augmentation_debug"] = {
        **dict(metadata.get("augmentation_debug", {}) if isinstance(metadata.get("augmentation_debug"), Mapping) else {}),
        "flipped_lr": True,
    }
    updated["metadata"] = metadata
    return image, updated


def apply_photometric(
    image_tensor: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    rng: random.Random,
) -> torch.Tensor:
    image = _tensor_to_image(image_tensor)
    brightness = float(cfg.get("brightness", 0.0) or 0.0)
    contrast = float(cfg.get("contrast", 0.0) or 0.0)
    gamma = float(cfg.get("gamma", 0.0) or 0.0)
    hsv_h = float(cfg.get("hsv_h", 0.0) or 0.0)
    hsv_s = float(cfg.get("hsv_s", 0.0) or 0.0)
    hsv_v = float(cfg.get("hsv_v", 0.0) or 0.0)
    blur = float(cfg.get("blur", 0.0) or 0.0)
    noise = float(cfg.get("noise", 0.0) or 0.0)

    if brightness > 0:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.0 - brightness, 1.0 + brightness))
    if contrast > 0:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.0 - contrast, 1.0 + contrast))
    if blur > 0 and rng.random() < blur:
        blur_radius_min, blur_radius_max = _blur_radius_range(cfg)
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(blur_radius_min, blur_radius_max)))

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
        hsv = np.array(image.convert("HSV"), dtype=np.float32)
        if hsv_h > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hsv_h, hsv_h) * 255.0) % 255.0
        if hsv_s > 0:
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1.0 - hsv_s, 1.0 + hsv_s), 0.0, 255.0)
        if hsv_v > 0:
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(1.0 - hsv_v, 1.0 + hsv_v), 0.0, 255.0)
        array = np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.float32) / 255.0
    if gamma > 0:
        gamma_value = rng.uniform(max(0.1, 1.0 - gamma), 1.0 + gamma)
        array = np.power(np.clip(array, 0.0, 1.0), gamma_value)
    if noise > 0:
        array = np.clip(array + rng.normalvariate(0.0, noise) * np.random.default_rng(rng.randrange(2**32)).standard_normal(array.shape), 0.0, 1.0)
    return torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous()


class AugmentedFaceKeypointDataset(Dataset):
    """Training-time augmentation wrapper for face ellipse/keypoint targets."""

    def __init__(
        self,
        base_dataset: Dataset,
        augmentation: Mapping[str, Any] | None,
        *,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.augmentation = dict(augmentation or {})
        self.seed = int(seed)
        # DataLoader persistent workers keep their own Dataset object, so plain
        # Python attributes changed by the parent process are not visible after
        # the workers have started. Store schedule state in shared memory to
        # keep per-epoch RNG and close_mosaic synchronized without respawning
        # workers on every epoch.
        self._epoch_state = torch.tensor([1, -1], dtype=torch.int64).share_memory_()
        self.mosaic_enabled = bool(float(self.augmentation.get("mosaic", 0.0) or 0.0) > 0.0)
        self.mixup_enabled = bool(float(self.augmentation.get("mixup", 0.0) or 0.0) > 0.0)

    def __len__(self) -> int:
        return len(self.base_dataset)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state[0].item())

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch_state[0] = int(value)

    @property
    def total_epochs(self) -> int | None:
        value = int(self._epoch_state[1].item())
        return value if value >= 0 else None

    @total_epochs.setter
    def total_epochs(self, value: int | None) -> None:
        self._epoch_state[1] = -1 if value is None else int(value)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.epoch = int(epoch)
        self.total_epochs = int(total_epochs) if total_epochs is not None else None

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * 1_000_003 + int(index))

    def _mosaic_active(self) -> bool:
        if not self.mosaic_enabled:
            return False
        close_mosaic = int(self.augmentation.get("close_mosaic", 0) or 0)
        if self.total_epochs is not None and close_mosaic > 0 and self.epoch > self.total_epochs - close_mosaic:
            return False
        return True

    def _effective_augmentation(self) -> dict[str, Any]:
        """Return the scheduled augmentation without mutating the run config."""

        cfg = dict(self.augmentation)
        weak_last_epochs = int(cfg.get("weak_last_epochs", 0) or 0)
        weak = (
            self.total_epochs is not None
            and weak_last_epochs > 0
            and self.epoch > self.total_epochs - weak_last_epochs
        )
        if not weak:
            return cfg
        factor = float(cfg.get("weak_augmentation_scale", 0.5) or 0.5)
        cfg["mosaic"] = 0.0
        for key in (
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "brightness",
            "contrast",
            "gamma",
            "blur",
            "noise",
        ):
            if key in cfg:
                cfg[key] = float(cfg.get(key, 0.0) or 0.0) * factor
        return cfg

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image, target = self.base_dataset[index]
        if not _enabled(self.augmentation):
            return image, target
        rng = self._rng(index)
        cfg = self._effective_augmentation()

        if self._mosaic_active() and rng.random() < float(cfg.get("mosaic", 0.0) or 0.0) and len(self.base_dataset) >= 4:
            extra_indices = [rng.randrange(len(self.base_dataset)) for _ in range(3)]
            samples = [(image, target)] + [self.base_dataset[item] for item in extra_indices]
            image, target = apply_mosaic(samples, rng=rng)

        if self.mixup_enabled and rng.random() < float(cfg.get("mixup", 0.0) or 0.0) and len(self.base_dataset) >= 2:
            other = self.base_dataset[rng.randrange(len(self.base_dataset))]
            image, target = apply_mixup(
                (image, target),
                other,
                alpha=float(cfg.get("mixup_alpha", 0.2) or 0.2),
                rng=rng,
            )

        geometry_keys = ("degrees", "translate", "scale", "shear", "perspective")
        if any(float(cfg.get(key, 0.0) or 0.0) > 0.0 for key in geometry_keys):
            image, target = apply_geometry(image, target, cfg, rng=rng)

        ellipse_border_crop = float(cfg.get("ellipse_border_crop", 0.0) or 0.0)
        if ellipse_border_crop > 0.0 and rng.random() < ellipse_border_crop:
            image, target = apply_ellipse_border_crop(image, target, cfg, rng=rng)

        fliplr = float(cfg.get("fliplr", 0.0) or 0.0)
        if fliplr > 0.0 and rng.random() < fliplr:
            image, target = apply_horizontal_flip(image, target)

        photometric_keys = ("hsv_h", "hsv_s", "hsv_v", "brightness", "contrast", "gamma", "blur", "noise")
        if any(float(cfg.get(key, 0.0) or 0.0) > 0.0 for key in photometric_keys):
            image = apply_photometric(image, cfg, rng=rng)
        landmark_occlusion = cfg.get("landmark_guided_occlusion")
        if isinstance(landmark_occlusion, Mapping):
            image, target = apply_landmark_occlusion_tensor(
                image,
                target,
                landmark_occlusion,
                rng=rng,
            )
        pad_color = cfg.get(
            "pad_color", getattr(self.base_dataset, "pad_color", None)
        )
        image = _restore_padding_pixels(image, target, pad_color=pad_color)
        return image, target
