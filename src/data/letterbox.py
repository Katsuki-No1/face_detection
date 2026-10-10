from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_PAD_COLOR = (124, 116, 104)


def normalize_canvas_size(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = int(value)
    else:
        if len(value) != 2:
            raise ValueError(f"canvas size must be an int or [height, width], got {value}")
        height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"canvas dimensions must be positive, got {(height, width)}")
    return height, width


def normalize_pad_color(value: int | Sequence[int] | None) -> tuple[int, int, int]:
    if value is None:
        return DEFAULT_PAD_COLOR
    if isinstance(value, int):
        channels = (int(value),) * 3
    else:
        if len(value) != 3:
            raise ValueError(f"pad color must be an int or three RGB values, got {value}")
        channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"pad color values must be in [0,255], got {channels}")
    return channels


@dataclass(frozen=True)
class LetterboxGeometry:
    original_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @classmethod
    def for_image(
        cls,
        original_hw: tuple[int, int],
        canvas_hw: int | Sequence[int],
        *,
        divisor: int | None = None,
    ) -> "LetterboxGeometry":
        original_h, original_w = int(original_hw[0]), int(original_hw[1])
        canvas_h, canvas_w = normalize_canvas_size(canvas_hw)
        if original_h <= 0 or original_w <= 0:
            raise ValueError(f"original image dimensions must be positive, got {original_hw}")
        if divisor is not None and (
            canvas_h % int(divisor) != 0 or canvas_w % int(divisor) != 0
        ):
            raise ValueError(
                f"letterbox canvas {(canvas_h, canvas_w)} must be divisible by {int(divisor)}"
            )
        scale = min(canvas_w / float(original_w), canvas_h / float(original_h))
        resized_w = min(canvas_w, max(1, int(round(original_w * scale))))
        resized_h = min(canvas_h, max(1, int(round(original_h * scale))))
        pad_w = canvas_w - resized_w
        pad_h = canvas_h - resized_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        return cls(
            original_hw=(original_h, original_w),
            canvas_hw=(canvas_h, canvas_w),
            resized_hw=(resized_h, resized_w),
            scale=float(scale),
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_w - pad_left,
            pad_bottom=pad_h - pad_top,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LetterboxGeometry":
        return cls(
            original_hw=tuple(int(item) for item in value["original_hw"]),
            canvas_hw=tuple(int(item) for item in value["canvas_hw"]),
            resized_hw=tuple(int(item) for item in value["resized_hw"]),
            scale=float(value["scale"]),
            pad_left=int(value["pad_left"]),
            pad_top=int(value["pad_top"]),
            pad_right=int(value["pad_right"]),
            pad_bottom=int(value["pad_bottom"]),
        )

    @property
    def valid_xyxy(self) -> tuple[float, float, float, float]:
        resized_h, resized_w = self.resized_hw
        return (
            float(self.pad_left),
            float(self.pad_top),
            float(self.pad_left + resized_w),
            float(self.pad_top + resized_h),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def letterbox_image(
    image: Image.Image,
    canvas_hw: int | Sequence[int],
    *,
    pad_color: int | Sequence[int] | None = None,
    divisor: int | None = 32,
) -> tuple[Image.Image, LetterboxGeometry, torch.Tensor]:
    image = image.convert("RGB")
    geometry = LetterboxGeometry.for_image(
        (image.height, image.width), canvas_hw, divisor=divisor
    )
    resized_h, resized_w = geometry.resized_hw
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas_h, canvas_w = geometry.canvas_hw
    canvas = Image.new("RGB", (canvas_w, canvas_h), normalize_pad_color(pad_color))
    canvas.paste(resized, (geometry.pad_left, geometry.pad_top))
    padding_mask = torch.ones((canvas_h, canvas_w), dtype=torch.bool)
    x1, y1, x2, y2 = (int(value) for value in geometry.valid_xyxy)
    padding_mask[y1:y2, x1:x2] = False
    return canvas, geometry, padding_mask


def _clone_target(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def transform_target(
    target: Mapping[str, Any],
    geometry: LetterboxGeometry,
    *,
    padding_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    transformed = _clone_target(target)
    scale = float(geometry.scale)
    offset_x = float(geometry.pad_left)
    offset_y = float(geometry.pad_top)

    boxes = transformed.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + offset_x
        boxes[:, 1::2] = boxes[:, 1::2] * scale + offset_y
        transformed["boxes"] = boxes

    keypoints = transformed.get("keypoints")
    if isinstance(keypoints, torch.Tensor) and keypoints.numel() > 0:
        keypoints = keypoints.clone()
        keypoints[..., 0] = keypoints[..., 0] * scale + offset_x
        keypoints[..., 1] = keypoints[..., 1] * scale + offset_y
        transformed["keypoints"] = keypoints

    ellipses = transformed.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        ellipses = ellipses.clone()
        ellipses[:, 0] = ellipses[:, 0] * scale + offset_x
        ellipses[:, 1] = ellipses[:, 1] * scale + offset_y
        ellipses[:, 2:4] *= scale
        transformed["ellipses"] = ellipses

    canvas_h, canvas_w = geometry.canvas_hw
    if padding_mask is None:
        padding_mask = torch.ones((canvas_h, canvas_w), dtype=torch.bool)
        x1, y1, x2, y2 = (int(value) for value in geometry.valid_xyxy)
        padding_mask[y1:y2, x1:x2] = False
    transformed["padding_mask"] = padding_mask.clone().to(dtype=torch.bool)
    transformed["valid_image_region_xyxy"] = torch.tensor(
        geometry.valid_xyxy, dtype=torch.float32
    )
    transformed["size"] = geometry.canvas_hw
    transformed["orig_size"] = tuple(
        int(value) for value in target.get("orig_size", geometry.original_hw)
    )
    metadata = dict(
        transformed.get("metadata", {})
        if isinstance(transformed.get("metadata"), Mapping)
        else {}
    )
    metadata["letterbox"] = geometry.to_dict()
    transformed["metadata"] = metadata
    return transformed


def _transform_xyxy_to_original(
    boxes: torch.Tensor, geometry: LetterboxGeometry
) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2] = (boxes[..., 0::2] - float(geometry.pad_left)) / float(
        geometry.scale
    )
    boxes[..., 1::2] = (boxes[..., 1::2] - float(geometry.pad_top)) / float(
        geometry.scale
    )
    original_h, original_w = geometry.original_hw
    boxes[..., 0::2].clamp_(0.0, float(original_w))
    boxes[..., 1::2].clamp_(0.0, float(original_h))
    return boxes


def restore_detections(
    detections: Mapping[str, Any], geometry: LetterboxGeometry
) -> dict[str, Any]:
    restored = _clone_target(detections)
    boxes = restored.get("boxes")
    count = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) and boxes.ndim > 0 else 0
    valid: torch.Tensor | None = None
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes = _transform_xyxy_to_original(boxes, geometry)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        restored["boxes"] = boxes

    ellipses = restored.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        ellipses = ellipses.clone()
        ellipses[:, 0] = (ellipses[:, 0] - float(geometry.pad_left)) / float(
            geometry.scale
        )
        ellipses[:, 1] = (ellipses[:, 1] - float(geometry.pad_top)) / float(
            geometry.scale
        )
        ellipses[:, 2:4] /= float(geometry.scale)
        restored["ellipses"] = ellipses
        from src.models.dinov3_deimv2_ellipse import ellipse_to_aabb

        ellipse_boxes = ellipse_to_aabb(ellipses)
        original_h, original_w = geometry.original_hw
        ellipse_boxes[:, 0::2].clamp_(0.0, float(original_w))
        ellipse_boxes[:, 1::2].clamp_(0.0, float(original_h))
        restored["ellipse_boxes"] = ellipse_boxes
    elif isinstance(restored.get("ellipse_boxes"), torch.Tensor):
        restored["ellipse_boxes"] = _transform_xyxy_to_original(
            restored["ellipse_boxes"], geometry
        )

    keypoints = restored.get("keypoints")
    if isinstance(keypoints, torch.Tensor) and keypoints.numel() > 0:
        keypoints = keypoints.clone()
        keypoints[..., 0] = (
            keypoints[..., 0] - float(geometry.pad_left)
        ) / float(geometry.scale)
        keypoints[..., 1] = (
            keypoints[..., 1] - float(geometry.pad_top)
        ) / float(geometry.scale)
        restored["keypoints"] = keypoints

    if valid is not None and count > 0 and not bool(valid.all()):
        for key, value in tuple(restored.items()):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == count:
                restored[key] = value[valid]
    return restored


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


__all__ = [
    "DEFAULT_PAD_COLOR",
    "LetterboxGeometry",
    "image_to_tensor",
    "letterbox_image",
    "normalize_canvas_size",
    "normalize_pad_color",
    "restore_detections",
    "transform_target",
]
