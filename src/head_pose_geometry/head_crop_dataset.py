from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .dataset import HeadCentricFaceDataset, ellipse_aabb


FACE_FIELDS = (
    "boxes",
    "ellipses",
    "ellipse_mask",
    "labels",
    "keypoints",
    "labeled",
    "visible",
    "rotation_matrix",
    "pose_supervision_mask",
    "pose_weight",
    "instance_ids",
    "sample_is_synthetic",
    "template_vertices",
    "weak_perspective_scale",
    "translation",
    "shape_code",
)


def square_head_crop_box(
    box_xyxy: Sequence[float] | torch.Tensor,
    *,
    context_scale: float = 1.20,
) -> tuple[float, float, float, float]:
    box = torch.as_tensor(box_xyxy, dtype=torch.float64).reshape(4)
    if not torch.isfinite(box).all():
        raise ValueError(f"HEAD box contains non-finite values: {box.tolist()}")
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"HEAD box has non-positive area: {box.tolist()}")
    scale = float(context_scale)
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError("HEAD crop context_scale must be finite and at least 1.0")
    side = max(2.0, scale * max(width, height))
    center_x = float((box[0] + box[2]) * 0.5)
    center_y = float((box[1] + box[3]) * 0.5)
    return (
        center_x - side * 0.5,
        center_y - side * 0.5,
        center_x + side * 0.5,
        center_y + side * 0.5,
    )


def crop_tensor_to_square(
    image: torch.Tensor,
    crop_xyxy: Sequence[float] | torch.Tensor,
    *,
    output_size: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Sample an amodal square crop, preserving out-of-frame padding."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, got {tuple(image.shape)}")
    size = int(output_size)
    if size < 2:
        raise ValueError("HEAD crop output_size must be at least 2")
    x0, y0, x1, y1 = (
        float(value) for value in torch.as_tensor(crop_xyxy).reshape(4)
    )
    side_x, side_y = x1 - x0, y1 - y0
    if side_x <= 0.0 or side_y <= 0.0:
        raise ValueError("HEAD crop must have positive width and height")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    xs = x0 + (torch.arange(size, device=image.device, dtype=torch.float32) + 0.5) * (
        side_x / size
    )
    ys = y0 + (torch.arange(size, device=image.device, dtype=torch.float32) + 0.5) * (
        side_y / size
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack(
        (
            2.0 * grid_x / max(1, width) - 1.0,
            2.0 * grid_y / max(1, height) - 1.0,
        ),
        dim=-1,
    )[None]
    source = image[None].to(dtype=torch.float32)
    if float(pad_value) != 0.0:
        source = source - float(pad_value)
    crop = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    if float(pad_value) != 0.0:
        crop = crop + float(pad_value)
    return crop.to(dtype=image.dtype).contiguous()


def map_crop_points_to_image(
    values: torch.Tensor,
    crop_xyxy: Sequence[float] | torch.Tensor,
    *,
    input_size: int,
) -> torch.Tensor:
    crop = torch.as_tensor(crop_xyxy, dtype=values.dtype, device=values.device)
    scale = (crop[2:4] - crop[:2]) / float(input_size)
    return values * scale + crop[:2]


class HeadCropCooperativeDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """One cooperative-DINO sample per GT or cached predicted HEAD crop."""

    def __init__(
        self,
        base: HeadCentricFaceDataset,
        *,
        output_size: int,
        context_scale: float = 1.20,
        pad_value: float = 0.0,
    ) -> None:
        self.base = base
        self.output_size = int(output_size)
        self.context_scale = float(context_scale)
        self.pad_value = float(pad_value)
        self.sample_indices: list[tuple[int, int]] = []
        for image_index, sample in enumerate(base.samples):
            predicted = sample.get("predicted_heads")
            count = (
                len(predicted)
                if predicted is not None
                else len(sample.get("annotations", ()))
            )
            self.sample_indices.extend(
                (image_index, head_index) for head_index in range(count)
            )
        if not self.sample_indices:
            raise ValueError("HEAD-crop dataset contains no GT or predicted HEAD crops")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    @staticmethod
    def _select_face_fields(
        target: Mapping[str, Any],
        face_index: int,
    ) -> dict[str, Any]:
        output = dict(target)
        face_count = int(torch.as_tensor(target["boxes"]).shape[0])
        for name in FACE_FIELDS:
            value = target.get(name)
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            if int(value.shape[0]) != face_count:
                continue
            output[name] = (
                value[face_index : face_index + 1].clone()
                if 0 <= face_index < face_count
                else value[:0].clone()
            )
        return output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_index, head_index = self.sample_indices[index]
        _resized_image, original_target = self.base[image_index]
        sample = self.base.samples[image_index]
        with Image.open(sample["path"]) as source_image:
            image_array = np.asarray(
                source_image.convert("RGB"),
                dtype=np.float32,
            ).copy()
        image = torch.from_numpy(image_array).permute(2, 0, 1).contiguous() / 255.0
        predicted = sample.get("predicted_heads")
        if predicted is not None:
            heads = torch.tensor(
                [row["box_xyxy"] for row in predicted],
                dtype=torch.float32,
            ).reshape(-1, 4)
        else:
            heads = torch.tensor(
                [
                    (
                        float(row["bbox"][0]),
                        float(row["bbox"][1]),
                        float(row["bbox"][0] + row["bbox"][2]),
                        float(row["bbox"][1] + row["bbox"][3]),
                    )
                    for row in sample.get("annotations", ())
                ],
                dtype=torch.float32,
            ).reshape(-1, 4)
        if head_index >= int(heads.shape[0]):
            raise IndexError(
                f"HEAD index {head_index} is unavailable for image sample {image_index}"
            )
        mapping = torch.as_tensor(
            original_target.get("geometry_target_indices", []),
            dtype=torch.long,
        ).reshape(-1)
        face_index = int(mapping[head_index]) if head_index < mapping.numel() else -1
        target = self._select_face_fields(original_target, face_index)
        if 0 <= face_index < len(sample.get("annotations", ())):
            raw_row = sample["annotations"][face_index]
            raw_ellipse = torch.tensor(raw_row["ellipse"], dtype=torch.float32)
            target["ellipses"] = raw_ellipse.reshape(1, 5)
            raw_box = torch.tensor(ellipse_aabb(raw_row["ellipse"])).reshape(1, 4)
            raw_box[:, 0::2].clamp_(0.0, float(image.shape[-1]))
            raw_box[:, 1::2].clamp_(0.0, float(image.shape[-2]))
            target["boxes"] = raw_box
        crop_box = square_head_crop_box(
            heads[head_index],
            context_scale=self.context_scale,
        )
        cropped = crop_tensor_to_square(
            image,
            crop_box,
            output_size=self.output_size,
            pad_value=self.pad_value,
        )
        x0, y0, x1, y1 = crop_box
        scale_x = self.output_size / (x1 - x0)
        scale_y = self.output_size / (y1 - y0)
        if target["boxes"].numel():
            boxes = target["boxes"]
            boxes[:, 0::2] = (boxes[:, 0::2] - x0) * scale_x
            boxes[:, 1::2] = (boxes[:, 1::2] - y0) * scale_y
            boxes[:, 0::2].clamp_(0.0, float(self.output_size))
            boxes[:, 1::2].clamp_(0.0, float(self.output_size))
            ellipse = target["ellipses"]
            ellipse[:, 0] = (ellipse[:, 0] - x0) * scale_x
            ellipse[:, 1] = (ellipse[:, 1] - y0) * scale_y
            ellipse[:, 2] *= scale_x
            ellipse[:, 3] *= scale_y
            keypoints = target["keypoints"]
            source_height, source_width = image.shape[-2:]
            keypoints[..., 0] = (
                keypoints[..., 0] * float(source_width) - x0
            ) * scale_x / float(self.output_size)
            keypoints[..., 1] = (
                keypoints[..., 1] * float(source_height) - y0
            ) * scale_y / float(self.output_size)

        transformed_head = heads[head_index : head_index + 1].clone()
        transformed_head[:, 0::2] = (transformed_head[:, 0::2] - x0) * scale_x
        transformed_head[:, 1::2] = (transformed_head[:, 1::2] - y0) * scale_y
        target["head_boxes"] = transformed_head
        target["head_detection_boxes"] = transformed_head.clone()
        target["head_detection_labels"] = torch.ones(1, dtype=torch.long)
        target["head_detection_to_face_indices"] = torch.tensor(
            [0 if target["boxes"].shape[0] else -1],
            dtype=torch.long,
        )
        target["geometry_target_indices"] = torch.tensor(
            [0 if target["boxes"].shape[0] else -1],
            dtype=torch.long,
        )
        confidence = torch.as_tensor(
            original_target.get("head_confidence", torch.ones(len(heads)))
        ).reshape(-1)
        target["head_confidence"] = confidence[head_index : head_index + 1].clone()
        sources = list(original_target.get("head_sources", ("unknown",) * len(heads)))
        target["head_sources"] = [sources[head_index]]
        target["head_crop"] = {
            "parent_image_id": int(torch.as_tensor(target["image_id"]).item()),
            "head_index": int(head_index),
            "source_head_box_xyxy": [float(value) for value in heads[head_index]],
            "crop_xyxy": [float(value) for value in crop_box],
            "source_image_size": [int(image.shape[-2]), int(image.shape[-1])],
            "output_size": self.output_size,
            "context_scale": self.context_scale,
        }
        return cropped, target


__all__ = [
    "HeadCropCooperativeDataset",
    "crop_tensor_to_square",
    "map_crop_points_to_image",
    "square_head_crop_box",
]
