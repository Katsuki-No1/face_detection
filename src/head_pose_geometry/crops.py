from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .heads import HeadDetection


@dataclass(frozen=True)
class HeadCropAugmentation:
    center_translation: float = 0.08
    log_scale_error: float = 0.12
    asymmetric_context: float = 0.12
    log_aspect_error: float = 0.08
    boundary_truncation_probability: float = 0.20
    boundary_truncation_fraction: tuple[float, float] = (0.05, 0.35)


@dataclass
class HeadCropBatch:
    rgb: torch.Tensor
    valid_mask: torch.Tensor
    border_distances: torch.Tensor
    original_image_size: torch.Tensor
    crop_to_image: torch.Tensor
    image_to_crop: torch.Tensor
    head_boxes_xyxy: torch.Tensor
    head_confidence: torch.Tensor
    valid_image_regions_xyxy: torch.Tensor
    instance_ids: list[str | int | None]
    sources: list[str]

    def to(self, device: torch.device | str) -> HeadCropBatch:
        return HeadCropBatch(
            rgb=self.rgb.to(device),
            valid_mask=self.valid_mask.to(device),
            border_distances=self.border_distances.to(device),
            original_image_size=self.original_image_size.to(device),
            crop_to_image=self.crop_to_image.to(device),
            image_to_crop=self.image_to_crop.to(device),
            head_boxes_xyxy=self.head_boxes_xyxy.to(device),
            head_confidence=self.head_confidence.to(device),
            valid_image_regions_xyxy=self.valid_image_regions_xyxy.to(device),
            instance_ids=list(self.instance_ids),
            sources=list(self.sources),
        )


def transform_points_homogeneous(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """Apply a batched 3x3 homogeneous transform to 2D points."""

    if points.shape[-1] != 2 or transform.shape[-2:] != (3, 3):
        raise ValueError("points must end in 2 and transform in (3,3)")
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    # Matrices are stored in the conventional column-vector form. Insert
    # singleton point axes between the transform batch and matrix axes so a
    # [B,3,3] transform works for [B,N,2] as well as [B,H,W,2] points.
    while transform.ndim < homogeneous.ndim + 1:
        transform = transform.unsqueeze(-3)
    transformed = (transform @ homogeneous.unsqueeze(-1)).squeeze(-1)
    return transformed[..., :2] / transformed[..., 2:].clamp_min(1.0e-12)


class HeadCropBuilder:
    """Context crop construction with explicit padding and affine metadata."""

    def __init__(
        self,
        *,
        output_size: int | tuple[int, int] = 224,
        context_scale: float = 1.35,
        pad_value: float = 0.0,
        augmentation: HeadCropAugmentation | None = None,
    ) -> None:
        if isinstance(output_size, int):
            self.output_height = self.output_width = int(output_size)
        else:
            self.output_height, self.output_width = (
                int(output_size[0]),
                int(output_size[1]),
            )
        if self.output_height < 2 or self.output_width < 2:
            raise ValueError("Head crop dimensions must be at least 2")
        if context_scale < 1.0:
            raise ValueError("context_scale must be >= 1")
        self.context_scale = float(context_scale)
        self.pad_value = float(pad_value)
        self.augmentation = augmentation or HeadCropAugmentation()

    @staticmethod
    def _uniform(
        generator: torch.Generator | None,
        low: float,
        high: float,
    ) -> float:
        value = torch.rand((), generator=generator).item()
        return low + (high - low) * float(value)

    def _crop_rectangle(
        self,
        box: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
        x1, y1, x2, y2 = (float(value) for value in box)
        width, height = max(1.0e-4, x2 - x1), max(1.0e-4, y2 - y1)
        center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        left = right = top = bottom = 0.5 * (self.context_scale - 1.0)
        if training:
            cfg = self.augmentation
            center_x += width * self._uniform(
                generator,
                -cfg.center_translation,
                cfg.center_translation,
            )
            center_y += height * self._uniform(
                generator,
                -cfg.center_translation,
                cfg.center_translation,
            )
            scale = math.exp(
                self._uniform(
                    generator,
                    -cfg.log_scale_error,
                    cfg.log_scale_error,
                )
            )
            aspect = math.exp(
                self._uniform(
                    generator,
                    -cfg.log_aspect_error,
                    cfg.log_aspect_error,
                )
            )
            width *= scale * math.sqrt(aspect)
            height *= scale / math.sqrt(aspect)
            asym = cfg.asymmetric_context
            left += self._uniform(generator, -asym, asym)
            right += self._uniform(generator, -asym, asym)
            top += self._uniform(generator, -asym, asym)
            bottom += self._uniform(generator, -asym, asym)
            left, right, top, bottom = (
                max(0.0, left),
                max(0.0, right),
                max(0.0, top),
                max(0.0, bottom),
            )
        crop_width = width * (1.0 + left + right)
        crop_height = height * (1.0 + top + bottom)
        crop_center_x = center_x + 0.5 * width * (right - left)
        crop_center_y = center_y + 0.5 * height * (bottom - top)
        rectangle = box.new_tensor(
            (
                crop_center_x - 0.5 * crop_width,
                crop_center_y - 0.5 * crop_height,
                crop_center_x + 0.5 * crop_width,
                crop_center_y + 0.5 * crop_height,
            )
        )
        return rectangle, (left, top, right, bottom)

    def _augmented_valid_region(
        self,
        region: torch.Tensor,
        box: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if not training:
            return region
        cfg = self.augmentation
        if self._uniform(generator, 0.0, 1.0) >= cfg.boundary_truncation_probability:
            return region
        output = region.clone()
        fraction = self._uniform(
            generator,
            float(cfg.boundary_truncation_fraction[0]),
            float(cfg.boundary_truncation_fraction[1]),
        )
        side = int(self._uniform(generator, 0.0, 4.0))
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if side == 0:
            output[0] = min(float(output[2]) - 1.0, float(box[0]) + fraction * width)
        elif side == 1:
            output[2] = max(float(output[0]) + 1.0, float(box[2]) - fraction * width)
        elif side == 2:
            output[1] = min(float(output[3]) - 1.0, float(box[1]) + fraction * height)
        else:
            output[3] = max(float(output[1]) + 1.0, float(box[3]) - fraction * height)
        return output

    def build(
        self,
        image: torch.Tensor,
        detections: list[HeadDetection],
        *,
        training: bool = False,
        generator: torch.Generator | None = None,
    ) -> HeadCropBatch:
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")
        channels, image_height, image_width = image.shape
        if channels != 3:
            raise ValueError(f"Expected RGB image, got {channels} channels")
        if not detections:
            empty_rgb = image.new_zeros((0, 3, self.output_height, self.output_width))
            return HeadCropBatch(
                rgb=empty_rgb,
                valid_mask=image.new_zeros(
                    (0, 1, self.output_height, self.output_width)
                ),
                border_distances=image.new_zeros((0, 4)),
                original_image_size=image.new_zeros((0, 2)),
                crop_to_image=image.new_zeros((0, 3, 3)),
                image_to_crop=image.new_zeros((0, 3, 3)),
                head_boxes_xyxy=image.new_zeros((0, 4)),
                head_confidence=image.new_zeros((0,)),
                valid_image_regions_xyxy=image.new_zeros((0, 4)),
                instance_ids=[],
                sources=[],
            )

        boxes = []
        confidences = []
        valid_regions = []
        rectangles = []
        ids: list[str | int | None] = []
        sources = []
        for raw in detections:
            item = raw.validated((image_height, image_width))
            box = image.new_tensor(item.box_xyxy)
            region = image.new_tensor(item.valid_image_region_xyxy)
            region = self._augmented_valid_region(
                region,
                box,
                training=training,
                generator=generator,
            )
            rectangle, _ = self._crop_rectangle(
                box,
                training=training,
                generator=generator,
            )
            boxes.append(box)
            confidences.append(float(item.confidence))
            valid_regions.append(region)
            rectangles.append(rectangle)
            ids.append(item.instance_id)
            sources.append(item.source)

        box_tensor = torch.stack(boxes)
        region_tensor = torch.stack(valid_regions)
        rectangle_tensor = torch.stack(rectangles)
        crop_width = (rectangle_tensor[:, 2] - rectangle_tensor[:, 0]).clamp_min(1.0e-6)
        crop_height = (rectangle_tensor[:, 3] - rectangle_tensor[:, 1]).clamp_min(
            1.0e-6
        )
        scale_x = crop_width / float(self.output_width - 1)
        scale_y = crop_height / float(self.output_height - 1)
        zeros = torch.zeros_like(scale_x)
        ones = torch.ones_like(scale_x)
        crop_to_image = torch.stack(
            (
                torch.stack((scale_x, zeros, rectangle_tensor[:, 0]), dim=-1),
                torch.stack((zeros, scale_y, rectangle_tensor[:, 1]), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
        image_to_crop = torch.linalg.inv(crop_to_image)

        crop_y, crop_x = torch.meshgrid(
            torch.arange(
                self.output_height,
                device=image.device,
                dtype=image.dtype,
            ),
            torch.arange(
                self.output_width,
                device=image.device,
                dtype=image.dtype,
            ),
            indexing="ij",
        )
        crop_points = torch.stack((crop_x, crop_y), dim=-1)
        image_points = transform_points_homogeneous(
            crop_points.unsqueeze(0).expand(len(detections), -1, -1, -1),
            crop_to_image,
        )
        if image_width > 1:
            normalized_x = 2.0 * image_points[..., 0] / (image_width - 1) - 1.0
        else:
            normalized_x = torch.zeros_like(image_points[..., 0])
        if image_height > 1:
            normalized_y = 2.0 * image_points[..., 1] / (image_height - 1) - 1.0
        else:
            normalized_y = torch.zeros_like(image_points[..., 1])
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        source = image.unsqueeze(0).expand(len(detections), -1, -1, -1)
        sampled = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid = (
            (image_points[..., 0] >= region_tensor[:, None, None, 0])
            & (image_points[..., 0] <= region_tensor[:, None, None, 2] - 1.0)
            & (image_points[..., 1] >= region_tensor[:, None, None, 1])
            & (image_points[..., 1] <= region_tensor[:, None, None, 3] - 1.0)
        ).unsqueeze(1)
        sampled = sampled * valid.to(sampled.dtype)
        if self.pad_value:
            sampled = sampled + (~valid).to(sampled.dtype) * self.pad_value

        normalizer = box_tensor.new_tensor(
            (
                float(image_width),
                float(image_height),
                float(image_width),
                float(image_height),
            )
        )
        border = (
            torch.stack(
                (
                    box_tensor[:, 0] - region_tensor[:, 0],
                    box_tensor[:, 1] - region_tensor[:, 1],
                    region_tensor[:, 2] - box_tensor[:, 2],
                    region_tensor[:, 3] - box_tensor[:, 3],
                ),
                dim=-1,
            )
            / normalizer
        )
        return HeadCropBatch(
            rgb=sampled,
            valid_mask=valid.to(sampled.dtype),
            border_distances=border,
            original_image_size=box_tensor.new_tensor(
                (image_height, image_width)
            ).expand(len(detections), -1),
            crop_to_image=crop_to_image,
            image_to_crop=image_to_crop,
            head_boxes_xyxy=box_tensor,
            head_confidence=box_tensor.new_tensor(confidences),
            valid_image_regions_xyxy=region_tensor,
            instance_ids=ids,
            sources=sources,
        )


__all__ = [
    "HeadCropAugmentation",
    "HeadCropBatch",
    "HeadCropBuilder",
    "transform_points_homogeneous",
]
