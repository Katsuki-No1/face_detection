from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _ceil_to_multiple(value: int, divisor: int) -> int:
    if value <= 0:
        raise ValueError(f"Image dimensions must be positive, got {value}")
    return ((int(value) + divisor - 1) // divisor) * divisor


def _prepare_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a CHW RGB tensor, got shape={tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32).div_(255.0)
    elif not image.dtype.is_floating_point:
        raise TypeError(f"Images must be uint8 or floating point, got {image.dtype}")
    else:
        image = image.to(dtype=torch.float32)
    # Dataset inputs are normally CPU tensors. Avoid scalar GPU synchronization
    # in the inference hot path while retaining strict validation at ingestion.
    if image.device.type == "cpu":
        if not torch.isfinite(image).all():
            raise ValueError("Image contains NaN or infinity")
        if image.numel() and (
            float(image.min()) < -1.0e-6 or float(image.max()) > 1.0 + 1.0e-6
        ):
            raise ValueError("Floating-point RGB images must be in the [0, 1] range")
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


@dataclass(frozen=True)
class HeadImageBatch:
    """Normalized, right/bottom-padded images and their valid-pixel mask.

    ``padding_mask`` follows DETR/MMDetection convention: ``True`` denotes
    padding and ``False`` denotes a valid image pixel.
    """

    tensors: torch.Tensor
    padding_mask: torch.Tensor
    original_sizes: torch.Tensor
    individual_padded_sizes: torch.Tensor
    original_size_list: tuple[tuple[int, int], ...]
    individual_padded_size_list: tuple[tuple[int, int], ...]

    @classmethod
    def from_images(
        cls,
        images: Sequence[torch.Tensor] | torch.Tensor,
        *,
        size_divisor: int = 32,
    ) -> "HeadImageBatch":
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                image_list = [images]
            elif images.ndim == 4:
                image_list = list(images.unbind(0))
            else:
                raise ValueError(f"Expected CHW or BCHW images, got shape={tuple(images.shape)}")
        else:
            image_list = list(images)
        if not image_list:
            raise ValueError("At least one image is required")
        if size_divisor <= 0:
            raise ValueError(f"size_divisor must be positive, got {size_divisor}")

        normalized = [_prepare_image(image) for image in image_list]
        device = normalized[0].device
        if any(image.device != device for image in normalized):
            raise ValueError("All images in a batch must be on the same device")

        original_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in normalized]
        padded_sizes = [
            (_ceil_to_multiple(height, size_divisor), _ceil_to_multiple(width, size_divisor))
            for height, width in original_sizes
        ]
        batch_h = max(height for height, _width in padded_sizes)
        batch_w = max(width for _height, width in padded_sizes)
        dtype = normalized[0].dtype
        batch = torch.zeros((len(normalized), 3, batch_h, batch_w), dtype=dtype, device=device)
        mask = torch.ones((len(normalized), batch_h, batch_w), dtype=torch.bool, device=device)
        for index, (image, (height, width)) in enumerate(zip(normalized, original_sizes, strict=True)):
            if image.dtype != dtype:
                image = image.to(dtype=dtype)
            batch[index, :, :height, :width] = image
            mask[index, :height, :width] = False
        return cls(
            tensors=batch,
            padding_mask=mask,
            original_sizes=torch.tensor(original_sizes, dtype=torch.long, device=device),
            individual_padded_sizes=torch.tensor(padded_sizes, dtype=torch.long, device=device),
            original_size_list=tuple(original_sizes),
            individual_padded_size_list=tuple(padded_sizes),
        )

    @property
    def batch_size(self) -> int:
        return int(self.tensors.shape[0])

    @property
    def padded_size(self) -> tuple[int, int]:
        return int(self.tensors.shape[-2]), int(self.tensors.shape[-1])

    def image_list(self) -> list[torch.Tensor]:
        """Return individually padded BCHW tensors for DINOv3 list-forward."""

        images: list[torch.Tensor] = []
        for index, (height, width) in enumerate(self.individual_padded_size_list):
            images.append(self.tensors[index : index + 1, :, :height, :width])
        return images

    def level_mask(
        self,
        size: tuple[int, int],
        *,
        stride: int | None = None,
    ) -> torch.Tensor:
        if stride is None:
            return F.interpolate(
                self.padding_mask[:, None].to(dtype=torch.float32),
                size=size,
                mode="nearest",
            ).to(dtype=torch.bool)[:, 0]
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        height, width = size
        rows = torch.arange(height, device=self.tensors.device) * stride
        columns = torch.arange(width, device=self.tensors.device) * stride
        valid_rows = rows[None, :] < self.original_sizes[:, 0:1]
        valid_columns = columns[None, :] < self.original_sizes[:, 1:2]
        return ~(valid_rows[:, :, None] & valid_columns[:, None, :])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "HeadImageBatch":
        return HeadImageBatch(
            tensors=self.tensors.to(device, non_blocking=non_blocking),
            padding_mask=self.padding_mask.to(device, non_blocking=non_blocking),
            original_sizes=self.original_sizes.to(device, non_blocking=non_blocking),
            individual_padded_sizes=self.individual_padded_sizes.to(device, non_blocking=non_blocking),
            original_size_list=self.original_size_list,
            individual_padded_size_list=self.individual_padded_size_list,
        )

    def pin_memory(self) -> "HeadImageBatch":
        return HeadImageBatch(
            tensors=self.tensors.pin_memory(),
            padding_mask=self.padding_mask.pin_memory(),
            original_sizes=self.original_sizes.pin_memory(),
            individual_padded_sizes=self.individual_padded_sizes.pin_memory(),
            original_size_list=self.original_size_list,
            individual_padded_size_list=self.individual_padded_size_list,
        )


def collate_head_detection_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[HeadImageBatch, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return HeadImageBatch.from_images(images), list(targets)


__all__ = [
    "HeadImageBatch",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "collate_head_detection_batch",
]
