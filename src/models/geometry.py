from __future__ import annotations

import torch


def feature_strides(height: int, width: int, image_size: tuple[int, int]) -> tuple[float, float]:
    """Return image-space stride as (stride_x, stride_y) for a feature map."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Feature map shape must be positive, got height={height}, width={width}")
    image_h, image_w = float(image_size[0]), float(image_size[1])
    return image_w / float(width), image_h / float(height)


def grid_centers(
    height: int,
    width: int,
    image_size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    flatten: bool = False,
) -> torch.Tensor:
    """Return cell centers in image coordinates as [..., (x, y)]."""

    stride_x, stride_y = feature_strides(height, width, image_size)
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride_y
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([grid_x, grid_y], dim=-1)
    return centers.reshape(-1, 2) if flatten else centers


def grid_indices(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return flattened cell indices as [..., (col, row)]."""

    rows = torch.arange(height, device=device, dtype=dtype)
    cols = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
