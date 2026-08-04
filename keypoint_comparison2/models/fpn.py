from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.GroupNorm(32, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_channels),
        nn.GELU(),
    )


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style Simple FPN over one dense DINOv3 patch-feature map."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.scale4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
        )
        self.scale2 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.scale1 = nn.Identity()
        self.scale_half = nn.MaxPool2d(kernel_size=2, stride=2)
        self.projections = nn.ModuleList(
            (
                _projection(in_channels // 4, out_channels),
                _projection(in_channels // 2, out_channels),
                _projection(in_channels, out_channels),
                _projection(in_channels, out_channels),
            )
        )
        self.level_weights = nn.Parameter(torch.ones(4))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        raw_levels = (
            self.scale4(feature),
            self.scale2(feature),
            self.scale1(feature),
            self.scale_half(feature),
        )
        levels = tuple(layer(value) for layer, value in zip(self.projections, raw_levels))
        target_size = levels[0].shape[-2:]
        weights = self.level_weights.softmax(0)
        fused = sum(
            weight * F.interpolate(level, size=target_size, mode="bilinear", align_corners=False)
            for weight, level in zip(weights, levels)
        )
        return self.fuse(fused), levels
