from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


EXPECTED_SFP_PARAMETERS = 4_699_264


class LayerNorm2d(nn.Module):
    """Affine LayerNorm over channels for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"LayerNorm2d expects NCHW input, got {tuple(value.shape)}")
        return F.layer_norm(
            value.permute(0, 2, 3, 1),
            (value.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2).contiguous()


class _P4Projection(nn.Sequential):
    def __init__(self, in_channels: int = 384, out_channels: int = 256) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )


class SimpleFeaturePyramid(nn.Module):
    """Checkpoint-compatible P3/P4/P5/P6 pyramid from one stride-16 map."""

    out_channels = 256
    strides = OrderedDict((("p3", 8), ("p4", 16), ("p5", 32), ("p6", 64)))

    def __init__(self, in_channels: int = 384) -> None:
        super().__init__()
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, self.out_channels, kernel_size=1, bias=False),
            LayerNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.out_channels),
        )
        self.p4 = _P4Projection(in_channels, self.out_channels)
        self.p5_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5 = _P4Projection(in_channels, self.out_channels)
        self.p6_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p6 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
            LayerNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.out_channels),
        )
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != EXPECTED_SFP_PARAMETERS:
            raise RuntimeError(
                f"Unexpected SFP parameter count: expected={EXPECTED_SFP_PARAMETERS}, actual={count}"
            )

    def forward(self, feature: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if feature.ndim != 4 or feature.shape[1] != 384:
            raise ValueError(f"SFP expects [B,384,H,W], got {tuple(feature.shape)}")
        p3 = self.p3(F.interpolate(feature, scale_factor=2.0, mode="bilinear", align_corners=False))
        p4 = self.p4(feature)
        p5 = self.p5(self.p5_pool(feature))
        p6 = self.p6(self.p6_pool(feature))
        return OrderedDict((("p3", p3), ("p4", p4), ("p5", p5), ("p6", p6)))


__all__ = ["EXPECTED_SFP_PARAMETERS", "LayerNorm2d", "SimpleFeaturePyramid"]
