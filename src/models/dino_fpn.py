from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_norm(norm: str | None, channels: int) -> nn.Module:
    if norm is None or norm.lower() in {"", "none", "identity"}:
        return nn.Identity()
    normalized = norm.lower()
    if normalized in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"gn", "groupnorm", "group_norm"}:
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"ln", "layernorm", "layer_norm"}:
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm: {norm}")


def make_activation(activation: str | None) -> nn.Module:
    if activation is None or activation.lower() in {"", "none", "identity"}:
        return nn.Identity()
    normalized = activation.lower()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=norm is None or str(norm).lower() in {"", "none", "identity"},
            ),
            make_norm(norm, out_channels),
            make_activation(activation),
        )


class DinoFeatureAdapter(nn.Module):
    """1x1 + 3x3 adapter from DINO token channels to detector channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        *,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        self.proj = ConvNormAct(in_channels, out_channels, kernel_size=1, norm=norm, activation=activation)
        self.refine = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, activation=activation)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"DinoFeatureAdapter expects (B, C, H, W), got {tuple(feature.shape)}")
        return self.refine(self.proj(feature))


class SimpleFPN(nn.Module):
    """Builds a compact P3/P4/P5 pyramid from one DINO feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        *,
        num_levels: int = 3,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        if num_levels not in {1, 2, 3}:
            raise ValueError("SimpleFPN currently supports 1, 2, or 3 levels")
        self.num_levels = int(num_levels)
        self.same = ConvNormAct(in_channels, out_channels, kernel_size=3, norm=norm, activation=activation)
        self.up = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, activation=activation)
        self.down = ConvNormAct(out_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"SimpleFPN expects (B, C, H, W), got {tuple(feature.shape)}")
        p4 = self.same(feature)
        if self.num_levels == 1:
            return [p4]
        p3 = self.up(F.interpolate(p4, scale_factor=2.0, mode="nearest"))
        if self.num_levels == 2:
            return [p3, p4]
        p5 = self.down(p4)
        return [p3, p4, p5]
