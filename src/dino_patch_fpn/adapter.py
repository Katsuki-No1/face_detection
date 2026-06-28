from __future__ import annotations

import torch
from torch import nn


def make_norm(norm: str | None, channels: int) -> nn.Module:
    if norm is None or norm.lower() in {"", "none", "identity"}:
        return nn.Identity()
    normalized = norm.lower()
    if normalized in {"bn", "batch", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"gn", "group", "groupnorm", "group_norm"}:
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"ln", "layer", "layernorm", "layer_norm"}:
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm: {norm}")


def make_activation(activation: str | None) -> nn.Module:
    if activation is None or activation.lower() in {"", "none", "identity"}:
        return nn.Identity()
    normalized = activation.lower()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
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
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=norm in {None, "", "none"}),
            make_norm(norm, out_channels),
            make_activation(activation),
        )


class CNNAdapter(nn.Module):
    """Adapts a DINO token map into CNN-friendly feature channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        *,
        num_layers: int = 2,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for layer_index in range(num_layers):
            kernel_size = 1 if layer_index == 0 else 3
            layers.append(
                ConvNormAct(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                )
            )
            current_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"CNNAdapter expects (B, C, H, W), got shape={tuple(feature.shape)}")
        return self.layers(feature)
