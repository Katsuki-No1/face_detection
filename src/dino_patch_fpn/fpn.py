from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import ConvNormAct


class SimpleFPN(nn.Module):
    """Builds P2/P3/P4/P5 features from one DINO-derived feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        *,
        levels: tuple[str, ...] = ("p2", "p3", "p4", "p5"),
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        allowed = {"p2", "p3", "p4", "p5"}
        unknown = set(levels) - allowed
        if unknown:
            raise ValueError(f"Unsupported FPN levels: {sorted(unknown)}")
        self.levels = tuple(levels)
        self.p3 = ConvNormAct(in_channels, out_channels, kernel_size=3, norm=norm, activation=activation)
        self.p2 = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, activation=activation)
        self.p4 = ConvNormAct(out_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)
        self.p5 = ConvNormAct(out_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"SimpleFPN expects (B, C, H, W), got shape={tuple(feature.shape)}")
        p3 = self.p3(feature)
        p2 = self.p2(F.interpolate(p3, scale_factor=2.0, mode="nearest"))
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        features = {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
        return {level: features[level] for level in self.levels}
