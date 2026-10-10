from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class CoDINOChannelMapper(nn.Module):
    """Project a backbone pyramid and append Co-DINO feature levels.

    This is the current-PyTorch equivalent of the five-scale ChannelMapper
    used by the upstream Co-DINO configurations. Input feature order is from
    highest to lowest resolution; extra levels are produced with stride-2
    convolutions from the last output.
    """

    def __init__(
        self,
        *,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_outs: int = 5,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in in_channels)
        if not channels:
            raise ValueError("in_channels must contain at least one feature level")
        if int(num_outs) < len(channels):
            raise ValueError("num_outs cannot be smaller than the input pyramid")
        if int(out_channels) % int(norm_groups):
            raise ValueError("out_channels must be divisible by norm_groups")
        self.in_channels = channels
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel, self.out_channels, kernel_size=1),
                nn.GroupNorm(int(norm_groups), self.out_channels),
            )
            for channel in channels
        )
        self.extra_levels = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(int(norm_groups), self.out_channels),
            )
            for _ in range(self.num_outs - len(channels))
        )
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if len(features) != len(self.projections):
            raise ValueError(
                f"ChannelMapper expects {len(self.projections)} inputs, got "
                f"{len(features)}"
            )
        outputs = [
            projection(feature)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        for layer in self.extra_levels:
            outputs.append(layer(outputs[-1]))
        return tuple(outputs)


__all__ = ["CoDINOChannelMapper"]
