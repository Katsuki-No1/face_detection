from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dino_fpn import ConvNormAct


class SCRFDHead(nn.Module):
    """Lightweight SCRFD-style anchor-free head for face bbox and 5 landmarks."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_classes: int = 1,
        num_keypoints: int = 5,
        head_channels: int | None = None,
        num_convs: int = 2,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("This face detector currently expects one foreground class")
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        if num_convs < 1:
            raise ValueError("num_convs must be >= 1")

        self.num_classes = int(num_classes)
        self.num_keypoints = int(num_keypoints)
        hidden_channels = int(head_channels or in_channels)

        def tower() -> nn.Sequential:
            layers: list[nn.Module] = []
            current_channels = in_channels
            for _ in range(num_convs):
                layers.append(
                    ConvNormAct(
                        current_channels,
                        hidden_channels,
                        kernel_size=3,
                        norm=norm,
                        activation=activation,
                    )
                )
                current_channels = hidden_channels
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.box_tower = tower()
        self.kpt_tower = tower()
        self.cls_logits = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.bbox_reg = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)
        self.kpt_reg = nn.Conv2d(hidden_channels, self.num_keypoints * 2, kernel_size=3, padding=1)

        nn.init.constant_(self.cls_logits.bias, -4.595)

    def forward(self, features: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        cls_logits: list[torch.Tensor] = []
        bbox_reg: list[torch.Tensor] = []
        kpt_reg: list[torch.Tensor] = []
        for feature in features:
            cls_logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(F.softplus(self.bbox_reg(self.box_tower(feature))))
            kpt_reg.append(self.kpt_reg(self.kpt_tower(feature)))
        return {"cls_logits": cls_logits, "bbox_reg": bbox_reg, "kpt_reg": kpt_reg}
