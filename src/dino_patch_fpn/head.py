from __future__ import annotations

import torch
from torch import nn

from .adapter import ConvNormAct


class DenseFaceKeypointHead(nn.Module):
    """Anchor-free dense prediction head for class, bbox, centerness, and keypoints."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_classes: int = 2,
        num_keypoints: int = 5,
        keypoint_dims: int = 3,
        head_channels: int | None = None,
        num_convs: int = 2,
        use_ellipse: bool = False,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        if keypoint_dims not in {2, 3}:
            raise ValueError("keypoint_dims must be 2 or 3")
        if num_convs < 1:
            raise ValueError("num_convs must be >= 1")

        self.num_classes = int(num_classes)
        self.num_keypoints = int(num_keypoints)
        self.keypoint_dims = int(keypoint_dims)
        self.use_ellipse = bool(use_ellipse)
        hidden_channels = int(head_channels or in_channels)

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
        self.tower = nn.Sequential(*layers)
        self.cls_logits = nn.Conv2d(hidden_channels, self.num_classes, kernel_size=3, padding=1)
        self.bbox_reg = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.keypoints = nn.Conv2d(hidden_channels, self.num_keypoints * self.keypoint_dims, kernel_size=3, padding=1)
        self.ellipse = nn.Conv2d(hidden_channels, 5, kernel_size=3, padding=1) if self.use_ellipse else None

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for level, feature in features.items():
            hidden = self.tower(feature)
            level_outputs = {
                "cls_logits": self.cls_logits(hidden),
                "bbox_reg": self.bbox_reg(hidden),
                "centerness": self.centerness(hidden),
                "keypoints": self.keypoints(hidden),
            }
            if self.ellipse is not None:
                level_outputs["ellipse"] = self.ellipse(hidden)
            outputs[level] = level_outputs
        return outputs
