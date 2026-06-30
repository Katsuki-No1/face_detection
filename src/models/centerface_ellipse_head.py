from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dino_fpn import ConvNormAct


class CenterFaceEllipseHead(nn.Module):
    """CenterFace/FCOS-style anchor-free head with ellipse as the main target."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_keypoints: int = 5,
        predict_bbox: bool = True,
        predict_theta: bool = True,
        head_channels: int | None = None,
        num_convs: int = 2,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        if num_keypoints < 0:
            raise ValueError("num_keypoints must be >= 0")
        if num_convs < 1:
            raise ValueError("num_convs must be >= 1")
        self.num_keypoints = int(num_keypoints)
        self.predict_bbox = bool(predict_bbox)
        self.predict_theta = bool(predict_theta)
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

        self.center_tower = tower()
        self.ellipse_tower = tower()
        self.kpt_tower = tower() if self.num_keypoints > 0 else nn.Identity()
        self.center_logits = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.ellipse_reg = nn.Conv2d(hidden_channels, 5, kernel_size=3, padding=1)
        self.kpt_reg = nn.Conv2d(hidden_channels, self.num_keypoints * 2, kernel_size=3, padding=1) if self.num_keypoints > 0 else None
        self.bbox_reg = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1) if self.predict_bbox else None

        nn.init.constant_(self.center_logits.bias, -4.595)

    def _ellipse_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        center_offset = torch.sigmoid(raw[:, 0:2])
        radii = F.softplus(raw[:, 2:4]).clamp_min(1e-4)
        if self.predict_theta:
            theta = raw[:, 4:5]
        else:
            theta = torch.zeros_like(raw[:, 4:5])
        return torch.cat([center_offset, radii, theta], dim=1)

    def forward(self, features: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        center_logits: list[torch.Tensor] = []
        ellipse_reg: list[torch.Tensor] = []
        kpt_reg: list[torch.Tensor] = []
        bbox_reg: list[torch.Tensor] = []
        for feature in features:
            center_feature = self.center_tower(feature)
            ellipse_feature = self.ellipse_tower(feature)
            center_logits.append(self.center_logits(center_feature))
            ellipse_reg.append(self._ellipse_from_raw(self.ellipse_reg(ellipse_feature)))
            if self.kpt_reg is not None:
                kpt_reg.append(self.kpt_reg(self.kpt_tower(feature)))
            else:
                batch, _channels, height, width = feature.shape
                kpt_reg.append(feature.new_zeros((batch, 0, height, width)))
            if self.bbox_reg is not None:
                bbox_reg.append(F.softplus(self.bbox_reg(ellipse_feature)).clamp_min(0.0))

        outputs = {
            "center_logits": center_logits,
            "ellipse_reg": ellipse_reg,
            "kpt_reg": kpt_reg,
        }
        if self.predict_bbox:
            outputs["bbox_reg"] = bbox_reg
        return outputs
