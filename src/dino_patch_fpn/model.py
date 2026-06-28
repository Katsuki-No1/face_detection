from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .adapter import CNNAdapter
from .backbone import DinoPatchFeatureMapBackbone, FakeTokenBackbone
from .data import DEFAULT_CLASS_NAMES, DEFAULT_KEYPOINT_NAMES
from .fpn import SimpleFPN
from .head import DenseFaceKeypointHead


class DinoPatchFPNFaceKeypointModel(nn.Module):
    """DINO patch-token detector scaffold for face/head bbox and keypoints."""

    def __init__(
        self,
        dino_backbone: nn.Module,
        *,
        patch_size: int | tuple[int, int] = 14,
        dino_embed_dim: int = 768,
        num_prefix_tokens: int = 1,
        token_key: str | None = None,
        freeze_backbone: bool = True,
        adapter_out_channels: int = 256,
        adapter_layers: int = 2,
        fpn_out_channels: int | None = None,
        fpn_levels: tuple[str, ...] = ("p2", "p3", "p4", "p5"),
        num_classes: int = 2,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
        num_keypoints: int = 5,
        keypoint_names: tuple[str, ...] = DEFAULT_KEYPOINT_NAMES,
        keypoint_dims: int = 3,
        use_ellipse: bool = False,
        norm: str | None = "gn",
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.keypoint_names = tuple(keypoint_names)
        self.patch_backbone = DinoPatchFeatureMapBackbone(
            dino_backbone,
            patch_size=patch_size,
            num_prefix_tokens=num_prefix_tokens,
            token_key=token_key,
            freeze=freeze_backbone,
        )
        self.adapter = CNNAdapter(
            dino_embed_dim,
            adapter_out_channels,
            num_layers=adapter_layers,
            norm=norm,
            activation=activation,
        )
        fpn_channels = int(fpn_out_channels or adapter_out_channels)
        self.fpn = SimpleFPN(
            adapter_out_channels,
            fpn_channels,
            levels=fpn_levels,
            norm=norm,
            activation=activation,
        )
        self.head = DenseFaceKeypointHead(
            fpn_channels,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            keypoint_dims=keypoint_dims,
            use_ellipse=use_ellipse,
            norm=norm,
            activation=activation,
        )

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_map = self.patch_backbone(images)
        adapter_feature = self.adapter(token_map)
        fpn_features = self.fpn(adapter_feature)
        return {"token_map": token_map, "adapter_feature": adapter_feature, "fpn": fpn_features}

    def forward(self, images: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        features = self.forward_features(images)
        return self.head(features["fpn"])  # type: ignore[arg-type]


def build_dino_patch_fpn_model(config: dict[str, Any], backbone: nn.Module | None = None) -> DinoPatchFPNFaceKeypointModel:
    model_cfg = config.get("model", config)
    backbone_cfg = model_cfg.get("backbone", {})
    adapter_cfg = model_cfg.get("adapter", {})
    fpn_cfg = model_cfg.get("fpn", {})
    head_cfg = model_cfg.get("head", {})

    patch_size = int(backbone_cfg.get("patch_size", 14))
    embed_dim = int(backbone_cfg.get("embed_dim", 768))
    num_prefix_tokens = int(backbone_cfg.get("num_prefix_tokens", 1))
    backbone_type = str(backbone_cfg.get("type", "dino")).lower()
    token_key = backbone_cfg.get("token_key")

    if backbone is None:
        if backbone_type not in {"fake", "fake_dino", "dummy", "dummy_dino"}:
            raise ValueError(
                "A real DINO backbone must be passed as backbone=. "
                "Use backbone.type=fake only for dummy shape tests."
            )
        backbone = FakeTokenBackbone(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_prefix_tokens=num_prefix_tokens,
            output_key=str(backbone_cfg.get("output_key", "tokens")),
        )

    return DinoPatchFPNFaceKeypointModel(
        backbone,
        patch_size=patch_size,
        dino_embed_dim=embed_dim,
        num_prefix_tokens=num_prefix_tokens,
        token_key=token_key,
        freeze_backbone=bool(backbone_cfg.get("freeze", True)),
        adapter_out_channels=int(adapter_cfg.get("out_channels", 256)),
        adapter_layers=int(adapter_cfg.get("num_layers", 2)),
        fpn_out_channels=int(fpn_cfg.get("out_channels", adapter_cfg.get("out_channels", 256))),
        fpn_levels=tuple(fpn_cfg.get("levels", ("p2", "p3", "p4", "p5"))),
        num_classes=int(head_cfg.get("num_classes", 2)),
        class_names=tuple(head_cfg.get("class_names", DEFAULT_CLASS_NAMES)),
        num_keypoints=int(head_cfg.get("num_keypoints", 5)),
        keypoint_names=tuple(head_cfg.get("keypoint_names", DEFAULT_KEYPOINT_NAMES)),
        keypoint_dims=int(head_cfg.get("keypoint_dims", 3)),
        use_ellipse=bool(head_cfg.get("use_ellipse", False)),
        norm=str(adapter_cfg.get("norm", fpn_cfg.get("norm", "gn"))),
        activation=str(adapter_cfg.get("activation", fpn_cfg.get("activation", "silu"))),
    )
