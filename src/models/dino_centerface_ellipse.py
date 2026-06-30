from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.losses.dino_centerface_ellipse_loss import DinoCenterFaceEllipseLoss

from .centerface_ellipse_head import CenterFaceEllipseHead
from .dino_backbone import DinoBackbone, build_dino_backbone
from .dino_fpn import DinoFeatureAdapter, SimpleFPN
from .ellipse_postprocess import DinoCenterFaceEllipsePostProcessor
from .ellipse_target_assigner import CenterFaceEllipseTargetAssigner


class DinoCenterFaceEllipseDetector(nn.Module):
    """DINO backbone with CNN/FPN adapter and CenterFace-style ellipse head."""

    def __init__(
        self,
        *,
        backbone: DinoBackbone,
        adapter: DinoFeatureAdapter,
        fpn: SimpleFPN,
        head: CenterFaceEllipseHead,
        loss_fn: DinoCenterFaceEllipseLoss | None = None,
        postprocessor: DinoCenterFaceEllipsePostProcessor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.fpn = fpn
        self.head = head
        self.loss_fn = loss_fn or DinoCenterFaceEllipseLoss()
        self.postprocessor = postprocessor or DinoCenterFaceEllipsePostProcessor()

        for module in (self.adapter, self.fpn, self.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def forward_raw(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        dino_feature = self.backbone(images)
        return self.forward_raw_from_dino_feature(dino_feature)

    def forward_raw_from_dino_feature(self, dino_feature: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        adapted = self.adapter(dino_feature)
        pyramid = self.fpn(adapted)
        return self.head(pyramid)

    def forward_from_dino_feature(
        self,
        dino_feature: torch.Tensor,
        *,
        image_size: tuple[int, int],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        predictions = self.forward_raw_from_dino_feature(dino_feature)
        if targets is not None:
            return self.loss_fn(predictions, targets, image_size)
        return self.postprocessor.decode(predictions, image_size)

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        predictions = self.forward_raw(images)
        image_size = (int(images.shape[-2]), int(images.shape[-1]))
        if targets is not None:
            return self.loss_fn(predictions, targets, image_size)
        return self.postprocessor.decode(predictions, image_size)


def build_dino_centerface_ellipse_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DinoCenterFaceEllipseDetector:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    adapter_cfg = dict(model_cfg.get("adapter", {}))
    neck_cfg = dict(model_cfg.get("neck", model_cfg.get("fpn", {})))
    head_cfg = dict(model_cfg.get("head", {}))
    loss_cfg = dict(cfg.get("loss", model_cfg.get("loss", {})))
    inference_cfg = dict(cfg.get("inference", cfg.get("postprocess", model_cfg.get("postprocess", {}))))

    backbone = build_dino_backbone(backbone_cfg, backbone=backbone_module)
    embed_dim = int(backbone_cfg.get("embed_dim", backbone.embed_dim))
    adapter_channels = int(adapter_cfg.get("out_channels", neck_cfg.get("out_channels", 256)))
    fpn_channels = int(neck_cfg.get("out_channels", adapter_channels))
    norm = adapter_cfg.get("norm", neck_cfg.get("norm", "gn"))
    activation = adapter_cfg.get("activation", neck_cfg.get("activation", "silu"))

    adapter = DinoFeatureAdapter(
        embed_dim,
        adapter_channels,
        norm=norm,
        activation=activation,
    )
    fpn = SimpleFPN(
        adapter_channels,
        fpn_channels,
        num_levels=int(neck_cfg.get("num_levels", 3)),
        norm=neck_cfg.get("norm", norm),
        activation=neck_cfg.get("activation", activation),
    )
    head = CenterFaceEllipseHead(
        fpn_channels,
        num_keypoints=int(head_cfg.get("num_keypoints", 5)),
        predict_bbox=bool(head_cfg.get("predict_bbox", True)),
        predict_theta=bool(head_cfg.get("predict_theta", True)),
        head_channels=head_cfg.get("head_channels"),
        num_convs=int(head_cfg.get("num_convs", 2)),
        norm=head_cfg.get("norm", norm),
        activation=head_cfg.get("activation", activation),
    )

    assigner = CenterFaceEllipseTargetAssigner()
    loss_fn = DinoCenterFaceEllipseLoss(
        assigner=assigner,
        center_weight=float(loss_cfg.get("center", loss_cfg.get("center_weight", 1.0))),
        ellipse_weight=float(loss_cfg.get("ellipse", loss_cfg.get("ellipse_weight", 5.0))),
        kpt_weight=float(loss_cfg.get("keypoint", loss_cfg.get("kpt_weight", 1.0))),
        bbox_weight=float(loss_cfg.get("bbox", loss_cfg.get("bbox_weight", 0.5))),
        focal_alpha=float(loss_cfg.get("focal_alpha", 2.0)),
        focal_beta=float(loss_cfg.get("focal_beta", 4.0)),
    )
    postprocessor = DinoCenterFaceEllipsePostProcessor(
        score_threshold=float(inference_cfg.get("score_threshold", 0.3)),
        nms_threshold=float(inference_cfg.get("nms_threshold", 0.5)),
        topk=int(inference_cfg.get("topk", 100)),
        detections_per_image=int(inference_cfg.get("detections_per_image", inference_cfg.get("topk", 100))),
        bbox_source=str(inference_cfg.get("bbox_source", "bbox")),
    )
    return DinoCenterFaceEllipseDetector(
        backbone=backbone,
        adapter=adapter,
        fpn=fpn,
        head=head,
        loss_fn=loss_fn,
        postprocessor=postprocessor,
    )
