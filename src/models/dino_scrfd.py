from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.losses.dino_scrfd_loss import DinoSCRFDLoss

from .dino_backbone import DinoBackbone, build_dino_backbone
from .dino_fpn import DinoFeatureAdapter, SimpleFPN
from .postprocess import DinoSCRFDPostProcessor
from .scrfd_head import SCRFDHead
from .target_assigner import CenterFaceTargetAssigner


class DinoSCRFDFaceDetector(nn.Module):
    """Frozen DINO backbone with adapter/FPN/SCRFD-style dense detection head."""

    def __init__(
        self,
        *,
        backbone: DinoBackbone,
        adapter: DinoFeatureAdapter,
        fpn: SimpleFPN,
        head: SCRFDHead,
        loss_fn: DinoSCRFDLoss | None = None,
        postprocessor: DinoSCRFDPostProcessor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.fpn = fpn
        self.head = head
        self.loss_fn = loss_fn or DinoSCRFDLoss()
        self.postprocessor = postprocessor or DinoSCRFDPostProcessor()

        for module in (self.adapter, self.fpn, self.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def forward_raw(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        dino_feature = self.backbone(images)
        adapted = self.adapter(dino_feature)
        pyramid = self.fpn(adapted)
        return self.head(pyramid)

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


def build_dino_scrfd_face_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DinoSCRFDFaceDetector:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    neck_cfg = dict(model_cfg.get("neck", model_cfg.get("fpn", {})))
    adapter_cfg = dict(model_cfg.get("adapter", {}))
    head_cfg = dict(model_cfg.get("head", {}))
    loss_cfg = dict(cfg.get("loss", model_cfg.get("loss", {})))
    post_cfg = dict(cfg.get("postprocess", model_cfg.get("postprocess", {})))

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
    head = SCRFDHead(
        fpn_channels,
        num_classes=int(head_cfg.get("num_classes", 1)),
        num_keypoints=int(head_cfg.get("num_keypoints", 5)),
        head_channels=head_cfg.get("head_channels"),
        num_convs=int(head_cfg.get("num_convs", 2)),
        norm=head_cfg.get("norm", norm),
        activation=head_cfg.get("activation", activation),
    )

    assigner = CenterFaceTargetAssigner(center_sampling_radius=int(loss_cfg.get("center_sampling_radius", 0)))
    loss_fn = DinoSCRFDLoss(
        assigner=assigner,
        cls_weight=float(loss_cfg.get("cls_weight", 1.0)),
        bbox_weight=float(loss_cfg.get("bbox_weight", 2.0)),
        kpt_weight=float(loss_cfg.get("kpt_weight", 1.0)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )
    postprocessor = DinoSCRFDPostProcessor(
        score_threshold=float(post_cfg.get("score_threshold", 0.05)),
        nms_threshold=float(post_cfg.get("nms_threshold", 0.4)),
        topk=int(post_cfg.get("topk", 1000)),
        detections_per_image=int(post_cfg.get("detections_per_image", 300)),
    )
    return DinoSCRFDFaceDetector(
        backbone=backbone,
        adapter=adapter,
        fpn=fpn,
        head=head,
        loss_fn=loss_fn,
        postprocessor=postprocessor,
    )
