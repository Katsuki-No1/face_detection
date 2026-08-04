from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .centerface_ellipse_head import CenterFaceEllipseHead
from .dino_backbone import DinoBackbone, DummyDinoBackbone, build_dino_backbone
from .dino_centerface_ellipse import DinoCenterFaceEllipseDetector, build_dino_centerface_ellipse_detector
from .dino_deim_style import (
    DETRStyleDecoder,
    DINODeimStyleCriterion,
    DINODeimStyleDetector,
    DINODeimStyleMatcher,
    DINODeimStylePostProcessor,
    HybridEncoder,
    SpatialTuningAdapter,
    build_dino_deim_style_detector,
)
from .dinov3_deimv2 import DINOv3DEIMv2Detector, build_dinov3_deimv2_detector
from .dinov3_deimv2_ellipse import DINOv3DEIMv2EllipseDetector, build_dinov3_deimv2_ellipse_detector
from .face_keypoint_cooperative import (
    FaceKeypointCooperativeDETR,
    build_face_keypoint_cooperative_detr,
)
from .head_codino import HeadCoDINODetector, build_head_codino_detector
from .dino_fpn import DinoFeatureAdapter, SimpleFPN
from .dino_scrfd import DinoSCRFDFaceDetector, build_dino_scrfd_face_detector
from .ellipse_postprocess import DinoCenterFaceEllipsePostProcessor, draw_ellipse_detections, ellipse_aware_nms, save_ellipse_overlay
from .ellipse_target_assigner import CenterFaceEllipseTargetAssigner
from .postprocess import DinoSCRFDPostProcessor, nms
from .scrfd_head import SCRFDHead
from .target_assigner import CenterFaceTargetAssigner


def build_detector_from_config(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> nn.Module:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    model_type = str(model_cfg.get("type", "dino_scrfd")).lower()
    if model_type in {"dino_centerface_ellipse", "dino_ellipse", "centerface_ellipse"}:
        return build_dino_centerface_ellipse_detector(cfg, backbone_module=backbone_module)
    if model_type in {"dinov3_deimv2", "dino_deimv2"}:
        return build_dinov3_deimv2_detector(cfg, backbone_module=backbone_module)
    if model_type in {"dinov3_deimv2_ellipse", "dino_deimv2_ellipse"}:
        return build_dinov3_deimv2_ellipse_detector(cfg, backbone_module=backbone_module)
    if model_type in {
        "face_keypoint_cooperative_detr",
        "face_keypoint_cooperative_deim",
        "cooperative_face_keypoint",
    }:
        if backbone_module is not None:
            raise ValueError(
                "FaceKeypointCooperativeDETR builds the shared pretrained "
                "DEIMv2-L backbone from its face_detector configuration."
            )
        return build_face_keypoint_cooperative_detr(cfg)
    if model_type in {"head_codino", "head_co_dino"}:
        return build_head_codino_detector(cfg, backbone_module=backbone_module)
    if model_type in {"dino_deim_style", "dino_rt_detr_style", "dino_rtdetr_style", "dino_deim"}:
        return build_dino_deim_style_detector(cfg, backbone_module=backbone_module)
    if model_type in {"dino_scrfd", "scrfd_dino", "dino_scrfd_face"}:
        return build_dino_scrfd_face_detector(cfg, backbone_module=backbone_module)
    raise ValueError(f"Unsupported model.type: {model_type}")


__all__ = [
    "CenterFaceTargetAssigner",
    "CenterFaceEllipseHead",
    "CenterFaceEllipseTargetAssigner",
    "DinoBackbone",
    "DinoCenterFaceEllipseDetector",
    "DinoCenterFaceEllipsePostProcessor",
    "DETRStyleDecoder",
    "DINODeimStyleCriterion",
    "DINODeimStyleDetector",
    "DINODeimStyleMatcher",
    "DINODeimStylePostProcessor",
    "DINOv3DEIMv2Detector",
    "DINOv3DEIMv2EllipseDetector",
    "FaceKeypointCooperativeDETR",
    "HeadCoDINODetector",
    "DinoFeatureAdapter",
    "DinoSCRFDFaceDetector",
    "DinoSCRFDPostProcessor",
    "DummyDinoBackbone",
    "HybridEncoder",
    "SCRFDHead",
    "SimpleFPN",
    "SpatialTuningAdapter",
    "build_detector_from_config",
    "build_dino_backbone",
    "build_dino_centerface_ellipse_detector",
    "build_dino_deim_style_detector",
    "build_dinov3_deimv2_detector",
    "build_dinov3_deimv2_ellipse_detector",
    "build_face_keypoint_cooperative_detr",
    "build_head_codino_detector",
    "build_dino_scrfd_face_detector",
    "draw_ellipse_detections",
    "ellipse_aware_nms",
    "nms",
    "save_ellipse_overlay",
]
