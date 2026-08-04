from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .dinov3 import DinoV3FPNRTMCC, DinoV3ORFormerRTMCC
from .joint_deim import DEIMv2JointFaceLandmarks
from .rtmpose import RTMPoseFace5


MODEL_NAMES = ("rtmpose_face5", "dinov3_fpn_rtmcc", "dinov3_orformer_rtmcc", "deimv2_joint")


def build_model(
    name: str,
    config: Mapping[str, Any],
    *,
    backbone_module: nn.Module | None = None,
    detector: nn.Module | None = None,
) -> nn.Module:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    model_cfg = dict(config.get("models", config)[name])
    if name == "rtmpose_face5":
        return RTMPoseFace5(model_cfg)
    if name == "dinov3_fpn_rtmcc":
        return DinoV3FPNRTMCC(model_cfg, backbone_module=backbone_module)
    if name == "dinov3_orformer_rtmcc":
        return DinoV3ORFormerRTMCC(model_cfg, backbone_module=backbone_module)
    joint_cfg = dict(model_cfg)
    joint_cfg["detector"] = dict(config.get("detector", joint_cfg.get("detector", {})))
    return DEIMv2JointFaceLandmarks(joint_cfg, detector=detector)
