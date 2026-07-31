from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.dino_backbone import DinoBackbone, build_dino_backbone

from .heads import (
    LEGACY_BINARY_STATE_HEAD,
    LandmarkQueryHead,
    LandmarkQueryRefinementHead,
    ViTPoseHeatmapHead,
)


VITPOSE_VARIANTS = ("vitpose", "vitpose_376")
VARIANTS = (*VITPOSE_VARIANTS, "query", "query_refinement")


@lru_cache(maxsize=16)
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    state: Any = value.get("ema", value.get("model", value))
    if isinstance(state, Mapping) and isinstance(state.get("module"), Mapping):
        state = state["module"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint has no usable model state: {path}")
    return {str(key): tensor for key, tensor in state.items() if isinstance(tensor, torch.Tensor)}


def initialize_dino_from_deim_checkpoint(
    dino: DinoBackbone,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"DEIM checkpoint was not found: {path}")
    source = _checkpoint_state(path)
    prefixes = ("backbone.dino.backbone.", "module.backbone.dino.backbone.")
    extracted: dict[str, torch.Tensor] = {}
    for key, tensor in source.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = tensor
                break
    if not extracted:
        raise RuntimeError(f"No DINOv3 backbone tensors found in DEIM checkpoint: {path}")
    missing, unexpected = dino.backbone.load_state_dict(extracted, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "DINOv3 extraction was not strict: "
            f"missing={list(missing)[:20]}, unexpected={list(unexpected)[:20]}"
        )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(str(path.resolve())),
        "loaded_tensors": len(extracted),
        "loaded_elements": sum(int(value.numel()) for value in extracted.values()),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


class DinoFaceLandmarkModel(nn.Module):
    def __init__(self, backbone: DinoBackbone, head: nn.Module, *, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported variant={variant!r}; expected one of {VARIANTS}")
        self.backbone = backbone
        self.head = head
        self.variant = variant

    def forward(self, images: torch.Tensor, *, return_attention: bool = False) -> dict[str, Any]:
        feature_map = self.backbone(images)
        output = self.head(feature_map, return_attention=return_attention)
        output["feature_map_shape"] = tuple(feature_map.shape)
        return output

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()


def build_landmark_model(
    config: Mapping[str, Any],
    *,
    variant: str,
    backbone_module: nn.Module | None = None,
    initialize_from_detector: bool = True,
) -> tuple[DinoFaceLandmarkModel, dict[str, Any] | None]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant={variant!r}; expected one of {VARIANTS}")
    model_cfg = dict(config.get("model", config))
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone = build_dino_backbone(backbone_cfg, backbone=backbone_module)
    initialization = None
    detector_checkpoint = backbone_cfg.get("initialize_from_deim_checkpoint")
    if initialize_from_detector and detector_checkpoint:
        initialization = initialize_dino_from_deim_checkpoint(
            backbone,
            detector_checkpoint,
            strict=bool(backbone_cfg.get("strict_detector_initialization", True)),
        )
    in_channels = int(backbone_cfg.get("embed_dim", 384))
    num_landmarks = int(model_cfg.get("num_landmarks", 5))
    variants_cfg = dict(model_cfg.get("variants", {}))
    head_cfg = dict(variants_cfg.get(variant, {}))
    if variant in VITPOSE_VARIANTS:
        head: nn.Module = ViTPoseHeatmapHead(
            in_channels,
            num_landmarks=num_landmarks,
            hidden_channels=int(head_cfg.get("hidden_channels", 256)),
            deconv_layers=int(head_cfg.get("deconv_layers", 2)),
            deconv_kernel=int(head_cfg.get("deconv_kernel", 4)),
            softargmax_temperature=float(head_cfg.get("softargmax_temperature", 0.10)),
            state_head_mode=str(
                head_cfg.get(
                    "state_head_mode",
                    model_cfg.get(
                        "state_head_mode",
                        LEGACY_BINARY_STATE_HEAD,
                    ),
                )
            ),
        )
    else:
        query_cfg = dict(variants_cfg.get("query", {}))
        if variant == "query_refinement":
            query_cfg.update({key: value for key, value in head_cfg.items() if key != "stages"})
        query_head = LandmarkQueryHead(
            in_channels,
            num_landmarks=num_landmarks,
            dimension=int(query_cfg.get("dimension", 256)),
            decoder_layers=int(query_cfg.get("decoder_layers", 4)),
            heads=int(query_cfg.get("heads", 8)),
            feedforward=int(query_cfg.get("feedforward", 1024)),
            dropout=float(query_cfg.get("dropout", 0.0)),
            auxiliary_supervision=bool(query_cfg.get("auxiliary_supervision", True)),
            state_head_mode=str(
                query_cfg.get(
                    "state_head_mode",
                    model_cfg.get("state_head_mode", LEGACY_BINARY_STATE_HEAD),
                )
            ),
        )
        if variant == "query":
            head = query_head
        else:
            stages = list(
                head_cfg.get(
                    "stages",
                    [
                        {"radius_in_feature_cells": 2.0, "maximum_delta": 0.08, "samples_per_axis": 3},
                        {"radius_in_feature_cells": 1.0, "maximum_delta": 0.04, "samples_per_axis": 3},
                    ],
                )
            )
            head = LandmarkQueryRefinementHead(query_head, stages=stages)
    return DinoFaceLandmarkModel(backbone, head, variant=variant), initialization
