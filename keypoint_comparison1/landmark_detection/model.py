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
    ConvKVFeaturePyramid,
    IntermediateFeaturePyramid,
    LEGACY_BINARY_STATE_HEAD,
    LandmarkQueryHead,
    LandmarkQueryRefinementHead,
    RGBRefinementFeaturePyramid,
    ViTPoseHeatmapHead,
)
from .messenger_recovery import MessengerRecoveryBlock, MessengerRecoveryConfig
from .continuous_recovery import ContinuousRecoveryBlock, ContinuousRecoveryConfig


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


def initialize_rgb_refinement_from_deim_checkpoint(
    module: RGBRefinementFeaturePyramid,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Copy the matching DEIM STA stride-4/8 stem into the crop CNN."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"DEIM checkpoint was not found: {path}")
    source = _checkpoint_state(path)
    source_suffixes = {
        "stem_conv.weight": "backbone.sta.stem.0.weight",
        "stem_norm.weight": "backbone.sta.stem.1.weight",
        "stem_norm.bias": "backbone.sta.stem.1.bias",
        "stride8_conv.weight": "backbone.sta.conv2.0.weight",
        "stride8_norm.weight": "backbone.sta.conv2.1.weight",
        "stride8_norm.bias": "backbone.sta.conv2.1.bias",
    }
    target = module.state_dict()
    extracted: dict[str, torch.Tensor] = {}
    for target_key, source_suffix in source_suffixes.items():
        matches = [
            tensor
            for key, tensor in source.items()
            if key == source_suffix or key.endswith("." + source_suffix)
        ]
        if len(matches) == 1 and tuple(matches[0].shape) == tuple(
            target[target_key].shape
        ):
            extracted[target_key] = matches[0]
    missing_required = sorted(set(source_suffixes) - set(extracted))
    if strict and missing_required:
        raise RuntimeError(
            "RGB refinement STA initialization is incomplete: "
            f"missing={missing_required}"
        )
    missing, unexpected = module.load_state_dict(extracted, strict=False)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(str(path.resolve())),
        "loaded_tensors": sorted(extracted),
        "missing_required": missing_required,
        "new_tensors": list(missing),
        "unexpected": list(unexpected),
        "normalization_transfer": "batch_norm_affine_to_group_norm",
    }


class DinoFaceLandmarkModel(nn.Module):
    def __init__(
        self,
        backbone: DinoBackbone,
        head: nn.Module,
        *,
        variant: str,
        intermediate_layers: tuple[int, ...] = (),
        rgb_refinement: RGBRefinementFeaturePyramid | None = None,
        messenger_recovery: MessengerRecoveryBlock | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported variant={variant!r}; expected one of {VARIANTS}")
        self.backbone = backbone
        self.head = head
        self.variant = variant
        self.intermediate_layers = tuple(int(index) for index in intermediate_layers)
        self.rgb_refinement = rgb_refinement
        self.messenger_recovery = messenger_recovery

    @property
    def face_overlap_enabled(self) -> bool:
        query_head = getattr(self.head, "query_head", self.head)
        return getattr(query_head, "face_overlap", None) is not None

    def forward(
        self,
        images: torch.Tensor,
        *,
        return_attention: bool = False,
        augmentation_valid_mask: torch.Tensor | None = None,
        recovery_mode: str = "recovered",
        return_recovery_diagnostics: bool = False,
        return_recovery_attention: bool = False,
        face_query_features: torch.Tensor | None = None,
        face_geometry: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if recovery_mode not in {"recovered", "raw"}:
            raise ValueError("recovery_mode must be 'recovered' or 'raw'")
        face_kwargs = {}
        if self.face_overlap_enabled or face_query_features is not None or face_geometry is not None:
            if not self.face_overlap_enabled:
                raise ValueError("FOR inputs supplied while face_overlap_robustness is disabled")
            face_kwargs = {"face_query_features": face_query_features, "face_geometry": face_geometry}
        feature_map: torch.Tensor | tuple[torch.Tensor, ...]
        reference_maps = None
        continuous = self.messenger_recovery if isinstance(self.messenger_recovery, ContinuousRecoveryBlock) else None
        if continuous is not None and continuous.config.multi_level:
            reference_maps = self.backbone.forward_intermediate_feature_maps(
                images, layer_indexes=continuous.config.reference_layers)
            feature_map = reference_maps[-1]
        elif self.intermediate_layers:
            feature_map = self.backbone.forward_intermediate_feature_maps(
                images,
                layer_indexes=self.intermediate_layers,
            )
        else:
            feature_map = self.backbone(images)
        recovery_diagnostics: dict[str, torch.Tensor] | None = None
        if self.messenger_recovery is not None:
            if not isinstance(feature_map, torch.Tensor):
                raise ValueError(
                    "Messenger v1 supports only the final DINO patch-token map"
                )
            recovery_kwargs = {"reference_maps": reference_maps} if continuous is not None else {}
            recovered_map, recovery_diagnostics = self.messenger_recovery(
                feature_map, return_diagnostics=return_recovery_attention, **recovery_kwargs)
            head_feature_map = (
                recovered_map if recovery_mode == "recovered" else feature_map
            )
        else:
            head_feature_map = feature_map
        if self.variant == "query_refinement":
            refinement_feature_maps = (
                self.rgb_refinement(images)
                if self.rgb_refinement is not None
                else None
            )
            output = self.head(
                head_feature_map,
                return_attention=return_attention,
                augmentation_valid_mask=augmentation_valid_mask,
                refinement_feature_maps=refinement_feature_maps,
                **face_kwargs,
            )
        else:
            if augmentation_valid_mask is not None:
                raise ValueError(
                    "augmentation_valid_mask is supported only by query_refinement"
                )
            output = self.head(head_feature_map, return_attention=return_attention, **face_kwargs)
        output["feature_map_shape"] = (
            tuple(feature_map.shape)
            if isinstance(feature_map, torch.Tensor)
            else tuple(tuple(feature.shape) for feature in feature_map)
        )
        if self.messenger_recovery is not None:
            output["recovery_mode"] = recovery_mode
        if return_recovery_diagnostics:
            if recovery_diagnostics is None:
                output["recovery_diagnostics"] = None
            else:
                output["recovery_diagnostics"] = recovery_diagnostics
        return output

    def extract_backbone_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Extract raw final DINO features for the stop-gradient clean teacher."""

        if self.intermediate_layers:
            raise ValueError(
                "Messenger v1 clean teacher does not support intermediate features"
            )
        return self.backbone(images)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def rgb_refinement_parameters(self):
        if self.rgb_refinement is None:
            return iter(())
        return self.rgb_refinement.parameters()

    def messenger_parameters(self):
        if self.messenger_recovery is None:
            return iter(())
        return self.messenger_recovery.parameters()


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
    pyramid_cfg = dict(model_cfg.get("feature_pyramid", {}) or {})
    semantic_cfg = dict(model_cfg.get("semantic_feature_levels", {}) or {})
    kv_pyramid_cfg = dict(model_cfg.get("kv_feature_pyramid", {}) or {})
    rgb_refinement_cfg = dict(model_cfg.get("rgb_refinement", {}) or {})
    messenger_cfg = MessengerRecoveryConfig.from_mapping(
        model_cfg.get("messenger_recovery", {}) or {}
    )
    pyramid_enabled = bool(pyramid_cfg.get("enabled", False))
    semantic_enabled = bool(semantic_cfg.get("enabled", False))
    kv_pyramid_enabled = bool(kv_pyramid_cfg.get("enabled", False))
    rgb_refinement_enabled = bool(rgb_refinement_cfg.get("enabled", False))
    continuous_cfg = ContinuousRecoveryConfig.from_mapping(
        model_cfg.get("continuous_recovery"), embed_dim=int(backbone_cfg.get("embed_dim", 384))
    )
    if messenger_cfg.enabled and continuous_cfg.enabled:
        raise ValueError("Legacy and continuous recovery cannot both be enabled")
    messenger_enabled = bool(messenger_cfg.enabled or continuous_cfg.enabled)
    if pyramid_enabled and semantic_enabled:
        raise ValueError(
            "feature_pyramid and semantic_feature_levels are mutually exclusive"
        )
    if kv_pyramid_enabled and not semantic_enabled:
        raise ValueError("kv_feature_pyramid requires semantic_feature_levels")
    if rgb_refinement_enabled and variant != "query_refinement":
        raise ValueError("rgb_refinement is supported only by query_refinement")
    if rgb_refinement_enabled and not semantic_enabled:
        raise ValueError("rgb_refinement requires semantic_feature_levels")
    if messenger_enabled and (pyramid_enabled or semantic_enabled or rgb_refinement_enabled):
        raise ValueError(
            "Messenger v1 cannot be combined with feature pyramids or RGB refinement"
        )
    if messenger_enabled and variant != "query_refinement":
        raise ValueError("Messenger v1 is supported only by query_refinement")
    active_intermediate_cfg = semantic_cfg if semantic_enabled else pyramid_cfg
    intermediate_layers = tuple(
        int(index)
        for index in active_intermediate_cfg.get("intermediate_layers", ())
    )
    if pyramid_enabled:
        if variant not in {"query", "query_refinement"}:
            raise ValueError(
                "The DINO intermediate FPN is supported by query variants only"
            )
        if len(intermediate_layers) != 3:
            raise ValueError(
                "model.feature_pyramid.intermediate_layers must define exactly "
                "three DINO block indexes"
            )
    elif intermediate_layers:
        if not semantic_enabled:
            raise ValueError(
                "model.feature_pyramid.intermediate_layers requires enabled=true"
            )
    if semantic_enabled:
        if variant not in {"query", "query_refinement"}:
            raise ValueError(
                "DINO semantic feature levels are supported by query variants only"
            )
        if len(intermediate_layers) != 3:
            raise ValueError(
                "model.semantic_feature_levels.intermediate_layers must define "
                "exactly three DINO block indexes"
            )
    for_cfg = dict(model_cfg.get("face_overlap_robustness", {}) or {})
    if for_cfg.get("enabled", False) and variant not in {"query", "query_refinement"}:
        raise ValueError("FOR supports only query and query_refinement variants")
    backbone = build_dino_backbone(backbone_cfg, backbone=backbone_module)
    if continuous_cfg.enabled and continuous_cfg.multi_level:
        blocks = getattr(backbone.backbone, "blocks", ())
        if len(blocks) != 12 or int(backbone_cfg.get("embed_dim", 384)) != 384:
            raise ValueError("Multi-level recovery requires the 12-block, 384-dimensional ViT-S/16")
    initialization: dict[str, Any] | None = None
    detector_checkpoint = backbone_cfg.get("initialize_from_deim_checkpoint")
    if initialize_from_detector and detector_checkpoint:
        initialization = initialize_dino_from_deim_checkpoint(
            backbone,
            detector_checkpoint,
            strict=bool(backbone_cfg.get("strict_detector_initialization", True)),
        )
    rgb_refinement = (
        RGBRefinementFeaturePyramid(
            base_channels=int(rgb_refinement_cfg.get("base_channels", 32)),
            projection_channels=int(
                rgb_refinement_cfg.get("projection_channels", 128)
            ),
        )
        if rgb_refinement_enabled
        else None
    )
    rgb_checkpoint = rgb_refinement_cfg.get("initialize_from_deim_checkpoint")
    if initialize_from_detector and rgb_refinement is not None and rgb_checkpoint:
        rgb_initialization = initialize_rgb_refinement_from_deim_checkpoint(
            rgb_refinement,
            rgb_checkpoint,
            strict=bool(rgb_refinement_cfg.get("strict_initialization", True)),
        )
        initialization = dict(initialization or {})
        initialization["rgb_refinement"] = rgb_initialization
    backbone_channels = int(backbone_cfg.get("embed_dim", 384))
    messenger_recovery = (
        (ContinuousRecoveryBlock(backbone_channels, continuous_cfg)
         if continuous_cfg.enabled else MessengerRecoveryBlock(backbone_channels, messenger_cfg))
        if messenger_enabled
        else None
    )
    pyramid_channels = int(pyramid_cfg.get("out_channels", backbone_channels))
    in_channels = pyramid_channels if pyramid_enabled else backbone_channels
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
        feature_pyramid = (
            IntermediateFeaturePyramid(
                backbone_channels,
                pyramid_channels,
                num_levels=3,
            )
            if pyramid_enabled
            else None
        )
        kv_feature_pyramid = (
            ConvKVFeaturePyramid(
                backbone_channels,
                semantic_levels=len(intermediate_layers),
            )
            if kv_pyramid_enabled
            else None
        )
        feature_levels = 3 if pyramid_enabled or semantic_enabled else 1
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
            iterative_reference_points=bool(
                query_cfg.get("iterative_reference_points", False)
            ),
            initial_reference_points=query_cfg.get("initial_reference_points"),
            coordinate_update_mode=str(
                query_cfg.get("coordinate_update_mode", "absolute")
            ),
            feature_pyramid=feature_pyramid,
            kv_feature_pyramid=kv_feature_pyramid,
            feature_levels=feature_levels,
            face_query_dim=(int(for_cfg.get("query_dim", 256)) if for_cfg.get("enabled", False) else None),
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
            if rgb_refinement is not None:
                stage_sources = tuple(
                    str(stage.get("feature_source", "")) for stage in stages
                )
                if stage_sources != rgb_refinement.feature_sources:
                    raise ValueError(
                        "RGB refinement stages must consume stride8 then stride4"
                    )
            head = LandmarkQueryRefinementHead(
                query_head,
                stages=stages,
                input_keypoint_error_augmentation=head_cfg.get(
                    "input_keypoint_error_augmentation"
                ),
                refinement_feature_channels=(
                    rgb_refinement.out_channels
                    if rgb_refinement is not None
                    else None
                ),
            )
    if rgb_refinement is not None and bool(
        rgb_refinement_cfg.get("freeze", False)
    ):
        for parameter in rgb_refinement.parameters():
            parameter.requires_grad_(False)
    if bool(head_cfg.get("freeze", False)):
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    return (
        DinoFaceLandmarkModel(
            backbone,
            head,
            variant=variant,
            intermediate_layers=(
                intermediate_layers
                if pyramid_enabled or semantic_enabled
                else ()
            ),
            rgb_refinement=rgb_refinement,
            messenger_recovery=messenger_recovery,
        ),
        initialization,
    )
