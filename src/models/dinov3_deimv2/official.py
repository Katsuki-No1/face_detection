from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dino_backbone import DinoBackbone, build_dino_backbone
from src.third_party.deimv2.engine.deim.deim_criterion import DEIMCriterion
from src.third_party.deimv2.engine.deim.deim_decoder import DEIMTransformer
from src.third_party.deimv2.engine.deim.hybrid_encoder import HybridEncoder
from src.third_party.deimv2.engine.deim.matcher import HungarianMatcher
from src.third_party.deimv2.engine.deim.postprocessor import PostProcessor
from src.third_party.deimv2.engine.backbone.vit_tiny import VisionTransformer


def _cfg_section(cfg: Mapping[str, Any], model_cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = model_cfg.get(name)
    if value is None:
        value = cfg.get(name, {})
    return dict(value or {})


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    return int(value[0]), int(value[1])


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


class SpatialPriorModuleV2(nn.Module):
    """Official DEIMv2-style spatial prior module for DINOv3 STA."""

    def __init__(self, inplanes: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )

    def forward_with_high_resolution(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the existing stride-4 stem and the detector pyramid.

        ``c1`` was always computed because it feeds ``c2`` but the historical
        detector contract discarded it.  Exposing it through a separate method
        lets optional dense landmark heads reuse the RGB detail without adding
        another image encoder or changing the normal backbone output.
        """

        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c1, (c2, c3, c4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _high_resolution, pyramid = self.forward_with_high_resolution(x)
        return pyramid


class DINOv3STABackbone(nn.Module):
    """DINOv3 + official DEIMv2 STA-style multi-scale feature adapter.

    The upstream DEIMv2 ``DINOv3STAs`` builds its own DINOv3/ViT-Tiny. This
    repository already has a local DINO loader and checkpoint configuration, so
    this module preserves the official STA data flow while reusing
    ``DinoBackbone`` for the DINO model.
    """

    def __init__(
        self,
        *,
        backbone_cfg: Mapping[str, Any],
        hidden_dim: int,
        interaction_indexes: tuple[int, ...] = (3, 7, 11),
        conv_inplane: int = 16,
        use_sta: bool = True,
    ) -> None:
        super().__init__()
        self.dino = build_dino_backbone(backbone_cfg)
        self.embed_dim = int(backbone_cfg.get("embed_dim") or self.dino.embed_dim)
        self.patch_size = _as_pair(backbone_cfg.get("patch_size", self.dino.patch_size))
        self.interaction_indexes = tuple(int(index) for index in interaction_indexes)
        self.use_sta = bool(use_sta)
        self.hidden_dim = int(hidden_dim)
        self.spatial_high_resolution_channels = (
            int(conv_inplane) if self.use_sta else 0
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if self.use_sta:
            self.sta = SpatialPriorModuleV2(inplanes=int(conv_inplane))
        else:
            conv_inplane = 0
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 2, self.hidden_dim, kernel_size=1, bias=False),
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 4, self.hidden_dim, kernel_size=1, bias=False),
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 4, self.hidden_dim, kernel_size=1, bias=False),
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.SyncBatchNorm(self.hidden_dim),
                nn.SyncBatchNorm(self.hidden_dim),
                nn.SyncBatchNorm(self.hidden_dim),
            ]
        )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.image_mean.to(dtype=images.dtype)) / self.image_std.to(dtype=images.dtype)

    def _tokens_to_map(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"Expected DINO tokens with 3 or 4 dims, got {tuple(tokens.shape)}")
        grid_h, grid_w = grid_hw
        expected = grid_h * grid_w
        if tokens.shape[1] != expected:
            raise ValueError(f"DINO token count mismatch: got={tokens.shape[1]}, expected={expected}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w).contiguous()

    @staticmethod
    def _layer_tokens(layer_output: Any) -> torch.Tensor:
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        if isinstance(layer_output, (tuple, list)) and layer_output and isinstance(layer_output[0], torch.Tensor):
            return layer_output[0]
        raise TypeError(f"Unsupported DINO intermediate layer output: {type(layer_output)!r}")

    def forward_with_high_resolution(
        self,
        images: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Return normal DEIM features plus the shared RGB stride-4 map."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected images with shape [B, 3, H, W], got "
                f"{tuple(images.shape)}"
            )
        if not self.use_sta:
            raise RuntimeError(
                "High-resolution STA features require use_sta=true"
            )
        images = self._normalize(images)
        semantic_features = self._semantic_features(images)
        high_resolution, detail_features = (
            self.sta.forward_with_high_resolution(images)
        )
        fused_features = [
            torch.cat((semantic, detail), dim=1)
            for semantic, detail in zip(
                semantic_features,
                detail_features,
                strict=True,
            )
        ]
        outputs = tuple(
            norm(conv(feature))
            for conv, norm, feature in zip(
                self.convs,
                self.norms,
                fused_features,
                strict=True,
            )
        )
        return outputs, high_resolution

    def _semantic_features_with_final_stride16(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return the STA semantic pyramid and the final DINO stride-16 map.

        The three STA levels are built from blocks 3, 7, and 11 and are then
        rescaled to strides 8, 16, and 32.  The Co-DINO HEAD must instead see
        the unscaled block-11 map, matching the standalone HEAD backbone's
        ``x_norm_patchtokens`` contract.  Both outputs come from this one DINO
        invocation.
        """
        grid_h = int(images.shape[-2]) // self.patch_size[0]
        grid_w = int(images.shape[-1]) // self.patch_size[1]
        raw_dino = self.dino.backbone
        if hasattr(raw_dino, "get_intermediate_layers") and self.interaction_indexes:
            if self.dino.freeze and not any(parameter.requires_grad for parameter in raw_dino.parameters()):
                with torch.no_grad():
                    layers = raw_dino.get_intermediate_layers(
                        images,
                        n=self.interaction_indexes,
                        return_class_token=True,
                    )
            else:
                layers = raw_dino.get_intermediate_layers(
                    images,
                    n=self.interaction_indexes,
                    return_class_token=True,
                )
            tokens = [self._layer_tokens(layer) for layer in layers]
            if len(tokens) == 1:
                tokens = [tokens[0], tokens[0], tokens[0]]
            base_features = [self._tokens_to_map(token, (grid_h, grid_w)) for token in tokens]
        else:
            base = self.dino(images)
            base_features = [base, base, base]

        if len(base_features) != 3:
            raise ValueError(f"DEIMv2 STA expects exactly 3 DINO feature levels, got {len(base_features)}")
        final_stride16 = base_features[-1]
        num_scales = len(base_features) - 2
        sem_feats: list[torch.Tensor] = []
        for index, feature in enumerate(base_features):
            scale = 2 ** (num_scales - index)
            target_h = max(1, int(grid_h * scale))
            target_w = max(1, int(grid_w * scale))
            sem_feats.append(F.interpolate(feature, size=(target_h, target_w), mode="bilinear", align_corners=False))
        return sem_feats, final_stride16

    def _semantic_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        semantic_features, _final_stride16 = (
            self._semantic_features_with_final_stride16(images)
        )
        return semantic_features

    def _forward_feature_bundle(
        self,
        images: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}")
        images = self._normalize(images)
        sem_feats, final_stride16 = (
            self._semantic_features_with_final_stride16(images)
        )
        if self.use_sta:
            detail_feats = self.sta(images)
            fused_feats = [torch.cat([sem_feat, detail_feat], dim=1) for sem_feat, detail_feat in zip(sem_feats, detail_feats)]
        else:
            fused_feats = sem_feats
        outputs = tuple(
            norm(conv(feat))
            for conv, norm, feat in zip(self.convs, self.norms, fused_feats)
        )
        return outputs, tuple(sem_feats), final_stride16

    def forward_with_semantic(
        self,
        images: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the DEIM features and the shared raw DINO feature pyramid.

        The second return value lets an optional detector head consume the
        same single DINO forward pass.  ``forward`` deliberately preserves the
        historical one-value contract for every existing configuration.
        """
        outputs, semantic_features, _final_stride16 = (
            self._forward_feature_bundle(images)
        )
        return outputs, semantic_features

    def forward_with_semantic_and_final_stride16(
        self,
        images: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Return DEIM, STA semantic, and block-11 HEAD features in one pass."""

        return self._forward_feature_bundle(images)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, _semantic = self.forward_with_semantic(images)
        return outputs


class OfficialTinySTABackbone(nn.Module):
    """Official DEIMv2 ViT-Tiny/TinyPlus STA backbone.

    The public DEIMv2 DINOv3-S/M checkpoints use this lightweight ViT wrapper
    with state-dict keys under ``backbone.dinov3._model``. Keeping the same
    module names lets us warm-start from those checkpoints while still using the
    local detector/loss integration.
    """

    def __init__(
        self,
        *,
        backbone_cfg: Mapping[str, Any],
        hidden_dim: int,
        interaction_indexes: tuple[int, ...] = (3, 7, 11),
        conv_inplane: int = 16,
        use_sta: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(backbone_cfg.get("embed_dim", 256))
        self.patch_size = int(backbone_cfg.get("patch_size", 16))
        self.interaction_indexes = tuple(int(index) for index in interaction_indexes)
        self.freeze = bool(backbone_cfg.get("freeze", True))
        self.use_sta = bool(use_sta)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.dinov3 = VisionTransformer(
            embed_dim=self.embed_dim,
            num_heads=int(backbone_cfg.get("num_heads", 4)),
            patch_size=self.patch_size,
            return_layers=list(self.interaction_indexes),
        )
        if self.freeze:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        if self.use_sta:
            self.sta = SpatialPriorModuleV2(inplanes=int(conv_inplane))
        else:
            conv_inplane = 0
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 2, self.hidden_dim, kernel_size=1, bias=False),
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 4, self.hidden_dim, kernel_size=1, bias=False),
                nn.Conv2d(self.embed_dim + int(conv_inplane) * 4, self.hidden_dim, kernel_size=1, bias=False),
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.SyncBatchNorm(self.hidden_dim),
                nn.SyncBatchNorm(self.hidden_dim),
                nn.SyncBatchNorm(self.hidden_dim),
            ]
        )

    def train(self, mode: bool = True) -> "OfficialTinySTABackbone":
        super().train(mode)
        if self.freeze and not any(parameter.requires_grad for parameter in self.dinov3.parameters()):
            self.dinov3.eval()
        return self

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.image_mean.to(dtype=images.dtype)) / self.image_std.to(dtype=images.dtype)

    def _semantic_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        batch_size = int(images.shape[0])
        grid_h = int(images.shape[-2]) // self.patch_size
        grid_w = int(images.shape[-1]) // self.patch_size
        if self.freeze and not any(parameter.requires_grad for parameter in self.dinov3.parameters()):
            with torch.no_grad():
                all_layers = self.dinov3(images)
        else:
            all_layers = self.dinov3(images)
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        if len(all_layers) != 3:
            raise ValueError(f"DEIMv2 STA expects exactly 3 ViT feature levels, got {len(all_layers)}")

        num_scales = len(all_layers) - 2
        sem_feats: list[torch.Tensor] = []
        for index, layer_output in enumerate(all_layers):
            tokens = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output
            feature = tokens.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w).contiguous()
            scale = 2 ** (num_scales - index)
            target_h = max(1, int(grid_h * scale))
            target_w = max(1, int(grid_w * scale))
            sem_feats.append(F.interpolate(feature, size=(target_h, target_w), mode="bilinear", align_corners=False))
        return sem_feats

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}")
        images = self._normalize(images)
        sem_feats = self._semantic_features(images)
        if self.use_sta:
            detail_feats = self.sta(images)
            fused_feats = [torch.cat([sem_feat, detail_feat], dim=1) for sem_feat, detail_feat in zip(sem_feats, detail_feats)]
        else:
            fused_feats = sem_feats
        return tuple(norm(conv(feat)) for conv, norm, feat in zip(self.convs, self.norms, fused_feats))


class DINOv3DEIMv2Detector(nn.Module):
    """Official DEIMv2-style detector integrated with local DINOv3 data paths."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        encoder: HybridEncoder,
        decoder: DEIMTransformer,
        criterion: DEIMCriterion,
        postprocessor: PostProcessor,
        category_ids: tuple[int, ...],
        score_threshold: float = 0.0,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.category_ids = tuple(int(value) for value in category_ids)
        self.category_to_label = {category_id: index for index, category_id in enumerate(self.category_ids)}
        self.current_epoch = 0
        self.score_threshold = float(score_threshold)
        self.detections_per_image = int(detections_per_image)
        self.pretrained_load_info: dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _targets_to_official(
        self,
        targets: list[dict[str, Any]],
        image_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, torch.Tensor]]:
        image_h, image_w = int(image_size[0]), int(image_size[1])
        normalizer = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=dtype).clamp_min(1.0)
        converted: list[dict[str, torch.Tensor]] = []
        for target in targets:
            boxes = torch.as_tensor(target.get("boxes"), dtype=dtype, device=device).reshape(-1, 4)
            raw_labels = torch.as_tensor(target.get("labels"), dtype=torch.long, device=device).reshape(-1)
            if raw_labels.numel() != boxes.shape[0]:
                raw_labels = torch.ones((boxes.shape[0],), dtype=torch.long, device=device)
            if boxes.numel() == 0:
                converted.append(
                    {
                        "boxes": boxes.reshape(0, 4),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )
                continue
            mapped_labels = []
            keep_indices = []
            for index, label in enumerate(raw_labels.tolist()):
                mapped = self.category_to_label.get(int(label))
                if mapped is None:
                    continue
                keep_indices.append(index)
                mapped_labels.append(mapped)
            if not keep_indices:
                converted.append(
                    {
                        "boxes": boxes.new_zeros((0, 4)),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )
                continue
            keep = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
            boxes = boxes[keep]
            labels = torch.as_tensor(mapped_labels, dtype=torch.long, device=device)
            boxes = _xyxy_to_cxcywh(boxes) / normalizer
            boxes = boxes.clamp(0.0, 1.0)
            valid = (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
            converted.append({"boxes": boxes[valid], "labels": labels[valid]})
        return converted

    @staticmethod
    def _loss_total(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = [value for value in losses.values() if isinstance(value, torch.Tensor)]
        if not values:
            raise RuntimeError("Official DEIMv2 criterion returned no tensor losses.")
        total = values[0].sum() * 0.0
        for value in values:
            total = total + value
        return total

    def _format_losses(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        formatted: dict[str, torch.Tensor] = {"loss_total": self._loss_total(losses)}
        formatted.update(losses)
        for group_name, needle in {
            "loss_aux": "_aux_",
            "loss_dn": "_dn_",
            "loss_enc": "_enc_",
            "loss_pre": "_pre",
        }.items():
            group_values = [value for key, value in losses.items() if needle in key and isinstance(value, torch.Tensor)]
            if group_values:
                formatted[group_name] = sum(group_values[1:], group_values[0])
        if "loss_mal" in losses:
            formatted["loss_cls"] = losses["loss_mal"]
        elif "loss_focal" in losses:
            formatted["loss_cls"] = losses["loss_focal"]
        if "loss_bbox" in losses:
            formatted["loss_bbox_l1"] = losses["loss_bbox"]
        return formatted

    @staticmethod
    def _resolve_padding_mask(
        images: torch.Tensor,
        targets: list[dict[str, Any]] | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if padding_mask is None and targets:
            values = [target.get("padding_mask") for target in targets]
            if all(isinstance(value, torch.Tensor) for value in values):
                padding_mask = torch.stack(
                    [value.to(dtype=torch.bool) for value in values], dim=0
                )
        if padding_mask is None:
            return None
        padding_mask = torch.as_tensor(
            padding_mask, dtype=torch.bool, device=images.device
        )
        expected = (images.shape[0], images.shape[-2], images.shape[-1])
        if tuple(padding_mask.shape) != tuple(expected):
            raise ValueError(
                f"padding_mask must have shape {tuple(expected)}, got {tuple(padding_mask.shape)}"
            )
        return padding_mask

    @staticmethod
    def _feature_padding_masks(
        padding_mask: torch.Tensor | None,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> list[torch.Tensor] | None:
        if padding_mask is None:
            return None
        return [
            F.adaptive_avg_pool2d(
                padding_mask[:, None].to(dtype=torch.float32),
                output_size=feature.shape[-2:],
            )[:, 0]
            >= (1.0 - 1.0e-6)
            for feature in features
        ]

    def forward_raw(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        official_targets = None
        if targets is not None:
            official_targets = self._targets_to_official(
                targets,
                (int(images.shape[-2]), int(images.shape[-1])),
                device=images.device,
                dtype=images.dtype,
            )
        padding_mask = self._resolve_padding_mask(images, targets, padding_mask)
        features = self.backbone(images)
        feature_padding_masks = self._feature_padding_masks(padding_mask, features)
        encoded = self.encoder(
            list(features), feature_padding_masks=feature_padding_masks
        )
        return self.decoder(
            encoded,
            official_targets,
            feature_padding_masks=feature_padding_masks,
        )

    def forward_raw_with_features(
        self,
        images: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run inference once and expose the final raw DINO feature map.

        This opt-in path is intentionally separate from ``forward_raw`` and
        ``forward`` so training, checkpoints, and the historical inference
        contract remain unchanged.  The returned appearance map is the
        unscaled final DINO block map (stride 16 for the deployed ViT-S/16).
        """

        feature_forward = getattr(
            self.backbone,
            "forward_with_semantic_and_final_stride16",
            None,
        )
        if not callable(feature_forward):
            raise RuntimeError(
                "This DEIM backbone does not expose the final DINO feature "
                "map required by tracking"
            )
        padding_mask = self._resolve_padding_mask(images, None, padding_mask)
        features, _semantic_features, appearance_features = feature_forward(
            images
        )
        feature_padding_masks = self._feature_padding_masks(
            padding_mask,
            features,
        )
        encoded = self.encoder(
            list(features),
            feature_padding_masks=feature_padding_masks,
        )
        outputs = self.decoder(
            encoded,
            None,
            feature_padding_masks=feature_padding_masks,
        )
        return outputs, appearance_features

    def forward_with_features(
        self,
        images: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Return normal postprocessed detections plus DINO appearance map."""

        outputs, appearance_features = self.forward_raw_with_features(
            images,
            padding_mask=padding_mask,
        )
        image_size = (int(images.shape[-2]), int(images.shape[-1]))
        orig_sizes = torch.tensor(
            [[images.shape[-1], images.shape[-2]] for _ in range(images.shape[0])],
            dtype=images.dtype,
            device=images.device,
        )
        results = self.postprocessor(outputs, orig_sizes)
        return (
            [self._clip_and_filter(result, image_size) for result in results],
            appearance_features,
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        outputs = self.forward_raw(
            images,
            targets if self.training else None,
            padding_mask=padding_mask,
        )
        image_size = (int(images.shape[-2]), int(images.shape[-1]))
        if targets is not None:
            official_targets = self._targets_to_official(targets, image_size, device=images.device, dtype=images.dtype)
            losses = self.criterion(outputs, official_targets, epoch=self.current_epoch)
            return self._format_losses(losses)
        orig_sizes = torch.tensor(
            [[images.shape[-1], images.shape[-2]] for _ in range(images.shape[0])],
            dtype=images.dtype,
            device=images.device,
        )
        results = self.postprocessor(outputs, orig_sizes)
        return [self._clip_and_filter(result, image_size) for result in results]

    def _clip_and_filter(self, result: dict[str, torch.Tensor], image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        image_h, image_w = image_size
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"].long()
        if boxes.numel() > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_h))
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (scores >= self.score_threshold)
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]
        if self.detections_per_image > 0 and scores.numel() > self.detections_per_image:
            scores, top_indices = torch.topk(scores, self.detections_per_image)
            boxes = boxes[top_indices]
            labels = labels[top_indices]
        return {"boxes": boxes, "scores": scores, "labels": labels}


def _is_official_tiny_backbone(backbone_cfg: Mapping[str, Any]) -> bool:
    name = str(backbone_cfg.get("name", "")).lower()
    backbone_type = str(backbone_cfg.get("type", "")).lower()
    return name in {"vit_tiny", "vit_tinyplus"} or backbone_type in {"official_vit_tiny", "official_vit_tinyplus", "vit_tiny", "vit_tinyplus"}


def _checkpoint_state_dict(path: str | Path, *, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Loading an official DEIMv2 .safetensors checkpoint requires safetensors. "
                "Install requirements.txt before model construction."
            ) from error
        checkpoint: Any = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, Mapping):
        search_keys = ("ema", "model", "state_dict", "module") if prefer_ema else ("model", "state_dict", "ema", "module")
        for key in search_keys:
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if prefer_ema and isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("module"), Mapping):
        checkpoint = checkpoint["module"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized = str(key)
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        normalized = normalized.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        state_dict[normalized] = value
    return state_dict


def _target_checkpoint_keys(source_key: str) -> tuple[str, ...]:
    """Return local key aliases for official DEIMv2 checkpoint modules.

    Upstream DEIMv2 stores a full DINOv3 backbone directly as
    ``backbone.dinov3``.  The reusable local wrapper stores the same module as
    ``backbone.dino.backbone``.  Tiny/TinyPlus checkpoints already use the
    direct prefix, so the original key always remains the first candidate.
    """

    candidates = [source_key]
    official_prefix = "backbone.dinov3."
    local_prefix = "backbone.dino.backbone."
    if source_key.startswith(official_prefix):
        candidates.append(local_prefix + source_key[len(official_prefix) :])
    elif source_key.startswith(local_prefix):
        candidates.append(official_prefix + source_key[len(local_prefix) :])
    return tuple(candidates)


_LAYER_EXPANSION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "encoder": (
        re.compile(r"^(encoder\.encoder\.\d+\.layers\.)(\d+)(\..+)$"),
    ),
    "decoder": (
        re.compile(r"^(decoder\.decoder\.layers\.)(\d+)(\..+)$"),
        re.compile(r"^(decoder\.decoder\.lqe_layers\.)(\d+)(\..+)$"),
        re.compile(r"^(decoder\.dec_bbox_head\.)(\d+)(\..+)$"),
    ),
}


def _expand_pretrained_layers(
    loadable: dict[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    expansion_cfg: Mapping[str, Any],
) -> dict[str, str]:
    """Initialize added encoder/decoder layers from the last pretrained layer.

    The public DEIMv2-L checkpoint has one Hybrid Encoder transformer layer and
    four decoder layers.  Depth ablations must not silently leave added layers
    at unrelated random initialization, so an explicitly configured
    ``repeat_last`` policy copies the deepest compatible pretrained layer into
    every added target layer.  Classification heads are intentionally excluded
    because custom datasets reinitialize them for their own class inventory.
    """

    source_keys = tuple(loadable)
    expanded: dict[str, str] = {}
    for family, patterns in _LAYER_EXPANSION_PATTERNS.items():
        policy = str(expansion_cfg.get(family, "none")).strip().lower()
        if policy in {"", "none", "false"}:
            continue
        if policy != "repeat_last":
            raise ValueError(
                f"Unsupported init_checkpoint.layer_expansion.{family}: {policy!r}"
            )
        for target_key, target_value in model_state.items():
            if target_key in loadable:
                continue
            for pattern in patterns:
                target_match = pattern.match(target_key)
                if target_match is None:
                    continue
                prefix, target_index_text, suffix = target_match.groups()
                target_index = int(target_index_text)
                candidates: list[tuple[int, str]] = []
                for source_key in source_keys:
                    source_match = pattern.match(source_key)
                    if source_match is None:
                        continue
                    source_prefix, source_index_text, source_suffix = source_match.groups()
                    if source_prefix == prefix and source_suffix == suffix:
                        source_index = int(source_index_text)
                        if source_index < target_index:
                            candidates.append((source_index, source_key))
                if not candidates:
                    break
                _source_index, source_key = max(candidates)
                source_value = loadable[source_key]
                if tuple(source_value.shape) != tuple(target_value.shape):
                    raise RuntimeError(
                        "Layer expansion shape mismatch: "
                        f"source={source_key}{tuple(source_value.shape)}, "
                        f"target={target_key}{tuple(target_value.shape)}"
                    )
                loadable[target_key] = source_value.clone()
                expanded[target_key] = source_key
                break
    return expanded


def _load_pretrained_checkpoint(model: DINOv3DEIMv2Detector, init_cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = init_cfg.get("path") or init_cfg.get("checkpoint") or init_cfg.get("weights")
    if not path:
        return {}
    skip_class_heads = bool(init_cfg.get("skip_class_heads", True))
    skip_prefixes = tuple(
        str(value)
        for value in init_cfg.get("skip_prefixes", ())
    )
    class_head_prefixes = (
        "decoder.denoising_class_embed.",
        "decoder.enc_score_head.",
        "decoder.dec_score_head.",
    )
    checkpoint = _checkpoint_state_dict(path, prefer_ema=bool(init_cfg.get("prefer_ema", False)))
    model_state = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped_shape: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    skipped_class: list[str] = []
    skipped_prefix: list[str] = []
    unexpected: list[str] = []
    remapped_keys = 0
    loaded_numel = 0
    for source_key, value in checkpoint.items():
        if skip_prefixes and source_key.startswith(skip_prefixes):
            skipped_prefix.append(source_key)
            continue
        if skip_class_heads and source_key.startswith(class_head_prefixes):
            skipped_class.append(source_key)
            continue
        key = next((candidate for candidate in _target_checkpoint_keys(source_key) if candidate in model_state), source_key)
        target = model_state.get(key)
        if target is None:
            unexpected.append(source_key)
            continue
        if tuple(value.shape) != tuple(target.shape):
            skipped_shape[key] = (tuple(value.shape), tuple(target.shape))
            continue
        loadable[key] = value
        loaded_numel += int(value.numel())
        remapped_keys += int(key != source_key)
    layer_expansion_cfg = init_cfg.get("layer_expansion", {}) or {}
    if not isinstance(layer_expansion_cfg, Mapping):
        raise TypeError("init_checkpoint.layer_expansion must be a mapping")
    expanded_layers = _expand_pretrained_layers(
        loadable,
        model_state,
        layer_expansion_cfg,
    )
    loaded_numel += sum(int(loadable[key].numel()) for key in expanded_layers)
    missing, model_unexpected = model.load_state_dict(loadable, strict=False)
    added_layer_initialization = {
        "applied": False,
        "added_layer_indices": [],
        "zeroed_bbox_output_layers": [],
        "copied_score_head_layers": [],
    }
    decoder = getattr(model, "decoder", None)
    initialize_added_layers = getattr(
        decoder, "apply_added_layer_initialization", None
    )
    if expanded_layers and callable(initialize_added_layers):
        added_layer_initialization = initialize_added_layers()
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    info: dict[str, Any] = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "loaded": len(loadable),
        "checkpoint_tensors": len(checkpoint),
        "missing": len(missing),
        "missing_keys": list(missing),
        "unexpected": len(unexpected) + len(model_unexpected),
        "unexpected_keys": [*unexpected, *model_unexpected],
        "skipped_shape": len(skipped_shape),
        "skipped_shape_keys": sorted(skipped_shape),
        "skipped_class_heads": len(skipped_class),
        "skipped_class_head_keys": skipped_class,
        "skipped_prefixes": list(skip_prefixes),
        "skipped_prefix": len(skipped_prefix),
        "skipped_prefix_keys": skipped_prefix,
        "remapped_keys": remapped_keys,
        "layer_expansion": {
            str(key): str(value) for key, value in layer_expansion_cfg.items()
        },
        "expanded_layer_keys": sorted(expanded_layers),
        "expanded_layer_sources": {
            key: expanded_layers[key] for key in sorted(expanded_layers)
        },
        "expanded_layers": len(expanded_layers),
        "added_layer_initialization": added_layer_initialization,
        "loaded_numel": loaded_numel,
        "model_state_numel": sum(int(value.numel()) for value in model_state.values()),
        "loaded_state_numel_fraction": loaded_numel / max(1, sum(int(value.numel()) for value in model_state.values())),
    }
    model.pretrained_load_info = info
    if bool(init_cfg.get("verbose", True)):
        print(f"pretrained_deimv2: {info}")
    return info


def build_dinov3_deimv2_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DINOv3DEIMv2Detector:
    if backbone_module is not None:
        raise ValueError("Official DINOv3 DEIMv2 path expects backbone configuration, not backbone_module injection.")
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    data_cfg = dict(cfg.get("data", {}))
    backbone_cfg = _cfg_section(cfg, model_cfg, "backbone")
    adapter_cfg = _cfg_section(cfg, model_cfg, "adapter")
    encoder_cfg = _cfg_section(cfg, model_cfg, "encoder")
    decoder_cfg = _cfg_section(cfg, model_cfg, "decoder")
    loss_cfg = dict(cfg.get("loss", model_cfg.get("loss", {})))
    inference_cfg = dict(cfg.get("inference", cfg.get("postprocess", model_cfg.get("postprocess", {}))))

    num_classes = int(model_cfg.get("num_classes", 2))
    category_ids = tuple(int(value) for value in data_cfg.get("category_ids", range(1, num_classes + 1)))
    if len(category_ids) != num_classes:
        category_ids = tuple(range(1, num_classes + 1))

    adapter_hidden_dim = int(adapter_cfg.get("hidden_dim", adapter_cfg.get("out_channels", 256)))
    backbone_cls = OfficialTinySTABackbone if _is_official_tiny_backbone(backbone_cfg) else DINOv3STABackbone
    backbone = backbone_cls(
        backbone_cfg=backbone_cfg,
        hidden_dim=adapter_hidden_dim,
        interaction_indexes=tuple(adapter_cfg.get("interaction_indexes", (3, 7, 11))),
        conv_inplane=int(adapter_cfg.get("conv_inplane", 16)),
        use_sta=bool(adapter_cfg.get("use_sta", True)),
    )

    encoder_hidden_dim = int(encoder_cfg.get("hidden_dim", adapter_hidden_dim))
    encoder = HybridEncoder(
        in_channels=list(encoder_cfg.get("in_channels", [adapter_hidden_dim, adapter_hidden_dim, adapter_hidden_dim])),
        feat_strides=list(encoder_cfg.get("feat_strides", [8, 16, 32])),
        hidden_dim=encoder_hidden_dim,
        nhead=int(encoder_cfg.get("nhead", encoder_cfg.get("num_heads", 8))),
        dim_feedforward=int(encoder_cfg.get("dim_feedforward", encoder_hidden_dim * 4)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
        enc_act=str(encoder_cfg.get("enc_act", "gelu")),
        use_encoder_idx=list(encoder_cfg.get("use_encoder_idx", [2])),
        num_encoder_layers=int(encoder_cfg.get("num_encoder_layers", encoder_cfg.get("num_layers", 1))),
        expansion=float(encoder_cfg.get("expansion", 1.0)),
        depth_mult=float(encoder_cfg.get("depth_mult", 1.0)),
        act=str(encoder_cfg.get("act", "silu")),
        eval_spatial_size=encoder_cfg.get("eval_spatial_size"),
        version=str(encoder_cfg.get("version", "deim")),
        csp_type=str(encoder_cfg.get("csp_type", "csp2")),
        fuse_op=str(encoder_cfg.get("fuse_op", "sum")),
    )

    decoder_hidden_dim = int(decoder_cfg.get("hidden_dim", encoder_hidden_dim))
    added_layer_cfg = dict(decoder_cfg.get("added_layer_initialization", {}) or {})
    decoder = DEIMTransformer(
        num_classes=num_classes,
        hidden_dim=decoder_hidden_dim,
        num_queries=int(decoder_cfg.get("num_queries", 300)),
        feat_channels=list(decoder_cfg.get("feat_channels", [encoder_hidden_dim, encoder_hidden_dim, encoder_hidden_dim])),
        feat_strides=list(decoder_cfg.get("feat_strides", [8, 16, 32])),
        num_levels=int(decoder_cfg.get("num_levels", 3)),
        num_points=decoder_cfg.get("num_points", [3, 6, 3]),
        nhead=int(decoder_cfg.get("nhead", decoder_cfg.get("num_heads", 8))),
        num_layers=int(decoder_cfg.get("num_layers", 6)),
        dim_feedforward=int(decoder_cfg.get("dim_feedforward", decoder_hidden_dim * 8)),
        dropout=float(decoder_cfg.get("dropout", 0.0)),
        activation=str(decoder_cfg.get("activation", "silu")),
        num_denoising=int(decoder_cfg.get("num_denoising", 100)),
        label_noise_ratio=float(decoder_cfg.get("label_noise_ratio", 0.5)),
        box_noise_scale=float(decoder_cfg.get("box_noise_scale", 1.0)),
        learn_query_content=bool(decoder_cfg.get("learn_query_content", False)),
        eval_spatial_size=decoder_cfg.get("eval_spatial_size"),
        eval_idx=int(decoder_cfg.get("eval_idx", -1)),
        aux_loss=bool(decoder_cfg.get("aux_loss", True)),
        cross_attn_method=str(decoder_cfg.get("cross_attn_method", "default")),
        query_select_method=str(decoder_cfg.get("query_select_method", "default")),
        reg_max=int(decoder_cfg.get("reg_max", loss_cfg.get("reg_max", 32))),
        reg_scale=float(decoder_cfg.get("reg_scale", 4.0)),
        layer_scale=float(decoder_cfg.get("layer_scale", 1.0)),
        mlp_act=str(decoder_cfg.get("mlp_act", "silu")),
        use_gateway=bool(decoder_cfg.get("use_gateway", True)),
        share_bbox_head=bool(decoder_cfg.get("share_bbox_head", False)),
        share_score_head=bool(decoder_cfg.get("share_score_head", False)),
        added_layer_start=added_layer_cfg.get("start_layer"),
        added_layer_residual_gate=float(
            added_layer_cfg.get("residual_gate", 0.1)
        ),
        zero_added_bbox_output=bool(
            added_layer_cfg.get("zero_bbox_output", False)
        ),
        copy_added_score_head=bool(
            added_layer_cfg.get("copy_score_head", False)
        ),
        self_attention_fp32=bool(decoder_cfg.get("self_attention_fp32", False)),
        global_context_attention=bool(
            decoder_cfg.get("global_context_attention", False)
        ),
        global_context_initial_gate=float(
            decoder_cfg.get("global_context_initial_gate", 0.0)
        ),
    )

    weight_dict = dict(loss_cfg.get("weight_dict", {"loss_mal": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_fgl": 0.15, "loss_ddf": 1.5}))
    matcher_cfg = dict(loss_cfg.get("matcher", {}))
    matcher = HungarianMatcher(
        weight_dict=dict(matcher_cfg.get("weight_dict", {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0})),
        use_focal_loss=bool(loss_cfg.get("use_focal_loss", True)),
        alpha=float(matcher_cfg.get("alpha", 0.25)),
        gamma=float(matcher_cfg.get("gamma", 2.0)),
        change_matcher=bool(matcher_cfg.get("change_matcher", True)),
        iou_order_alpha=float(matcher_cfg.get("iou_order_alpha", 4.0)),
        matcher_change_epoch=int(matcher_cfg.get("matcher_change_epoch", 100)),
    )
    criterion = DEIMCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=list(loss_cfg.get("losses", ["mal", "boxes", "local"])),
        alpha=float(loss_cfg.get("alpha", 0.75)),
        gamma=float(loss_cfg.get("gamma", 1.5)),
        num_classes=num_classes,
        reg_max=int(loss_cfg.get("reg_max", decoder_cfg.get("reg_max", 32))),
        boxes_weight_format=loss_cfg.get("boxes_weight_format"),
        share_matched_indices=bool(loss_cfg.get("share_matched_indices", False)),
        mal_alpha=loss_cfg.get("mal_alpha"),
        use_uni_set=bool(loss_cfg.get("use_uni_set", True)),
        aux_layer_weights=loss_cfg.get("aux_layer_weights"),
        dn_layer_weights=loss_cfg.get("dn_layer_weights"),
    )
    postprocessor = PostProcessor(
        num_classes=num_classes,
        use_focal_loss=bool(loss_cfg.get("use_focal_loss", True)),
        num_top_queries=int(inference_cfg.get("topk", inference_cfg.get("num_top_queries", 300))),
        remap_mscoco_category=bool(inference_cfg.get("remap_mscoco_category", False)),
    )
    detector = DINOv3DEIMv2Detector(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        postprocessor=postprocessor,
        category_ids=category_ids,
        score_threshold=float(inference_cfg.get("score_thr", inference_cfg.get("score_threshold", 0.0))),
        detections_per_image=int(inference_cfg.get("detections_per_image", 100)),
    )
    init_cfg = dict(model_cfg.get("init_checkpoint", cfg.get("init_checkpoint", {})) or {})
    legacy_init = model_cfg.get("pretrained_checkpoint") or cfg.get("pretrained_checkpoint")
    if legacy_init and not init_cfg:
        init_cfg = {"path": legacy_init}
    if init_cfg:
        _load_pretrained_checkpoint(detector, init_cfg)
    return detector
