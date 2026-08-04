from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dino_backbone import DinoBackbone, build_dino_backbone

from .fpn import SimpleFeaturePyramid
from .simcc import RTMCCHead, SimCCLoss


def _layer_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
            tensor = value.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor
    raise TypeError(f"Unsupported DINOv3 intermediate output: {type(value)!r}")


class DinoV3PatchEncoder(nn.Module):
    """Strict pretrained DINOv3 patch encoder with last-block feature fusion."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backbone_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(config)
        weights = cfg.get("weights")
        if bool(cfg.get("require_pretrained", True)) and backbone_module is None:
            if not weights or not Path(str(weights)).is_file():
                raise FileNotFoundError(
                    "DINOv3 pretrained weights are required. Run "
                    "`python keypoint_comparison2/prepare_weights.py --dinov3-weights <downloaded-file>` "
                    "after Meta has approved DINOv3 access."
                )
        self.dino: DinoBackbone = build_dino_backbone(cfg, backbone=backbone_module)
        self.embed_dim = int(cfg.get("embed_dim", self.dino.embed_dim))
        self.patch_size = int(cfg.get("patch_size", self.dino.patch_size[0]))
        self.intermediate_layers = tuple(int(value) for value in cfg.get("intermediate_layers", (2, 5, 8, 11)))
        self.layer_weights = nn.Parameter(torch.ones(len(self.intermediate_layers)))
        self.pretrained_report = {
            "checkpoint": str(weights) if weights else None,
            "strict": bool(cfg.get("strict_weights", False)),
            "architecture": str(cfg.get("name", "dinov3_vits16")),
            "intermediate_layers": list(self.intermediate_layers),
        }
        if bool(cfg.get("activation_checkpointing", False)):
            self._enable_activation_checkpointing()
        self.register_buffer(
            "mean",
            torch.tensor(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(cfg.get("std", [0.229, 0.224, 0.225]), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _enable_activation_checkpointing(self) -> None:
        raw = self.dino.backbone
        blocks = getattr(raw, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError("DINOv3 activation checkpointing requires a ModuleList named `blocks`")
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError as error:
            raise RuntimeError("This PyTorch build does not provide activation checkpoint wrappers") from error
        for index, block in enumerate(blocks):
            blocks[index] = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)

    def _to_map(self, tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(f"DINOv3 tokens must have 3 or 4 dimensions, got {tuple(tensor.shape)}")
        expected = height * width
        if tensor.shape[1] != expected:
            # Some entrypoints return class/register tokens together with patches.
            if tensor.shape[1] > expected:
                tensor = tensor[:, -expected:, :]
            else:
                raise ValueError(f"DINOv3 token count {tensor.shape[1]} does not match grid {height}x{width}")
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], height, width).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("DINOv3 crop dimensions must be divisible by patch_size")
        normalized = self._normalize(images)
        raw = self.dino.backbone
        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        if hasattr(raw, "get_intermediate_layers") and self.intermediate_layers:
            values = raw.get_intermediate_layers(
                normalized,
                n=self.intermediate_layers,
                return_class_token=False,
            )
            maps = [self._to_map(_layer_tensor(value), grid_h, grid_w) for value in values]
            if len(maps) != len(self.intermediate_layers):
                raise RuntimeError(
                    f"DINOv3 returned {len(maps)} intermediate maps; expected {len(self.intermediate_layers)}"
                )
            weights = self.layer_weights.softmax(0)
            return sum(weight * feature for weight, feature in zip(weights, maps))
        return self.dino(normalized)


class DinoV3FPNRTMCC(nn.Module):
    """DINOv3 patch features -> Simple FPN -> RTMCC -> SimCCLabel."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backbone_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(config)
        self.backbone = DinoV3PatchEncoder(cfg["backbone"], backbone_module=backbone_module)
        self.pretrained_report = self.backbone.pretrained_report
        fpn_cfg = dict(cfg.get("fpn", {}))
        head_cfg = dict(cfg.get("head", {}))
        out_channels = int(fpn_cfg.get("out_channels", 256))
        self.fpn = SimpleFeaturePyramid(self.backbone.embed_dim, out_channels)
        input_size = int(cfg.get("input_size", 320))
        self.head = RTMCCHead(
            out_channels,
            num_keypoints=5,
            input_size=(input_size, input_size),
            featuremap_size=tuple(head_cfg.get("featuremap_size", (8, 8))),
            hidden_dimension=int(head_cfg.get("hidden_dimension", 256)),
            split_ratio=float(head_cfg.get("simcc_split_ratio", 2.0)),
            sigma=tuple(head_cfg.get("sigma", (5.66, 5.66))),
            final_kernel_size=int(head_cfg.get("final_kernel_size", 7)),
            gau_attention_dimension=int(head_cfg.get("gau_attention_dimension", 128)),
            gau_expansion_factor=float(head_cfg.get("gau_expansion_factor", 2.0)),
            dropout=float(head_cfg.get("dropout", 0.0)),
        )
        self.loss_module = SimCCLoss(
            self.head.codec,
            beta=float(head_cfg.get("kl_beta", 10.0)),
            visibility_weight=float(head_cfg.get("visibility_loss_weight", 0.2)),
        )

    def extract_features(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return self.fpn(self.backbone(images))

    def forward(
        self,
        images: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        feature, levels = self.extract_features(images)
        output: dict[str, Any] = self.head(feature)
        output["fpn_features"] = levels
        if targets is not None:
            output.update(
                self.loss_module(
                    output,
                    targets["coordinates"],
                    targets["visible"],
                    targets.get("labeled"),
                )
            )
        return output


class RecoveryDecoderLayer(nn.Module):
    """Visibility-aware ORFormer-style token recovery layer."""

    def __init__(self, dimension: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.visible_cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        visible_memory: torch.Tensor,
        *,
        position: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.self_attention(query + position, query + position, query, need_weights=False)
        query = self.norm1(query + self.dropout(attended))
        recovered, _ = self.visible_cross_attention(
            query + position,
            visible_memory + position,
            visible_memory,
            need_weights=False,
        )
        query = self.norm2(query + self.dropout(recovered))
        return self.norm3(query + self.dropout(self.ffn(query)))


def _sine_position(height: int, width: int, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("ORFormer dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )
    scale = torch.arange(dimension // 4, device=device, dtype=dtype)
    scale = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), -scale / max(1, dimension // 4 - 1))
    position = torch.cat(
        (
            torch.sin(x.flatten().unsqueeze(-1) * scale),
            torch.cos(x.flatten().unsqueeze(-1) * scale),
            torch.sin(y.flatten().unsqueeze(-1) * scale),
            torch.cos(y.flatten().unsqueeze(-1) * scale),
        ),
        dim=-1,
    )
    return position.unsqueeze(0)


class OcclusionRecoveryFormer(nn.Module):
    """Predict patch occlusion and reconstruct hidden tokens from visible context."""

    def __init__(
        self,
        in_channels: int,
        *,
        dimension: int = 256,
        layers: int = 4,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection: nn.Module
        self.output_projection: nn.Module
        if in_channels == dimension:
            # Preserve pretrained DINO features before recovery has learned
            # anything. The residual recovery path then only changes tokens
            # predicted as occluded.
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()
        else:
            self.input_projection = nn.Conv2d(in_channels, dimension, kernel_size=1)
            self.output_projection = nn.Conv2d(dimension, in_channels, kernel_size=1)
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(in_channels, dimension, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dimension, 1, kernel_size=1),
        )
        nn.init.constant_(self.occlusion_head[-1].bias, -4.0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.layers = nn.ModuleList(
            RecoveryDecoderLayer(dimension, heads, feedforward, dropout) for _ in range(layers)
        )
        self.gate = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.Sigmoid())

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = feature.shape
        occlusion_logits = self.occlusion_head(feature)
        probability = occlusion_logits.sigmoid().flatten(2).transpose(1, 2)
        visible = self.input_projection(feature).flatten(2).transpose(1, 2)
        mask_token = self.mask_token.expand(batch, height * width, -1)
        query = visible * (1.0 - probability) + mask_token * probability
        visible_memory = visible * (1.0 - probability) + mask_token * (0.05 * probability)
        position = _sine_position(height, width, visible.shape[-1], visible.device, visible.dtype)
        for layer in self.layers:
            query = layer(query, visible_memory, position=position)
        gate = self.gate(torch.cat((visible, query), dim=-1))
        recovered_tokens = visible + probability * gate * (query - visible)
        recovered = recovered_tokens.transpose(1, 2).reshape(batch, -1, height, width)
        return self.output_projection(recovered), occlusion_logits


class DinoV3ORFormerRTMCC(DinoV3FPNRTMCC):
    """DINOv3 -> occlusion classifier -> ORFormer recovery -> FPN/RTMCC/SimCC."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backbone_module: nn.Module | None = None,
    ) -> None:
        super().__init__(config, backbone_module=backbone_module)
        recovery_cfg = dict(config.get("recovery", {}))
        self.recovery = OcclusionRecoveryFormer(
            self.backbone.embed_dim,
            dimension=int(recovery_cfg.get("dimension", 256)),
            layers=int(recovery_cfg.get("layers", 4)),
            heads=int(recovery_cfg.get("heads", 8)),
            feedforward=int(recovery_cfg.get("feedforward", 1024)),
            dropout=float(recovery_cfg.get("dropout", 0.0)),
        )
        self.occlusion_loss_weight = float(recovery_cfg.get("occlusion_loss_weight", 1.0))
        self.reconstruction_loss_weight = float(recovery_cfg.get("reconstruction_loss_weight", 2.0))
        self.sparsity_weight = float(recovery_cfg.get("sparsity_weight", 0.01))

    def forward(
        self,
        images: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None = None,
        *,
        clean_images: torch.Tensor | None = None,
        occlusion_masks: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        patch_feature = self.backbone(images)
        recovered, occlusion_logits = self.recovery(patch_feature)
        feature, levels = self.fpn(recovered)
        output: dict[str, Any] = self.head(feature)
        output.update(
            {
                "fpn_features": levels,
                "patch_features": patch_feature,
                "recovered_patch_features": recovered,
                "occlusion_logits": occlusion_logits,
            }
        )
        if targets is None:
            return output
        losses = self.loss_module(
            output,
            targets["coordinates"],
            targets["visible"],
            targets.get("labeled"),
        )
        if occlusion_masks is not None:
            mask = F.interpolate(
                occlusion_masks.to(dtype=occlusion_logits.dtype),
                size=occlusion_logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            loss_occlusion = F.binary_cross_entropy_with_logits(occlusion_logits, mask)
        else:
            mask = occlusion_logits.new_zeros(occlusion_logits.shape)
            loss_occlusion = occlusion_logits.sigmoid().mean() * self.sparsity_weight
        loss_reconstruction = recovered.sum() * 0.0
        if clean_images is not None and bool(mask.any()):
            with torch.no_grad():
                clean_feature = self.backbone(clean_images)
            patch_mask = mask.expand_as(recovered)
            loss_reconstruction = F.smooth_l1_loss(
                recovered * patch_mask,
                clean_feature * patch_mask,
                reduction="sum",
            ) / patch_mask.sum().clamp_min(1.0)
        losses["loss_occlusion"] = loss_occlusion
        losses["loss_reconstruction"] = loss_reconstruction
        losses["loss_total"] = (
            losses["loss_total"]
            + self.occlusion_loss_weight * loss_occlusion
            + self.reconstruction_loss_weight * loss_reconstruction
        )
        output.update(losses)
        return output
