from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .heads import sine_position_encoding_2d


@dataclass(frozen=True)
class MessengerRecoveryConfig:
    enabled: bool = False
    layers: int = 1
    heads: int = 6
    feedforward_ratio: float = 4.0
    dropout: float = 0.0
    positional_encoding: str = "sine_2d"
    attention_scope: str = "full_face_crop"
    self_patch_mask: bool = True
    gate: str = "soft_scalar"
    initial_gate: float = 0.10
    zero_init_projection: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "MessengerRecoveryConfig":
        raw = dict(value or {})
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            layers=int(raw.get("layers", 1)),
            heads=int(raw.get("heads", 6)),
            feedforward_ratio=float(raw.get("feedforward_ratio", 4.0)),
            dropout=float(raw.get("dropout", 0.0)),
            positional_encoding=str(raw.get("positional_encoding", "sine_2d")),
            attention_scope=str(raw.get("attention_scope", "full_face_crop")),
            self_patch_mask=bool(raw.get("self_patch_mask", True)),
            gate=str(raw.get("gate", "soft_scalar")),
            initial_gate=float(raw.get("initial_gate", 0.10)),
            zero_init_projection=bool(raw.get("zero_init_projection", True)),
        )
        config.validate()
        return config

    def validate(self, *, embed_dim: int | None = None) -> None:
        if self.layers != 1:
            raise ValueError("Messenger v1 requires exactly one recovery layer")
        if self.heads != 6:
            raise ValueError("Messenger v1 requires messenger_recovery.heads=6")
        if embed_dim is not None and int(embed_dim) % self.heads != 0:
            raise ValueError(
                "DINO embed_dim must be divisible by messenger_recovery.heads"
            )
        if self.feedforward_ratio <= 0.0:
            raise ValueError("messenger_recovery.feedforward_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("messenger_recovery.dropout must be in [0, 1)")
        if self.positional_encoding != "sine_2d":
            raise ValueError("Messenger v1 requires positional_encoding=sine_2d")
        if self.attention_scope != "full_face_crop":
            raise ValueError("Messenger v1 requires attention_scope=full_face_crop")
        if not self.self_patch_mask:
            raise ValueError("Messenger v1 requires self_patch_mask=true")
        if self.gate != "soft_scalar":
            raise ValueError("Messenger v1 requires gate=soft_scalar")
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("messenger_recovery.initial_gate must be in (0, 1)")
        if not self.zero_init_projection:
            raise ValueError("Messenger v1 requires zero_init_projection=true")


class MessengerRecoveryBlock(nn.Module):
    """Predict each DINO patch from every *other* patch in the same face crop."""

    def __init__(
        self,
        embed_dim: int,
        config: Mapping[str, Any] | MessengerRecoveryConfig,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.config = (
            config
            if isinstance(config, MessengerRecoveryConfig)
            else MessengerRecoveryConfig.from_mapping(config)
        )
        self.config.validate(embed_dim=self.embed_dim)
        hidden = max(1, int(round(self.embed_dim * self.config.feedforward_ratio)))
        self.messenger_token = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.memory_norm = nn.LayerNorm(self.embed_dim)
        self.attention = nn.MultiheadAttention(
            self.embed_dim,
            self.config.heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, self.embed_dim),
        )
        gate_hidden = max(16, self.embed_dim // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.correction_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.messenger_token, std=0.02)
        final_gate = self.gate_mlp[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(
            final_gate.bias,
            math.log(self.config.initial_gate / (1.0 - self.config.initial_gate)),
        )
        nn.init.zeros_(self.correction_projection.weight)
        nn.init.zeros_(self.correction_projection.bias)

    @staticmethod
    def self_patch_attention_mask(
        token_count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if token_count < 2:
            raise ValueError("Messenger recovery requires at least two patch tokens")
        return torch.eye(token_count, dtype=torch.bool, device=device)

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if feature_map.ndim != 4:
            raise ValueError(
                "Messenger feature_map must be BxCxHxW, got "
                f"{tuple(feature_map.shape)}"
            )
        batch, channels, height, width = feature_map.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Messenger channel mismatch: {channels} != {self.embed_dim}"
            )
        patches = feature_map.flatten(2).transpose(1, 2).contiguous()
        position = sine_position_encoding_2d(
            height,
            width,
            self.embed_dim,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        queries = self.messenger_token.to(dtype=feature_map.dtype).expand(
            batch, height * width, -1
        ) + position
        normalized_patches = self.memory_norm(patches)
        predicted, attention = self.attention(
            self.query_norm(queries),
            normalized_patches,
            normalized_patches,
            attn_mask=self.self_patch_attention_mask(
                height * width,
                device=feature_map.device,
            ),
            need_weights=return_diagnostics,
            average_attn_weights=False,
        )
        predicted = predicted + self.ffn(self.output_norm(predicted))
        gate_input = torch.cat(
            [patches, predicted, (patches - predicted).abs()],
            dim=-1,
        )
        alpha = self.gate_mlp(gate_input).sigmoid()
        recovered = patches + alpha * self.correction_projection(predicted - patches)
        recovered_map = recovered.transpose(1, 2).reshape(
            batch, channels, height, width
        ).contiguous()
        diagnostics = {
            "original_tokens": patches,
            "messenger_tokens": predicted,
            "recovered_tokens": recovered,
            "gate_alpha": alpha.squeeze(-1),
        }
        if return_diagnostics:
            if attention is None:
                raise RuntimeError("Messenger attention diagnostics were requested")
            diagnostics["attention"] = attention
        return recovered_map, diagnostics


def patch_occlusion_ratios(
    pixel_mask: torch.Tensor,
    *,
    patch_size: tuple[int, int] | int,
) -> torch.Tensor:
    """Return the exact masked-pixel fraction for every non-overlapping patch."""

    if pixel_mask.ndim == 3:
        pixel_mask = pixel_mask[:, None]
    if pixel_mask.ndim != 4 or pixel_mask.shape[1] != 1:
        raise ValueError(
            "pixel_mask must have shape BxHxW or Bx1xHxW, got "
            f"{tuple(pixel_mask.shape)}"
        )
    if isinstance(patch_size, int):
        kernel = (int(patch_size), int(patch_size))
    else:
        kernel = (int(patch_size[0]), int(patch_size[1]))
    height, width = pixel_mask.shape[-2:]
    if height % kernel[0] != 0 or width % kernel[1] != 0:
        raise ValueError("pixel mask size must be divisible by DINO patch size")
    ratios = F.avg_pool2d(
        pixel_mask.to(dtype=torch.float32),
        kernel_size=kernel,
        stride=kernel,
    )
    return ratios.flatten(1)


__all__ = [
    "MessengerRecoveryBlock",
    "MessengerRecoveryConfig",
    "patch_occlusion_ratios",
]
