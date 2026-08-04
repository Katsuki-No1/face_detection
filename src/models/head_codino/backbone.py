from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


EXPECTED_DINOV3_VITS16PLUS_PARAMETERS = 28_697_472


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"DINOv3 checkpoint does not contain a state dict: {path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = str(key)
        for prefix in ("module.", "backbone.model.", "backbone.backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        state[clean] = value
    return state


def _reject_non_plus_checkpoint(path: Path) -> None:
    state = _state_dict_from_checkpoint(path)
    has_standard_mlp = any(key.startswith("blocks.0.mlp.fc1.") for key in state)
    has_swiglu = "blocks.0.mlp.w1.weight" in state and "blocks.0.mlp.w2.weight" in state
    if has_standard_mlp and not has_swiglu:
        raise ValueError(
            f"{path} is a standard DINOv3 ViT-S/16 checkpoint, not ViT-S+/16 "
            "(standard MLP keys were found instead of SwiGLU w1/w2/w3 keys)."
        )


class DINOv3ViTS16PlusBackbone(nn.Module):
    """Official DINOv3 ViT-S+/16 final-patch-token wrapper."""

    patch_size = 16
    embed_dim = 384

    def __init__(
        self,
        *,
        source_root: str | Path,
        weights: str | Path | None = None,
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        source_root = Path(source_root).expanduser().resolve()
        if not (source_root / "dinov3" / "hub" / "backbones.py").exists():
            raise FileNotFoundError(f"Official DINOv3 source was not found under {source_root}")
        source_str = str(source_root)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)
        from dinov3.hub.backbones import dinov3_vits16plus

        weight_path = Path(weights).expanduser().resolve() if weights else None
        if weight_path is not None:
            if not weight_path.exists():
                raise FileNotFoundError(f"DINOv3 ViT-S+/16 weights were not found: {weight_path}")
            _reject_non_plus_checkpoint(weight_path)
        self.model = dinov3_vits16plus(
            pretrained=weight_path is not None,
            weights=str(weight_path) if weight_path is not None else "LVD1689M",
            check_hash=False,
        )
        self.freeze = bool(freeze)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        if self.unfreeze_last_n_blocks < 0 or self.unfreeze_last_n_blocks > len(self.model.blocks):
            raise ValueError(
                f"unfreeze_last_n_blocks must be between 0 and {len(self.model.blocks)}, "
                f"got {self.unfreeze_last_n_blocks}"
            )
        self._configure_trainability()
        count = sum(parameter.numel() for parameter in self.model.parameters())
        if count != EXPECTED_DINOV3_VITS16PLUS_PARAMETERS:
            raise RuntimeError(
                "Unexpected DINOv3 ViT-S+/16 parameter count: "
                f"expected={EXPECTED_DINOV3_VITS16PLUS_PARAMETERS}, actual={count}"
            )

    def _configure_trainability(self) -> None:
        if not self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            return
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.unfreeze_last_n_blocks:
            for block in self.model.blocks[-self.unfreeze_last_n_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            self.model.eval()

    @property
    def fully_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.model.parameters())

    def train(self, mode: bool = True) -> "DINOv3ViTS16PlusBackbone":
        super().train(mode)
        if self.fully_frozen:
            self.model.eval()
        return self

    def forward(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        image_list = list(images)
        if not image_list:
            raise ValueError("DINOv3 requires at least one image")
        for image in image_list:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(
                    "DINOv3 list-forward expects individual tensors shaped [1,3,H,W], "
                    f"got {tuple(image.shape)}"
                )
            if image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
                raise ValueError(f"DINOv3 input must be divisible by 16, got {tuple(image.shape[-2:])}")

        context = torch.no_grad() if self.fully_frozen else torch.enable_grad()
        with context:
            outputs = self.model.forward_features(image_list, [None] * len(image_list))
        features: list[torch.Tensor] = []
        for image, output in zip(image_list, outputs, strict=True):
            tokens = output["x_norm_patchtokens"]
            height = int(image.shape[-2]) // self.patch_size
            width = int(image.shape[-1]) // self.patch_size
            if tokens.shape[1] != height * width:
                raise RuntimeError(
                    f"DINOv3 patch token mismatch: tokens={tokens.shape[1]}, grid={height}x{width}"
                )
            feature = tokens.transpose(1, 2).reshape(1, self.embed_dim, height, width).contiguous()
            features.append(feature[0])
        return features


__all__ = ["DINOv3ViTS16PlusBackbone", "EXPECTED_DINOV3_VITS16PLUS_PARAMETERS"]
