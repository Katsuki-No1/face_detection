from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


PATCH_TOKEN_KEYS = ("x_norm_patchtokens", "patch_tokens", "patchtokens")
SEQUENCE_TOKEN_KEYS = ("tokens", "x", "last_hidden_state")


def normalize_patch_size(patch_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    return int(patch_h), int(patch_w)


def infer_grid_size(
    image_size: tuple[int, int] | torch.Size,
    patch_size: int | tuple[int, int],
    *,
    allow_non_divisible: bool = False,
) -> tuple[int, int]:
    if len(image_size) == 4:
        height, width = int(image_size[-2]), int(image_size[-1])
    elif len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError(f"image_size must be (H, W) or tensor shape, got {tuple(image_size)}")

    patch_h, patch_w = normalize_patch_size(patch_size)
    if not allow_non_divisible and (height % patch_h != 0 or width % patch_w != 0):
        raise ValueError(
            "Input image size must be divisible by patch_size: "
            f"image_size=({height}, {width}), patch_size=({patch_h}, {patch_w})"
        )
    return height // patch_h, width // patch_w


def square_grid_from_token_count(token_count: int) -> tuple[int, int]:
    side = int(round(token_count**0.5))
    if side * side != token_count:
        raise ValueError(
            "image_size or grid_size is required when patch token count is not a square: "
            f"token_count={token_count}"
        )
    return side, side


class PatchTokenToFeatureMap(nn.Module):
    """Converts a DINO token sequence from (B, N, C) into (B, C, H, W)."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        *,
        num_prefix_tokens: int = 0,
        allow_non_divisible: bool = False,
    ) -> None:
        super().__init__()
        if num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be >= 0")
        self.patch_size = normalize_patch_size(patch_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.allow_non_divisible = bool(allow_non_divisible)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        image_size: tuple[int, int] | torch.Size | None = None,
        grid_size: tuple[int, int] | None = None,
        num_prefix_tokens: int | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape (B, N, C), got shape={tuple(tokens.shape)}")

        prefix_tokens = self.num_prefix_tokens if num_prefix_tokens is None else int(num_prefix_tokens)
        if prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be >= 0")
        if prefix_tokens:
            if tokens.shape[1] <= prefix_tokens:
                raise ValueError(
                    "Token sequence is shorter than the configured prefix token count: "
                    f"token_count={tokens.shape[1]}, num_prefix_tokens={prefix_tokens}"
                )
            tokens = tokens[:, prefix_tokens:, :]

        batch_size, token_count, channels = tokens.shape
        if grid_size is None:
            if image_size is None:
                grid_size = square_grid_from_token_count(int(token_count))
            else:
                grid_size = infer_grid_size(
                    image_size,
                    self.patch_size,
                    allow_non_divisible=self.allow_non_divisible,
                )

        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
        expected_tokens = grid_h * grid_w
        if token_count != expected_tokens:
            image_text = "unknown" if image_size is None else tuple(int(v) for v in image_size[-2:])
            raise ValueError(
                "Patch token count mismatch: "
                f"got {token_count} patch tokens after removing {prefix_tokens} prefix tokens, "
                f"expected {expected_tokens} for grid=({grid_h}, {grid_w}), "
                f"image_size={image_text}, patch_size={self.patch_size}"
            )
        return tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w).contiguous()


def _tensor_from_output(output: Any, token_key: str | None) -> tuple[torch.Tensor, bool, str]:
    if isinstance(output, torch.Tensor):
        return output, False, "tensor"

    if isinstance(output, Mapping):
        if token_key is not None:
            if token_key not in output:
                raise KeyError(f"token_key={token_key!r} was not found in DINO output keys={list(output.keys())}")
            value = output[token_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"DINO output[{token_key!r}] is not a Tensor: {type(value)!r}")
            return value, token_key in PATCH_TOKEN_KEYS, token_key

        for key in PATCH_TOKEN_KEYS:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value, True, key
        for key in SEQUENCE_TOKEN_KEYS:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value, False, key
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value, False, str(key)
        raise TypeError(f"Could not find a 3D token tensor in DINO output keys={list(output.keys())}")

    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for index, value in enumerate(output):
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value, False, f"sequence[{index}]"

    raise TypeError(
        "DINO backbone output must be a Tensor, mapping, or sequence containing a "
        f"(B, N, C) token Tensor, got {type(output)!r}"
    )


class DinoPatchFeatureMapBackbone(nn.Module):
    """Wraps a DINO-like backbone and returns a 2D patch-token feature map."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_size: int | tuple[int, int],
        num_prefix_tokens: int = 0,
        token_key: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.token_key = token_key
        self.freeze = bool(freeze)
        self.converter = PatchTokenToFeatureMap(patch_size, num_prefix_tokens=num_prefix_tokens)
        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DinoPatchFeatureMapBackbone":
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be (B, 3, H, W), got shape={tuple(images.shape)}")
        if self.freeze:
            with torch.no_grad():
                output = self.backbone(images)
        else:
            output = self.backbone(images)

        tokens, patch_only, _source = _tensor_from_output(output, self.token_key)
        prefix_tokens = 0 if patch_only else self.converter.num_prefix_tokens
        return self.converter(tokens, image_size=images.shape[-2:], num_prefix_tokens=prefix_tokens)


class FakeTokenBackbone(nn.Module):
    """Tiny DINO stand-in for CPU shape tests. It never loads external weights."""

    def __init__(
        self,
        *,
        embed_dim: int = 64,
        patch_size: int | tuple[int, int] = 14,
        num_prefix_tokens: int = 1,
        output_key: str = "tokens",
        include_prefix_tokens: bool | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = int(embed_dim)
        self.patch_size = normalize_patch_size(patch_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.output_key = output_key
        if include_prefix_tokens is None:
            include_prefix_tokens = output_key not in PATCH_TOKEN_KEYS
        self.include_prefix_tokens = bool(include_prefix_tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        grid_h, grid_w = infer_grid_size(images.shape[-2:], self.patch_size)
        patch_count = grid_h * grid_w
        batch_size = images.shape[0]
        dtype = images.dtype
        device = images.device

        spatial = torch.linspace(0.0, 1.0, patch_count, device=device, dtype=dtype).view(1, patch_count, 1)
        channels = torch.linspace(0.0, 0.5, self.embed_dim, device=device, dtype=dtype).view(1, 1, self.embed_dim)
        patch_tokens = (spatial + channels).expand(batch_size, patch_count, self.embed_dim).contiguous()

        tokens = patch_tokens
        if self.include_prefix_tokens and self.num_prefix_tokens > 0:
            prefix = torch.zeros(batch_size, self.num_prefix_tokens, self.embed_dim, device=device, dtype=dtype)
            tokens = torch.cat([prefix, patch_tokens], dim=1)

        if self.output_key == "tensor":
            return tokens
        return {self.output_key: tokens}


DummyDINOBackbone = FakeTokenBackbone
