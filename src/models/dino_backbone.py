from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


PATCH_TOKEN_KEYS = ("x_norm_patchtokens", "patch_tokens", "patchtokens")
SEQUENCE_TOKEN_KEYS = ("tokens", "x", "last_hidden_state")

DINO_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}


def _normalize_patch_size(patch_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    return int(patch_h), int(patch_w)


def _grid_size_from_image(
    image_size: tuple[int, int] | torch.Size,
    patch_size: tuple[int, int],
    *,
    allow_non_divisible: bool = False,
) -> tuple[int, int]:
    height, width = int(image_size[-2]), int(image_size[-1])
    patch_h, patch_w = patch_size
    if not allow_non_divisible and (height % patch_h != 0 or width % patch_w != 0):
        raise ValueError(
            "DINO input size must be divisible by patch_size for token reshape: "
            f"image_size=({height}, {width}), patch_size=({patch_h}, {patch_w})"
        )
    return height // patch_h, width // patch_w


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint at {path} does not contain a state dict")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            clean_key = str(key)
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            state_dict[clean_key] = value
    return state_dict


def _patch_torch_for_dinov3_backbone() -> None:
    """Provide small Torch 2.0 compatibility shims needed by local DINOv3 backbones."""

    try:
        import torch._dynamo as torch_dynamo  # type: ignore[attr-defined]
    except Exception:
        torch_dynamo = None
    if torch_dynamo is not None:
        if not hasattr(torch, "_dynamo"):
            setattr(torch, "_dynamo", torch_dynamo)
        if not hasattr(torch_dynamo, "reset_code_caches"):
            setattr(torch_dynamo, "reset_code_caches", lambda: None)
        config = getattr(torch_dynamo, "config", None)
        allowed_keys = getattr(config, "_allowed_keys", None)
        config_values = getattr(config, "_config", None)
        if isinstance(allowed_keys, set) and isinstance(config_values, dict):
            for key, default in {
                "automatic_dynamic_shapes": False,
                "accumulated_cache_size_limit": 1024,
            }.items():
                allowed_keys.add(key)
                config_values.setdefault(key, default)

    if not hasattr(torch, "compiler"):

        class _CompilerShim:
            @staticmethod
            def allow_in_graph(function):
                return function

        setattr(torch, "compiler", _CompilerShim())


def _load_local_entrypoint(
    *,
    repo: str | Path,
    entrypoint: str,
    module_name: str,
    pretrained: bool = False,
) -> nn.Module:
    repo_path = Path(repo).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Local DINO repo was not found: {repo_path}")
    repo_str = str(repo_path)
    inserted = False
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        inserted = True
    try:
        if module_name.startswith("dinov3."):
            _patch_torch_for_dinov3_backbone()
        module = importlib.import_module(module_name)
        factory = getattr(module, entrypoint)
        model = factory(pretrained=pretrained)
    finally:
        if inserted:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass
    if not isinstance(model, nn.Module):
        raise TypeError(f"Local DINO entrypoint {module_name}.{entrypoint} did not return nn.Module: {type(model)!r}")
    return model


def _load_local_hub_model(
    *,
    repo: str | Path,
    entrypoint: str,
    pretrained: bool = False,
    module_name: str | None = None,
) -> nn.Module:
    if module_name:
        return _load_local_entrypoint(repo=repo, entrypoint=entrypoint, module_name=module_name, pretrained=pretrained)

    try:
        return torch.hub.load(
            str(repo),
            entrypoint,
            source="local",
            pretrained=pretrained,
        )
    except ImportError:
        if str(entrypoint).startswith("dinov3_"):
            return _load_local_entrypoint(
                repo=repo,
                entrypoint=entrypoint,
                module_name="dinov3.hub.backbones",
                pretrained=pretrained,
            )
        raise


def _extract_token_tensor(output: Any, token_key: str | None) -> tuple[torch.Tensor, bool]:
    if isinstance(output, torch.Tensor):
        if output.ndim == 4:
            return output, True
        if output.ndim == 3:
            return output, False
        raise ValueError(f"Expected DINO tensor output with 3 or 4 dims, got {tuple(output.shape)}")

    if isinstance(output, Mapping):
        if token_key is not None:
            if token_key not in output:
                raise KeyError(f"token_key={token_key!r} was not found in DINO output keys={list(output.keys())}")
            value = output[token_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"DINO output[{token_key!r}] is not a Tensor: {type(value)!r}")
            return value, token_key in PATCH_TOKEN_KEYS or value.ndim == 4

        for key in PATCH_TOKEN_KEYS:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value, True
        for key in SEQUENCE_TOKEN_KEYS:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value, False
        for value in output.values():
            if isinstance(value, torch.Tensor) and value.ndim in {3, 4}:
                return value, value.ndim == 4
        raise TypeError(f"Could not find patch tokens in DINO output keys={list(output.keys())}")

    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for value in output:
            if isinstance(value, torch.Tensor) and value.ndim in {3, 4}:
                return value, value.ndim == 4

    raise TypeError(f"Unsupported DINO output type: {type(output)!r}")


def tokens_to_feature_map(
    tokens: torch.Tensor,
    *,
    image_size: tuple[int, int] | torch.Size,
    patch_size: int | tuple[int, int],
    num_prefix_tokens: int = 0,
    patch_tokens_only: bool = False,
    allow_non_divisible: bool = False,
) -> torch.Tensor:
    if tokens.ndim == 4:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, N, C), got {tuple(tokens.shape)}")

    if not patch_tokens_only and num_prefix_tokens > 0:
        if tokens.shape[1] <= num_prefix_tokens:
            raise ValueError(
                "Token sequence is shorter than the configured prefix count: "
                f"tokens={tokens.shape[1]}, num_prefix_tokens={num_prefix_tokens}"
            )
        tokens = tokens[:, num_prefix_tokens:, :]

    grid_h, grid_w = _grid_size_from_image(
        image_size,
        _normalize_patch_size(patch_size),
        allow_non_divisible=allow_non_divisible,
    )
    expected_tokens = grid_h * grid_w
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            "Patch token count mismatch: "
            f"got={tokens.shape[1]}, expected={expected_tokens}, grid=({grid_h}, {grid_w})"
        )
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w).contiguous()


class DummyDinoBackbone(nn.Module):
    """Small local DINO stand-in used for smoke tests without downloads."""

    def __init__(
        self,
        *,
        embed_dim: int = 384,
        patch_size: int | tuple[int, int] = 14,
        num_prefix_tokens: int = 1,
        output_key: str = "tokens",
    ) -> None:
        super().__init__()
        patch_h, patch_w = _normalize_patch_size(patch_size)
        self.embed_dim = int(embed_dim)
        self.patch_size = (patch_h, patch_w)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.output_key = output_key
        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            bias=False,
        )
        self.prefix_tokens = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, self.embed_dim))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_map = self.patch_embed(images)
        patch_tokens = patch_map.flatten(2).transpose(1, 2).contiguous()
        if self.output_key in PATCH_TOKEN_KEYS:
            return {self.output_key: patch_tokens}
        prefix = self.prefix_tokens.expand(images.shape[0], -1, -1)
        return {self.output_key: torch.cat([prefix, patch_tokens], dim=1)}


class DinoBackbone(nn.Module):
    """DINOv2-style patch-token wrapper that returns a 2D feature map.

    This wrapper never downloads weights. Pass a local DINO module through
    ``backbone=`` for real training, or use ``dummy=True`` explicitly for
    local shape tests.
    """

    def __init__(
        self,
        *,
        name: str = "dinov2_vits14",
        patch_size: int | tuple[int, int] = 14,
        embed_dim: int | None = None,
        backbone: nn.Module | None = None,
        hub_repo: str | Path | None = None,
        hub_model: str | None = None,
        hub_module: str | None = None,
        weights: str | Path | None = None,
        pretrained: bool = False,
        dummy: bool = False,
        token_key: str | None = None,
        num_prefix_tokens: int = 1,
        call_method: str = "auto",
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 0,
        strict_weights: bool = False,
        allow_non_divisible: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.patch_size = _normalize_patch_size(patch_size)
        self.embed_dim = int(embed_dim or DINO_EMBED_DIMS.get(name, 384))
        self.token_key = token_key
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.call_method = call_method
        self.freeze = bool(freeze)
        self.allow_non_divisible = bool(allow_non_divisible)

        if backbone is None:
            if hub_repo is not None:
                backbone = _load_local_hub_model(
                    repo=hub_repo,
                    entrypoint=hub_model or name,
                    pretrained=False,
                    module_name=hub_module,
                )
            elif dummy:
                backbone = DummyDinoBackbone(
                    embed_dim=self.embed_dim,
                    patch_size=self.patch_size,
                    num_prefix_tokens=self.num_prefix_tokens,
                    output_key=str(token_key or "tokens"),
                )
            else:
                raise ValueError(
                    "Automatic DINO downloads are disabled. Pass backbone=, configure a local hub_repo, "
                    "or set dummy=True explicitly for smoke tests."
                )
        self.backbone = backbone

        if weights is not None:
            state_dict = _load_state_dict(weights)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=strict_weights)
            if strict_weights and (missing or unexpected):
                raise RuntimeError(f"Failed to load strict DINO weights: missing={missing}, unexpected={unexpected}")

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            if unfreeze_last_n_blocks > 0:
                self._unfreeze_last_blocks(unfreeze_last_n_blocks)
            self.backbone.eval()

    def _unfreeze_last_blocks(self, count: int) -> None:
        block_container = None
        for attr in ("blocks", "layers"):
            value = getattr(self.backbone, attr, None)
            if isinstance(value, (nn.ModuleList, list, tuple)):
                block_container = value
                break
        if block_container is None:
            return
        for block in list(block_container)[-int(count) :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "DinoBackbone":
        super().train(mode)
        if self.freeze and not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def _call_backbone(self, images: torch.Tensor) -> Any:
        if self.call_method == "forward":
            return self.backbone(images)
        if self.call_method == "forward_features":
            return self.backbone.forward_features(images)
        if self.call_method != "auto":
            raise ValueError(f"Unsupported DINO call_method: {self.call_method}")
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(images)
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape (B, 3, H, W), got {tuple(images.shape)}")

        if self.freeze and not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            with torch.no_grad():
                output = self._call_backbone(images)
        else:
            output = self._call_backbone(images)

        tokens, patch_tokens_only = _extract_token_tensor(output, self.token_key)
        return tokens_to_feature_map(
            tokens,
            image_size=images.shape[-2:],
            patch_size=self.patch_size,
            num_prefix_tokens=self.num_prefix_tokens,
            patch_tokens_only=patch_tokens_only,
            allow_non_divisible=self.allow_non_divisible,
        )


def build_dino_backbone(config: Mapping[str, Any] | None = None, *, backbone: nn.Module | None = None) -> DinoBackbone:
    cfg = dict(config or {})
    return DinoBackbone(
        name=str(cfg.get("name", "dinov2_vits14")),
        patch_size=cfg.get("patch_size", 14),
        embed_dim=cfg.get("embed_dim"),
        backbone=backbone,
        hub_repo=cfg.get("hub_repo"),
        hub_model=cfg.get("hub_model"),
        hub_module=cfg.get("hub_module"),
        weights=cfg.get("weights", cfg.get("pretrained_path")),
        pretrained=bool(cfg.get("pretrained", False)),
        dummy=bool(cfg.get("dummy", False)),
        token_key=cfg.get("token_key"),
        num_prefix_tokens=int(cfg.get("num_prefix_tokens", 1)),
        call_method=str(cfg.get("call_method", "auto")),
        freeze=bool(cfg.get("freeze", True)),
        unfreeze_last_n_blocks=int(cfg.get("unfreeze_last_n_blocks", 0)),
        strict_weights=bool(cfg.get("strict_weights", False)),
        allow_non_divisible=bool(cfg.get("allow_non_divisible", False)),
    )
