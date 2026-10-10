from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .query_error_augmentation import (
    QueryErrorAugmentationConfig,
    apply_query_error_augmentation,
)


LEGACY_BINARY_STATE_HEAD = "legacy_binary"
FACTORIZED_STATE_HEAD = "factorized_presence_visibility"
STATE_HEAD_MODES = (LEGACY_BINARY_STATE_HEAD, FACTORIZED_STATE_HEAD)


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while channels % groups:
        groups -= 1
    return groups


class RGBRefinementFeaturePyramid(nn.Module):
    """Build stride-8/4 RGB detail maps for coarse-to-fine refinement."""

    feature_sources = ("stride8", "stride4")

    def __init__(
        self,
        *,
        base_channels: int = 32,
        projection_channels: int = 128,
    ) -> None:
        super().__init__()
        if base_channels <= 0 or projection_channels <= 0:
            raise ValueError("RGB refinement channels must be positive")
        self.base_channels = int(base_channels)
        self.projection_channels = int(projection_channels)
        self.stem_conv = nn.Conv2d(
            3,
            self.base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(
            _group_count(self.base_channels), self.base_channels
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stride8_conv = nn.Conv2d(
            self.base_channels,
            self.base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.stride8_norm = nn.GroupNorm(
            _group_count(self.base_channels * 2), self.base_channels * 2
        )
        self.activation = nn.GELU()

        def adapter(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(channels), channels),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    self.projection_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(self.projection_channels),
                    self.projection_channels,
                ),
                nn.GELU(),
            )

        self.stride4_adapter = adapter(self.base_channels)
        self.stride8_adapter = adapter(self.base_channels * 2)

    @property
    def out_channels(self) -> tuple[int, int]:
        return (self.projection_channels, self.projection_channels)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "RGB refinement expects Bx3xHxW images, got "
                f"{tuple(images.shape)}"
            )
        stride2 = self.activation(self.stem_norm(self.stem_conv(images)))
        stride4 = self.stem_pool(stride2)
        stride8 = self.activation(
            self.stride8_norm(self.stride8_conv(stride4))
        )
        return (
            self.stride8_adapter(stride8),
            self.stride4_adapter(stride4),
        )


class ConvKVFeaturePyramid(nn.Module):
    """Build stride-16/32/64 K/V maps from same-grid DINO semantic levels."""

    def __init__(self, channels: int, *, semantic_levels: int = 3) -> None:
        super().__init__()
        if semantic_levels != 3:
            raise ValueError("ConvKVFeaturePyramid requires three semantic levels")
        self.semantic_levels = int(semantic_levels)
        self.num_levels = 3
        self.out_channels = int(channels)
        groups = 8 if channels % 8 == 0 else 1
        initial_logits = torch.zeros(self.num_levels, self.semantic_levels)
        initial_logits.diagonal().fill_(2.0)
        self.semantic_logits = nn.Parameter(initial_logits)

        def conv_block(*, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )

        self.local_branch = conv_block(stride=1)
        self.medium_branch = conv_block(stride=2)
        self.global_branch = nn.Sequential(
            conv_block(stride=2),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            conv_block(stride=2),
        )

    def forward(
        self,
        semantic_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = tuple(semantic_features)
        if len(features) != self.semantic_levels:
            raise ValueError(
                "Conv K/V semantic input count mismatch: "
                f"expected={self.semantic_levels}, got={len(features)}"
            )
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("Every Conv K/V semantic feature must be BxCxHxW")
        if any(feature.shape != features[0].shape for feature in features[1:]):
            raise ValueError(
                "Conv K/V semantic features must share channels and one patch grid"
            )
        weights = self.semantic_logits.softmax(dim=-1)
        mixed = tuple(
            sum(
                weights[scale_index, level_index].to(dtype=feature.dtype)
                * feature
                for level_index, feature in enumerate(features)
            )
            for scale_index in range(self.num_levels)
        )
        return (
            mixed[0] + self.local_branch(mixed[0]),
            self.medium_branch(mixed[1]),
            self.global_branch(mixed[2]),
        )


class IntermediateFeaturePyramid(nn.Module):
    """Fuse three DINO block outputs into stride-8/16/32 FPN features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_levels: int = 3,
    ) -> None:
        super().__init__()
        if num_levels != 3:
            raise ValueError(
                "IntermediateFeaturePyramid currently requires exactly three levels"
            )
        self.num_levels = int(num_levels)
        self.out_channels = int(out_channels)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for _ in range(self.num_levels)
            ]
        )
        groups = 8 if out_channels % 8 == 0 else 1
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, out_channels),
                    nn.GELU(),
                )
                for _ in range(self.num_levels)
            ]
        )

    def forward(
        self,
        intermediate_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = tuple(intermediate_features)
        if len(features) != self.num_levels:
            raise ValueError(
                "Intermediate FPN input count mismatch: "
                f"expected={self.num_levels}, got={len(features)}"
            )
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("Every intermediate DINO feature must be BxCxHxW")
        base_size = features[1].shape[-2:]
        if any(feature.shape[-2:] != base_size for feature in features):
            raise ValueError(
                "Transformer intermediate features must share one patch grid before FPN"
            )

        lateral = [
            projection(feature)
            for projection, feature in zip(
                self.lateral_convs,
                features,
                strict=True,
            )
        ]
        base_height, base_width = int(base_size[0]), int(base_size[1])
        c3 = F.interpolate(
            lateral[0],
            size=(base_height * 2, base_width * 2),
            mode="bilinear",
            align_corners=False,
        )
        c4 = lateral[1]
        c5 = F.adaptive_avg_pool2d(
            lateral[2],
            output_size=(max(1, base_height // 2), max(1, base_width // 2)),
        )

        p5 = c5
        p4 = c4 + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest",
        )
        p3 = c3 + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest",
        )
        outputs = (p3, p4, p5)
        return tuple(
            output_conv(feature)
            for output_conv, feature in zip(
                self.output_convs,
                outputs,
                strict=True,
            )
        )


def factorized_state_probabilities(
    presence_logits: torch.Tensor,
    visibility_logits: torch.Tensor,
) -> torch.Tensor:
    """Return probabilities ordered as missing, occluded, and visible."""

    if presence_logits.shape != visibility_logits.shape:
        raise ValueError(
            "presence_logits and visibility_logits must have the same shape, "
            f"got {tuple(presence_logits.shape)} and "
            f"{tuple(visibility_logits.shape)}"
        )
    presence = presence_logits.sigmoid()
    visible_given_present = visibility_logits.sigmoid()
    return torch.stack(
        [
            1.0 - presence,
            presence * (1.0 - visible_given_present),
            presence * visible_given_present,
        ],
        dim=-1,
    )


def soft_argmax_2d(heatmaps: torch.Tensor, *, temperature: float = 0.10) -> torch.Tensor:
    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must be BxKxHxW, got {tuple(heatmaps.shape)}")
    batch, landmarks, height, width = heatmaps.shape
    probabilities = torch.softmax(heatmaps.flatten(2) / max(float(temperature), 1e-4), dim=-1)
    xs = torch.linspace(0.0, 1.0, width, dtype=heatmaps.dtype, device=heatmaps.device)
    ys = torch.linspace(0.0, 1.0, height, dtype=heatmaps.dtype, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
    y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
    return torch.stack([x, y], dim=-1).reshape(batch, landmarks, 2)


def decode_heatmap_argmax(heatmaps: torch.Tensor, *, quarter_pixel: bool = True) -> torch.Tensor:
    """Decode heatmaps with the common quarter-pixel local-gradient correction."""

    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must be BxKxHxW, got {tuple(heatmaps.shape)}")
    batch, landmarks, height, width = heatmaps.shape
    indices = heatmaps.flatten(2).argmax(dim=-1)
    xs = (indices % width).to(dtype=heatmaps.dtype)
    ys = torch.div(indices, width, rounding_mode="floor").to(dtype=heatmaps.dtype)
    if quarter_pixel and height >= 3 and width >= 3:
        padded = F.pad(heatmaps, (1, 1, 1, 1), mode="replicate")
        batch_index = torch.arange(batch, device=heatmaps.device)[:, None]
        landmark_index = torch.arange(landmarks, device=heatmaps.device)[None, :]
        px = xs.long() + 1
        py = ys.long() + 1
        dx = padded[batch_index, landmark_index, py, px + 1] - padded[batch_index, landmark_index, py, px - 1]
        dy = padded[batch_index, landmark_index, py + 1, px] - padded[batch_index, landmark_index, py - 1, px]
        xs = xs + 0.25 * torch.sign(dx)
        ys = ys + 0.25 * torch.sign(dy)
    x_norm = xs / float(max(1, width - 1))
    y_norm = ys / float(max(1, height - 1))
    return torch.stack([x_norm, y_norm], dim=-1).clamp(0.0, 1.0)


class ViTPoseHeatmapHead(nn.Module):
    """ViTPose-style CNN decoder over a ViT patch-token feature map."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_landmarks: int = 5,
        hidden_channels: int = 256,
        deconv_layers: int = 2,
        deconv_kernel: int = 4,
        softargmax_temperature: float = 0.10,
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
    ) -> None:
        super().__init__()
        if deconv_layers < 1:
            raise ValueError("deconv_layers must be positive")
        if deconv_kernel not in {2, 4}:
            raise ValueError("deconv_kernel must be 2 or 4")
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"Unsupported state_head_mode={state_head_mode!r}; "
                f"expected one of {STATE_HEAD_MODES}"
            )
        self.state_head_mode = str(state_head_mode)
        padding = 1 if deconv_kernel == 4 else 0
        layers: list[nn.Module] = []
        channels = int(in_channels)
        for _ in range(int(deconv_layers)):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        hidden_channels,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = hidden_channels
        self.deconv = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(channels, num_landmarks, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_landmarks),
        )
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_landmarks),
            )
        self.softargmax_temperature = float(softargmax_temperature)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final_bias = self.final_layer.bias
        if final_bias is not None:
            nn.init.zeros_(final_bias)

    def forward(self, feature_map: torch.Tensor, *, return_attention: bool = False) -> dict[str, Any]:
        del return_attention
        heatmaps = self.final_layer(self.deconv(feature_map))
        visibility_logits = self.visibility_head(feature_map)
        output = {
            "heatmaps": heatmaps,
            "coordinates": soft_argmax_2d(heatmaps, temperature=self.softargmax_temperature),
            "decoded_coordinates": decode_heatmap_argmax(heatmaps),
            "visibility_logits": visibility_logits,
            "auxiliary_coordinates": [],
            "self_attention": [],
            "cross_attention": [],
        }
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            presence_logits = self.presence_head(feature_map)
            output["presence_logits"] = presence_logits
            output["state_probabilities"] = factorized_state_probabilities(
                presence_logits,
                visibility_logits,
            )
        return output


def sine_position_encoding_2d(
    height: int,
    width: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dimension % 4 != 0:
        raise ValueError("2D sine position encoding dimension must be divisible by four")
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    features_per_axis = dimension // 2
    feature_index = torch.arange(features_per_axis, device=device, dtype=dtype)
    divisor = 10000.0 ** (2.0 * torch.div(feature_index, 2, rounding_mode="floor") / features_per_axis)

    def encode(values: torch.Tensor) -> torch.Tensor:
        phase = values[..., None] * (2.0 * math.pi) / divisor
        return torch.stack([phase[..., 0::2].sin(), phase[..., 1::2].cos()], dim=-1).flatten(-2)

    position = torch.cat([encode(grid_y), encode(grid_x)], dim=-1)
    return position.reshape(1, height * width, dimension)


def sine_reference_point_encoding(
    reference_points: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    """Encode normalized x/y query references without adding learned weights."""

    if reference_points.ndim != 3 or reference_points.shape[-1] != 2:
        raise ValueError(
            "reference_points must be BxKx2, got "
            f"{tuple(reference_points.shape)}"
        )
    if dimension % 4 != 0:
        raise ValueError("2D sine position encoding dimension must be divisible by four")
    features_per_axis = dimension // 2
    feature_index = torch.arange(
        features_per_axis,
        device=reference_points.device,
        dtype=reference_points.dtype,
    )
    divisor = 10000.0 ** (
        2.0
        * torch.div(feature_index, 2, rounding_mode="floor")
        / features_per_axis
    )

    def encode(values: torch.Tensor) -> torch.Tensor:
        phase = values[..., None] * (2.0 * math.pi) / divisor
        return torch.stack(
            [phase[..., 0::2].sin(), phase[..., 1::2].cos()],
            dim=-1,
        ).flatten(-2)

    return torch.cat(
        [encode(reference_points[..., 1]), encode(reference_points[..., 0])],
        dim=-1,
    )


def inverse_sigmoid(value: torch.Tensor, *, epsilon: float = 1.0e-5) -> torch.Tensor:
    bounded = value.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log(bounded / (1.0 - bounded))


class LandmarkDecoderLayer(nn.Module):
    def __init__(self, dimension: int, *, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
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
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_position: torch.Tensor,
        query_position: torch.Tensor | None = None,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normalized = self.norm1(queries)
        self_query = (
            normalized
            if query_position is None
            else normalized + query_position
        )
        self_result, self_weights = self.self_attention(
            self_query,
            self_query,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        queries = queries + self.dropout(self_result)
        normalized = self.norm2(queries)
        cross_query = (
            normalized
            if query_position is None
            else normalized + query_position
        )
        cross_result, cross_weights = self.cross_attention(
            cross_query,
            memory + memory_position,
            memory,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        queries = queries + self.dropout(cross_result)
        queries = queries + self.dropout(self.ffn(self.norm3(queries)))
        return queries, self_weights, cross_weights


class LandmarkQueryHead(nn.Module):
    """Five semantic landmark queries with explicit query-query self-attention."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_landmarks: int = 5,
        dimension: int = 256,
        decoder_layers: int = 4,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
        auxiliary_supervision: bool = True,
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
        iterative_reference_points: bool = False,
        initial_reference_points: list[list[float]] | None = None,
        coordinate_update_mode: str = "absolute",
        feature_pyramid: IntermediateFeaturePyramid | None = None,
        kv_feature_pyramid: ConvKVFeaturePyramid | None = None,
        feature_levels: int = 1,
        face_query_dim: int | None = None,
    ) -> None:
        super().__init__()
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"Unsupported state_head_mode={state_head_mode!r}; "
                f"expected one of {STATE_HEAD_MODES}"
            )
        self.num_landmarks = int(num_landmarks)
        self.dimension = int(dimension)
        self.state_head_mode = str(state_head_mode)
        self.iterative_reference_points = bool(iterative_reference_points)
        self.coordinate_update_mode = str(coordinate_update_mode)
        self.feature_pyramid = feature_pyramid
        self.kv_feature_pyramid = kv_feature_pyramid
        self.feature_levels = int(feature_levels)
        if self.feature_levels <= 0:
            raise ValueError("feature_levels must be positive")
        if self.feature_pyramid is not None and (
            self.feature_pyramid.num_levels != self.feature_levels
        ):
            raise ValueError(
                "feature_pyramid.num_levels must match feature_levels"
            )
        if self.feature_pyramid is not None and self.kv_feature_pyramid is not None:
            raise ValueError(
                "feature_pyramid and kv_feature_pyramid are mutually exclusive"
            )
        if self.kv_feature_pyramid is not None and (
            self.kv_feature_pyramid.num_levels != self.feature_levels
        ):
            raise ValueError(
                "kv_feature_pyramid.num_levels must match feature_levels"
            )
        if self.coordinate_update_mode not in {"absolute", "inverse_sigmoid_delta"}:
            raise ValueError(
                "coordinate_update_mode must be 'absolute' or "
                "'inverse_sigmoid_delta'"
            )
        if (
            self.coordinate_update_mode == "inverse_sigmoid_delta"
            and not self.iterative_reference_points
        ):
            raise ValueError(
                "inverse_sigmoid_delta requires iterative_reference_points=true"
            )
        self.memory_projection = nn.Conv2d(in_channels, dimension, kernel_size=1)
        self.landmark_queries = nn.Parameter(torch.empty(1, num_landmarks, dimension))
        if self.feature_levels > 1:
            self.level_embeddings = nn.Parameter(
                torch.empty(self.feature_levels, dimension)
            )
        if self.iterative_reference_points:
            default_reference_points = (
                initial_reference_points
                if initial_reference_points is not None
                else self._default_reference_points(num_landmarks)
            )
            reference_tensor = torch.as_tensor(
                default_reference_points,
                dtype=torch.float32,
            )
            if reference_tensor.shape != (num_landmarks, 2):
                raise ValueError(
                    "initial_reference_points must have shape "
                    f"({num_landmarks}, 2), got {tuple(reference_tensor.shape)}"
                )
            if not bool(
                ((reference_tensor > 0.0) & (reference_tensor < 1.0)).all()
            ):
                raise ValueError(
                    "initial_reference_points values must be strictly between zero and one"
                )
            self.initial_reference_point_logits = nn.Parameter(
                inverse_sigmoid(reference_tensor).unsqueeze(0)
            )
            self.missing_query_embedding = nn.Parameter(
                torch.empty(1, 1, dimension)
            )
        self.layers = nn.ModuleList(
            [
                LandmarkDecoderLayer(dimension, heads=heads, feedforward=feedforward, dropout=dropout)
                for _ in range(decoder_layers)
            ]
        )
        self.coordinate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, dimension),
                    nn.GELU(),
                    nn.Linear(dimension, 2),
                )
                for _ in range(decoder_layers)
            ]
        )
        self.visibility_head = nn.Linear(dimension, 1)
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            self.presence_head = nn.Linear(dimension, 1)
        self.auxiliary_supervision = bool(auxiliary_supervision)
        nn.init.normal_(self.landmark_queries, std=0.02)
        if self.feature_levels > 1:
            nn.init.normal_(self.level_embeddings, std=0.02)
        if self.iterative_reference_points:
            nn.init.normal_(self.missing_query_embedding, std=0.02)
        for head in self.coordinate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # Construct after legacy initialization; disabled has no new state keys.
        from .face_overlap import FaceOverlapConditioner
        self.face_overlap = (FaceOverlapConditioner(face_query_dim, dimension)
                             if face_query_dim is not None else None)

    @staticmethod
    def _default_reference_points(num_landmarks: int) -> list[list[float]]:
        if num_landmarks == 5:
            return [
                [0.32, 0.38],
                [0.68, 0.38],
                [0.50, 0.52],
                [0.37, 0.69],
                [0.63, 0.69],
            ]
        return [[0.50, 0.50] for _ in range(num_landmarks)]

    def forward(
        self,
        feature_map: torch.Tensor | Sequence[torch.Tensor],
        *,
        return_attention: bool = False,
        input_error_augmentation: QueryErrorAugmentationConfig | None = None,
        augmentation_valid_mask: torch.Tensor | None = None,
        face_query_features: torch.Tensor | None = None,
        face_geometry: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.kv_feature_pyramid is not None:
            if isinstance(feature_map, torch.Tensor):
                raise TypeError(
                    "The Conv K/V pyramid requires multiple DINO semantic maps"
                )
            feature_maps = self.kv_feature_pyramid(tuple(feature_map))
        elif self.feature_pyramid is not None:
            if isinstance(feature_map, torch.Tensor):
                raise TypeError(
                    "The intermediate feature pyramid requires multiple DINO feature maps"
                )
            feature_maps = self.feature_pyramid(tuple(feature_map))
        elif isinstance(feature_map, torch.Tensor):
            feature_maps = (feature_map,)
        else:
            feature_maps = tuple(feature_map)
        if len(feature_maps) != self.feature_levels:
            raise ValueError(
                "Query feature-level count mismatch: "
                f"expected={self.feature_levels}, got={len(feature_maps)}"
            )

        projected_maps = tuple(
            self.memory_projection(feature) for feature in feature_maps
        )
        # Keep the batch dimension symbolic during ONNX export.  Converting the
        # value to ``int`` bakes the example batch (normally one) into
        # MultiheadAttention reshape nodes and breaks the advertised dynamic
        # batch contract used by TensorRT.
        batch = projected_maps[0].shape[0]
        memories: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for level_index, projected in enumerate(projected_maps):
            _, _, height, width = projected.shape
            memories.append(
                projected.flatten(2).transpose(1, 2).contiguous()
            )
            level_position = sine_position_encoding_2d(
                height,
                width,
                self.dimension,
                device=projected.device,
                dtype=projected.dtype,
            )
            if self.feature_levels > 1:
                level_position = level_position + self.level_embeddings[
                    level_index
                ].reshape(1, 1, -1).to(dtype=projected.dtype)
            positions.append(level_position)
        memory = torch.cat(memories, dim=1)
        position = torch.cat(positions, dim=1)
        projected = projected_maps[0]
        queries = self.landmark_queries.expand(batch, -1, -1)
        if self.face_overlap is not None:
            if face_query_features is None or face_geometry is None:
                raise ValueError("FOR requires both face_query_features and face_geometry")
            if face_query_features.shape[0] != batch:
                raise ValueError("FOR face query batch must match crop batch")
            queries = queries + self.face_overlap(face_query_features, face_geometry).unsqueeze(1)
        elif face_query_features is not None or face_geometry is not None:
            raise ValueError("FOR inputs supplied while face_overlap_robustness is disabled")
        clean_initial_reference_points: torch.Tensor | None = None
        decoder_input_reference_points: torch.Tensor | None = None
        reference_points: torch.Tensor | None = None
        augmentation_masks = {
            mode: torch.zeros(
                (batch, self.num_landmarks),
                dtype=torch.bool,
                device=projected.device,
            )
            for mode in ("jitter", "miss", "swap")
        }
        if self.iterative_reference_points:
            clean_initial_reference_points = (
                self.initial_reference_point_logits.sigmoid()
                .to(dtype=projected.dtype)
                .expand(batch, -1, -1)
            )
            reference_points = clean_initial_reference_points
            if (
                self.training
                and input_error_augmentation is not None
                and input_error_augmentation.enabled
            ):
                reference_points, augmentation_masks = (
                    apply_query_error_augmentation(
                        reference_points,
                        valid_mask=augmentation_valid_mask,
                        config=input_error_augmentation,
                    )
                )
                queries = queries + (
                    augmentation_masks["miss"][..., None].to(queries.dtype)
                    * self.missing_query_embedding
                )
            decoder_input_reference_points = reference_points
        elif input_error_augmentation is not None and input_error_augmentation.enabled:
            raise ValueError(
                "input keypoint error augmentation requires "
                "iterative_reference_points=true"
            )
        decoder_input_query_features = queries
        coordinates: list[torch.Tensor] = []
        self_attention: list[torch.Tensor] = []
        cross_attention: list[torch.Tensor] = []
        for layer, coordinate_head in zip(self.layers, self.coordinate_heads, strict=True):
            query_position = (
                sine_reference_point_encoding(reference_points, self.dimension)
                if reference_points is not None
                else None
            )
            queries, self_weights, cross_weights = layer(
                queries,
                memory,
                memory_position=position,
                query_position=query_position,
                return_attention=return_attention,
            )
            coordinate_output = coordinate_head(queries)
            if self.coordinate_update_mode == "inverse_sigmoid_delta":
                if reference_points is None:
                    raise RuntimeError(
                        "inverse_sigmoid_delta requires reference points"
                    )
                reference_points = torch.sigmoid(
                    inverse_sigmoid(reference_points) + coordinate_output
                )
                coordinates.append(reference_points)
            else:
                absolute_coordinates = torch.sigmoid(coordinate_output)
                coordinates.append(absolute_coordinates)
                if self.iterative_reference_points:
                    reference_points = absolute_coordinates
            if self_weights is not None:
                self_attention.append(self_weights)
            if cross_weights is not None:
                cross_attention.append(cross_weights)
        visibility_logits = self.visibility_head(queries).select(-1, 0)
        output = {
            "coordinates": coordinates[-1],
            "decoded_coordinates": coordinates[-1],
            "visibility_logits": visibility_logits,
            "auxiliary_coordinates": coordinates[:-1] if self.auxiliary_supervision else [],
            "query_features": queries,
            "projected_feature_map": projected,
            "projected_feature_maps": projected_maps,
            "self_attention": self_attention,
            "cross_attention": cross_attention,
            "clean_initial_reference_points": clean_initial_reference_points,
            "decoder_input_reference_points": decoder_input_reference_points,
            "decoder_reference_points": coordinates,
            "decoder_input_query_features": decoder_input_query_features,
            "input_keypoint_error_masks": augmentation_masks,
        }
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            presence_logits = self.presence_head(queries).select(-1, 0)
            output["presence_logits"] = presence_logits
            output["state_probabilities"] = factorized_state_probabilities(
                presence_logits,
                visibility_logits,
            )
        return output


class CoordinateRefinementStage(nn.Module):
    """Coordinate-conditioned local sampling followed by bounded residual correction."""

    def __init__(
        self,
        dimension: int,
        *,
        radius_in_feature_cells: float,
        maximum_delta: float,
        samples_per_axis: int = 3,
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
        feature_levels: int = 1,
        feature_channels: int | None = None,
        feature_source: str | None = None,
    ) -> None:
        super().__init__()
        if samples_per_axis < 1 or samples_per_axis % 2 == 0:
            raise ValueError("samples_per_axis must be a positive odd integer")
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"Unsupported state_head_mode={state_head_mode!r}; "
                f"expected one of {STATE_HEAD_MODES}"
            )
        self.radius_in_feature_cells = float(radius_in_feature_cells)
        self.maximum_delta = float(maximum_delta)
        self.samples_per_axis = int(samples_per_axis)
        self.state_head_mode = str(state_head_mode)
        self.feature_levels = int(feature_levels)
        self.feature_source = (
            str(feature_source) if feature_source is not None else None
        )
        self.feature_channels = int(
            dimension if feature_channels is None else feature_channels
        )
        if self.feature_levels <= 0:
            raise ValueError("feature_levels must be positive")
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        local_dimension = (
            self.feature_channels
            * samples_per_axis
            * samples_per_axis
            * self.feature_levels
        )
        self.mlp = nn.Sequential(
            nn.Linear(dimension + local_dimension + 2, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(dimension, 2)
        self.visibility_delta = nn.Linear(dimension, 1)
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            self.presence_delta = nn.Linear(dimension, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.visibility_delta.weight)
        nn.init.zeros_(self.visibility_delta.bias)
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            nn.init.zeros_(self.presence_delta.weight)
            nn.init.zeros_(self.presence_delta.bias)

    def _sample(self, feature_map: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        offsets_1d = torch.linspace(
            -1.0,
            1.0,
            self.samples_per_axis,
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack([offset_x, offset_y], dim=-1).reshape(1, 1, -1, 2)
        scale = torch.tensor(
            [
                self.radius_in_feature_cells / float(max(1, width - 1)),
                self.radius_in_feature_cells / float(max(1, height - 1)),
            ],
            dtype=feature_map.dtype,
            device=feature_map.device,
        ).reshape(1, 1, 1, 2)
        normalized_grid = (coordinates[:, :, None, :] + offsets * scale).mul(2.0).sub(1.0)
        grid = normalized_grid.reshape(batch, -1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.select(-1, 0).transpose(1, 2)
        return sampled.reshape(batch, coordinates.shape[1], -1)

    def forward(
        self,
        feature_map: torch.Tensor | Sequence[torch.Tensor],
        query_features: torch.Tensor,
        coordinates: torch.Tensor,
        visibility_logits: torch.Tensor,
        presence_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature_maps = (
            (feature_map,) if isinstance(feature_map, torch.Tensor) else tuple(feature_map)
        )
        if len(feature_maps) != self.feature_levels:
            raise ValueError(
                "Refinement feature-level count mismatch: "
                f"expected={self.feature_levels}, got={len(feature_maps)}"
            )
        local = torch.cat(
            [self._sample(feature, coordinates) for feature in feature_maps],
            dim=-1,
        )
        hidden = self.mlp(torch.cat([query_features, local, coordinates], dim=-1))
        delta = torch.tanh(self.delta_head(hidden)) * self.maximum_delta
        refined = (coordinates + delta).clamp(0.0, 1.0)
        visibility = (
            visibility_logits
            + self.visibility_delta(hidden).select(-1, 0)
        )
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            if presence_logits is None:
                raise ValueError(
                    "factorized refinement requires presence_logits"
                )
            presence = (
                presence_logits
                + self.presence_delta(hidden).select(-1, 0)
            )
            return refined, presence, visibility, hidden
        return refined, visibility, hidden


class LandmarkQueryRefinementHead(nn.Module):
    def __init__(
        self,
        query_head: LandmarkQueryHead,
        *,
        stages: list[dict[str, Any]],
        input_keypoint_error_augmentation: dict[str, Any] | None = None,
        refinement_feature_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.query_head = query_head
        self.input_keypoint_error_augmentation = (
            QueryErrorAugmentationConfig.from_mapping(
                input_keypoint_error_augmentation,
                num_landmarks=query_head.num_landmarks,
            )
        )
        external_channels = (
            tuple(int(value) for value in refinement_feature_channels)
            if refinement_feature_channels is not None
            else None
        )
        if external_channels is not None and len(external_channels) != len(stages):
            raise ValueError(
                "refinement_feature_channels must define one entry per stage"
            )
        self.uses_external_refinement_features = external_channels is not None
        self.refinement_stages = nn.ModuleList(
            [
                CoordinateRefinementStage(
                    query_head.dimension,
                    radius_in_feature_cells=float(stage["radius_in_feature_cells"]),
                    maximum_delta=float(stage["maximum_delta"]),
                    samples_per_axis=int(stage.get("samples_per_axis", 3)),
                    state_head_mode=query_head.state_head_mode,
                    feature_levels=(
                        1
                        if external_channels is not None
                        else query_head.feature_levels
                    ),
                    feature_channels=(
                        external_channels[index]
                        if external_channels is not None
                        else query_head.dimension
                    ),
                    feature_source=stage.get("feature_source"),
                )
                for index, stage in enumerate(stages)
            ]
        )

    def forward(
        self,
        feature_map: torch.Tensor | Sequence[torch.Tensor],
        *,
        return_attention: bool = False,
        augmentation_valid_mask: torch.Tensor | None = None,
        refinement_feature_maps: Sequence[torch.Tensor] | None = None,
        face_query_features: torch.Tensor | None = None,
        face_geometry: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        output = self.query_head(
            feature_map,
            return_attention=return_attention,
            input_error_augmentation=self.input_keypoint_error_augmentation,
            augmentation_valid_mask=augmentation_valid_mask,
            face_query_features=face_query_features,
            face_geometry=face_geometry,
        )
        coordinates = output["coordinates"]
        visibility = output["visibility_logits"]
        presence = output.get("presence_logits")
        queries = output["query_features"]
        clean_base_coordinates = coordinates
        refinement_input_coordinates = coordinates
        refinement_coordinates: list[torch.Tensor] = []
        projected_features = output["projected_feature_maps"]
        external_features = (
            tuple(refinement_feature_maps)
            if refinement_feature_maps is not None
            else None
        )
        if self.uses_external_refinement_features:
            if external_features is None:
                raise ValueError(
                    "RGB refinement stages require external refinement features"
                )
            if len(external_features) != len(self.refinement_stages):
                raise ValueError(
                    "External refinement feature count must match stage count"
                )
        elif external_features is not None:
            raise ValueError(
                "External refinement features were provided to a legacy head"
            )
        for stage_index, stage in enumerate(self.refinement_stages):
            stage_features: torch.Tensor | Sequence[torch.Tensor] = (
                external_features[stage_index]
                if external_features is not None
                else projected_features
            )
            if self.query_head.state_head_mode == FACTORIZED_STATE_HEAD:
                coordinates, presence, visibility, residual_features = stage(
                    stage_features,
                    queries,
                    coordinates,
                    visibility,
                    presence,
                )
            else:
                coordinates, visibility, residual_features = stage(
                    stage_features,
                    queries,
                    coordinates,
                    visibility,
                )
            queries = queries + residual_features
            refinement_coordinates.append(coordinates)
        output["base_coordinates"] = clean_base_coordinates
        output["refinement_input_coordinates"] = refinement_input_coordinates
        output["coordinates"] = coordinates
        output["decoded_coordinates"] = coordinates
        output["visibility_logits"] = visibility
        if presence is not None:
            output["presence_logits"] = presence
            output["state_probabilities"] = factorized_state_probabilities(
                presence,
                visibility,
            )
        output["final_query_features"] = queries
        output["refinement_coordinates"] = refinement_coordinates
        if external_features is not None:
            output["rgb_refinement_feature_maps"] = external_features
        return output
