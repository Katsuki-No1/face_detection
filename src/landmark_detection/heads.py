from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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
    ) -> None:
        super().__init__()
        if deconv_layers < 1:
            raise ValueError("deconv_layers must be positive")
        if deconv_kernel not in {2, 4}:
            raise ValueError("deconv_kernel must be 2 or 4")
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
        return {
            "heatmaps": heatmaps,
            "coordinates": soft_argmax_2d(heatmaps, temperature=self.softargmax_temperature),
            "decoded_coordinates": decode_heatmap_argmax(heatmaps),
            "visibility_logits": self.visibility_head(feature_map),
            "auxiliary_coordinates": [],
            "self_attention": [],
            "cross_attention": [],
        }


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
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normalized = self.norm1(queries)
        self_result, self_weights = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        queries = queries + self.dropout(self_result)
        normalized = self.norm2(queries)
        cross_result, cross_weights = self.cross_attention(
            normalized,
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
    ) -> None:
        super().__init__()
        self.num_landmarks = int(num_landmarks)
        self.dimension = int(dimension)
        self.memory_projection = nn.Conv2d(in_channels, dimension, kernel_size=1)
        self.landmark_queries = nn.Parameter(torch.empty(1, num_landmarks, dimension))
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
        self.auxiliary_supervision = bool(auxiliary_supervision)
        nn.init.normal_(self.landmark_queries, std=0.02)
        for head in self.coordinate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, feature_map: torch.Tensor, *, return_attention: bool = False) -> dict[str, Any]:
        projected = self.memory_projection(feature_map)
        batch, _, height, width = projected.shape
        memory = projected.flatten(2).transpose(1, 2).contiguous()
        position = sine_position_encoding_2d(
            height,
            width,
            self.dimension,
            device=projected.device,
            dtype=projected.dtype,
        )
        queries = self.landmark_queries.expand(batch, -1, -1)
        coordinates: list[torch.Tensor] = []
        self_attention: list[torch.Tensor] = []
        cross_attention: list[torch.Tensor] = []
        for layer, coordinate_head in zip(self.layers, self.coordinate_heads, strict=True):
            queries, self_weights, cross_weights = layer(
                queries,
                memory,
                memory_position=position,
                return_attention=return_attention,
            )
            coordinates.append(torch.sigmoid(coordinate_head(queries)))
            if self_weights is not None:
                self_attention.append(self_weights)
            if cross_weights is not None:
                cross_attention.append(cross_weights)
        return {
            "coordinates": coordinates[-1],
            "decoded_coordinates": coordinates[-1],
            "visibility_logits": self.visibility_head(queries).squeeze(-1),
            "auxiliary_coordinates": coordinates[:-1] if self.auxiliary_supervision else [],
            "query_features": queries,
            "projected_feature_map": projected,
            "self_attention": self_attention,
            "cross_attention": cross_attention,
        }


class CoordinateRefinementStage(nn.Module):
    """Coordinate-conditioned local sampling followed by bounded residual correction."""

    def __init__(
        self,
        dimension: int,
        *,
        radius_in_feature_cells: float,
        maximum_delta: float,
        samples_per_axis: int = 3,
    ) -> None:
        super().__init__()
        if samples_per_axis < 1 or samples_per_axis % 2 == 0:
            raise ValueError("samples_per_axis must be a positive odd integer")
        self.radius_in_feature_cells = float(radius_in_feature_cells)
        self.maximum_delta = float(maximum_delta)
        self.samples_per_axis = int(samples_per_axis)
        local_dimension = dimension * samples_per_axis * samples_per_axis
        self.mlp = nn.Sequential(
            nn.Linear(dimension + local_dimension + 2, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(dimension, 2)
        self.visibility_delta = nn.Linear(dimension, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.visibility_delta.weight)
        nn.init.zeros_(self.visibility_delta.bias)

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
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled.reshape(batch, coordinates.shape[1], -1)

    def forward(
        self,
        feature_map: torch.Tensor,
        query_features: torch.Tensor,
        coordinates: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self._sample(feature_map, coordinates)
        hidden = self.mlp(torch.cat([query_features, local, coordinates], dim=-1))
        delta = torch.tanh(self.delta_head(hidden)) * self.maximum_delta
        refined = (coordinates + delta).clamp(0.0, 1.0)
        visibility = visibility_logits + self.visibility_delta(hidden).squeeze(-1)
        return refined, visibility, hidden


class LandmarkQueryRefinementHead(nn.Module):
    def __init__(self, query_head: LandmarkQueryHead, *, stages: list[dict[str, float]]) -> None:
        super().__init__()
        self.query_head = query_head
        self.refinement_stages = nn.ModuleList(
            [
                CoordinateRefinementStage(
                    query_head.dimension,
                    radius_in_feature_cells=float(stage["radius_in_feature_cells"]),
                    maximum_delta=float(stage["maximum_delta"]),
                    samples_per_axis=int(stage.get("samples_per_axis", 3)),
                )
                for stage in stages
            ]
        )

    def forward(self, feature_map: torch.Tensor, *, return_attention: bool = False) -> dict[str, Any]:
        output = self.query_head(feature_map, return_attention=return_attention)
        coordinates = output["coordinates"]
        visibility = output["visibility_logits"]
        queries = output["query_features"]
        refinement_coordinates: list[torch.Tensor] = []
        for stage in self.refinement_stages:
            coordinates, visibility, residual_features = stage(
                output["projected_feature_map"],
                queries,
                coordinates,
                visibility,
            )
            queries = queries + residual_features
            refinement_coordinates.append(coordinates)
        output["base_coordinates"] = output["coordinates"]
        output["coordinates"] = coordinates
        output["decoded_coordinates"] = coordinates
        output["visibility_logits"] = visibility
        output["refinement_coordinates"] = refinement_coordinates
        return output
