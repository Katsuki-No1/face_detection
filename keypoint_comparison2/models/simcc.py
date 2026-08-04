from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SimCCSpec:
    input_size: tuple[int, int] = (256, 256)
    split_ratio: float = 2.0
    sigma: tuple[float, float] = (5.66, 5.66)

    @property
    def x_bins(self) -> int:
        return int(round(self.input_size[0] * self.split_ratio))

    @property
    def y_bins(self) -> int:
        return int(round(self.input_size[1] * self.split_ratio))


class SimCCLabel:
    """Encode continuous five-point coordinates as independent 1-D Gaussians."""

    def __init__(self, spec: SimCCSpec) -> None:
        self.spec = spec

    def encode(
        self,
        coordinates: torch.Tensor,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(f"coordinates must be [B,K,2], got {tuple(coordinates.shape)}")
        device, dtype = coordinates.device, coordinates.dtype
        x_axis = torch.arange(self.spec.x_bins, device=device, dtype=dtype)
        y_axis = torch.arange(self.spec.y_bins, device=device, dtype=dtype)
        mu_x = coordinates[..., 0].clamp(0, 1) * (self.spec.x_bins - 1)
        mu_y = coordinates[..., 1].clamp(0, 1) * (self.spec.y_bins - 1)
        sigma_x = self.spec.sigma[0] * self.spec.split_ratio
        sigma_y = self.spec.sigma[1] * self.spec.split_ratio
        label_x = torch.exp(-0.5 * ((x_axis - mu_x.unsqueeze(-1)) / sigma_x) ** 2)
        label_y = torch.exp(-0.5 * ((y_axis - mu_y.unsqueeze(-1)) / sigma_y) ** 2)
        weights = visible.to(dtype=dtype)
        label_x = label_x / label_x.sum(-1, keepdim=True).clamp_min(1e-8)
        label_y = label_y / label_y.sum(-1, keepdim=True).clamp_min(1e-8)
        return label_x, label_y, weights

    def decode(self, logits_x: torch.Tensor, logits_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability_x = logits_x.softmax(-1)
        probability_y = logits_y.softmax(-1)
        axis_x = torch.linspace(0, 1, logits_x.shape[-1], device=logits_x.device, dtype=logits_x.dtype)
        axis_y = torch.linspace(0, 1, logits_y.shape[-1], device=logits_y.device, dtype=logits_y.dtype)
        coordinates = torch.stack(
            ((probability_x * axis_x).sum(-1), (probability_y * axis_y).sum(-1)),
            dim=-1,
        )
        confidence = torch.sqrt(probability_x.amax(-1) * probability_y.amax(-1))
        return coordinates, confidence


class ScaleNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(dimension), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.scale / x.norm(dim=-1, keepdim=True).clamp_min(self.eps))


class RTMCCBlock(nn.Module):
    """Gated attention unit used by the RTMCC coordinate-classification head."""

    def __init__(
        self,
        dimension: int,
        *,
        expansion_factor: float = 2.0,
        attention_dimension: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        expanded = int(round(dimension * expansion_factor))
        self.norm = ScaleNorm(dimension)
        self.uv = nn.Linear(dimension, expanded * 2)
        self.qk = nn.Linear(dimension, attention_dimension * 2)
        self.gamma = nn.Parameter(torch.ones(2, attention_dimension))
        self.beta = nn.Parameter(torch.zeros(2, attention_dimension))
        self.out = nn.Linear(expanded, dimension)
        self.dropout = nn.Dropout(dropout)
        self.attention_dimension = attention_dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = self.norm(inputs)
        u, v = self.uv(x).chunk(2, dim=-1)
        q, k = self.qk(x).chunk(2, dim=-1)
        q = q * self.gamma[0] + self.beta[0]
        k = k * self.gamma[1] + self.beta[1]
        attention = F.relu(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_dimension)).square()
        attention = self.dropout(attention / attention.sum(-1, keepdim=True).clamp_min(1e-6))
        gated = F.silu(u) * torch.matmul(attention, F.silu(v))
        return residual + self.dropout(self.out(gated))


class RTMCCHead(nn.Module):
    """Five-point RTMCC head returning SimCC logits, coordinates, and visibility."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_keypoints: int = 5,
        input_size: tuple[int, int] = (256, 256),
        featuremap_size: tuple[int, int] = (8, 8),
        hidden_dimension: int = 256,
        split_ratio: float = 2.0,
        sigma: tuple[float, float] = (5.66, 5.66),
        final_kernel_size: int = 7,
        gau_attention_dimension: int = 128,
        gau_expansion_factor: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_keypoints != 5:
            raise ValueError("This comparison fixes num_keypoints=5")
        self.num_keypoints = num_keypoints
        self.featuremap_size = tuple(int(value) for value in featuremap_size)
        self.spec = SimCCSpec(input_size=input_size, split_ratio=split_ratio, sigma=sigma)
        padding = final_kernel_size // 2
        self.final_layer = nn.Conv2d(
            in_channels,
            num_keypoints,
            kernel_size=final_kernel_size,
            stride=1,
            padding=padding,
        )
        flattened = self.featuremap_size[0] * self.featuremap_size[1]
        self.mlp = nn.Sequential(nn.Linear(flattened, hidden_dimension), ScaleNorm(hidden_dimension))
        self.gau = RTMCCBlock(
            hidden_dimension,
            expansion_factor=gau_expansion_factor,
            attention_dimension=gau_attention_dimension,
            dropout=dropout,
        )
        self.cls_x = nn.Linear(hidden_dimension, self.spec.x_bins, bias=False)
        self.cls_y = nn.Linear(hidden_dimension, self.spec.y_bins, bias=False)
        self.visibility = nn.Linear(hidden_dimension, 1)
        self.codec = SimCCLabel(self.spec)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = F.adaptive_avg_pool2d(feature, self.featuremap_size)
        token = self.final_layer(feature).flatten(2)
        token = self.gau(self.mlp(token))
        logits_x = self.cls_x(token)
        logits_y = self.cls_y(token)
        coordinates, simcc_confidence = self.codec.decode(logits_x, logits_y)
        return {
            "simcc_x": logits_x,
            "simcc_y": logits_y,
            "coordinates": coordinates,
            "simcc_confidence": simcc_confidence,
            "visibility_logits": self.visibility(token).squeeze(-1),
        }


class SimCCLoss(nn.Module):
    def __init__(self, codec: SimCCLabel, *, beta: float = 10.0, visibility_weight: float = 0.2) -> None:
        super().__init__()
        self.codec = codec
        self.beta = float(beta)
        self.visibility_weight = float(visibility_weight)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        visible: torch.Tensor,
        labeled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        coordinate_mask = visible if labeled is None else labeled
        label_x, label_y, weight = self.codec.encode(coordinates, coordinate_mask)
        loss_x = F.kl_div(
            F.log_softmax(output["simcc_x"] * self.beta, dim=-1),
            label_x,
            reduction="none",
        ).sum(-1)
        loss_y = F.kl_div(
            F.log_softmax(output["simcc_y"] * self.beta, dim=-1),
            label_y,
            reduction="none",
        ).sum(-1)
        denominator = weight.sum().clamp_min(1.0)
        loss_simcc = ((loss_x + loss_y) * weight).sum() / denominator
        visibility_error = F.binary_cross_entropy_with_logits(
            output["visibility_logits"],
            visible.to(dtype=output["visibility_logits"].dtype),
            reduction="none",
        )
        loss_visibility = (visibility_error * weight).sum() / denominator
        coordinate_error = F.smooth_l1_loss(
            output["coordinates"],
            coordinates,
            reduction="none",
        ).sum(-1)
        loss_coordinate = (coordinate_error * weight).sum() / denominator
        total = loss_simcc + 0.5 * loss_coordinate + self.visibility_weight * loss_visibility
        return {
            "loss_total": total,
            "loss_simcc": loss_simcc,
            "loss_coordinate": loss_coordinate,
            "loss_visibility": loss_visibility,
        }
