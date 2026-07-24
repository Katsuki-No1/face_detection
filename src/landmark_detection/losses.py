from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def stack_targets(targets: Sequence[dict[str, Any]], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([target[key] for target in targets], dim=0).to(device, non_blocking=True)


def wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    width: float = 0.10,
    curvature: float = 0.02,
) -> torch.Tensor:
    difference = (prediction - target).abs()
    constant = width - width * torch.log(torch.tensor(1.0 + width / curvature, device=difference.device))
    value = torch.where(
        difference < width,
        width * torch.log1p(difference / curvature),
        difference - constant,
    )
    expanded = mask[..., None].expand_as(value)
    if not bool(expanded.any()):
        return prediction.sum() * 0.0
    return value[expanded].mean()


def gaussian_heatmap_targets(
    coordinates: torch.Tensor,
    visible: torch.Tensor,
    *,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    ys = torch.arange(height, device=coordinates.device, dtype=coordinates.dtype).reshape(1, 1, height, 1)
    xs = torch.arange(width, device=coordinates.device, dtype=coordinates.dtype).reshape(1, 1, 1, width)
    center_x = coordinates[..., 0].clamp(0.0, 1.0).mul(width - 1).unsqueeze(-1).unsqueeze(-1)
    center_y = coordinates[..., 1].clamp(0.0, 1.0).mul(height - 1).unsqueeze(-1).unsqueeze(-1)
    heatmaps = torch.exp(-((xs - center_x).square() + (ys - center_y).square()) / (2.0 * sigma * sigma))
    return heatmaps * visible[..., None, None].to(dtype=heatmaps.dtype)


class LandmarkLoss(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.heatmap_weight = float(cfg.get("heatmap_weight", 1.0))
        self.coordinate_weight = float(cfg.get("coordinate_weight", 2.0))
        self.visibility_weight = float(cfg.get("visibility_weight", 0.10))
        self.auxiliary_weight = float(cfg.get("auxiliary_weight", 0.50))
        self.refinement_auxiliary_weight = float(cfg.get("refinement_auxiliary_weight", 0.75))
        self.heatmap_sigma = float(cfg.get("heatmap_sigma", 1.5))
        self.wing_width = float(cfg.get("wing_width", 0.10))
        self.wing_curvature = float(cfg.get("wing_curvature", 0.02))

    def _coordinate_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        return wing_loss(
            prediction,
            target,
            visible,
            width=self.wing_width,
            curvature=self.wing_curvature,
        )

    def per_landmark_coordinate_losses(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor,
    ) -> list[torch.Tensor]:
        losses = []
        for index in range(prediction.shape[1]):
            losses.append(self._coordinate_loss(prediction[:, index : index + 1], target[:, index : index + 1], visible[:, index : index + 1]))
        return losses

    def forward(self, output: dict[str, Any], targets: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        device = output["coordinates"].device
        coordinates = stack_targets(targets, "keypoints", device)
        visible = stack_targets(targets, "visible", device).bool()
        losses: dict[str, torch.Tensor] = {}
        losses["loss_coordinates"] = self._coordinate_loss(output["coordinates"], coordinates, visible)
        losses["loss_visibility"] = F.binary_cross_entropy_with_logits(
            output["visibility_logits"],
            visible.to(dtype=output["visibility_logits"].dtype),
        )
        total = self.coordinate_weight * losses["loss_coordinates"] + self.visibility_weight * losses["loss_visibility"]

        if "heatmaps" in output:
            heatmaps = output["heatmaps"]
            target_heatmaps = gaussian_heatmap_targets(
                coordinates,
                visible,
                height=heatmaps.shape[-2],
                width=heatmaps.shape[-1],
                sigma=self.heatmap_sigma,
            )
            # ViTPose/SimpleBaseline heatmap regression operates on raw logits.
            # A sigmoid would initialize every background pixel around 0.5 and
            # let the overwhelming background dominate the sparse Gaussian.
            per_pixel = (heatmaps - target_heatmaps).square()
            landmark_weight = visible[..., None, None].to(dtype=per_pixel.dtype)
            denominator = landmark_weight.sum().clamp_min(1.0) * heatmaps.shape[-2] * heatmaps.shape[-1]
            losses["loss_heatmap"] = (per_pixel * landmark_weight).sum() / denominator
            total = total + self.heatmap_weight * losses["loss_heatmap"]

        auxiliary = list(output.get("auxiliary_coordinates", []))
        if auxiliary:
            auxiliary_losses = [self._coordinate_loss(value, coordinates, visible) for value in auxiliary]
            losses["loss_auxiliary"] = torch.stack(auxiliary_losses).mean()
            total = total + self.auxiliary_weight * losses["loss_auxiliary"]

        refinement = list(output.get("refinement_coordinates", []))
        if refinement:
            refinement_losses = [self._coordinate_loss(value, coordinates, visible) for value in refinement]
            weights = torch.linspace(
                0.5,
                1.0,
                len(refinement_losses),
                device=device,
                dtype=refinement_losses[0].dtype,
            )
            losses["loss_refinement_auxiliary"] = (
                torch.stack(refinement_losses) * weights
            ).sum() / weights.sum()
            total = total + self.refinement_auxiliary_weight * losses["loss_refinement_auxiliary"]
        losses["loss_total"] = total
        return losses
