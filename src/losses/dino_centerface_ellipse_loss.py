from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ellipse_target_assigner import CenterFaceEllipseTargetAssigner


def center_heatmap_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    normalizer: torch.Tensor | float | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = targets.eq(1.0).to(logits.dtype)
    negative = targets.lt(1.0).to(logits.dtype)
    negative_weights = (1.0 - targets).pow(beta)
    positive_loss = -torch.log(probability) * (1.0 - probability).pow(alpha) * positive
    negative_loss = -torch.log(1.0 - probability) * probability.pow(alpha) * negative_weights * negative
    loss = positive_loss + negative_loss
    if normalizer is None:
        return loss.mean()
    if not isinstance(normalizer, torch.Tensor):
        normalizer = logits.new_tensor(float(normalizer))
    return loss.sum() / normalizer.clamp_min(1.0)


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalizer: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    expanded_mask = mask.expand_as(prediction)
    loss = F.smooth_l1_loss(prediction[expanded_mask], target[expanded_mask], reduction="sum")
    return loss / normalizer.clamp_min(1.0)


def _masked_channel_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalizer: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[1] == 0 or not bool(mask.any().item()):
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction[mask], target[mask], reduction="sum")
    return loss / normalizer.clamp_min(1.0)


class DinoCenterFaceEllipseLoss(nn.Module):
    """Center heatmap, ellipse, keypoint, and optional bbox losses."""

    def __init__(
        self,
        *,
        assigner: CenterFaceEllipseTargetAssigner | None = None,
        center_weight: float = 1.0,
        ellipse_weight: float = 5.0,
        kpt_weight: float = 1.0,
        bbox_weight: float = 0.5,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.assigner = assigner or CenterFaceEllipseTargetAssigner()
        self.center_weight = float(center_weight)
        self.ellipse_weight = float(ellipse_weight)
        self.kpt_weight = float(kpt_weight)
        self.bbox_weight = float(bbox_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)

    def forward(
        self,
        predictions: dict[str, list[torch.Tensor]],
        targets: list[dict[str, Any]] | None,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        assignments = self.assigner.assign(predictions, targets, image_size)
        positive_count = assignments["positive_count"]
        visible_kpt_count = assignments["visible_kpt_count"]
        assert isinstance(positive_count, torch.Tensor)
        assert isinstance(visible_kpt_count, torch.Tensor)

        loss_center = predictions["center_logits"][0].sum() * 0.0
        loss_ellipse = predictions["ellipse_reg"][0].sum() * 0.0
        loss_kpt = predictions["kpt_reg"][0].sum() * 0.0
        bbox_predictions = predictions.get("bbox_reg", [])
        loss_bbox = bbox_predictions[0].sum() * 0.0 if bbox_predictions else loss_center * 0.0

        center_targets = assignments["center_targets"]
        ellipse_targets = assignments["ellipse_targets"]
        kpt_targets = assignments["kpt_targets"]
        bbox_targets = assignments["bbox_targets"]
        positive_masks = assignments["positive_masks"]
        kpt_masks = assignments["kpt_masks"]
        assert isinstance(center_targets, list)
        assert isinstance(ellipse_targets, list)
        assert isinstance(kpt_targets, list)
        assert isinstance(bbox_targets, list)
        assert isinstance(positive_masks, list)
        assert isinstance(kpt_masks, list)

        for logits, target in zip(predictions["center_logits"], center_targets):
            loss_center = loss_center + center_heatmap_focal_loss(
                logits,
                target,
                alpha=self.focal_alpha,
                beta=self.focal_beta,
                normalizer=positive_count,
            )

        for pred_ellipse, target_ellipse, pos_mask in zip(predictions["ellipse_reg"], ellipse_targets, positive_masks):
            loss_ellipse = loss_ellipse + _masked_smooth_l1(pred_ellipse, target_ellipse, pos_mask, positive_count)

        for pred_kpt, target_kpt, kpt_mask in zip(predictions["kpt_reg"], kpt_targets, kpt_masks):
            loss_kpt = loss_kpt + _masked_channel_smooth_l1(pred_kpt, target_kpt, kpt_mask, visible_kpt_count)

        for pred_bbox, target_bbox, pos_mask in zip(bbox_predictions, bbox_targets, positive_masks):
            loss_bbox = loss_bbox + _masked_smooth_l1(pred_bbox, target_bbox, pos_mask, positive_count)

        level_count = max(1, len(predictions["center_logits"]))
        loss_center = loss_center / level_count
        loss_ellipse = loss_ellipse / level_count
        loss_kpt = loss_kpt / level_count
        loss_bbox = loss_bbox / level_count
        loss_total = (
            self.center_weight * loss_center
            + self.ellipse_weight * loss_ellipse
            + self.kpt_weight * loss_kpt
            + self.bbox_weight * loss_bbox
        )
        return {
            "loss_center": loss_center,
            "loss_ellipse": loss_ellipse,
            "loss_kpt": loss_kpt,
            "loss_bbox": loss_bbox,
            "loss_total": loss_total,
            "num_pos": positive_count.detach().to(torch.float32),
        }
