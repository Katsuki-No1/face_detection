from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.target_assigner import CenterFaceTargetAssigner


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    normalizer: torch.Tensor | float | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
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


class DinoSCRFDLoss(nn.Module):
    """Classification, bbox, and 5-point landmark losses for DinoSCRFD."""

    def __init__(
        self,
        *,
        assigner: CenterFaceTargetAssigner | None = None,
        cls_weight: float = 1.0,
        bbox_weight: float = 2.0,
        kpt_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.assigner = assigner or CenterFaceTargetAssigner()
        self.cls_weight = float(cls_weight)
        self.bbox_weight = float(bbox_weight)
        self.kpt_weight = float(kpt_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

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

        loss_cls = predictions["cls_logits"][0].sum() * 0.0
        loss_bbox = predictions["bbox_reg"][0].sum() * 0.0
        loss_kpt = predictions["kpt_reg"][0].sum() * 0.0

        cls_targets = assignments["cls_targets"]
        bbox_targets = assignments["bbox_targets"]
        kpt_targets = assignments["kpt_targets"]
        positive_masks = assignments["positive_masks"]
        kpt_masks = assignments["kpt_masks"]
        assert isinstance(cls_targets, list)
        assert isinstance(bbox_targets, list)
        assert isinstance(kpt_targets, list)
        assert isinstance(positive_masks, list)
        assert isinstance(kpt_masks, list)

        for logits, target in zip(predictions["cls_logits"], cls_targets):
            loss_cls = loss_cls + sigmoid_focal_loss(
                logits,
                target,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                normalizer=positive_count,
            )

        for pred_bbox, target_bbox, pos_mask in zip(predictions["bbox_reg"], bbox_targets, positive_masks):
            loss_bbox = loss_bbox + _masked_smooth_l1(pred_bbox, target_bbox, pos_mask, positive_count)

        for pred_kpt, target_kpt, kpt_mask in zip(predictions["kpt_reg"], kpt_targets, kpt_masks):
            loss_kpt = loss_kpt + _masked_smooth_l1(pred_kpt, target_kpt, kpt_mask, visible_kpt_count)

        level_count = max(1, len(predictions["cls_logits"]))
        loss_cls = loss_cls / level_count
        loss_bbox = loss_bbox / level_count
        loss_kpt = loss_kpt / level_count
        loss_total = self.cls_weight * loss_cls + self.bbox_weight * loss_bbox + self.kpt_weight * loss_kpt
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_kpt": loss_kpt,
            "loss_total": loss_total,
            "num_pos": positive_count.detach().to(torch.float32),
        }
