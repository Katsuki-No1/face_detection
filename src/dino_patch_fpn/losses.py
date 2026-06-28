from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .targets import DenseTargetAssigner


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    expanded_mask = mask.expand_as(prediction)
    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(prediction[expanded_mask], target[expanded_mask], reduction="mean")
    elif loss_type == "l1":
        loss = F.l1_loss(prediction[expanded_mask], target[expanded_mask], reduction="mean")
    else:
        raise ValueError(f"Unsupported regression loss_type: {loss_type}")
    return loss


class DenseFaceKeypointLoss(nn.Module):
    """Minimal loss entry point for future training.

    The default assigner intentionally creates no positive samples yet. This
    lets shape tests and dummy forward passes exercise the loss API without
    pretending that real target assignment is implemented.
    """

    def __init__(
        self,
        *,
        assigner: DenseTargetAssigner | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bbox_loss: str = "l1",
        keypoint_loss: str = "smooth_l1",
        ellipse_loss: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.assigner = assigner or DenseTargetAssigner()
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.bbox_loss = bbox_loss
        self.keypoint_loss = keypoint_loss
        self.ellipse_loss = ellipse_loss

    def forward(
        self,
        predictions: dict[str, dict[str, torch.Tensor]],
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        assignments = self.assigner.assign(predictions, targets)
        first_tensor = next(iter(next(iter(predictions.values())).values()))
        loss_cls = first_tensor.sum() * 0.0
        loss_bbox = first_tensor.sum() * 0.0
        loss_centerness = first_tensor.sum() * 0.0
        loss_keypoints = first_tensor.sum() * 0.0
        loss_ellipse = first_tensor.sum() * 0.0

        for level, level_predictions in predictions.items():
            level_targets = assignments[level]
            loss_cls = loss_cls + sigmoid_focal_loss(
                level_predictions["cls_logits"],
                level_targets["cls_targets"],
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            loss_centerness = loss_centerness + F.binary_cross_entropy_with_logits(
                level_predictions["centerness"],
                level_targets["centerness_targets"],
                reduction="mean",
            )
            loss_bbox = loss_bbox + _masked_regression_loss(
                level_predictions["bbox_reg"],
                level_targets["bbox_targets"],
                level_targets["positive_mask"],
                loss_type=self.bbox_loss,
            )
            loss_keypoints = loss_keypoints + _masked_regression_loss(
                level_predictions["keypoints"],
                level_targets["keypoint_targets"],
                level_targets["keypoint_mask"],
                loss_type=self.keypoint_loss,
            )
            if "ellipse" in level_predictions:
                loss_ellipse = loss_ellipse + _masked_regression_loss(
                    level_predictions["ellipse"],
                    level_targets["ellipse_targets"],
                    level_targets["positive_mask"],
                    loss_type=self.ellipse_loss,
                )

        level_count = max(1, len(predictions))
        losses = {
            "loss_cls": loss_cls / level_count,
            "loss_bbox": loss_bbox / level_count,
            "loss_centerness": loss_centerness / level_count,
            "loss_keypoints": loss_keypoints / level_count,
            "loss_ellipse": loss_ellipse / level_count,
        }
        losses["loss_total"] = sum(losses.values())
        return losses
