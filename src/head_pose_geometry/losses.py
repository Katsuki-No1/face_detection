from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .rotations import doubled_angle_distance, geodesic_rotation_distance


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.to(value)
    while effective.ndim < value.ndim:
        effective = effective.unsqueeze(-1)
    if weight is not None:
        weights = weight.to(value)
        while weights.ndim < value.ndim:
            weights = weights.unsqueeze(-1)
        effective = effective * weights
    return (value * effective).sum() / effective.sum().clamp_min(1.0)


@dataclass(frozen=True)
class GeometryLossWeights:
    rotation: float = 1.0
    projected_keypoints: float = 5.0
    amodal_ellipse_center: float = 2.0
    amodal_ellipse_axes: float = 2.0
    amodal_ellipse_angle: float = 1.0
    head_relative: float = 1.0
    observability: float = 1.0
    visibility: float = 0.5
    scale_prior: float = 0.05
    shape_prior: float = 0.01
    uncertainty: float = 0.05
    template_3d: float = 1.0
    scale_translation: float = 1.0
    augmentation_consistency: float = 0.5


class HeadPoseGeometryLoss(nn.Module):
    """Masked synthetic/real objective for the pose and geometry branch."""

    component_names = (
        "loss_pose_rotation",
        "loss_geometry_keypoints",
        "loss_geometry_ellipse_center",
        "loss_geometry_ellipse_axes",
        "loss_geometry_ellipse_angle",
        "loss_geometry_head_relative",
        "loss_geometry_observability",
        "loss_geometry_visibility",
        "loss_geometry_scale_prior",
        "loss_geometry_shape_prior",
        "loss_geometry_uncertainty",
        "loss_geometry_template_3d",
        "loss_geometry_scale_translation",
        "loss_geometry_augmentation_consistency",
    )

    def __init__(
        self,
        *,
        weights: GeometryLossWeights | None = None,
        minimum_scale: float = 1.0e-4,
        maximum_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.weights = weights or GeometryLossWeights()
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)

    def zero_losses(
        self,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the complete loss contract when no geometry target contributes."""

        zero = reference.sum() * 0.0
        losses = {name: zero for name in self.component_names}
        losses["loss_head_pose_geometry"] = zero
        return losses

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        reference = output["rotation_matrix"]
        count = int(reference.shape[0])
        zero = reference.sum() * 0.0
        pose_mask = target.get(
            "pose_supervision_mask",
            torch.zeros(count, dtype=torch.bool, device=reference.device),
        ).to(device=reference.device, dtype=torch.bool)
        pose_weight = target.get(
            "pose_weight",
            torch.ones(count, device=reference.device, dtype=reference.dtype),
        ).to(reference)
        rotation_target = target.get("rotation_matrix")
        if rotation_target is not None and bool(pose_mask.any()):
            rotation_error = geodesic_rotation_distance(
                output["rotation_matrix"],
                rotation_target.to(reference),
            )
            loss_rotation = _masked_mean(
                rotation_error,
                pose_mask,
                weight=pose_weight,
            )
        else:
            rotation_error = reference.new_zeros((count,))
            loss_rotation = zero

        ellipse_target = target.get("ellipses")
        ellipse_mask = target.get(
            "ellipse_mask",
            torch.ones(count, dtype=torch.bool, device=reference.device),
        ).to(device=reference.device, dtype=torch.bool)
        predicted_ellipse = output["amodal_ellipse"]
        if ellipse_target is not None and bool(ellipse_mask.any()):
            ellipse_target = ellipse_target.to(predicted_ellipse)
            loss_center = _masked_mean(
                F.smooth_l1_loss(
                    predicted_ellipse[:, :2],
                    ellipse_target[:, :2],
                    reduction="none",
                ),
                ellipse_mask,
            )
            loss_axes = _masked_mean(
                F.smooth_l1_loss(
                    torch.log(predicted_ellipse[:, 2:4].clamp_min(1.0e-6)),
                    torch.log(ellipse_target[:, 2:4].clamp_min(1.0e-6)),
                    reduction="none",
                ),
                ellipse_mask,
            )
            angle_error = doubled_angle_distance(
                output["ellipse_angle_vector"],
                ellipse_target[:, 4],
            )
            angle_reliability = output.get(
                "ellipse_angle_reliability",
                torch.ones_like(angle_error),
            )
            target_ratio = ellipse_target[:, 2] / ellipse_target[:, 3].clamp_min(1.0e-6)
            target_reliability = ((target_ratio - 1.0) / 0.10).clamp(0.0, 1.0)
            loss_angle = _masked_mean(
                angle_error,
                ellipse_mask,
                weight=angle_reliability * target_reliability,
            )
        else:
            loss_center = loss_axes = loss_angle = zero

        keypoint_target = target.get("keypoints")
        labeled = target.get(
            "keypoint_labeled",
            torch.zeros(
                (count, output["projected_keypoints"].shape[1]),
                dtype=torch.bool,
                device=reference.device,
            ),
        ).to(device=reference.device, dtype=torch.bool)
        if keypoint_target is not None and bool(labeled.any()):
            difference = F.smooth_l1_loss(
                output["projected_keypoints"],
                keypoint_target.to(output["projected_keypoints"]),
                reduction="none",
            ).sum(dim=-1)
            loss_keypoints = _masked_mean(difference, labeled)
        else:
            loss_keypoints = zero

        head_boxes = target.get("head_boxes_xyxy")
        if (
            head_boxes is not None
            and ellipse_target is not None
            and bool(ellipse_mask.any())
        ):
            head_boxes = head_boxes.to(predicted_ellipse)
            head_size = (head_boxes[:, 2:] - head_boxes[:, :2]).clamp_min(1.0e-6)
            head_center = 0.5 * (head_boxes[:, :2] + head_boxes[:, 2:])
            predicted_relative = (predicted_ellipse[:, :2] - head_center) / head_size
            target_relative = (ellipse_target[:, :2] - head_center) / head_size
            relative_error = F.smooth_l1_loss(
                predicted_relative,
                target_relative,
                reduction="none",
            )
            loss_head_relative = _masked_mean(relative_error, ellipse_mask)
        else:
            loss_head_relative = zero

        observability_target = target.get("face_observability")
        if observability_target is None:
            observability_target = labeled.any(dim=-1).to(reference)
        observability_mask = target.get(
            "observability_mask",
            # Absence of all keypoints is ambiguous (rear view, blur,
            # occlusion, scale, or annotation policy). It is not a negative
            # observability label unless the dataset explicitly supplies one.
            labeled.any(dim=-1),
        ).to(device=reference.device, dtype=torch.bool)
        observability_logits = output["face_observability_logits"]
        observability_error = F.binary_cross_entropy_with_logits(
            observability_logits,
            observability_target.to(observability_logits),
            reduction="none",
        )
        loss_observability = _masked_mean(
            observability_error,
            observability_mask,
        )

        visible_target = target.get("keypoint_visible")
        if visible_target is not None and bool(labeled.any()):
            visibility_logits = output["keypoint_visibility_logits"]
            visibility_error = F.binary_cross_entropy_with_logits(
                visibility_logits,
                visible_target.to(visibility_logits),
                reduction="none",
            )
            loss_visibility = _masked_mean(visibility_error, labeled)
        else:
            loss_visibility = zero

        template_target = target.get("template_vertices")
        template_mask = target.get(
            "template_supervision_mask",
            torch.ones(count, dtype=torch.bool, device=reference.device),
        ).to(device=reference.device, dtype=torch.bool)
        if template_target is not None and bool(template_mask.any()):
            template_error = F.smooth_l1_loss(
                output["rotated_vertices"],
                template_target.to(output["rotated_vertices"]),
                reduction="none",
            )
            loss_template = _masked_mean(template_error, template_mask)
        else:
            loss_template = zero
        shape_target = target.get("shape_code")
        if (
            shape_target is not None
            and output.get("shape_code") is not None
            and output["shape_code"].numel()
            and bool(template_mask.any())
        ):
            loss_template = loss_template + _masked_mean(
                F.smooth_l1_loss(
                    output["shape_code"],
                    shape_target.to(output["shape_code"]),
                    reduction="none",
                ),
                template_mask,
            )

        provided_scale = target.get("weak_perspective_scale")
        provided_translation = target.get("translation")
        projection_parameter_mask = target.get(
            "projection_parameter_mask",
            torch.ones(count, dtype=torch.bool, device=reference.device),
        ).to(device=reference.device, dtype=torch.bool)
        projection_parameter_terms = []
        if provided_scale is not None and bool(projection_parameter_mask.any()):
            projection_parameter_terms.append(
                _masked_mean(
                    F.smooth_l1_loss(
                        torch.log(output["weak_perspective_scale"].clamp_min(1.0e-6)),
                        torch.log(
                            provided_scale.to(
                                output["weak_perspective_scale"]
                            ).clamp_min(1.0e-6)
                        ),
                        reduction="none",
                    ),
                    projection_parameter_mask,
                )
            )
        if provided_translation is not None and bool(projection_parameter_mask.any()):
            projection_parameter_terms.append(
                _masked_mean(
                    F.smooth_l1_loss(
                        output["translation"],
                        provided_translation.to(output["translation"]),
                        reduction="none",
                    ),
                    projection_parameter_mask,
                )
            )
        loss_scale_translation = (
            sum(projection_parameter_terms[1:], projection_parameter_terms[0])
            if projection_parameter_terms
            else zero
        )

        consistency_rotation = output.get("consistency_rotation_matrix")
        if consistency_rotation is not None:
            consistency_terms = [
                geodesic_rotation_distance(
                    output["rotation_matrix"],
                    consistency_rotation.detach().to(reference),
                ).mean()
            ]
            consistency_ellipse = output.get("consistency_amodal_ellipse")
            if consistency_ellipse is not None:
                consistency_ellipse = consistency_ellipse.detach().to(predicted_ellipse)
                consistency_terms.extend(
                    (
                        F.smooth_l1_loss(
                            predicted_ellipse[:, :2],
                            consistency_ellipse[:, :2],
                        ),
                        F.smooth_l1_loss(
                            torch.log(predicted_ellipse[:, 2:4].clamp_min(1.0e-6)),
                            torch.log(consistency_ellipse[:, 2:4].clamp_min(1.0e-6)),
                        ),
                        doubled_angle_distance(
                            output["ellipse_angle_vector"],
                            consistency_ellipse[:, 4],
                        ).mean(),
                    )
                )
            consistency_keypoints = output.get("consistency_projected_keypoints")
            if consistency_keypoints is not None:
                consistency_terms.append(
                    F.smooth_l1_loss(
                        output["projected_keypoints"],
                        consistency_keypoints.detach().to(
                            output["projected_keypoints"]
                        ),
                    )
                )
            loss_consistency = sum(
                consistency_terms[1:],
                consistency_terms[0],
            )
        else:
            loss_consistency = zero

        scale = output["weak_perspective_scale"]
        scale_penalty = (
            F.relu(self.minimum_scale - scale).square()
            + F.relu(scale - self.maximum_scale).square()
        ).mean()
        shape_code = output.get("shape_code")
        shape_prior = (
            shape_code.square().mean()
            if shape_code is not None and shape_code.numel()
            else zero
        )
        combined_error = rotation_error.detach()
        if ellipse_target is not None:
            combined_error = (
                combined_error
                + torch.linalg.vector_norm(
                    predicted_ellipse[:, :2] - ellipse_target[:, :2],
                    dim=-1,
                ).detach()
            )
        uncertainty = output["geometry_uncertainty"]
        uncertainty_nll = (
            torch.exp(-uncertainty) * combined_error + uncertainty
        ).mean()
        pose_uncertainty = output.get("pose_uncertainty")
        if pose_uncertainty is not None and bool(pose_mask.any()):
            pose_uncertainty_nll = _masked_mean(
                torch.exp(-pose_uncertainty) * rotation_error.detach()
                + pose_uncertainty,
                pose_mask,
                weight=pose_weight,
            )
            uncertainty_nll = uncertainty_nll + pose_uncertainty_nll

        weighted = {
            "loss_pose_rotation": loss_rotation * self.weights.rotation,
            "loss_geometry_keypoints": (
                loss_keypoints * self.weights.projected_keypoints
            ),
            "loss_geometry_ellipse_center": (
                loss_center * self.weights.amodal_ellipse_center
            ),
            "loss_geometry_ellipse_axes": (
                loss_axes * self.weights.amodal_ellipse_axes
            ),
            "loss_geometry_ellipse_angle": (
                loss_angle * self.weights.amodal_ellipse_angle
            ),
            "loss_geometry_head_relative": (
                loss_head_relative * self.weights.head_relative
            ),
            "loss_geometry_observability": (
                loss_observability * self.weights.observability
            ),
            "loss_geometry_visibility": (loss_visibility * self.weights.visibility),
            "loss_geometry_scale_prior": (scale_penalty * self.weights.scale_prior),
            "loss_geometry_shape_prior": (shape_prior * self.weights.shape_prior),
            "loss_geometry_uncertainty": (uncertainty_nll * self.weights.uncertainty),
            "loss_geometry_template_3d": (loss_template * self.weights.template_3d),
            "loss_geometry_scale_translation": (
                loss_scale_translation * self.weights.scale_translation
            ),
            "loss_geometry_augmentation_consistency": (
                loss_consistency * self.weights.augmentation_consistency
            ),
        }
        weighted["loss_head_pose_geometry"] = sum(weighted.values())
        return weighted


def geometry_loss_weights_from_mapping(
    raw: dict[str, Any] | None,
) -> GeometryLossWeights:
    values = dict(raw or {})
    return GeometryLossWeights(
        **{
            field: float(values.get(field, getattr(GeometryLossWeights(), field)))
            for field in GeometryLossWeights.__dataclass_fields__
        }
    )


__all__ = [
    "GeometryLossWeights",
    "HeadPoseGeometryLoss",
    "geometry_loss_weights_from_mapping",
]
