from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .rotations import doubled_angle_vector


@dataclass
class FusionEvidence:
    pose_uncertainty: torch.Tensor
    geometry_uncertainty: torch.Tensor
    direct_ellipse_uncertainty: torch.Tensor
    mean_keypoint_visibility: torch.Tensor
    visible_keypoint_fraction: torch.Tensor
    face_observability: torch.Tensor
    truncation_ratio: torch.Tensor
    minimum_border_distance: torch.Tensor
    head_confidence: torch.Tensor

    def tensor(self, *, like: torch.Tensor) -> torch.Tensor:
        values = (
            self.pose_uncertainty,
            self.geometry_uncertainty,
            self.direct_ellipse_uncertainty,
            self.mean_keypoint_visibility,
            self.visible_keypoint_fraction,
            self.face_observability,
            self.truncation_ratio,
            self.minimum_border_distance,
            self.head_confidence,
        )
        return torch.stack(
            [value.to(like).reshape(-1) for value in values],
            dim=-1,
        )


class UncertaintyGatedGeometryFusion(nn.Module):
    """Residual geometry proposal fused through uncertainty-aware gates."""

    evidence_dimension = 9

    def __init__(
        self,
        pose_embedding_dimension: int,
        *,
        hidden_dimension: int = 128,
        keypoints: int = 5,
    ) -> None:
        super().__init__()
        self.keypoints = int(keypoints)
        input_dimension = pose_embedding_dimension + self.evidence_dimension
        correction_dimension = 2 + 2 + 2 + 2 * self.keypoints
        self.correction = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, correction_dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 2),
        )
        correction_output = self.correction[-1]
        gate_output = self.gate[-1]
        assert isinstance(correction_output, nn.Linear)
        assert isinstance(gate_output, nn.Linear)
        nn.init.zeros_(correction_output.weight)
        nn.init.zeros_(correction_output.bias)
        nn.init.normal_(gate_output.weight, std=1.0e-3)
        nn.init.constant_(gate_output.bias, -1.5)

    def forward(
        self,
        *,
        pose_embedding: torch.Tensor,
        direct_ellipse: torch.Tensor,
        direct_keypoints: torch.Tensor,
        geometry_ellipse: torch.Tensor,
        geometry_keypoints: torch.Tensor,
        evidence: FusionEvidence,
    ) -> dict[str, torch.Tensor]:
        count = int(pose_embedding.shape[0])
        if direct_ellipse.shape != (count, 5):
            raise ValueError("direct_ellipse must have shape [N,5]")
        if direct_keypoints.shape != (count, self.keypoints, 2):
            raise ValueError(f"direct_keypoints must have shape [N,{self.keypoints},2]")
        evidence_tensor = evidence.tensor(like=pose_embedding)
        latent = torch.cat((pose_embedding, evidence_tensor), dim=-1)
        correction = self.correction(latent)
        cursor = 0
        center_delta = correction[:, cursor : cursor + 2]
        cursor += 2
        log_axis_delta = correction[:, cursor : cursor + 2]
        cursor += 2
        angle_delta = correction[:, cursor : cursor + 2]
        cursor += 2
        keypoint_delta = correction[:, cursor:].reshape(
            count,
            self.keypoints,
            2,
        )

        corrected_center = geometry_ellipse[:, :2] + center_delta
        corrected_axes = geometry_ellipse[:, 2:4].clamp_min(1.0e-6) * torch.exp(
            log_axis_delta.clamp(-2.0, 2.0)
        )
        geometry_angle = doubled_angle_vector(geometry_ellipse[:, 4])
        corrected_raw_angle = geometry_angle + 0.25 * angle_delta
        corrected_angle = torch.where(
            torch.linalg.vector_norm(
                corrected_raw_angle,
                dim=-1,
                keepdim=True,
            )
            > 1.0e-8,
            F.normalize(corrected_raw_angle, dim=-1, eps=1.0e-8),
            geometry_angle,
        )
        corrected_keypoints = geometry_keypoints + keypoint_delta.reshape(
            count,
            self.keypoints,
            2,
        )

        learned_gate = torch.sigmoid(self.gate(latent))
        pose_reliability = torch.exp(
            -evidence.pose_uncertainty.to(pose_embedding)
        ).clamp(0.0, 1.0)
        geometry_reliability = torch.exp(
            -evidence.geometry_uncertainty.to(pose_embedding)
        ).clamp(0.0, 1.0)
        direct_distrust = evidence.direct_ellipse_uncertainty.to(pose_embedding).clamp(
            0.0, 1.0
        )
        safety = (
            pose_reliability * geometry_reliability * (0.25 + 0.75 * direct_distrust)
        )
        ellipse_gate = learned_gate[:, 0] * safety
        keypoint_gate = learned_gate[:, 1] * safety

        final_center = torch.lerp(
            direct_ellipse[:, :2],
            corrected_center,
            ellipse_gate.unsqueeze(-1),
        )
        # Semi-axis fusion is performed in log space, preserving positivity and
        # order instead of averaging raw axis parameters.
        final_log_axes = torch.lerp(
            torch.log(direct_ellipse[:, 2:4].clamp_min(1.0e-6)),
            torch.log(corrected_axes.clamp_min(1.0e-6)),
            ellipse_gate.unsqueeze(-1),
        )
        final_axes = torch.exp(final_log_axes)
        final_major = torch.maximum(final_axes[:, 0], final_axes[:, 1])
        final_minor = torch.minimum(final_axes[:, 0], final_axes[:, 1])
        direct_angle = doubled_angle_vector(direct_ellipse[:, 4])
        mixed_angle = (
            1.0 - ellipse_gate.unsqueeze(-1)
        ) * direct_angle + ellipse_gate.unsqueeze(-1) * corrected_angle
        final_angle_vector = torch.where(
            torch.linalg.vector_norm(
                mixed_angle,
                dim=-1,
                keepdim=True,
            )
            > 1.0e-8,
            F.normalize(mixed_angle, dim=-1, eps=1.0e-8),
            direct_angle,
        )
        final_theta = 0.5 * torch.atan2(
            final_angle_vector[:, 1],
            final_angle_vector[:, 0],
        )
        final_ellipse = torch.stack(
            (
                final_center[:, 0],
                final_center[:, 1],
                final_major,
                final_minor,
                final_theta,
            ),
            dim=-1,
        )
        final_keypoints = torch.lerp(
            direct_keypoints,
            corrected_keypoints,
            keypoint_gate[:, None, None],
        )
        return {
            "fused_ellipse": final_ellipse,
            "fused_ellipse_angle_vector": final_angle_vector,
            "fused_keypoints": final_keypoints,
            "geometry_residual_ellipse": torch.cat(
                (
                    corrected_center,
                    corrected_axes,
                    0.5
                    * torch.atan2(
                        corrected_angle[:, 1],
                        corrected_angle[:, 0],
                    ).unsqueeze(-1),
                ),
                dim=-1,
            ),
            "geometry_residual_keypoints": corrected_keypoints,
            "ellipse_geometry_gate": ellipse_gate,
            "keypoint_geometry_gate": keypoint_gate,
            "gate_evidence": evidence_tensor,
        }


__all__ = [
    "FusionEvidence",
    "UncertaintyGatedGeometryFusion",
]
