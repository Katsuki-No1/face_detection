from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def differentiable_covariance_ellipse(
    points: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    epsilon: float = 1.0e-6,
    contour_calibration: float = math.sqrt(2.0),
    circle_threshold: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Fit an amodal ellipse by weighted mean and regularized covariance.

    For uniformly sampled ellipse-boundary points, covariance eigenvalues are
    ``a^2/2`` and ``b^2/2``; the default ``sqrt(2)`` calibration therefore
    recovers semi-axis lengths.
    """

    if points.shape[-1] != 2 or points.shape[-2] < 3:
        raise ValueError(f"Expected [...,N>=3,2] points, got {tuple(points.shape)}")
    if weights is None:
        weights = torch.ones_like(points[..., 0])
    if weights.shape != points.shape[:-1]:
        raise ValueError(
            f"Weight shape {tuple(weights.shape)} does not match points "
            f"{tuple(points.shape)}"
        )
    weights = weights.to(points).clamp_min(0.0)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    normalized = weights / denominator
    center = (points * normalized.unsqueeze(-1)).sum(dim=-2)
    centered = points - center.unsqueeze(-2)
    covariance = torch.einsum(
        "...n,...ni,...nj->...ij",
        normalized,
        centered,
        centered,
    )
    identity = torch.eye(2, device=points.device, dtype=points.dtype)
    covariance = covariance + float(epsilon) * identity
    xx = covariance[..., 0, 0]
    xy = covariance[..., 0, 1]
    yy = covariance[..., 1, 1]
    trace = xx + yy
    anisotropy = torch.sqrt(((xx - yy).square() + 4.0 * xy.square()).clamp_min(0.0))
    major_eigenvalue = (0.5 * (trace + anisotropy)).clamp_min(epsilon)
    minor_eigenvalue = (0.5 * (trace - anisotropy)).clamp_min(epsilon)
    major = float(contour_calibration) * torch.sqrt(major_eigenvalue)
    minor = float(contour_calibration) * torch.sqrt(minor_eigenvalue)

    raw_angle = torch.stack((xx - yy, 2.0 * xy), dim=-1)
    raw_norm = torch.linalg.vector_norm(raw_angle, dim=-1, keepdim=True)
    default_angle = torch.zeros_like(raw_angle)
    default_angle[..., 0] = 1.0
    angle_vector = torch.where(
        raw_norm
        > float(circle_threshold) * trace.abs().unsqueeze(-1).clamp_min(epsilon),
        raw_angle / raw_norm.clamp_min(epsilon),
        default_angle,
    )
    theta = 0.5 * torch.atan2(angle_vector[..., 1], angle_vector[..., 0])
    reliability = (anisotropy / trace.abs().clamp_min(epsilon)).clamp(0.0, 1.0)
    ellipse = torch.cat(
        (
            center,
            major.unsqueeze(-1),
            minor.unsqueeze(-1),
            theta.unsqueeze(-1),
        ),
        dim=-1,
    )
    return {
        "ellipse": ellipse,
        "center": center,
        "axes": torch.stack((major, minor), dim=-1),
        "angle_vector": angle_vector,
        "angle_reliability": reliability,
        "covariance": covariance,
        "eigenvalues": torch.stack(
            (major_eigenvalue, minor_eigenvalue),
            dim=-1,
        ),
    }


def _canonical_template(
    contour_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = torch.arange(contour_points, dtype=torch.float32) * (
        2.0 * math.pi / contour_points
    )
    x = 0.72 * torch.cos(angles)
    y = 1.00 * torch.sin(angles)
    # A shallow, curved facial cap. Depth variation makes weak-perspective
    # profile projection informative while keeping the model lightweight.
    radial = (x / 0.82).square() + (y / 1.12).square()
    z = -0.58 - 0.22 * torch.sqrt((1.0 - 0.72 * radial).clamp_min(0.0))
    contour = torch.stack((x, y, z), dim=-1)
    contour_normals = F.normalize(
        torch.stack((x / 0.82**2, y / 1.12**2, z / 0.82**2), dim=-1),
        dim=-1,
    )

    keypoints = torch.tensor(
        (
            (-0.30, -0.24, -0.89),
            (0.30, -0.24, -0.89),
            (0.00, 0.02, -1.00),
            (-0.23, 0.40, -0.84),
            (0.23, 0.40, -0.84),
        ),
        dtype=torch.float32,
    )
    keypoint_normals = F.normalize(
        torch.stack(
            (
                keypoints[:, 0] / 0.82**2,
                keypoints[:, 1] / 1.12**2,
                keypoints[:, 2] / 0.82**2,
            ),
            dim=-1,
        ),
        dim=-1,
    )
    return contour, keypoints, contour_normals, keypoint_normals


class CanonicalHeadGeometryAdapter(nn.Module):
    """Fixed-template weak-perspective head geometry with optional shape code."""

    def __init__(
        self,
        pose_embedding_dimension: int,
        *,
        hidden_dimension: int = 128,
        contour_points: int = 48,
        shape_dimension: int = 0,
        shape_limit: float = 0.15,
        minimum_scale: float = 1.0e-4,
        visibility_sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        if contour_points < 8:
            raise ValueError("contour_points must be at least 8")
        if shape_dimension < 0 or shape_dimension > 4:
            raise ValueError("shape_dimension must be between 0 and 4")
        self.shape_dimension = int(shape_dimension)
        self.shape_limit = float(shape_limit)
        self.minimum_scale = float(minimum_scale)
        self.visibility_sharpness = float(visibility_sharpness)
        contour, keypoints, contour_normals, keypoint_normals = _canonical_template(
            contour_points
        )
        vertices = torch.cat((contour, keypoints), dim=0)
        normals = torch.cat((contour_normals, keypoint_normals), dim=0)
        self.contour_count = int(contour.shape[0])
        self.keypoint_count = int(keypoints.shape[0])
        self.register_buffer("template_vertices", vertices)
        self.register_buffer("template_normals", normals)
        basis = torch.zeros((4, vertices.shape[0], 3), dtype=torch.float32)
        basis[0, :, 0] = vertices[:, 0]
        basis[1, :, 1] = vertices[:, 1]
        basis[2, :, 2] = vertices[:, 2]
        basis[3, :, 0] = vertices[:, 1] * 0.25
        self.register_buffer("shape_basis", basis)

        context_dimension = pose_embedding_dimension + 6
        output_dimension = 1 + 2 + 1 + 1 + 1 + self.shape_dimension
        self.predictor = nn.Sequential(
            nn.Linear(context_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension),
        )
        final = self.predictor[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _deformed_vertices(
        self,
        shape_code: torch.Tensor,
    ) -> torch.Tensor:
        base = self.template_vertices.to(shape_code)
        if self.shape_dimension == 0:
            return base.unsqueeze(0).expand(shape_code.shape[0], -1, -1)
        basis = self.shape_basis[: self.shape_dimension].to(shape_code)
        return base.unsqueeze(0) + torch.einsum(
            "bi,ivc->bvc",
            shape_code,
            basis,
        )

    def forward(
        self,
        *,
        rotation_matrix: torch.Tensor,
        pose_embedding: torch.Tensor,
        head_boxes_xyxy: torch.Tensor,
        border_distances: torch.Tensor,
        head_confidence: torch.Tensor,
        valid_fraction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = int(rotation_matrix.shape[0])
        if rotation_matrix.shape != (count, 3, 3):
            raise ValueError("rotation_matrix must have shape [N,3,3]")
        if head_boxes_xyxy.shape != (count, 4):
            raise ValueError("head_boxes_xyxy must have shape [N,4]")
        context_features = torch.cat(
            (
                border_distances.to(pose_embedding),
                head_confidence.to(pose_embedding).reshape(count, 1),
                valid_fraction.to(pose_embedding).reshape(count, 1),
            ),
            dim=-1,
        )
        raw = self.predictor(torch.cat((pose_embedding, context_features), dim=-1))
        offset = 0
        log_scale_delta = raw[:, offset]
        offset += 1
        translation_delta = raw[:, offset : offset + 2]
        offset += 2
        uncertainty_raw = raw[:, offset]
        offset += 1
        observability_delta = raw[:, offset]
        offset += 1
        amodal_logit = raw[:, offset]
        offset += 1
        if self.shape_dimension:
            shape_code = self.shape_limit * torch.tanh(
                raw[:, offset : offset + self.shape_dimension]
            )
        else:
            shape_code = raw.new_zeros((count, 0))

        head_size = (head_boxes_xyxy[:, 2:] - head_boxes_xyxy[:, :2]).clamp_min(
            self.minimum_scale
        )
        head_center = 0.5 * (head_boxes_xyxy[:, :2] + head_boxes_xyxy[:, 2:])
        base_scale = 0.42 * head_size.amax(dim=-1)
        scale = (base_scale * torch.exp(log_scale_delta.clamp(-2.0, 2.0))).clamp_min(
            self.minimum_scale
        )
        # Translation is intentionally not sigmoid-clamped; this is the amodal
        # image coordinate path and may leave [0,1].
        translation = head_center + torch.tanh(translation_delta) * head_size

        vertices = self._deformed_vertices(shape_code)
        rotated = torch.einsum("bij,bvj->bvi", rotation_matrix, vertices)
        projected = scale[:, None, None] * rotated[..., :2] + translation[:, None]

        normals = self.template_normals.to(rotation_matrix)
        rotated_normals = torch.einsum(
            "bij,vj->bvi",
            rotation_matrix,
            normals,
        )
        view_direction = rotation_matrix.new_tensor((0.0, 0.0, -1.0))
        facing = (rotated_normals * view_direction).sum(dim=-1)
        visibility_logits = self.visibility_sharpness * facing
        visibility = torch.sigmoid(visibility_logits)

        contour = projected[:, : self.contour_count]
        keypoints = projected[:, self.contour_count :]
        contour_visibility = visibility[:, : self.contour_count]
        keypoint_visibility = visibility[:, self.contour_count :]
        keypoint_visibility_logits = visibility_logits[:, self.contour_count :]
        # The amodal prior uses the complete projected canonical contour.
        # Self-occlusion weighting produces a separate visible estimate; using
        # it for the amodal ellipse would bias profile centers toward the
        # camera-facing half of the template.
        fitted = differentiable_covariance_ellipse(contour)
        visible_fitted = differentiable_covariance_ellipse(
            contour,
            weights=contour_visibility,
        )

        canonical_face_normal = rotation_matrix.new_tensor((0.0, 0.0, -1.0))
        rotated_face_normal = torch.einsum(
            "bij,j->bi",
            rotation_matrix,
            canonical_face_normal,
        )
        facing_cosine = (rotated_face_normal * view_direction).sum(dim=-1)
        base_observability_logit = 6.0 * facing_cosine
        observability_logits = base_observability_logit + observability_delta
        observability = torch.sigmoid(observability_logits)
        uncertainty = F.softplus(uncertainty_raw)
        geometry_confidence = torch.exp(-uncertainty).clamp(0.0, 1.0)
        amodal_confidence = torch.sigmoid(amodal_logit) * geometry_confidence
        visible_confidence = amodal_confidence * observability
        return {
            "projected_contour": contour,
            "projected_keypoints": keypoints,
            "amodal_ellipse": fitted["ellipse"],
            "visible_ellipse": visible_fitted["ellipse"],
            "ellipse_angle_vector": fitted["angle_vector"],
            "ellipse_angle_reliability": fitted["angle_reliability"],
            "ellipse_covariance": fitted["covariance"],
            "visible_ellipse_covariance": visible_fitted["covariance"],
            "contour_visibility": contour_visibility,
            "keypoint_visibility": keypoint_visibility,
            "keypoint_visibility_logits": keypoint_visibility_logits,
            "face_observability": observability,
            "face_observability_logits": observability_logits,
            "visible_face_confidence": visible_confidence,
            "amodal_face_confidence": amodal_confidence,
            "geometry_uncertainty": uncertainty,
            "geometry_confidence": geometry_confidence,
            "weak_perspective_scale": scale,
            "translation": translation,
            "shape_code": shape_code,
            "rotated_vertices": rotated,
            "facing_cosine": facing_cosine,
        }


__all__ = [
    "CanonicalHeadGeometryAdapter",
    "differentiable_covariance_ellipse",
]
