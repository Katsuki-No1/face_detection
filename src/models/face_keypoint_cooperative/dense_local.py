from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from keypoint_comparison1.landmark_detection.heads import (
    FACTORIZED_STATE_HEAD,
    factorized_state_probabilities,
)
from src.models.dinov3_deimv2_ellipse import canonicalize_ellipses


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    dtype_eps = torch.finfo(value.dtype).eps if value.is_floating_point() else eps
    effective_eps = max(float(eps), float(dtype_eps))
    value = value.clamp(effective_eps, 1.0 - effective_eps)
    return torch.log(value / (1.0 - value))


def _rotated_offsets(
    offsets: torch.Tensor,
    axes: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Map canonical ellipse offsets to normalized full-frame coordinates."""

    scaled = offsets.view(1, 1, 1, -1, 2) * axes[:, :, None, None]
    cosine = torch.cos(theta)[:, :, None, None]
    sine = torch.sin(theta)[:, :, None, None]
    x = cosine * scaled[..., 0] - sine * scaled[..., 1]
    y = sine * scaled[..., 0] + cosine * scaled[..., 1]
    return torch.stack((x, y), dim=-1)


class _DenseLocalStage(nn.Module):
    def __init__(
        self,
        dimension: int,
        *,
        radius_in_face_axes: float,
        maximum_delta_in_face_axes: float,
        minimum_association_gate: float,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.radius_in_face_axes = float(radius_in_face_axes)
        self.maximum_delta_in_face_axes = float(maximum_delta_in_face_axes)
        self.minimum_association_gate = float(minimum_association_gate)
        self.query_norm = nn.LayerNorm(dimension)
        self.sample_norm = nn.LayerNorm(dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        combined = 3 * dimension + 1
        self.update = nn.Sequential(
            nn.Linear(combined, 2 * dimension),
            nn.GELU(),
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(combined, dimension),
            nn.GELU(),
            nn.Linear(dimension, 1),
            nn.Sigmoid(),
        )
        self.coordinate_delta = nn.Linear(dimension, 2)
        self.presence_delta = nn.Linear(dimension, 1)
        self.visibility_delta = nn.Linear(dimension, 1)
        self.log_scale = nn.Linear(dimension, 2)
        self.pose_update = nn.Linear(dimension, dimension)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.coordinate_delta,
            self.presence_delta,
            self.visibility_delta,
            self.pose_update,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.constant_(self.log_scale.bias, math.log(0.05))
        gate_output = self.gate[-2]
        assert isinstance(gate_output, nn.Linear)
        nn.init.normal_(gate_output.weight, std=1.0e-3)
        nn.init.constant_(gate_output.bias, math.log(0.1 / 0.9))

    def forward(
        self,
        keypoint_embeddings: torch.Tensor,
        pose_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        keypoint_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
        samples: torch.Tensor,
        face_context: torch.Tensor,
        associated_axes: torch.Tensor,
        associated_theta: torch.Tensor,
        association_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query = self.query_norm(keypoint_embeddings)
        normalized_samples = self.sample_norm(samples)
        keys = self.key(normalized_samples)
        values = self.value(normalized_samples)
        attention = torch.einsum("bpkd,bpksd->bpks", query, keys)
        attention = (attention / math.sqrt(self.dimension)).softmax(dim=-1)
        context = torch.einsum("bpks,bpksd->bpkd", attention, values)
        expanded_face = face_context.unsqueeze(2).expand_as(query)
        expanded_confidence = association_confidence[:, :, None, None].expand(
            *query.shape[:-1],
            1,
        )
        combined = torch.cat(
            (query, context, expanded_face, expanded_confidence),
            dim=-1,
        )
        residual = self.update(combined)
        learned_gate = self.gate(combined)
        association_gate = (
            self.minimum_association_gate
            + (1.0 - self.minimum_association_gate)
            * association_confidence[:, :, None, None]
        )
        gate = learned_gate * association_gate
        keypoint_embeddings = keypoint_embeddings + gate * residual

        local_delta = (
            torch.tanh(self.coordinate_delta(keypoint_embeddings))
            * self.maximum_delta_in_face_axes
            * associated_axes[:, :, None]
        )
        cosine = torch.cos(associated_theta)[:, :, None]
        sine = torch.sin(associated_theta)[:, :, None]
        global_delta = torch.stack(
            (
                cosine * local_delta[..., 0] - sine * local_delta[..., 1],
                sine * local_delta[..., 0] + cosine * local_delta[..., 1],
            ),
            dim=-1,
        )
        coordinates = (coordinates + gate * global_delta).clamp(0.0, 1.0)
        keypoint_logits = (
            keypoint_logits
            + (gate * self.presence_delta(keypoint_embeddings)).squeeze(-1)
        )
        visibility_logits = (
            visibility_logits
            + (gate * self.visibility_delta(keypoint_embeddings)).squeeze(-1)
        )
        pose_embeddings = pose_embeddings + self.pose_update(
            (gate * residual).mean(dim=2)
        )
        log_scales = self.log_scale(keypoint_embeddings).clamp(-7.0, 0.0)
        return {
            "pose_embeddings": pose_embeddings,
            "keypoint_embeddings": keypoint_embeddings,
            "pred_keypoints": coordinates,
            "pred_keypoint_logits": keypoint_logits,
            "pred_visibility_logits": visibility_logits,
            "pred_keypoint_log_scales": log_scales,
            "local_refinement_gate": gate.squeeze(-1),
            "local_attention": attention,
        }


class EllipseAlignedDensePoseRefiner(nn.Module):
    """Feature-level face normalization and coarse-to-fine point refinement.

    The module never crops or re-runs the RGB backbone.  It builds a stride-4
    feature from the shared pose feature and a shallow RGB edge stem, associates
    every independent pose proposal with soft top-k Face proposals, and samples
    progressively smaller ellipse-aligned neighborhoods around all five points.
    """

    def __init__(
        self,
        feature_dimension: int,
        face_dimension: int,
        *,
        stages: Sequence[Mapping[str, float]],
        samples_per_axis: int = 3,
        feature_levels: int = 3,
        association_topk: int = 4,
        association_temperature: float = 1.0,
        roi_scale: float = 1.35,
        minimum_association_gate: float = 0.25,
        state_head_mode: str = "legacy_binary",
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("dense local refinement requires at least one stage")
        if samples_per_axis < 1 or samples_per_axis % 2 == 0:
            raise ValueError("samples_per_axis must be a positive odd integer")
        self.dimension = int(feature_dimension)
        self.feature_levels = int(feature_levels)
        self.association_topk = int(association_topk)
        self.association_temperature = float(association_temperature)
        self.roi_scale = float(roi_scale)
        self.samples_per_axis = int(samples_per_axis)
        self.state_head_mode = str(state_head_mode)
        stem_dimension = max(32, feature_dimension // 2)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, stem_dimension, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(1, stem_dimension),
            nn.GELU(),
            nn.Conv2d(
                stem_dimension,
                feature_dimension,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, feature_dimension),
            nn.GELU(),
        )
        self.p2_lateral = nn.Conv2d(feature_dimension, feature_dimension, 1)
        self.p2_fusion = nn.Sequential(
            nn.Conv2d(2 * feature_dimension, feature_dimension, 3, padding=1, bias=False),
            nn.GroupNorm(1, feature_dimension),
            nn.GELU(),
        )
        self.face_projection = (
            nn.Identity()
            if face_dimension == feature_dimension
            else nn.Linear(face_dimension, feature_dimension)
        )
        self.level_embeddings = nn.Parameter(
            torch.zeros(max(1, feature_levels), feature_dimension)
        )
        self.stages = nn.ModuleList(
            [
                _DenseLocalStage(
                    feature_dimension,
                    radius_in_face_axes=float(stage["radius_in_face_axes"]),
                    maximum_delta_in_face_axes=float(
                        stage["maximum_delta_in_face_axes"]
                    ),
                    minimum_association_gate=minimum_association_gate,
                )
                for stage in stages
            ]
        )
        self.pose_evidence = nn.Sequential(
            nn.Linear(3 * feature_dimension + 1, feature_dimension),
            nn.GELU(),
            nn.Linear(feature_dimension, 1),
        )
        evidence_output = self.pose_evidence[-1]
        assert isinstance(evidence_output, nn.Linear)
        nn.init.zeros_(evidence_output.weight)
        nn.init.zeros_(evidence_output.bias)
        axis = torch.linspace(-1.0, 1.0, samples_per_axis)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "sampling_offsets",
            torch.stack((x, y), dim=-1).reshape(-1, 2),
            persistent=False,
        )

    def _pyramid(
        self,
        images: torch.Tensor,
        projected_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not projected_features:
            raise ValueError("dense local refinement requires projected features")
        rgb = self.rgb_stem(images)
        p3 = self.p2_lateral(projected_features[0])
        p3 = F.interpolate(
            p3,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.p2_fusion(torch.cat((rgb, p3), dim=1))
        available = [p2, *projected_features]
        return available[: min(self.feature_levels, len(available))]

    def _associated_faces(
        self,
        face: Mapping[str, torch.Tensor],
        association_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # B,F,P -> B,P,F
        pair_logits = association_logits.transpose(1, 2)
        faces = pair_logits.shape[-1]
        if faces == 0:
            poses = pair_logits.shape[1]
            dimension = face["embeddings"].shape[-1]
            zero = pair_logits.new_zeros((pair_logits.shape[0], poses))
            return {
                "boxes": pair_logits.new_zeros((pair_logits.shape[0], poses, 4)),
                "ellipses": pair_logits.new_zeros((pair_logits.shape[0], poses, 5)),
                "context": pair_logits.new_zeros(
                    (pair_logits.shape[0], poses, dimension)
                ),
                "confidence": zero,
                "weights": pair_logits,
            }
        count = min(self.association_topk, faces)
        top_values, top_indices = torch.topk(pair_logits, count, dim=-1)
        masked = torch.full_like(pair_logits, torch.finfo(pair_logits.dtype).min)
        masked.scatter_(-1, top_indices, top_values)
        weights = (masked / max(self.association_temperature, 1.0e-4)).softmax(dim=-1)
        boxes = torch.einsum("bpf,bfd->bpd", weights, face["pred_boxes"])
        ellipses = face["pred_ellipses"]
        center_axes = torch.einsum("bpf,bfd->bpd", weights, ellipses[..., :4])
        angle_vector = torch.stack(
            (torch.cos(2.0 * ellipses[..., 4]), torch.sin(2.0 * ellipses[..., 4])),
            dim=-1,
        )
        associated_angle = torch.einsum("bpf,bfd->bpd", weights, angle_vector)
        theta = 0.5 * torch.atan2(associated_angle[..., 1], associated_angle[..., 0])
        associated_ellipses = canonicalize_ellipses(
            torch.cat((center_axes, theta.unsqueeze(-1)), dim=-1)
        )
        projected_face = self.face_projection(face["embeddings"])
        context = torch.einsum("bpf,bfd->bpd", weights, projected_face)
        confidence = (weights * pair_logits.sigmoid()).sum(dim=-1)
        return {
            "boxes": boxes,
            "ellipses": associated_ellipses,
            "context": context,
            "confidence": confidence,
            "weights": weights,
        }

    def _samples(
        self,
        features: Sequence[torch.Tensor],
        coordinates: torch.Tensor,
        ellipses: torch.Tensor,
        *,
        radius: float,
    ) -> torch.Tensor:
        axes = ellipses[..., 2:4].clamp_min(1.0e-4) * self.roi_scale * radius
        offsets = _rotated_offsets(
            self.sampling_offsets.to(dtype=coordinates.dtype),
            axes,
            ellipses[..., 4],
        )
        grid = coordinates.unsqueeze(-2) + offsets
        batch, poses, keypoints, samples = grid.shape[:4]
        grid = grid.reshape(batch, poses * keypoints, samples, 2) * 2.0 - 1.0
        sampled_levels = []
        for level, feature in enumerate(features):
            sampled = F.grid_sample(
                feature,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.permute(0, 2, 3, 1).reshape(
                batch,
                poses,
                keypoints,
                samples,
                self.dimension,
            )
            sampled_levels.append(sampled + self.level_embeddings[level])
        return torch.cat(sampled_levels, dim=-2)

    def forward(
        self,
        images: torch.Tensor,
        face: Mapping[str, torch.Tensor],
        pose: Mapping[str, Any],
        association: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        output: dict[str, Any] = dict(pose)
        poses = int(pose["pred_keypoints"].shape[1])
        faces = int(face["pred_boxes"].shape[1])
        if poses == 0 or faces == 0:
            output["dense_local_auxiliary_outputs"] = []
            output["dense_local_enabled"] = torch.ones(
                (), device=images.device, dtype=images.dtype
            )
            return output
        features = self._pyramid(images, pose["projected_features"])
        associated = self._associated_faces(face, association["logits"])
        state: dict[str, torch.Tensor] = {
            "pose_embeddings": pose["pose_embeddings"],
            "keypoint_embeddings": pose["keypoint_embeddings"],
            "pred_keypoints": pose["pred_keypoints"],
            "pred_keypoint_logits": pose["pred_keypoint_logits"],
            "pred_visibility_logits": pose["pred_visibility_logits"],
        }
        auxiliary: list[dict[str, torch.Tensor]] = []
        for stage in self.stages:
            samples = self._samples(
                features,
                state["pred_keypoints"],
                associated["ellipses"],
                radius=stage.radius_in_face_axes,
            )
            state = stage(
                state["keypoint_embeddings"],
                state["pose_embeddings"],
                state["pred_keypoints"],
                state["pred_keypoint_logits"],
                state["pred_visibility_logits"],
                samples,
                associated["context"],
                associated["ellipses"][..., 2:4].clamp_min(1.0e-4),
                associated["ellipses"][..., 4],
                associated["confidence"],
            )
            auxiliary.append(
                {
                    **{
                        key: value
                        for key, value in output.items()
                        if key.startswith("pred_pose_")
                    },
                    "pred_pose_logits": output["pred_pose_logits"],
                    "pred_pose_boxes": output["pred_pose_boxes"],
                    "pred_keypoints": state["pred_keypoints"],
                    "pred_keypoint_logits": state["pred_keypoint_logits"],
                    "pred_visibility_logits": state["pred_visibility_logits"],
                    "pred_keypoint_log_scales": state[
                        "pred_keypoint_log_scales"
                    ],
                }
            )
        evidence_input = torch.cat(
            (
                state["pose_embeddings"],
                state["keypoint_embeddings"].mean(dim=2),
                associated["context"],
                associated["confidence"].unsqueeze(-1),
            ),
            dim=-1,
        )
        pose_evidence = self.pose_evidence(evidence_input).squeeze(-1)
        output.update(state)
        output["pred_pose_logits"] = (
            pose["pred_pose_logits"]
            + associated["confidence"] * pose_evidence
        )
        output["pose_evidence_logits"] = pose_evidence
        output["associated_face_boxes"] = associated["boxes"]
        output["associated_face_ellipses"] = associated["ellipses"]
        output["association_confidence"] = associated["confidence"]
        output["association_weights"] = associated["weights"]
        output["dense_p2_spatial_shape"] = tuple(
            int(value) for value in features[0].shape[-2:]
        )
        output["dense_local_auxiliary_outputs"] = auxiliary[:-1]
        output["dense_local_enabled"] = torch.ones(
            (), device=images.device, dtype=images.dtype
        )
        if self.state_head_mode == FACTORIZED_STATE_HEAD:
            output["pred_state_probabilities"] = factorized_state_probabilities(
                output["pred_keypoint_logits"],
                output["pred_visibility_logits"],
            )
        return output


class PoseEvidenceFaceFusion(nn.Module):
    """Expose and supervise pose evidence instead of hiding it in a Face logit."""

    def __init__(
        self,
        dimension: int,
        geometry_dimension: int,
        *,
        topk_associations: int = 4,
        initial_gate: float = 0.10,
        maximum_box_delta: float = 0.10,
    ) -> None:
        super().__init__()
        self.topk_associations = int(topk_associations)
        self.maximum_box_delta = float(maximum_box_delta)
        inputs = 2 * dimension + geometry_dimension + 5
        self.encoder = nn.Sequential(
            nn.Linear(inputs, 2 * dimension),
            nn.GELU(),
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
        )
        self.pose_evidence_head = nn.Linear(dimension, 1)
        self.localization_quality_head = nn.Linear(dimension, 1)
        self.confidence_gate = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())
        self.box_gate = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())
        self.box_delta = nn.Linear(dimension, 4)
        self.ellipse_delta = nn.Linear(dimension, 6)
        for head in (
            self.pose_evidence_head,
            self.localization_quality_head,
            self.box_delta,
            self.ellipse_delta,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for gate in (self.confidence_gate, self.box_gate):
            layer = gate[0]
            assert isinstance(layer, nn.Linear)
            nn.init.normal_(layer.weight, std=1.0e-3)
            nn.init.constant_(layer.bias, math.log(initial_gate / (1.0 - initial_gate)))

    def forward(self, refined: Mapping[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = dict(refined)
        logits = refined["association_logits"]
        faces, poses = logits.shape[1:]
        if faces == 0 or poses == 0:
            return output
        count = min(self.topk_associations, poses)
        top_values, top_indices = torch.topk(logits, count, dim=-1)
        weights = top_values.softmax(dim=-1)
        batch_indices = torch.arange(
            logits.shape[0],
            device=logits.device,
        )[:, None, None]
        pose_embeddings = refined["pose_embeddings"][batch_indices, top_indices]
        pose_context = (weights.unsqueeze(-1) * pose_embeddings).sum(dim=2)
        face_indices = torch.arange(
            faces,
            device=logits.device,
        )[None, :, None]
        geometry = refined["association_geometry"][
            batch_indices,
            face_indices,
            top_indices,
        ]
        geometry_context = (weights.unsqueeze(-1) * geometry).sum(dim=2)
        pose_probability = refined["pred_pose_logits"].sigmoid()[
            batch_indices,
            top_indices,
        ]
        presence = refined["pred_keypoint_logits"].sigmoid().mean(dim=-1)[
            batch_indices,
            top_indices,
        ]
        visibility = refined["pred_visibility_logits"].sigmoid().mean(dim=-1)[
            batch_indices,
            top_indices,
        ]
        association_probability = top_values.sigmoid()
        if "pred_keypoint_log_scales" in refined:
            uncertainty = refined["pred_keypoint_log_scales"].exp().mean(dim=(-1, -2))
            uncertainty = uncertainty[batch_indices, top_indices]
        else:
            uncertainty = torch.zeros_like(pose_probability)
        scalars = torch.stack(
            (
                (weights * pose_probability).sum(dim=-1),
                (weights * presence).sum(dim=-1),
                (weights * visibility).sum(dim=-1),
                (weights * association_probability).sum(dim=-1),
                (weights * uncertainty).sum(dim=-1),
            ),
            dim=-1,
        )
        encoded = self.encoder(
            torch.cat(
                (
                    refined["face_embeddings"],
                    pose_context,
                    geometry_context,
                    scalars,
                ),
                dim=-1,
            )
        )
        pose_evidence = self.pose_evidence_head(encoded)
        localization_quality = self.localization_quality_head(encoded)
        confidence_gate = self.confidence_gate(encoded)
        box_gate = self.box_gate(encoded)
        base_boxes = refined["pred_face_boxes"]
        boxes = torch.sigmoid(
            _inverse_sigmoid(base_boxes)
            + box_gate
            * self.maximum_box_delta
            * torch.tanh(self.box_delta(encoded))
        )
        base_ellipses = refined["pred_face_ellipses"]
        raw_ellipse = self.ellipse_delta(encoded)
        ellipse_center = torch.sigmoid(
            _inverse_sigmoid(base_ellipses[..., :2])
            + box_gate * self.maximum_box_delta * torch.tanh(raw_ellipse[..., :2])
        )
        ellipse_axes = base_ellipses[..., 2:4] * torch.exp(
            box_gate * self.maximum_box_delta * torch.tanh(raw_ellipse[..., 2:4])
        )
        angle_vector = torch.stack(
            (
                torch.cos(2.0 * base_ellipses[..., 4]),
                torch.sin(2.0 * base_ellipses[..., 4]),
            ),
            dim=-1,
        )
        angle_vector = F.normalize(
            angle_vector
            + box_gate
            * self.maximum_box_delta
            * torch.tanh(raw_ellipse[..., 4:6]),
            dim=-1,
            eps=1.0e-6,
        )
        ellipses = canonicalize_ellipses(
            torch.cat(
                (
                    ellipse_center,
                    ellipse_axes,
                    (0.5 * torch.atan2(angle_vector[..., 1], angle_vector[..., 0])).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        output["pred_face_objectness_logits"] = refined["pred_face_logits"]
        output["pred_face_pose_evidence_logits"] = pose_evidence.squeeze(-1)
        output["pred_face_localization_quality_logits"] = localization_quality.squeeze(-1)
        output["face_evidence_gate"] = confidence_gate.squeeze(-1)
        output["face_box_update_gate"] = box_gate.squeeze(-1)
        output["pre_evidence_face_boxes"] = base_boxes
        output["pre_evidence_face_ellipses"] = base_ellipses
        output["pred_face_boxes"] = boxes
        output["pred_face_ellipses"] = ellipses
        output["pred_face_logits"] = (
            refined["pred_face_logits"]
            + confidence_gate * pose_evidence.expand_as(refined["pred_face_logits"])
        )
        return output


__all__ = ["EllipseAlignedDensePoseRefiner", "PoseEvidenceFaceFusion"]
