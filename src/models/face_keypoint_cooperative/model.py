from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from keypoint_comparison1.landmark_detection.heads import (
    FACTORIZED_STATE_HEAD,
    LEGACY_BINARY_STATE_HEAD,
    STATE_HEAD_MODES,
    CoordinateRefinementStage,
    factorized_state_probabilities,
)
from src.models.dinov3_deimv2_ellipse import canonicalize_ellipses
from src.third_party.deimv2.engine.deim.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from src.third_party.deimv2.engine.deim.dfine_decoder import MSDeformableAttention

from .cascade import ExternalHeadCascadeRouter, detections_from_targets
from .dense_local import EllipseAlignedDensePoseRefiner, PoseEvidenceFaceFusion
from .sequential import StrictSequentialCascade
from .shared_head import SharedDINOHeadBranch


KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    # In FP16, 1 - 1e-5 rounds back to exactly 1 and produces log(inf).
    # Use an epsilon that is representable below one for the active dtype.
    dtype_eps = torch.finfo(value.dtype).eps if torch.is_floating_point(value) else eps
    effective_eps = max(float(eps), float(dtype_eps))
    value = value.clamp(min=effective_eps, max=1.0 - effective_eps)
    return torch.log(value / (1.0 - value))


def _gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    for _ in range(tensor.ndim - 2):
        view = view.unsqueeze(-1)
    return torch.gather(tensor, 1, view.expand(-1, -1, *tensor.shape[2:]))


def _float_top_level_tensors(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote refinement inputs without changing indices or nested metadata."""

    return {
        key: (
            value.float()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        )
        for key, value in values.items()
    }


def _candidate_indices(
    logits: torch.Tensor,
    *,
    maximum: int,
    matched: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    scores = logits.sigmoid().amax(-1)
    count = min(int(maximum), int(scores.shape[1]))
    if count <= 0:
        return torch.empty(
            (scores.shape[0], 0),
            dtype=torch.long,
            device=scores.device,
        )
    top = torch.topk(scores, count, dim=1).indices
    if matched is None:
        return top
    rows: list[torch.Tensor] = []
    for batch_index, (prediction_indices, _target_indices) in enumerate(matched):
        required = prediction_indices.to(device=top.device, dtype=torch.long)
        ordered = torch.cat((required, top[batch_index]))
        # torch.unique sorts on some backends. Preserve the required matches
        # first so a low-scoring matched query cannot be dropped from training.
        keep: list[int] = []
        seen: set[int] = set()
        for value in ordered.tolist():
            if value not in seen:
                keep.append(value)
                seen.add(value)
            if len(keep) == count:
                break
        rows.append(torch.tensor(keep, device=top.device, dtype=torch.long))
    return torch.stack(rows)


def _grid_reference_points(
    features: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    points = []
    for feature in features:
        height, width = int(feature.shape[-2]), int(feature.shape[-1])
        y, x = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        points.append(
            torch.stack(
                ((x + 0.5) / max(1, width), (y + 0.5) / max(1, height)),
                dim=-1,
            ).reshape(-1, 2)
        )
    return torch.cat(points, dim=0)


def _deformable_value(
    memory: torch.Tensor,
    spatial_shapes: Sequence[Sequence[int]],
    *,
    heads: int,
    memory_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    value = memory
    if memory_mask is not None:
        value = value * memory_mask.to(value.dtype).unsqueeze(-1)
    batch, length, dimension = value.shape
    if dimension % heads:
        raise ValueError(f"dimension={dimension} must be divisible by heads={heads}")
    value = value.reshape(batch, length, heads, dimension // heads)
    split_shape = [int(height) * int(width) for height, width in spatial_shapes]
    return tuple(value.permute(0, 2, 3, 1).split(split_shape, dim=-1))


def _soft_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * cross_entropy
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


class PoseHungarianMatcher(nn.Module):
    """One-to-one pose matching with object, box, and visible-keypoint costs."""

    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_box: float = 2.0,
        cost_keypoint: float = 5.0,
    ) -> None:
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_box = float(cost_box)
        self.cost_keypoint = float(cost_keypoint)

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        logits = outputs["pred_pose_logits"]
        boxes = outputs["pred_pose_boxes"]
        keypoints = outputs["pred_keypoints"]
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target in enumerate(targets):
            target_boxes = target["boxes"]
            count = int(target_boxes.shape[0])
            if count == 0 or logits.shape[1] == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue
            class_cost = -logits[batch_index].sigmoid().reshape(-1, 1).expand(-1, count)
            box_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            target_keypoints = target["keypoints"]
            labeled = target["labeled"].to(dtype=keypoints.dtype)
            difference = (
                keypoints[batch_index, :, None]
                - target_keypoints[None]
            ).abs().sum(dim=-1)
            keypoint_cost = (
                difference * labeled.unsqueeze(0)
            ).sum(dim=-1) / labeled.sum(dim=-1).clamp_min(1.0).unsqueeze(0)
            cost = (
                self.cost_class * class_cost
                + self.cost_box * box_cost
                + self.cost_keypoint * keypoint_cost
            )
            cost = torch.nan_to_num(cost, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            source, destination = linear_sum_assignment(cost.detach().cpu())
            matches.append(
                (
                    torch.as_tensor(source, dtype=torch.long, device=logits.device),
                    torch.as_tensor(destination, dtype=torch.long, device=logits.device),
                )
            )
        return matches


class GroupedPoseDecoderLayer(nn.Module):
    """Grouped self-attention plus DEIM multi-scale deformable attention."""

    def __init__(
        self,
        dimension: int,
        *,
        heads: int,
        levels: int,
        points: int | list[int],
        feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.group_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = MSDeformableAttention(
            dimension,
            heads,
            levels,
            points,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.norm_group = nn.LayerNorm(dimension)
        self.norm_cross = nn.LayerNorm(dimension)
        self.norm_ffn = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        instance_queries: torch.Tensor,
        keypoint_queries: torch.Tensor,
        references: torch.Tensor,
        value: tuple[torch.Tensor, ...],
        spatial_shapes: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, poses, landmarks, dimension = keypoint_queries.shape
        grouped = torch.cat(
            (instance_queries.unsqueeze(2), keypoint_queries),
            dim=2,
        ).reshape(batch * poses, landmarks + 1, dimension)
        normalized = self.norm_group(grouped)
        attended, self_weights = self.group_attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        grouped = grouped + self.dropout(attended)
        grouped = grouped.reshape(batch, poses, landmarks + 1, dimension)

        flattened = grouped.reshape(batch, poses * (landmarks + 1), dimension)
        flat_references = references.reshape(
            batch,
            poses * (landmarks + 1),
            references.shape[-2],
            4,
        )
        cross = self.cross_attention(
            self.norm_cross(flattened),
            flat_references,
            value,
            spatial_shapes,
        )
        flattened = flattened + self.dropout(cross)
        flattened = flattened + self.dropout(self.ffn(self.norm_ffn(flattened)))
        grouped = flattened.reshape(batch, poses, landmarks + 1, dimension)
        return grouped[:, :, 0], grouped[:, :, 1:], self_weights


class IndependentPoseQueryDecoder(nn.Module):
    """Full-frame pose proposals independent of DEIM face proposals."""

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        dimension: int = 256,
        num_queries: int = 100,
        num_keypoints: int = 5,
        layers: int = 4,
        heads: int = 8,
        feedforward: int = 1024,
        deformable_points: int | list[int] = (4, 4, 4),
        dropout: float = 0.0,
        auxiliary_loss: bool = True,
        local_refinement: Sequence[Mapping[str, float]] = (),
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
    ) -> None:
        super().__init__()
        if num_keypoints != len(KEYPOINT_NAMES):
            raise ValueError(
                f"This dataset contract requires {len(KEYPOINT_NAMES)} keypoints, got {num_keypoints}"
            )
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"Unsupported state_head_mode={state_head_mode!r}; "
                f"expected one of {STATE_HEAD_MODES}"
            )
        self.dimension = int(dimension)
        self.num_queries = int(num_queries)
        self.num_keypoints = int(num_keypoints)
        self.heads = int(heads)
        self.auxiliary_loss = bool(auxiliary_loss)
        self.state_head_mode = str(state_head_mode)
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(channels), dimension, kernel_size=1, bias=False),
                    nn.GroupNorm(1, dimension),
                )
                for channels in in_channels
            ]
        )
        self.proposal_score = nn.Linear(dimension, 1)
        self.proposal_center = nn.Linear(dimension, 2)
        self.proposal_size = nn.Linear(dimension, 2)
        self.pose_query_embedding = nn.Parameter(torch.empty(num_queries, dimension))
        self.landmark_queries = nn.Parameter(torch.empty(num_keypoints, dimension))
        self.initial_keypoint_offset = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 2),
        )
        point_spec: int | list[int]
        if isinstance(deformable_points, Sequence) and not isinstance(deformable_points, (str, bytes)):
            point_spec = [int(value) for value in deformable_points]
        else:
            point_spec = int(deformable_points)
        self.layers = nn.ModuleList(
            [
                GroupedPoseDecoderLayer(
                    dimension,
                    heads=heads,
                    levels=len(in_channels),
                    points=point_spec,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.coordinate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, dimension),
                    nn.GELU(),
                    nn.Linear(dimension, 2),
                )
                for _ in range(layers)
            ]
        )
        self.center_heads = nn.ModuleList([nn.Linear(dimension, 2) for _ in range(layers)])
        self.size_heads = nn.ModuleList([nn.Linear(dimension, 2) for _ in range(layers)])
        self.pose_confidence_heads = nn.ModuleList(
            [nn.Linear(dimension, 1) for _ in range(layers)]
        )
        self.keypoint_confidence_heads = nn.ModuleList(
            [nn.Linear(dimension, 1) for _ in range(layers)]
        )
        # A shared head deliberately matches the trained Query+refinement
        # checkpoint and is refined additively at every decoder layer.
        self.visibility_head = nn.Linear(dimension, 1)
        self.visibility_delta_heads = nn.ModuleList(
            [nn.Linear(dimension, 1) for _ in range(layers)]
        )
        self.refinement_stages = nn.ModuleList(
            [
                CoordinateRefinementStage(
                    dimension,
                    radius_in_feature_cells=float(stage["radius_in_feature_cells"]),
                    maximum_delta=float(stage["maximum_delta"]),
                    samples_per_axis=int(stage.get("samples_per_axis", 3)),
                    state_head_mode=self.state_head_mode,
                )
                for stage in local_refinement
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pose_query_embedding, std=0.02)
        nn.init.trunc_normal_(self.landmark_queries, std=0.02)
        prior = math.log(0.01 / 0.99)
        nn.init.constant_(self.proposal_score.bias, prior)
        for head in self.pose_confidence_heads:
            nn.init.zeros_(head.bias)
        for collection in (
            self.coordinate_heads,
            self.center_heads,
            self.size_heads,
            self.visibility_delta_heads,
        ):
            for head in collection:
                output = head[-1] if isinstance(head, nn.Sequential) else head
                if isinstance(output, nn.Linear):
                    nn.init.zeros_(output.weight)
                    nn.init.zeros_(output.bias)
        final_offset = self.initial_keypoint_offset[-1]
        assert isinstance(final_offset, nn.Linear)
        nn.init.zeros_(final_offset.weight)
        template = torch.tensor(
            (
                (-0.18, -0.13),
                (0.18, -0.13),
                (0.00, 0.00),
                (-0.13, 0.17),
                (0.13, 0.17),
            ),
            dtype=final_offset.bias.dtype,
        )
        # The semantic template is injected through the learned landmark
        # embeddings; keep the shared output bias neutral.
        nn.init.zeros_(final_offset.bias)
        with torch.no_grad():
            self.landmark_queries.add_(
                F.pad(template, (0, self.dimension - 2)) * 0.1
            )

    def _project(
        self,
        encoded: Sequence[torch.Tensor],
        padding_mask: torch.Tensor | None,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        list[list[int]],
        torch.Tensor | None,
    ]:
        if len(encoded) != len(self.level_projections):
            raise ValueError(
                f"Expected {len(self.level_projections)} feature levels, got {len(encoded)}"
            )
        projected = [
            projection(feature)
            for projection, feature in zip(self.level_projections, encoded, strict=True)
        ]
        spatial_shapes = [
            [int(feature.shape[-2]), int(feature.shape[-1])]
            for feature in projected
        ]
        memory = torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in projected],
            dim=1,
        )
        memory_mask = None
        if padding_mask is not None:
            masks = [
                F.interpolate(
                    padding_mask[:, None].to(dtype=torch.float32),
                    size=feature.shape[-2:],
                    mode="nearest",
                ).squeeze(1).to(dtype=torch.bool)
                for feature in projected
            ]
            # True is padded externally; deformable values use one for valid.
            memory_mask = torch.cat([~mask.flatten(1) for mask in masks], dim=1)
            memory = memory * memory_mask.to(memory.dtype).unsqueeze(-1)
        return projected, memory, spatial_shapes, memory_mask

    def forward(
        self,
        encoded: Sequence[torch.Tensor],
        *,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        projected, memory, spatial_shapes, memory_mask = self._project(
            encoded,
            padding_mask,
        )
        proposal_logits_all = self.proposal_score(memory).squeeze(-1)
        if memory_mask is not None:
            proposal_logits_all = proposal_logits_all.masked_fill(
                ~memory_mask,
                torch.finfo(proposal_logits_all.dtype).min,
            )
        query_count = min(self.num_queries, int(memory.shape[1]))
        proposal_indices = torch.topk(
            proposal_logits_all,
            query_count,
            dim=1,
        ).indices
        selected_proposal_logits = _gather(
            proposal_logits_all.unsqueeze(-1),
            proposal_indices,
        ).squeeze(-1)
        gathered_memory = _gather(memory, proposal_indices)
        instance_queries = (
            gathered_memory
            + self.pose_query_embedding[:query_count].unsqueeze(0)
        )
        grid = _grid_reference_points(
            projected,
            dtype=memory.dtype,
            device=memory.device,
        ).unsqueeze(0).expand(memory.shape[0], -1, -1)
        centers = _gather(grid, proposal_indices)
        centers = torch.sigmoid(
            _inverse_sigmoid(centers) + self.proposal_center(instance_queries)
        )
        sizes = (0.02 + 0.48 * torch.sigmoid(self.proposal_size(instance_queries))).clamp(
            1.0e-4,
            1.0,
        )
        keypoint_queries = (
            instance_queries.unsqueeze(2)
            + self.landmark_queries.view(1, 1, self.num_keypoints, self.dimension)
        )
        initial_delta = torch.tanh(self.initial_keypoint_offset(keypoint_queries))
        coordinates = torch.sigmoid(
            _inverse_sigmoid(centers.unsqueeze(2).expand_as(initial_delta))
            + initial_delta
        )
        visibility = self.visibility_head(keypoint_queries).squeeze(-1)
        value = _deformable_value(
            memory,
            spatial_shapes,
            heads=self.heads,
            memory_mask=memory_mask,
        )
        layer_outputs: list[dict[str, torch.Tensor]] = []
        attention_weights: list[torch.Tensor] = []
        refinement_offset = max(0, len(self.layers) - len(self.refinement_stages))
        for layer_index, layer in enumerate(self.layers):
            pose_reference = torch.cat((centers, sizes), dim=-1)
            references = torch.cat(
                (
                    pose_reference.unsqueeze(2),
                    F.pad(coordinates, (0, 2), value=0.0),
                ),
                dim=2,
            )
            # DEIM's MSDA represents the levels by contiguous sampling-point
            # groups. Its 4-D reference form therefore uses a singleton level
            # axis and broadcasts xywh over those groups.
            references_xy = references.unsqueeze(-2)
            instance_queries, keypoint_queries, self_attention = layer(
                instance_queries,
                keypoint_queries,
                references_xy,
                value,
                spatial_shapes,
            )
            centers = torch.sigmoid(
                _inverse_sigmoid(centers)
                + self.center_heads[layer_index](instance_queries)
            )
            sizes = torch.sigmoid(
                _inverse_sigmoid(sizes)
                + self.size_heads[layer_index](instance_queries)
            ).clamp_min(1.0e-4)
            # The pretrained crop Query head predicts coordinates in normalized
            # face-crop space. Reinterpret them as local pose-box coordinates
            # instead of incorrectly treating those absolute logits as global
            # residuals. The retained previous reference makes the update
            # iterative and keeps deformable sampling stable early in training.
            local_coordinates = torch.sigmoid(
                self.coordinate_heads[layer_index](keypoint_queries)
            )
            predicted_coordinates = (
                centers.unsqueeze(2)
                + (local_coordinates - 0.5) * sizes.unsqueeze(2)
            ).clamp(0.0, 1.0)
            coordinates = (
                0.25 * coordinates + 0.75 * predicted_coordinates
            ).clamp(0.0, 1.0)
            visibility = (
                visibility
                + self.visibility_delta_heads[layer_index](keypoint_queries).squeeze(-1)
            )
            keypoint_presence = None
            refinement_index = layer_index - refinement_offset
            if 0 <= refinement_index < len(self.refinement_stages):
                flat_coordinates = coordinates.flatten(1, 2)
                flat_queries = keypoint_queries.flatten(1, 2)
                flat_visibility = visibility.flatten(1, 2)
                if self.state_head_mode == FACTORIZED_STATE_HEAD:
                    keypoint_presence = self.keypoint_confidence_heads[
                        layer_index
                    ](keypoint_queries).squeeze(-1)
                    (
                        flat_coordinates,
                        flat_presence,
                        flat_visibility,
                        residual,
                    ) = self.refinement_stages[refinement_index](
                        projected[0],
                        flat_queries,
                        flat_coordinates,
                        flat_visibility,
                        keypoint_presence.flatten(1, 2),
                    )
                    keypoint_presence = flat_presence.reshape_as(
                        keypoint_presence
                    )
                else:
                    (
                        flat_coordinates,
                        flat_visibility,
                        residual,
                    ) = self.refinement_stages[refinement_index](
                        projected[0],
                        flat_queries,
                        flat_coordinates,
                        flat_visibility,
                    )
                coordinates = flat_coordinates.reshape_as(coordinates)
                visibility = flat_visibility.reshape_as(visibility)
                keypoint_queries = (
                    keypoint_queries + residual.reshape_as(keypoint_queries)
                )
            if keypoint_presence is None:
                keypoint_presence = self.keypoint_confidence_heads[layer_index](
                    keypoint_queries
                ).squeeze(-1)
            pose_boxes = torch.cat((centers, sizes), dim=-1)
            layer_output = {
                "pred_pose_logits": self.pose_confidence_heads[layer_index](
                    instance_queries
                ).squeeze(-1)
                + selected_proposal_logits,
                "pred_pose_boxes": pose_boxes,
                "pred_keypoints": coordinates,
                "pred_keypoint_logits": keypoint_presence,
                "pred_visibility_logits": visibility,
                "pose_embeddings": instance_queries,
                "keypoint_embeddings": keypoint_queries,
            }
            if self.state_head_mode == FACTORIZED_STATE_HEAD:
                layer_output["pred_state_probabilities"] = (
                    factorized_state_probabilities(
                        keypoint_presence,
                        visibility,
                    )
                )
            layer_outputs.append(layer_output)
            if return_attention:
                attention_weights.append(self_attention)
        output: dict[str, Any] = dict(layer_outputs[-1])
        output.update(
            {
                "proposal_logits": selected_proposal_logits,
                "proposal_indices": proposal_indices,
                "projected_features": projected,
                "memory": memory,
                "memory_mask": memory_mask,
                "spatial_shapes": spatial_shapes,
                "attention_weights": attention_weights,
                "auxiliary_outputs": (
                    layer_outputs[:-1]
                    if self.auxiliary_loss
                    else []
                ),
            }
        )
        return output


class FacePoseAssociation(nn.Module):
    """Learned appearance-and-geometry association for independent candidates."""

    def __init__(
        self,
        face_dimension: int,
        pose_dimension: int,
        *,
        appearance_dimension: int = 128,
        hidden_dimension: int = 256,
        inside_temperature: float = 12.0,
    ) -> None:
        super().__init__()
        self.inside_temperature = float(inside_temperature)
        self.face_dimension = int(face_dimension)
        self.pose_dimension = int(pose_dimension)
        self.face_projection = nn.Linear(face_dimension, appearance_dimension)
        self.refined_face_projection = (
            nn.Identity()
            if face_dimension == pose_dimension
            else nn.Linear(pose_dimension, appearance_dimension)
        )
        self.pose_projection = nn.Linear(pose_dimension, appearance_dimension)
        self.geometry_dimension = 15
        self.scorer = nn.Sequential(
            nn.Linear(appearance_dimension + self.geometry_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        final_scorer = self.scorer[-1]
        assert isinstance(final_scorer, nn.Linear)
        nn.init.normal_(final_scorer.weight, std=0.01)
        nn.init.constant_(final_scorer.bias, -2.0)

    def _geometry(
        self,
        face_boxes: torch.Tensor,
        face_ellipses: torch.Tensor,
        face_logits: torch.Tensor,
        pose_boxes: torch.Tensor,
        keypoints: torch.Tensor,
        pose_logits: torch.Tensor,
        keypoint_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = 1.0e-5
        face_center = face_boxes[..., :2]
        face_size = face_boxes[..., 2:].clamp_min(epsilon)
        weights = (
            keypoint_logits.sigmoid() * (0.25 + 0.75 * visibility_logits.sigmoid())
        ).clamp_min(epsilon)
        pose_center = (
            keypoints * weights.unsqueeze(-1)
        ).sum(dim=-2) / weights.sum(dim=-1, keepdim=True)
        centered = keypoints - pose_center.unsqueeze(-2)
        spread = torch.sqrt(
            (
                centered.square() * weights.unsqueeze(-1)
            ).sum(dim=-2)
            / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
            + epsilon
        )
        center_delta = (
            pose_center[:, None] - face_center[:, :, None]
        ) / face_size[:, :, None].clamp_min(epsilon)
        log_size_ratio = torch.log(
            (pose_boxes[:, None, :, 2:] + epsilon)
            / (face_size[:, :, None] + epsilon)
        )
        spread_ratio = spread[:, None] / face_size[:, :, None].clamp_min(epsilon)

        face_top_left = face_center - 0.5 * face_size
        local = (
            keypoints[:, None] - face_top_left[:, :, None, None]
        ) / face_size[:, :, None, None].clamp_min(epsilon)
        inside_margin = torch.minimum(local, 1.0 - local).amin(dim=-1)
        inside_box = torch.sigmoid(
            self.inside_temperature * inside_margin
        ).mean(dim=-1)

        ellipse_center = face_ellipses[..., :2]
        axes = face_ellipses[..., 2:4].clamp_min(epsilon)
        theta = face_ellipses[..., 4]
        ellipse_delta = (
            keypoints[:, None] - ellipse_center[:, :, None, None]
        )
        cosine = torch.cos(theta)[:, :, None, None]
        sine = torch.sin(theta)[:, :, None, None]
        local_x = (
            cosine * ellipse_delta[..., 0]
            + sine * ellipse_delta[..., 1]
        ) / axes[..., 0, None, None]
        local_y = (
            -sine * ellipse_delta[..., 0]
            + cosine * ellipse_delta[..., 1]
        ) / axes[..., 1, None, None]
        radial = local_x.square() + local_y.square()
        inside_ellipse = torch.sigmoid(
            self.inside_temperature * (1.0 - radial)
        ).mean(dim=-1)

        eye_vector = keypoints[..., 1, :] - keypoints[..., 0, :]
        mouth_vector = keypoints[..., 4, :] - keypoints[..., 3, :]
        direction = F.normalize(eye_vector + mouth_vector, dim=-1, eps=epsilon)
        face_direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        direction_dot = (
            face_direction[:, :, None] * direction[:, None]
        ).sum(dim=-1)
        angle_consistency = 2.0 * direction_dot.square() - 1.0

        face_confidence = face_logits.sigmoid().amax(dim=-1)[:, :, None]
        pose_confidence = pose_logits.sigmoid()[:, None]
        mean_visibility = visibility_logits.sigmoid().mean(dim=-1)[:, None]
        mean_keypoint_confidence = keypoint_logits.sigmoid().mean(dim=-1)[:, None]
        geometry = torch.cat(
            (
                center_delta,
                log_size_ratio,
                spread_ratio,
                inside_box.unsqueeze(-1),
                inside_ellipse.unsqueeze(-1),
                angle_consistency.unsqueeze(-1),
                face_confidence.expand(-1, -1, pose_boxes.shape[1]).unsqueeze(-1),
                pose_confidence.expand(-1, face_boxes.shape[1], -1).unsqueeze(-1),
                mean_visibility.expand(-1, face_boxes.shape[1], -1).unsqueeze(-1),
                mean_keypoint_confidence.expand(
                    -1,
                    face_boxes.shape[1],
                    -1,
                ).unsqueeze(-1),
                (pose_boxes[:, None, :, :2] - face_center[:, :, None]),
            ),
            dim=-1,
        )
        if geometry.shape[-1] != self.geometry_dimension:
            raise RuntimeError(
                f"Association geometry contract changed: {geometry.shape[-1]} "
                f"!= {self.geometry_dimension}"
            )
        return geometry

    def forward(
        self,
        face_embeddings: torch.Tensor,
        pose_embeddings: torch.Tensor,
        *,
        face_boxes: torch.Tensor,
        face_ellipses: torch.Tensor,
        face_logits: torch.Tensor,
        pose_boxes: torch.Tensor,
        keypoints: torch.Tensor,
        pose_logits: torch.Tensor,
        keypoint_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, faces = face_embeddings.shape[:2]
        poses = pose_embeddings.shape[1]
        if faces == 0 or poses == 0:
            logits = face_embeddings.new_zeros((batch, faces, poses))
            geometry = face_embeddings.new_zeros(
                (batch, faces, poses, self.geometry_dimension)
            )
            return {"logits": logits, "geometry": geometry}
        if face_embeddings.shape[-1] == self.face_dimension:
            projected_face = self.face_projection(face_embeddings)
        elif face_embeddings.shape[-1] == self.pose_dimension:
            projected_face = self.refined_face_projection(face_embeddings)
        else:
            raise ValueError(
                "Association received an unsupported face embedding width: "
                f"{face_embeddings.shape[-1]}"
            )
        face = F.normalize(projected_face, dim=-1)
        pose = F.normalize(self.pose_projection(pose_embeddings), dim=-1)
        appearance = (face[:, :, None] - pose[:, None]).abs()
        geometry = self._geometry(
            face_boxes,
            face_ellipses,
            face_logits,
            pose_boxes,
            keypoints,
            pose_logits,
            keypoint_logits,
            visibility_logits,
        )
        logits = self.scorer(torch.cat((appearance, geometry), dim=-1)).squeeze(-1)
        return {"logits": logits, "geometry": geometry}


class CooperativeDecoderLayer(nn.Module):
    """Association-masked bidirectional attention with confidence-aware gates."""

    def __init__(
        self,
        face_dimension: int,
        dimension: int,
        *,
        heads: int,
        levels: int,
        deformable_points: int | list[int],
        feedforward: int,
        topk_associations: int,
        dropout: float,
        geometry_dimension: int,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.topk_associations = int(topk_associations)
        self.face_input = (
            nn.Identity()
            if face_dimension == dimension
            else nn.Linear(face_dimension, dimension)
        )
        self.face_to_pose_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pose_to_face_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_inputs = 2 * dimension + geometry_dimension + 4
        self.keypoint_to_face_gate = nn.Sequential(
            nn.Linear(gate_inputs, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Sigmoid(),
        )
        self.face_to_keypoint_gate = nn.Sequential(
            nn.Linear(gate_inputs, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Sigmoid(),
        )
        for gate in (self.keypoint_to_face_gate, self.face_to_keypoint_gate):
            output_layer = gate[-2]
            assert isinstance(output_layer, nn.Linear)
            nn.init.normal_(output_layer.weight, std=1.0e-3)
            nn.init.constant_(output_layer.bias, math.log(0.1 / 0.9))
        self.face_cross_attention = MSDeformableAttention(
            dimension,
            heads,
            levels,
            deformable_points,
        )
        self.pose_cross_attention = MSDeformableAttention(
            dimension,
            heads,
            levels,
            deformable_points,
        )
        self.face_ffn = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.pose_ffn = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.face_norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(3)])
        self.pose_norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(3)])
        self.face_box_head = nn.Linear(dimension, 4)
        self.face_ellipse_head = nn.Linear(dimension, 6)
        self.face_confidence_head = nn.Linear(dimension, 1)
        self.pose_box_head = nn.Linear(dimension, 4)
        self.pose_confidence_head = nn.Linear(dimension, 1)
        self.keypoint_coordinate_head = nn.Linear(dimension, 2)
        self.keypoint_confidence_head = nn.Linear(dimension, 1)
        self.visibility_head = nn.Linear(dimension, 1)
        self.dropout = nn.Dropout(dropout)
        for head in (
            self.face_box_head,
            self.face_ellipse_head,
            self.face_confidence_head,
            self.pose_box_head,
            self.pose_confidence_head,
            self.keypoint_coordinate_head,
            self.keypoint_confidence_head,
            self.visibility_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _topk_pose_context(
        pose_tokens: torch.Tensor,
        association_logits: torch.Tensor,
        maximum: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, faces, poses = association_logits.shape
        count = min(maximum, poses)
        indices = torch.topk(association_logits, count, dim=-1).indices
        batch_indices = torch.arange(batch, device=indices.device)[:, None, None]
        gathered = pose_tokens[batch_indices, indices]
        return gathered.flatten(2, 3), indices

    @staticmethod
    def _topk_face_context(
        face_tokens: torch.Tensor,
        association_logits: torch.Tensor,
        maximum: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transposed = association_logits.transpose(1, 2)
        batch, poses, faces = transposed.shape
        count = min(maximum, faces)
        indices = torch.topk(transposed, count, dim=-1).indices
        batch_indices = torch.arange(batch, device=indices.device)[:, None, None]
        return face_tokens[batch_indices, indices], indices

    @staticmethod
    def _selected_pair_features(
        pair_features: torch.Tensor,
        indices: torch.Tensor,
        *,
        pose_to_face: bool,
    ) -> torch.Tensor:
        batch = pair_features.shape[0]
        batch_indices = torch.arange(batch, device=indices.device)[:, None, None]
        if pose_to_face:
            # pair_features: B,F,P,G; indices: B,P,K
            pose_indices = torch.arange(
                pair_features.shape[2],
                device=indices.device,
            )[None, :, None]
            return pair_features[batch_indices, indices, pose_indices]
        # pair_features: B,F,P,G; indices: B,F,K
        face_indices = torch.arange(
            pair_features.shape[1],
            device=indices.device,
        )[None, :, None]
        return pair_features[batch_indices, face_indices, indices]

    def forward(
        self,
        face_embeddings: torch.Tensor,
        pose_embeddings: torch.Tensor,
        keypoint_embeddings: torch.Tensor,
        *,
        association_logits: torch.Tensor,
        association_geometry: torch.Tensor,
        face_boxes: torch.Tensor,
        face_ellipses: torch.Tensor,
        face_logits: torch.Tensor,
        pose_boxes: torch.Tensor,
        pose_logits: torch.Tensor,
        keypoints: torch.Tensor,
        keypoint_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
        memory_value: tuple[torch.Tensor, ...],
        spatial_shapes: Sequence[Sequence[int]],
        allow_face_to_pose: torch.Tensor,
        allow_pose_to_face: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        face_embeddings = self.face_input(face_embeddings)
        batch, faces, dimension = face_embeddings.shape
        poses = pose_embeddings.shape[1]
        landmarks = keypoint_embeddings.shape[2]
        if faces and poses:
            pose_tokens = torch.cat(
                (pose_embeddings.unsqueeze(2), keypoint_embeddings),
                dim=2,
            )
            pose_context, face_pair_indices = self._topk_pose_context(
                pose_tokens,
                association_logits,
                self.topk_associations,
            )
            face_flat = face_embeddings.reshape(batch * faces, 1, dimension)
            context_flat = pose_context.reshape(
                batch * faces,
                pose_context.shape[2],
                dimension,
            )
            face_delta, _ = self.pose_to_face_attention(
                self.face_norms[0](face_flat),
                context_flat,
                context_flat,
                need_weights=False,
            )
            face_delta = face_delta.reshape(batch, faces, dimension)
            face_geometry = self._selected_pair_features(
                association_geometry,
                face_pair_indices,
                pose_to_face=False,
            ).mean(dim=2)
            selected_pose_confidence = _gather(
                pose_logits.unsqueeze(-1),
                face_pair_indices.reshape(batch, -1),
            ).reshape(batch, faces, -1).sigmoid().mean(dim=-1, keepdim=True)
            selected_visibility = _gather(
                visibility_logits.mean(dim=-1, keepdim=True),
                face_pair_indices.reshape(batch, -1),
            ).reshape(batch, faces, -1).sigmoid().mean(dim=-1, keepdim=True)
            face_association = association_logits.topk(
                min(self.topk_associations, poses),
                dim=-1,
            ).values.sigmoid().mean(dim=-1, keepdim=True)
            face_gate_input = torch.cat(
                (
                    face_embeddings,
                    face_delta,
                    face_geometry,
                    selected_pose_confidence,
                    selected_visibility,
                    face_association,
                    face_logits.sigmoid().amax(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            face_gate = self.keypoint_to_face_gate(face_gate_input)
            face_embeddings = (
                face_embeddings
                + allow_pose_to_face[:, None, None]
                * face_gate
                * self.dropout(face_delta)
            )

            face_context, pose_pair_indices = self._topk_face_context(
                face_embeddings,
                association_logits,
                self.topk_associations,
            )
            pose_tokens_flat = pose_tokens.reshape(
                batch * poses,
                landmarks + 1,
                dimension,
            )
            face_context_flat = face_context.reshape(
                batch * poses,
                face_context.shape[2],
                dimension,
            )
            pose_delta, _ = self.face_to_pose_attention(
                self.pose_norms[0](pose_tokens_flat),
                face_context_flat,
                face_context_flat,
                need_weights=False,
            )
            pose_delta = pose_delta.reshape(
                batch,
                poses,
                landmarks + 1,
                dimension,
            )
            pose_geometry = self._selected_pair_features(
                association_geometry,
                pose_pair_indices,
                pose_to_face=True,
            ).mean(dim=2)
            selected_face_logits = _gather(
                face_logits.sigmoid().amax(dim=-1, keepdim=True),
                pose_pair_indices.reshape(batch, -1),
            ).reshape(batch, poses, -1).mean(dim=-1, keepdim=True)
            pose_association = association_logits.transpose(1, 2).topk(
                min(self.topk_associations, faces),
                dim=-1,
            ).values.sigmoid().mean(dim=-1, keepdim=True)
            source_visibility = visibility_logits.sigmoid().mean(
                dim=-1,
                keepdim=True,
            )
            scalar_features = torch.cat(
                (
                    selected_face_logits,
                    source_visibility,
                    pose_association,
                    pose_logits.sigmoid().unsqueeze(-1),
                ),
                dim=-1,
            )
            scalar_features = scalar_features.unsqueeze(2).expand(
                -1,
                -1,
                landmarks + 1,
                -1,
            )
            geometry_features = pose_geometry.unsqueeze(2).expand(
                -1,
                -1,
                landmarks + 1,
                -1,
            )
            pose_gate_input = torch.cat(
                (
                    pose_tokens,
                    pose_delta,
                    geometry_features,
                    scalar_features,
                ),
                dim=-1,
            )
            pose_gate = self.face_to_keypoint_gate(pose_gate_input)
            pose_tokens = (
                pose_tokens
                + allow_face_to_pose[:, None, None, None]
                * pose_gate
                * self.dropout(pose_delta)
            )
            pose_embeddings = pose_tokens[:, :, 0]
            keypoint_embeddings = pose_tokens[:, :, 1:]
        else:
            face_gate = face_embeddings.new_zeros(face_embeddings.shape)
            pose_gate = keypoint_embeddings.new_zeros(
                (
                    batch,
                    poses,
                    landmarks + 1,
                    dimension,
                )
            )

        face_reference = face_boxes.unsqueeze(-2)
        if faces:
            face_cross = self.face_cross_attention(
                self.face_norms[1](face_embeddings),
                face_reference,
                memory_value,
                spatial_shapes,
            )
            face_embeddings = face_embeddings + self.dropout(face_cross)
            face_embeddings = face_embeddings + self.dropout(
                self.face_ffn(self.face_norms[2](face_embeddings))
            )

        pose_tokens = torch.cat(
            (pose_embeddings.unsqueeze(2), keypoint_embeddings),
            dim=2,
        )
        if poses:
            keypoint_references = torch.cat(
                (
                    keypoints,
                    pose_boxes[..., 2:].unsqueeze(2).expand(
                        -1,
                        -1,
                        landmarks,
                        -1,
                    ),
                ),
                dim=-1,
            )
            references = torch.cat(
                (pose_boxes.unsqueeze(2), keypoint_references),
                dim=2,
            ).reshape(batch, poses * (landmarks + 1), 1, 4)
            flat = pose_tokens.reshape(batch, poses * (landmarks + 1), dimension)
            pose_cross = self.pose_cross_attention(
                self.pose_norms[1](flat),
                references,
                memory_value,
                spatial_shapes,
            )
            flat = flat + self.dropout(pose_cross)
            flat = flat + self.dropout(self.pose_ffn(self.pose_norms[2](flat)))
            pose_tokens = flat.reshape(batch, poses, landmarks + 1, dimension)
            pose_embeddings = pose_tokens[:, :, 0]
            keypoint_embeddings = pose_tokens[:, :, 1:]

        face_boxes = torch.sigmoid(
            _inverse_sigmoid(face_boxes)
            + self.face_box_head(face_embeddings)
        )
        raw_ellipse_delta = self.face_ellipse_head(face_embeddings)
        ellipse_center = torch.sigmoid(
            _inverse_sigmoid(face_ellipses[..., :2])
            + raw_ellipse_delta[..., :2]
        )
        axes = (
            face_ellipses[..., 2:4].clamp_min(1.0e-5)
            * torch.exp(raw_ellipse_delta[..., 2:4].clamp(-2.0, 2.0))
        )
        angle_vector = torch.stack(
            (
                torch.cos(2.0 * face_ellipses[..., 4]),
                torch.sin(2.0 * face_ellipses[..., 4]),
            ),
            dim=-1,
        )
        angle_vector = F.normalize(
            angle_vector + raw_ellipse_delta[..., 4:6],
            dim=-1,
            eps=1.0e-6,
        )
        face_ellipses = canonicalize_ellipses(
            torch.cat(
                (
                    ellipse_center,
                    axes,
                    (
                        0.5
                        * torch.atan2(
                            angle_vector[..., 1],
                            angle_vector[..., 0],
                        )
                    ).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        face_logits = (
            face_logits
            + self.face_confidence_head(face_embeddings).expand_as(face_logits)
        )
        pose_boxes = torch.sigmoid(
            _inverse_sigmoid(pose_boxes)
            + self.pose_box_head(pose_embeddings)
        )
        pose_logits = (
            pose_logits + self.pose_confidence_head(pose_embeddings).squeeze(-1)
        )
        keypoints = torch.sigmoid(
            _inverse_sigmoid(keypoints)
            + self.keypoint_coordinate_head(keypoint_embeddings)
        )
        keypoint_logits = (
            keypoint_logits
            + self.keypoint_confidence_head(keypoint_embeddings).squeeze(-1)
        )
        visibility_logits = (
            visibility_logits
            + self.visibility_head(keypoint_embeddings).squeeze(-1)
        )
        return {
            "face_embeddings": face_embeddings,
            "pose_embeddings": pose_embeddings,
            "keypoint_embeddings": keypoint_embeddings,
            "pred_face_boxes": face_boxes,
            "pred_face_ellipses": face_ellipses,
            "pred_face_logits": face_logits,
            "pred_pose_boxes": pose_boxes,
            "pred_pose_logits": pose_logits,
            "pred_keypoints": keypoints,
            "pred_keypoint_logits": keypoint_logits,
            "pred_visibility_logits": visibility_logits,
            "gate_keypoint_to_face": face_gate,
            "gate_face_to_pose": pose_gate,
        }


class BidirectionalCooperativeDecoder(nn.Module):
    def __init__(
        self,
        face_dimension: int,
        pose_dimension: int,
        association: FacePoseAssociation,
        *,
        layers: int = 3,
        heads: int = 8,
        levels: int = 3,
        deformable_points: int | list[int] = (4, 4, 4),
        feedforward: int = 1024,
        topk_associations: int = 4,
        dropout: float = 0.0,
        drop_face_to_pose: float = 0.10,
        drop_pose_to_face: float = 0.10,
        auxiliary_loss: bool = True,
    ) -> None:
        super().__init__()
        self.association = association
        self.dimension = int(pose_dimension)
        self.heads = int(heads)
        self.drop_face_to_pose = float(drop_face_to_pose)
        self.drop_pose_to_face = float(drop_pose_to_face)
        self.auxiliary_loss = bool(auxiliary_loss)
        self.layers = nn.ModuleList(
            [
                CooperativeDecoderLayer(
                    face_dimension if index == 0 else pose_dimension,
                    pose_dimension,
                    heads=heads,
                    levels=levels,
                    deformable_points=deformable_points,
                    feedforward=feedforward,
                    topk_associations=topk_associations,
                    dropout=dropout,
                    geometry_dimension=association.geometry_dimension,
                )
                for index in range(layers)
            ]
        )

    def enforce_face_to_pose_only(self) -> None:
        """Disable the reverse keypoint-to-FACE path for a causal cascade."""

        for layer in self.layers:
            for module in (
                layer.pose_to_face_attention,
                layer.keypoint_to_face_gate,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _branch_masks(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        override: tuple[bool, bool] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if override is not None:
            return (
                torch.full((batch,), float(override[0]), device=device, dtype=dtype),
                torch.full((batch,), float(override[1]), device=device, dtype=dtype),
            )
        if not self.training:
            ones = torch.ones((batch,), device=device, dtype=dtype)
            return ones, ones
        face_to_pose = (
            torch.rand((batch,), device=device) >= self.drop_face_to_pose
        ).to(dtype)
        pose_to_face = (
            torch.rand((batch,), device=device) >= self.drop_pose_to_face
        ).to(dtype)
        return face_to_pose, pose_to_face

    def forward(
        self,
        face: Mapping[str, torch.Tensor],
        pose: Mapping[str, torch.Tensor],
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        spatial_shapes: Sequence[Sequence[int]],
        branch_override: tuple[bool, bool] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, torch.Tensor] = {
            "face_embeddings": face["embeddings"],
            "pose_embeddings": pose["pose_embeddings"],
            "keypoint_embeddings": pose["keypoint_embeddings"],
            "pred_face_boxes": face["pred_boxes"],
            "pred_face_ellipses": face["pred_ellipses"],
            "pred_face_logits": face["pred_logits"],
            "pred_pose_boxes": pose["pred_pose_boxes"],
            "pred_pose_logits": pose["pred_pose_logits"],
            "pred_keypoints": pose["pred_keypoints"],
            "pred_keypoint_logits": pose["pred_keypoint_logits"],
            "pred_visibility_logits": pose["pred_visibility_logits"],
        }
        allow_face_to_pose, allow_pose_to_face = self._branch_masks(
            state["face_embeddings"].shape[0],
            device=state["face_embeddings"].device,
            dtype=state["face_embeddings"].dtype,
            override=branch_override,
        )
        memory_value = _deformable_value(
            memory,
            spatial_shapes,
            heads=self.heads,
            memory_mask=memory_mask,
        )
        layer_outputs: list[dict[str, torch.Tensor]] = []
        associations: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            association = self.association(
                state["face_embeddings"],
                state["pose_embeddings"],
                face_boxes=state["pred_face_boxes"],
                face_ellipses=state["pred_face_ellipses"],
                face_logits=state["pred_face_logits"],
                pose_boxes=state["pred_pose_boxes"],
                keypoints=state["pred_keypoints"],
                pose_logits=state["pred_pose_logits"],
                keypoint_logits=state["pred_keypoint_logits"],
                visibility_logits=state["pred_visibility_logits"],
            )
            associations.append(association)
            state = layer(
                state["face_embeddings"],
                state["pose_embeddings"],
                state["keypoint_embeddings"],
                association_logits=association["logits"],
                association_geometry=association["geometry"],
                face_boxes=state["pred_face_boxes"],
                face_ellipses=state["pred_face_ellipses"],
                face_logits=state["pred_face_logits"],
                pose_boxes=state["pred_pose_boxes"],
                pose_logits=state["pred_pose_logits"],
                keypoints=state["pred_keypoints"],
                keypoint_logits=state["pred_keypoint_logits"],
                visibility_logits=state["pred_visibility_logits"],
                memory_value=memory_value,
                spatial_shapes=spatial_shapes,
                allow_face_to_pose=allow_face_to_pose,
                allow_pose_to_face=allow_pose_to_face,
            )
            layer_outputs.append(dict(state))
        final_association = self.association(
            state["face_embeddings"],
            state["pose_embeddings"],
            face_boxes=state["pred_face_boxes"],
            face_ellipses=state["pred_face_ellipses"],
            face_logits=state["pred_face_logits"],
            pose_boxes=state["pred_pose_boxes"],
            keypoints=state["pred_keypoints"],
            pose_logits=state["pred_pose_logits"],
            keypoint_logits=state["pred_keypoint_logits"],
            visibility_logits=state["pred_visibility_logits"],
        )
        output: dict[str, Any] = dict(state)
        output.update(
            {
                "association_logits": final_association["logits"],
                "association_geometry": final_association["geometry"],
                "association_layers": associations,
                "auxiliary_outputs": (
                    layer_outputs[:-1]
                    if self.auxiliary_loss
                    else []
                ),
                "branch_face_to_pose_enabled": allow_face_to_pose,
                "branch_pose_to_face_enabled": allow_pose_to_face,
            }
        )
        return output


class CooperativeCriterion(nn.Module):
    """Independent, association, and refined objectives in one criterion."""

    def __init__(
        self,
        pose_matcher: PoseHungarianMatcher,
        *,
        weights: Mapping[str, float] | None = None,
        auxiliary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        configured = dict(weights or {})
        self.weights = {
            "keypoint_independent": float(
                configured.get("keypoint_independent", 10.0)
            ),
            "association": float(configured.get("association", 1.0)),
            "face_refined": float(configured.get("face_refined", 1.0)),
            "keypoint_refined": float(configured.get("keypoint_refined", 10.0)),
            "visibility": float(configured.get("visibility", 1.0)),
            "pose_box": float(configured.get("pose_box", 2.0)),
            "ellipse_refined": float(configured.get("ellipse_refined", 2.0)),
            "keypoint_local": float(configured.get("keypoint_local", 1.0)),
            "keypoint_uncertainty": float(
                configured.get("keypoint_uncertainty", 0.1)
            ),
            "face_evidence": float(configured.get("face_evidence", 1.0)),
            "localization_quality": float(
                configured.get("localization_quality", 0.5)
            ),
            "non_regression": float(configured.get("non_regression", 0.5)),
        }
        self.auxiliary_weight = float(auxiliary_weight)
        self.pose_matcher = pose_matcher

    @staticmethod
    def _matched_positions(
        candidates: torch.Tensor,
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        selected = []
        for batch_index, (source, target) in enumerate(matches):
            positions = []
            targets = []
            for source_index, target_index in zip(source.tolist(), target.tolist()):
                found = torch.nonzero(
                    candidates[batch_index] == int(source_index),
                    as_tuple=False,
                )
                if found.numel():
                    positions.append(int(found[0, 0]))
                    targets.append(int(target_index))
            selected.append(
                (
                    torch.tensor(
                        positions,
                        dtype=torch.long,
                        device=candidates.device,
                    ),
                    torch.tensor(
                        targets,
                        dtype=torch.long,
                        device=candidates.device,
                    ),
                )
            )
        return selected

    def _pose_losses(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        logits = output["pred_pose_logits"]
        keypoints = output["pred_keypoints"]
        boxes = output["pred_pose_boxes"]
        keypoint_logits = output["pred_keypoint_logits"]
        visibility_logits = output["pred_visibility_logits"]
        object_target = torch.zeros_like(logits)
        coordinate_terms = []
        box_terms = []
        keypoint_presence_terms = []
        visibility_terms = []
        positives = 0
        for batch_index, ((source, destination), target) in enumerate(
            zip(matches, targets, strict=True)
        ):
            if source.numel() == 0:
                continue
            object_target[batch_index, source] = 1.0
            labeled = target["labeled"][destination].to(dtype=keypoints.dtype)
            visible = target["visible"][destination].to(dtype=keypoints.dtype)
            denominator = labeled.sum().clamp_min(1.0)
            coordinate_terms.append(
                (
                    F.smooth_l1_loss(
                        keypoints[batch_index, source],
                        target["keypoints"][destination],
                        reduction="none",
                    ).sum(dim=-1)
                    * labeled
                ).sum()
                / denominator
            )
            box_terms.append(
                F.l1_loss(
                    boxes[batch_index, source],
                    target["boxes"][destination],
                    reduction="mean",
                )
            )
            keypoint_presence_terms.append(
                F.binary_cross_entropy_with_logits(
                    keypoint_logits[batch_index, source],
                    labeled,
                    reduction="mean",
                )
            )
            visibility_terms.append(
                (
                    F.binary_cross_entropy_with_logits(
                        visibility_logits[batch_index, source],
                        visible,
                        reduction="none",
                    )
                    * labeled
                ).sum()
                / denominator
            )
            positives += int(source.numel())
        zero = logits.sum() * 0.0
        normalizer = max(1, positives)
        return {
            f"{prefix}_object": _soft_focal_loss(
                logits,
                object_target,
                reduction="sum",
            )
            / normalizer,
            f"{prefix}_keypoint": (
                torch.stack(coordinate_terms).mean()
                if coordinate_terms
                else zero
            ),
            f"{prefix}_pose_box": (
                torch.stack(box_terms).mean()
                if box_terms
                else zero
            )
            * self.weights["pose_box"],
            f"{prefix}_keypoint_presence": (
                torch.stack(keypoint_presence_terms).mean()
                if keypoint_presence_terms
                else zero
            ),
            f"{prefix}_visibility": (
                torch.stack(visibility_terms).mean()
                if visibility_terms
                else zero
            ),
        }

    def _refined_face_losses(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        logits = output["pred_face_logits"].amax(dim=-1)
        boxes = output["pred_face_boxes"]
        ellipses = output["pred_face_ellipses"]
        object_target = torch.zeros_like(logits)
        box_l1_terms = []
        giou_terms = []
        ellipse_terms = []
        angle_terms = []
        positives = 0
        for batch_index, ((source, destination), target) in enumerate(
            zip(matches, targets, strict=True)
        ):
            if source.numel() == 0:
                continue
            object_target[batch_index, source] = 1.0
            predicted_boxes = boxes[batch_index, source]
            target_boxes = target["boxes"][destination]
            if not bool(torch.isfinite(predicted_boxes).all()):
                raise FloatingPointError(
                    f"Non-finite refined Face boxes in {prefix}"
                )
            box_l1_terms.append(
                F.l1_loss(predicted_boxes, target_boxes, reduction="mean")
            )
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(predicted_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
            giou_terms.append((1.0 - giou.diag()).mean())
            if "ellipses" in target:
                predicted_ellipses = ellipses[batch_index, source]
                target_ellipses = target["ellipses"][destination]
                ellipse_terms.append(
                    F.smooth_l1_loss(
                        predicted_ellipses[..., :4],
                        target_ellipses[..., :4],
                        reduction="mean",
                    )
                )
                angle_terms.append(
                    (
                        1.0
                        - torch.cos(
                            2.0
                            * (
                                predicted_ellipses[..., 4]
                                - target_ellipses[..., 4]
                            )
                        )
                    ).mean()
                )
            positives += int(source.numel())
        zero = logits.sum() * 0.0
        normalizer = max(1, positives)
        return {
            f"{prefix}_object": _soft_focal_loss(
                logits,
                object_target,
                reduction="sum",
            )
            / normalizer,
            f"{prefix}_bbox_l1": (
                torch.stack(box_l1_terms).mean()
                if box_l1_terms
                else zero
            ),
            f"{prefix}_giou": (
                torch.stack(giou_terms).mean()
                if giou_terms
                else zero
            ),
            f"{prefix}_ellipse": (
                torch.stack(ellipse_terms).mean()
                if ellipse_terms
                else zero
            )
            * self.weights["ellipse_refined"],
            f"{prefix}_ellipse_angle": (
                torch.stack(angle_terms).mean()
                if angle_terms
                else zero
            )
            * self.weights["ellipse_refined"],
        }

    @staticmethod
    def _keypoint_uncertainty_loss(
        output: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        keypoints = output["pred_keypoints"]
        log_scales = output.get("pred_keypoint_log_scales")
        if log_scales is None:
            return keypoints.sum() * 0.0
        terms = []
        for batch_index, ((source, destination), target) in enumerate(
            zip(matches, targets, strict=True)
        ):
            if source.numel() == 0:
                continue
            labeled = target["labeled"][destination].to(keypoints.dtype)
            error = (
                keypoints[batch_index, source]
                - target["keypoints"][destination]
            ).abs()
            scale = log_scales[batch_index, source]
            laplace = error * torch.exp(-scale) + scale
            terms.append(
                (laplace.sum(dim=-1) * labeled).sum()
                / labeled.sum().clamp_min(1.0)
            )
        return torch.stack(terms).mean() if terms else keypoints.sum() * 0.0

    @staticmethod
    def _elementwise_box_iou(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = box_cxcywh_to_xyxy(first)
        second = box_cxcywh_to_xyxy(second)
        upper = torch.maximum(first[..., :2], second[..., :2])
        lower = torch.minimum(first[..., 2:], second[..., 2:])
        intersection = (lower - upper).clamp_min(0.0).prod(dim=-1)
        first_area = (first[..., 2:] - first[..., :2]).clamp_min(0.0).prod(dim=-1)
        second_area = (
            (second[..., 2:] - second[..., :2]).clamp_min(0.0).prod(dim=-1)
        )
        return intersection / (first_area + second_area - intersection).clamp_min(1.0e-8)

    def _face_evidence_losses(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        face_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        pose_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        evidence_logits = output["pred_face_pose_evidence_logits"]
        quality_logits = output["pred_face_localization_quality_logits"]
        evidence_target = torch.zeros_like(evidence_logits)
        quality_target = torch.zeros_like(quality_logits)
        non_regression_terms = []
        positive_count = 0
        for batch_index, (face_match, pose_match, target) in enumerate(
            zip(face_matches, pose_matches, targets, strict=True)
        ):
            source, destination = face_match
            if source.numel() == 0:
                continue
            pose_targets = set(pose_match[1].tolist())
            evidence_target[batch_index, source] = torch.tensor(
                [
                    float(target_index in pose_targets)
                    for target_index in destination.tolist()
                ],
                device=evidence_logits.device,
                dtype=evidence_logits.dtype,
            )
            target_boxes = target["boxes"][destination]
            before = output["pre_evidence_face_boxes"][batch_index, source]
            after = output["pred_face_boxes"][batch_index, source]
            quality_target[batch_index, source] = self._elementwise_box_iou(
                before,
                target_boxes,
            ).detach()
            before_error = (before - target_boxes).abs().mean(dim=-1)
            after_error = (after - target_boxes).abs().mean(dim=-1)
            non_regression_terms.append(
                F.relu(after_error - before_error).mean()
            )
            positive_count += int(source.numel())
        normalizer = max(1, positive_count)
        evidence_loss = _soft_focal_loss(
            evidence_logits,
            evidence_target,
            alpha=0.50,
            reduction="sum",
        ) / normalizer
        quality_loss = _soft_focal_loss(
            quality_logits,
            quality_target,
            alpha=0.50,
            reduction="sum",
        ) / normalizer
        zero = evidence_logits.sum() * 0.0
        return {
            "loss_face_pose_evidence": evidence_loss,
            "loss_face_localization_quality": quality_loss,
            "loss_face_non_regression": (
                torch.stack(non_regression_terms).mean()
                if non_regression_terms
                else zero
            ),
        }

    @staticmethod
    def _association_loss(
        logits: torch.Tensor,
        face_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        pose_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        positive_count = 0
        correct = logits.new_zeros(())
        matched_count = 0
        for batch_index, (face_match, pose_match) in enumerate(
            zip(face_matches, pose_matches, strict=True)
        ):
            face_by_target = {
                int(target_index): int(source_index)
                for source_index, target_index in zip(
                    face_match[0].tolist(),
                    face_match[1].tolist(),
                )
            }
            pose_by_target = {
                int(target_index): int(source_index)
                for source_index, target_index in zip(
                    pose_match[0].tolist(),
                    pose_match[1].tolist(),
                )
            }
            for target_index in set(face_by_target) & set(pose_by_target):
                face_index = face_by_target[target_index]
                pose_index = pose_by_target[target_index]
                target[batch_index, face_index, pose_index] = 1.0
                positive_count += 1
                if logits.shape[2] and logits.shape[1]:
                    correct = correct + (
                        logits[batch_index, face_index].argmax()
                        == pose_index
                    ).to(logits.dtype)
                    matched_count += 1
        elementwise = _soft_focal_loss(
            logits[valid],
            target[valid],
            alpha=0.50,
            gamma=2.0,
            reduction="none",
        )
        flattened_targets = target[valid] > 0.5
        positive_losses = elementwise[flattened_targets]
        negative_losses = elementwise[~flattened_targets]
        positive_term = (
            positive_losses.mean()
            if positive_losses.numel()
            else logits.sum() * 0.0
        )
        # Full F x P scoring is retained for association quality, while hard
        # negative mining prevents 10k easy pairs from overwhelming the few
        # positive instance pairs.
        negative_limit = max(
            1,
            10 * positive_count,
            logits.shape[1] + logits.shape[2],
        )
        if negative_losses.numel() > negative_limit:
            negative_losses = torch.topk(
                negative_losses,
                negative_limit,
            ).values
        negative_term = (
            negative_losses.mean()
            if negative_losses.numel()
            else logits.sum() * 0.0
        )
        loss = positive_term + negative_term
        accuracy = correct / max(1, matched_count)
        return loss, accuracy

    @staticmethod
    def _sum_branch_losses(
        losses: Mapping[str, torch.Tensor],
        *,
        prefix: str,
        suffixes: Sequence[str],
    ) -> torch.Tensor:
        selected = [
            value
            for key, value in losses.items()
            if key.startswith(prefix)
            and any(key.endswith(suffix) for suffix in suffixes)
        ]
        if not selected:
            raise RuntimeError(
                f"No losses matched prefix={prefix!r}, suffixes={suffixes}"
            )
        return sum(selected[1:], selected[0])

    def forward(
        self,
        *,
        independent_face_losses: Mapping[str, torch.Tensor],
        independent_pose: Mapping[str, Any],
        dense_pose: Mapping[str, Any] | None = None,
        refined: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        face_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        pose_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {
            f"loss_face_independent_{key.removeprefix('loss_')}": value
            for key, value in independent_face_losses.items()
            if isinstance(value, torch.Tensor) and key != "loss_total"
        }
        pose_losses = self._pose_losses(
            independent_pose,
            targets,
            pose_matches,
            prefix="loss_pose_independent",
        )
        losses.update(pose_losses)
        for layer_index, auxiliary in enumerate(
            independent_pose.get("auxiliary_outputs", [])
        ):
            auxiliary_losses = self._pose_losses(
                auxiliary,
                targets,
                pose_matches,
                prefix=f"loss_pose_independent_aux_{layer_index}",
            )
            losses.update(
                {
                    key: value * self.auxiliary_weight
                    for key, value in auxiliary_losses.items()
                }
            )

        dense_keypoint_total = independent_pose["pred_pose_logits"].sum() * 0.0
        dense_visibility = dense_keypoint_total
        dense_uncertainty = dense_keypoint_total
        if dense_pose is not None:
            dense_losses = self._pose_losses(
                dense_pose,
                targets,
                pose_matches,
                prefix="loss_pose_dense_local",
            )
            losses.update(dense_losses)
            auxiliary_uncertainty_terms = []
            for layer_index, auxiliary in enumerate(
                dense_pose.get("dense_local_auxiliary_outputs", [])
            ):
                auxiliary_losses = self._pose_losses(
                    auxiliary,
                    targets,
                    pose_matches,
                    prefix=f"loss_pose_dense_local_aux_{layer_index}",
                )
                losses.update(
                    {
                        key: value * self.auxiliary_weight
                        for key, value in auxiliary_losses.items()
                    }
                )
                auxiliary_uncertainty = (
                    self._keypoint_uncertainty_loss(
                        auxiliary,
                        targets,
                        pose_matches,
                    )
                    * self.auxiliary_weight
                )
                losses[
                    f"loss_keypoint_uncertainty_aux_{layer_index}"
                ] = auxiliary_uncertainty
                auxiliary_uncertainty_terms.append(auxiliary_uncertainty)
            dense_keypoint_total = self._sum_branch_losses(
                losses,
                prefix="loss_pose_dense_local",
                suffixes=(
                    "_object",
                    "_keypoint",
                    "_keypoint_presence",
                    "_pose_box",
                ),
            )
            dense_visibility = self._sum_branch_losses(
                losses,
                prefix="loss_pose_dense_local",
                suffixes=("_visibility",),
            )
            dense_uncertainty = self._keypoint_uncertainty_loss(
                dense_pose,
                targets,
                pose_matches,
            )
            dense_uncertainty = dense_uncertainty + sum(
                auxiliary_uncertainty_terms,
                dense_uncertainty.new_zeros(()),
            )
            losses["loss_keypoint_uncertainty"] = dense_uncertainty

        refined_face = self._refined_face_losses(
            refined,
            targets,
            face_matches,
            prefix="loss_face_refined",
        )
        refined_pose = self._pose_losses(
            refined,
            targets,
            pose_matches,
            prefix="loss_pose_refined",
        )
        losses.update(refined_face)
        losses.update(refined_pose)
        for layer_index, auxiliary in enumerate(
            refined.get("auxiliary_outputs", [])
        ):
            layer_face = self._refined_face_losses(
                auxiliary,
                targets,
                face_matches,
                prefix=f"loss_face_refined_aux_{layer_index}",
            )
            layer_pose = self._pose_losses(
                auxiliary,
                targets,
                pose_matches,
                prefix=f"loss_pose_refined_aux_{layer_index}",
            )
            losses.update(
                {
                    key: value * self.auxiliary_weight
                    for key, value in {**layer_face, **layer_pose}.items()
                }
            )

        association_loss, association_accuracy = self._association_loss(
            refined["association_logits"],
            face_matches,
            pose_matches,
        )
        losses["loss_association"] = association_loss
        losses["association_accuracy"] = association_accuracy.detach()

        evidence_total = association_loss * 0.0
        quality_total = association_loss * 0.0
        non_regression_total = association_loss * 0.0
        if "pred_face_pose_evidence_logits" in refined:
            evidence_losses = self._face_evidence_losses(
                refined,
                targets,
                face_matches,
                pose_matches,
            )
            losses.update(evidence_losses)
            evidence_total = evidence_losses["loss_face_pose_evidence"]
            quality_total = evidence_losses["loss_face_localization_quality"]
            non_regression_total = evidence_losses["loss_face_non_regression"]

        face_independent_total = independent_face_losses["loss_total"]
        independent_keypoint_total = self._sum_branch_losses(
            losses,
            prefix="loss_pose_independent",
            suffixes=(
                "_object",
                "_keypoint",
                "_keypoint_presence",
                "_pose_box",
            ),
        )
        independent_visibility = self._sum_branch_losses(
            losses,
            prefix="loss_pose_independent",
            suffixes=("_visibility",),
        )
        face_refined_total = self._sum_branch_losses(
            losses,
            prefix="loss_face_refined",
            suffixes=(
                "_object",
                "_bbox_l1",
                "_giou",
                "_ellipse",
                "_ellipse_angle",
            ),
        )
        pose_refined_total = self._sum_branch_losses(
            losses,
            prefix="loss_pose_refined",
            suffixes=(
                "_object",
                "_keypoint",
                "_keypoint_presence",
                "_pose_box",
            ),
        )
        refined_visibility = self._sum_branch_losses(
            losses,
            prefix="loss_pose_refined",
            suffixes=("_visibility",),
        )
        losses["loss_face_independent"] = face_independent_total
        losses["loss_keypoint_independent"] = independent_keypoint_total
        losses["loss_face_refined"] = face_refined_total
        losses["loss_keypoint_refined"] = pose_refined_total
        losses["loss_keypoint_local"] = dense_keypoint_total
        losses["loss_face_evidence"] = evidence_total
        losses["loss_visibility"] = (
            independent_visibility + dense_visibility + refined_visibility
        )
        losses["loss_total"] = (
            face_independent_total
            + self.weights["keypoint_independent"] * independent_keypoint_total
            + self.weights["association"] * association_loss
            + self.weights["face_refined"] * face_refined_total
            + self.weights["keypoint_refined"] * pose_refined_total
            + self.weights["keypoint_local"] * dense_keypoint_total
            + self.weights["keypoint_uncertainty"] * dense_uncertainty
            + self.weights["face_evidence"] * evidence_total
            + self.weights["localization_quality"] * quality_total
            + self.weights["non_regression"] * non_regression_total
            + self.weights["visibility"]
            * (independent_visibility + dense_visibility + refined_visibility)
        )
        return losses


class FaceKeypointCooperativeDETR(nn.Module):
    """Cooperative face/keypoint model with an optional causal cascade."""

    def __init__(
        self,
        face_detector: nn.Module,
        *,
        pose_decoder: IndependentPoseQueryDecoder,
        association: FacePoseAssociation,
        cooperative_decoder: BidirectionalCooperativeDecoder,
        criterion: CooperativeCriterion,
        head_pose_geometry: nn.Module | None = None,
        external_head_cascade: ExternalHeadCascadeRouter | None = None,
        shared_head_branch: SharedDINOHeadBranch | None = None,
        strict_sequential_cascade: StrictSequentialCascade | None = None,
        dense_local_refiner: EllipseAlignedDensePoseRefiner | None = None,
        face_evidence_fusion: PoseEvidenceFaceFusion | None = None,
        maximum_face_queries: int = 100,
        face_score_threshold: float = 0.05,
        pose_score_threshold: float = 0.05,
        association_threshold: float = 0.50,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.face_detector = face_detector
        self.pose_decoder = pose_decoder
        self.cooperative_decoder = cooperative_decoder
        if self.cooperative_decoder.association is not association:
            raise ValueError(
                "The top-level association module must be the same instance "
                "used by the cooperative decoder."
            )
        self.criterion = criterion
        # Optional by construction: when disabled, this attribute contributes
        # no parameters/state keys and the historical forward path is exact.
        self.head_pose_geometry = head_pose_geometry
        self.external_head_cascade = external_head_cascade
        self.shared_head_branch = shared_head_branch
        self.strict_sequential_cascade = strict_sequential_cascade
        self.dense_local_refiner = dense_local_refiner
        self.face_evidence_fusion = face_evidence_fusion
        if self.strict_sequential_cascade is not None:
            if self.shared_head_branch is None:
                raise ValueError(
                    "Strict sequential cascade requires the shared HEAD branch"
                )
            if self.external_head_cascade is None:
                raise ValueError(
                    "Strict sequential cascade requires HEAD-to-FACE routing"
                )
            if self.strict_sequential_cascade.enforce_unidirectional_refinement:
                self.cooperative_decoder.enforce_face_to_pose_only()
        self.maximum_face_queries = int(maximum_face_queries)
        self.face_score_threshold = float(face_score_threshold)
        self.pose_score_threshold = float(pose_score_threshold)
        self.association_threshold = float(association_threshold)
        self.detections_per_image = int(detections_per_image)
        self.current_epoch = 0
        self.pretrained_load_info: dict[str, Any] = {}
        self._face_layer_hidden: list[torch.Tensor] = []
        face_layers = self.face_detector.decoder.decoder.layers
        self._face_hooks = [
            layer.register_forward_hook(self._capture_face_hidden)
            for layer in face_layers
        ]

    @property
    def association(self) -> FacePoseAssociation:
        """Expose the decoder-owned association without duplicate state keys."""
        return self.cooperative_decoder.association

    def _capture_face_hidden(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        self._face_layer_hidden.append(output)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        if hasattr(self.face_detector, "set_epoch"):
            self.face_detector.set_epoch(epoch)

    def _normalized_targets(
        self,
        targets: Sequence[Mapping[str, Any]],
        official_targets: Sequence[Mapping[str, torch.Tensor]],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        height, width = image_size
        normalized: list[dict[str, torch.Tensor]] = []
        for raw, official in zip(targets, official_targets, strict=True):
            keypoints = torch.as_tensor(
                raw.get("keypoints"),
                device=official["boxes"].device,
                dtype=official["boxes"].dtype,
            ).reshape(-1, len(KEYPOINT_NAMES), 2)
            labeled = torch.as_tensor(
                raw.get(
                    "labeled",
                    torch.ones(keypoints.shape[:-1], dtype=torch.bool),
                ),
                device=keypoints.device,
                dtype=torch.bool,
            ).reshape(-1, len(KEYPOINT_NAMES))
            visible = torch.as_tensor(
                raw.get("visible", labeled),
                device=keypoints.device,
                dtype=torch.bool,
            ).reshape(-1, len(KEYPOINT_NAMES))
            raw_labels = torch.as_tensor(
                raw.get("labels"),
                device=keypoints.device,
                dtype=torch.long,
            ).reshape(-1)
            keep = [
                index
                for index, label in enumerate(raw_labels.tolist())
                if int(label) in self.face_detector.category_to_label
            ]
            keep_tensor = torch.tensor(
                keep,
                device=keypoints.device,
                dtype=torch.long,
            )
            keypoints = (
                keypoints[keep_tensor]
                if keep
                else keypoints.new_zeros((0, len(KEYPOINT_NAMES), 2))
            )
            labeled = (
                labeled[keep_tensor]
                if keep
                else labeled.new_zeros((0, len(KEYPOINT_NAMES)))
            )
            visible = (
                visible[keep_tensor]
                if keep
                else visible.new_zeros((0, len(KEYPOINT_NAMES)))
            )
            if keypoints.numel() and float(keypoints.detach().amax()) > 1.5:
                normalizer = keypoints.new_tensor((width, height))
                keypoints = keypoints / normalizer
            count = min(int(official["boxes"].shape[0]), int(keypoints.shape[0]))
            target = {
                "boxes": official["boxes"][:count],
                "labels": official["labels"][:count],
                "keypoints": keypoints[:count].clamp(0.0, 1.0),
                "labeled": labeled[:count],
                "visible": visible[:count],
                "instance_ids": torch.arange(
                    count,
                    device=keypoints.device,
                    dtype=torch.long,
                ),
            }
            if "ellipses" in official:
                target["ellipses"] = official["ellipses"][:count]
            else:
                boxes = target["boxes"]
                target["ellipses"] = torch.cat(
                    (
                        boxes[..., :2],
                        0.5 * boxes[..., 2:],
                        boxes.new_zeros((boxes.shape[0], 1)),
                    ),
                    dim=-1,
                )
            normalized.append(target)
        return normalized

    def _face_branch(
        self,
        encoded: Sequence[torch.Tensor],
        official_targets: Sequence[Mapping[str, torch.Tensor]] | None,
    ) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]] | None]:
        self._face_layer_hidden.clear()
        face_decoder = self.face_detector.decoder
        restore_eval_mode = official_targets is not None and not face_decoder.training
        if restore_eval_mode:
            # The official DEIM criterion requires the decoder's auxiliary,
            # encoder, and pre-decoder outputs. DEIM emits those only in
            # training mode, even when the surrounding model is evaluating a
            # validation loss. Keep the backbone, shared encoder, pose branch,
            # and cooperative branch in evaluation mode and restore the face
            # decoder immediately after producing its loss inputs.
            face_decoder.train()
        try:
            # D-FINE decodes a distribution over edge distances before
            # converting it to cxcywh boxes.  With FP16 autocast, a finite set
            # of weights can still overflow in the decoder activations after a
            # few optimizer updates and produce NaN boxes during validation.
            # Keep the outer training policy in FP16, but evaluate this
            # numerically sensitive decoder in FP32.  The casts remain in the
            # autograd graph, so gradients still reach the shared encoder.
            decoder_device = encoded[0].device.type
            with torch.autocast(device_type=decoder_device, enabled=False):
                raw = face_decoder(
                    [feature.float() for feature in encoded],
                    (
                        list(official_targets)
                        if official_targets is not None
                        else None
                    ),
                )
        finally:
            if restore_eval_mode:
                face_decoder.eval()
        if not self._face_layer_hidden:
            raise RuntimeError("DEIM face decoder hidden-state capture failed")
        if not bool(torch.isfinite(raw["pred_boxes"]).all()):
            raise FloatingPointError(
                "Non-finite independent Face boxes after FP32 DEIM decoding"
            )
        number_of_queries = int(raw["pred_boxes"].shape[1])
        final_embeddings = self._face_layer_hidden[-1][:, -number_of_queries:]
        matches = None
        if official_targets is not None:
            matches = self.face_detector.criterion.matcher(
                {
                    "pred_logits": raw["pred_logits"],
                    "pred_boxes": raw["pred_boxes"],
                },
                list(official_targets),
                epoch=self.current_epoch,
            )["indices"]
        candidates = _candidate_indices(
            raw["pred_logits"],
            maximum=self.maximum_face_queries,
            matched=matches,
        )
        selected = {
            "pred_logits": _gather(raw["pred_logits"], candidates),
            "pred_boxes": _gather(raw["pred_boxes"], candidates),
            "embeddings": _gather(final_embeddings, candidates),
            "candidate_indices": candidates,
            "raw_output": raw,
            "decoder_embeddings": [
                _gather(hidden[:, -number_of_queries:], candidates)
                for hidden in self._face_layer_hidden
            ],
        }
        if "pred_ellipses" in raw:
            selected["pred_ellipses"] = _gather(
                raw["pred_ellipses"],
                candidates,
            )
        else:
            boxes = selected["pred_boxes"]
            selected["pred_ellipses"] = torch.cat(
                (
                    boxes[..., :2],
                    0.5 * boxes[..., 2:],
                    boxes.new_zeros((*boxes.shape[:-1], 1)),
                ),
                dim=-1,
            )
        return selected, matches

    @staticmethod
    def _remap_face_matches(
        candidates: torch.Tensor,
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return CooperativeCriterion._matched_positions(candidates, matches)

    @staticmethod
    def _shared_head_target_indices(
        predictions: Sequence[Mapping[str, torch.Tensor]],
        targets: Sequence[Mapping[str, Any]],
        *,
        minimum_iou: float = 0.10,
    ) -> list[torch.Tensor]:
        """Map internal HEAD predictions to FACE target indices.

        HEAD supervision includes annotations without valid FACE attributes.
        ``head_detection_to_face_indices`` therefore carries ``-1`` for those
        rows; the router/geometry loss only treats matched valid FACE rows as
        positives.
        """

        mappings: list[torch.Tensor] = []
        for prediction, target in zip(predictions, targets, strict=True):
            boxes = torch.as_tensor(prediction["boxes"]).detach()
            target_boxes = torch.as_tensor(
                target.get("head_detection_boxes", target.get("head_boxes", [])),
                device=boxes.device,
                dtype=boxes.dtype,
            ).reshape(-1, 4)
            head_to_face = torch.as_tensor(
                target.get(
                    "head_detection_to_face_indices",
                    torch.arange(target_boxes.shape[0]),
                ),
                device=boxes.device,
                dtype=torch.long,
            ).reshape(-1)
            if target_boxes.shape[0] != head_to_face.shape[0]:
                raise ValueError(
                    "head_detection_to_face_indices must align with "
                    "head_detection_boxes"
                )
            mapping = torch.full(
                (boxes.shape[0],),
                -1,
                device=boxes.device,
                dtype=torch.long,
            )
            if boxes.numel() and target_boxes.numel():
                upper = torch.maximum(boxes[:, None, :2], target_boxes[None, :, :2])
                lower = torch.minimum(boxes[:, None, 2:], target_boxes[None, :, 2:])
                intersection = (lower - upper).clamp_min(0.0).prod(dim=-1)
                box_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0).prod(dim=-1)
                target_area = (
                    (target_boxes[:, 2:] - target_boxes[:, :2])
                    .clamp_min(0.0)
                    .prod(dim=-1)
                )
                iou = intersection / (
                    box_area[:, None] + target_area[None, :] - intersection
                ).clamp_min(1.0e-12)
                best_iou, best_target = iou.max(dim=1)
                valid = best_iou >= float(minimum_iou)
                mapping[valid] = head_to_face[best_target[valid]]
            mappings.append(mapping)
        return mappings

    def forward(
        self,
        images: torch.Tensor,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        return_raw: bool = False,
        return_attention: bool = False,
        branch_override: tuple[bool, bool] | None = None,
        head_detections: Sequence[Sequence[Any]] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        image_size = (int(images.shape[-2]), int(images.shape[-1]))
        official_targets = None
        joint_targets = None
        if targets is not None:
            official_targets = self.face_detector._targets_to_official(
                list(targets),
                image_size,
                device=images.device,
                dtype=images.dtype,
            )
            joint_targets = self._normalized_targets(
                targets,
                official_targets,
                image_size,
            )
        shared_head_output = None
        if self.shared_head_branch is not None:
            shared_forward = getattr(
                self.face_detector.backbone,
                "forward_with_semantic_and_final_stride16",
                None,
            )
            if shared_forward is None:
                raise TypeError(
                    "model.shared_head_branch requires a face backbone with "
                    "forward_with_semantic_and_final_stride16(); no second "
                    "DINO pass is allowed"
                )
            features, semantic_features, final_stride16 = shared_forward(images)
            if len(semantic_features) != 3:
                raise ValueError(
                    "The shared DINO backbone must return three semantic levels"
                )
            if final_stride16.ndim != 4:
                raise ValueError(
                    "The shared DINO backbone block-11 feature must have shape "
                    f"[B,C,H/16,W/16], got {tuple(final_stride16.shape)}"
                )
            # MMCV's CUDA multi-scale deformable-attention extension supports
            # the production FP16 path used by this shared branch.
            with torch.autocast(
                device_type=final_stride16.device.type,
                dtype=torch.float16,
                enabled=final_stride16.is_cuda,
            ):
                shared_head_output = self.shared_head_branch(
                    final_stride16,
                    image_size=image_size,
                    targets=targets,
                    padding_mask=padding_mask,
                )
            # The internal branch is authoritative.  It is derived from the
            # exact same full-frame DINO invocation as the FACE branch.
            head_detections = shared_head_output["detections"]
        else:
            features = self.face_detector.backbone(images)
        shared_head_target_indices = None
        geometry_targets = targets
        if shared_head_output is not None and targets is not None:
            shared_head_target_indices = self._shared_head_target_indices(
                shared_head_output["predictions"],
                targets,
            )
            geometry_targets = [
                {
                    **dict(target),
                    "geometry_target_indices": mapping,
                }
                for target, mapping in zip(
                    targets,
                    shared_head_target_indices,
                    strict=True,
                )
            ]
        encoded = self.face_detector.encoder(list(features))
        needs_external_heads = (
            self.external_head_cascade is not None
            or self.head_pose_geometry is not None
            or self.strict_sequential_cascade is not None
        )
        if needs_external_heads and head_detections is None:
            if targets is None:
                raise ValueError(
                    "The three-stage cascade requires predicted "
                    "head_detections at inference; there is no GT fallback."
                )
            head_detections = detections_from_targets(
                targets,
                image_size=image_size,
            )
        head_threshold_diagnostics = None
        if self.strict_sequential_cascade is not None:
            assert head_detections is not None
            head_detections, head_threshold_diagnostics = (
                self.strict_sequential_cascade.filter_head_detections(
                    head_detections
                )
            )
        sequential_diagnostics: dict[str, Any] | None = None
        face_encoded = encoded
        sequential_training_rois = self.training and targets is not None
        if self.strict_sequential_cascade is not None:
            assert head_detections is not None
            face_encoded, head_stage = (
                self.strict_sequential_cascade.head_conditioned_features(
                    encoded,
                    head_detections,
                    image_size=image_size,
                    training=sequential_training_rois,
                )
            )
            sequential_diagnostics = {
                "execution_order": ("head", "face", "keypoint"),
                "head_threshold": head_threshold_diagnostics,
                "head_to_face": head_stage,
                "unidirectional_refinement": True,
            }
        independent_face, raw_face_matches = self._face_branch(
            face_encoded,
            official_targets,
        )
        cascade_routing = None
        if self.external_head_cascade is not None:
            assert head_detections is not None
            cascade_routing = self.external_head_cascade(
                face_boxes=independent_face["pred_boxes"],
                face_logits=independent_face["pred_logits"],
                head_detections=head_detections,
                image_size=image_size,
            )
        pose_encoded = encoded
        if self.strict_sequential_cascade is not None:
            assert sequential_diagnostics is not None
            if sequential_training_rois:
                assert joint_targets is not None
                face_rois = self.strict_sequential_cascade.target_face_boxes(
                    joint_targets
                )
            else:
                if cascade_routing is None:
                    raise RuntimeError(
                        "Strict sequential inference requires HEAD-routed FACE predictions"
                    )
                assert self.external_head_cascade is not None
                face_rois = self.strict_sequential_cascade.predicted_face_boxes(
                    independent_face,
                    cascade_routing,
                    routing_threshold=(
                        self.external_head_cascade.face_to_head_threshold
                    ),
                )
            pose_encoded, face_stage = (
                self.strict_sequential_cascade.face_conditioned_features(
                    encoded,
                    face_rois,
                    training=sequential_training_rois,
                )
            )
            sequential_diagnostics["face_to_keypoint"] = face_stage
        independent_pose = self.pose_decoder(
            pose_encoded,
            padding_mask=padding_mask,
            return_attention=return_attention,
        )
        geometry_outputs = None
        if self.head_pose_geometry is not None:
            assert head_detections is not None
            geometry_outputs = self.head_pose_geometry(
                images,
                head_detections,
                training_crop=self.training and targets is not None,
            )
            independent_pose = self.head_pose_geometry.inject_pose_queries(
                independent_pose,
                geometry_outputs,
            )
        dense_pose = None
        pose_for_cooperation = independent_pose
        if self.dense_local_refiner is not None:
            dense_device = independent_pose["memory"].device.type
            with torch.autocast(device_type=dense_device, enabled=False):
                initial_association = self.association(
                    independent_face["embeddings"].float(),
                    independent_pose["pose_embeddings"].float(),
                    face_boxes=independent_face["pred_boxes"].float(),
                    face_ellipses=independent_face["pred_ellipses"].float(),
                    face_logits=independent_face["pred_logits"].float(),
                    pose_boxes=independent_pose["pred_pose_boxes"].float(),
                    keypoints=independent_pose["pred_keypoints"].float(),
                    pose_logits=independent_pose["pred_pose_logits"].float(),
                    keypoint_logits=independent_pose[
                        "pred_keypoint_logits"
                    ].float(),
                    visibility_logits=independent_pose[
                        "pred_visibility_logits"
                    ].float(),
                )
                dense_pose = self.dense_local_refiner(
                    images.float(),
                    _float_top_level_tensors(independent_face),
                    _float_top_level_tensors(independent_pose),
                    initial_association,
                )
            pose_for_cooperation = dense_pose
        # FP16 remains the outer AMP policy, but the bidirectional refinement
        # repeatedly applies inverse-sigmoid box updates and deformable
        # attention. Running this compact block in FP32 prevents a single
        # half-precision attention overflow from turning all later boxes into
        # NaNs. Gradients still propagate through the input casts.
        refinement_device = independent_pose["memory"].device.type
        with torch.autocast(
            device_type=refinement_device,
            enabled=False,
        ):
            if self.strict_sequential_cascade is not None:
                if branch_override not in (None, (True, False)):
                    raise ValueError(
                        "Strict sequential cascade forbids keypoint-to-FACE refinement"
                    )
                branch_override = (True, False)
            refined = self.cooperative_decoder(
                _float_top_level_tensors(independent_face),
                _float_top_level_tensors(pose_for_cooperation),
                memory=pose_for_cooperation["memory"].float(),
                memory_mask=pose_for_cooperation["memory_mask"],
                spatial_shapes=pose_for_cooperation["spatial_shapes"],
                branch_override=branch_override,
            )
            if self.face_evidence_fusion is not None:
                refined = self.face_evidence_fusion(refined)
        if self.pose_decoder.state_head_mode == FACTORIZED_STATE_HEAD:
            refined["pred_state_probabilities"] = (
                factorized_state_probabilities(
                    refined["pred_keypoint_logits"],
                    refined["pred_visibility_logits"],
                )
            )
        output: dict[str, Any] = {
            "independent_face": independent_face,
            "independent_pose": independent_pose,
            "refined": refined,
        }
        if dense_pose is not None:
            output["dense_local_pose"] = dense_pose
        if shared_head_output is not None:
            output["shared_head"] = shared_head_output
        if geometry_outputs is not None:
            output["head_pose_geometry"] = geometry_outputs
        if cascade_routing is not None:
            output["external_head_cascade"] = cascade_routing
        if sequential_diagnostics is not None:
            output["strict_sequential_cascade"] = sequential_diagnostics
        if targets is not None:
            assert official_targets is not None
            assert joint_targets is not None
            assert raw_face_matches is not None
            face_matches = self._remap_face_matches(
                independent_face["candidate_indices"],
                raw_face_matches,
            )
            pose_matches = self.criterion.pose_matcher(
                independent_pose,
                joint_targets,
            )
            if geometry_outputs is not None:
                geometry_target_indices = [
                    torch.as_tensor(
                        target.get(
                            "geometry_target_indices",
                            torch.arange(
                                geometry_output["rotation_matrix"].shape[0],
                                device=images.device,
                            ),
                        ),
                        device=images.device,
                        dtype=torch.long,
                    )
                    for target, geometry_output in zip(
                        geometry_targets,
                        geometry_outputs,
                        strict=True,
                    )
                ]
                refined = self.head_pose_geometry.fuse_training_queries(
                    refined,
                    geometry_outputs,
                    face_matches=face_matches,
                    pose_matches=pose_matches,
                    geometry_target_indices=geometry_target_indices,
                )
                output["refined"] = refined
            face_loss_values = self.face_detector.criterion(
                independent_face["raw_output"],
                list(official_targets),
                epoch=self.current_epoch,
            )
            independent_face_losses = self.face_detector._format_losses(
                face_loss_values
            )
            losses = self.criterion(
                independent_face_losses=independent_face_losses,
                independent_pose=independent_pose,
                dense_pose=dense_pose,
                refined=refined,
                targets=joint_targets,
                face_matches=face_matches,
                pose_matches=pose_matches,
            )
            if shared_head_output is not None:
                shared_head_losses = {
                    f"loss_shared_head_{name.removeprefix('loss_')}": value.sum()
                    for name, value in shared_head_output["losses"].items()
                    if name != "loss_total"
                }
                weighted_head_total = (
                    shared_head_output["losses"]["loss_total"].sum()
                    * self.shared_head_branch.loss_weight
                )
                losses.update(shared_head_losses)
                losses["loss_shared_head_total"] = weighted_head_total
                losses["loss_total"] = losses["loss_total"] + weighted_head_total
            if cascade_routing is not None:
                assert self.external_head_cascade is not None
                routing_target_indices = (
                    shared_head_target_indices
                    if shared_head_target_indices is not None
                    else [
                        torch.as_tensor(
                            target.get(
                                "geometry_target_indices",
                                torch.arange(
                                    int(cascade_routing["head_valid"][batch_index].sum()),
                                    device=images.device,
                                ),
                            ),
                            device=images.device,
                            dtype=torch.long,
                        )
                        for batch_index, target in enumerate(targets)
                    ]
                )
                routing_losses = self.external_head_cascade.loss(
                    cascade_routing,
                    face_matches=face_matches,
                    geometry_target_indices=routing_target_indices,
                )
                losses.update(routing_losses)
                losses["loss_total"] = (
                    losses["loss_total"]
                    + routing_losses["loss_external_head_routing"]
                )
            if geometry_outputs is not None:
                geometry_losses = self.head_pose_geometry.losses(
                    geometry_outputs,
                    geometry_targets,
                    image_size=image_size,
                )
                losses.update(geometry_losses)
                losses["loss_total"] = (
                    losses["loss_total"]
                    + geometry_losses["loss_head_pose_geometry"]
                )
            output["matches"] = {
                "face": face_matches,
                "pose": pose_matches,
            }
            output.update(losses)
            return output
        if geometry_outputs is not None:
            refined = self.head_pose_geometry.fuse_inference_queries(
                refined,
                geometry_outputs,
            )
            output["refined"] = refined
        if return_raw:
            return output
        return self.decode(output, image_size=image_size)

    def _decode_face(
        self,
        boxes: torch.Tensor,
        ellipses: torch.Tensor,
        logits: torch.Tensor,
        *,
        image_size: tuple[int, int],
        threshold: float,
    ) -> list[dict[str, torch.Tensor]]:
        height, width = image_size
        scores = logits.sigmoid().amax(dim=-1)
        results = []
        for batch_index in range(boxes.shape[0]):
            keep = torch.nonzero(
                scores[batch_index] >= threshold,
                as_tuple=False,
            ).squeeze(-1)
            if self.detections_per_image > 0 and keep.numel() > self.detections_per_image:
                _, order = torch.topk(
                    scores[batch_index, keep],
                    self.detections_per_image,
                )
                keep = keep[order]
            selected_boxes = boxes[batch_index, keep]
            xyxy = box_cxcywh_to_xyxy(selected_boxes)
            xyxy = xyxy * xyxy.new_tensor((width, height, width, height))
            selected_ellipses = ellipses[batch_index, keep].clone()
            selected_ellipses[..., 0] *= width
            selected_ellipses[..., 1] *= height
            selected_ellipses[..., 2:4] *= max(width, height)
            results.append(
                {
                    "boxes": xyxy,
                    "ellipses": selected_ellipses,
                    "scores": scores[batch_index, keep],
                    "labels": torch.zeros_like(keep),
                    "query_indices": keep,
                }
            )
        return results

    def _decode_pose(
        self,
        output: Mapping[str, torch.Tensor],
        *,
        image_size: tuple[int, int],
        threshold: float,
    ) -> list[dict[str, torch.Tensor]]:
        height, width = image_size
        scores = output["pred_pose_logits"].sigmoid()
        boxes = output["pred_pose_boxes"]
        keypoints = output["pred_keypoints"]
        results = []
        for batch_index in range(boxes.shape[0]):
            keep = torch.nonzero(
                scores[batch_index] >= threshold,
                as_tuple=False,
            ).squeeze(-1)
            if self.detections_per_image > 0 and keep.numel() > self.detections_per_image:
                _, order = torch.topk(
                    scores[batch_index, keep],
                    self.detections_per_image,
                )
                keep = keep[order]
            selected_boxes = box_cxcywh_to_xyxy(boxes[batch_index, keep])
            selected_boxes = selected_boxes * selected_boxes.new_tensor(
                (width, height, width, height)
            )
            selected_keypoints = (
                keypoints[batch_index, keep]
                * keypoints.new_tensor((width, height))
            )
            results.append(
                {
                    "boxes": selected_boxes,
                    "scores": scores[batch_index, keep],
                    "keypoints": selected_keypoints,
                    "keypoint_scores": output["pred_keypoint_logits"][
                        batch_index,
                        keep,
                    ].sigmoid(),
                    "visibility": output["pred_visibility_logits"][
                        batch_index,
                        keep,
                    ].sigmoid(),
                    "query_indices": keep,
                }
            )
            if self.pose_decoder.state_head_mode == FACTORIZED_STATE_HEAD:
                results[-1]["state_probabilities"] = (
                    factorized_state_probabilities(
                        output["pred_keypoint_logits"][batch_index, keep],
                        output["pred_visibility_logits"][batch_index, keep],
                    )
                )
        return results

    def decode(
        self,
        output: Mapping[str, Any],
        *,
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        independent_face = output["independent_face"]
        independent_pose = output["independent_pose"]
        refined = output["refined"]
        face_before = self._decode_face(
            independent_face["pred_boxes"],
            independent_face["pred_ellipses"],
            independent_face["pred_logits"],
            image_size=image_size,
            threshold=self.face_score_threshold,
        )
        face_after = self._decode_face(
            refined["pred_face_boxes"],
            refined["pred_face_ellipses"],
            refined["pred_face_logits"],
            image_size=image_size,
            threshold=self.face_score_threshold,
        )
        pose_before = self._decode_pose(
            independent_pose,
            image_size=image_size,
            threshold=self.pose_score_threshold,
        )
        dense_pose = (
            self._decode_pose(
                output["dense_local_pose"],
                image_size=image_size,
                threshold=self.pose_score_threshold,
            )
            if "dense_local_pose" in output
            else None
        )
        pose_after = self._decode_pose(
            refined,
            image_size=image_size,
            threshold=self.pose_score_threshold,
        )
        decoded = []
        association_probability = refined["association_logits"].sigmoid()
        for batch_index in range(association_probability.shape[0]):
            cooperative_face = face_after[batch_index]
            cooperative_pose = pose_after[batch_index]
            cascade_diagnostics: dict[str, torch.Tensor] | None = None
            if self.external_head_cascade is not None:
                routing = output.get("external_head_cascade")
                if routing is None:
                    raise RuntimeError(
                        "External HEAD cascade output is missing during decode"
                    )
                face_indices = cooperative_face["query_indices"]
                route_probability = routing["best_head_probability"][
                    batch_index,
                    face_indices,
                ]
                face_mask = (
                    route_probability
                    >= self.external_head_cascade.face_to_head_threshold
                )
                routed_face = {
                    key: (
                        value[face_mask]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == face_indices.shape[0]
                        else value
                    )
                    for key, value in cooperative_face.items()
                }
                routed_face_indices = routed_face["query_indices"]
                pose_indices = cooperative_pose["query_indices"]
                if routed_face_indices.numel() and pose_indices.numel():
                    routed_association = association_probability[batch_index][
                        routed_face_indices
                    ][:, pose_indices]
                    pose_mask = (
                        routed_association.amax(dim=0)
                        >= self.association_threshold
                    )
                else:
                    pose_mask = torch.zeros(
                        pose_indices.shape,
                        device=pose_indices.device,
                        dtype=torch.bool,
                    )
                routed_pose = {
                    key: (
                        value[pose_mask]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == pose_indices.shape[0]
                        else value
                    )
                    for key, value in cooperative_pose.items()
                }
                face_after[batch_index] = routed_face
                pose_after[batch_index] = routed_pose
                cascade_diagnostics = {
                    "face_to_head_probability": route_probability[face_mask],
                    "face_to_head_index": routing["best_head_index"][
                        batch_index,
                        face_indices[face_mask],
                    ],
                    "head_boxes": routing["head_boxes_xyxy"][
                        batch_index,
                        routing["head_valid"][batch_index],
                    ]
                    * routing["head_boxes_xyxy"].new_tensor(
                        (
                            image_size[1],
                            image_size[0],
                            image_size[1],
                            image_size[0],
                        )
                    ),
                    "head_confidence": routing["head_confidence"][
                        batch_index,
                        routing["head_valid"][batch_index],
                    ],
                }
            face_keep = face_after[batch_index]["query_indices"]
            pose_keep = pose_after[batch_index]["query_indices"]
            if face_keep.numel() and pose_keep.numel():
                matrix = association_probability[batch_index][face_keep][:, pose_keep]
                pairs = torch.nonzero(
                    matrix >= self.association_threshold,
                    as_tuple=False,
                )
                association_rows = torch.cat(
                    (
                        pairs.to(matrix.dtype),
                        matrix[pairs[:, 0], pairs[:, 1]].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            else:
                association_rows = association_probability.new_zeros((0, 3))
            row: dict[str, Any] = {
                    "independent_face": face_before[batch_index],
                    "independent_pose": pose_before[batch_index],
                    "refined_face": face_after[batch_index],
                    "refined_pose": pose_after[batch_index],
                    "associations": association_rows,
                    "branch_face_to_pose_enabled": refined[
                        "branch_face_to_pose_enabled"
                    ][batch_index],
                    "branch_pose_to_face_enabled": refined[
                        "branch_pose_to_face_enabled"
                    ][batch_index],
                }
            if dense_pose is not None:
                row["dense_local_pose"] = dense_pose[batch_index]
            if "pred_face_pose_evidence_logits" in refined:
                evidence_indices = face_after[batch_index]["query_indices"]
                row["face_evidence"] = {
                    "pose_evidence": refined[
                        "pred_face_pose_evidence_logits"
                    ][batch_index, evidence_indices].sigmoid(),
                    "localization_quality": refined[
                        "pred_face_localization_quality_logits"
                    ][batch_index, evidence_indices].sigmoid(),
                    "fusion_gate": refined["face_evidence_gate"][
                        batch_index,
                        evidence_indices,
                    ],
                    "box_update_gate": refined["face_box_update_gate"][
                        batch_index,
                        evidence_indices,
                    ],
                }
            if cascade_diagnostics is not None:
                row["cooperative_refined_face"] = cooperative_face
                row["cooperative_refined_pose"] = cooperative_pose
                row["external_head_cascade"] = cascade_diagnostics
            shared_head = output.get("shared_head")
            if shared_head is not None:
                prediction = shared_head["detector_predictions"][batch_index]
                row["shared_head"] = {
                    "boxes": prediction["boxes"],
                    "scores": prediction["scores"],
                    "labels": prediction["labels"],
                }
            geometry_rows = output.get("head_pose_geometry")
            if geometry_rows is not None:
                geometry = geometry_rows[batch_index]
                head_geometry: dict[str, Any] = {
                    "head_boxes": geometry["crop"].head_boxes_xyxy,
                    "head_confidence": geometry["crop"].head_confidence,
                }
                for key in (
                    "rotation_matrix",
                    "euler_xyz",
                    "pose_uncertainty",
                    "pose_confidence",
                    "visible_face_confidence",
                    "amodal_face_confidence",
                    "face_observability",
                    "geometry_uncertainty",
                    "geometry_confidence",
                    "amodal_ellipse",
                    "visible_ellipse",
                    "projected_keypoints",
                ):
                    if key in geometry:
                        value = geometry[key]
                        if key in {"amodal_ellipse", "visible_ellipse"}:
                            value = value.clone()
                            value[..., 0] *= image_size[1]
                            value[..., 1] *= image_size[0]
                            value[..., 2:4] *= max(image_size)
                        elif key == "projected_keypoints":
                            value = value * value.new_tensor(
                                (image_size[1], image_size[0])
                            )
                        head_geometry[key] = value
                row["head_pose_geometry"] = head_geometry
            decoded.append(row)
        return decoded

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {
            "backbone": [],
            "sta_neck": [],
            "shared_encoder": list(self.face_detector.encoder.parameters()),
            "face_decoder": list(self.face_detector.decoder.parameters()),
            "keypoint_decoder": list(self.pose_decoder.parameters()),
            "association": list(self.association.parameters()),
            "cooperative_decoder": [],
            "dense_local_refiner": (
                list(self.dense_local_refiner.parameters())
                if self.dense_local_refiner is not None
                else []
            ),
            "face_evidence_fusion": (
                list(self.face_evidence_fusion.parameters())
                if self.face_evidence_fusion is not None
                else []
            ),
            "head_pose_geometry": (
                list(self.head_pose_geometry.parameters())
                if self.head_pose_geometry is not None
                else []
            ),
            "external_head_cascade": (
                list(self.external_head_cascade.parameters())
                if self.external_head_cascade is not None
                else []
            ),
            "shared_head_branch": (
                list(self.shared_head_branch.parameters())
                if self.shared_head_branch is not None
                else []
            ),
        }
        backbone = self.face_detector.backbone
        for name, parameter in backbone.named_parameters():
            if "dino" in name or "dinov3" in name:
                groups["backbone"].append(parameter)
            else:
                groups["sta_neck"].append(parameter)
        association_ids = {id(parameter) for parameter in self.association.parameters()}
        groups["cooperative_decoder"] = [
            parameter
            for parameter in self.cooperative_decoder.parameters()
            if id(parameter) not in association_ids
        ]
        return groups


__all__ = [
    "BidirectionalCooperativeDecoder",
    "CooperativeCriterion",
    "EllipseAlignedDensePoseRefiner",
    "FaceKeypointCooperativeDETR",
    "FacePoseAssociation",
    "IndependentPoseQueryDecoder",
    "KEYPOINT_NAMES",
    "PoseHungarianMatcher",
    "PoseEvidenceFaceFusion",
]
