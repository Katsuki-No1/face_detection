from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from keypoint_comparison1.landmark_detection.heads import (
    CoordinateRefinementStage,
    FACTORIZED_STATE_HEAD,
    LEGACY_BINARY_STATE_HEAD,
    STATE_HEAD_MODES,
    factorized_state_probabilities,
    sine_position_encoding_2d,
)
from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import load_checkpoint, load_config


def _gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    for _ in range(tensor.ndim - 2):
        view = view.unsqueeze(-1)
    return torch.gather(tensor, 1, view.expand(-1, -1, *tensor.shape[2:]))


def _candidate_indices(
    logits: torch.Tensor,
    *,
    maximum: int,
    matched: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    scores = logits.sigmoid().amax(-1)
    count = min(int(maximum), scores.shape[1])
    top = torch.topk(scores, count, dim=1).indices
    if matched is None:
        return top
    rows: list[torch.Tensor] = []
    for batch_index, (prediction_indices, _target_indices) in enumerate(matched):
        required = prediction_indices.to(device=top.device)
        merged = torch.unique(torch.cat((required, top[batch_index])), sorted=False)
        if merged.numel() < count:
            merged = torch.cat((merged, top[batch_index, : count - merged.numel()]))
        rows.append(merged[:count])
    return torch.stack(rows)


def _block_diagonal_landmark_mask(faces: int, landmarks: int, device: torch.device) -> torch.Tensor:
    groups = torch.arange(faces, device=device).repeat_interleave(landmarks)
    return groups[:, None] != groups[None, :]


class LandmarkQueryDecoder(nn.Module):
    """Generate five semantic queries from every DEIMv2 face object query."""

    def __init__(
        self,
        dimension: int,
        *,
        layers: int = 4,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
    ) -> None:
        super().__init__()
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"state_head_mode must be one of {STATE_HEAD_MODES}, got "
                f"{state_head_mode!r}"
            )
        self.state_head_mode = state_head_mode
        self.landmark_queries = nn.Parameter(torch.empty(5, dimension))
        nn.init.trunc_normal_(self.landmark_queries, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=dimension,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dimension)
        self.coordinate_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 2),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, 1),
        )
        self.presence_head = (
            nn.Sequential(
                nn.Linear(dimension, dimension // 2),
                nn.GELU(),
                nn.Linear(dimension // 2, 1),
            )
            if state_head_mode == FACTORIZED_STATE_HEAD
            else None
        )

    @staticmethod
    def _inside_boxes(local: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        center = boxes[..., :2].unsqueeze(2)
        size = boxes[..., 2:].clamp_min(1e-5).unsqueeze(2)
        top_left = center - 0.5 * size
        return (top_left + local * size).clamp(0.0, 1.0)

    @staticmethod
    def _inside_ellipses(
        local: torch.Tensor,
        ellipses: torch.Tensor,
    ) -> torch.Tensor:
        """Map local landmark coordinates through the predicted Face ellipse."""

        centered = local.mul(2.0).sub(1.0)
        major = ellipses[..., 2].clamp_min(1.0e-5).unsqueeze(-1)
        minor = ellipses[..., 3].clamp_min(1.0e-5).unsqueeze(-1)
        local_x = centered[..., 0] * major
        local_y = centered[..., 1] * minor
        theta = ellipses[..., 4].unsqueeze(-1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = cosine * local_x - sine * local_y
        y = sine * local_x + cosine * local_y
        return (
            ellipses[..., None, :2] + torch.stack((x, y), dim=-1)
        ).clamp(0.0, 1.0)

    def forward(
        self,
        face_queries: torch.Tensor,
        face_boxes: torch.Tensor,
        memory: torch.Tensor,
        *,
        face_ellipses: torch.Tensor | None = None,
        return_query_features: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        list[torch.Tensor],
    ] | tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
    ]:
        batch, faces, dimension = face_queries.shape
        query = face_queries.unsqueeze(2) + self.landmark_queries.view(1, 1, 5, dimension)
        query = query.reshape(batch, faces * 5, dimension)
        mask = _block_diagonal_landmark_mask(faces, 5, query.device)
        auxiliary: list[torch.Tensor] = []
        presence = None
        visibility = query.new_zeros((batch, faces, 5))
        for layer in self.layers:
            query = layer(query, memory, tgt_mask=mask)
            normalized = self.norm(query).reshape(batch, faces, 5, dimension)
            local = self.coordinate_head(normalized).sigmoid()
            auxiliary.append(
                self._inside_ellipses(local, face_ellipses)
                if face_ellipses is not None
                else self._inside_boxes(local, face_boxes)
            )
            if self.presence_head is not None:
                presence = self.presence_head(normalized).squeeze(-1)
            visibility = self.visibility_head(normalized).squeeze(-1)
        result = (auxiliary[-1], presence, visibility, auxiliary[:-1])
        if return_query_features:
            return (*result, normalized)
        return result


class EllipseConditionedLocalRefinement(nn.Module):
    """Fuse shared stride-4/8 Face ROIs and refine global landmark queries."""

    def __init__(
        self,
        dimension: int,
        *,
        high_resolution_channels: int,
        roi_size: int = 16,
        roi_scale: float = 1.25,
        layers: int = 2,
        heads: int = 8,
        feedforward: int | None = None,
        dropout: float = 0.0,
        maximum_base_delta: float = 0.10,
        minimum_radius: float = 1.0e-3,
        detach_ellipse: bool = True,
        stages: Sequence[Mapping[str, float]] | None = None,
        state_head_mode: str = LEGACY_BINARY_STATE_HEAD,
    ) -> None:
        super().__init__()
        if high_resolution_channels < 1:
            raise ValueError("high_resolution_channels must be positive")
        if roi_size < 2:
            raise ValueError("roi_size must be at least two")
        if roi_scale <= 0.0:
            raise ValueError("roi_scale must be positive")
        if layers < 1:
            raise ValueError("local query decoder layers must be positive")
        if dimension % heads != 0:
            raise ValueError("local query dimension must be divisible by heads")
        if dimension % 4 != 0:
            raise ValueError(
                "local query dimension must support 2D sine position encoding"
            )
        if state_head_mode not in STATE_HEAD_MODES:
            raise ValueError(
                f"state_head_mode must be one of {STATE_HEAD_MODES}, got "
                f"{state_head_mode!r}"
            )
        configured_stages = list(
            stages
            if stages is not None
            else (
                {
                    "radius_in_feature_cells": 2.0,
                    "maximum_delta": 0.08,
                    "samples_per_axis": 3,
                },
                {
                    "radius_in_feature_cells": 1.0,
                    "maximum_delta": 0.04,
                    "samples_per_axis": 3,
                },
            )
        )
        if len(configured_stages) != 2:
            raise ValueError("ellipse local refinement requires exactly two stages")

        self.dimension = int(dimension)
        self.roi_size = int(roi_size)
        self.roi_scale = float(roi_scale)
        self.maximum_base_delta = float(maximum_base_delta)
        self.minimum_radius = float(minimum_radius)
        self.detach_ellipse = bool(detach_ellipse)
        self.state_head_mode = str(state_head_mode)
        self.deim_projection = nn.Conv2d(
            dimension, dimension, kernel_size=1, bias=False
        )
        self.high_resolution_projection = nn.Conv2d(
            high_resolution_channels, dimension, kernel_size=1, bias=False
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                2 * dimension,
                dimension,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, dimension),
            nn.GELU(),
        )
        self.face_projection = nn.Linear(dimension, dimension, bias=False)
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.query_norm = nn.LayerNorm(dimension)
        hidden = int(feedforward or dimension * 4)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=dimension,
                nhead=heads,
                dim_feedforward=hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.base_delta_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 2),
        )
        output_layer = self.base_delta_head[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.refinement_stages = nn.ModuleList(
            [
                CoordinateRefinementStage(
                    dimension,
                    radius_in_feature_cells=float(
                        stage["radius_in_feature_cells"]
                    ),
                    maximum_delta=float(stage["maximum_delta"]),
                    samples_per_axis=int(stage.get("samples_per_axis", 3)),
                    state_head_mode=state_head_mode,
                )
                for stage in configured_stages
            ]
        )

    def _ellipse_grid(self, ellipses: torch.Tensor) -> torch.Tensor:
        axis = (
            (torch.arange(
                self.roi_size,
                device=ellipses.device,
                dtype=ellipses.dtype,
            ) + 0.5)
            / float(self.roi_size)
        ).mul(2.0).sub(1.0)
        local_y, local_x = torch.meshgrid(axis, axis, indexing="ij")
        local_x = local_x.view(1, 1, self.roi_size, self.roi_size)
        local_y = local_y.view(1, 1, self.roi_size, self.roi_size)
        center = ellipses[..., :2]
        half_side = (
            ellipses[..., 2].clamp_min(self.minimum_radius) * self.roi_scale
        )
        theta = ellipses[..., 4]
        cosine = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        sine = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        scaled_x = local_x * half_side.unsqueeze(-1).unsqueeze(-1)
        scaled_y = local_y * half_side.unsqueeze(-1).unsqueeze(-1)
        x = (
            center[..., 0].unsqueeze(-1).unsqueeze(-1)
            + cosine * scaled_x
            - sine * scaled_y
        )
        y = (
            center[..., 1].unsqueeze(-1).unsqueeze(-1)
            + sine * scaled_x
            + cosine * scaled_y
        )
        return torch.stack((x, y), dim=-1).mul(2.0).sub(1.0)

    @staticmethod
    def _sample_rois(
        feature_map: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        batch, faces, roi_height, roi_width = grid.shape[:4]
        packed_grid = grid.reshape(batch, faces * roi_height, roi_width, 2)
        sampled = F.grid_sample(
            feature_map,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        channels = sampled.shape[1]
        return (
            sampled.reshape(batch, channels, faces, roi_height, roi_width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _global_to_local(
        self,
        coordinates: torch.Tensor,
        ellipses: torch.Tensor,
    ) -> torch.Tensor:
        delta = coordinates - ellipses[..., None, :2]
        theta = ellipses[..., 4].unsqueeze(-1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        local_x = cosine * delta[..., 0] + sine * delta[..., 1]
        local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
        diameter = (
            2.0
            * ellipses[..., 2].clamp_min(self.minimum_radius)
            * self.roi_scale
        ).unsqueeze(-1)
        return (
            0.5 + torch.stack((local_x, local_y), dim=-1) / diameter.unsqueeze(-1)
        ).clamp(0.0, 1.0)

    def _local_to_global(
        self,
        coordinates: torch.Tensor,
        ellipses: torch.Tensor,
    ) -> torch.Tensor:
        half_side = (
            ellipses[..., 2].clamp_min(self.minimum_radius) * self.roi_scale
        ).unsqueeze(-1)
        local = coordinates.mul(2.0).sub(1.0) * half_side.unsqueeze(-1)
        theta = ellipses[..., 4].unsqueeze(-1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = cosine * local[..., 0] - sine * local[..., 1]
        y = sine * local[..., 0] + cosine * local[..., 1]
        return (
            ellipses[..., None, :2] + torch.stack((x, y), dim=-1)
        ).clamp(0.0, 1.0)

    def forward(
        self,
        *,
        global_query_features: torch.Tensor,
        face_queries: torch.Tensor,
        coarse_coordinates: torch.Tensor,
        presence_logits: torch.Tensor | None,
        visibility_logits: torch.Tensor,
        ellipses: torch.Tensor,
        deim_feature: torch.Tensor,
        high_resolution_feature: torch.Tensor,
    ) -> dict[str, Any]:
        geometry = ellipses.detach() if self.detach_ellipse else ellipses
        grid = self._ellipse_grid(geometry)
        deim_roi = self._sample_rois(deim_feature, grid)
        high_resolution_roi = self._sample_rois(high_resolution_feature, grid)
        batch, faces = deim_roi.shape[:2]
        deim_roi = deim_roi.reshape(
            batch * faces, *deim_roi.shape[2:]
        )
        high_resolution_roi = high_resolution_roi.reshape(
            batch * faces, *high_resolution_roi.shape[2:]
        )
        fused = self.fusion(
            torch.cat(
                (
                    self.deim_projection(deim_roi),
                    self.high_resolution_projection(high_resolution_roi),
                ),
                dim=1,
            )
        )
        memory = fused.flatten(2).transpose(1, 2).contiguous()
        position = sine_position_encoding_2d(
            self.roi_size,
            self.roi_size,
            self.dimension,
            device=fused.device,
            dtype=fused.dtype,
        )
        geometry_vector = torch.cat(
            (
                geometry[..., :4],
                torch.cos(2.0 * geometry[..., 4:5]),
                torch.sin(2.0 * geometry[..., 4:5]),
            ),
            dim=-1,
        )
        queries = self.query_norm(
            global_query_features
            + self.face_projection(face_queries).unsqueeze(2)
            + self.geometry_projection(geometry_vector).unsqueeze(2)
        ).reshape(batch * faces, 5, self.dimension)
        for layer in self.layers:
            queries = layer(queries, memory + position)

        local_coordinates = self._global_to_local(
            coarse_coordinates,
            geometry,
        ).reshape(batch * faces, 5, 2)
        local_coordinates = (
            local_coordinates
            + torch.tanh(self.base_delta_head(queries))
            * self.maximum_base_delta
        ).clamp(0.0, 1.0)
        presence = (
            presence_logits.reshape(batch * faces, 5)
            if presence_logits is not None
            else None
        )
        visibility = visibility_logits.reshape(batch * faces, 5)
        base_global = self._local_to_global(
            local_coordinates.reshape(batch, faces, 5, 2),
            geometry,
        )
        auxiliary_global = [base_global]
        refinement_global: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.refinement_stages):
            if self.state_head_mode == FACTORIZED_STATE_HEAD:
                local_coordinates, presence, visibility, residual = stage(
                    fused,
                    queries,
                    local_coordinates,
                    visibility,
                    presence,
                )
            else:
                local_coordinates, visibility, residual = stage(
                    fused,
                    queries,
                    local_coordinates,
                    visibility,
                )
            queries = queries + residual
            global_coordinates = self._local_to_global(
                local_coordinates.reshape(batch, faces, 5, 2),
                geometry,
            )
            refinement_global.append(global_coordinates)
            if stage_index + 1 < len(self.refinement_stages):
                auxiliary_global.append(global_coordinates)

        return {
            "coordinates": refinement_global[-1],
            "presence_logits": (
                presence.reshape(batch, faces, 5)
                if presence is not None
                else None
            ),
            "visibility_logits": visibility.reshape(batch, faces, 5),
            "auxiliary_coordinates": auxiliary_global,
            "base_coordinates": base_global,
            "refinement_coordinates": refinement_global,
            "roi_shape": (self.roi_size, self.roi_size),
        }


class DEIMv2JointFaceLandmarks(nn.Module):
    """DEIMv2 Face geometry with five child landmark queries per selected face."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        detector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(config)
        detector_cfg = dict(cfg.get("detector", {}))
        if detector is None:
            config_path = Path(str(detector_cfg["config"]))
            checkpoint_path = Path(str(detector_cfg["checkpoint"]))
            if not config_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "The trained DEIMv2-L face detector config/checkpoint is required for the joint model"
                )
            detector = build_detector_from_config(load_config(config_path))
            missing, unexpected = load_checkpoint(detector, checkpoint_path, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"DEIMv2-L detector checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
        self.face_detector = detector
        self.face_output_geometry = str(
            cfg.get("face_output_geometry", "bbox")
        ).lower()
        if self.face_output_geometry not in {"bbox", "ellipse"}:
            raise ValueError(
                "face_output_geometry must be either 'bbox' or 'ellipse'"
            )
        dimension = int(cfg.get("dimension", getattr(self.face_detector.decoder, "hidden_dim", 224)))
        self.maximum_face_queries = int(cfg.get("maximum_face_queries", 100))
        state_head_mode = str(
            cfg.get("state_head_mode", LEGACY_BINARY_STATE_HEAD)
        )
        self.landmark_decoder = LandmarkQueryDecoder(
            dimension,
            layers=int(cfg.get("landmark_decoder_layers", 4)),
            heads=int(cfg.get("heads", 8)),
            feedforward=int(cfg.get("feedforward", dimension * 4)),
            dropout=float(cfg.get("dropout", 0.0)),
            state_head_mode=state_head_mode,
        )
        local_cfg = dict(cfg.get("ellipse_local_refinement", {}))
        self.ellipse_local_refinement: (
            EllipseConditionedLocalRefinement | None
        ) = None
        if bool(local_cfg.get("enabled", False)):
            if self.face_output_geometry != "ellipse":
                raise ValueError(
                    "ellipse_local_refinement requires "
                    "face_output_geometry='ellipse'"
                )
            high_resolution_channels = int(
                local_cfg.get(
                    "high_resolution_channels",
                    getattr(
                        self.face_detector.backbone,
                        "spatial_high_resolution_channels",
                        0,
                    ),
                )
            )
            self.ellipse_local_refinement = EllipseConditionedLocalRefinement(
                dimension,
                high_resolution_channels=high_resolution_channels,
                roi_size=int(local_cfg.get("roi_size", 16)),
                roi_scale=float(local_cfg.get("roi_scale", 1.25)),
                layers=int(local_cfg.get("layers", 2)),
                heads=int(local_cfg.get("heads", cfg.get("heads", 8))),
                feedforward=int(
                    local_cfg.get("feedforward", dimension * 4)
                ),
                dropout=float(local_cfg.get("dropout", 0.0)),
                maximum_base_delta=float(
                    local_cfg.get("maximum_base_delta", 0.10)
                ),
                minimum_radius=float(local_cfg.get("minimum_radius", 1.0e-3)),
                detach_ellipse=bool(local_cfg.get("detach_ellipse", True)),
                stages=local_cfg.get("stages"),
                state_head_mode=state_head_mode,
            )
        self.keypoint_loss_weight = float(cfg.get("keypoint_loss_weight", 10.0))
        self.presence_loss_weight = float(cfg.get("presence_loss_weight", 1.0))
        self.visibility_loss_weight = float(cfg.get("visibility_loss_weight", 1.0))
        self.auxiliary_loss_weight = float(cfg.get("auxiliary_loss_weight", 0.5))
        self._latest_face_queries: torch.Tensor | None = None
        decoder_layers = self.face_detector.decoder.decoder.layers
        decoder_layers[-1].register_forward_hook(self._capture_face_queries)

    def _capture_face_queries(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        self._latest_face_queries = output

    @staticmethod
    def _memory(encoded: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([feature.flatten(2).transpose(1, 2) for feature in encoded], dim=1)

    def _landmark_losses(
        self,
        predictions: torch.Tensor,
        presence_logits: torch.Tensor | None,
        visibility_logits: torch.Tensor,
        auxiliary: list[torch.Tensor],
        candidates: torch.Tensor,
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        coordinate_terms: list[torch.Tensor] = []
        presence_terms: list[torch.Tensor] = []
        visibility_terms: list[torch.Tensor] = []
        auxiliary_terms: list[torch.Tensor] = []
        for batch_index, ((prediction_indices, target_indices), target) in enumerate(zip(matches, targets)):
            if prediction_indices.numel() == 0:
                continue
            candidate_row = candidates[batch_index]
            positions = []
            valid_targets = []
            for prediction_index, target_index in zip(prediction_indices.tolist(), target_indices.tolist()):
                found = torch.nonzero(candidate_row == int(prediction_index), as_tuple=False)
                if found.numel():
                    positions.append(int(found[0, 0]))
                    valid_targets.append(int(target_index))
            if not positions:
                continue
            position_tensor = torch.tensor(positions, device=predictions.device, dtype=torch.long)
            target_tensor = torch.tensor(valid_targets, device=predictions.device, dtype=torch.long)
            target_coordinates = target["keypoints"].to(predictions.device)[target_tensor]
            target_visible = target["visible"].to(predictions.device)[target_tensor].to(predictions.dtype)
            target_labeled = target.get("labeled", target["visible"]).to(predictions.device)[target_tensor].to(
                predictions.dtype
            )
            prediction = predictions[batch_index, position_tensor]
            weight = target_labeled
            denominator = weight.sum().clamp_min(1.0)
            coordinate_terms.append(
                (F.smooth_l1_loss(prediction, target_coordinates, reduction="none").sum(-1) * weight).sum()
                / denominator
            )
            if presence_logits is not None:
                # Presence is supervised for all five semantic slots:
                # missing=0 versus occluded/visible=1.  Unlike coordinate and
                # visibility losses, missing points must not be masked here.
                presence_terms.append(
                    F.binary_cross_entropy_with_logits(
                        presence_logits[batch_index, position_tensor],
                        target_labeled,
                        reduction="mean",
                    )
                )
            visibility_terms.append(
                (
                    F.binary_cross_entropy_with_logits(
                        visibility_logits[batch_index, position_tensor],
                        target_visible,
                        reduction="none",
                    )
                    * weight
                )
                .sum()
                / denominator
            )
            if auxiliary:
                auxiliary_terms.append(
                    sum(
                        (
                            F.smooth_l1_loss(
                                value[batch_index, position_tensor],
                                target_coordinates,
                                reduction="none",
                            ).sum(-1)
                            * weight
                        ).sum()
                        / denominator
                        for value in auxiliary
                    )
                    / len(auxiliary)
                )
        zero = predictions.sum() * 0.0
        loss_keypoint = torch.stack(coordinate_terms).mean() if coordinate_terms else zero
        loss_presence = torch.stack(presence_terms).mean() if presence_terms else zero
        loss_visibility = torch.stack(visibility_terms).mean() if visibility_terms else zero
        loss_auxiliary = torch.stack(auxiliary_terms).mean() if auxiliary_terms else zero
        losses = {
            "loss_keypoint": loss_keypoint,
            "loss_visibility": loss_visibility,
            "loss_landmark_aux": loss_auxiliary,
        }
        if presence_logits is not None:
            losses["loss_presence"] = loss_presence
        return losses

    def forward(
        self,
        images: torch.Tensor,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        official_targets = None
        if targets is not None:
            official_targets = self.face_detector._targets_to_official(
                list(targets),
                (int(images.shape[-2]), int(images.shape[-1])),
                device=images.device,
                dtype=images.dtype,
            )
        high_resolution_feature = None
        if self.ellipse_local_refinement is not None:
            if images.shape[-2] != images.shape[-1]:
                raise RuntimeError(
                    "ellipse_local_refinement currently requires a square "
                    "input because ellipse radii are normalized by the "
                    "maximum image side"
                )
            high_resolution_forward = getattr(
                self.face_detector.backbone,
                "forward_with_high_resolution",
                None,
            )
            if not callable(high_resolution_forward):
                raise RuntimeError(
                    "ellipse_local_refinement requires a DINO+STA backbone "
                    "with forward_with_high_resolution"
                )
            features, high_resolution_feature = high_resolution_forward(images)
        else:
            features = self.face_detector.backbone(images)
        encoded = self.face_detector.encoder(list(features))
        self._latest_face_queries = None
        raw = self.face_detector.decoder(encoded, official_targets)
        if self._latest_face_queries is None:
            raise RuntimeError("Failed to capture DEIMv2 decoder face object queries")
        number_of_queries = raw["pred_boxes"].shape[1]
        face_queries = self._latest_face_queries[:, -number_of_queries:, :]

        matches = None
        if official_targets is not None:
            matches = self.face_detector.criterion.matcher(
                {"pred_logits": raw["pred_logits"], "pred_boxes": raw["pred_boxes"]},
                official_targets,
                epoch=int(getattr(self.face_detector, "current_epoch", 0)),
            )["indices"]
        candidates = _candidate_indices(
            raw["pred_logits"],
            maximum=self.maximum_face_queries,
            matched=matches,
        )
        selected_queries = _gather(face_queries, candidates)
        selected_boxes = _gather(raw["pred_boxes"], candidates)
        selected_logits = _gather(raw["pred_logits"], candidates)
        if self.face_output_geometry == "ellipse" and "pred_ellipses" not in raw:
            raise RuntimeError(
                "The new DEIM Joint requires pred_ellipses from its Face decoder"
            )
        selected_ellipses = (
            _gather(raw["pred_ellipses"], candidates)
            if "pred_ellipses" in raw
            else None
        )
        landmark_ellipses = (
            selected_ellipses
            if self.face_output_geometry == "ellipse"
            else None
        )
        if self.ellipse_local_refinement is not None:
            (
                keypoints,
                presence_logits,
                visibility_logits,
                auxiliary,
                global_query_features,
            ) = self.landmark_decoder(
                selected_queries,
                selected_boxes,
                self._memory(encoded),
                face_ellipses=landmark_ellipses,
                return_query_features=True,
            )
        else:
            (
                keypoints,
                presence_logits,
                visibility_logits,
                auxiliary,
            ) = self.landmark_decoder(
                selected_queries,
                selected_boxes,
                self._memory(encoded),
                face_ellipses=landmark_ellipses,
            )
        local_output: dict[str, Any] | None = None
        if self.ellipse_local_refinement is not None:
            if selected_ellipses is None:
                raise RuntimeError(
                    "ellipse_local_refinement requires pred_ellipses from "
                    "the Face decoder"
                )
            if high_resolution_feature is None:
                raise RuntimeError(
                    "ellipse_local_refinement did not receive the STA c1 feature"
                )
            global_coarse_keypoints = keypoints
            local_output = self.ellipse_local_refinement(
                global_query_features=global_query_features,
                face_queries=selected_queries,
                coarse_coordinates=keypoints,
                presence_logits=presence_logits,
                visibility_logits=visibility_logits,
                ellipses=selected_ellipses,
                deim_feature=encoded[0],
                high_resolution_feature=high_resolution_feature,
            )
            keypoints = local_output["coordinates"]
            presence_logits = local_output["presence_logits"]
            visibility_logits = local_output["visibility_logits"]
            auxiliary = [
                *auxiliary,
                *local_output["auxiliary_coordinates"],
            ]
        output: dict[str, Any] = {
            "pred_logits": selected_logits,
            "pred_boxes": selected_boxes,
            "pred_keypoints": keypoints,
            "visibility_logits": visibility_logits,
            "candidate_indices": candidates,
            "auxiliary_keypoints": auxiliary,
            "raw_detector_output": raw,
        }
        if self.face_output_geometry == "ellipse":
            assert selected_ellipses is not None
            output["face_geometry_type"] = "ellipse"
            output["pred_face_geometry"] = selected_ellipses
            output["pred_detector_boxes"] = selected_boxes
        if matches is not None:
            output["match_indices"] = matches
        if presence_logits is not None:
            output["presence_logits"] = presence_logits
        if selected_ellipses is not None:
            output["pred_ellipses"] = selected_ellipses
        if local_output is not None:
            output["global_coarse_keypoints"] = global_coarse_keypoints
            output["local_base_keypoints"] = local_output["base_coordinates"]
            output["local_refinement_keypoints"] = local_output[
                "refinement_coordinates"
            ]
            output["local_roi_shape"] = local_output["roi_shape"]
        if targets is None or official_targets is None or matches is None:
            output["scores"] = selected_logits.sigmoid().amax(-1)
            if presence_logits is not None:
                output["presence"] = presence_logits.sigmoid()
                output["state_probabilities"] = factorized_state_probabilities(
                    presence_logits,
                    visibility_logits,
                )
            output["visibility"] = visibility_logits.sigmoid()
            return output
        face_losses = self.face_detector.criterion(
            raw,
            official_targets,
            epoch=int(getattr(self.face_detector, "current_epoch", 0)),
        )
        formatted_face = self.face_detector._format_losses(face_losses)
        landmark_losses = self._landmark_losses(
            keypoints,
            presence_logits,
            visibility_logits,
            auxiliary,
            candidates,
            matches,
            targets,
        )
        output.update({f"face_{key}": value for key, value in formatted_face.items()})
        output.update(landmark_losses)
        output["loss_total"] = (
            formatted_face["loss_total"]
            + self.keypoint_loss_weight * landmark_losses["loss_keypoint"]
            + self.presence_loss_weight
            * landmark_losses.get("loss_presence", keypoints.sum() * 0.0)
            + self.visibility_loss_weight * landmark_losses["loss_visibility"]
            + self.auxiliary_loss_weight * landmark_losses["loss_landmark_aux"]
        )
        return output
