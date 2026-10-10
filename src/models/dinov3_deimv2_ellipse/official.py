from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dinov3_deimv2.official import (
    DINOv3DEIMv2Detector,
    _load_pretrained_checkpoint,
    build_dinov3_deimv2_detector,
)
from src.third_party.deimv2.engine.deim.deim_criterion import DEIMCriterion
from src.third_party.deimv2.engine.deim.deim_decoder import DEIMTransformer, TransformerDecoder
from src.third_party.deimv2.engine.deim.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from src.third_party.deimv2.engine.deim.dfine_utils import distance2bbox, weighting_function
from src.third_party.deimv2.engine.deim.postprocessor import PostProcessor, mod
from src.third_party.deimv2.engine.deim.utils import inverse_sigmoid


def canonicalize_ellipses(ellipses: torch.Tensor) -> torch.Tensor:
    """Canonicalize ``[cx, cy, rx, ry, theta]`` to major/minor-axis form."""

    if ellipses.shape[-1] != 5:
        raise ValueError(f"Expected ellipse tensors ending in 5 values, got {tuple(ellipses.shape)}")
    cx, cy, first_radius, second_radius, theta = ellipses.unbind(-1)
    swap = second_radius > first_radius
    major = torch.maximum(first_radius, second_radius)
    minor = torch.minimum(first_radius, second_radius)
    theta = theta + swap.to(theta.dtype) * (math.pi * 0.5)
    theta = torch.remainder(theta + math.pi * 0.5, math.pi) - math.pi * 0.5
    return torch.stack((cx, cy, major, minor, theta), dim=-1)


def ellipse_to_aabb(ellipses: torch.Tensor) -> torch.Tensor:
    """Return the enclosing XYXY box for pixel-space canonical ellipses."""

    cx, cy, major, minor, theta = ellipses.unbind(-1)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    half_width = torch.sqrt((major * cosine).square() + (minor * sine).square()).clamp_min(0.0)
    half_height = torch.sqrt((major * sine).square() + (minor * cosine).square()).clamp_min(0.0)
    return torch.stack((cx - half_width, cy - half_height, cx + half_width, cy + half_height), dim=-1)


def scale_ellipses_anisotropic(
    ellipses: torch.Tensor,
    *,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:
    """Apply an axis-aligned affine scale and refit the exact transformed ellipse."""

    if ellipses.numel() == 0:
        return ellipses.clone()
    canonical = canonicalize_ellipses(ellipses)
    cx, cy, major, minor, theta = canonical.unbind(-1)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    major_squared = major.square()
    minor_squared = minor.square()
    covariance_xx = (major_squared * cosine.square() + minor_squared * sine.square()) * float(scale_x) ** 2
    covariance_yy = (major_squared * sine.square() + minor_squared * cosine.square()) * float(scale_y) ** 2
    covariance_xy = (major_squared - minor_squared) * sine * cosine * float(scale_x) * float(scale_y)
    covariance = torch.stack(
        (
            torch.stack((covariance_xx, covariance_xy), dim=-1),
            torch.stack((covariance_xy, covariance_yy), dim=-1),
        ),
        dim=-2,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    transformed_major = eigenvalues[..., 1].clamp_min(0.0).sqrt()
    transformed_minor = eigenvalues[..., 0].clamp_min(0.0).sqrt()
    major_vector = eigenvectors[..., :, 1]
    transformed_theta = torch.atan2(major_vector[..., 1], major_vector[..., 0])
    transformed = torch.stack(
        (
            cx * float(scale_x),
            cy * float(scale_y),
            transformed_major,
            transformed_minor,
            transformed_theta,
        ),
        dim=-1,
    )
    return canonicalize_ellipses(transformed)


def ellipse_contour_points(ellipses: torch.Tensor, *, points: int = 36) -> torch.Tensor:
    """Sample normalized contour points from ``[..., cx, cy, a, b, theta]``."""

    angles = torch.arange(points, dtype=ellipses.dtype, device=ellipses.device) * (2.0 * math.pi / points)
    unit_x = torch.cos(angles)
    unit_y = torch.sin(angles)
    cx, cy, major, minor, theta = ellipses.unbind(-1)
    cosine = torch.cos(theta).unsqueeze(-1)
    sine = torch.sin(theta).unsqueeze(-1)
    local_x = major.unsqueeze(-1) * unit_x
    local_y = minor.unsqueeze(-1) * unit_y
    x = cx.unsqueeze(-1) + local_x * cosine - local_y * sine
    y = cy.unsqueeze(-1) + local_x * sine + local_y * cosine
    return torch.stack((x, y), dim=-1)


def _ellipse_angle_vectors(ellipses: torch.Tensor) -> torch.Tensor:
    """Return the pi-periodic unit-vector representation of ellipse angles."""

    doubled = 2.0 * ellipses[..., 4]
    return torch.stack((torch.cos(doubled), torch.sin(doubled)), dim=-1)


class EllipseLocalQueryRefinementStage(nn.Module):
    """Update Face queries from an ellipse-aligned local multi-scale ROI."""

    def __init__(
        self,
        dimension: int,
        *,
        num_levels: int,
        heads: int,
        feedforward: int,
        roi_size: int,
        roi_scale: float,
        minimum_axis_ratio: float,
        maximum_center_delta: float,
        maximum_log_radius_delta: float,
        maximum_ratio_logit_delta: float,
        maximum_angle_delta_degrees: float,
        dropout: float,
        detach_reference: bool,
        attention_fp32: bool,
        residual_gate: float,
    ) -> None:
        super().__init__()
        if dimension <= 0 or dimension % heads != 0:
            raise ValueError("Ellipse refinement dimension must be positive and divisible by heads")
        if num_levels <= 0:
            raise ValueError("Ellipse refinement requires at least one feature level")
        if roi_size < 2:
            raise ValueError("Ellipse refinement roi_size must be at least two")
        if roi_scale <= 0.0:
            raise ValueError("Ellipse refinement roi_scale must be positive")
        if not 0.0 < minimum_axis_ratio <= 1.0:
            raise ValueError("Ellipse refinement minimum_axis_ratio must be in (0, 1]")
        for name, value in {
            "maximum_center_delta": maximum_center_delta,
            "maximum_log_radius_delta": maximum_log_radius_delta,
            "maximum_ratio_logit_delta": maximum_ratio_logit_delta,
            "maximum_angle_delta_degrees": maximum_angle_delta_degrees,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Ellipse refinement {name} must be finite and non-negative")

        self.dimension = int(dimension)
        self.num_levels = int(num_levels)
        self.roi_size = int(roi_size)
        self.roi_scale = float(roi_scale)
        self.minimum_axis_ratio = float(minimum_axis_ratio)
        self.maximum_center_delta = float(maximum_center_delta)
        self.maximum_log_radius_delta = float(maximum_log_radius_delta)
        self.maximum_ratio_logit_delta = float(maximum_ratio_logit_delta)
        self.maximum_angle_delta = math.radians(float(maximum_angle_delta_degrees))
        self.detach_reference = bool(detach_reference)
        self.attention_fp32 = bool(attention_fp32)

        self.memory_projection = nn.Linear(dimension, dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.query_norm = nn.LayerNorm(dimension)
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.local_position = nn.Parameter(
            torch.zeros(num_levels, roi_size * roi_size, dimension)
        )
        self.cross_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_gate = nn.Parameter(torch.tensor(float(residual_gate)))
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.delta_head = nn.Linear(dimension, 5)
        nn.init.normal_(self.local_position, std=0.02)
        # An enabled refinement model can warm-start from the one-shot ellipse
        # checkpoint without changing its initial predictions. Gradients first
        # establish the residual head and subsequently train the local attention.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _ellipse_grid(self, ellipses: torch.Tensor) -> torch.Tensor:
        axis = (
            (torch.arange(
                self.roi_size,
                dtype=ellipses.dtype,
                device=ellipses.device,
            ) + 0.5)
            / float(self.roi_size)
        ).mul(2.0).sub(1.0)
        local_y, local_x = torch.meshgrid(axis, axis, indexing="ij")
        local_x = local_x.reshape(1, 1, self.roi_size, self.roi_size)
        local_y = local_y.reshape(1, 1, self.roi_size, self.roi_size)

        center = ellipses[..., :2]
        major = ellipses[..., 2].clamp_min(1.0e-6)
        minor = torch.maximum(
            ellipses[..., 3].clamp_min(1.0e-6),
            major * self.minimum_axis_ratio,
        )
        cosine = torch.cos(ellipses[..., 4]).unsqueeze(-1).unsqueeze(-1)
        sine = torch.sin(ellipses[..., 4]).unsqueeze(-1).unsqueeze(-1)
        scaled_x = local_x * (major * self.roi_scale).unsqueeze(-1).unsqueeze(-1)
        scaled_y = local_y * (minor * self.roi_scale).unsqueeze(-1).unsqueeze(-1)
        x = center[..., 0, None, None] + cosine * scaled_x - sine * scaled_y
        y = center[..., 1, None, None] + sine * scaled_x + cosine * scaled_y
        return torch.stack((x, y), dim=-1).mul(2.0).sub(1.0)

    @staticmethod
    def _sample_level(feature_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch, queries, roi_height, roi_width = grid.shape[:4]
        packed_grid = grid.reshape(batch, queries * roi_height, roi_width, 2)
        sampled = F.grid_sample(
            feature_map,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        channels = sampled.shape[1]
        return (
            sampled.reshape(batch, channels, queries, roi_height * roi_width)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    @staticmethod
    def _geometry_features(ellipses: torch.Tensor) -> torch.Tensor:
        major = ellipses[..., 2].clamp_min(1.0e-6)
        ratio = (ellipses[..., 3] / major).clamp(1.0e-4, 1.0 - 1.0e-4)
        theta = ellipses[..., 4]
        return torch.stack(
            (
                ellipses[..., 0],
                ellipses[..., 1],
                torch.log(major),
                torch.logit(ratio),
                torch.cos(2.0 * theta),
                torch.sin(2.0 * theta),
            ),
            dim=-1,
        )

    def _update_ellipse(
        self,
        ellipses: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.delta_head(query)
        major = ellipses[..., 2].clamp_min(1.0e-6)
        center = ellipses[..., :2] + (
            torch.tanh(raw[..., :2])
            * self.maximum_center_delta
            * major.unsqueeze(-1)
        )
        refined_major = major * torch.exp(
            torch.tanh(raw[..., 2]) * self.maximum_log_radius_delta
        )
        ratio = (ellipses[..., 3] / major).clamp(1.0e-4, 1.0 - 1.0e-4)
        refined_ratio = torch.sigmoid(
            torch.logit(ratio)
            + torch.tanh(raw[..., 3]) * self.maximum_ratio_logit_delta
        ).clamp(1.0e-4, 1.0 - 1.0e-4)
        theta = ellipses[..., 4] + (
            torch.tanh(raw[..., 4]) * self.maximum_angle_delta
        )
        refined = torch.stack(
            (
                center[..., 0],
                center[..., 1],
                refined_major,
                refined_major * refined_ratio,
                theta,
            ),
            dim=-1,
        )
        return canonicalize_ellipses(refined)

    def forward(
        self,
        query: torch.Tensor,
        ellipses: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(feature_maps) != self.num_levels:
            raise ValueError(
                "Ellipse refinement feature-level mismatch: "
                f"expected={self.num_levels}, got={len(feature_maps)}"
            )
        reference = ellipses.detach() if self.detach_reference else ellipses
        grid = self._ellipse_grid(reference)
        sampled_levels = [self._sample_level(feature, grid) for feature in feature_maps]
        local_memory = torch.stack(sampled_levels, dim=2)
        batch, queries, levels, samples, channels = local_memory.shape
        if channels != self.dimension:
            raise ValueError(
                f"Ellipse refinement expected memory dimension {self.dimension}, got {channels}"
            )
        local_memory = self.memory_projection(local_memory)
        local_memory = self.memory_norm(local_memory)
        position = self.local_position.to(dtype=local_memory.dtype).reshape(
            1, 1, levels, samples, channels
        )
        keys = (local_memory + position).reshape(
            batch * queries, levels * samples, channels
        )
        values = local_memory.reshape(batch * queries, levels * samples, channels)
        normalized_query = self.query_norm(query)
        query_input = normalized_query + self.geometry_projection(
            self._geometry_features(reference).to(dtype=query.dtype)
        )
        packed_query = query_input.reshape(batch * queries, 1, channels)
        if self.attention_fp32:
            with torch.autocast(device_type=query.device.type, enabled=False):
                attended, _ = self.cross_attention(
                    packed_query.float(),
                    keys.float(),
                    values.float(),
                    need_weights=False,
                )
            attended = attended.to(dtype=query.dtype)
        else:
            attended, _ = self.cross_attention(
                packed_query,
                keys,
                values,
                need_weights=False,
            )
        attended = attended.reshape(batch, queries, channels)
        query = query + self.attention_gate.to(dtype=query.dtype) * self.dropout(attended)
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query, self._update_ellipse(ellipses, query)


class EllipseLocalQueryRefinement(nn.Module):
    """Two-stage coarse-to-fine ellipse refinement over local image features."""

    def __init__(
        self,
        dimension: int,
        *,
        num_levels: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        stages = list(config.get("stages", ()))
        if len(stages) != 2:
            raise ValueError("Ellipse local query refinement requires exactly two stages")
        heads = int(config.get("heads", 8))
        feedforward = int(config.get("feedforward", dimension * 4))
        dropout = float(config.get("dropout", 0.0))
        common = {
            "num_levels": num_levels,
            "heads": heads,
            "feedforward": feedforward,
            "dropout": dropout,
            "detach_reference": bool(config.get("detach_reference", True)),
            "attention_fp32": bool(config.get("attention_fp32", True)),
            "residual_gate": float(config.get("residual_gate", 0.1)),
        }
        self.stages = nn.ModuleList(
            [
                EllipseLocalQueryRefinementStage(
                    dimension,
                    roi_size=int(stage.get("roi_size", 5)),
                    roi_scale=float(stage["roi_scale"]),
                    minimum_axis_ratio=float(stage.get("minimum_axis_ratio", 0.50)),
                    maximum_center_delta=float(stage.get("maximum_center_delta", 0.50)),
                    maximum_log_radius_delta=float(stage.get("maximum_log_radius_delta", 0.35)),
                    maximum_ratio_logit_delta=float(stage.get("maximum_ratio_logit_delta", 0.50)),
                    maximum_angle_delta_degrees=float(stage.get("maximum_angle_delta_degrees", 20.0)),
                    **common,
                )
                for stage in stages
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        ellipses: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        stage_outputs: list[torch.Tensor] = []
        for stage in self.stages:
            query, ellipses = stage(query, ellipses, feature_maps)
            stage_outputs.append(ellipses)
        return query, ellipses, stage_outputs


class EllipseTransformerDecoder(TransformerDecoder):
    """DEIM decoder variant that also exposes query states used by the ellipse head."""

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        query_conditioner=None,
        global_context=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_hidden = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
        search_reference = ref_points_detach
        if query_conditioner is not None:
            output, search_reference = query_conditioner(
                stage_index=0,
                content=output,
                face_reference=ref_points_detach,
                dn_meta=dn_meta,
                final=False,
            )

        for index, layer in enumerate(self.layers):
            eval_terminal = not self.training and index == self.eval_idx
            ref_points_input = search_reference.unsqueeze(2)

            if index >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            previous_output = output
            refined_output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
                global_context,
            )
            output = self.apply_added_layer_residual(
                index, previous_output, refined_output
            )

            if index == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[index](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or index == self.eval_idx:
                scores = score_head[index](output)
                scores = self.lqe_layers[index](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_hidden.append(output)
                if not self.training and query_conditioner is None:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            search_reference = ref_points_detach
            if query_conditioner is not None:
                output, search_reference = query_conditioner(
                    stage_index=index + 1,
                    content=output,
                    face_reference=ref_points_detach,
                    dn_meta=dn_meta,
                    final=eval_terminal or index + 1 == len(self.layers),
                )
            if eval_terminal:
                break
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            torch.stack(dec_out_hidden),
        )


class EllipseDEIMTransformer(DEIMTransformer):
    """DEIMv2 transformer with an additive direct ellipse regression head."""

    def initialize_ellipse_head(
        self,
        *,
        hidden_dim: int,
        prior_major: float = 0.05,
        prior_axis_ratio: float = 0.70,
        prior_angle_degrees: float = 90.0,
        refinement: Mapping[str, Any] | None = None,
    ) -> None:
        self.decoder.__class__ = EllipseTransformerDecoder
        self.ellipse_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        output_layer = self.ellipse_head[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        major = max(float(prior_major), 1.0e-4)
        ratio = min(max(float(prior_axis_ratio), 1.0e-4), 1.0 - 1.0e-4)
        angle = math.radians(float(prior_angle_degrees))
        with torch.no_grad():
            output_layer.bias[2] = math.log(math.expm1(major))
            output_layer.bias[3] = math.log(ratio / (1.0 - ratio))
            output_layer.bias[4] = math.cos(2.0 * angle)
            output_layer.bias[5] = math.sin(2.0 * angle)
        refinement_cfg = dict(refinement or {})
        self.ellipse_refinement = (
            EllipseLocalQueryRefinement(
                hidden_dim,
                num_levels=int(getattr(self, "num_levels", 3)),
                config=refinement_cfg,
            )
            if bool(refinement_cfg.get("enabled", False))
            else None
        )
        self.depth_supervision_layers: tuple[int, ...] = ()

    def configure_depth_supervision(self, layers: list[int] | tuple[int, ...]) -> None:
        """Select one-based decoder layers exposed to depth-specific losses."""

        layer_count = len(self.decoder.layers)
        resolved = tuple(sorted({int(layer) for layer in layers}))
        if any(layer < 1 or layer > layer_count for layer in resolved):
            raise ValueError(
                f"Depth supervision layers must be in [1, {layer_count}], got {resolved}"
            )
        self.depth_supervision_layers = resolved

    def _predict_ellipses(self, hidden: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.ellipse_head(hidden)
        center = boxes[..., :2].detach() + raw[..., :2]
        major = F.softplus(raw[..., 2]).clamp_min(1.0e-6)
        minor = major * torch.sigmoid(raw[..., 3]).clamp(1.0e-4, 1.0 - 1.0e-4)
        angle_vector = F.normalize(raw[..., 4:6], dim=-1, eps=1.0e-6)
        theta = 0.5 * torch.atan2(angle_vector[..., 1], angle_vector[..., 0])
        ellipses = torch.stack((center[..., 0], center[..., 1], major, minor, theta), dim=-1)
        return ellipses, angle_vector

    @staticmethod
    def _memory_feature_maps(
        memory: torch.Tensor,
        feats: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        split_sizes = [int(feature.shape[-2] * feature.shape[-1]) for feature in feats]
        if sum(split_sizes) != int(memory.shape[1]):
            raise ValueError(
                "Ellipse refinement memory size does not match the feature pyramid: "
                f"memory={memory.shape[1]}, pyramid={split_sizes}"
            )
        levels = memory.split(split_sizes, dim=1)
        return tuple(
            level.transpose(1, 2).reshape(
                memory.shape[0],
                memory.shape[2],
                int(feature.shape[-2]),
                int(feature.shape[-1]),
            ).contiguous()
            for level, feature in zip(levels, feats, strict=True)
        )

    def forward(
        self,
        feats,
        targets=None,
        feature_padding_masks=None,
        query_conditioner=None,
    ):
        memory, spatial_shapes = self._get_encoder_input(feats)
        global_context = (
            self.global_context_projection(feats[-1])
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
            if self.global_context_attention
            else None
        )
        memory_padding_mask = self._flatten_feature_padding_masks(
            feats, feature_padding_masks
        )

        if self.training and self.num_denoising > 0:
            from src.third_party.deimv2.engine.deim.denoising import get_contrastive_denoising_training_group

            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=1.0,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
            memory_padding_mask,
        )

        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            out_hidden,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            query_conditioner=query_conditioner,
            global_context=global_context,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            _dn_hidden, out_hidden = torch.split(out_hidden, dn_meta["dn_num_split"], dim=2)

        base_ellipses, base_angle_vectors = self._predict_ellipses(
            out_hidden[-1], out_bboxes[-1]
        )
        pred_ellipses = base_ellipses
        pred_angle_vectors = base_angle_vectors
        ellipse_refinement_aux_outputs: list[dict[str, Any]] = []
        if self.ellipse_refinement is not None:
            refinement_memory = memory
            if memory_padding_mask is not None:
                refinement_memory = refinement_memory * (
                    ~memory_padding_mask
                ).unsqueeze(-1).to(dtype=refinement_memory.dtype)
            refinement_features = self._memory_feature_maps(
                refinement_memory,
                feats,
            )
            _refined_query, pred_ellipses, refinement_stages = (
                self.ellipse_refinement(
                    out_hidden[-1],
                    base_ellipses,
                    refinement_features,
                )
            )
            pred_angle_vectors = _ellipse_angle_vectors(pred_ellipses)
            if self.training:
                ellipse_refinement_aux_outputs = [
                    {
                        "refinement_stage": 0,
                        "pred_ellipses": base_ellipses,
                        "pred_ellipse_vectors": base_angle_vectors,
                    },
                    {
                        "refinement_stage": 1,
                        "pred_ellipses": refinement_stages[0],
                        "pred_ellipse_vectors": _ellipse_angle_vectors(
                            refinement_stages[0]
                        ),
                    },
                ]
        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "pred_ellipses": pred_ellipses,
                "pred_ellipse_vectors": pred_angle_vectors,
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_ellipses": pred_ellipses,
                "pred_ellipse_vectors": pred_angle_vectors,
            }

        if getattr(self, "return_face_query_features", False):
            # The same final decoder query that predicts the FACE bbox/ellipse.
            # Routing metadata only: no new detector parameters or losses.
            out["face_query_features"] = out_hidden[-1]

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta
        if ellipse_refinement_aux_outputs:
            out["ellipse_refinement_aux_outputs"] = (
                ellipse_refinement_aux_outputs
            )
        if self.training and self.depth_supervision_layers:
            depth_outputs = []
            final_layer = int(out_hidden.shape[0])
            for layer in self.depth_supervision_layers:
                index = layer - 1
                if index >= final_layer:
                    raise RuntimeError(
                        f"Requested decoder layer {layer}, but only {final_layer} layers were emitted"
                    )
                if layer == final_layer:
                    layer_ellipses = pred_ellipses
                    layer_vectors = pred_angle_vectors
                else:
                    layer_ellipses, layer_vectors = self._predict_ellipses(
                        out_hidden[index], out_bboxes[index]
                    )
                depth_outputs.append(
                    {
                        "decoder_layer": layer,
                        "pred_logits": out_logits[index],
                        "pred_boxes": out_bboxes[index],
                        "pred_ellipses": layer_ellipses,
                        "pred_ellipse_vectors": layer_vectors,
                    }
                )
            # The name intentionally contains ``aux`` so the base DEIM matcher
            # never mistakes these diagnostic branches for the deployed output.
            out["depth_aux_outputs"] = depth_outputs
        return out


class EllipseDEIMCriterion(DEIMCriterion):
    """Original DEIM losses plus direct ellipse geometry losses."""

    def __init__(
        self,
        *args,
        ellipse_weights: Mapping[str, float] | None = None,
        contour_points: int = 36,
        circle_ratio_threshold: float = 1.05,
        ellipse_aux_layer_weights: Mapping[int, float] | None = None,
        ellipse_refinement_aux_weights: Mapping[int, float] | None = None,
        depth_consistency: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        weights = dict(ellipse_weights or {})
        self.ellipse_weights = {
            "center": float(weights.get("center", 5.0)),
            "axes": float(weights.get("axes", 5.0)),
            "angle": float(weights.get("angle", 1.0)),
            "chamfer": float(weights.get("chamfer", 2.0)),
        }
        self.contour_points = int(contour_points)
        self.circle_ratio_threshold = float(circle_ratio_threshold)
        self.ellipse_aux_layer_weights = self._validate_depth_layer_weights(
            ellipse_aux_layer_weights,
            "ellipse_aux_layer_weights",
        )
        self.ellipse_refinement_aux_weights = {
            int(stage): float(weight)
            for stage, weight in dict(
                ellipse_refinement_aux_weights or {}
            ).items()
        }
        if any(stage < 0 for stage in self.ellipse_refinement_aux_weights):
            raise ValueError("Ellipse refinement auxiliary stage numbers cannot be negative")
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in self.ellipse_refinement_aux_weights.values()
        ):
            raise ValueError(
                "Ellipse refinement auxiliary weights must be finite and non-negative"
            )
        consistency = dict(depth_consistency or {})
        self.consistency_teacher_layer = (
            int(consistency["teacher_layer"])
            if consistency.get("teacher_layer") is not None
            else None
        )
        self.consistency_student_layer_weights = self._validate_depth_layer_weights(
            consistency.get("student_layer_weights"),
            "depth_consistency.student_layer_weights",
        )
        consistency_weights = dict(consistency.get("weights", {}))
        self.depth_consistency_weights = {
            "bbox": float(consistency_weights.get("bbox", 1.0)),
            "score": float(consistency_weights.get("score", 0.25)),
            "ellipse": float(consistency_weights.get("ellipse", 0.5)),
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.depth_consistency_weights.values()
        ):
            raise ValueError("Depth consistency weights must be finite and non-negative")
        if self.consistency_teacher_layer is not None:
            if self.consistency_teacher_layer < 1:
                raise ValueError("Depth consistency teacher_layer must be positive")
            if self.consistency_teacher_layer in self.consistency_student_layer_weights:
                raise ValueError("Depth consistency teacher cannot also be a student")

    @staticmethod
    def _validate_depth_layer_weights(
        weights: Mapping[int, float] | None,
        name: str,
    ) -> dict[int, float]:
        resolved = {int(layer): float(weight) for layer, weight in dict(weights or {}).items()}
        if any(layer < 1 for layer in resolved):
            raise ValueError(f"{name} layer numbers must be positive")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in resolved.values()):
            raise ValueError(f"{name} weights must be finite and non-negative")
        return resolved

    def _ellipse_losses(self, outputs, targets, indices) -> dict[str, torch.Tensor]:
        predictions = outputs["pred_ellipses"]
        zero = predictions.sum() * 0.0
        matched_predictions = []
        matched_targets = []
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            if source_indices.numel() == 0:
                continue
            matched_predictions.append(predictions[batch_index, source_indices.to(predictions.device)])
            matched_targets.append(targets[batch_index]["ellipses"][target_indices.to(predictions.device)])
        if not matched_predictions:
            return {
                "loss_ellipse_center": zero,
                "loss_ellipse_axes": zero,
                "loss_ellipse_angle": zero,
                "loss_ellipse_chamfer": zero,
            }

        predicted = torch.cat(matched_predictions, dim=0)
        target = torch.cat(matched_targets, dim=0).to(dtype=predicted.dtype)
        count = max(1, int(predicted.shape[0]))
        center_loss = F.smooth_l1_loss(predicted[:, :2], target[:, :2], reduction="sum") / count
        axes_loss = F.smooth_l1_loss(predicted[:, 2:4], target[:, 2:4], reduction="sum") / count

        axis_ratio = target[:, 2] / target[:, 3].clamp_min(1.0e-6)
        angle_mask = axis_ratio >= self.circle_ratio_threshold
        angle_distance = 1.0 - torch.cos(2.0 * (predicted[:, 4] - target[:, 4]))
        if angle_mask.any():
            angle_loss = angle_distance[angle_mask].mean()
        else:
            angle_loss = zero

        predicted_points = ellipse_contour_points(predicted, points=self.contour_points)
        target_points = ellipse_contour_points(target, points=self.contour_points)
        pairwise = torch.cdist(predicted_points, target_points, p=2)
        chamfer_loss = 0.5 * (pairwise.min(dim=2).values.mean() + pairwise.min(dim=1).values.mean())
        return {
            "loss_ellipse_center": center_loss * self.ellipse_weights["center"],
            "loss_ellipse_axes": axes_loss * self.ellipse_weights["axes"],
            "loss_ellipse_angle": angle_loss * self.ellipse_weights["angle"],
            "loss_ellipse_chamfer": chamfer_loss * self.ellipse_weights["chamfer"],
        }

    @staticmethod
    def _positive_prediction_indices(indices):
        batch_indices = []
        source_indices = []
        for batch_index, (source, _target) in enumerate(indices):
            if source.numel() == 0:
                continue
            batch_indices.append(torch.full_like(source, batch_index))
            source_indices.append(source)
        if not source_indices:
            return None
        return torch.cat(batch_indices), torch.cat(source_indices)

    @staticmethod
    def _target_values(targets, indices, key: str) -> torch.Tensor | None:
        values = [
            target[key][target_indices]
            for target, (_source_indices, target_indices) in zip(targets, indices)
            if target_indices.numel() > 0
        ]
        return torch.cat(values, dim=0) if values else None

    def _ellipse_error_per_instance(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        center = F.smooth_l1_loss(
            predicted[:, :2], target[:, :2], reduction="none"
        ).sum(dim=-1) * self.ellipse_weights["center"]
        axes = F.smooth_l1_loss(
            predicted[:, 2:4], target[:, 2:4], reduction="none"
        ).sum(dim=-1) * self.ellipse_weights["axes"]
        axis_ratio = target[:, 2] / target[:, 3].clamp_min(1.0e-6)
        angle = 1.0 - torch.cos(2.0 * (predicted[:, 4] - target[:, 4]))
        angle = angle * (axis_ratio >= self.circle_ratio_threshold).to(angle.dtype)
        angle = angle * self.ellipse_weights["angle"]
        predicted_points = ellipse_contour_points(predicted, points=self.contour_points)
        target_points = ellipse_contour_points(target, points=self.contour_points)
        pairwise = torch.cdist(predicted_points, target_points, p=2)
        chamfer = 0.5 * (
            pairwise.min(dim=2).values.mean(dim=1)
            + pairwise.min(dim=1).values.mean(dim=1)
        ) * self.ellipse_weights["chamfer"]
        return center + axes + angle + chamfer

    def _depth_non_regression_losses(
        self,
        teacher: Mapping[str, torch.Tensor],
        student: Mapping[str, torch.Tensor],
        targets,
        indices,
        *,
        student_layer: int,
        layer_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Penalize only target error that exceeds the detached layer-4 teacher."""

        student_boxes = student["pred_boxes"]
        zero = student_boxes.sum() * 0.0
        positive_indices = self._positive_prediction_indices(indices)
        target_boxes = self._target_values(targets, indices, "boxes")
        target_ellipses = self._target_values(targets, indices, "ellipses")

        if positive_indices is None or target_boxes is None or target_ellipses is None:
            bbox_loss = zero
            ellipse_loss = zero
        else:
            batch_index, source_index = positive_indices
            student_matched_boxes = student_boxes[batch_index, source_index]
            teacher_matched_boxes = teacher["pred_boxes"][batch_index, source_index].detach()
            target_boxes = target_boxes.to(
                device=student_matched_boxes.device,
                dtype=student_matched_boxes.dtype,
            )

            def bbox_error(boxes: torch.Tensor) -> torch.Tensor:
                safe_boxes = torch.cat(
                    (boxes[:, :2], boxes[:, 2:].clamp_min(1.0e-6)), dim=-1
                )
                l1 = F.smooth_l1_loss(
                    safe_boxes, target_boxes, reduction="none"
                ).sum(dim=-1)
                giou = 1.0 - torch.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(safe_boxes),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                )
                return l1 + giou

            bbox_loss = F.relu(
                bbox_error(student_matched_boxes)
                - bbox_error(teacher_matched_boxes).detach()
            ).mean()

            student_ellipses = student["pred_ellipses"][batch_index, source_index]
            teacher_ellipses = teacher["pred_ellipses"][batch_index, source_index].detach()
            target_ellipses = target_ellipses.to(
                device=student_ellipses.device,
                dtype=student_ellipses.dtype,
            )
            ellipse_loss = F.relu(
                self._ellipse_error_per_instance(student_ellipses, target_ellipses)
                - self._ellipse_error_per_instance(
                    teacher_ellipses, target_ellipses
                ).detach()
            ).mean()

        student_logits = student["pred_logits"]
        teacher_logits = teacher["pred_logits"].detach()
        score_targets = torch.zeros_like(student_logits)
        positive_mask = torch.zeros(
            student_logits.shape[:2],
            dtype=torch.bool,
            device=student_logits.device,
        )
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            if source_indices.numel() == 0:
                continue
            source_indices = source_indices.to(student_logits.device)
            labels = targets[batch_index]["labels"][target_indices].to(
                device=student_logits.device,
                dtype=torch.long,
            )
            score_targets[batch_index, source_indices, labels] = 1.0
            positive_mask[batch_index, source_indices] = True
        student_score_error = F.binary_cross_entropy_with_logits(
            student_logits, score_targets, reduction="none"
        ).mean(dim=-1)
        teacher_score_error = F.binary_cross_entropy_with_logits(
            teacher_logits, score_targets, reduction="none"
        ).mean(dim=-1)
        score_delta = F.relu(student_score_error - teacher_score_error.detach())
        score_parts = []
        if positive_mask.any():
            score_parts.append(score_delta[positive_mask].mean())
        if (~positive_mask).any():
            score_parts.append(score_delta[~positive_mask].mean())
        score_loss = sum(score_parts) / len(score_parts) if score_parts else zero

        common_weight = float(layer_weight)
        return {
            f"loss_depth_bbox_consistency_l{student_layer}": (
                bbox_loss * self.depth_consistency_weights["bbox"] * common_weight
            ),
            f"loss_depth_score_consistency_l{student_layer}": (
                score_loss * self.depth_consistency_weights["score"] * common_weight
            ),
            f"loss_ellipse_consistency_l{student_layer}": (
                ellipse_loss * self.depth_consistency_weights["ellipse"] * common_weight
            ),
        }

    def forward(self, outputs, targets, epoch=0, **kwargs):
        losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        outputs_without_aux = {key: value for key, value in outputs.items() if "aux" not in key}
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)["indices"]
        losses.update(self._ellipse_losses(outputs, targets, indices))
        refinement_outputs = {
            int(stage_output["refinement_stage"]): stage_output
            for stage_output in outputs.get(
                "ellipse_refinement_aux_outputs", []
            )
        }
        for stage, weight in self.ellipse_refinement_aux_weights.items():
            if stage not in refinement_outputs:
                raise RuntimeError(
                    f"Ellipse refinement auxiliary stage {stage} was not emitted"
                )
            for key, value in self._ellipse_losses(
                refinement_outputs[stage], targets, indices
            ).items():
                losses[f"{key}_aux_refine_s{stage}"] = (
                    value * float(weight)
                )
        depth_outputs = {
            int(layer_output["decoder_layer"]): layer_output
            for layer_output in outputs.get("depth_aux_outputs", [])
        }
        for layer, weight in self.ellipse_aux_layer_weights.items():
            if layer not in depth_outputs:
                raise RuntimeError(f"Ellipse auxiliary decoder layer {layer} was not emitted")
            layer_output = depth_outputs[layer]
            layer_indices = self.matcher(layer_output, targets, epoch=epoch)["indices"]
            for key, value in self._ellipse_losses(
                layer_output, targets, layer_indices
            ).items():
                losses[f"{key}_aux_l{layer}"] = value * float(weight)

        if self.consistency_teacher_layer is not None:
            teacher_layer = self.consistency_teacher_layer
            if teacher_layer not in depth_outputs:
                raise RuntimeError(
                    f"Depth consistency teacher layer {teacher_layer} was not emitted"
                )
            teacher = depth_outputs[teacher_layer]
            teacher_indices = self.matcher(teacher, targets, epoch=epoch)["indices"]
            for student_layer, layer_weight in self.consistency_student_layer_weights.items():
                if student_layer not in depth_outputs:
                    raise RuntimeError(
                        f"Depth consistency student layer {student_layer} was not emitted"
                    )
                losses.update(
                    self._depth_non_regression_losses(
                        teacher,
                        depth_outputs[student_layer],
                        targets,
                        teacher_indices,
                        student_layer=student_layer,
                        layer_weight=layer_weight,
                    )
                )
        return losses


class EllipsePostProcessor(PostProcessor):
    """Gather ellipse predictions with the exact queries selected for boxes."""

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs["pred_logits"]
        normalized_boxes = outputs["pred_boxes"]
        normalized_ellipses = outputs["pred_ellipses"]
        bbox_pred = torch.empty_like(normalized_boxes)
        cx, cy, width, height = normalized_boxes.unbind(-1)
        bbox_pred[..., 0] = cx - width * 0.5
        bbox_pred[..., 1] = cy - height * 0.5
        bbox_pred[..., 2] = cx + width * 0.5
        bbox_pred[..., 3] = cy + height * 0.5
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        ellipses = normalized_ellipses.clone()
        ellipses[..., 0] *= orig_target_sizes[:, 0].unsqueeze(1)
        ellipses[..., 1] *= orig_target_sizes[:, 1].unsqueeze(1)
        max_size = orig_target_sizes.max(dim=1).values.unsqueeze(1)
        ellipses[..., 2] *= max_size
        ellipses[..., 3] *= max_size

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            flat_scores = scores.flatten(1)
            scores, flat_index = torch.topk(
                flat_scores,
                min(self.num_top_queries, int(flat_scores.shape[1])),
                dim=-1,
            )
            labels = mod(flat_index, self.num_classes)
            query_index = flat_index // self.num_classes
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, query_index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=query_index)
            else:
                query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand_as(scores)

        boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).expand(-1, -1, 4))
        ellipses = ellipses.gather(dim=1, index=query_index.unsqueeze(-1).expand(-1, -1, 5))
        ellipse_boxes = ellipse_to_aabb(ellipses)

        if self.deploy_mode and "face_query_features" in outputs:
            raise ValueError("FOR query output requires the eager detector; legacy export is unchanged")
        if self.deploy_mode:
            return labels, boxes, scores, ellipses
        if self.remap_mscoco_category:
            from src.third_party.deimv2.engine.data.dataset import mscoco_label2category

            labels = torch.tensor(
                [mscoco_label2category[int(value.item())] for value in labels.flatten()],
                device=boxes.device,
            ).reshape(labels.shape)
        results = [
            {
                "labels": labels_item,
                "boxes": boxes_item,
                "scores": scores_item,
                "ellipses": ellipses_item,
                "ellipse_boxes": ellipse_boxes_item,
            }
            for labels_item, boxes_item, scores_item, ellipses_item, ellipse_boxes_item in zip(
                labels,
                boxes,
                scores,
                ellipses,
                ellipse_boxes,
            )
        ]
        if "face_query_features" in outputs:
            features = outputs["face_query_features"]
            selected = features.gather(1, query_index.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
            for result, indices, queries in zip(results, query_index, selected, strict=True):
                result["face_query_index"] = indices
                result["face_query_features"] = queries
        return results


class DINOv3DEIMv2EllipseDetector(DINOv3DEIMv2Detector):
    """Additive DEIMv2-L variant whose public result includes learned ellipses."""

    def set_face_query_output(self, enabled: bool = True) -> None:
        """Opt-in FOR transport; preserves weights and the default output contract."""
        self.decoder.return_face_query_features = bool(enabled)

    def _targets_to_official(self, targets, image_size, *, device, dtype):
        image_h, image_w = int(image_size[0]), int(image_size[1])
        box_normalizer = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=dtype).clamp_min(1.0)
        ellipse_normalizer = torch.tensor(
            [image_w, image_h, max(image_h, image_w), max(image_h, image_w), 1.0],
            device=device,
            dtype=dtype,
        )
        converted = []
        for target in targets:
            boxes = torch.as_tensor(target.get("boxes"), dtype=dtype, device=device).reshape(-1, 4)
            ellipses = torch.as_tensor(target.get("ellipses"), dtype=dtype, device=device).reshape(-1, 5)
            raw_labels = torch.as_tensor(target.get("labels"), dtype=torch.long, device=device).reshape(-1)
            if boxes.shape[0] != ellipses.shape[0]:
                raise ValueError(f"Ellipse count must match boxes: ellipses={ellipses.shape[0]}, boxes={boxes.shape[0]}")
            if raw_labels.numel() != boxes.shape[0]:
                raw_labels = torch.ones((boxes.shape[0],), dtype=torch.long, device=device)

            keep_indices = []
            mapped_labels = []
            for index, label in enumerate(raw_labels.tolist()):
                mapped = self.category_to_label.get(int(label))
                if mapped is not None:
                    keep_indices.append(index)
                    mapped_labels.append(mapped)
            if not keep_indices:
                converted.append(
                    {
                        "boxes": boxes.new_zeros((0, 4)),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                        "ellipses": ellipses.new_zeros((0, 5)),
                    }
                )
                continue

            keep = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
            boxes = boxes[keep]
            ellipses = canonicalize_ellipses(ellipses[keep]) / ellipse_normalizer
            labels = torch.as_tensor(mapped_labels, dtype=torch.long, device=device)
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)
            boxes = (boxes / box_normalizer).clamp(0.0, 1.0)
            valid = (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
            converted.append({"boxes": boxes[valid], "labels": labels[valid], "ellipses": ellipses[valid]})
        return converted

    def _format_losses(self, losses):
        formatted = super()._format_losses(losses)
        ellipse_parts = [
            value
            for key, value in losses.items()
            if key.startswith("loss_ellipse_") and isinstance(value, torch.Tensor)
        ]
        if ellipse_parts:
            formatted["loss_ellipse"] = sum(ellipse_parts[1:], ellipse_parts[0])
        consistency_parts = [
            value
            for key, value in losses.items()
            if "_consistency_" in key and isinstance(value, torch.Tensor)
        ]
        if consistency_parts:
            formatted["loss_depth_consistency"] = sum(
                consistency_parts[1:], consistency_parts[0]
            )
        return formatted

    def _clip_and_filter(self, result, image_size):
        image_h, image_w = image_size
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"].long()
        ellipses = result["ellipses"]
        ellipse_boxes = result["ellipse_boxes"]
        context = {key: result[key] for key in ("face_query_index", "face_query_features") if key in result}
        if boxes.numel() > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_h))
            ellipse_boxes[:, 0::2] = ellipse_boxes[:, 0::2].clamp(0, float(image_w))
            ellipse_boxes[:, 1::2] = ellipse_boxes[:, 1::2].clamp(0, float(image_h))
            valid = (
                (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
                & (scores >= self.score_threshold)
                & torch.isfinite(ellipses).all(dim=-1)
                & (ellipses[:, 2] > 0.0)
                & (ellipses[:, 3] > 0.0)
            )
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]
            ellipses = ellipses[valid]
            ellipse_boxes = ellipse_boxes[valid]
            context = {key: value[valid] for key, value in context.items()}
        if self.detections_per_image > 0 and scores.numel() > self.detections_per_image:
            scores, top_indices = torch.topk(scores, self.detections_per_image)
            boxes = boxes[top_indices]
            labels = labels[top_indices]
            ellipses = ellipses[top_indices]
            ellipse_boxes = ellipse_boxes[top_indices]
            context = {key: value[top_indices] for key, value in context.items()}
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "ellipses": ellipses,
            "ellipse_boxes": ellipse_boxes,
            **context,
        }


def _ellipse_criterion_from_base(base: DEIMCriterion, config: Mapping[str, Any]) -> EllipseDEIMCriterion:
    loss_cfg = dict(config.get("loss", {}))
    ellipse_cfg = dict(loss_cfg.get("ellipse", {}))
    depth_cfg = dict(loss_cfg.get("depth_supervision", {}))
    return EllipseDEIMCriterion(
        matcher=base.matcher,
        weight_dict=base.weight_dict,
        losses=base.losses,
        alpha=base.alpha,
        gamma=base.gamma,
        num_classes=base.num_classes,
        reg_max=base.reg_max,
        boxes_weight_format=base.boxes_weight_format,
        share_matched_indices=base.share_matched_indices,
        mal_alpha=base.mal_alpha,
        use_uni_set=base.use_uni_set,
        aux_layer_weights=base.aux_layer_weights,
        dn_layer_weights=base.dn_layer_weights,
        ellipse_weights=ellipse_cfg.get("weights"),
        contour_points=int(ellipse_cfg.get("contour_points", 36)),
        circle_ratio_threshold=float(ellipse_cfg.get("circle_ratio_threshold", 1.05)),
        ellipse_aux_layer_weights=depth_cfg.get("ellipse_aux_layer_weights"),
        ellipse_refinement_aux_weights=ellipse_cfg.get(
            "refinement_aux_weights"
        ),
        depth_consistency=depth_cfg.get("consistency"),
    )


def build_dinov3_deimv2_ellipse_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DINOv3DEIMv2EllipseDetector:
    if backbone_module is not None:
        raise ValueError("DINOv3 DEIMv2 ellipse path expects configuration, not backbone_module injection.")
    cfg = copy.deepcopy(dict(config or {}))
    model_cfg = cfg.setdefault("model", {})
    init_cfg = dict(model_cfg.pop("init_checkpoint", cfg.pop("init_checkpoint", {})) or {})
    legacy_init = model_cfg.pop("pretrained_checkpoint", None) or cfg.pop("pretrained_checkpoint", None)
    if legacy_init and not init_cfg:
        init_cfg = {"path": legacy_init}
    model_cfg["type"] = "dinov3_deimv2"
    base = build_dinov3_deimv2_detector(cfg)

    ellipse_decoder = base.decoder
    ellipse_decoder.__class__ = EllipseDEIMTransformer
    ellipse_cfg = dict(dict(config or {}).get("model", {}).get("ellipse_head", {}))
    ellipse_decoder.initialize_ellipse_head(
        hidden_dim=int(getattr(ellipse_decoder, "hidden_dim")),
        prior_major=float(ellipse_cfg.get("prior_major", 0.05)),
        prior_axis_ratio=float(ellipse_cfg.get("prior_axis_ratio", 0.70)),
        prior_angle_degrees=float(ellipse_cfg.get("prior_angle_degrees", 90.0)),
        refinement=ellipse_cfg.get("refinement"),
    )
    loss_cfg = dict(dict(config or {}).get("loss", {}))
    depth_cfg = dict(loss_cfg.get("depth_supervision", {}))
    consistency_cfg = dict(depth_cfg.get("consistency", {}))
    supervised_layers = {
        int(layer)
        for layer in dict(depth_cfg.get("ellipse_aux_layer_weights", {}))
    }
    if consistency_cfg.get("teacher_layer") is not None:
        supervised_layers.add(int(consistency_cfg["teacher_layer"]))
    supervised_layers.update(
        int(layer)
        for layer in dict(consistency_cfg.get("student_layer_weights", {}))
    )
    ellipse_decoder.configure_depth_supervision(tuple(supervised_layers))

    inference_cfg = dict(dict(config or {}).get("inference", {}))
    postprocessor = EllipsePostProcessor(
        num_classes=base.postprocessor.num_classes,
        use_focal_loss=base.postprocessor.use_focal_loss,
        num_top_queries=base.postprocessor.num_top_queries,
        remap_mscoco_category=base.postprocessor.remap_mscoco_category,
    )
    detector = DINOv3DEIMv2EllipseDetector(
        backbone=base.backbone,
        encoder=base.encoder,
        decoder=ellipse_decoder,
        criterion=_ellipse_criterion_from_base(base.criterion, dict(config or {})),
        postprocessor=postprocessor,
        category_ids=base.category_ids,
        score_threshold=float(inference_cfg.get("score_thr", inference_cfg.get("score_threshold", 0.0))),
        detections_per_image=int(inference_cfg.get("detections_per_image", 100)),
    )
    if init_cfg:
        init_cfg.setdefault("prefer_ema", True)
        _load_pretrained_checkpoint(detector, init_cfg)
    return detector


__all__ = [
    "DINOv3DEIMv2EllipseDetector",
    "EllipseDEIMCriterion",
    "EllipseDEIMTransformer",
    "EllipseLocalQueryRefinement",
    "EllipseLocalQueryRefinementStage",
    "EllipsePostProcessor",
    "build_dinov3_deimv2_ellipse_detector",
    "canonicalize_ellipses",
    "ellipse_contour_points",
    "ellipse_to_aabb",
    "scale_ellipses_anisotropic",
]
