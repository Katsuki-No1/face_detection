from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Sequence

import torch
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor, nn

from src.models.dinov3_deimv2_ellipse import (
    canonicalize_ellipses,
    ellipse_contour_points,
)
from src.models.head_codino.codino import CompactCoDINO

from .o2m_head import PositiveCoordinates


class CoDINOQueryHead(CompactCoDINO):
    """Configurable Co-DINO O2O branch with cooperative O2M queries.

    The maintained MMDetection 3.x DINO encoder, decoder, attention, CDN,
    Hungarian assigner, QFL, L1 and GIoU implementations provide the core
    query path. ATSS assignments are converted into a second decoder pass,
    matching the cooperative training direction of the original Co-DETR code.
    Only the O2O predictions are exposed at inference.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dims: int = 256,
        num_feature_levels: int = 5,
        num_queries: int = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        feedforward_channels: int = 2048,
        num_denoising_queries: int = 100,
        stable_self_attention: bool = False,
        cooperative_loss_weight: float = 1.0,
        ellipse: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_feature_levels=num_feature_levels,
            num_queries=num_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            feedforward_channels=feedforward_channels,
            num_denoising_queries=num_denoising_queries,
            stable_self_attention=stable_self_attention,
            enforce_compact_parameter_contract=False,
        )
        self.cooperative_loss_weight = float(cooperative_loss_weight)
        if self.cooperative_loss_weight < 0.0:
            raise ValueError("cooperative_loss_weight must be non-negative")
        self.auxiliary_query_embedding = nn.Sequential(
            nn.Linear(4, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        for module in self.auxiliary_query_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        ellipse_cfg = dict(ellipse or {})
        self.ellipse_enabled = bool(ellipse_cfg.get("enabled", False))
        self.ellipse_contour_points = int(ellipse_cfg.get("contour_points", 36))
        self.ellipse_circle_ratio_threshold = float(
            ellipse_cfg.get("circle_ratio_threshold", 1.05)
        )
        ellipse_weights = dict(ellipse_cfg.get("weights", {}) or {})
        self.ellipse_weights = {
            "center": float(ellipse_weights.get("center", 5.0)),
            "axes": float(ellipse_weights.get("axes", 5.0)),
            "angle": float(ellipse_weights.get("angle", 1.0)),
            "chamfer": float(ellipse_weights.get("chamfer", 2.0)),
        }
        self.ellipse_head: nn.Module | None = None
        if self.ellipse_enabled:
            self.ellipse_head = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.SiLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.SiLU(),
                nn.Linear(self.embed_dims, 6),
            )
            output = self.ellipse_head[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            prior_major = max(float(ellipse_cfg.get("prior_major", 0.05)), 1.0e-4)
            prior_ratio = min(
                max(float(ellipse_cfg.get("prior_axis_ratio", 0.70)), 1.0e-4),
                1.0 - 1.0e-4,
            )
            prior_angle = math.radians(
                float(ellipse_cfg.get("prior_angle_degrees", 90.0))
            )
            with torch.no_grad():
                output.bias[2] = math.log(math.expm1(prior_major))
                output.bias[3] = math.log(prior_ratio / (1.0 - prior_ratio))
                output.bias[4] = math.cos(2.0 * prior_angle)
                output.bias[5] = math.sin(2.0 * prior_angle)

    def _predict_ellipses(self, hidden: Tensor, boxes: Tensor) -> Tensor:
        if self.ellipse_head is None:
            raise RuntimeError("Ellipse prediction requested while ellipse_head is disabled")
        raw = self.ellipse_head(hidden)
        center = boxes[..., :2].detach() + raw[..., :2]
        major = torch.nn.functional.softplus(raw[..., 2]).clamp_min(1.0e-6)
        minor = major * torch.sigmoid(raw[..., 3]).clamp(1.0e-4, 1.0 - 1.0e-4)
        angle_vector = torch.nn.functional.normalize(raw[..., 4:6], dim=-1, eps=1.0e-6)
        theta = 0.5 * torch.atan2(angle_vector[..., 1], angle_vector[..., 0])
        return torch.stack(
            (center[..., 0], center[..., 1], major, minor, theta), dim=-1
        )

    def _decode_positive_coordinates(
        self,
        raw_outputs: dict[str, Any],
        positives: PositiveCoordinates,
    ) -> dict[str, Tensor]:
        anchors = positives.anchors.clamp(1.0e-4, 1.0 - 1.0e-4)
        query = self.auxiliary_query_embedding(anchors)
        hidden_states, references = self.decoder(
            query=query,
            value=raw_outputs["_memory"],
            key_padding_mask=raw_outputs["_memory_mask"],
            self_attn_mask=None,
            reference_points=anchors,
            spatial_shapes=raw_outputs["_spatial_shapes"],
            level_start_index=raw_outputs["_level_start_index"],
            valid_ratios=raw_outputs["_valid_ratios"],
            reg_branches=self.bbox_head.reg_branches,
        )
        logits, boxes = self.bbox_head(hidden_states, list(references))
        return {"logits": logits, "boxes": boxes}

    def forward(
        self,
        features: Sequence[Tensor],
        masks: Sequence[Tensor],
        *,
        batch_data_samples: Sequence[DetDataSample] | None = None,
        positive_coordinates: PositiveCoordinates | None = None,
    ) -> dict[str, Any]:
        outputs = super().forward(
            features,
            masks,
            batch_data_samples=batch_data_samples,
        )
        if self.ellipse_enabled:
            # CDN queries are prepended during training. The final num_queries
            # hidden states are the deployed one-to-one matching queries.
            matching_hidden = outputs["_hidden_states"][-1, :, -self.num_queries :]
            outputs["pred_ellipses"] = self._predict_ellipses(
                matching_hidden,
                outputs["pred_boxes"],
            )
        if positive_coordinates is not None:
            outputs["cooperative_outputs"] = self._decode_positive_coordinates(
                outputs,
                positive_coordinates,
            )
            outputs["_positive_coordinates"] = positive_coordinates
        return outputs

    def _ellipse_loss(
        self,
        raw_outputs: dict[str, Any],
        batch_data_samples: Sequence[DetDataSample],
    ) -> dict[str, Tensor]:
        predictions = raw_outputs["pred_ellipses"].float()
        logits = raw_outputs["pred_logits"].float()
        boxes = raw_outputs["pred_boxes"].float()
        zero = predictions.sum() * 0.0
        matched_predictions: list[Tensor] = []
        matched_targets: list[Tensor] = []
        for batch_index, sample in enumerate(batch_data_samples):
            gt = sample.gt_instances
            if not hasattr(gt, "ellipses"):
                raise ValueError(
                    "Ellipse-enabled Co-DINO requires an ellipse for every FACE target"
                )
            image_height, image_width = sample.metainfo["img_shape"][:2]
            factor = boxes.new_tensor(
                [image_width, image_height, image_width, image_height]
            )
            predicted_boxes = bbox_cxcywh_to_xyxy(boxes[batch_index]) * factor
            assignment = self.bbox_head.assigner.assign(
                pred_instances=InstanceData(
                    scores=logits[batch_index].detach(),
                    bboxes=predicted_boxes.detach(),
                ),
                gt_instances=gt,
                img_meta=sample.metainfo,
            )
            positive = torch.nonzero(
                assignment.gt_inds > 0, as_tuple=False
            ).squeeze(-1)
            if positive.numel() == 0:
                continue
            target_indices = assignment.gt_inds[positive] - 1
            maximum_side = max(image_width, image_height)
            normalizer = predictions.new_tensor(
                [image_width, image_height, maximum_side, maximum_side, 1.0]
            )
            target_ellipses = canonicalize_ellipses(
                gt.ellipses[target_indices].to(dtype=predictions.dtype)
            ) / normalizer
            matched_predictions.append(predictions[batch_index, positive])
            matched_targets.append(target_ellipses)

        if not matched_predictions:
            return {
                "loss_ellipse_center": zero,
                "loss_ellipse_axes": zero,
                "loss_ellipse_angle": zero,
                "loss_ellipse_chamfer": zero,
            }
        predicted = torch.cat(matched_predictions, dim=0)
        target = torch.cat(matched_targets, dim=0)
        count = max(1, int(predicted.shape[0]))
        center_loss = torch.nn.functional.smooth_l1_loss(
            predicted[:, :2], target[:, :2], reduction="sum"
        ) / count
        axes_loss = torch.nn.functional.smooth_l1_loss(
            predicted[:, 2:4], target[:, 2:4], reduction="sum"
        ) / count
        target_axis_ratio = target[:, 2] / target[:, 3].clamp_min(1.0e-6)
        angle_mask = target_axis_ratio >= self.ellipse_circle_ratio_threshold
        angle_distance = 1.0 - torch.cos(
            2.0 * (predicted[:, 4] - target[:, 4])
        )
        angle_loss = angle_distance[angle_mask].mean() if angle_mask.any() else zero
        predicted_points = ellipse_contour_points(
            predicted, points=self.ellipse_contour_points
        )
        target_points = ellipse_contour_points(
            target, points=self.ellipse_contour_points
        )
        pairwise = torch.cdist(predicted_points, target_points, p=2)
        chamfer_loss = 0.5 * (
            pairwise.min(dim=2).values.mean()
            + pairwise.min(dim=1).values.mean()
        )
        return {
            "loss_ellipse_center": center_loss * self.ellipse_weights["center"],
            "loss_ellipse_axes": axes_loss * self.ellipse_weights["axes"],
            "loss_ellipse_angle": angle_loss * self.ellipse_weights["angle"],
            "loss_ellipse_chamfer": chamfer_loss * self.ellipse_weights["chamfer"],
        }

    def loss(
        self,
        raw_outputs: dict[str, Any],
        batch_data_samples: Sequence[Any],
    ) -> dict[str, Tensor]:
        losses = super().loss(raw_outputs, batch_data_samples)
        if self.ellipse_enabled:
            losses.pop("loss_total", None)
            losses.update(self._ellipse_loss(raw_outputs, batch_data_samples))
            losses["loss_total"] = sum(losses.values())
        return losses

    def cooperative_loss(
        self,
        raw_outputs: dict[str, Any],
        batch_data_samples: Sequence[DetDataSample],
    ) -> dict[str, Tensor]:
        if "cooperative_outputs" not in raw_outputs:
            return {}
        positives: PositiveCoordinates = raw_outputs["_positive_coordinates"]
        cooperative = raw_outputs["cooperative_outputs"]
        all_logits = cooperative["logits"].float()
        all_boxes = cooperative["boxes"].float()
        targets = positives.targets.float()
        valid = positives.valid
        labels = positives.labels
        valid_count = max(int(valid.sum().item()), 1)
        label_weights = valid.to(dtype=all_logits.dtype)
        bbox_weights = valid.unsqueeze(-1).expand_as(targets).to(dtype=all_boxes.dtype)

        factors = []
        for sample in batch_data_samples:
            height, width = sample.metainfo["img_shape"][:2]
            factors.append(all_boxes.new_tensor([width, height, width, height]))
        factor = torch.stack(factors, dim=0).unsqueeze(1)

        losses: dict[str, Tensor] = {}
        last_layer = all_logits.shape[0] - 1
        for layer, (logits, boxes) in enumerate(
            zip(all_logits, all_boxes, strict=True)
        ):
            flat_logits = logits.reshape(-1, self.bbox_head.cls_out_channels)
            flat_labels = labels.reshape(-1)
            flat_weights = label_weights.reshape(-1)
            quality = flat_weights.new_zeros(flat_labels.shape)
            flat_valid = valid.reshape(-1)
            if flat_valid.any():
                quality[flat_valid] = bbox_overlaps(
                    bbox_cxcywh_to_xyxy(boxes.reshape(-1, 4)[flat_valid].detach()),
                    bbox_cxcywh_to_xyxy(targets.reshape(-1, 4)[flat_valid]),
                    is_aligned=True,
                )
            loss_cls = self.bbox_head.loss_cls(
                flat_logits,
                (flat_labels, quality),
                flat_weights,
                avg_factor=valid_count,
            )
            predicted_xyxy = bbox_cxcywh_to_xyxy(boxes) * factor
            target_xyxy = bbox_cxcywh_to_xyxy(targets) * factor
            loss_bbox = self.bbox_head.loss_bbox(
                boxes,
                targets,
                bbox_weights,
                avg_factor=valid_count,
            )
            loss_giou = self.bbox_head.loss_iou(
                predicted_xyxy,
                target_xyxy,
                bbox_weights,
                avg_factor=valid_count,
            )
            prefix = "aux_o2m" if layer == last_layer else f"d{layer}.aux_o2m"
            losses[f"{prefix}.loss_cls"] = loss_cls * self.cooperative_loss_weight
            losses[f"{prefix}.loss_bbox"] = loss_bbox * self.cooperative_loss_weight
            losses[f"{prefix}.loss_giou"] = loss_giou * self.cooperative_loss_weight
        return losses


__all__ = ["CoDINOQueryHead"]
