from __future__ import annotations

import copy
import math
from collections.abc import Mapping
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

        for index, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if index >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

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
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
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

    def _predict_ellipses(self, hidden: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.ellipse_head(hidden)
        center = boxes[..., :2].detach() + raw[..., :2]
        major = F.softplus(raw[..., 2]).clamp_min(1.0e-6)
        minor = major * torch.sigmoid(raw[..., 3]).clamp(1.0e-4, 1.0 - 1.0e-4)
        angle_vector = F.normalize(raw[..., 4:6], dim=-1, eps=1.0e-6)
        theta = 0.5 * torch.atan2(angle_vector[..., 1], angle_vector[..., 0])
        ellipses = torch.stack((center[..., 0], center[..., 1], major, minor, theta), dim=-1)
        return ellipses, angle_vector

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

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
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            _dn_hidden, out_hidden = torch.split(out_hidden, dn_meta["dn_num_split"], dim=2)

        pred_ellipses, pred_angle_vectors = self._predict_ellipses(out_hidden[-1], out_bboxes[-1])
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
        return out


class EllipseDEIMCriterion(DEIMCriterion):
    """Original DEIM losses plus direct ellipse geometry losses."""

    def __init__(
        self,
        *args,
        ellipse_weights: Mapping[str, float] | None = None,
        contour_points: int = 36,
        circle_ratio_threshold: float = 1.05,
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

    def forward(self, outputs, targets, epoch=0, **kwargs):
        losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        outputs_without_aux = {key: value for key, value in outputs.items() if "aux" not in key}
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)["indices"]
        losses.update(self._ellipse_losses(outputs, targets, indices))
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

        if self.deploy_mode:
            return labels, boxes, scores, ellipses
        if self.remap_mscoco_category:
            from src.third_party.deimv2.engine.data.dataset import mscoco_label2category

            labels = torch.tensor(
                [mscoco_label2category[int(value.item())] for value in labels.flatten()],
                device=boxes.device,
            ).reshape(labels.shape)
        return [
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


class DINOv3DEIMv2EllipseDetector(DINOv3DEIMv2Detector):
    """Additive DEIMv2-L variant whose public result includes learned ellipses."""

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
        return formatted

    def _clip_and_filter(self, result, image_size):
        image_h, image_w = image_size
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"].long()
        ellipses = result["ellipses"]
        ellipse_boxes = result["ellipse_boxes"]
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
        if self.detections_per_image > 0 and scores.numel() > self.detections_per_image:
            scores, top_indices = torch.topk(scores, self.detections_per_image)
            boxes = boxes[top_indices]
            labels = labels[top_indices]
            ellipses = ellipses[top_indices]
            ellipse_boxes = ellipse_boxes[top_indices]
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "ellipses": ellipses,
            "ellipse_boxes": ellipse_boxes,
        }


def _ellipse_criterion_from_base(base: DEIMCriterion, config: Mapping[str, Any]) -> EllipseDEIMCriterion:
    loss_cfg = dict(config.get("loss", {}))
    ellipse_cfg = dict(loss_cfg.get("ellipse", {}))
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
        ellipse_weights=ellipse_cfg.get("weights"),
        contour_points=int(ellipse_cfg.get("contour_points", 36)),
        circle_ratio_threshold=float(ellipse_cfg.get("circle_ratio_threshold", 1.05)),
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
    )

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
    "EllipsePostProcessor",
    "build_dinov3_deimv2_ellipse_detector",
    "canonicalize_ellipses",
    "ellipse_contour_points",
    "ellipse_to_aabb",
    "scale_ellipses_anisotropic",
]
