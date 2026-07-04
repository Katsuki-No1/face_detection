from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .dino_backbone import DinoBackbone, build_dino_backbone
from .dino_fpn import ConvNormAct
from .postprocess import nms


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = (area1[:, None] + area2[None, :] - inter).clamp_min(1e-6)
    iou = inter / union

    enclosing_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp_min(0.0)
    enclosing_area = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp_min(1e-6)
    return iou - (enclosing_area - union) / enclosing_area


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = (area1[:, None] + area2[None, :] - inter).clamp_min(1e-6)
    return inter / union


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def target_boxes_to_normalized_cxcywh(target: dict[str, Any], image_size: tuple[int, int]) -> torch.Tensor:
    boxes = torch.as_tensor(target.get("boxes"), dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.reshape(-1, 4)
    size = target.get("size", image_size)
    image_h, image_w = int(size[0]), int(size[1])
    normalizer = boxes.new_tensor([image_w, image_h, image_w, image_h]).clamp_min(1.0)
    return box_xyxy_to_cxcywh(boxes) / normalizer


def target_labels_zero_based(target: dict[str, Any], count: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = target.get("labels")
    if labels is None:
        return torch.zeros((count,), dtype=torch.long, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    if labels_tensor.numel() != count:
        return torch.zeros((count,), dtype=torch.long, device=device)
    labels_tensor = labels_tensor - 1
    return labels_tensor.clamp(0, max(0, num_classes - 1))


def _sine_position_encoding(height: int, width: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    num_bands = max(1, dim // 4)
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(num_bands, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1, num_bands)))
    x_embed = xx.flatten()[:, None] * omega[None, :]
    y_embed = yy.flatten()[:, None] * omega[None, :]
    pos = torch.cat((x_embed.sin(), x_embed.cos(), y_embed.sin(), y_embed.cos()), dim=1)
    if pos.shape[1] < dim:
        pos = F.pad(pos, (0, dim - pos.shape[1]))
    return pos[:, :dim].unsqueeze(0)


class SpatialTuningAdapter(nn.Module):
    """Project one DINO patch feature map into lightweight P3/P4/P5 features.

    Input shape is ``[B, C, H, W]``. Output features are ordered as P3, P4 and
    P5 by default. This is a DINO single-scale to detector multi-scale adapter,
    not a full RTMDet/YOLO FPN copy.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        *,
        out_scales: tuple[str, ...] = ("p3", "p4", "p5"),
        norm: str | None = "gn",
        activation: str | None = "silu",
        refine_layers: int = 0,
        cross_scale_fusion: bool = False,
    ) -> None:
        super().__init__()
        allowed = {"p3", "p4", "p5"}
        if not out_scales or any(scale not in allowed for scale in out_scales):
            raise ValueError(f"out_scales must be a non-empty subset of {sorted(allowed)}, got {out_scales}")
        self.out_scales = tuple(out_scales)
        self.cross_scale_fusion = bool(cross_scale_fusion)
        self.proj = ConvNormAct(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation)
        self.p4 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, norm=norm, activation=activation)
        self.p3 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, norm=norm, activation=activation)
        self.p5 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=2, norm=norm, activation=activation)
        self.refine = nn.ModuleDict()
        for scale in ("p3", "p4", "p5"):
            layers = [
                ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, norm=norm, activation=activation)
                for _ in range(int(refine_layers))
            ]
            self.refine[scale] = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"SpatialTuningAdapter expects [B, C, H, W], got {tuple(feature.shape)}")
        base = self.proj(feature)
        p4 = self.p4(base)
        features = {
            "p3": self.p3(F.interpolate(p4, scale_factor=2.0, mode="nearest")),
            "p4": p4,
            "p5": self.p5(p4),
        }
        if self.cross_scale_fusion:
            features["p4"] = features["p4"] + F.interpolate(features["p5"], size=features["p4"].shape[-2:], mode="nearest")
            features["p4"] = features["p4"] + F.interpolate(features["p3"], size=features["p4"].shape[-2:], mode="nearest")
            features["p3"] = features["p3"] + F.interpolate(features["p4"], size=features["p3"].shape[-2:], mode="nearest")
            features["p5"] = features["p5"] + F.interpolate(features["p4"], size=features["p5"].shape[-2:], mode="nearest")
        features = {name: self.refine[name](value) for name, value in features.items()}
        return [features[name] for name in self.out_scales]


@dataclass(frozen=True)
class EncoderOutput:
    memory: torch.Tensor
    spatial_shapes: tuple[tuple[int, int], ...]


class HybridEncoder(nn.Module):
    """Small RT-DETR-style encoder facade for multi-scale DINO features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        *,
        encoder_type: str = "identity",
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder_type = str(encoder_type).lower()
        self.level_embed = nn.Parameter(torch.zeros(8, self.hidden_dim))
        if self.encoder_type in {"identity", "none"} or int(num_layers) <= 0:
            self.encoder = None
        elif self.encoder_type in {"lightweight_transformer", "transformer"}:
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=self.hidden_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        else:
            raise ValueError(f"Unsupported encoder.type: {encoder_type}")

    def forward(self, features: list[torch.Tensor]) -> EncoderOutput:
        tokens = []
        spatial_shapes: list[tuple[int, int]] = []
        for level, feature in enumerate(features):
            if feature.ndim != 4:
                raise ValueError(f"encoder feature {level} must be [B, C, H, W], got {tuple(feature.shape)}")
            batch, channels, height, width = feature.shape
            if channels != self.hidden_dim:
                raise ValueError(f"encoder feature {level} has {channels} channels, expected {self.hidden_dim}")
            spatial_shapes.append((int(height), int(width)))
            flattened = feature.flatten(2).transpose(1, 2).contiguous()
            pos = _sine_position_encoding(height, width, self.hidden_dim, device=feature.device, dtype=feature.dtype)
            level_embed = self.level_embed[level].view(1, 1, -1)
            tokens.append(flattened + pos + level_embed)
        memory = torch.cat(tokens, dim=1) if tokens else torch.empty(0)
        if self.encoder is not None:
            memory = self.encoder(memory)
        return EncoderOutput(memory=memory, spatial_shapes=tuple(spatial_shapes))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        for index in range(num_layers):
            in_dim = input_dim if index == 0 else hidden_dim
            out_dim = output_dim if index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


class DETRStyleDecoder(nn.Module):
    """Learned-query DETR-style decoder for bbox detection."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_layers: int = 3,
        num_heads: int = 8,
        num_classes: int = 1,
        dropout: float = 0.0,
        refine_boxes: bool = False,
        box_refine_steps: int = 0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.refine_boxes = bool(refine_boxes)
        self.bbox_refiners = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(int(box_refine_steps))]
        )

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = int(memory.shape[0])
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tgt = torch.zeros_like(query)
        hidden = self.decoder(tgt + query, memory)
        boxes = self.bbox_embed(hidden).sigmoid()
        if self.refine_boxes:
            for refiner in self.bbox_refiners:
                boxes = (inverse_sigmoid(boxes) + refiner(hidden)).sigmoid()
        return {"pred_logits": self.class_embed(hidden), "pred_boxes": boxes}


class DINODeimStyleMatcher:
    """Hungarian matcher for normalized DETR-style boxes."""

    def __init__(
        self,
        *,
        cls_cost: float = 2.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        dense_o2o_enabled: bool = False,
        extra_matches_per_gt: int = 0,
        min_iou: float = 0.0,
        max_cost: float | None = None,
    ) -> None:
        self.cls_cost = float(cls_cost)
        self.bbox_cost = float(bbox_cost)
        self.giou_cost = float(giou_cost)
        self.dense_o2o_enabled = bool(dense_o2o_enabled)
        self.extra_matches_per_gt = int(extra_matches_per_gt)
        self.min_iou = float(min_iou)
        self.max_cost = float(max_cost) if max_cost is not None else None

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        targets: list[dict[str, Any]] | None,
        image_size: tuple[int, int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        logits = predictions["pred_logits"]
        boxes = predictions["pred_boxes"]
        batch_size, num_queries, num_classes = logits.shape
        targets = targets or [None for _ in range(batch_size)]  # type: ignore[list-item]
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        probabilities = logits.sigmoid()
        for batch_index in range(batch_size):
            target = targets[batch_index] if batch_index < len(targets) and targets[batch_index] is not None else {}
            target_boxes = target_boxes_to_normalized_cxcywh(target, image_size).to(device=boxes.device, dtype=boxes.dtype)
            target_count = int(target_boxes.shape[0])
            if target_count == 0:
                empty = torch.empty((0,), dtype=torch.long, device=boxes.device)
                matches.append((empty, empty))
                continue
            labels = target_labels_zero_based(target, target_count, num_classes, boxes.device)
            cls_cost = -probabilities[batch_index][:, labels]
            bbox_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(boxes[batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            pred_xyxy = box_cxcywh_to_xyxy(boxes[batch_index])
            target_xyxy = box_cxcywh_to_xyxy(target_boxes)
            cost = self.cls_cost * cls_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
            src, dst = self._linear_sum_assignment(cost)
            if self.dense_o2o_enabled and self.extra_matches_per_gt > 0:
                src, dst = self._add_dense_o2o_matches(
                    cost=cost,
                    ious=box_iou(pred_xyxy, target_xyxy),
                    base_src=src,
                    base_dst=dst,
                )
            matches.append((src.to(boxes.device), dst.to(boxes.device)))
        return matches

    def _add_dense_o2o_matches(
        self,
        *,
        cost: torch.Tensor,
        ious: torch.Tensor,
        base_src: torch.Tensor,
        base_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_values = [int(value) for value in base_src.tolist()]
        dst_values = [int(value) for value in base_dst.tolist()]
        used_src = set(src_values)
        for target_index in range(cost.shape[1]):
            added = 0
            for query_index in torch.argsort(cost[:, target_index]).tolist():
                query_index = int(query_index)
                if query_index in used_src:
                    continue
                if self.max_cost is not None and float(cost[query_index, target_index]) > self.max_cost:
                    continue
                if self.min_iou > 0.0 and float(ious[query_index, target_index]) < self.min_iou:
                    continue
                src_values.append(query_index)
                dst_values.append(int(target_index))
                used_src.add(query_index)
                added += 1
                if added >= self.extra_matches_per_gt:
                    break
        return torch.as_tensor(src_values, dtype=torch.long), torch.as_tensor(dst_values, dtype=torch.long)

    @staticmethod
    def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)
        except Exception:
            # Fallback is greedy and only used when SciPy is unavailable.
            remaining_rows = set(range(cost.shape[0]))
            remaining_cols = set(range(cost.shape[1]))
            pairs: list[tuple[int, int]] = []
            detached = cost.detach().cpu()
            while remaining_rows and remaining_cols:
                best_pair = min(
                    ((float(detached[row, col]), row, col) for row in remaining_rows for col in remaining_cols),
                    key=lambda item: item[0],
                )
                _, row, col = best_pair
                pairs.append((row, col))
                remaining_rows.remove(row)
                remaining_cols.remove(col)
            if not pairs:
                return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
            rows, cols = zip(*pairs)
            return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


class DINODeimStyleCriterion(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 1,
        matcher: DINODeimStyleMatcher | None = None,
        cls_weight: float = 2.0,
        bbox_l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        no_object_weight: float = 0.1,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mal_enabled: bool = False,
        mal_quality_power: float = 1.0,
        mal_min_quality: float = 0.25,
        mal_cls_quality_target: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher or DINODeimStyleMatcher(cls_cost=cls_weight, bbox_cost=bbox_l1_weight, giou_cost=giou_weight)
        self.cls_weight = float(cls_weight)
        self.bbox_l1_weight = float(bbox_l1_weight)
        self.giou_weight = float(giou_weight)
        self.no_object_weight = float(no_object_weight)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.mal_enabled = bool(mal_enabled)
        self.mal_quality_power = float(mal_quality_power)
        self.mal_min_quality = float(mal_min_quality)
        self.mal_cls_quality_target = bool(mal_cls_quality_target)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: list[dict[str, Any]] | None,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]
        batch_size, num_queries, num_classes = logits.shape
        targets = targets or [None for _ in range(batch_size)]  # type: ignore[list-item]
        matches = self.matcher(predictions, targets, image_size)

        target_classes = torch.zeros_like(logits)
        matched_pred_boxes = []
        matched_target_boxes = []
        matched_weights = []
        matched_qualities = []
        for batch_index, (src_indices, target_indices) in enumerate(matches):
            if src_indices.numel() == 0:
                continue
            target = targets[batch_index]
            target_boxes = target_boxes_to_normalized_cxcywh(target, image_size).to(device=pred_boxes.device, dtype=pred_boxes.dtype)
            labels = target_labels_zero_based(target, int(target_boxes.shape[0]), num_classes, pred_boxes.device)
            src_boxes = pred_boxes[batch_index, src_indices]
            dst_boxes = target_boxes[target_indices]
            quality = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(dst_boxes))).detach().clamp(0.0, 1.0)
            if self.mal_enabled:
                cls_target = quality.clamp_min(self.mal_min_quality) if self.mal_cls_quality_target else torch.ones_like(quality)
                weight = quality.clamp_min(self.mal_min_quality).pow(self.mal_quality_power)
            else:
                cls_target = torch.ones_like(quality)
                weight = torch.ones_like(quality)
            target_classes[batch_index, src_indices, labels[target_indices]] = cls_target
            matched_pred_boxes.append(src_boxes)
            matched_target_boxes.append(dst_boxes)
            matched_weights.append(weight)
            matched_qualities.append(quality)

        positive_count = sum(int(src.numel()) for src, _ in matches)
        normalizer = logits.new_tensor(float(max(1, positive_count)))
        if self.use_focal_loss:
            prob = logits.sigmoid()
            ce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
            p_t = prob * target_classes + (1.0 - prob) * (1.0 - target_classes)
            alpha_t = self.focal_alpha * target_classes + (1.0 - self.focal_alpha) * (1.0 - target_classes)
            negative_weight = torch.where(target_classes > 0.0, torch.ones_like(target_classes), torch.full_like(target_classes, self.no_object_weight))
            loss_cls = (alpha_t * negative_weight * ce * ((1.0 - p_t) ** self.focal_gamma)).sum() / normalizer
        else:
            negative_weight = torch.where(target_classes > 0.0, torch.ones_like(target_classes), torch.full_like(target_classes, self.no_object_weight))
            loss_cls = (F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none") * negative_weight).sum() / normalizer

        if matched_pred_boxes:
            src_boxes = torch.cat(matched_pred_boxes, dim=0)
            dst_boxes = torch.cat(matched_target_boxes, dim=0)
            weights = torch.cat(matched_weights, dim=0).to(device=src_boxes.device, dtype=src_boxes.dtype)
            bbox_normalizer = weights.sum().clamp_min(1.0)
            l1_per_pair = (src_boxes - dst_boxes).abs().sum(dim=-1)
            loss_bbox_l1 = (l1_per_pair * weights).sum() / bbox_normalizer
            giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(dst_boxes))
            loss_giou = ((1.0 - torch.diag(giou)) * weights).sum() / bbox_normalizer
            match_quality = torch.cat(matched_qualities, dim=0).mean()
        else:
            loss_bbox_l1 = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0
            match_quality = pred_boxes.sum() * 0.0

        loss_total = self.cls_weight * loss_cls + self.bbox_l1_weight * loss_bbox_l1 + self.giou_weight * loss_giou
        return {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox_l1,
            "loss_bbox_l1": loss_bbox_l1,
            "loss_giou": loss_giou,
            "match_quality": match_quality.detach(),
            "num_pos": logits.new_tensor(float(positive_count)),
        }


class DINODeimStylePostProcessor:
    def __init__(
        self,
        *,
        score_threshold: float = 0.3,
        use_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        topk: int = 300,
        detections_per_image: int = 100,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.use_nms = bool(use_nms)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.topk = int(topk)
        self.detections_per_image = int(detections_per_image)

    def decode(self, predictions: dict[str, torch.Tensor], image_size: tuple[int, int]) -> list[dict[str, torch.Tensor]]:
        logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]
        batch_size, _num_queries, num_classes = logits.shape
        image_h, image_w = int(image_size[0]), int(image_size[1])
        scale = logits.new_tensor([image_w, image_h, image_w, image_h])
        results = []
        for batch_index in range(batch_size):
            scores_per_class = logits[batch_index].sigmoid()
            scores, labels = scores_per_class.max(dim=-1)
            keep = scores >= self.score_threshold
            if not bool(keep.any().item()):
                empty = logits.new_zeros((0,))
                results.append({"boxes": logits.new_zeros((0, 4)), "scores": empty, "labels": torch.zeros((0,), dtype=torch.long, device=logits.device)})
                continue
            scores = scores[keep]
            labels = labels[keep]
            boxes = box_cxcywh_to_xyxy(pred_boxes[batch_index][keep]) * scale
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_h))
            if self.topk > 0 and scores.numel() > self.topk:
                scores, top_indices = torch.topk(scores, self.topk)
                boxes = boxes[top_indices]
                labels = labels[top_indices]
            if self.use_nms and boxes.numel() > 0:
                keep_indices = nms(boxes, scores, self.nms_iou_threshold)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
            if self.detections_per_image > 0:
                boxes = boxes[: self.detections_per_image]
                scores = scores[: self.detections_per_image]
                labels = labels[: self.detections_per_image]
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results


class DINODeimStyleDetector(nn.Module):
    """DINOv3 + spatial adapter + hybrid encoder + DETR-style decoder.

    This is an experimental DEIMv2 / RT-DETR-inspired detector baseline. It is
    not a full DEIMv2 reproduction. Dense O2O, MAL and decoder refinement are
    implemented as experimental style toggles for ablation, not as official
    DEIMv2-equivalent components.
    """

    def __init__(
        self,
        *,
        backbone: DinoBackbone,
        adapter: SpatialTuningAdapter,
        encoder: HybridEncoder,
        decoder: DETRStyleDecoder,
        criterion: DINODeimStyleCriterion | None = None,
        postprocessor: DINODeimStylePostProcessor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion or DINODeimStyleCriterion(num_classes=decoder.num_classes)
        self.postprocessor = postprocessor or DINODeimStylePostProcessor()
        for module in (self.adapter, self.encoder, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def forward_raw(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        dino_feature = self.backbone(images)
        return self.forward_raw_from_dino_feature(dino_feature)

    def forward_raw_from_dino_feature(self, dino_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.adapter(dino_feature)
        encoder_output = self.encoder(features)
        predictions = self.decoder(encoder_output.memory)
        predictions["spatial_shapes"] = encoder_output.spatial_shapes  # type: ignore[assignment]
        return predictions

    def forward_from_dino_feature(
        self,
        dino_feature: torch.Tensor,
        *,
        image_size: tuple[int, int],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        predictions = self.forward_raw_from_dino_feature(dino_feature)
        if targets is not None:
            return self.criterion(predictions, targets, image_size)
        return self.postprocessor.decode(predictions, image_size)

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        predictions = self.forward_raw(images)
        image_size = (int(images.shape[-2]), int(images.shape[-1]))
        if targets is not None:
            return self.criterion(predictions, targets, image_size)
        return self.postprocessor.decode(predictions, image_size)


def _cfg_section(cfg: Mapping[str, Any], model_cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = model_cfg.get(name)
    if value is None:
        value = cfg.get(name, {})
    return dict(value or {})


def build_dino_deim_style_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DINODeimStyleDetector:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    backbone_cfg = _cfg_section(cfg, model_cfg, "backbone")
    adapter_cfg = _cfg_section(cfg, model_cfg, "adapter")
    encoder_cfg = _cfg_section(cfg, model_cfg, "encoder")
    decoder_cfg = _cfg_section(cfg, model_cfg, "decoder")
    loss_cfg = dict(cfg.get("loss", model_cfg.get("loss", {})))
    inference_cfg = dict(cfg.get("inference", cfg.get("postprocess", model_cfg.get("postprocess", {}))))

    backbone = build_dino_backbone(backbone_cfg, backbone=backbone_module)
    hidden_dim = int(
        adapter_cfg.get(
            "hidden_dim",
            adapter_cfg.get("out_channels", encoder_cfg.get("hidden_dim", decoder_cfg.get("hidden_dim", 256))),
        )
    )
    out_scales = tuple(adapter_cfg.get("out_scales", ("p3", "p4", "p5")))
    norm = adapter_cfg.get("norm", "gn")
    activation = adapter_cfg.get("activation", "silu")
    adapter = SpatialTuningAdapter(
        int(backbone_cfg.get("embed_dim") or backbone.embed_dim),
        hidden_dim,
        out_scales=out_scales,
        norm=norm,
        activation=activation,
        refine_layers=int(adapter_cfg.get("refine_layers", 0)),
        cross_scale_fusion=bool(adapter_cfg.get("cross_scale_fusion", False)),
    )
    encoder = HybridEncoder(
        hidden_dim,
        encoder_type=str(encoder_cfg.get("type", "identity")),
        num_layers=int(encoder_cfg.get("num_layers", 0)),
        num_heads=int(encoder_cfg.get("num_heads", 8)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
    )
    num_classes = int(model_cfg.get("num_classes", decoder_cfg.get("num_classes", 1)))
    decoder_hidden_dim = int(decoder_cfg.get("hidden_dim", hidden_dim))
    if decoder_hidden_dim != hidden_dim:
        raise ValueError(f"decoder.hidden_dim must match adapter/encoder hidden_dim: {decoder_hidden_dim} != {hidden_dim}")
    decoder = DETRStyleDecoder(
        hidden_dim=decoder_hidden_dim,
        num_queries=int(decoder_cfg.get("num_queries", 100)),
        num_layers=int(decoder_cfg.get("num_layers", 3)),
        num_heads=int(decoder_cfg.get("num_heads", 8)),
        num_classes=num_classes,
        dropout=float(decoder_cfg.get("dropout", 0.0)),
        refine_boxes=bool(decoder_cfg.get("refine_boxes", False)),
        box_refine_steps=int(decoder_cfg.get("box_refine_steps", 0)),
    )
    dense_cfg = dict(loss_cfg.get("dense_o2o", {}))
    mal_cfg = dict(loss_cfg.get("mal", {}))
    matcher = DINODeimStyleMatcher(
        cls_cost=float(loss_cfg.get("cls_weight", 2.0)),
        bbox_cost=float(loss_cfg.get("bbox_l1_weight", 5.0)),
        giou_cost=float(loss_cfg.get("giou_weight", 2.0)),
        dense_o2o_enabled=bool(dense_cfg.get("enabled", False)),
        extra_matches_per_gt=int(dense_cfg.get("extra_matches_per_gt", 0)),
        min_iou=float(dense_cfg.get("min_iou", 0.0)),
        max_cost=dense_cfg.get("max_cost"),
    )
    criterion = DINODeimStyleCriterion(
        num_classes=num_classes,
        matcher=matcher,
        cls_weight=float(loss_cfg.get("cls_weight", 2.0)),
        bbox_l1_weight=float(loss_cfg.get("bbox_l1_weight", 5.0)),
        giou_weight=float(loss_cfg.get("giou_weight", 2.0)),
        no_object_weight=float(loss_cfg.get("no_object_weight", 0.1)),
        use_focal_loss=bool(loss_cfg.get("use_focal_loss", True)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        mal_enabled=bool(mal_cfg.get("enabled", False)),
        mal_quality_power=float(mal_cfg.get("quality_power", 1.0)),
        mal_min_quality=float(mal_cfg.get("min_quality", 0.25)),
        mal_cls_quality_target=bool(mal_cfg.get("cls_quality_target", True)),
    )
    postprocessor = DINODeimStylePostProcessor(
        score_threshold=float(inference_cfg.get("score_thr", inference_cfg.get("score_threshold", 0.3))),
        use_nms=bool(inference_cfg.get("use_nms", False)),
        nms_iou_threshold=float(inference_cfg.get("nms_iou_thr", inference_cfg.get("nms_threshold", 0.5))),
        topk=int(inference_cfg.get("topk", 300)),
        detections_per_image=int(inference_cfg.get("detections_per_image", inference_cfg.get("topk", 100))),
    )
    return DINODeimStyleDetector(
        backbone=backbone,
        adapter=adapter,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        postprocessor=postprocessor,
    )
