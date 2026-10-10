from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from mmdet.models.dense_heads import ATSSHead
from mmdet.structures import DetDataSample
from torch import Tensor


def _xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    size = boxes[..., 2:] - boxes[..., :2]
    return torch.cat((center, size), dim=-1)


@dataclass(frozen=True)
class PositiveCoordinates:
    """Padded O2M assignments used as cooperative decoder queries."""

    anchors: Tensor
    labels: Tensor
    targets: Tensor
    valid: Tensor
    counts: tuple[int, ...]


class CoATSSHead(ATSSHead):
    """ATSS O2M head plus deterministic positive-coordinate export.

    Co-DETR trains its deployed one-to-one query branch together with one or
    more one-to-many heads. The O2M assignments are also injected into the
    shared DINO decoder as cooperative queries. MMDetection 3.x no longer
    exposes the Co-DETR-specific ``pos_coords`` return value, so this class
    restores that small contract on top of the maintained ATSS implementation.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 1,
        strides: Sequence[int] = (8, 16, 32),
        topk: int = 9,
        max_positive_queries: int = 300,
        loss_weight: float = 1.0,
    ) -> None:
        self.max_positive_queries = int(max_positive_queries)
        if self.max_positive_queries <= 0:
            raise ValueError("max_positive_queries must be positive")
        weight = float(loss_weight)
        super().__init__(
            num_classes=int(num_classes),
            in_channels=int(in_channels),
            feat_channels=int(feat_channels),
            stacked_convs=int(stacked_convs),
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[int(value) for value in strides],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=weight,
            ),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0 * weight),
            loss_centerness=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=weight,
            ),
            train_cfg=dict(
                assigner=dict(type="ATSSAssigner", topk=int(topk)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.0,
                nms=dict(type="nms", iou_threshold=0.6),
                max_per_img=100,
            ),
        )
        self.init_weights()

    @staticmethod
    def _evenly_spaced_indices(count: int, maximum: int, device: torch.device) -> Tensor:
        if count <= maximum:
            return torch.arange(count, device=device)
        return torch.linspace(0, count - 1, steps=maximum, device=device).round().long()

    def positive_coordinates(
        self,
        features: Sequence[Tensor],
        batch_data_samples: Sequence[DetDataSample],
    ) -> PositiveCoordinates:
        """Return ATSS-positive anchors and assigned boxes in normalized cxcywh."""

        feature_sizes = [feature.shape[-2:] for feature in features]
        image_metas = [sample.metainfo for sample in batch_data_samples]
        gt_instances = [sample.gt_instances for sample in batch_data_samples]
        anchors_per_image, flags_per_image = self.get_anchors(
            feature_sizes,
            image_metas,
            device=features[0].device,
        )
        level_counts = [anchors.shape[0] for anchors in anchors_per_image[0]]

        per_image: list[tuple[Tensor, Tensor, Tensor]] = []
        counts: list[int] = []
        for anchors, flags, gt, meta in zip(
            anchors_per_image,
            flags_per_image,
            gt_instances,
            image_metas,
            strict=True,
        ):
            flat_anchors = torch.cat(anchors)
            flat_flags = torch.cat(flags)
            (
                assigned_anchors,
                assigned_labels,
                _label_weights,
                assigned_targets,
                _bbox_weights,
                _positive_indices,
                _negative_indices,
                _sampling_result,
            ) = self._get_targets_single(
                flat_anchors,
                flat_flags,
                level_counts,
                gt,
                meta,
                None,
                unmap_outputs=True,
            )
            positive = (assigned_labels >= 0) & (assigned_labels < self.num_classes)
            anchor_boxes = assigned_anchors[positive]
            labels = assigned_labels[positive]
            target_boxes = assigned_targets[positive]
            keep = self._evenly_spaced_indices(
                int(anchor_boxes.shape[0]),
                self.max_positive_queries,
                anchor_boxes.device,
            )
            anchor_boxes = anchor_boxes[keep]
            labels = labels[keep]
            target_boxes = target_boxes[keep]
            height, width = meta["img_shape"][:2]
            factor = anchor_boxes.new_tensor([width, height, width, height]).clamp_min(1.0)
            per_image.append(
                (
                    _xyxy_to_cxcywh(anchor_boxes / factor).clamp(0.0, 1.0),
                    labels,
                    _xyxy_to_cxcywh(target_boxes / factor).clamp(0.0, 1.0),
                )
            )
            counts.append(int(labels.numel()))

        padded_count = max(1, max(counts, default=0))
        batch_size = len(per_image)
        dtype = features[0].dtype
        device = features[0].device
        padded_anchors = torch.zeros((batch_size, padded_count, 4), dtype=dtype, device=device)
        padded_targets = torch.zeros_like(padded_anchors)
        padded_labels = torch.full(
            (batch_size, padded_count),
            self.num_classes,
            dtype=torch.long,
            device=device,
        )
        valid = torch.zeros((batch_size, padded_count), dtype=torch.bool, device=device)
        for index, (anchors, labels, targets) in enumerate(per_image):
            count = labels.numel()
            if count:
                padded_anchors[index, :count] = anchors.to(dtype=dtype)
                padded_targets[index, :count] = targets.to(dtype=dtype)
                padded_labels[index, :count] = labels
                valid[index, :count] = True
        return PositiveCoordinates(
            anchors=padded_anchors,
            labels=padded_labels,
            targets=padded_targets,
            valid=valid,
            counts=tuple(counts),
        )

    def loss_and_positive_coordinates(
        self,
        features: Sequence[Tensor],
        batch_data_samples: Sequence[DetDataSample],
    ) -> tuple[dict[str, Tensor | list[Tensor]], PositiveCoordinates]:
        outputs = self(tuple(features))
        gt_instances = [sample.gt_instances for sample in batch_data_samples]
        image_metas = [sample.metainfo for sample in batch_data_samples]
        losses = self.loss_by_feat(*outputs, gt_instances, image_metas)
        positives = self.positive_coordinates(features, batch_data_samples)
        return losses, positives


__all__ = ["CoATSSHead", "PositiveCoordinates"]
