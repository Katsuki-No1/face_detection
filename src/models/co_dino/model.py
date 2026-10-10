from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from torch import Tensor, nn

from src.models.dinov3_deimv2.official import (
    DINOv3STABackbone,
    OfficialTinySTABackbone,
    _load_pretrained_checkpoint,
)
from src.models.dinov3_deimv2_ellipse import ellipse_to_aabb

from .o2m_head import CoATSSHead, PositiveCoordinates
from .neck import CoDINOChannelMapper
from .query_head import CoDINOQueryHead


def _config_section(
    root: Mapping[str, Any],
    model: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    return dict(model.get(name, root.get(name, {})) or {})


def _sum_loss_value(value: Tensor | Sequence[Tensor]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    values = list(value)
    if not values:
        raise ValueError("O2M loss list must not be empty")
    return sum(values[1:], values[0])


class DINOv3CoDINODetector(nn.Module):
    """Full-image DINOv3 + Co-DINO detector, independent of the DEIM path."""

    deployed_inference_topology = "non-cascade: full-image FACE bbox"

    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: CoDINOChannelMapper,
        query_head: CoDINOQueryHead,
        o2m_head: CoATSSHead | None,
        category_ids: Sequence[int],
        score_threshold: float = 0.001,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.query_head = query_head
        self.o2m_head = o2m_head
        self.category_ids = tuple(int(value) for value in category_ids)
        if len(self.category_ids) != self.query_head.num_classes:
            raise ValueError(
                "category_ids must contain exactly one id per detector class"
            )
        self.category_to_label = {
            category_id: index for index, category_id in enumerate(self.category_ids)
        }
        self.score_threshold = float(score_threshold)
        self.detections_per_image = int(detections_per_image)
        if self.query_head.ellipse_enabled:
            self.deployed_inference_topology = (
                "non-cascade: full-image FACE bbox/ellipse"
            )
        self.current_epoch = 0
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.detections_per_image <= 0:
            raise ValueError("detections_per_image must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _resolve_padding_mask(
        images: Tensor,
        targets: Sequence[Mapping[str, Any]] | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        if padding_mask is None and targets:
            target_masks = [target.get("padding_mask") for target in targets]
            if all(isinstance(mask, Tensor) for mask in target_masks):
                padding_mask = torch.stack(
                    [mask.to(dtype=torch.bool) for mask in target_masks],
                    dim=0,
                )
        if padding_mask is None:
            return torch.zeros(
                (images.shape[0], images.shape[-2], images.shape[-1]),
                dtype=torch.bool,
                device=images.device,
            )
        mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=images.device)
        expected = (images.shape[0], images.shape[-2], images.shape[-1])
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"padding_mask must have shape {expected}, got {tuple(mask.shape)}"
            )
        return mask

    @staticmethod
    def _feature_masks(mask: Tensor, features: Sequence[Tensor]) -> tuple[Tensor, ...]:
        # A coarse Co-DINO level is valid when any source pixel in its pooling
        # region is valid. Nearest-neighbor sampling can otherwise turn P6/P7
        # entirely invalid for letterboxed or geometrically augmented images,
        # yielding zero valid ratios and NaNs in deformable attention.
        valid = (~mask).to(dtype=torch.float32)[:, None]
        return tuple(
            ~F.adaptive_max_pool2d(
                valid,
                output_size=feature.shape[-2:],
            )[:, 0].to(dtype=torch.bool)
            for feature in features
        )

    def _build_data_samples(
        self,
        images: Tensor,
        targets: Sequence[Mapping[str, Any]],
    ) -> list[DetDataSample]:
        if len(targets) != images.shape[0]:
            raise ValueError("targets length must match the image batch")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        samples: list[DetDataSample] = []
        for index, target in enumerate(targets):
            boxes = torch.as_tensor(
                target.get("boxes", []),
                dtype=torch.float32,
                device=images.device,
            ).reshape(-1, 4)
            source_labels = torch.as_tensor(
                target.get("labels", torch.zeros(boxes.shape[0], dtype=torch.long)),
                dtype=torch.long,
                device=images.device,
            ).reshape(-1)
            if source_labels.numel() != boxes.shape[0]:
                raise ValueError("Each target must contain one label per box")
            mapped_labels: list[int] = []
            kept_boxes: list[Tensor] = []
            raw_ellipses = target.get("ellipses")
            ellipse_tensor = None
            if raw_ellipses is not None:
                ellipse_tensor = torch.as_tensor(
                    raw_ellipses,
                    dtype=torch.float32,
                    device=images.device,
                ).reshape(-1, 5)
                if ellipse_tensor.shape[0] != boxes.shape[0]:
                    raise ValueError("Each target must contain one ellipse per box")
            kept_ellipses: list[Tensor] = []
            for source_index, (box, source_label) in enumerate(
                zip(boxes, source_labels, strict=True)
            ):
                raw_label = int(source_label.item())
                mapped = self.category_to_label.get(raw_label)
                if mapped is None and 0 <= raw_label < len(self.category_ids):
                    mapped = raw_label
                if mapped is None and 1 <= raw_label <= len(self.category_ids):
                    # Synthetic smoke data and some already-adapted loaders
                    # expose contiguous one-based labels rather than original
                    # COCO category ids. The configured dataset filter remains
                    # responsible for selecting FACE before this conversion.
                    mapped = raw_label - 1
                if mapped is not None:
                    kept_boxes.append(box)
                    mapped_labels.append(mapped)
                    if ellipse_tensor is not None:
                        kept_ellipses.append(ellipse_tensor[source_index])
            if kept_boxes:
                boxes = torch.stack(kept_boxes)
                labels = torch.tensor(mapped_labels, dtype=torch.long, device=images.device)
                ellipses = (
                    torch.stack(kept_ellipses)
                    if ellipse_tensor is not None
                    else None
                )
                boxes[:, 0::2].clamp_(0, width)
                boxes[:, 1::2].clamp_(0, height)
                valid = (
                    torch.isfinite(boxes).all(dim=-1)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                )
                boxes = boxes[valid]
                labels = labels[valid]
                if ellipses is not None:
                    ellipses = ellipses[valid]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)
                labels = torch.zeros((0,), dtype=torch.long, device=images.device)
                ellipses = (
                    torch.zeros((0, 5), dtype=torch.float32, device=images.device)
                    if ellipse_tensor is not None
                    else None
                )
            sample = DetDataSample()
            sample.gt_instances = InstanceData(bboxes=boxes, labels=labels)
            if ellipses is not None:
                sample.gt_instances.ellipses = ellipses
            sample.set_metainfo(
                {
                    "img_id": target.get("image_id", index),
                    "img_shape": (height, width),
                    "ori_shape": (height, width),
                    "pad_shape": (height, width),
                    "batch_input_shape": (height, width),
                    "scale_factor": (1.0, 1.0),
                }
            )
            samples.append(sample)
        return samples

    def extract_features(self, images: Tensor) -> tuple[Tensor, ...]:
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, Mapping):
            backbone_features = tuple(backbone_features.values())
        else:
            backbone_features = tuple(backbone_features)
        features = self.neck(backbone_features)
        if len(features) != self.query_head.num_feature_levels:
            raise ValueError(
                f"Backbone returned {len(features)} levels; Co-DINO expects "
                f"{self.query_head.num_feature_levels}"
            )
        return features

    def forward_raw(
        self,
        images: Tensor,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        padding_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        mask = self._resolve_padding_mask(images, targets, padding_mask)
        features = self.extract_features(images)
        masks = self._feature_masks(mask, features)
        samples = self._build_data_samples(images, targets) if targets is not None else None
        positives: PositiveCoordinates | None = None
        o2m_losses: dict[str, Tensor | list[Tensor]] = {}
        if self.training and samples is not None and self.o2m_head is not None:
            o2m_losses, positives = self.o2m_head.loss_and_positive_coordinates(
                features,
                samples,
            )
        outputs = self.query_head(
            features,
            masks,
            batch_data_samples=samples,
            positive_coordinates=positives,
        )
        outputs["_batch_data_samples"] = samples
        outputs["_o2m_losses"] = o2m_losses
        outputs["_features"] = features
        outputs["_feature_masks"] = masks
        return outputs

    def loss(
        self,
        images: Tensor,
        targets: Sequence[Mapping[str, Any]],
        *,
        padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        outputs = self.forward_raw(images, targets, padding_mask=padding_mask)
        samples = outputs["_batch_data_samples"]
        o2o_losses = self.query_head.loss(outputs, samples)
        o2o_losses.pop("loss_total", None)
        losses: dict[str, Tensor] = dict(o2o_losses)
        for name, value in outputs["_o2m_losses"].items():
            losses[f"o2m.{name}"] = _sum_loss_value(value)
        losses.update(self.query_head.cooperative_loss(outputs, samples))
        total_values = list(losses.values())
        if not total_values:
            raise RuntimeError("Co-DINO produced no tensor losses")
        losses["loss_total"] = sum(total_values[1:], total_values[0])
        if "loss_bbox" in losses:
            losses["loss_bbox_l1"] = losses["loss_bbox"]
        ellipse = [
            value
            for name, value in losses.items()
            if name.startswith("loss_ellipse_")
        ]
        if ellipse:
            losses["loss_ellipse"] = sum(ellipse[1:], ellipse[0])
        cooperative = [
            value for name, value in losses.items() if ".aux_o2m." in name
        ]
        if cooperative:
            losses["loss_cooperative"] = sum(cooperative[1:], cooperative[0])
        o2m = [value for name, value in losses.items() if name.startswith("o2m.")]
        if o2m:
            losses["loss_o2m"] = sum(o2m[1:], o2m[0])
        return losses

    def predict_from_raw(self, outputs: Mapping[str, Any], image_size: tuple[int, int]) -> list[dict[str, Tensor]]:
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        normalized_ellipses = outputs.get("pred_ellipses")
        height, width = image_size
        results: list[dict[str, Tensor]] = []
        for image_logits, image_boxes in zip(logits, boxes, strict=True):
            scores = image_logits.sigmoid().flatten()
            count = min(self.detections_per_image, int(scores.numel()))
            scores, indices = torch.topk(scores, count, sorted=True)
            labels = indices.remainder(self.query_head.num_classes).long()
            query_indices = torch.div(
                indices,
                self.query_head.num_classes,
                rounding_mode="floor",
            )
            selected = image_boxes[query_indices]
            xyxy = torch.cat(
                (
                    selected[:, :2] - 0.5 * selected[:, 2:],
                    selected[:, :2] + 0.5 * selected[:, 2:],
                ),
                dim=-1,
            )
            xyxy = xyxy * xyxy.new_tensor([width, height, width, height])
            xyxy[:, 0::2].clamp_(0, width)
            xyxy[:, 1::2].clamp_(0, height)
            valid = (
                (scores >= self.score_threshold)
                & (xyxy[:, 2] > xyxy[:, 0])
                & (xyxy[:, 3] > xyxy[:, 1])
            )
            result = {
                "boxes": xyxy[valid],
                "scores": scores[valid],
                "labels": labels[valid],
            }
            if normalized_ellipses is not None:
                ellipses = normalized_ellipses[len(results), query_indices].clone()
                ellipses[..., 0] *= width
                ellipses[..., 1] *= height
                ellipses[..., 2:4] *= max(width, height)
                ellipses = ellipses[valid]
                result["ellipses"] = ellipses
                result["ellipse_boxes"] = ellipse_to_aabb(ellipses)
            results.append(result)
        return results

    def forward(
        self,
        images: Tensor,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        padding_mask: Tensor | None = None,
        return_raw: bool = False,
    ) -> dict[str, Tensor] | dict[str, Any] | list[dict[str, Tensor]]:
        if targets is not None:
            if return_raw:
                return self.forward_raw(images, targets, padding_mask=padding_mask)
            return self.loss(images, targets, padding_mask=padding_mask)
        outputs = self.forward_raw(images, padding_mask=padding_mask)
        if return_raw:
            return outputs
        return self.predict_from_raw(outputs, (int(images.shape[-2]), int(images.shape[-1])))


def build_dinov3_codino_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> DINOv3CoDINODetector:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    data_cfg = dict(cfg.get("data", {}))
    backbone_cfg = _config_section(cfg, model_cfg, "backbone")
    adapter_cfg = _config_section(cfg, model_cfg, "adapter")
    neck_cfg = _config_section(cfg, model_cfg, "neck")
    query_cfg = _config_section(cfg, model_cfg, "query_head")
    o2m_cfg = _config_section(cfg, model_cfg, "o2m_head")
    ellipse_cfg = _config_section(cfg, model_cfg, "ellipse_head")
    inference_cfg = dict(cfg.get("inference", model_cfg.get("inference", {})) or {})

    num_classes = int(model_cfg.get("num_classes", 1))
    embed_dims = int(query_cfg.get("embed_dims", adapter_cfg.get("hidden_dim", 256)))
    if embed_dims != int(adapter_cfg.get("hidden_dim", embed_dims)):
        raise ValueError("adapter.hidden_dim and query_head.embed_dims must match")
    if backbone_module is None:
        backbone_type = str(backbone_cfg.get("type", "")).lower()
        backbone_class = (
            OfficialTinySTABackbone
            if "tiny" in backbone_type
            else DINOv3STABackbone
        )
        backbone = backbone_class(
            backbone_cfg=backbone_cfg,
            hidden_dim=embed_dims,
            interaction_indexes=tuple(adapter_cfg.get("interaction_indexes", (5, 8, 11))),
            conv_inplane=int(adapter_cfg.get("conv_inplane", 32)),
            use_sta=bool(adapter_cfg.get("use_sta", True)),
        )
    else:
        backbone = backbone_module

    neck = CoDINOChannelMapper(
        in_channels=tuple(neck_cfg.get("in_channels", (embed_dims, embed_dims, embed_dims))),
        out_channels=embed_dims,
        num_outs=int(neck_cfg.get("num_outs", query_cfg.get("num_feature_levels", 5))),
        norm_groups=int(neck_cfg.get("norm_groups", 32)),
    )

    query_head = CoDINOQueryHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_feature_levels=int(query_cfg.get("num_feature_levels", 5)),
        num_queries=int(query_cfg.get("num_queries", 300)),
        num_encoder_layers=int(query_cfg.get("num_encoder_layers", 6)),
        num_decoder_layers=int(query_cfg.get("num_decoder_layers", 6)),
        feedforward_channels=int(query_cfg.get("feedforward_channels", 2048)),
        num_denoising_queries=int(query_cfg.get("num_denoising_queries", 100)),
        stable_self_attention=bool(query_cfg.get("stable_self_attention", False)),
        cooperative_loss_weight=float(query_cfg.get("cooperative_loss_weight", 1.0)),
        ellipse=ellipse_cfg,
    )
    o2m_enabled = bool(o2m_cfg.get("enabled", True))
    o2m_head = None
    if o2m_enabled:
        strides = tuple(o2m_cfg.get("strides", (8, 16, 32, 64, 128)))
        if len(strides) != query_head.num_feature_levels:
            raise ValueError("o2m_head.strides must match the number of feature levels")
        o2m_head = CoATSSHead(
            num_classes=num_classes,
            in_channels=embed_dims,
            feat_channels=int(o2m_cfg.get("feat_channels", embed_dims)),
            stacked_convs=int(o2m_cfg.get("stacked_convs", 1)),
            strides=strides,
            topk=int(o2m_cfg.get("topk", 9)),
            max_positive_queries=int(o2m_cfg.get("max_positive_queries", 300)),
            loss_weight=float(o2m_cfg.get("loss_weight", 1.0)),
        )
    category_ids = tuple(int(value) for value in data_cfg.get("category_ids", range(1, num_classes + 1)))
    detector = DINOv3CoDINODetector(
        backbone=backbone,
        neck=neck,
        query_head=query_head,
        o2m_head=o2m_head,
        category_ids=category_ids,
        score_threshold=float(inference_cfg.get("score_thr", 0.001)),
        detections_per_image=int(inference_cfg.get("detections_per_image", 100)),
    )
    init_cfg = dict(
        model_cfg.get("init_checkpoint", cfg.get("init_checkpoint", {})) or {}
    )
    if init_cfg:
        _load_pretrained_checkpoint(detector, init_cfg)
    return detector


__all__ = ["DINOv3CoDINODetector", "build_dinov3_codino_detector"]
