from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from torch import nn

from src.head_pose_geometry import HeadDetection
from src.models.head_codino import CompactCoDINO, SimpleFeaturePyramid
from src.models.head_codino.geometry import postprocess_head_detections


class SharedDetectorFeaturePyramid(nn.Module):
    """The three SFP levels actually consumed by ``CompactCoDINO``."""

    def __init__(self, in_channels: int = 384) -> None:
        super().__init__()
        complete = SimpleFeaturePyramid(in_channels=int(in_channels))
        self.p3 = complete.p3
        self.p4 = complete.p4
        self.p5_pool = complete.p5_pool
        self.p5 = complete.p5

    def forward(
        self,
        feature: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        p3 = self.p3(
            F.interpolate(
                feature,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
        )
        p4 = self.p4(feature)
        p5 = self.p5(self.p5_pool(feature))
        return OrderedDict((("p3", p3), ("p4", p4), ("p5", p5)))


class SharedDINOHeadBranch(nn.Module):
    """Co-DINO HEAD branch fed by the Face branch's single DINO forward.

    This module intentionally contains no backbone.  It consumes the stride-16
    block-11 stride-16 map returned by
    ``DINOv3STABackbone.forward_with_semantic_and_final_stride16`` and adds
    only the SFP and compact one-class Co-DINO query head.
    """

    def __init__(
        self,
        *,
        in_channels: int = 384,
        category_id: int = 1,
        score_threshold: float = 0.001,
        max_detections: int = 100,
        loss_weight: float = 1.0,
        teacher_force_targets: bool = False,
        stable_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.sfp = SharedDetectorFeaturePyramid(
            in_channels=int(in_channels)
        )
        self.query_head = CompactCoDINO(
            stable_self_attention=bool(stable_self_attention)
        )
        self.category_id = int(category_id)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.loss_weight = float(loss_weight)
        self.teacher_force_targets = bool(teacher_force_targets)
        if self.category_id <= 0:
            raise ValueError("category_id must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0,1]")
        if not 0 < self.max_detections <= self.query_head.num_queries:
            raise ValueError(
                f"max_detections must be in [1,{self.query_head.num_queries}]"
            )
        if self.loss_weight <= 0.0:
            raise ValueError("loss_weight must be positive")

    @staticmethod
    def _feature_masks(
        pyramid: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks: list[torch.Tensor] = []
        for level in ("p3", "p4", "p5"):
            feature = pyramid[level]
            if padding_mask is None:
                mask = torch.zeros(
                    feature.shape[0],
                    feature.shape[-2],
                    feature.shape[-1],
                    dtype=torch.bool,
                    device=feature.device,
                )
            else:
                if padding_mask.ndim != 3:
                    raise ValueError(
                        "padding_mask must have shape [B,H,W], got "
                        f"{tuple(padding_mask.shape)}"
                    )
                mask = F.interpolate(
                    padding_mask[:, None].float(),
                    size=feature.shape[-2:],
                    mode="nearest",
                )[:, 0].to(dtype=torch.bool)
            masks.append(mask)
        return tuple(masks)  # type: ignore[return-value]

    def _build_data_samples(
        self,
        targets: Sequence[Mapping[str, Any]],
        *,
        image_size: tuple[int, int],
        device: torch.device,
    ) -> list[DetDataSample]:
        height, width = image_size
        samples: list[DetDataSample] = []
        for index, target in enumerate(targets):
            boxes = torch.as_tensor(
                target.get("head_detection_boxes", target.get("head_boxes", [])),
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 4)
            finite = torch.isfinite(boxes).all(dim=-1)
            positive = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[finite & positive].clone()
            if boxes.numel():
                boxes[:, 0::2].clamp_(0.0, float(width))
                boxes[:, 1::2].clamp_(0.0, float(height))
                boxes = boxes[
                    (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                ]
            sample = DetDataSample()
            sample.gt_instances = InstanceData(
                bboxes=boxes,
                labels=torch.zeros(
                    boxes.shape[0],
                    dtype=torch.long,
                    device=device,
                ),
            )
            image_id = target.get("image_id", index)
            if isinstance(image_id, torch.Tensor) and image_id.numel() == 1:
                image_id = int(image_id.item())
            sample.set_metainfo(
                {
                    "img_id": image_id,
                    "img_shape": (height, width),
                    "ori_shape": (height, width),
                    "batch_input_shape": (height, width),
                    "scale_factor": (1.0, 1.0),
                }
            )
            samples.append(sample)
        return samples

    @staticmethod
    def _as_head_detections(
        predictions: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[list[HeadDetection]]:
        batches: list[list[HeadDetection]] = []
        for result in predictions:
            boxes = result["boxes"].detach().cpu().reshape(-1, 4)
            scores = result["scores"].detach().cpu().reshape(-1)
            batches.append(
                [
                    HeadDetection(
                        box_xyxy=tuple(float(value) for value in box),
                        confidence=float(score),
                        instance_id=query_index,
                        source="shared_dinov3_vits16plus_codino",
                    )
                    for query_index, (box, score) in enumerate(
                        zip(boxes, scores, strict=True)
                    )
                ]
            )
        return batches

    def forward(
        self,
        semantic_feature: torch.Tensor,
        *,
        image_size: tuple[int, int],
        targets: Sequence[Mapping[str, Any]] | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        pyramid = self.sfp(semantic_feature)
        features = tuple(pyramid[level] for level in ("p3", "p4", "p5"))
        masks = self._feature_masks(pyramid, padding_mask)
        samples = (
            self._build_data_samples(
                targets,
                image_size=image_size,
                device=semantic_feature.device,
            )
            if targets is not None
            else None
        )
        restore_eval_mode = samples is not None and not self.query_head.training
        if restore_eval_mode:
            self.query_head.train()
        try:
            raw = self.query_head(
                features,
                masks,
                batch_data_samples=samples,
            )
        finally:
            if restore_eval_mode:
                self.query_head.eval()
        predictions = postprocess_head_detections(
            raw["pred_logits"],
            raw["pred_boxes"],
            [image_size] * int(semantic_feature.shape[0]),
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
        )
        routing_predictions = predictions
        if self.training and self.teacher_force_targets and targets is not None:
            routing_predictions = []
            for target in targets:
                boxes = torch.as_tensor(
                    target.get(
                        "head_detection_boxes",
                        target.get("head_boxes", []),
                    ),
                    dtype=torch.float32,
                    device=semantic_feature.device,
                ).reshape(-1, 4)
                routing_predictions.append(
                    {
                        "boxes": boxes,
                        "scores": torch.ones(
                            boxes.shape[0],
                            dtype=semantic_feature.dtype,
                            device=semantic_feature.device,
                        ),
                        "labels": torch.zeros(
                            boxes.shape[0],
                            dtype=torch.long,
                            device=semantic_feature.device,
                        ),
                        "original_size": torch.tensor(
                            image_size,
                            dtype=torch.long,
                            device=semantic_feature.device,
                        ),
                    }
                )
        result: dict[str, Any] = {
            "raw_output": raw,
            "predictions": routing_predictions,
            "detector_predictions": predictions,
            "detections": self._as_head_detections(routing_predictions),
        }
        if samples is not None:
            result["losses"] = self.query_head.loss(raw, samples)
        return result


__all__ = ["SharedDINOHeadBranch", "SharedDetectorFeaturePyramid"]
