from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.head_pose_geometry.heads import HeadDetection
from src.third_party.deimv2.engine.deim.box_ops import box_cxcywh_to_xyxy


def validate_strict_sequential_config(config: Mapping[str, Any]) -> None:
    """Reject configurations that silently turn the cascade back into a joint model."""

    model = config.get("model", {})
    if not isinstance(model, Mapping):
        return
    sequential = model.get("strict_sequential_cascade", {})
    if not isinstance(sequential, Mapping) or not bool(
        sequential.get("enabled", False)
    ):
        return

    shared_head = model.get("shared_head_branch", {})
    if not isinstance(shared_head, Mapping) or not bool(
        shared_head.get("enabled", False)
    ):
        raise ValueError(
            "strict_sequential_cascade requires model.shared_head_branch.enabled=true"
        )
    if not bool(shared_head.get("teacher_force_targets", False)):
        raise ValueError(
            "strict_sequential_cascade requires GT HEAD teacher forcing during training"
        )
    router = model.get("external_head_cascade", {})
    if not isinstance(router, Mapping) or not bool(router.get("enabled", False)):
        raise ValueError(
            "strict_sequential_cascade requires external_head_cascade so FACE "
            "predictions remain attached to predicted HEADs"
        )
    if not bool(sequential.get("enforce_unidirectional_refinement", False)):
        raise ValueError(
            "strict_sequential_cascade must enforce unidirectional FACE-to-keypoint refinement"
        )
    if str(sequential.get("training_head_source")) != "ground_truth":
        raise ValueError(
            "strict_sequential_cascade training_head_source must be ground_truth"
        )
    if str(sequential.get("training_face_source")) != "ground_truth":
        raise ValueError(
            "strict_sequential_cascade training_face_source must be ground_truth"
        )
    if str(sequential.get("evaluation_head_source")) != "predicted":
        raise ValueError(
            "strict_sequential_cascade evaluation_head_source must be predicted"
        )
    if str(sequential.get("evaluation_face_source")) != "predicted":
        raise ValueError(
            "strict_sequential_cascade evaluation_face_source must be predicted"
        )
    data = config.get("data", {})
    head_crop = data.get("head_crop", {}) if isinstance(data, Mapping) else {}
    if isinstance(head_crop, Mapping) and bool(head_crop.get("enabled", False)):
        raise ValueError(
            "strict_sequential_cascade uses full-image shared features and cannot be "
            "combined with the separate pixel HEAD-crop cooperative dataset"
        )
    verification = config.get("verification", {})
    verification = verification if isinstance(verification, Mapping) else {}
    if list(verification.get("expected_execution_order", ())) != [
        "head",
        "face",
        "keypoint",
    ]:
        raise ValueError(
            "strict_sequential_cascade requires verification.expected_execution_order "
            "to be [head, face, keypoint]"
        )
    if verification.get("expected_pose_to_face_refinement") is not False:
        raise ValueError(
            "strict_sequential_cascade requires "
            "verification.expected_pose_to_face_refinement=false"
        )

    for stage_name in ("head_to_face", "face_to_keypoint"):
        stage = sequential.get(stage_name, {})
        if not isinstance(stage, Mapping):
            raise ValueError(f"strict_sequential_cascade.{stage_name} is required")
        context_scale = float(stage.get("context_scale", 0.0))
        if not math.isfinite(context_scale) or context_scale < 1.0:
            raise ValueError(
                f"strict_sequential_cascade.{stage_name}.context_scale must be at least 1"
            )
        for key in ("center_jitter", "log_scale_jitter"):
            value = float(stage.get(key, -1.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"strict_sequential_cascade.{stage_name}.{key} must be non-negative"
                )
        if float(stage.get("center_jitter", 0.0)) == 0.0 and float(
            stage.get("log_scale_jitter", 0.0)
        ) == 0.0:
            raise ValueError(
                f"strict_sequential_cascade.{stage_name} must add localization noise"
            )


class StrictSequentialCascade:
    """Build causal HEAD and FACE ROI feature views without another DINO pass.

    The object intentionally has no parameters or state-dict entries.  When it is
    absent, the historical cooperative model path and checkpoint contract remain
    unchanged.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.head_to_face = dict(config.get("head_to_face", {}))
        self.face_to_keypoint = dict(config.get("face_to_keypoint", {}))
        self.face_score_threshold = float(
            config.get("face_score_threshold", 0.80)
        )
        self.maximum_face_rois = int(config.get("maximum_face_rois", 100))
        self.enforce_unidirectional_refinement = bool(
            config.get("enforce_unidirectional_refinement", True)
        )
        if not 0.0 <= self.face_score_threshold <= 1.0:
            raise ValueError("strict cascade face_score_threshold must be in [0,1]")
        if self.maximum_face_rois <= 0:
            raise ValueError("strict cascade maximum_face_rois must be positive")

    @staticmethod
    def _augment_boxes(
        boxes: torch.Tensor,
        stage: Mapping[str, Any],
        *,
        training: bool,
    ) -> torch.Tensor:
        boxes = boxes.reshape(-1, 4)
        if boxes.numel() == 0:
            return boxes
        center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1.0e-4)
        if bool(stage.get("square", True)):
            side = size.amax(dim=-1, keepdim=True)
            size = side.expand_as(size)
        if training:
            center_jitter = float(stage.get("center_jitter", 0.0))
            if center_jitter > 0.0:
                offset = torch.empty_like(center).uniform_(-1.0, 1.0)
                center = center + offset * size * center_jitter
            log_scale_jitter = float(stage.get("log_scale_jitter", 0.0))
            if log_scale_jitter > 0.0:
                scale_noise = torch.empty(
                    (boxes.shape[0], 1),
                    device=boxes.device,
                    dtype=boxes.dtype,
                ).uniform_(-log_scale_jitter, log_scale_jitter)
                size = size * scale_noise.exp()
        size = size * float(stage.get("context_scale", 1.0))
        output = torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)
        output = output.clamp(0.0, 1.0)
        valid = (output[:, 2] - output[:, 0] > 1.0e-5) & (
            output[:, 3] - output[:, 1] > 1.0e-5
        )
        return output[valid]

    @staticmethod
    def head_boxes(
        detections: Sequence[Sequence[HeadDetection]],
        *,
        image_size: tuple[int, int],
        like: torch.Tensor,
    ) -> list[torch.Tensor]:
        height, width = image_size
        normalizer = like.new_tensor((width, height, width, height)).clamp_min(1.0)
        batches: list[torch.Tensor] = []
        for rows in detections:
            boxes = [row.validated(image_size).box_xyxy for row in rows]
            batches.append(
                like.new_tensor(boxes).reshape(-1, 4) / normalizer
                if boxes
                else like.new_zeros((0, 4))
            )
        return batches

    @staticmethod
    def target_face_boxes(
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        return [box_cxcywh_to_xyxy(target["boxes"]) for target in targets]

    def predicted_face_boxes(
        self,
        face: Mapping[str, torch.Tensor],
        routing: Mapping[str, torch.Tensor],
        *,
        routing_threshold: float,
    ) -> list[torch.Tensor]:
        boxes = box_cxcywh_to_xyxy(face["pred_boxes"])
        scores = face["pred_logits"].sigmoid().amax(dim=-1)
        route_scores = routing["best_head_probability"]
        batches: list[torch.Tensor] = []
        for batch_index in range(boxes.shape[0]):
            keep = torch.nonzero(
                (scores[batch_index] >= self.face_score_threshold)
                & (route_scores[batch_index] >= float(routing_threshold)),
                as_tuple=False,
            ).squeeze(-1)
            if keep.numel() > self.maximum_face_rois:
                order = torch.topk(
                    scores[batch_index, keep], self.maximum_face_rois
                ).indices
                keep = keep[order]
            batches.append(boxes[batch_index, keep].detach())
        return batches

    @staticmethod
    def _condition_features(
        features: Sequence[torch.Tensor],
        boxes: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        if not features:
            raise ValueError("strict cascade requires at least one feature level")
        batch = int(features[0].shape[0])
        if len(boxes) != batch:
            raise ValueError("ROI boxes must align with the image batch")
        conditioned: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        coverage_rows: list[torch.Tensor] = []
        for feature in features:
            height, width = int(feature.shape[-2]), int(feature.shape[-1])
            ys = (
                torch.arange(height, device=feature.device, dtype=feature.dtype)
                + 0.5
            ) / max(1, height)
            xs = (
                torch.arange(width, device=feature.device, dtype=feature.dtype)
                + 0.5
            ) / max(1, width)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            batch_masks: list[torch.Tensor] = []
            for rows in boxes:
                rows = rows.to(device=feature.device, dtype=feature.dtype)
                if rows.numel() == 0:
                    mask = torch.zeros(
                        (height, width), device=feature.device, dtype=torch.bool
                    )
                else:
                    inside = (
                        (grid_x[None] >= rows[:, 0, None, None])
                        & (grid_x[None] <= rows[:, 2, None, None])
                        & (grid_y[None] >= rows[:, 1, None, None])
                        & (grid_y[None] <= rows[:, 3, None, None])
                    )
                    mask = inside.any(dim=0)
                batch_masks.append(mask)
            mask = torch.stack(batch_masks)
            conditioned.append(feature * mask[:, None].to(feature.dtype))
            masks.append(mask)
            coverage_rows.append(mask.float().mean(dim=(-2, -1)))
        coverage = torch.stack(coverage_rows, dim=1)
        return conditioned, masks, coverage

    def head_conditioned_features(
        self,
        features: Sequence[torch.Tensor],
        detections: Sequence[Sequence[HeadDetection]],
        *,
        image_size: tuple[int, int],
        training: bool,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        boxes = self.head_boxes(
            detections,
            image_size=image_size,
            like=features[0],
        )
        boxes = [
            self._augment_boxes(row, self.head_to_face, training=training)
            for row in boxes
        ]
        conditioned, _masks, coverage = self._condition_features(features, boxes)
        return conditioned, {
            "roi_boxes": boxes,
            "roi_count": torch.as_tensor(
                [row.shape[0] for row in boxes],
                device=features[0].device,
                dtype=torch.long,
            ),
            "feature_coverage": coverage,
            "source": "ground_truth_augmented" if training else "predicted",
        }

    def face_conditioned_features(
        self,
        features: Sequence[torch.Tensor],
        boxes: Sequence[torch.Tensor],
        *,
        training: bool,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        augmented = [
            self._augment_boxes(row, self.face_to_keypoint, training=training)
            for row in boxes
        ]
        conditioned, _masks, coverage = self._condition_features(
            features, augmented
        )
        return conditioned, {
            "roi_boxes": augmented,
            "roi_count": torch.as_tensor(
                [row.shape[0] for row in augmented],
                device=features[0].device,
                dtype=torch.long,
            ),
            "feature_coverage": coverage,
            "source": "ground_truth_augmented" if training else "predicted",
        }


__all__ = [
    "StrictSequentialCascade",
    "validate_strict_sequential_config",
]
