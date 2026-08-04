from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.head_pose_geometry.heads import HeadDetection


def detections_from_targets(
    targets: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
) -> list[list[HeadDetection]]:
    """Build detector-independent HEAD rows without introducing a GT fallback."""

    height, width = image_size
    batches: list[list[HeadDetection]] = []
    for target in targets:
        raw_boxes = target.get("head_boxes")
        if raw_boxes is None:
            raise ValueError(
                "External HEAD cascade training requires target['head_boxes']"
            )
        boxes = torch.as_tensor(raw_boxes).detach().cpu().reshape(-1, 4)
        confidence = (
            torch.as_tensor(
                target.get("head_confidence", torch.ones(boxes.shape[0]))
            )
            .detach()
            .cpu()
            .reshape(-1)
        )
        sources = target.get("head_sources")
        if sources is None:
            sources = ["ground_truth"] * int(boxes.shape[0])
        if len(sources) != boxes.shape[0]:
            raise ValueError("head_sources length must match head_boxes")
        batches.append(
            [
                HeadDetection(
                    box_xyxy=tuple(float(value) for value in box.tolist()),
                    confidence=float(score),
                    instance_id=index,
                    valid_image_region_xyxy=(
                        0.0,
                        0.0,
                        float(width),
                        float(height),
                    ),
                    source=str(source),
                )
                for index, (box, score, source) in enumerate(
                    zip(boxes, confidence, sources, strict=True)
                )
            ]
        )
    return batches


class ExternalHeadCascadeRouter(nn.Module):
    """Route DEIM FACE candidates through external HEAD detections.

    The router is deliberately independent of the HEAD detector implementation.
    It learns only a residual on top of a deterministic containment prior, so
    an untrained checkpoint already implements a meaningful cascade.
    """

    feature_dimension = 9

    def __init__(
        self,
        *,
        hidden_dimension: int = 64,
        face_to_head_threshold: float = 0.50,
        loss_weight: float = 2.0,
        hard_negative_ratio: int = 10,
    ) -> None:
        super().__init__()
        if not 0.0 <= face_to_head_threshold <= 1.0:
            raise ValueError("face_to_head_threshold must be inside [0,1]")
        if hard_negative_ratio < 1:
            raise ValueError("hard_negative_ratio must be positive")
        self.face_to_head_threshold = float(face_to_head_threshold)
        self.loss_weight = float(loss_weight)
        self.hard_negative_ratio = int(hard_negative_ratio)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, 1),
        )
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _head_tensors(
        detections: Sequence[Sequence[HeadDetection]],
        *,
        image_size: tuple[int, int],
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = image_size
        batch = len(detections)
        maximum = max(1, max((len(rows) for rows in detections), default=0))
        boxes = like.new_zeros((batch, maximum, 4))
        confidence = like.new_zeros((batch, maximum))
        valid = torch.zeros(
            (batch, maximum),
            device=like.device,
            dtype=torch.bool,
        )
        normalizer = like.new_tensor((width, height, width, height)).clamp_min(1.0)
        for batch_index, rows in enumerate(detections):
            for head_index, raw in enumerate(rows):
                item = raw.validated(image_size)
                boxes[batch_index, head_index] = like.new_tensor(
                    item.box_xyxy
                ) / normalizer
                confidence[batch_index, head_index] = float(item.confidence)
                valid[batch_index, head_index] = True
        return boxes, confidence, valid

    def forward(
        self,
        *,
        face_boxes: torch.Tensor,
        face_logits: torch.Tensor,
        head_detections: Sequence[Sequence[HeadDetection]],
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if face_boxes.ndim != 3 or face_boxes.shape[-1] != 4:
            raise ValueError("face_boxes must have shape [B,F,4] in cxcywh")
        if face_boxes.shape[:2] != face_logits.shape[:2]:
            raise ValueError("face_boxes and face_logits must align")
        if face_boxes.shape[0] != len(head_detections):
            raise ValueError("HEAD detections must align with the image batch")
        head_xyxy, head_confidence, head_valid = self._head_tensors(
            head_detections,
            image_size=image_size,
            like=face_boxes,
        )
        head_center = 0.5 * (head_xyxy[..., :2] + head_xyxy[..., 2:])
        head_size = (head_xyxy[..., 2:] - head_xyxy[..., :2]).clamp_min(1.0e-4)
        face_center = face_boxes[..., :2]
        face_size = face_boxes[..., 2:].clamp_min(1.0e-4)

        signed_delta = (
            face_center[:, :, None, :] - head_center[:, None, :, :]
        ) / head_size[:, None, :, :]
        absolute_delta = signed_delta.abs()
        log_size_ratio = torch.log(
            face_size[:, :, None, :] / head_size[:, None, :, :]
        ).clamp(-6.0, 6.0)
        face_score = face_logits.sigmoid().amax(dim=-1)
        containment = 1.0 - 2.0 * absolute_delta.amax(dim=-1)
        face_area_ratio = (
            face_size.prod(dim=-1)[:, :, None]
            / head_size.prod(dim=-1)[:, None, :]
        ).clamp(0.0, 4.0)
        features = torch.cat(
            (
                signed_delta,
                absolute_delta,
                log_size_ratio,
                face_score[:, :, None, None].expand(
                    -1,
                    -1,
                    head_xyxy.shape[1],
                    -1,
                ),
                head_confidence[:, None, :, None].expand(
                    -1,
                    face_boxes.shape[1],
                    -1,
                    -1,
                ),
                face_area_ratio.unsqueeze(-1),
            ),
            dim=-1,
        )
        if features.shape[-1] != self.feature_dimension:
            raise RuntimeError(
                f"HEAD routing feature contract changed: {features.shape[-1]}"
            )

        # A FACE center inside a HEAD is the strongest detector-independent
        # cue. Oversized faces and very small face/head area ratios are
        # discouraged but remain recoverable by the learned residual.
        oversize = F.relu(log_size_ratio).sum(dim=-1)
        too_small = F.relu(0.03 - face_area_ratio) * 8.0
        heuristic = (
            5.0 * containment
            - 2.0 * oversize
            - too_small
            + torch.logit(
                head_confidence[:, None, :].clamp(1.0e-4, 1.0 - 1.0e-4)
            )
        )
        logits = heuristic + self.residual(features).squeeze(-1)
        logits = logits.masked_fill(~head_valid[:, None, :], -30.0)
        probability = torch.sigmoid(logits) * head_valid[:, None, :].to(logits)
        best_probability, best_head_index = probability.max(dim=-1)
        any_head = head_valid.any(dim=-1)
        best_probability = torch.where(
            any_head[:, None],
            best_probability,
            torch.zeros_like(best_probability),
        )
        best_head_index = torch.where(
            any_head[:, None],
            best_head_index,
            torch.full_like(best_head_index, -1),
        )
        return {
            "face_to_head_logits": logits,
            "face_to_head_probability": probability,
            "best_head_probability": best_probability,
            "best_head_index": best_head_index,
            "head_boxes_xyxy": head_xyxy,
            "head_confidence": head_confidence,
            "head_valid": head_valid,
            "routing_features": features,
        }

    def loss(
        self,
        routing: Mapping[str, torch.Tensor],
        *,
        face_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        geometry_target_indices: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits = routing["face_to_head_logits"]
        head_valid = routing["head_valid"]
        targets = torch.zeros_like(logits)
        supervised = head_valid[:, None, :].expand_as(logits).clone()
        positive_pairs: list[tuple[int, int, int]] = []
        for batch_index, ((face_source, face_target), raw_mapping) in enumerate(
            zip(face_matches, geometry_target_indices, strict=True)
        ):
            mapping = torch.as_tensor(
                raw_mapping,
                device=logits.device,
                dtype=torch.long,
            ).reshape(-1)
            head_count = int(head_valid[batch_index].sum())
            if mapping.shape[0] != head_count:
                raise ValueError(
                    "geometry_target_indices must align with predicted HEADs: "
                    f"{mapping.shape[0]} != {head_count}"
                )
            face_for_target = {
                int(target): int(source)
                for source, target in zip(
                    face_source.tolist(),
                    face_target.tolist(),
                    strict=True,
                )
            }
            for head_index, target_index in enumerate(mapping.tolist()):
                if int(target_index) < 0:
                    supervised[batch_index, :, head_index] = False
                    continue
                face_index = face_for_target.get(int(target_index))
                if face_index is None:
                    continue
                targets[batch_index, face_index, head_index] = 1.0
                positive_pairs.append((batch_index, face_index, head_index))

        if not bool(supervised.any()):
            zero = logits.sum() * 0.0
            return {
                "loss_external_head_routing": zero,
                "external_head_routing_accuracy": zero.detach(),
            }
        elementwise = F.binary_cross_entropy_with_logits(
            logits[supervised],
            targets[supervised],
            reduction="none",
        )
        selected_targets = targets[supervised] > 0.5
        positives = elementwise[selected_targets]
        negatives = elementwise[~selected_targets]
        negative_limit = max(
            1,
            self.hard_negative_ratio * max(1, positives.numel()),
        )
        if negatives.numel() > negative_limit:
            negatives = torch.topk(negatives, negative_limit).values
        positive_term = (
            positives.mean() if positives.numel() else logits.sum() * 0.0
        )
        negative_term = (
            negatives.mean() if negatives.numel() else logits.sum() * 0.0
        )
        correct = logits.new_zeros(())
        for batch_index, face_index, head_index in positive_pairs:
            correct = correct + (
                logits[batch_index, :, head_index].argmax() == face_index
            ).to(logits)
        accuracy = correct / max(1, len(positive_pairs))
        return {
            "loss_external_head_routing": self.loss_weight
            * (positive_term + negative_term),
            "external_head_routing_accuracy": accuracy.detach(),
        }


__all__ = [
    "ExternalHeadCascadeRouter",
    "detections_from_targets",
]
