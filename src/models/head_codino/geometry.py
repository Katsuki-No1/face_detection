from __future__ import annotations

from collections.abc import Sequence

import torch


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
        dim=-1,
    )


def normalize_boxes_xyxy(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    factor = boxes.new_tensor((width, height, width, height)).clamp_min(1)
    return xyxy_to_cxcywh(boxes) / factor


def denormalize_boxes_cxcywh(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    factor = boxes.new_tensor((width, height, width, height))
    return cxcywh_to_xyxy(boxes) * factor


def convert_normalized_cxcywh_size(
    boxes: torch.Tensor,
    *,
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> torch.Tensor:
    """Convert normalized boxes when right/bottom padding changes the canvas."""

    source_height, source_width = source_size
    destination_height, destination_width = destination_size
    if min(source_height, source_width, destination_height, destination_width) <= 0:
        raise ValueError("source_size and destination_size must be positive")
    scale = boxes.new_tensor(
        (
            source_width / destination_width,
            source_height / destination_height,
            source_width / destination_width,
            source_height / destination_height,
        )
    )
    return boxes * scale


def original_to_padded_normalized_cxcywh(
    boxes: torch.Tensor,
    original_size: tuple[int, int],
    padded_size: tuple[int, int],
) -> torch.Tensor:
    return convert_normalized_cxcywh_size(
        boxes,
        source_size=original_size,
        destination_size=padded_size,
    )


def padded_to_original_normalized_cxcywh(
    boxes: torch.Tensor,
    original_size: tuple[int, int],
    padded_size: tuple[int, int],
) -> torch.Tensor:
    return convert_normalized_cxcywh_size(
        boxes,
        source_size=padded_size,
        destination_size=original_size,
    )


def clip_boxes_xyxy(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    clipped = boxes.clone()
    clipped[..., 0::2].clamp_(0, float(width))
    clipped[..., 1::2].clamp_(0, float(height))
    return clipped


def expand_square_roi(
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    *,
    scale: float = 1.20,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    side = (boxes[..., 2:] - boxes[..., :2]).amax(dim=-1, keepdim=True).clamp_min(0) * scale
    square = torch.cat((centers - side * 0.5, centers + side * 0.5), dim=-1)
    return clip_boxes_xyxy(square, image_size)


def postprocess_head_detections(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    original_sizes: torch.Tensor | Sequence[Sequence[int]],
    *,
    score_threshold: float = 0.30,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(f"score_threshold must be in [0,1], got {score_threshold}")
    if max_detections <= 0:
        raise ValueError(f"max_detections must be positive, got {max_detections}")
    if pred_logits.ndim != 3 or pred_logits.shape[-1] != 1:
        raise ValueError(f"Expected pred_logits [B,Q,1], got {tuple(pred_logits.shape)}")
    if pred_boxes.shape != (*pred_logits.shape[:2], 4):
        raise ValueError(f"Expected pred_boxes [B,Q,4], got {tuple(pred_boxes.shape)}")
    if isinstance(original_sizes, torch.Tensor):
        if original_sizes.shape != (pred_logits.shape[0], 2):
            raise ValueError(f"Expected original_sizes [B,2], got {tuple(original_sizes.shape)}")
        size_values: Sequence[Sequence[int] | torch.Tensor] = original_sizes
    else:
        size_values = original_sizes
        if len(size_values) != pred_logits.shape[0] or any(
            len(size) != 2 for size in size_values
        ):
            raise ValueError("Expected original_sizes as B pairs of (height, width)")
    results: list[dict[str, torch.Tensor]] = []
    for logits, normalized_boxes, size in zip(
        pred_logits, pred_boxes, size_values, strict=True
    ):
        height, width = int(size[0]), int(size[1])
        scores = logits[:, 0].sigmoid()
        boxes = clip_boxes_xyxy(denormalize_boxes_cxcywh(normalized_boxes, (height, width)), (height, width))
        valid = (
            torch.isfinite(scores)
            & torch.isfinite(boxes).all(dim=-1)
            & (scores >= float(score_threshold))
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        scores = scores[valid]
        boxes = boxes[valid]
        if scores.numel():
            order = torch.argsort(scores, descending=True, stable=True)
            order = order[:max_detections]
            scores = scores[order]
            boxes = boxes[order]
        labels = torch.zeros((scores.shape[0],), dtype=torch.long, device=scores.device)
        results.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "original_size": (
                    size.clone()
                    if isinstance(size, torch.Tensor)
                    else torch.tensor(size, dtype=torch.long, device=scores.device)
                ),
            }
        )
    return results


__all__ = [
    "clip_boxes_xyxy",
    "cxcywh_to_xyxy",
    "denormalize_boxes_cxcywh",
    "expand_square_roi",
    "normalize_boxes_xyxy",
    "original_to_padded_normalized_cxcywh",
    "padded_to_original_normalized_cxcywh",
    "postprocess_head_detections",
    "xyxy_to_cxcywh",
]
