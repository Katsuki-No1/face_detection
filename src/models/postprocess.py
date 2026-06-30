from __future__ import annotations

import csv
from pathlib import Path

import torch

from .geometry import feature_strides, grid_centers, grid_indices


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    box_area = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return inter / (box_area + boxes_area - inter).clamp_min(1e-6)


def _fallback_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    try:
        from torchvision.ops import nms as torchvision_nms

        return torchvision_nms(boxes, scores, iou_threshold)
    except Exception:
        return _fallback_nms(boxes, scores, iou_threshold)


def _centers_for_prediction(prediction: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    return grid_centers(
        int(prediction.shape[-2]),
        int(prediction.shape[-1]),
        image_size,
        device=prediction.device,
        dtype=prediction.dtype,
        flatten=True,
    )


class DinoSCRFDPostProcessor:
    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.4,
        topk: int = 1000,
        detections_per_image: int = 300,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.topk = int(topk)
        self.detections_per_image = int(detections_per_image)

    def decode(
        self,
        predictions: dict[str, list[torch.Tensor]],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        cls_logits = predictions["cls_logits"]
        center_offset = predictions.get("center_offset")
        bbox_reg = predictions["bbox_reg"]
        kpt_reg = predictions["kpt_reg"]
        batch_size = cls_logits[0].shape[0]
        results: list[dict[str, torch.Tensor]] = []

        for batch_index in range(batch_size):
            boxes_per_level: list[torch.Tensor] = []
            scores_per_level: list[torch.Tensor] = []
            kpts_per_level: list[torch.Tensor] = []

            for level_index, (level_logits, level_boxes, level_kpts) in enumerate(zip(cls_logits, bbox_reg, kpt_reg)):
                height, width = level_logits.shape[-2:]
                stride_x, stride_y = feature_strides(height, width, image_size)
                if center_offset is None:
                    offsets = level_logits.new_full((height * width, 2), 0.5)
                else:
                    offsets = center_offset[level_index][batch_index].permute(1, 2, 0).reshape(-1, 2).clamp(0.0, 1.0)
                grid = grid_indices(height, width, device=level_logits.device, dtype=level_logits.dtype)
                centers = torch.stack(
                    [
                        (grid[:, 0] + offsets[:, 0]) * stride_x,
                        (grid[:, 1] + offsets[:, 1]) * stride_y,
                    ],
                    dim=1,
                )
                scores = level_logits[batch_index].sigmoid().reshape(-1)
                keep = scores >= self.score_threshold
                if not bool(keep.any().item()):
                    continue

                scores = scores[keep]
                centers_kept = centers[keep]
                distances = level_boxes[batch_index].permute(1, 2, 0).reshape(-1, 4)[keep].clamp_min(0.0)
                keypoint_offsets = level_kpts[batch_index].permute(1, 2, 0).reshape(-1, level_kpts.shape[1])[keep]

                boxes = torch.stack(
                    [
                        centers_kept[:, 0] - distances[:, 0],
                        centers_kept[:, 1] - distances[:, 1],
                        centers_kept[:, 0] + distances[:, 2],
                        centers_kept[:, 1] + distances[:, 3],
                    ],
                    dim=1,
                )
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_size[1]))
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_size[0]))

                keypoints = keypoint_offsets.reshape(keypoint_offsets.shape[0], -1, 2)
                keypoints = keypoints + centers_kept[:, None, :]
                boxes_per_level.append(boxes)
                scores_per_level.append(scores)
                kpts_per_level.append(keypoints)

            if not boxes_per_level:
                empty = cls_logits[0].new_zeros((0,))
                results.append(
                    {
                        "boxes": cls_logits[0].new_zeros((0, 4)),
                        "scores": empty,
                        "keypoints": cls_logits[0].new_zeros((0, kpt_reg[0].shape[1] // 2, 2)),
                    }
                )
                continue

            boxes = torch.cat(boxes_per_level, dim=0)
            scores = torch.cat(scores_per_level, dim=0)
            keypoints = torch.cat(kpts_per_level, dim=0)
            if self.topk > 0 and scores.numel() > self.topk:
                top_scores, top_indices = torch.topk(scores, self.topk)
                boxes = boxes[top_indices]
                keypoints = keypoints[top_indices]
                scores = top_scores

            keep_indices = nms(boxes, scores, self.nms_threshold)
            if self.detections_per_image > 0:
                keep_indices = keep_indices[: self.detections_per_image]
            results.append(
                {
                    "boxes": boxes[keep_indices],
                    "scores": scores[keep_indices],
                    "keypoints": keypoints[keep_indices],
                }
            )
        return results


def write_detection_csv(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "score", "x1", "y1", "x2", "y2", "keypoints"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
