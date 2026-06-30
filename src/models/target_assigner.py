from __future__ import annotations

import math
from typing import Any

import torch

from .geometry import feature_strides, grid_centers


def _empty_target(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "keypoints": torch.zeros((0, 5, 2), dtype=torch.float32, device=device),
        "keypoint_mask": torch.zeros((0, 5), dtype=torch.bool, device=device),
    }


def _target_to_device(target: dict[str, Any] | None, device: torch.device) -> dict[str, torch.Tensor]:
    if target is None:
        return _empty_target(device)
    boxes = target.get("boxes")
    if boxes is None:
        return _empty_target(device)
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return _empty_target(device)

    raw_keypoints = torch.as_tensor(
        target.get("keypoints", torch.zeros((boxes.shape[0], 5, 2))),
        dtype=torch.float32,
        device=device,
    )
    if raw_keypoints.numel() == 0:
        keypoints = torch.zeros((boxes.shape[0], 5, 2), dtype=torch.float32, device=device)
    else:
        keypoints = raw_keypoints.reshape(boxes.shape[0], -1, 2)
    keypoint_mask = torch.as_tensor(
        target.get("keypoint_mask", torch.zeros((boxes.shape[0], keypoints.shape[1]))),
        dtype=torch.bool,
        device=device,
    ).reshape(boxes.shape[0], keypoints.shape[1])
    return {"boxes": boxes, "keypoints": keypoints, "keypoint_mask": keypoint_mask}


def _level_centers(height: int, width: int, image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    return grid_centers(height, width, image_size, device=device, dtype=torch.float32)


class CenterFaceTargetAssigner:
    """Simple center-cell assignment for the first DINO-SCRFD training scaffold."""

    def __init__(
        self,
        *,
        size_ranges: tuple[tuple[float, float], ...] | None = None,
        center_sampling_radius: int = 0,
    ) -> None:
        self.size_ranges = size_ranges
        self.center_sampling_radius = int(center_sampling_radius)

    def _level_for_box(self, box: torch.Tensor, num_levels: int) -> int:
        if num_levels == 1:
            return 0
        size = math.sqrt(max(0.0, float((box[2] - box[0]) * (box[3] - box[1]))))
        ranges = self.size_ranges
        if ranges is None:
            if num_levels == 2:
                ranges = ((0.0, 96.0), (96.0, float("inf")))
            else:
                ranges = ((0.0, 64.0), (64.0, 128.0), (128.0, float("inf")))
        for index, (lower, upper) in enumerate(ranges[:num_levels]):
            if lower <= size < upper:
                return index
        return min(num_levels - 1, len(ranges) - 1)

    def assign(
        self,
        predictions: dict[str, list[torch.Tensor]],
        targets: list[dict[str, Any]] | None,
        image_size: tuple[int, int],
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        cls_logits = predictions["cls_logits"]
        center_offset = predictions["center_offset"]
        bbox_reg = predictions["bbox_reg"]
        kpt_reg = predictions["kpt_reg"]
        device = cls_logits[0].device
        batch_size = cls_logits[0].shape[0]
        num_keypoints = kpt_reg[0].shape[1] // 2
        targets = targets or [None for _ in range(batch_size)]  # type: ignore[list-item]

        cls_targets = [torch.zeros_like(item) for item in cls_logits]
        center_offset_targets = [torch.full_like(item, 0.5) for item in center_offset]
        bbox_targets = [torch.zeros_like(item) for item in bbox_reg]
        kpt_targets = [torch.zeros_like(item) for item in kpt_reg]
        positive_masks = [
            torch.zeros((item.shape[0], 1, item.shape[2], item.shape[3]), dtype=torch.bool, device=device)
            for item in cls_logits
        ]
        kpt_masks = [torch.zeros_like(item, dtype=torch.bool) for item in kpt_reg]
        area_maps = [
            torch.full((item.shape[0], 1, item.shape[2], item.shape[3]), float("inf"), device=device)
            for item in cls_logits
        ]

        for batch_index in range(batch_size):
            target = _target_to_device(targets[batch_index] if batch_index < len(targets) else None, device)
            boxes = target["boxes"]
            if boxes.numel() == 0:
                continue
            keypoints = target["keypoints"]
            keypoint_mask = target["keypoint_mask"]
            for gt_index, box in enumerate(boxes):
                x1, y1, x2, y2 = box.tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                level_index = self._level_for_box(box, len(cls_logits))
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                height, width = cls_logits[level_index].shape[-2:]
                stride_x, stride_y = feature_strides(height, width, image_size)
                grid_x = center_x / stride_x
                grid_y = center_y / stride_y
                col = min(width - 1, max(0, int(math.floor(grid_x))))
                row = min(height - 1, max(0, int(math.floor(grid_y))))

                gt_area = (x2 - x1) * (y2 - y1)
                if gt_area >= float(area_maps[level_index][batch_index, 0, row, col]):
                    continue

                cls_targets[level_index][batch_index, 0, row, col] = 1.0
                positive_masks[level_index][batch_index, 0, row, col] = True
                area_maps[level_index][batch_index, 0, row, col] = gt_area
                center_offset_targets[level_index][batch_index, :, row, col] = torch.tensor(
                    [grid_x - float(col), grid_y - float(row)],
                    dtype=torch.float32,
                    device=device,
                ).clamp(0.0, 1.0)
                bbox_targets[level_index][batch_index, :, row, col] = torch.stack(
                    [center_x - box[0], center_y - box[1], box[2] - center_x, box[3] - center_y]
                ).clamp_min(0.0)

                visible = keypoint_mask[gt_index]
                for point_index in range(min(num_keypoints, keypoints.shape[1])):
                    offset = point_index * 2
                    if not bool(visible[point_index].item()):
                        continue
                    kpt_targets[level_index][batch_index, offset, row, col] = keypoints[gt_index, point_index, 0] - center_x
                    kpt_targets[level_index][batch_index, offset + 1, row, col] = keypoints[gt_index, point_index, 1] - center_y
                    kpt_masks[level_index][batch_index, offset : offset + 2, row, col] = True

        positive_count = sum(mask.sum() for mask in positive_masks)
        visible_kpt_count = sum(mask.sum() for mask in kpt_masks)
        return {
            "cls_targets": cls_targets,
            "center_offset_targets": center_offset_targets,
            "bbox_targets": bbox_targets,
            "kpt_targets": kpt_targets,
            "positive_masks": positive_masks,
            "kpt_masks": kpt_masks,
            "positive_count": positive_count,
            "visible_kpt_count": visible_kpt_count,
        }
