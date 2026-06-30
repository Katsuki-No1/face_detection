from __future__ import annotations

import math
from typing import Any

import torch


def _ellipse_to_bbox(ellipses: torch.Tensor) -> torch.Tensor:
    if ellipses.numel() == 0:
        return ellipses.new_zeros((0, 4))
    cx = ellipses[:, 0]
    cy = ellipses[:, 1]
    rx = ellipses[:, 2].clamp_min(0.0)
    ry = ellipses[:, 3].clamp_min(0.0)
    theta = ellipses[:, 4]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    half_w = torch.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
    half_h = torch.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)


def _derive_ellipses_from_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 5))
    return torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
            (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * 0.5,
            (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0) * 0.5,
            boxes.new_zeros((boxes.shape[0],)),
        ],
        dim=1,
    )


def _target_to_device(target: dict[str, Any] | None, device: torch.device, num_keypoints: int) -> dict[str, torch.Tensor]:
    if target is None:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "ellipses": torch.zeros((0, 5), dtype=torch.float32, device=device),
            "keypoints": torch.zeros((0, num_keypoints, 2), dtype=torch.float32, device=device),
            "keypoint_mask": torch.zeros((0, num_keypoints), dtype=torch.bool, device=device),
        }

    boxes_value = target.get("boxes")
    ellipses_value = target.get("ellipses")
    if boxes_value is None and ellipses_value is None:
        return _target_to_device(None, device, num_keypoints)

    if ellipses_value is None:
        boxes = torch.as_tensor(boxes_value, dtype=torch.float32, device=device).reshape(-1, 4)
        ellipses = _derive_ellipses_from_boxes(boxes)
    else:
        ellipses = torch.as_tensor(ellipses_value, dtype=torch.float32, device=device).reshape(-1, 5)
        boxes = _ellipse_to_bbox(ellipses) if boxes_value is None else torch.as_tensor(boxes_value, dtype=torch.float32, device=device).reshape(-1, 4)
        if boxes.shape[0] != ellipses.shape[0]:
            boxes = _ellipse_to_bbox(ellipses)

    instance_count = ellipses.shape[0]
    keypoints_value = target.get("keypoints", torch.zeros((instance_count, num_keypoints, 2)))
    keypoints = torch.as_tensor(keypoints_value, dtype=torch.float32, device=device)
    if keypoints.numel() == 0:
        keypoints = torch.zeros((instance_count, num_keypoints, 2), dtype=torch.float32, device=device)
    else:
        keypoints = keypoints.reshape(instance_count, -1, 2)
        if keypoints.shape[1] < num_keypoints:
            pad = torch.zeros((instance_count, num_keypoints - keypoints.shape[1], 2), dtype=torch.float32, device=device)
            keypoints = torch.cat([keypoints, pad], dim=1)
        keypoints = keypoints[:, :num_keypoints]

    mask_value = target.get("keypoint_mask", torch.zeros((instance_count, num_keypoints)))
    keypoint_mask = torch.as_tensor(mask_value, dtype=torch.bool, device=device)
    if keypoint_mask.numel() == 0:
        keypoint_mask = torch.zeros((instance_count, num_keypoints), dtype=torch.bool, device=device)
    else:
        keypoint_mask = keypoint_mask.reshape(instance_count, -1)
        if keypoint_mask.shape[1] < num_keypoints:
            pad_mask = torch.zeros((instance_count, num_keypoints - keypoint_mask.shape[1]), dtype=torch.bool, device=device)
            keypoint_mask = torch.cat([keypoint_mask, pad_mask], dim=1)
        keypoint_mask = keypoint_mask[:, :num_keypoints]

    return {"boxes": boxes, "ellipses": ellipses, "keypoints": keypoints, "keypoint_mask": keypoint_mask}


def _feature_stride(prediction: torch.Tensor, image_size: tuple[int, int]) -> tuple[float, float]:
    height, width = prediction.shape[-2:]
    return float(image_size[1]) / float(width), float(image_size[0]) / float(height)


def _draw_gaussian(
    heatmap: torch.Tensor,
    *,
    row: int,
    col: int,
    radius_y: int,
    radius_x: int,
) -> None:
    height, width = heatmap.shape[-2:]
    y0 = max(0, row - radius_y)
    y1 = min(height, row + radius_y + 1)
    x0 = max(0, col - radius_x)
    x1 = min(width, col + radius_x + 1)
    if y1 <= y0 or x1 <= x0:
        return
    ys = torch.arange(y0, y1, dtype=heatmap.dtype, device=heatmap.device)
    xs = torch.arange(x0, x1, dtype=heatmap.dtype, device=heatmap.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sigma_y = max(float(radius_y), 1.0) / 3.0
    sigma_x = max(float(radius_x), 1.0) / 3.0
    gaussian = torch.exp(-(((grid_y - row) ** 2) / (2.0 * sigma_y**2) + ((grid_x - col) ** 2) / (2.0 * sigma_x**2)))
    heatmap[..., y0:y1, x0:x1] = torch.maximum(heatmap[..., y0:y1, x0:x1], gaussian)


class CenterFaceEllipseTargetAssigner:
    """Assign one positive center cell per ellipse and Gaussian center heatmaps."""

    def __init__(
        self,
        *,
        size_ranges: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        self.size_ranges = size_ranges

    def _level_for_ellipse(self, ellipse: torch.Tensor, num_levels: int) -> int:
        if num_levels == 1:
            return 0
        size = math.sqrt(max(0.0, float(ellipse[2] * 2.0 * ellipse[3] * 2.0)))
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
        center_logits = predictions["center_logits"]
        ellipse_reg = predictions["ellipse_reg"]
        kpt_reg = predictions["kpt_reg"]
        bbox_reg = predictions.get("bbox_reg")
        device = center_logits[0].device
        batch_size = center_logits[0].shape[0]
        num_keypoints = kpt_reg[0].shape[1] // 2 if kpt_reg else 0
        targets = targets or [None for _ in range(batch_size)]  # type: ignore[list-item]

        center_targets = [torch.zeros_like(item) for item in center_logits]
        ellipse_targets = [torch.zeros_like(item) for item in ellipse_reg]
        kpt_targets = [torch.zeros_like(item) for item in kpt_reg]
        bbox_targets = [torch.zeros_like(item) for item in bbox_reg] if bbox_reg is not None else []
        positive_masks = [
            torch.zeros((item.shape[0], 1, item.shape[2], item.shape[3]), dtype=torch.bool, device=device)
            for item in center_logits
        ]
        kpt_masks = [torch.zeros_like(item, dtype=torch.bool) for item in kpt_reg]
        area_maps = [
            torch.full((item.shape[0], 1, item.shape[2], item.shape[3]), float("inf"), device=device)
            for item in center_logits
        ]

        image_h, image_w = int(image_size[0]), int(image_size[1])
        for batch_index in range(batch_size):
            target = _target_to_device(targets[batch_index] if batch_index < len(targets) else None, device, num_keypoints)
            ellipses = target["ellipses"]
            if ellipses.numel() == 0:
                continue
            boxes = target["boxes"]
            keypoints = target["keypoints"]
            keypoint_mask = target["keypoint_mask"]
            for gt_index, ellipse in enumerate(ellipses):
                cx, cy, rx, ry, theta = ellipse.tolist()
                if rx <= 0.0 or ry <= 0.0:
                    continue
                if not (0.0 <= cx < float(image_w) and 0.0 <= cy < float(image_h)):
                    continue
                level_index = self._level_for_ellipse(ellipse, len(center_logits))
                stride_x, stride_y = _feature_stride(center_logits[level_index], image_size)
                grid_x = cx / stride_x
                grid_y = cy / stride_y
                col = int(math.floor(grid_x))
                row = int(math.floor(grid_y))
                height, width = center_logits[level_index].shape[-2:]
                if row < 0 or row >= height or col < 0 or col >= width:
                    continue

                area = float((rx * 2.0) * (ry * 2.0))
                if area >= float(area_maps[level_index][batch_index, 0, row, col]):
                    continue

                radius_x = max(1, int(rx / max(stride_x, 1e-6)))
                radius_y = max(1, int(ry / max(stride_y, 1e-6)))
                _draw_gaussian(
                    center_targets[level_index][batch_index, 0],
                    row=row,
                    col=col,
                    radius_y=radius_y,
                    radius_x=radius_x,
                )
                positive_masks[level_index][batch_index, 0, row, col] = True
                area_maps[level_index][batch_index, 0, row, col] = area
                ellipse_targets[level_index][batch_index, :, row, col] = torch.tensor(
                    [grid_x - col, grid_y - row, rx, ry, theta],
                    dtype=torch.float32,
                    device=device,
                )

                if bbox_reg is not None and gt_index < boxes.shape[0]:
                    box = boxes[gt_index]
                    center = ellipse.new_tensor([cx, cy])
                    bbox_targets[level_index][batch_index, :, row, col] = torch.stack(
                        [
                            center[0] - box[0],
                            center[1] - box[1],
                            box[2] - center[0],
                            box[3] - center[1],
                        ]
                    ).clamp_min(0.0)

                for point_index in range(min(num_keypoints, keypoints.shape[1])):
                    if not bool(keypoint_mask[gt_index, point_index].item()):
                        continue
                    offset = point_index * 2
                    kpt_targets[level_index][batch_index, offset, row, col] = keypoints[gt_index, point_index, 0] - cx
                    kpt_targets[level_index][batch_index, offset + 1, row, col] = keypoints[gt_index, point_index, 1] - cy
                    kpt_masks[level_index][batch_index, offset : offset + 2, row, col] = True

        positive_count = sum(mask.sum() for mask in positive_masks)
        visible_kpt_count = sum(mask.sum() for mask in kpt_masks)
        return {
            "center_targets": center_targets,
            "ellipse_targets": ellipse_targets,
            "kpt_targets": kpt_targets,
            "bbox_targets": bbox_targets,
            "positive_masks": positive_masks,
            "kpt_masks": kpt_masks,
            "positive_count": positive_count,
            "visible_kpt_count": visible_kpt_count,
        }
