from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .postprocess import nms


def ellipse_to_bbox_tensor(ellipses: torch.Tensor, image_size: tuple[int, int] | None = None) -> torch.Tensor:
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
    boxes = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)
    if image_size is not None:
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_size[1]))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_size[0]))
    return boxes


def ellipse_aware_nms(
    ellipses: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    _ = ellipses
    return nms(boxes, scores, iou_threshold)


def _level_candidates(
    *,
    scores: torch.Tensor,
    center_offsets: torch.Tensor,
    ellipse_reg: torch.Tensor,
    kpt_reg: torch.Tensor,
    bbox_reg: torch.Tensor | None,
    image_size: tuple[int, int],
    score_threshold: float,
    topk: int,
    bbox_source: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = scores.shape[-2:]
    pooled = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
    peak_mask = scores.eq(pooled)
    flat_scores = scores.reshape(-1)
    flat_peak_mask = peak_mask.reshape(-1)
    keep = flat_peak_mask & flat_scores.ge(score_threshold)
    if not bool(keep.any().item()):
        empty = scores.new_zeros((0,))
        return empty, scores.new_zeros((0, 5)), scores.new_zeros((0, kpt_reg.shape[0] // 2, 2)), scores.new_zeros((0, 4))

    indices = torch.nonzero(keep, as_tuple=False).flatten()
    candidate_scores = flat_scores[indices]
    if topk > 0 and candidate_scores.numel() > topk:
        candidate_scores, order = torch.topk(candidate_scores, topk)
        indices = indices[order]

    rows = torch.div(indices, width, rounding_mode="floor")
    cols = indices % width
    stride_x = float(image_size[1]) / float(width)
    stride_y = float(image_size[0]) / float(height)
    flat_ellipse = ellipse_reg.permute(1, 2, 0).reshape(-1, 5)[indices]
    offsets = center_offsets.permute(1, 2, 0).reshape(-1, 2)[indices]
    center_x = (cols.to(flat_ellipse.dtype) + offsets[:, 0]) * stride_x
    center_y = (rows.to(flat_ellipse.dtype) + offsets[:, 1]) * stride_y
    ellipses = torch.stack(
        [
            center_x,
            center_y,
            flat_ellipse[:, 2],
            flat_ellipse[:, 3],
            flat_ellipse[:, 4],
        ],
        dim=1,
    )

    keypoint_offsets = kpt_reg.permute(1, 2, 0).reshape(-1, kpt_reg.shape[0])[indices]
    keypoints = keypoint_offsets.reshape(keypoint_offsets.shape[0], -1, 2)
    keypoints = keypoints + ellipses[:, None, 0:2]

    if bbox_source == "bbox" and bbox_reg is not None:
        distances = bbox_reg.permute(1, 2, 0).reshape(-1, 4)[indices].clamp_min(0.0)
        boxes = torch.stack(
            [
                center_x - distances[:, 0],
                center_y - distances[:, 1],
                center_x + distances[:, 2],
                center_y + distances[:, 3],
            ],
            dim=1,
        )
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(image_size[1]))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(image_size[0]))
    else:
        boxes = ellipse_to_bbox_tensor(ellipses, image_size=image_size)
    return candidate_scores, ellipses, keypoints, boxes


class DinoCenterFaceEllipsePostProcessor:
    def __init__(
        self,
        *,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        topk: int = 100,
        detections_per_image: int = 100,
        bbox_source: str = "bbox",
    ) -> None:
        if bbox_source not in {"bbox", "ellipse"}:
            raise ValueError(f"bbox_source must be 'bbox' or 'ellipse', got {bbox_source!r}")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.topk = int(topk)
        self.detections_per_image = int(detections_per_image)
        self.bbox_source = bbox_source

    def decode(
        self,
        predictions: dict[str, list[torch.Tensor]],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        center_logits = predictions["center_logits"]
        ellipse_reg = predictions["ellipse_reg"]
        kpt_reg = predictions["kpt_reg"]
        bbox_reg = predictions.get("bbox_reg")
        batch_size = center_logits[0].shape[0]
        results: list[dict[str, torch.Tensor]] = []

        for batch_index in range(batch_size):
            scores_per_level: list[torch.Tensor] = []
            ellipses_per_level: list[torch.Tensor] = []
            keypoints_per_level: list[torch.Tensor] = []
            boxes_per_level: list[torch.Tensor] = []
            for level_index, (level_logits, level_ellipse, level_kpts) in enumerate(zip(center_logits, ellipse_reg, kpt_reg)):
                level_bbox = bbox_reg[level_index][batch_index] if bbox_reg is not None else None
                scores, ellipses, keypoints, boxes = _level_candidates(
                    scores=level_logits[batch_index].sigmoid(),
                    center_offsets=level_ellipse[batch_index, 0:2],
                    ellipse_reg=level_ellipse[batch_index],
                    kpt_reg=level_kpts[batch_index],
                    bbox_reg=level_bbox,
                    image_size=image_size,
                    score_threshold=self.score_threshold,
                    topk=self.topk,
                    bbox_source=self.bbox_source,
                )
                if scores.numel() == 0:
                    continue
                scores_per_level.append(scores)
                ellipses_per_level.append(ellipses)
                keypoints_per_level.append(keypoints)
                boxes_per_level.append(boxes)

            if not scores_per_level:
                empty = center_logits[0].new_zeros((0,))
                num_keypoints = kpt_reg[0].shape[1] // 2
                empty_boxes = center_logits[0].new_zeros((0, 4))
                results.append(
                    {
                        "scores": empty,
                        "ellipses": center_logits[0].new_zeros((0, 5)),
                        "keypoints": center_logits[0].new_zeros((0, num_keypoints, 2)),
                        "bboxes": empty_boxes,
                        "boxes": empty_boxes,
                    }
                )
                continue

            scores = torch.cat(scores_per_level, dim=0)
            ellipses = torch.cat(ellipses_per_level, dim=0)
            keypoints = torch.cat(keypoints_per_level, dim=0)
            boxes = torch.cat(boxes_per_level, dim=0)
            if self.topk > 0 and scores.numel() > self.topk:
                scores, order = torch.topk(scores, self.topk)
                ellipses = ellipses[order]
                keypoints = keypoints[order]
                boxes = boxes[order]

            keep_indices = ellipse_aware_nms(ellipses, boxes, scores, self.nms_threshold)
            if self.detections_per_image > 0:
                keep_indices = keep_indices[: self.detections_per_image]
            boxes = boxes[keep_indices]
            results.append(
                {
                    "scores": scores[keep_indices],
                    "ellipses": ellipses[keep_indices],
                    "keypoints": keypoints[keep_indices],
                    "bboxes": boxes,
                    "boxes": boxes,
                }
            )
        return results


def _ellipse_polygon(cx: float, cy: float, rx: float, ry: float, theta: float, *, steps: int = 72) -> list[tuple[float, float]]:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    points = []
    for index in range(steps):
        angle = 2.0 * math.pi * float(index) / float(steps)
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append((cx + x * cos_theta - y * sin_theta, cy + x * sin_theta + y * cos_theta))
    return points


def draw_ellipse_detections(
    image: Image.Image,
    detections: dict[str, torch.Tensor],
    *,
    draw_bbox: bool = True,
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    ellipses = detections.get("ellipses", torch.zeros((0, 5))).detach().cpu()
    scores = detections.get("scores", torch.zeros((ellipses.shape[0],))).detach().cpu()
    keypoints = detections.get("keypoints", torch.zeros((ellipses.shape[0], 0, 2))).detach().cpu()
    boxes = detections.get("bboxes", detections.get("boxes", torch.zeros((ellipses.shape[0], 4)))).detach().cpu()
    for index, ellipse in enumerate(ellipses):
        cx, cy, rx, ry, theta = [float(value) for value in ellipse.tolist()]
        polygon = _ellipse_polygon(cx, cy, rx, ry, theta)
        draw.line(polygon + [polygon[0]], fill="red", width=2)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="cyan")
        if draw_bbox and index < boxes.shape[0]:
            x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
            draw.rectangle((x1, y1, x2, y2), outline="orange", width=1)
        if index < keypoints.shape[0]:
            for x, y in keypoints[index].tolist():
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="yellow")
        if index < scores.shape[0]:
            draw.text((cx + 3, cy + 3), f"{float(scores[index]):.2f}", fill="white")
    return image


def save_ellipse_overlay(path: str | Path, image: Image.Image, detections: dict[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_ellipse_detections(image, detections).save(path)
