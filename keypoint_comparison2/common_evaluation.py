from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch


def normalized_to_original(
    coordinates: torch.Tensor,
    inverse_affine: torch.Tensor,
    *,
    input_size: int,
) -> torch.Tensor:
    """Map normalized crop coordinates back to original-image pixels."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(
            "coordinates must have shape [batch, keypoints, 2], "
            f"got {tuple(coordinates.shape)}"
        )
    if inverse_affine.shape != (coordinates.shape[0], 2, 3):
        raise ValueError(
            "inverse_affine must have shape [batch, 2, 3], "
            f"got {tuple(inverse_affine.shape)}"
        )
    crop_pixels = coordinates.float() * float(input_size - 1)
    homogeneous = torch.cat(
        (
            crop_pixels,
            torch.ones(
                (*crop_pixels.shape[:-1], 1),
                dtype=crop_pixels.dtype,
                device=crop_pixels.device,
            ),
        ),
        dim=-1,
    )
    return torch.bmm(
        homogeneous,
        inverse_affine.to(
            device=homogeneous.device,
            dtype=homogeneous.dtype,
        ).transpose(1, 2),
    )


def bbox_diagonal(box_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    return max(1.0e-6, float(np.hypot(x2 - x1, y2 - y1)))


def bbox_iou(left_xyxy: Sequence[float], right_xyxy: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left_xyxy)
    rx1, ry1, rx2, ry2 = (float(value) for value in right_xyxy)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(1.0e-9, left_area + right_area - intersection)


def greedy_bbox_matches(
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(ground_truth):
        for prediction_index, prediction_box in enumerate(predictions):
            overlap = bbox_iou(gt_box, prediction_box)
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, prediction_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for overlap, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matched.append((gt_index, prediction_index, overlap))
    missed = [
        index for index in range(len(ground_truth)) if index not in used_gt
    ]
    extra = [
        index
        for index in range(len(predictions))
        if index not in used_predictions
    ]
    return sorted(matched), missed, extra


@dataclass
class CommonKeypointMetrics:
    """GT-bbox-diagonal keypoint metrics shared by every model family."""

    thresholds: tuple[float, ...] = (0.05, 0.10)
    errors: list[np.ndarray] = field(default_factory=list)
    visible_targets: list[np.ndarray] = field(default_factory=list)
    visibility_predictions: list[np.ndarray] = field(default_factory=list)
    total_instances: int = 0
    total_labeled_points: int = 0
    evaluated_instances: int = 0

    def register_ground_truth(
        self,
        labeled: np.ndarray | torch.Tensor,
    ) -> None:
        mask = np.asarray(
            labeled.detach().cpu() if isinstance(labeled, torch.Tensor) else labeled,
            dtype=bool,
        )
        if mask.ndim == 1:
            mask = mask[None, :]
        self.total_instances += int(mask.shape[0])
        self.total_labeled_points += int(mask.sum())

    def update(
        self,
        prediction_xy: np.ndarray | torch.Tensor,
        target_xy: np.ndarray | torch.Tensor,
        labeled: np.ndarray | torch.Tensor,
        visible: np.ndarray | torch.Tensor,
        visibility_probability: np.ndarray | torch.Tensor,
        bbox_xyxy: np.ndarray | torch.Tensor,
    ) -> list[dict[str, Any]]:
        prediction = np.asarray(
            prediction_xy.detach().cpu()
            if isinstance(prediction_xy, torch.Tensor)
            else prediction_xy,
            dtype=np.float64,
        )
        target = np.asarray(
            target_xy.detach().cpu()
            if isinstance(target_xy, torch.Tensor)
            else target_xy,
            dtype=np.float64,
        )
        labeled_mask = np.asarray(
            labeled.detach().cpu() if isinstance(labeled, torch.Tensor) else labeled,
            dtype=bool,
        )
        visible_mask = np.asarray(
            visible.detach().cpu() if isinstance(visible, torch.Tensor) else visible,
            dtype=bool,
        )
        probabilities = np.asarray(
            visibility_probability.detach().cpu()
            if isinstance(visibility_probability, torch.Tensor)
            else visibility_probability,
            dtype=np.float64,
        )
        boxes = np.asarray(
            bbox_xyxy.detach().cpu()
            if isinstance(bbox_xyxy, torch.Tensor)
            else bbox_xyxy,
            dtype=np.float64,
        )
        if prediction.ndim == 2:
            prediction = prediction[None, ...]
            target = target[None, ...]
            labeled_mask = labeled_mask[None, ...]
            visible_mask = visible_mask[None, ...]
            probabilities = probabilities[None, ...]
            boxes = boxes[None, ...]
        expected = (prediction.shape[0], prediction.shape[1])
        for name, value in (
            ("target", target),
            ("labeled", labeled_mask),
            ("visible", visible_mask),
            ("visibility_probability", probabilities),
        ):
            if value.shape[:2] != expected:
                raise ValueError(
                    f"{name} shape {value.shape} does not match {expected}"
                )
        if boxes.shape != (prediction.shape[0], 4):
            raise ValueError(
                f"bbox_xyxy must have shape [{prediction.shape[0]}, 4], "
                f"got {boxes.shape}"
            )

        rows: list[dict[str, Any]] = []
        for index in range(prediction.shape[0]):
            normalization = bbox_diagonal(boxes[index])
            normalized_error = (
                np.linalg.norm(prediction[index] - target[index], axis=-1)
                / normalization
            )
            mask = labeled_mask[index]
            self.errors.append(normalized_error[mask])
            self.visible_targets.append(visible_mask[index][mask])
            self.visibility_predictions.append(probabilities[index][mask] >= 0.5)
            self.evaluated_instances += 1
            rows.append(
                {
                    "bbox_diagonal": normalization,
                    "normalized_errors": [
                        float(value) if bool(mask[position]) else None
                        for position, value in enumerate(normalized_error)
                    ],
                }
            )
        return rows

    @staticmethod
    def _keypoint_summary(
        errors: np.ndarray,
        thresholds: tuple[float, ...],
    ) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {
            "points": int(errors.size),
            "nme": float(errors.mean()) if errors.size else None,
        }
        for threshold in thresholds:
            result[f"pck_{threshold:.2f}"] = (
                float((errors <= threshold).mean())
                if errors.size
                else None
            )
        return result

    def compute(self) -> dict[str, Any]:
        errors = (
            np.concatenate(self.errors)
            if self.errors
            else np.empty((0,), dtype=np.float64)
        )
        visible = (
            np.concatenate(self.visible_targets)
            if self.visible_targets
            else np.empty((0,), dtype=bool)
        )
        predicted_visible = (
            np.concatenate(self.visibility_predictions)
            if self.visibility_predictions
            else np.empty((0,), dtype=bool)
        )
        summary: dict[str, Any] = {
            "total_instances": int(self.total_instances),
            "evaluated_instances": int(self.evaluated_instances),
            "total_labeled_points": int(self.total_labeled_points),
            "evaluated_labeled_points": int(errors.size),
            "keypoint_coverage": (
                float(errors.size / self.total_labeled_points)
                if self.total_labeled_points
                else None
            ),
            **self._keypoint_summary(errors, self.thresholds),
        }
        for threshold in self.thresholds:
            correct = int((errors <= threshold).sum())
            summary[f"end_to_end_pck_{threshold:.2f}"] = (
                float(correct / self.total_labeled_points)
                if self.total_labeled_points
                else None
            )
        natural_visible = self._keypoint_summary(
            errors[visible],
            self.thresholds,
        )
        natural_occluded = self._keypoint_summary(
            errors[~visible],
            self.thresholds,
        )
        summary.update(
            {
                f"natural_visible_{key}": value
                for key, value in natural_visible.items()
            }
        )
        summary.update(
            {
                f"natural_occluded_{key}": value
                for key, value in natural_occluded.items()
            }
        )
        if errors.size:
            true_positive = int((predicted_visible & visible).sum())
            false_positive = int((predicted_visible & ~visible).sum())
            false_negative = int((~predicted_visible & visible).sum())
            true_negative = int((~predicted_visible & ~visible).sum())
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / max(1, true_positive + false_negative)
            f1 = (
                2.0 * precision * recall
                / max(1.0e-8, precision + recall)
            )
            occluded_precision = true_negative / max(
                1,
                true_negative + false_negative,
            )
            occluded_recall = true_negative / max(
                1,
                true_negative + false_positive,
            )
            occluded_f1 = (
                2.0 * occluded_precision * occluded_recall
                / max(1.0e-8, occluded_precision + occluded_recall)
            )
            summary.update(
                {
                    "visibility_true_positive": true_positive,
                    "visibility_false_positive": false_positive,
                    "visibility_false_negative": false_negative,
                    "visibility_true_negative": true_negative,
                    "visibility_precision": precision,
                    "visibility_recall": recall,
                    "visibility_f1": f1,
                    "occluded_precision": occluded_precision,
                    "occluded_recall": occluded_recall,
                    "occluded_f1": occluded_f1,
                    "visibility_occluded_macro_f1": (
                        f1 + occluded_f1
                    )
                    / 2.0,
                    "visibility_confusion_matrix": {
                        "labels": ["visible", "occluded"],
                        "rows": "ground_truth",
                        "columns": "prediction",
                        "values": [
                            [true_positive, false_negative],
                            [false_positive, true_negative],
                        ],
                    },
                }
            )
        return summary


__all__ = [
    "CommonKeypointMetrics",
    "bbox_diagonal",
    "bbox_iou",
    "greedy_bbox_matches",
    "normalized_to_original",
]
