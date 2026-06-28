from __future__ import annotations

import torch


class DenseTargetAssigner:
    """Target assignment scaffold for the dense face/keypoint head.

    TODO: replace the all-background assignment with center sampling or an
    FCOS-style matcher that maps boxes/keypoints to FPN locations.
    """

    def assign(
        self,
        predictions: dict[str, dict[str, torch.Tensor]],
        targets: list[dict] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        _ = targets
        assignments: dict[str, dict[str, torch.Tensor]] = {}
        for level, level_predictions in predictions.items():
            cls_logits = level_predictions["cls_logits"]
            bbox_reg = level_predictions["bbox_reg"]
            centerness = level_predictions["centerness"]
            keypoints = level_predictions["keypoints"]
            positive_mask = torch.zeros_like(centerness, dtype=torch.bool)
            assignments[level] = {
                "cls_targets": torch.zeros_like(cls_logits),
                "bbox_targets": torch.zeros_like(bbox_reg),
                "centerness_targets": torch.zeros_like(centerness),
                "keypoint_targets": torch.zeros_like(keypoints),
                "positive_mask": positive_mask,
                "keypoint_mask": positive_mask.expand_as(keypoints),
            }
            if "ellipse" in level_predictions:
                assignments[level]["ellipse_targets"] = torch.zeros_like(level_predictions["ellipse"])
        return assignments
