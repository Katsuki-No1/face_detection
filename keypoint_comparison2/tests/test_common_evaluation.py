from __future__ import annotations

import numpy as np
import pytest
import torch

from keypoint_comparison2.common_evaluation import (
    CommonKeypointMetrics,
    greedy_bbox_matches,
    normalized_to_original,
)


def test_normalized_to_original_applies_inverse_affine() -> None:
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    inverse = torch.tensor([[[2.0, 0.0, 10.0], [0.0, 3.0, 20.0]]])
    result = normalized_to_original(
        coordinates,
        inverse,
        input_size=11,
    )
    assert torch.allclose(
        result,
        torch.tensor([[[10.0, 20.0], [30.0, 50.0]]]),
    )


def test_common_metrics_use_gt_bbox_diagonal_and_visibility_mask() -> None:
    metrics = CommonKeypointMetrics()
    labeled = np.asarray([[True, True, False]])
    visible = np.asarray([[True, False, False]])
    metrics.register_ground_truth(labeled)
    metrics.update(
        prediction_xy=np.asarray([[[3.0, 4.0], [6.0, 8.0], [0.0, 0.0]]]),
        target_xy=np.zeros((1, 3, 2)),
        labeled=labeled,
        visible=visible,
        visibility_probability=np.asarray([[0.9, 0.8, 0.1]]),
        bbox_xyxy=np.asarray([[0.0, 0.0, 6.0, 8.0]]),
    )
    result = metrics.compute()
    assert result["nme"] == pytest.approx(0.75)
    assert result["pck_0.05"] == 0.0
    assert result["keypoint_coverage"] == 1.0
    assert result["visibility_precision"] == 0.5
    assert result["visibility_recall"] == 1.0
    assert result["visibility_f1"] == pytest.approx(2.0 / 3.0)
    assert result["occluded_precision"] == 0.0
    assert result["occluded_recall"] == 0.0
    assert result["occluded_f1"] == 0.0
    assert result["visibility_occluded_macro_f1"] == pytest.approx(1.0 / 3.0)
    assert result["visibility_confusion_matrix"] == {
        "labels": ["visible", "occluded"],
        "rows": "ground_truth",
        "columns": "prediction",
        "values": [[1, 0], [1, 0]],
    }


def test_common_metrics_report_occluded_class_metrics() -> None:
    metrics = CommonKeypointMetrics()
    labeled = np.ones((1, 6), dtype=bool)
    visible = np.asarray([[True, True, True, False, False, False]])
    metrics.register_ground_truth(labeled)
    metrics.update(
        prediction_xy=np.zeros((1, 6, 2)),
        target_xy=np.zeros((1, 6, 2)),
        labeled=labeled,
        visible=visible,
        visibility_probability=np.asarray([[0.9, 0.8, 0.1, 0.9, 0.2, 0.1]]),
        bbox_xyxy=np.asarray([[0.0, 0.0, 3.0, 4.0]]),
    )
    result = metrics.compute()
    assert result["visibility_confusion_matrix"]["values"] == [[2, 1], [1, 2]]
    assert result["occluded_precision"] == pytest.approx(2.0 / 3.0)
    assert result["occluded_recall"] == pytest.approx(2.0 / 3.0)
    assert result["occluded_f1"] == pytest.approx(2.0 / 3.0)
    assert result["visibility_occluded_macro_f1"] == pytest.approx(2.0 / 3.0)


def test_common_metrics_end_to_end_pck_counts_missed_points() -> None:
    metrics = CommonKeypointMetrics()
    metrics.register_ground_truth(np.asarray([[True, True], [True, True]]))
    metrics.update(
        prediction_xy=np.asarray([[[0.0, 0.0], [0.0, 0.0]]]),
        target_xy=np.asarray([[[0.0, 0.0], [0.0, 0.0]]]),
        labeled=np.asarray([[True, True]]),
        visible=np.asarray([[True, True]]),
        visibility_probability=np.asarray([[1.0, 1.0]]),
        bbox_xyxy=np.asarray([[0.0, 0.0, 3.0, 4.0]]),
    )
    result = metrics.compute()
    assert result["pck_0.05"] == 1.0
    assert result["keypoint_coverage"] == 0.5
    assert result["end_to_end_pck_0.05"] == 0.5


def test_greedy_bbox_matches_is_one_to_one() -> None:
    matched, missed, extra = greedy_bbox_matches(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        [[1.0, 1.0, 9.0, 9.0], [50.0, 50.0, 60.0, 60.0]],
        iou_threshold=0.5,
    )
    assert matched == [(0, 0, pytest.approx(0.64))]
    assert missed == [1]
    assert extra == [1]
