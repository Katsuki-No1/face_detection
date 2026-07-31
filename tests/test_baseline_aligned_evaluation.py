from __future__ import annotations

import pytest
import torch

from src.evaluation.baseline_aligned import (
    EXPECTED,
    IGNORED_HEAD_IDS,
    THRESHOLDS,
    _face_metrics,
    _keypoint_metrics,
    _state_metrics,
)
from tools.evaluate_deim_query_baseline_aligned import _cuda_device_index


def _row() -> dict:
    return {
        "face_target": "positive",
        "face_predicted": True,
        "head_roi_side": 120.0,
        "gt_keypoints": [[10.0 * index, 0.0] for index in range(5)],
        "gt_keypoint_states": [2, 1, 2, 2, 1],
        "predicted_keypoints": [[10.0 * index, 0.0] for index in range(5)],
        "point_scores": [0.90, 0.90, 0.90, 0.90, 0.90],
        "state_probabilities": [
            [0.01, 0.01, 0.98],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.98, 0.01],
        ],
    }


def test_locked_contract_matches_report_inventory() -> None:
    assert THRESHOLDS["head_score"] == pytest.approx(0.30)
    assert THRESHOLDS["face_score"] == pytest.approx(0.80)
    assert THRESHOLDS["point_presence"] == pytest.approx(0.70)
    assert EXPECTED["val"]["images"] == 530
    assert EXPECTED["test"]["heads"] == 806
    assert IGNORED_HEAD_IDS == {"val": {1219, 4717}, "test": {4096}}


def test_query_evaluator_normalizes_explicit_cuda_device_index() -> None:
    assert _cuda_device_index(torch.device("cpu")) is None
    assert _cuda_device_index(torch.device("cuda:0")) == 0


def test_point_and_state_metrics_use_fixed_slots_and_three_states() -> None:
    keypoint = _keypoint_metrics([_row()], supported=True)
    assert keypoint["macro_f1_0.05"] == pytest.approx(1.0)
    assert keypoint["macro_f1_0.10"] == pytest.approx(1.0)
    assert keypoint["micro_recall_0.10"] == pytest.approx(1.0)
    assert keypoint["occluded_matched_points_0.10"] == 2

    state = _state_metrics([_row()], supported=True)
    assert state["state_accuracy"] == pytest.approx(1.0)
    assert state["state_macro_f1"] == pytest.approx(1.0)
    assert state["occluded_e2e_f1"] == pytest.approx(1.0)


def test_face_macro_f1_keeps_no_face_class_and_ignores_ambiguous() -> None:
    rows = [
        {"face_target": "positive", "face_predicted": True},
        {"face_target": "negative", "face_predicted": False},
        {"face_target": "ignored", "face_predicted": True},
    ]
    result = _face_metrics(rows)
    assert result["supervised_heads"] == 2
    assert result["macro_f1"] == pytest.approx(1.0)
