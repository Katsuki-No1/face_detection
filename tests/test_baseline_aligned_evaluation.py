from __future__ import annotations

import json
from pathlib import Path

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
from tools.evaluate_baseline_aligned_predictions import (
    _head_coco,
    _keypoint_comparison2,
)


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
    assert THRESHOLDS["face_score"] == pytest.approx(0.33)
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


def test_head_coco_adapter_preserves_empty_image_inventory(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "one.jpg"},
                    {"id": 2, "file_name": "two.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    adapted = _head_coco(predictions, annotations)
    assert adapted[1]["heads"] == [
        {"bbox_xyxy": [10.0, 20.0, 40.0, 60.0], "score": 0.9}
    ]
    assert adapted[2] == {"heads": [], "faces": [], "poses": []}


def test_keypoint_comparison2_adapter_exposes_face_ellipse_and_points(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"images": [{"id": 7, "file_name": "images/val/a.jpg"}]}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "file_name": "images/val/a.jpg",
                "predictions": [
                    {
                        "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                        "score": 0.85,
                        "ellipse": [2.0, 3.0, 1.0, 0.5, 0.0],
                        "keypoints": [[2.0, 3.0]] * 5,
                        "visibility": [0.9] * 5,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapted = _keypoint_comparison2(predictions, annotations)[7]
    assert adapted["heads"] == []
    assert adapted["faces"][0]["ellipse"] == [2.0, 3.0, 1.0, 0.5, 0.0]
    assert adapted["poses"][0]["point_scores"] == [0.9] * 5
    assert adapted["poses"][0]["state_probabilities"] is None
