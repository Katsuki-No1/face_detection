from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.face_bbox_iou50 import threshold_metrics, write_evaluation
from tools.compare_mal_alpha_ablation import select_optimal_alpha


def test_threshold_metrics_counts_duplicate_false_positive() -> None:
    ground_truth = {
        1: [{"annotation_id": 10, "category_id": 2, "bbox_xyxy": [0, 0, 10, 10]}],
        2: [],
    }
    predictions = [
        {"image_id": 1, "score": 0.90, "bbox_xyxy": [0, 0, 10, 10]},
        {"image_id": 1, "score": 0.80, "bbox_xyxy": [0, 0, 10, 10]},
        {"image_id": 2, "score": 0.70, "bbox_xyxy": [20, 20, 30, 30]},
    ]

    result = threshold_metrics(
        ground_truth,
        predictions,
        score_threshold=0.50,
        iou_threshold=0.50,
        num_images=2,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["duplicated_gt_count"] == 1
    assert result["duplicate_extra_prediction_count"] == 1
    assert result["duplicate_prediction_rate"] == pytest.approx(1 / 3)
    assert result["duplicate_image_rate"] == pytest.approx(0.5)
    assert result["duplicate_confidences"] == [pytest.approx(0.80)]
    assert result["duplicate_confidence_distribution"]["median"] == pytest.approx(0.80)


def test_write_evaluation_selects_threshold_on_val_and_locks_it_on_test(
    tmp_path: Path,
) -> None:
    annotations = {
        "images": [{"id": 1, "file_name": "one.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "iscrowd": 0,
            }
        ],
    }
    predictions = [
        {
            "image_id": 1,
            "category_id": 2,
            "score": 0.90,
            "bbox_xyxy": [0, 0, 10, 10],
        },
        {
            "image_id": 1,
            "category_id": 2,
            "score": 0.80,
            "bbox_xyxy": [0, 0, 10, 10],
        },
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    val_dir = tmp_path / "val"

    validation = write_evaluation(
        annotation_path=annotation_path,
        predictions_path=prediction_path,
        output_dir=val_dir,
        split="val",
        alpha=1.0,
        category_ids=[2],
    )
    assert validation["common_0_33"]["duplicate_extra_prediction_count"] == 1
    assert validation["validation_selected"]["precision"] == pytest.approx(1.0)
    assert validation["validation_selected"]["recall"] == pytest.approx(1.0)

    test = write_evaluation(
        annotation_path=annotation_path,
        predictions_path=prediction_path,
        output_dir=tmp_path / "test",
        split="test",
        alpha=1.0,
        category_ids=[2],
        validation_metrics_path=val_dir / "metrics.json",
    )
    assert test["validation_selected"]["threshold"] == pytest.approx(
        validation["validation_selected"]["threshold"]
    )


def _alpha_row(alpha: float, precision: float, recall: float, duplicates: int, rate: float):
    return {
        "alpha": alpha,
        "precision": precision,
        "recall": recall,
        "duplicate_extra_prediction_count": duplicates,
        "duplicate_prediction_rate": rate,
    }


def test_alpha_selection_requires_preserved_precision_recall_and_lower_duplication() -> None:
    selected, rows = select_optimal_alpha(
        [
            _alpha_row(1.0, 0.91, 0.92, 8, 0.010),
            _alpha_row(1.25, 0.92, 0.91, 4, 0.005),
            _alpha_row(1.5, 0.92, 0.93, 3, 0.004),
            _alpha_row(2.0, 0.90, 0.94, 2, 0.003),
        ]
    )
    assert selected == pytest.approx(1.5)
    eligibility = {float(row["alpha"]): bool(row["eligible"]) for row in rows}
    assert eligibility == {1.0: False, 1.25: False, 1.5: True, 2.0: False}

