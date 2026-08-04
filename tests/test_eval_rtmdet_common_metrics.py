from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.eval_rtmdet_with_dinov3_deimv2_metrics import prediction_rows


def test_prediction_rows_selects_requested_model_label() -> None:
    result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            bboxes=torch.tensor([[1.0, 2.0, 11.0, 12.0], [20.0, 30.0, 50.0, 70.0]]),
            scores=torch.tensor([0.95, 0.90]),
            labels=torch.tensor([0, 1]),
        )
    )

    rows = prediction_rows(
        {"id": 7, "file_name": "frame.jpg"},
        result,
        category_id=2,
        category_name="face",
        model_label_id=1,
        score_min=0.01,
        max_det=100,
    )

    assert len(rows) == 1
    assert rows[0]["category_name"] == "face"
    assert rows[0]["bbox_xyxy"] == [20.0, 30.0, 50.0, 70.0]
