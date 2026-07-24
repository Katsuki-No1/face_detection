from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.landmark_detection.evaluation import evaluate_deim_boxes, load_or_create_detector_predictions
from src.landmark_detection.geometry import (
    build_crop_transform,
    normalized_crop_to_original,
    original_to_normalized_crop,
)
from src.landmark_detection.losses import LandmarkLoss
from src.landmark_detection.metrics import LandmarkMetricAccumulator, match_detector_boxes
from src.landmark_detection.model import build_landmark_model, initialize_dino_from_deim_checkpoint
from src.landmark_detection.pipeline import EXPERIMENT_CONDITIONS, build_comparison
from src.landmark_detection.training import _optimizer
from src.models.dino_backbone import DinoBackbone, DummyDinoBackbone


def _model_config() -> dict:
    return {
        "model": {
            "num_landmarks": 5,
            "backbone": {
                "name": "dummy",
                "patch_size": 16,
                "embed_dim": 32,
                "dummy": True,
                "token_key": "x_norm_patchtokens",
                "num_prefix_tokens": 0,
                "freeze": False,
            },
            "variants": {
                "vitpose": {"hidden_channels": 16, "deconv_layers": 2},
                "query": {
                    "dimension": 32,
                    "decoder_layers": 2,
                    "heads": 4,
                    "feedforward": 64,
                    "auxiliary_supervision": True,
                },
                "query_refinement": {
                    "dimension": 32,
                    "decoder_layers": 2,
                    "heads": 4,
                    "feedforward": 64,
                    "auxiliary_supervision": True,
                    "stages": [
                        {"radius_in_feature_cells": 2.0, "maximum_delta": 0.08, "samples_per_axis": 3},
                        {"radius_in_feature_cells": 1.0, "maximum_delta": 0.04, "samples_per_axis": 3},
                    ],
                },
            },
        }
    }


def _targets(batch: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "keypoints": torch.tensor(
                [[0.30, 0.35], [0.70, 0.35], [0.50, 0.55], [0.38, 0.75], [0.62, 0.75]],
                dtype=torch.float32,
            ),
            "visible": torch.ones(5, dtype=torch.bool),
        }
        for _ in range(batch)
    ]


def test_crop_coordinate_round_trip() -> None:
    transform = build_crop_transform([20.0, 30.0, 120.0, 170.0], output_size=256, scale_factor=1.3, rotation_deg=13.0)
    points = np.asarray([[25.0, 40.0], [75.0, 88.0], [118.0, 160.0]], dtype=np.float32)
    normalized = original_to_normalized_crop(points, transform)
    recovered = normalized_crop_to_original(normalized, transform)
    np.testing.assert_allclose(recovered, points, atol=2e-4)


@pytest.mark.parametrize("variant", ["vitpose", "query", "query_refinement"])
def test_landmark_heads_forward_loss_and_backward(variant: str) -> None:
    model, initialization = build_landmark_model(
        _model_config(), variant=variant, initialize_from_detector=False
    )
    assert initialization is None
    images = torch.rand(2, 3, 64, 64)
    output = model(images, return_attention=True)
    assert output["coordinates"].shape == (2, 5, 2)
    assert output["decoded_coordinates"].shape == (2, 5, 2)
    assert output["visibility_logits"].shape == (2, 5)
    assert bool(torch.all((output["decoded_coordinates"] >= 0) & (output["decoded_coordinates"] <= 1)))
    if variant == "vitpose":
        assert output["heatmaps"].shape == (2, 5, 16, 16)
    elif variant == "query":
        assert len(output["self_attention"]) == 2
    else:
        assert len(output["refinement_coordinates"]) == 2
        assert output["base_coordinates"].shape == (2, 5, 2)
    losses = LandmarkLoss()(output, _targets(2))
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    assert any(parameter.grad is not None for parameter in model.head.parameters())


def test_strict_dino_extraction_from_deim_ema(tmp_path: Path) -> None:
    source = DummyDinoBackbone(
        embed_dim=16,
        patch_size=16,
        num_prefix_tokens=0,
        output_key="x_norm_patchtokens",
    )
    state = {
        f"backbone.dino.backbone.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
    }
    path = tmp_path / "deim.pt"
    torch.save({"ema": state}, path)
    target_raw = DummyDinoBackbone(
        embed_dim=16,
        patch_size=16,
        num_prefix_tokens=0,
        output_key="x_norm_patchtokens",
    )
    target = DinoBackbone(
        name="dummy",
        patch_size=16,
        embed_dim=16,
        backbone=target_raw,
        token_key="x_norm_patchtokens",
        num_prefix_tokens=0,
        freeze=False,
    )
    report = initialize_dino_from_deim_checkpoint(target, path, strict=True)
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["loaded_tensors"] == len(source.state_dict())


def test_optimizer_covers_parameters_once_and_excludes_tokens_from_decay() -> None:
    model, _ = build_landmark_model(
        _model_config(), variant="query", initialize_from_detector=False
    )
    optimizer = _optimizer(
        model,
        {
            "lr_backbone": 1e-5,
            "lr_head": 5e-4,
            "weight_decay": 0.05,
            "fused_optimizer": False,
        },
    )
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in optimized}) == len(optimized)
    assert {id(parameter) for parameter in optimized} == {id(parameter) for parameter in model.parameters()}
    query_parameter = model.head.landmark_queries
    query_group = next(group for group in optimizer.param_groups if any(parameter is query_parameter for parameter in group["params"]))
    assert query_group["weight_decay"] == 0.0


def test_detector_matching_is_one_to_one() -> None:
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    detections = [
        {"bbox_xyxy": [0, 0, 10, 10]},
        {"bbox_xyxy": [1, 1, 9, 9]},
        {"bbox_xyxy": [20, 20, 30, 30]},
    ]
    matches, missed, false_positives = match_detector_boxes(gt, detections, iou_threshold=0.5)
    assert [(left, right) for left, right, _iou in matches] == [(0, 0), (1, 2)]
    assert missed == []
    assert false_positives == [1]


def test_metrics_count_detector_miss_in_end_to_end_pck() -> None:
    accumulator = LandmarkMetricAccumulator()
    points = np.asarray([[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]], dtype=np.float32)
    visible = np.ones(5, dtype=bool)
    accumulator.add_prediction(
        predicted_keypoints=points,
        gt_keypoints=points,
        visible=visible,
        bbox_xyxy=[0, 0, 4, 4],
        image_id=1,
        annotation_id=1,
    )
    accumulator.add_missed(
        gt_keypoints=points,
        visible=visible,
        bbox_xyxy=[0, 0, 4, 4],
        image_id=2,
        annotation_id=2,
    )
    metrics = accumulator.compute()
    assert metrics["pck_0.05"] == 1.0
    assert metrics["keypoint_coverage"] == 0.5
    assert metrics["end_to_end_pck_0.05"] == 0.5
    assert metrics["detector_operating_point"]["recall"] == 0.5


def test_deim_evaluation_ignores_faces_without_landmark_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), (127, 127, 127)).save(image_path)
    keypoints = [20, 20, 2, 28, 20, 2, 24, 24, 2, 21, 28, 2, 27, 28, 2]
    annotation_path = tmp_path / "test.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 64, "height": 64}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [16, 16, 16, 16], "keypoints": keypoints},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [40, 40, 16, 16], "keypoints": [0] * 15},
                ],
                "categories": [{"id": 1, "name": "face"}],
            }
        ),
        encoding="utf-8",
    )

    class CenterModel(nn.Module):
        def forward(self, images: torch.Tensor, *, return_attention: bool = False) -> dict:
            del return_attention
            batch = images.shape[0]
            coordinates = torch.full((batch, 5, 2), 0.5, device=images.device)
            return {
                "coordinates": coordinates,
                "decoded_coordinates": coordinates,
                "visibility_logits": torch.zeros(batch, 5, device=images.device),
                "self_attention": [],
                "cross_attention": [],
            }

    detector_predictions = {
        1: [
            {"bbox_xyxy": [16, 16, 32, 32], "score": 0.9, "label": 0},
            {"bbox_xyxy": [40, 40, 56, 56], "score": 0.9, "label": 0},
        ]
    }
    metrics, rows = evaluate_deim_boxes(
        CenterModel(),
        annotation_path=annotation_path,
        image_root=tmp_path,
        detector_predictions=detector_predictions,
        device=torch.device("cpu"),
        batch_size=2,
        input_size=64,
        crop_scale_factor=1.0,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        matching_iou_threshold=0.5,
        category_ids=(1,),
    )
    assert metrics["instances"] == 1
    assert metrics["detector_false_positives"] == 0
    assert len(rows) == 1


def test_detector_cache_is_invalidated_by_runtime_settings(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16), (0, 0, 0)).save(tmp_path / "frame.jpg")
    annotation_path = tmp_path / "test.json"
    annotation_path.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "frame.jpg"}], "annotations": []}),
        encoding="utf-8",
    )
    config_path = tmp_path / "detector.yaml"
    checkpoint_path = tmp_path / "detector.pt"
    config_path.write_text("model: test\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeDetector:
        def __init__(self) -> None:
            self.config_path = config_path
            self.checkpoint_path = checkpoint_path
            self.score_threshold = 0.3
            self.image_size = 960
            self.maximum_detections = 100
            self.calls = 0

        def predict(self, _image: Image.Image) -> list[dict]:
            self.calls += 1
            return []

    detector = FakeDetector()
    cache = tmp_path / "cache.json"
    load_or_create_detector_predictions(
        detector=detector,  # type: ignore[arg-type]
        annotation_path=annotation_path,
        image_root=tmp_path,
        cache_path=cache,
    )
    load_or_create_detector_predictions(
        detector=detector,  # type: ignore[arg-type]
        annotation_path=annotation_path,
        image_root=tmp_path,
        cache_path=cache,
    )
    assert detector.calls == 1
    detector.score_threshold = 0.4
    load_or_create_detector_predictions(
        detector=detector,  # type: ignore[arg-type]
        annotation_path=annotation_path,
        image_root=tmp_path,
        cache_path=cache,
    )
    assert detector.calls == 2


def test_exact_a_to_e_condition_matrix_and_comparison() -> None:
    assert [(item.condition, item.bbox_mode, item.variant) for item in EXPERIMENT_CONDITIONS] == [
        ("A", "gt", "vitpose"),
        ("B", "gt", "query"),
        ("C", "deim", "vitpose"),
        ("D", "deim", "query"),
        ("E", "deim", "query_refinement"),
    ]
    results = {}
    for index, condition in enumerate("ABCDE"):
        results[condition] = {
            "instances": 2,
            "missed_instances": 0,
            "keypoint_coverage": 1.0,
            "nme": 0.05 - index * 0.002,
            "pck_0.03": 0.4,
            "pck_0.05": 0.7 + index * 0.01,
            "pck_0.10": 0.9,
            "end_to_end_pck_0.05": 0.7 + index * 0.01,
            "end_to_end_pck_0.10": 0.9,
        }
    results["C"]["paired_oracle_reference"] = {"nme": 0.05, "pck_0.05": 0.7, "pck_0.10": 0.9}
    results["D"]["paired_oracle_reference"] = {"nme": 0.048, "pck_0.05": 0.71, "pck_0.10": 0.9}
    results["E"]["paired_oracle_reference"] = {"nme": 0.046, "pck_0.05": 0.72, "pck_0.10": 0.9}
    comparison = build_comparison("fixed_test", results)
    assert len(comparison["conditions"]) == 5
    assert comparison["deltas"]["E_minus_D_refinement_gain"]["nme"] == pytest.approx(-0.002)


def test_config_example_is_valid_json_serializable() -> None:
    # Guards the lightweight test config itself against tensors/non-serializable values.
    json.dumps(_model_config())
