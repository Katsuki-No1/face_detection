from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from keypoint_comparison1.landmark_detection.dataset import (
    CropAugmentation,
    TopDownFaceLandmarkDataset,
    _parse_keypoints,
)
from keypoint_comparison1.landmark_detection.evaluation import evaluate_deim_boxes, load_or_create_detector_predictions
from keypoint_comparison1.landmark_detection.config import (
    KEYPOINT_TARGET_POLICY,
    LandmarkExperimentConfig,
    load_landmark_config,
)
from keypoint_comparison1.landmark_detection.geometry import (
    build_crop_transform,
    normalized_crop_to_original,
    original_to_normalized_crop,
)
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.metrics import LandmarkMetricAccumulator, match_detector_boxes
from keypoint_comparison1.landmark_detection.model import (
    VARIANTS,
    VITPOSE_VARIANTS,
    build_landmark_model,
    initialize_dino_from_deim_checkpoint,
)
from keypoint_comparison1.landmark_detection.heads import (
    FACTORIZED_STATE_HEAD,
    LandmarkQueryHead,
    ViTPoseHeatmapHead,
)
from keypoint_comparison1.landmark_detection.pipeline import (
    EXPERIMENT_CONDITIONS,
    build_comparison,
    build_subset_comparison,
    evaluate_conditions,
)
from keypoint_comparison1.landmark_detection.training import (
    _optimizer,
    initialize_refinement_from_query,
)
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
                "vitpose_376": {"hidden_channels": 20, "deconv_layers": 2},
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


def test_evaluation_can_defer_until_all_scheduled_heads_exist(
    tmp_path: Path,
) -> None:
    config = LandmarkExperimentConfig(
        path=tmp_path / "config.yaml",
        project_root=tmp_path,
        raw={
            "data": {"evaluation_sets": {"v2_test": {}}},
            "evaluation": {"defer_until_all_checkpoints": True},
            "outputs": {"root": str(tmp_path / "outputs")},
        },
    )
    result = evaluate_conditions(
        config,
        evaluation_sets=("v2_test",),
        device=torch.device("cpu"),
    )
    assert result["status"] == "deferred"
    assert len(result["missing_checkpoints"]) == len(VARIANTS)
    status = json.loads(
        (config.output_root / "evaluation/status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "deferred"


def test_query_vs_vitpose376_subset_comparison() -> None:
    results = {
        "B": {"nme": 0.02, "pck_0.05": 0.90},
        "D": {
            "nme": 0.03,
            "pck_0.05": 0.88,
            "end_to_end_pck_0.05": 0.82,
        },
        "F": {"nme": 0.04, "pck_0.05": 0.84},
        "G": {
            "nme": 0.05,
            "pck_0.05": 0.81,
            "end_to_end_pck_0.05": 0.76,
        },
    }
    comparison = build_subset_comparison("v2_test", results)
    assert [row["condition"] for row in comparison["conditions"]] == [
        "B",
        "D",
        "F",
        "G",
    ]
    assert comparison["deltas"][
        "B_minus_F_query_vs_vitpose376_gt"
    ]["nme"] == pytest.approx(-0.02)
    assert comparison["deltas"][
        "D_minus_G_query_vs_vitpose376_deim"
    ]["end_to_end_pck_0.05"] == pytest.approx(0.06)


def _targets(batch: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "keypoints": torch.tensor(
                [[0.30, 0.35], [0.70, 0.35], [0.50, 0.55], [0.38, 0.75], [0.62, 0.75]],
                dtype=torch.float32,
            ),
            "labeled": torch.ones(5, dtype=torch.bool),
            "visible": torch.ones(5, dtype=torch.bool),
        }
        for _ in range(batch)
    ]


def test_coco_visibility_is_split_into_labeled_and_visible_masks() -> None:
    points, labeled, visible = _parse_keypoints(
        [
            10,
            20,
            2,
            30,
            40,
            1,
            0,
            0,
            0,
            50,
            60,
            2,
            70,
            80,
            1,
        ]
    )
    np.testing.assert_array_equal(
        points,
        np.asarray(
            [[10, 20], [30, 40], [0, 0], [50, 60], [70, 80]],
            dtype=np.float32,
        ),
    )
    assert labeled.tolist() == [True, True, False, True, True]
    assert visible.tolist() == [True, False, False, True, False]


def test_landmark_loss_uses_labeled_for_coordinates_and_visible_for_bce() -> None:
    output = {
        "coordinates": torch.zeros((1, 5, 2), requires_grad=True),
        "visibility_logits": torch.tensor(
            [[0.0, 0.0, 100.0, 100.0, 100.0]],
            requires_grad=True,
        ),
    }
    target_coordinates = torch.zeros((5, 2))
    target_coordinates[1, 0] = 0.5
    targets = [
        {
            "keypoints": target_coordinates,
            "labeled": torch.tensor([True, True, False, False, False]),
            "visible": torch.tensor([True, False, False, False, False]),
        }
    ]
    losses = LandmarkLoss()(output, targets)
    assert losses["loss_coordinates"] > 0
    assert float(losses["loss_visibility"].detach()) == pytest.approx(
        float(np.log(2.0)),
    )
    losses["loss_total"].backward()
    assert output["coordinates"].grad is not None
    assert output["visibility_logits"].grad is not None
    assert output["visibility_logits"].grad[0, 2:].abs().sum() == 0


def test_factorized_loss_supervises_missing_presence_but_not_visibility() -> None:
    output = {
        "coordinates": torch.zeros((1, 5, 2), requires_grad=True),
        "presence_logits": torch.zeros((1, 5), requires_grad=True),
        "visibility_logits": torch.zeros((1, 5), requires_grad=True),
    }
    targets = [
        {
            "keypoints": torch.zeros((5, 2)),
            "labeled": torch.tensor([True, True, False, False, False]),
            "visible": torch.tensor([True, False, False, False, False]),
        }
    ]
    losses = LandmarkLoss()(output, targets)
    assert float(losses["loss_presence"].detach()) == pytest.approx(
        float(np.log(2.0))
    )
    assert float(losses["loss_visibility"].detach()) == pytest.approx(
        float(np.log(2.0))
    )
    losses["loss_total"].backward()
    assert bool((output["presence_logits"].grad != 0).all())
    assert output["visibility_logits"].grad[0, :2].abs().sum() > 0
    assert output["visibility_logits"].grad[0, 2:].abs().sum() == 0


def test_synthetic_occlusion_keeps_labels_and_clears_visibility(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (80, 80), (127, 127, 127)).save(
        tmp_path / "face.jpg"
    )
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "face.jpg",
                        "width": 80,
                        "height": 80,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [15, 15, 50, 50],
                        "ellipse": [40, 40, 25, 18, 0.0],
                        "keypoints": [40, 40, 2] * 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = TopDownFaceLandmarkDataset(
        annotation_path=annotation_path,
        image_root=tmp_path,
        input_size=64,
        training=True,
        category_ids=(1,),
        augmentation=CropAugmentation(
            scale_factor=1.0,
            scale_jitter=0.0,
            center_jitter=0.0,
            rotation_deg=0.0,
            horizontal_flip_probability=0.0,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            blur_probability=0.0,
            occlusion_probability=1.0,
            occlusion_area=(1.0, 1.0),
        ),
    )
    _image, target = dataset[0]
    assert bool(target["labeled"].all())
    assert not bool(target["visible"].any())
    assert bool(target["occlusion_mask"].any())


def test_crop_coordinate_round_trip() -> None:
    transform = build_crop_transform([20.0, 30.0, 120.0, 170.0], output_size=256, scale_factor=1.3, rotation_deg=13.0)
    points = np.asarray([[25.0, 40.0], [75.0, 88.0], [118.0, 160.0]], dtype=np.float32)
    normalized = original_to_normalized_crop(points, transform)
    recovered = normalized_crop_to_original(normalized, transform)
    np.testing.assert_allclose(recovered, points, atol=2e-4)


@pytest.mark.parametrize("variant", VARIANTS)
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
    if variant in VITPOSE_VARIANTS:
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


def test_query_refinement_factorizes_presence_and_conditional_visibility() -> None:
    config = _model_config()
    config["model"]["state_head_mode"] = FACTORIZED_STATE_HEAD
    model, _ = build_landmark_model(
        config,
        variant="query_refinement",
        initialize_from_detector=False,
    )
    output = model(torch.rand(2, 3, 64, 64))
    assert output["presence_logits"].shape == (2, 5)
    assert output["visibility_logits"].shape == (2, 5)
    assert output["state_probabilities"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["state_probabilities"].sum(dim=-1),
        torch.ones((2, 5)),
    )
    losses = LandmarkLoss()(output, _targets(2))
    losses["loss_total"].backward()
    assert all(
        stage.presence_delta.weight.grad is not None
        for stage in model.head.refinement_stages
    )


def test_legacy_query_checkpoint_contract_has_no_presence_parameters() -> None:
    model, _ = build_landmark_model(
        _model_config(),
        variant="query_refinement",
        initialize_from_detector=False,
    )
    assert not any(
        "presence" in key
        for key in model.state_dict()
    )
    output = model(torch.rand(1, 3, 64, 64))
    assert "presence_logits" not in output
    assert "state_probabilities" not in output


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
        labeled=visible,
        visible=visible,
        bbox_xyxy=[0, 0, 4, 4],
        image_id=1,
        annotation_id=1,
    )
    accumulator.add_missed(
        gt_keypoints=points,
        labeled=visible,
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


def test_metrics_separate_coordinate_labels_from_visibility_targets() -> None:
    accumulator = LandmarkMetricAccumulator()
    points = np.asarray(
        [[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]],
        dtype=np.float32,
    )
    labeled = np.asarray([True, True, True, True, False])
    visible = np.asarray([True, False, True, False, False])
    accumulator.add_prediction(
        predicted_keypoints=points,
        gt_keypoints=points,
        labeled=labeled,
        visible=visible,
        bbox_xyxy=[0, 0, 4, 4],
        image_id=1,
        annotation_id=1,
        landmark_scores=np.asarray([0.9, 0.8, 0.1, 0.2, 0.9]),
    )
    metrics = accumulator.compute()
    assert metrics["labeled_keypoints"] == 4
    assert metrics["visible_keypoints"] == 2
    assert metrics["occluded_keypoints"] == 2
    assert metrics["evaluated_keypoints"] == 4
    assert metrics["natural_visible_points"] == 2
    assert metrics["natural_occluded_points"] == 2
    assert metrics["visibility_confusion_matrix"]["values"] == [
        [1, 1],
        [1, 1],
    ]
    assert metrics["visibility_occluded_macro_f1"] == pytest.approx(0.5)


def test_metrics_report_presence_and_three_class_state() -> None:
    accumulator = LandmarkMetricAccumulator()
    points = np.asarray(
        [[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]],
        dtype=np.float32,
    )
    labeled = np.asarray([False, True, True, False, True])
    visible = np.asarray([False, False, True, False, True])
    state_probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    accumulator.add_prediction(
        predicted_keypoints=points,
        gt_keypoints=points,
        labeled=labeled,
        visible=visible,
        bbox_xyxy=[0, 0, 4, 4],
        image_id=1,
        annotation_id=1,
        landmark_scores=np.asarray([0.1, 0.1, 0.9, 0.2, 0.8]),
        presence_scores=np.asarray([0.1, 0.9, 0.9, 0.1, 0.9]),
        state_probabilities=state_probabilities,
    )
    metrics = accumulator.compute()
    assert metrics["presence_missing_macro_f1"] == pytest.approx(1.0)
    assert metrics["state_accuracy"] == pytest.approx(1.0)
    assert metrics["state_macro_f1"] == pytest.approx(1.0)
    assert metrics["state_confusion_matrix"]["values"] == [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]


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

    class FactorizedCenterModel(CenterModel):
        def forward(self, images: torch.Tensor, *, return_attention: bool = False) -> dict:
            output = super().forward(
                images,
                return_attention=return_attention,
            )
            batch = images.shape[0]
            output["presence_logits"] = torch.zeros(
                batch,
                5,
                device=images.device,
            )
            output["state_probabilities"] = torch.full(
                (batch, 5, 3),
                1.0 / 3.0,
                device=images.device,
            )
            return output

    factorized_metrics, factorized_rows = evaluate_deim_boxes(
        FactorizedCenterModel(),
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
        include_missing_keypoint_instances=True,
    )
    assert factorized_metrics["instances"] == 2
    assert factorized_metrics["presence_evaluated_keypoints"] == 10
    assert len(factorized_rows) == 2


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


def test_parameter_matched_vitpose_head_matches_query_head_size() -> None:
    vitpose = ViTPoseHeatmapHead(
        384,
        num_landmarks=5,
        hidden_channels=376,
        deconv_layers=2,
        deconv_kernel=4,
        state_head_mode=FACTORIZED_STATE_HEAD,
    )
    query = LandmarkQueryHead(
        384,
        num_landmarks=5,
        dimension=256,
        decoder_layers=4,
        heads=8,
        feedforward=1024,
        state_head_mode=FACTORIZED_STATE_HEAD,
    )
    vitpose_parameters = sum(parameter.numel() for parameter in vitpose.parameters())
    query_parameters = sum(parameter.numel() for parameter in query.parameters())
    assert vitpose_parameters == 4_579_399
    assert query_parameters == 4_579_338
    assert abs(query_parameters - vitpose_parameters) == 61


def test_exact_a_to_g_condition_matrix_and_comparison() -> None:
    assert [(item.condition, item.bbox_mode, item.variant) for item in EXPERIMENT_CONDITIONS] == [
        ("A", "gt", "vitpose"),
        ("B", "gt", "query"),
        ("C", "deim", "vitpose"),
        ("D", "deim", "query"),
        ("E", "deim", "query_refinement"),
        ("F", "gt", "vitpose_376"),
        ("G", "deim", "vitpose_376"),
    ]
    results = {}
    for index, condition in enumerate("ABCDEFG"):
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
    results["G"]["paired_oracle_reference"] = {"nme": 0.04, "pck_0.05": 0.75, "pck_0.10": 0.9}
    comparison = build_comparison("fixed_test", results)
    assert len(comparison["conditions"]) == 7
    assert comparison["deltas"]["E_minus_D_refinement_gain"]["nme"] == pytest.approx(-0.002)
    assert comparison["deltas"]["B_minus_F_query_vs_parameter_matched_vitpose_oracle_boxes"][
        "nme"
    ] == pytest.approx(0.008)


def test_config_example_is_valid_json_serializable() -> None:
    # Guards the lightweight test config itself against tensors/non-serializable values.
    json.dumps(_model_config())


def test_production_config_pins_factorized_state_target_policy() -> None:
    config = load_landmark_config(
        Path(
            "keypoint_comparison1/configs/"
            "dinov3_vits16_face_pose_drive_v2.yaml"
        )
    )
    assert config.raw["data"]["keypoint_target_policy"] == (
        KEYPOINT_TARGET_POLICY
    )
    assert config.raw["model"]["state_head_mode"] == FACTORIZED_STATE_HEAD
    assert config.raw["data"]["include_missing_keypoint_instances"] is True
    assert config.raw["experiment_name"].endswith("_factorized_state")
    assert config.output_root.name.endswith("_coco_visibility")


def test_refinement_rejects_legacy_visibility_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy_query.pt"
    torch.save({"model": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="legacy collapsed visibility"):
        initialize_refinement_from_query(nn.Linear(1, 1), checkpoint)
