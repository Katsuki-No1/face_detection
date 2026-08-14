from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from keypoint_comparison2.config import load_config
from keypoint_comparison2.data import CropAugmentation, FaceCropDataset, JointFaceDataset
from keypoint_comparison2.models.dinov3 import DinoV3FPNRTMCC, DinoV3ORFormerRTMCC
from keypoint_comparison2.models.joint_deim import (
    DEIMv2JointFaceLandmarks,
    EllipseConditionedLocalRefinement,
    LandmarkQueryDecoder,
)
from keypoint_comparison2.models.rtmpose import _semantic_face5_tensor
from keypoint_comparison2.models.simcc import SimCCLabel, SimCCSpec
from keypoint_comparison2.training import (
    BatchPlan,
    _set_dinov3_trainable,
    configured_batch_plans,
    select_batch_plan,
)
from src.models.dino_backbone import DummyDinoBackbone
from src.models.dinov3_deimv2.official import SpatialPriorModuleV2
from keypoint_comparison1.landmark_detection.heads import FACTORIZED_STATE_HEAD


REPO_ROOT = Path(__file__).resolve().parents[2]


def _dino_config(*, occlusion: bool = False) -> dict:
    config = {
        "input_size": 64,
        "backbone": {
            "name": "dinov3_test",
            "patch_size": 16,
            "embed_dim": 32,
            "require_pretrained": False,
            "freeze": False,
            "num_prefix_tokens": 0,
            "token_key": "x_norm_patchtokens",
            "intermediate_layers": [],
        },
        "fpn": {"out_channels": 32},
        "head": {
            "hidden_dimension": 32,
            "featuremap_size": [4, 4],
            "gau_attention_dimension": 16,
            "simcc_split_ratio": 2.0,
            "sigma": [2.0, 2.0],
        },
    }
    if occlusion:
        config["recovery"] = {
            "dimension": 32,
            "layers": 2,
            "heads": 4,
            "feedforward": 64,
        }
    return config


def _dummy_backbone() -> DummyDinoBackbone:
    return DummyDinoBackbone(
        embed_dim=32,
        patch_size=16,
        num_prefix_tokens=0,
        output_key="x_norm_patchtokens",
    )


def test_production_config_uses_dinov3_small_for_models_two_and_three() -> None:
    config = load_config(REPO_ROOT / "keypoint_comparison2/configs/production.yaml", project_root=REPO_ROOT)
    assert config.raw["train"]["amp_dtype"] == "float16"
    assert config.raw["train"]["grad_scaler_init_scale"] == 1024.0
    left = config.raw["models"]["dinov3_fpn_rtmcc"]["backbone"]
    right = config.raw["models"]["dinov3_orformer_rtmcc"]["backbone"]
    assert left["name"] == "dinov3_vits16"
    assert right["name"] == "dinov3_vits16"
    assert left["embed_dim"] == right["embed_dim"] == 384
    assert left["intermediate_layers"] == right["intermediate_layers"] == [2, 5, 8, 11]
    assert left["weights"] == right["weights"]
    assert config.raw["detector"]["variant"] == "DEIMv2-L"
    assert config.raw["data"]["rtmpose"]["category_ids"] == [1]
    assert config.raw["data"]["dino"]["category_ids"] == [2]
    assert (
        "dataset_adapters/face_pose_drive_v2_balanced_811/rtmpose"
        in config.raw["data"]["rtmpose"]["annotations"]["train"]
    )
    assert (
        "dataset_adapters/face_pose_drive_v2_balanced_811"
        in config.raw["data"]["dino"]["annotations"]["train"]
    )
    assert config.raw["data"]["dino"]["annotations"] == config.raw["data"]["joint"]["annotations"]
    assert config.raw["detector"]["config"].endswith("_balanced_811_960_ellipse_from_coco.yaml")
    assert config.raw["detector"]["checkpoint"].endswith("best_val_ellipse_map.pt")
    assert config.raw["detector"]["crop_geometry"] == "ellipse"
    assert "crop_cache" not in config.raw["detector"]
    assert config.raw["data"]["augmentation"]["scale_factor"] == 1.25
    assert (
        config.raw["data"]["augmentation"]["roi_geometry"]
        == "long_axis_square"
    )
    assert config.raw["data"]["ellipse_error_model"].endswith(
        "ellipse_deim_val_residuals.json"
    )
    calibration = config.raw["data"]["crop_error_calibration"]
    assert calibration["minimum_matched_samples"] >= 200
    assert calibration["independent_component_noise"] is False
    assert "keypoint_selection" in config.raw["data"]["dino"]["annotations"]["val"]
    assert config.output_root.name.endswith("_gt_ellipse_long_axis_square")
    joint_plans = configured_batch_plans(config.raw["models"]["deimv2_joint"])
    assert joint_plans == (
        BatchPlan(8, 2),
        BatchPlan(4, 4),
        BatchPlan(2, 8),
        BatchPlan(1, 16),
    )
    assert {plan.effective_batch_size for plan in joint_plans} == {16}


def test_deimv2_joint_batch_fallback_only_advances_on_cuda_oom() -> None:
    plans = (
        BatchPlan(8, 2),
        BatchPlan(4, 4),
        BatchPlan(2, 8),
        BatchPlan(1, 16),
    )
    probed: list[BatchPlan] = []

    def probe(plan: BatchPlan) -> dict[str, float]:
        probed.append(plan)
        if plan.batch_size > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {"peak_reserved_mib": 18_000.0}

    selected, attempts = select_batch_plan(plans, probe)
    assert selected == BatchPlan(2, 8)
    assert probed == list(plans[:3])
    assert [attempt["status"] for attempt in attempts] == [
        "cuda_oom",
        "cuda_oom",
        "selected",
    ]

    def unrelated_failure(_plan: BatchPlan) -> dict[str, float]:
        raise RuntimeError("unrelated failure")

    with pytest.raises(RuntimeError, match="unrelated failure"):
        select_batch_plan(plans, unrelated_failure)


def test_deimv2_joint_backbone_freeze_reaches_nested_detector_dino() -> None:
    model = torch.nn.Module()
    model.face_detector = torch.nn.Module()
    model.face_detector.backbone = torch.nn.Module()
    model.face_detector.backbone.dino = torch.nn.Linear(4, 4)

    assert _set_dinov3_trainable(model, False)
    assert not any(
        parameter.requires_grad
        for parameter in model.face_detector.backbone.dino.parameters()
    )
    assert _set_dinov3_trainable(model, True)
    assert all(
        parameter.requires_grad
        for parameter in model.face_detector.backbone.dino.parameters()
    )


def test_joint_dataset_keeps_deim_ellipse_targets(tmp_path: Path) -> None:
    Image.new("RGB", (100, 50), (127, 127, 127)).save(tmp_path / "face.jpg")
    annotation = {
        "images": [{"id": 1, "file_name": "face.jpg", "width": 100, "height": 50}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [20, 10, 60, 30],
                "ellipse": [50, 25, 30, 10, 0.0],
                "keypoints": [35, 20, 2, 65, 20, 1, 50, 25, 2, 40, 35, 2, 60, 35, 2],
            }
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    dataset = JointFaceDataset(
        annotation=path,
        image_root=tmp_path,
        image_size=100,
        category_ids=(2,),
        target_category_id=2,
        training=False,
    )
    _, target = dataset[0]
    assert target["ellipses"].shape == (1, 5)
    assert target["labels"].tolist() == [2]
    assert target["labeled"].tolist() == [[True, True, True, True, True]]
    assert target["visible"].tolist() == [[True, False, True, True, True]]
    # Original image is stretched vertically by 2x.
    torch.testing.assert_close(
        target["ellipses"][0],
        torch.tensor([50.0, 50.0, 30.0, 20.0, 0.0]),
        atol=1e-5,
        rtol=0.0,
    )


def test_simcc_label_round_trip() -> None:
    codec = SimCCLabel(SimCCSpec(input_size=(64, 64), split_ratio=2.0, sigma=(1.0, 1.0)))
    coordinates = torch.tensor([[[0.2, 0.3], [0.8, 0.3], [0.5, 0.5], [0.35, 0.8], [0.65, 0.8]]])
    visible = torch.ones(1, 5, dtype=torch.bool)
    label_x, label_y, _ = codec.encode(coordinates, visible)
    decoded, confidence = codec.decode(label_x.log(), label_y.log())
    torch.testing.assert_close(decoded, coordinates, atol=0.01, rtol=0.0)
    assert confidence.shape == (1, 5)


@pytest.mark.parametrize("model_class,occlusion", [(DinoV3FPNRTMCC, False), (DinoV3ORFormerRTMCC, True)])
def test_dinov3_models_forward_loss_backward(model_class, occlusion: bool) -> None:
    model = model_class(_dino_config(occlusion=occlusion), backbone_module=_dummy_backbone())
    images = torch.rand(2, 3, 64, 64)
    targets = {
        "coordinates": torch.rand(2, 5, 2),
        "visible": torch.ones(2, 5, dtype=torch.bool),
    }
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1
    output = model(images, targets, clean_images=images, occlusion_masks=mask)
    assert output["coordinates"].shape == (2, 5, 2)
    assert output["visibility_logits"].shape == (2, 5)
    assert torch.isfinite(output["loss_total"])
    output["loss_total"].backward()
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    if occlusion:
        assert output["occlusion_logits"].shape == (2, 1, 4, 4)
        assert output["loss_reconstruction"] >= 0


def test_landmark_query_decoder_generates_five_queries_per_face() -> None:
    decoder = LandmarkQueryDecoder(32, layers=2, heads=4, feedforward=64)
    face_queries = torch.randn(2, 3, 32, requires_grad=True)
    face_boxes = torch.tensor(
        [[[0.3, 0.3, 0.2, 0.2], [0.7, 0.3, 0.2, 0.2], [0.5, 0.7, 0.3, 0.3]]] * 2
    )
    memory = torch.randn(2, 20, 32)
    keypoints, presence, visibility, auxiliary = decoder(
        face_queries, face_boxes, memory
    )
    assert keypoints.shape == (2, 3, 5, 2)
    assert presence is None
    assert visibility.shape == (2, 3, 5)
    assert len(auxiliary) == 1
    assert bool(((keypoints >= 0) & (keypoints <= 1)).all())
    assert not any("presence_head" in key for key in decoder.state_dict())
    (keypoints.sum() + visibility.sum()).backward()
    assert decoder.landmark_queries.grad is not None


def test_landmark_query_decoder_factorizes_presence_and_visibility() -> None:
    decoder = LandmarkQueryDecoder(
        32,
        layers=2,
        heads=4,
        feedforward=64,
        state_head_mode=FACTORIZED_STATE_HEAD,
    )
    face_queries = torch.randn(2, 3, 32, requires_grad=True)
    face_boxes = torch.tensor(
        [[[0.3, 0.3, 0.2, 0.2], [0.7, 0.3, 0.2, 0.2], [0.5, 0.7, 0.3, 0.3]]] * 2
    )
    memory = torch.randn(2, 20, 32)
    keypoints, presence, visibility, _auxiliary = decoder(
        face_queries, face_boxes, memory
    )
    assert presence is not None
    assert presence.shape == visibility.shape == (2, 3, 5)
    state_probabilities = torch.stack(
        (
            1.0 - presence.sigmoid(),
            presence.sigmoid() * (1.0 - visibility.sigmoid()),
            presence.sigmoid() * visibility.sigmoid(),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        state_probabilities.sum(dim=-1),
        torch.ones_like(presence),
    )
    (keypoints.sum() + presence.sum() + visibility.sum()).backward()
    assert decoder.presence_head is not None
    assert any(parameter.grad is not None for parameter in decoder.presence_head.parameters())


def test_spatial_prior_exposes_existing_stride_four_feature() -> None:
    module = SpatialPriorModuleV2(inplanes=8).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        high_resolution, pyramid = module.forward_with_high_resolution(images)
        historical = module(images)
    assert high_resolution.shape == (2, 8, 16, 16)
    assert [tuple(value.shape) for value in pyramid] == [
        (2, 16, 8, 8),
        (2, 32, 4, 4),
        (2, 32, 2, 2),
    ]
    for actual, expected in zip(historical, pyramid, strict=True):
        torch.testing.assert_close(actual, expected)


def test_ellipse_local_refinement_fuses_stride_four_and_eight_features() -> None:
    refiner = EllipseConditionedLocalRefinement(
        32,
        high_resolution_channels=8,
        roi_size=8,
        roi_scale=1.25,
        layers=1,
        heads=4,
        feedforward=64,
        state_head_mode=FACTORIZED_STATE_HEAD,
        stages=[
            {
                "radius_in_feature_cells": 2.0,
                "maximum_delta": 0.08,
                "samples_per_axis": 3,
            },
            {
                "radius_in_feature_cells": 1.0,
                "maximum_delta": 0.04,
                "samples_per_axis": 3,
            },
        ],
    )
    ellipses = torch.tensor(
        [
            [
                [0.30, 0.30, 0.16, 0.10, 0.00],
                [0.70, 0.35, 0.14, 0.08, 0.25],
                [0.50, 0.70, 0.18, 0.11, -0.30],
            ]
        ]
        * 2,
        dtype=torch.float32,
    )
    offsets = torch.tensor(
        [
            [-0.04, -0.03],
            [0.04, -0.03],
            [0.00, 0.00],
            [-0.03, 0.04],
            [0.03, 0.04],
        ],
        dtype=torch.float32,
    )
    coarse = ellipses[..., None, :2] + offsets.view(1, 1, 5, 2)
    global_queries = torch.randn(2, 3, 5, 32, requires_grad=True)
    face_queries = torch.randn(2, 3, 32, requires_grad=True)
    presence = torch.randn(2, 3, 5, requires_grad=True)
    visibility = torch.randn(2, 3, 5, requires_grad=True)
    deim_feature = torch.randn(2, 32, 8, 8, requires_grad=True)
    high_resolution = torch.randn(2, 8, 16, 16, requires_grad=True)
    output = refiner(
        global_query_features=global_queries,
        face_queries=face_queries,
        coarse_coordinates=coarse,
        presence_logits=presence,
        visibility_logits=visibility,
        ellipses=ellipses,
        deim_feature=deim_feature,
        high_resolution_feature=high_resolution,
    )
    assert output["coordinates"].shape == (2, 3, 5, 2)
    assert output["presence_logits"].shape == (2, 3, 5)
    assert output["visibility_logits"].shape == (2, 3, 5)
    assert output["roi_shape"] == (8, 8)
    assert len(output["refinement_coordinates"]) == 2
    assert len(output["auxiliary_coordinates"]) == 2
    # Every residual output starts at zero, so enabling the new branch is an
    # identity mapping before training instead of a random checkpoint break.
    torch.testing.assert_close(output["coordinates"], coarse, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(output["presence_logits"], presence)
    torch.testing.assert_close(output["visibility_logits"], visibility)
    output["coordinates"].sum().backward()
    base_output = refiner.base_delta_head[-1]
    assert isinstance(base_output, torch.nn.Linear)
    assert base_output.weight.grad is not None
    for stage in refiner.refinement_stages:
        assert stage.delta_head.weight.grad is not None


def test_deim_joint_routes_global_queries_through_ellipse_local_refinement() -> None:
    class DummyBackbone(torch.nn.Module):
        spatial_high_resolution_channels = 8

        def __init__(self) -> None:
            super().__init__()
            self.f4 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1)
            self.f8 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=8, padding=1)

        def forward_with_high_resolution(self, images):
            high = self.f4(images)
            first = self.f8(images)
            return (
                first,
                torch.nn.functional.avg_pool2d(first, 2),
                torch.nn.functional.avg_pool2d(first, 4),
            ), high

        def forward(self, images):
            features, _high = self.forward_with_high_resolution(images)
            return features

    class DummyEncoder(torch.nn.Module):
        def forward(self, features):
            return features

    class DummyQueryLayer(torch.nn.Module):
        def forward(self, queries):
            return queries

    class DummyDecoder(torch.nn.Module):
        hidden_dim = 32

        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([DummyQueryLayer()])
            self.query_embedding = torch.nn.Parameter(torch.randn(3, 32))

        def forward(self, features, _targets=None):
            batch = features[0].shape[0]
            context = features[0].mean(dim=(-2, -1)).unsqueeze(1)
            queries = context + self.query_embedding.unsqueeze(0)
            queries = self.decoder.layers[-1](queries)
            boxes = queries.new_tensor(
                [
                    [0.30, 0.30, 0.24, 0.24],
                    [0.70, 0.35, 0.22, 0.22],
                    [0.50, 0.70, 0.28, 0.28],
                ]
            ).unsqueeze(0).expand(batch, -1, -1)
            ellipses = queries.new_tensor(
                [
                    [0.30, 0.30, 0.12, 0.08, 0.00],
                    [0.70, 0.35, 0.11, 0.07, 0.20],
                    [0.50, 0.70, 0.14, 0.09, -0.25],
                ]
            ).unsqueeze(0).expand(batch, -1, -1)
            return {
                "pred_logits": queries.mean(dim=-1, keepdim=True),
                "pred_boxes": boxes,
                "pred_ellipses": ellipses,
            }

    class DummyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = DummyBackbone()
            self.encoder = DummyEncoder()
            self.decoder = DummyDecoder()

    model = DEIMv2JointFaceLandmarks(
        {
            "dimension": 32,
            "maximum_face_queries": 3,
            "landmark_decoder_layers": 2,
            "heads": 4,
            "feedforward": 64,
            "state_head_mode": FACTORIZED_STATE_HEAD,
            "ellipse_local_refinement": {
                "enabled": True,
                "roi_size": 8,
                "roi_scale": 1.25,
                "layers": 1,
                "heads": 4,
                "feedforward": 64,
                "detach_ellipse": True,
            },
        },
        detector=DummyDetector(),
    )
    output = model(torch.rand(2, 3, 64, 64))
    assert output["pred_keypoints"].shape == (2, 3, 5, 2)
    assert output["global_coarse_keypoints"].shape == (2, 3, 5, 2)
    assert output["local_base_keypoints"].shape == (2, 3, 5, 2)
    assert len(output["local_refinement_keypoints"]) == 2
    assert output["local_roi_shape"] == (8, 8)
    torch.testing.assert_close(
        output["pred_keypoints"],
        output["global_coarse_keypoints"],
        atol=1.0e-6,
        rtol=0.0,
    )
    (output["pred_keypoints"].sum() + output["visibility_logits"].sum()).backward()
    assert model.landmark_decoder.landmark_queries.grad is not None
    with pytest.raises(RuntimeError, match="requires a square input"):
        model(torch.rand(1, 3, 64, 80))


def test_deim_joint_presence_loss_separates_missing_from_present() -> None:
    model = DEIMv2JointFaceLandmarks.__new__(DEIMv2JointFaceLandmarks)
    torch.nn.Module.__init__(model)
    predictions = torch.zeros(1, 1, 5, 2, requires_grad=True)
    presence_logits = torch.zeros(1, 1, 5, requires_grad=True)
    visibility_logits = torch.zeros(1, 1, 5, requires_grad=True)
    losses = model._landmark_losses(
        predictions,
        presence_logits,
        visibility_logits,
        [],
        torch.tensor([[0]]),
        [(torch.tensor([0]), torch.tensor([0]))],
        [
            {
                "keypoints": torch.zeros(1, 5, 2),
                "labeled": torch.tensor([[False, True, True, True, True]]),
                "visible": torch.tensor([[False, False, True, True, True]]),
            }
        ],
    )
    assert set(losses) == {
        "loss_keypoint",
        "loss_presence",
        "loss_visibility",
        "loss_landmark_aux",
    }
    (losses["loss_presence"] + losses["loss_visibility"]).backward()
    assert presence_logits.grad is not None
    assert presence_logits.grad[0, 0, 0] > 0  # missing: lower the presence logit
    assert presence_logits.grad[0, 0, 1] < 0  # occluded: keep the point present
    assert visibility_logits.grad is not None
    assert visibility_logits.grad[0, 0, 0] == 0  # missing is masked here
    assert visibility_logits.grad[0, 0, 1] > 0  # occluded: lower visibility only


def test_face106_semantic_transfer() -> None:
    source = torch.arange(106, dtype=torch.float32).view(106, 1)
    mapped = _semantic_face5_tensor(source).squeeze(1)
    torch.testing.assert_close(mapped, torch.tensor([70.0, 79.0, 54.0, 84.0, 90.0]))


def test_crop_dataset_returns_clean_image_and_occlusion_supervision(tmp_path: Path) -> None:
    Image.new("RGB", (80, 80), (127, 127, 127)).save(tmp_path / "face.jpg")
    annotation = {
        "images": [{"id": 1, "file_name": "face.jpg", "width": 80, "height": 80}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [15, 15, 50, 50],
                "ellipse": [40, 40, 25, 18, 0.0],
                "keypoints": [30, 30, 2, 50, 30, 2, 40, 40, 2, 32, 52, 2, 48, 52, 2],
            }
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    dataset = FaceCropDataset(
        annotation=path,
        image_root=tmp_path,
        input_size=64,
        training=False,
        category_ids=(1,),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["clean_image"].shape == (3, 64, 64)
    assert sample["occlusion_mask"].shape == (1, 64, 64)
    assert sample["coordinates"].shape == (5, 2)
    torch.testing.assert_close(
        sample["gt_ellipse"],
        torch.tensor([40.0, 40.0, 25.0, 18.0, 0.0]),
    )
    torch.testing.assert_close(sample["crop_ellipse"], sample["gt_ellipse"])
    torch.testing.assert_close(
        sample["bbox_xyxy"],
        torch.tensor([15.0, 22.0, 65.0, 58.0]),
    )
    # The perturbed/GT major diameter is 50, so the real-image source ROI is a
    # 62.5x62.5 theta-aligned square after the 1.25 context multiplier.
    affine = sample["affine"].numpy()
    assert abs(float(affine[0, 0])) == pytest.approx(63.0 / 62.5)
    assert abs(float(affine[1, 1])) == pytest.approx(63.0 / 62.5)
    torch.testing.assert_close(
        sample["crop_roi_size"],
        torch.tensor([62.5, 62.5]),
    )
    assert sample["crop_roi_is_square"] is True


def test_gt_ellipse_geometry_is_resampled_for_each_training_sample(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (100, 100), (127, 127, 127)).save(
        tmp_path / "face.jpg"
    )
    annotation = {
        "images": [
            {"id": 1, "file_name": "face.jpg", "width": 100, "height": 100}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 25, 60, 50],
                "ellipse": [50, 50, 30, 18, 0.2],
                "keypoints": [35, 40, 2, 65, 40, 2, 50, 50, 2, 40, 62, 2, 60, 62, 2],
            }
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    dataset = FaceCropDataset(
        annotation=path,
        image_root=tmp_path,
        input_size=64,
        training=True,
        category_ids=(1,),
        augmentation=CropAugmentation(
            scale_factor=1.25,
            scale_jitter=0.20,
            center_jitter=0.08,
            rotation_deg=25.0,
            horizontal_flip_probability=0.0,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            blur_probability=0.0,
            occlusion_probability=0.0,
        ),
    )
    first = dataset[0]
    second = dataset[0]
    torch.testing.assert_close(first["gt_ellipse"], second["gt_ellipse"])
    assert not torch.allclose(first["crop_ellipse"], second["crop_ellipse"])
    assert bool(torch.isfinite(first["coordinates"]).all())
    crop_points = first["coordinates"].numpy() * 63.0
    homogeneous = np.concatenate(
        (crop_points, np.ones((5, 1), dtype=np.float32)),
        axis=1,
    )
    recovered = homogeneous @ first["inverse_affine"].numpy().T
    np.testing.assert_allclose(
        recovered,
        np.asarray(annotation["annotations"][0]["keypoints"], dtype=np.float32)
        .reshape(5, 3)[:, :2],
        atol=2.0e-4,
    )


def test_crop_dataset_uses_long_axis_square_detector_roi(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (80, 80), (127, 127, 127)).save(
        tmp_path / "face.jpg"
    )
    annotation = {
        "images": [
            {"id": 1, "file_name": "face.jpg", "width": 80, "height": 80}
        ],
        "annotations": [
            {
                "id": 7,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 25, 40, 30],
                "keypoints": [
                    30, 32, 2, 50, 32, 2, 40, 40, 2,
                    32, 48, 2, 48, 48, 2,
                ],
            }
        ],
    }
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    cache_path = tmp_path / "crop_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {"crop_geometry": "ellipse_aligned"},
                "predictions": {
                    "1": [
                        {
                            "bbox_xyxy": [20, 25, 60, 55],
                            "detector_bbox_xyxy": [20, 25, 60, 55],
                            "ellipse": [40, 40, 20, 10, 0.0],
                            "score": 0.95,
                            "crop_geometry": "ellipse",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = FaceCropDataset(
        annotation=annotation_path,
        image_root=tmp_path,
        input_size=64,
        training=False,
        category_ids=(1,),
        augmentation=CropAugmentation(scale_factor=1.0),
        detector_crop_cache=cache_path,
        detector_matching_iou=0.5,
    )
    sample = dataset[0]
    torch.testing.assert_close(
        sample["bbox_xyxy"],
        torch.tensor([20.0, 30.0, 60.0, 50.0]),
    )
    assert sample["crop_rotation_degrees"] == pytest.approx(0.0)
    assert sample["detector_matching_iou"] == pytest.approx(1.0)
    # The 2:1 predicted ellipse defines a 40x40 long-axis square. The affine
    # is isotropic and the short-axis surplus comes from the real image.
    affine = sample["affine"].numpy()
    assert abs(float(affine[0, 0])) == pytest.approx(
        abs(float(affine[1, 1]))
    )
    torch.testing.assert_close(
        sample["crop_roi_size"],
        torch.tensor([40.0, 40.0]),
    )
    assert sample["crop_roi_is_square"] is True


def test_synthetic_occlusion_updates_landmark_visibility(tmp_path: Path) -> None:
    Image.new("RGB", (80, 80), (127, 127, 127)).save(tmp_path / "face.jpg")
    annotation = {
        "images": [{"id": 1, "file_name": "face.jpg", "width": 80, "height": 80}],
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
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    dataset = FaceCropDataset(
        annotation=path,
        image_root=tmp_path,
        input_size=64,
        training=True,
        category_ids=(1,),
        augmentation=CropAugmentation(
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
            occlusion_count=(1, 1),
        ),
    )
    sample = dataset[0]
    assert bool(sample["labeled"].all())
    assert not bool(sample["visible"].any())
    assert bool(sample["occlusion_mask"].any())


@pytest.mark.model_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local RTMPose weights and CUDA")
def test_real_rtmpose_pretrained_transfer_and_backward() -> None:
    from keypoint_comparison2.models import build_model

    config = load_config(REPO_ROOT / "keypoint_comparison2/configs/production.yaml", project_root=REPO_ROOT)
    model = build_model("rtmpose_face5", config.raw).cuda().train()
    output = model(
        torch.rand(1, 3, 256, 256, device="cuda"),
        {
            "coordinates": torch.rand(1, 5, 2, device="cuda"),
            "visible": torch.ones(1, 5, dtype=torch.bool, device="cuda"),
        },
    )
    output["loss_total"].backward()
    assert model.pretrained_report["loaded_backbone"] == model.pretrained_report["total_backbone"]


@pytest.mark.model_weights
@pytest.mark.parametrize("model_name", ["dinov3_fpn_rtmcc", "dinov3_orformer_rtmcc"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local DINOv3-S weights and CUDA")
def test_real_dinov3_small_forward_and_backward(model_name: str) -> None:
    from keypoint_comparison2.models import build_model

    config = load_config(REPO_ROOT / "keypoint_comparison2/configs/production.yaml", project_root=REPO_ROOT)
    weights = Path(config.raw["models"][model_name]["backbone"]["weights"])
    if not weights.is_file():
        pytest.skip(f"missing {weights}")
    model = build_model(model_name, config.raw).cuda().train()
    images = torch.rand(1, 3, 320, 320, device="cuda")
    mask = torch.zeros(1, 1, 320, 320, device="cuda")
    mask[:, :, 96:192, 96:192] = 1
    output = model(
        images,
        {
            "coordinates": torch.rand(1, 5, 2, device="cuda"),
            "visible": torch.ones(1, 5, dtype=torch.bool, device="cuda"),
            "labeled": torch.ones(1, 5, dtype=torch.bool, device="cuda"),
        },
        clean_images=images,
        occlusion_masks=mask,
    )
    output["loss_total"].backward()
    assert output["coordinates"].shape == (1, 5, 2)
    assert model.pretrained_report["architecture"] == "dinov3_vits16"


@pytest.mark.model_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local DEIMv2-L weights and CUDA")
def test_real_deimv2_joint_queries_forward_and_backward() -> None:
    from keypoint_comparison2.data import JointFaceDataset
    from keypoint_comparison2.models import build_model

    config = load_config(REPO_ROOT / "keypoint_comparison2/configs/production.yaml", project_root=REPO_ROOT)
    detector_checkpoint = Path(config.raw["detector"]["checkpoint"])
    if not detector_checkpoint.is_file():
        pytest.skip(f"missing {detector_checkpoint}")
    data = config.raw["data"]["joint"]
    dataset = JointFaceDataset(
        annotation=data["annotations"]["train"],
        image_root=data["image_root"],
        image_size=960,
        category_ids=data["category_ids"],
        target_category_id=data["target_category_id"],
    )
    image, target = dataset[0]
    model = build_model("deimv2_joint", config.raw).cuda().train()
    output = model(
        image.unsqueeze(0).cuda(),
        [{key: value.cuda() for key, value in target.items()}],
    )
    assert output["pred_keypoints"].shape == (1, 100, 5, 2)
    assert output["visibility_logits"].shape == (1, 100, 5)
    output["loss_total"].backward()
    assert any(parameter.grad is not None for parameter in model.landmark_decoder.parameters())
