from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from keypoint_comparison1.landmark_detection.config import (
    TWO_STAGE_FACE_KEYPOINT_PIPELINE,
    configured_evaluation_pipeline_stages,
    load_landmark_config,
)
from keypoint_comparison1.landmark_detection.dataset import (
    CropAugmentation,
    TopDownFaceLandmarkDataset,
)
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss, MessengerRecoveryLoss
from keypoint_comparison1.landmark_detection.messenger_recovery import (
    MessengerRecoveryBlock,
    patch_occlusion_ratios,
)
from keypoint_comparison1.landmark_detection.model import build_landmark_model


def _messenger_config(*, enabled: bool) -> dict:
    return {
        "model": {
            "num_landmarks": 5,
            "backbone": {
                "name": "dummy",
                "patch_size": 16,
                "embed_dim": 24,
                "dummy": True,
                "token_key": "x_norm_patchtokens",
                "num_prefix_tokens": 0,
                "freeze": False,
            },
            "messenger_recovery": {
                "enabled": enabled,
                "layers": 1,
                "heads": 6,
                "feedforward_ratio": 4.0,
                "dropout": 0.0,
                "positional_encoding": "sine_2d",
                "attention_scope": "full_face_crop",
                "self_patch_mask": True,
                "gate": "soft_scalar",
                "initial_gate": 0.10,
                "zero_init_projection": True,
            },
            "variants": {
                "query": {
                    "dimension": 24,
                    "decoder_layers": 1,
                    "heads": 6,
                    "feedforward": 48,
                    "auxiliary_supervision": True,
                },
                "query_refinement": {
                    "dimension": 24,
                    "decoder_layers": 1,
                    "heads": 6,
                    "feedforward": 48,
                    "auxiliary_supervision": True,
                    "stages": [
                        {
                            "radius_in_feature_cells": 1.0,
                            "maximum_delta": 0.05,
                            "samples_per_axis": 3,
                        }
                    ],
                },
            },
        }
    }


def test_self_patch_mask_has_zero_diagonal_and_prediction_excludes_own_patch() -> None:
    torch.manual_seed(7)
    block = MessengerRecoveryBlock(24, {"enabled": True, "heads": 6}).eval()
    feature = torch.randn(2, 24, 4, 4)
    _, first = block(feature, return_diagnostics=True)
    changed = feature.clone()
    changed[:, :, 1, 2] += 100.0
    _, second = block(changed, return_diagnostics=True)
    target_index = 1 * 4 + 2

    assert torch.count_nonzero(first["attention"].diagonal(dim1=-2, dim2=-1)) == 0
    torch.testing.assert_close(
        first["messenger_tokens"][:, target_index],
        second["messenger_tokens"][:, target_index],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_zero_initialized_residual_preserves_raw_tokens_and_landmark_output() -> None:
    torch.manual_seed(11)
    model, _ = build_landmark_model(
        _messenger_config(enabled=True),
        variant="query_refinement",
        initialize_from_detector=False,
    )
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    recovered = model(images, recovery_mode="recovered", return_recovery_diagnostics=True)
    raw = model(images, recovery_mode="raw")

    diagnostics = recovered["recovery_diagnostics"]
    torch.testing.assert_close(
        diagnostics["recovered_tokens"], diagnostics["original_tokens"], atol=0, rtol=0
    )
    torch.testing.assert_close(recovered["decoded_coordinates"], raw["decoded_coordinates"])
    assert float(diagnostics["gate_alpha"].mean().detach()) == pytest.approx(0.10, abs=1.0e-6)


def test_disabled_model_keeps_checkpoint_and_forward_contract() -> None:
    implicit = _messenger_config(enabled=False)
    implicit["model"].pop("messenger_recovery")
    old_model, _ = build_landmark_model(
        implicit, variant="query_refinement", initialize_from_detector=False
    )
    disabled_model, _ = build_landmark_model(
        _messenger_config(enabled=False),
        variant="query_refinement",
        initialize_from_detector=False,
    )
    assert old_model.state_dict().keys() == disabled_model.state_dict().keys()
    disabled_model.load_state_dict(old_model.state_dict(), strict=True)
    images = torch.randn(1, 3, 64, 64)
    old_model.eval()
    disabled_model.eval()
    old_output = old_model(images)
    disabled_output = disabled_model(images)
    assert old_output.keys() == disabled_output.keys()
    torch.testing.assert_close(
        old_output["decoded_coordinates"], disabled_output["decoded_coordinates"]
    )


def test_patch_ratios_and_recovery_loss_stop_clean_teacher_gradient() -> None:
    block = MessengerRecoveryBlock(24, {"enabled": True, "heads": 6})
    occluded = torch.randn(2, 24, 2, 2, requires_grad=True)
    clean = torch.randn(2, 24, 2, 2, requires_grad=True)
    _recovered, diagnostics = block(occluded)
    mask = torch.zeros(2, 32, 32)
    mask[0, :16, :16] = 1.0
    mask[1, 8:24, 8:24] = 1.0
    ratios = patch_occlusion_ratios(mask, patch_size=16)
    torch.testing.assert_close(ratios[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    criterion = MessengerRecoveryLoss(
        reconstruction_weight=0.10, gate_weight=0.01, patch_size=16
    )
    losses = criterion(diagnostics, clean, mask)
    losses["loss_messenger_total"].backward()

    assert clean.grad is None
    assert block.attention.in_proj_weight.grad is not None
    assert block.gate_mlp[-1].weight.grad is not None
    no_mask = criterion(diagnostics, clean, torch.zeros_like(mask))
    assert float(no_mask["loss_messenger_total"].detach()) == 0.0


def test_combined_task_and_recovery_losses_reach_head_and_messenger() -> None:
    model, _ = build_landmark_model(
        _messenger_config(enabled=True),
        variant="query_refinement",
        initialize_from_detector=False,
    )
    images = torch.randn(2, 3, 64, 64)
    output = model(images, return_recovery_diagnostics=True)
    targets = [
        {
            "keypoints": torch.rand(5, 2),
            "labeled": torch.ones(5, dtype=torch.bool),
            "visible": torch.tensor([True, True, False, True, False]),
        }
        for _ in range(2)
    ]
    task_losses = LandmarkLoss()(output, targets)
    clean_map = torch.randn(2, 24, 4, 4, requires_grad=True)
    mask = torch.zeros(2, 64, 64)
    mask[:, :16, :16] = 1.0
    recovery_losses = MessengerRecoveryLoss(patch_size=16)(
        output["recovery_diagnostics"], clean_map, mask
    )
    (task_losses["loss_total"] + recovery_losses["loss_messenger_total"]).backward()

    assert clean_map.grad is None
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    assert model.messenger_recovery.correction_projection.weight.grad is not None
    assert model.messenger_recovery.attention.in_proj_weight.grad is not None


def _write_dataset(root: Path) -> tuple[Path, Path]:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[..., 0] = np.arange(96, dtype=np.uint8)[None, :]
    image[..., 1] = np.arange(96, dtype=np.uint8)[:, None]
    image[..., 2] = 127
    Image.fromarray(image).save(root / "face.png")
    keypoints = [35, 40, 2, 61, 40, 2, 48, 51, 2, 39, 64, 2, 57, 64, 1]
    annotation = {
        "images": [{"id": 1, "file_name": "face.png", "width": 96, "height": 96}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 2,
                "bbox": [24, 24, 48, 52],
                "ellipse": [48, 50, 28, 23, 0],
                "keypoints": keypoints,
                "num_keypoints": 5,
            }
        ],
        "categories": [{"id": 2, "name": "face", "keypoints": [str(i) for i in range(5)]}],
    }
    path = root / "annotations.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    return path, root


def test_paired_occlusion_preserves_geometry_flips_mask_and_builds_boundary_slice(
    tmp_path: Path,
) -> None:
    annotation, image_root = _write_dataset(tmp_path)
    paired = {
        "probability": 1.0,
        "kinds": ["rectangle", "irregular", "boundary"],
        "kind_probabilities": [1.0 / 3.0] * 3,
        "face_overlap": [0.10, 0.40],
        "one_face_per_source_image": True,
    }
    augmentation = CropAugmentation(
        horizontal_flip_probability=1.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        blur_probability=0.0,
        occlusion_probability=0.0,
        paired_occlusion=paired,
    )
    random.seed(3)
    dataset = TopDownFaceLandmarkDataset(
        annotation_path=annotation,
        image_root=image_root,
        input_size=256,
        training=True,
        augmentation=augmentation,
        category_ids=[2],
        require_visible_keypoint=False,
        seed=5,
    )
    occluded, target = dataset[0]
    clean = target["clean_image"]
    mask = target["occlusion_mask"]
    assert target["flipped"] is True
    assert bool(mask.any())
    torch.testing.assert_close(occluded[:, ~mask], clean[:, ~mask], atol=0, rtol=0)
    assert target["paired_occlusion"]["pair_kind"] in paired["kinds"]

    boundary = TopDownFaceLandmarkDataset(
        annotation_path=annotation,
        image_root=image_root,
        input_size=256,
        training=False,
        augmentation=augmentation,
        category_ids=[2],
        require_visible_keypoint=False,
        seed=5,
        evaluation_occlusion_kind="boundary",
    )
    _boundary_image, boundary_target = boundary[0]
    assert bool(boundary_target["occlusion_mask"].any())
    assert boundary_target["paired_occlusion"]["deterministic_evaluation"] is True


def test_production_configs_are_matched_and_head_free() -> None:
    root = Path(__file__).resolve().parents[2]
    control = load_landmark_config(
        root / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_messenger_control.yaml"
    )
    messenger = load_landmark_config(
        root / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_messenger_recovery.yaml"
    )
    assert configured_evaluation_pipeline_stages(control.raw) == TWO_STAGE_FACE_KEYPOINT_PIPELINE
    assert configured_evaluation_pipeline_stages(messenger.raw) == TWO_STAGE_FACE_KEYPOINT_PIPELINE
    assert not control.raw["model"]["messenger_recovery"]["enabled"]
    assert messenger.raw["model"]["messenger_recovery"]["enabled"]
    assert control.raw["data"]["augmentation"]["paired_occlusion"] == messenger.raw["data"]["augmentation"]["paired_occlusion"]
    assert control.raw["train"]["microbatch_candidates"] == [4, 8, 16]
    assert control.raw["train"]["grad_scaler_init_scale"] == 1024.0
    assert messenger.raw["train"]["grad_scaler_init_scale"] == 1024.0
    assert messenger.raw["train"]["checkpoint_selection"]["metric"] == "conditional_semantic_macro_f1_nme_0.10"
