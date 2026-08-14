from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from keypoint_comparison2.config import load_config
from keypoint_comparison2.training import validate


SELECTION_METRIC = "conditional_semantic_macro_f1_nme_0.10"


class _PerfectJoint(nn.Module):
    def forward(self, images, targets):
        target = targets[0]
        return {
            "loss_total": images.sum() * 0.0 + 1.0,
            "pred_logits": torch.full((1, 1, 1), 10.0),
            "pred_keypoints": target["keypoints"].unsqueeze(0),
            "presence_logits": torch.full((1, 1, 5), 10.0),
            "visibility_logits": torch.full((1, 1, 5), 10.0),
            "candidate_indices": torch.zeros((1, 1), dtype=torch.long),
            "match_indices": [
                (
                    torch.zeros(1, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                )
            ],
        }


def _batch():
    keypoints = torch.tensor(
        [[[0.30, 0.30], [0.70, 0.30], [0.50, 0.50], [0.35, 0.70], [0.65, 0.70]]],
        dtype=torch.float32,
    )
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
        "head_boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        "ellipses": torch.tensor([[50.0, 50.0, 30.0, 20.0, 0.0]]),
        "labels": torch.tensor([2]),
        "keypoints": keypoints,
        "labeled": torch.ones((1, 5), dtype=torch.bool),
        "visible": torch.ones((1, 5), dtype=torch.bool),
        "image_id": torch.tensor(1),
    }
    return torch.zeros((1, 3, 100, 100)), [target]


def test_ellipse_local_config_is_independent_and_uses_semantic_f1() -> None:
    config = load_config(
        Path("keypoint_comparison2/configs/ellipse_local_refinement.yaml")
    )
    joint = config.raw["models"]["deimv2_joint"]
    selection = config.raw["train"]["checkpoint_selection"]

    assert joint["ellipse_local_refinement"]["enabled"] is True
    assert joint["face_output_geometry"] == "ellipse"
    assert joint["state_head_mode"] == "factorized_presence_visibility"
    assert config.raw["train"]["epochs"] == 40
    assert config.raw["train"]["early_stopping"] == {
        "enabled": True,
        "minimum_epoch": 10,
        "patience": 5,
        "min_delta": 0.0001,
    }
    assert selection["joint_metric"] == SELECTION_METRIC
    assert selection["joint_direction"] == "maximize"
    assert selection["point_presence_threshold"] == 0.70
    assert config.raw["data"]["dataset_version"] == (
        "face_pose_drive_v2_balanced_811"
    )
    assert "normal500_active500" not in config.raw["data"]["joint"][
        "annotations"
    ]["train"]
    assert config.output_root.name.endswith("ellipse_local_refinement_seed42")


def test_joint_validation_uses_final_presence_and_head_normalization() -> None:
    metrics = validate(
        _PerfectJoint(),
        "deimv2_joint",
        [_batch()],
        device=torch.device("cpu"),
        train_cfg={
            "amp": False,
            "checkpoint_selection": {
                "joint_metric": SELECTION_METRIC,
                "face_score_threshold": 0.33,
                "point_presence_threshold": 0.70,
                "nme_threshold": 0.10,
                "head_normalization_scale": 1.20,
            },
        },
    )

    assert metrics[SELECTION_METRIC] == 1.0
    assert metrics["conditional_eligible_faces"] == 1.0
    assert metrics["conditional_semantic_tp"] == 5.0
    assert metrics["conditional_semantic_fp"] == 0.0
    assert metrics["conditional_semantic_fn"] == 0.0
