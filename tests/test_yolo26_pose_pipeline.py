from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts_YOLO.yolo26_pose.pipeline import (
    PipelineError,
    load_pipeline_config,
    recommend_variant,
    select_model_keys,
    validate_production_training,
    validate_wandb_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml"
V2_CONFIG = ROOT / "configs/yolo/yolo26_pose/face_pose_drive_v2_full.yaml"


def test_full_config_contains_s_and_m_without_dataset_subsampling() -> None:
    config = load_pipeline_config(CONFIG)
    assert tuple(config.models) == ("s", "m")
    assert config.models["s"].official_checkpoint == "yolo26s-pose.pt"
    assert config.models["m"].official_checkpoint == "yolo26m-pose.pt"
    assert config.dataset.expected_images == {"train": 2460, "val": 300, "test": 330}
    assert config.dataset.expected_kpt_shape == (5, 3)
    assert config.training["fraction"] == 1.0
    assert config.training["epochs"] == 200
    assert config.training["patience"] == 50
    assert config.training["optimizer"] == "MuSGD"
    assert config.wandb.enabled is True
    assert config.wandb.entity == "sato-waseda-university"
    assert config.wandb.project == "face_detection"
    assert config.wandb.group == "face_pose_drive_v1_four_model_comparison"
    assert config.wandb.mode == "online"


def test_v2_config_uses_the_unified_full_dataset_and_separate_outputs() -> None:
    config = load_pipeline_config(V2_CONFIG)
    assert config.dataset.version == "face_pose_drive_v2_balanced_811"
    assert config.dataset.expected_images == {"train": 4237, "val": 530, "test": 528}
    assert config.dataset.expected_kpt_shape == (5, 3)
    assert config.training["fraction"] == 1.0
    assert config.training["epochs"] == 200
    assert config.models["m"].batch == 8
    assert config.outputs.train_project == (
        ROOT / "runs/yolo/yolo26_pose/face_pose_drive_v2_balanced_811"
    )
    assert config.wandb.group == "face_pose_drive_v2_balanced_811_deimv2_l_yolo26m"


def test_production_contract_rejects_subsampling_and_demo_epochs() -> None:
    config = load_pipeline_config(CONFIG)
    subsampled = deepcopy(config.training)
    subsampled["fraction"] = 0.25
    with pytest.raises(PipelineError, match="fraction=1.0"):
        validate_production_training(subsampled)
    short = deepcopy(config.training)
    short["epochs"] = 10
    with pytest.raises(PipelineError, match="at least"):
        validate_production_training(short)


def test_model_selection_is_stable_and_validated() -> None:
    assert select_model_keys("all") == ["s", "m"]
    assert select_model_keys(["m", "s"]) == ["s", "m"]
    assert select_model_keys("m") == ["m"]
    with pytest.raises(PipelineError, match="Unknown"):
        select_model_keys("n")


def test_wandb_production_contract_rejects_disabled_or_offline_logging() -> None:
    with pytest.raises(PipelineError, match="enabled"):
        validate_wandb_config(
            {"enabled": False, "entity": "team", "project": "project", "group": "group", "mode": "online"}
        )
    with pytest.raises(PipelineError, match="online"):
        validate_wandb_config(
            {"enabled": True, "entity": "team", "project": "project", "group": "group", "mode": "offline"}
        )


def _result(pose_map: float, box_map: float, nme: float) -> dict:
    return {
        "map": {
            "pose": {"map50_95": pose_map},
            "box": {"map50_95": box_map},
        },
        "keypoints": {"nme": nme},
    }


def test_recommendation_requires_meaningful_m_gain() -> None:
    selection = {
        "current_default": "s",
        "pose_map50_95_min_gain_for_m": 0.01,
        "bbox_map50_95_max_regression_for_m": 0.005,
        "nme_max_regression_for_m": 0.002,
    }
    promoted = recommend_variant(
        {"s": _result(0.70, 0.70, 0.050), "m": _result(0.72, 0.70, 0.049)},
        selection,
    )
    assert promoted["recommended"] == "m"
    retained = recommend_variant(
        {"s": _result(0.70, 0.70, 0.050), "m": _result(0.705, 0.72, 0.045)},
        selection,
    )
    assert retained["recommended"] == "s"


def test_recommendation_is_pending_until_both_runs_exist() -> None:
    result = recommend_variant({"s": _result(0.7, 0.7, 0.05)}, {"current_default": "s"})
    assert result == {
        "status": "pending",
        "recommended": "s",
        "reason": "Both YOLO26s and YOLO26m require results from the same fixed validation/test protocol.",
    }
