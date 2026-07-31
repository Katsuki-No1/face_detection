from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_config

from .heads import FACTORIZED_STATE_HEAD, LEGACY_BINARY_STATE_HEAD
from .model import VARIANTS


class LandmarkConfigError(ValueError):
    pass


LEGACY_KEYPOINT_TARGET_POLICY = {
    "coordinate_mask": "coco_v_gt_0",
    "visibility_target": "coco_v_gt_1",
    "visibility_loss_mask": "coco_v_gt_0",
}

KEYPOINT_TARGET_POLICY = {
    "coordinate_mask": "coco_v_gt_0",
    "presence_target": "coco_v_gt_0",
    "presence_loss_mask": "all_keypoint_slots",
    "visibility_target": "coco_v_gt_1",
    "visibility_loss_mask": "coco_v_gt_0",
}


def configured_state_head_mode(config: dict[str, Any]) -> str:
    return str(
        config.get("model", {}).get(
            "state_head_mode",
            LEGACY_BINARY_STATE_HEAD,
        )
    )


def configured_keypoint_target_policy(
    config: dict[str, Any],
) -> dict[str, str]:
    policy = config.get("data", {}).get("keypoint_target_policy")
    return dict(policy) if isinstance(policy, dict) else {}


@dataclass(frozen=True)
class LandmarkExperimentConfig:
    path: Path
    project_root: Path
    raw: dict[str, Any]

    @property
    def output_root(self) -> Path:
        return Path(self.raw["outputs"]["root"])

    @property
    def input_size(self) -> int:
        return int(self.raw["data"]["input_size"])


def _resolved(root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def load_landmark_config(path: str | Path, *, project_root: str | Path | None = None) -> LandmarkExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_config(config_path)
    root = Path(project_root).expanduser().resolve() if project_root else config_path.parents[2].resolve()
    if int(raw.get("schema_version", 0)) != 1:
        raise LandmarkConfigError("schema_version must be 1")
    raw["project_root"] = str(root)
    data = raw.get("data")
    model = raw.get("model")
    detector = raw.get("detector")
    outputs = raw.get("outputs")
    train = raw.get("train")
    if not all(isinstance(value, dict) for value in (data, model, detector, outputs, train)):
        raise LandmarkConfigError("data, model, detector, outputs, and train mappings are required")
    data["root"] = _resolved(root, data["root"])
    if data.get("ellipse_error_model"):
        data["ellipse_error_model"] = _resolved(root, data["ellipse_error_model"])
    for key, value in dict(data.get("annotations", {})).items():
        data["annotations"][key] = _resolved(root, value)
    for name, value in dict(data.get("evaluation_sets", {})).items():
        value["root"] = _resolved(root, value["root"])
        value["annotation"] = _resolved(root, value["annotation"])
    backbone = model.get("backbone", {})
    for key in ("hub_repo", "weights", "initialize_from_deim_checkpoint"):
        if backbone.get(key):
            backbone[key] = _resolved(root, backbone[key])
    detector["config"] = _resolved(root, detector["config"])
    detector["checkpoint"] = _resolved(root, detector["checkpoint"])
    outputs["root"] = _resolved(root, outputs["root"])
    validate_landmark_config(raw)
    return LandmarkExperimentConfig(path=config_path, project_root=root, raw=raw)


def validate_landmark_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    train = config["train"]
    input_size = int(data.get("input_size", 0))
    patch_size = int(model.get("backbone", {}).get("patch_size", 16))
    if input_size < 256 or input_size % patch_size != 0:
        raise LandmarkConfigError("Production landmark input_size must be >=256 and divisible by DINO patch_size")
    if int(model.get("num_landmarks", 0)) != 5:
        raise LandmarkConfigError("This comparison requires exactly five face landmarks")
    variants = set(model.get("variants", {}))
    if variants != set(VARIANTS):
        raise LandmarkConfigError(f"model.variants must define exactly {VARIANTS}")
    if train.get("production_mode") is not True:
        raise LandmarkConfigError("train.production_mode must be true")
    if int(train.get("epochs", 0)) < int(train.get("minimum_epochs", 100)):
        raise LandmarkConfigError("Production epochs are below train.minimum_epochs")
    early_stopping = train.get("early_stopping", {})
    if bool(early_stopping.get("enabled", True)) and int(early_stopping.get("minimum_epoch", 0)) < int(
        train.get("minimum_epochs", 100)
    ):
        raise LandmarkConfigError("early_stopping.minimum_epoch must not be below train.minimum_epochs")
    if float(train.get("fraction", 0.0)) != 1.0:
        raise LandmarkConfigError("Production training forbids dataset subsampling")
    if int(train.get("batch_size", 0)) <= 0 or int(train.get("gradient_accumulation_steps", 0)) <= 0:
        raise LandmarkConfigError("batch_size and gradient_accumulation_steps must be positive")
    if not bool(train.get("amp", False)):
        raise LandmarkConfigError("Production training requires AMP")
    if not bool(train.get("ema", {}).get("enabled", False)):
        raise LandmarkConfigError("Production training requires EMA")
    if not bool(data.get("leakage_audit", {}).get("enabled", False)):
        raise LandmarkConfigError("Production training requires data.leakage_audit.enabled=true")
    state_head_mode = configured_state_head_mode(config)
    if state_head_mode not in {
        LEGACY_BINARY_STATE_HEAD,
        FACTORIZED_STATE_HEAD,
    }:
        raise LandmarkConfigError(
            "model.state_head_mode must be either "
            f"{LEGACY_BINARY_STATE_HEAD!r} or {FACTORIZED_STATE_HEAD!r}"
        )
    expected_policy = (
        KEYPOINT_TARGET_POLICY
        if state_head_mode == FACTORIZED_STATE_HEAD
        else LEGACY_KEYPOINT_TARGET_POLICY
    )
    if data.get("keypoint_target_policy") != expected_policy:
        raise LandmarkConfigError(
            "data.keypoint_target_policy does not match "
            f"model.state_head_mode={state_head_mode!r}; expected "
            f"{expected_policy}"
        )
    if (
        state_head_mode == FACTORIZED_STATE_HEAD
        and data.get("include_missing_keypoint_instances") is not True
    ):
        raise LandmarkConfigError(
            "The factorized state head requires "
            "data.include_missing_keypoint_instances=true"
        )
    if str(config["detector"].get("crop_geometry")) != "ellipse":
        raise LandmarkConfigError(
            "detector.crop_geometry must be ellipse"
        )
    augmentation = data.get("augmentation", {})
    if str(augmentation.get("roi_geometry", "")) != "long_axis_square":
        raise LandmarkConfigError(
            "Production crops must use roi_geometry=long_axis_square"
        )
    if abs(float(augmentation.get("scale_factor", 0.0)) - 1.25) > 1.0e-8:
        raise LandmarkConfigError(
            "GT ellipse crop margin must remain scale_factor=1.25"
        )
    if not 0.0 < float(augmentation.get("scale_jitter", 0.0)) < 1.0:
        raise LandmarkConfigError(
            "Independent major/minor axis jitter must be between zero and one"
        )
    if float(augmentation.get("center_jitter", 0.0)) <= 0.0:
        raise LandmarkConfigError("GT ellipse center jitter must be positive")
    if float(augmentation.get("rotation_deg", 0.0)) <= 0.0:
        raise LandmarkConfigError("GT ellipse angle jitter must be positive")
    if not str(data.get("ellipse_error_model", "")).strip():
        raise LandmarkConfigError(
            "Production training requires data.ellipse_error_model calibrated "
            "from the completed ellipse-DEIM"
        )
