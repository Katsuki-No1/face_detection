from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MODEL_NAMES = ("rtmpose_face5", "dinov3_fpn_rtmcc", "dinov3_orformer_rtmcc", "deimv2_joint")


class ComparisonConfigError(ValueError):
    """Raised before an invalid production experiment can start."""


def data_key_for_model(model_name: str) -> str:
    if model_name == "rtmpose_face5":
        return "rtmpose"
    if model_name in {"dinov3_fpn_rtmcc", "dinov3_orformer_rtmcc"}:
        return "dino"
    if model_name == "deimv2_joint":
        return "joint"
    raise ComparisonConfigError(f"Unknown model: {model_name}")


def data_config_for_model(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    key = data_key_for_model(model_name)
    value = config.get("data", {}).get(key)
    if not isinstance(value, dict):
        raise ComparisonConfigError(f"data.{key} is required for {model_name}")
    return value


@dataclass(frozen=True)
class ComparisonConfig:
    path: Path
    project_root: Path
    raw: dict[str, Any]

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output"]["root"])


_PATH_KEYS = {
    "root",
    "annotation",
    "image_root",
    "config",
    "checkpoint",
    "source",
    "hub_repo",
    "weights",
    "five_keypoint_checkpoint",
    "initialize_from_deim_checkpoint",
    "ellipse_error_model",
    "error_model",
}


def _resolve_paths(value: Any, root: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        if key in {"annotations", "paths"}:
            return {name: _resolve_paths(item, root, "annotation") for name, item in value.items()}
        return {name: _resolve_paths(item, root, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, root, key) for item in value]
    if key in _PATH_KEYS and isinstance(value, str) and value and not value.startswith(("http://", "https://")):
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (root / path).resolve())
    return value


def load_config(path: str | Path, *, project_root: str | Path | None = None) -> ComparisonConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ComparisonConfigError(f"Config must contain a mapping: {config_path}")
    root = Path(project_root).expanduser().resolve() if project_root else config_path.parents[2]
    resolved = _resolve_paths(deepcopy(raw), root)
    validate_config(resolved)
    return ComparisonConfig(path=config_path, project_root=root, raw=resolved)


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 2:
        raise ComparisonConfigError("schema_version must be 2")
    if tuple(config.get("landmarks", {}).get("names", ())) != (
        "left_eye",
        "right_eye",
        "nose_tip",
        "left_mouth_corner",
        "right_mouth_corner",
    ):
        raise ComparisonConfigError("The five landmark names and their order are fixed for this comparison")
    models = config.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ComparisonConfigError(f"models must define exactly {MODEL_NAMES}")
    for name, model in models.items():
        if not isinstance(model, dict):
            raise ComparisonConfigError(f"models.{name} must be a mapping")
        if (
            int(model.get("batch_size", 0)) < 1
            or int(model.get("gradient_accumulation_steps", 0)) < 1
        ):
            raise ComparisonConfigError(
                f"{name} batch_size and gradient_accumulation_steps must be positive"
            )
    joint = models["deimv2_joint"]
    fallback = joint.get("batch_size_fallback", {})
    if isinstance(fallback, dict) and bool(fallback.get("enabled", False)):
        candidates = fallback.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ComparisonConfigError(
                "deimv2_joint batch_size_fallback requires candidates"
            )
        try:
            plans = [
                (
                    int(candidate["batch_size"]),
                    int(candidate["gradient_accumulation_steps"]),
                )
                for candidate in candidates
                if isinstance(candidate, dict)
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonConfigError(
                "Invalid deimv2_joint batch fallback candidate"
            ) from error
        if len(plans) != len(candidates) or any(
            batch < 1 or accumulation < 1
            for batch, accumulation in plans
        ):
            raise ComparisonConfigError(
                "DEIMv2 Joint batch fallback values must be positive"
            )
        configured = (
            int(joint["batch_size"]),
            int(joint["gradient_accumulation_steps"]),
        )
        if plans[0] != configured:
            raise ComparisonConfigError(
                "DEIMv2 Joint settings must match the first batch fallback candidate"
            )
        if len(set(plans)) != len(plans):
            raise ComparisonConfigError(
                "DEIMv2 Joint batch fallback candidates must be unique"
            )
        if len({batch * accumulation for batch, accumulation in plans}) != 1:
            raise ComparisonConfigError(
                "DEIMv2 Joint batch fallback must preserve effective batch size"
            )
    data = config.get("data", {})
    for key, expected_category in (("rtmpose", 1), ("dino", 2), ("joint", 2)):
        value = data.get(key)
        if not isinstance(value, dict):
            raise ComparisonConfigError(f"data.{key} must be defined")
        categories = tuple(int(item) for item in value.get("category_ids", ()))
        if categories != (expected_category,):
            raise ComparisonConfigError(
                f"data.{key}.category_ids must be [{expected_category}], got {categories}"
            )
        if set(value.get("annotations", {})) != {"train", "val", "test"}:
            raise ComparisonConfigError(f"data.{key}.annotations must define train/val/test")
    if data["joint"]["annotations"] != data["dino"]["annotations"]:
        raise ComparisonConfigError("DEIM joint and DINO models must share the dedicated DEIM/DINO export")
    if data["rtmpose"]["annotations"] == data["dino"]["annotations"]:
        raise ComparisonConfigError("RTMPose and DEIM/DINO must not share an annotation export")
    if int(data["joint"].get("target_category_id", 0)) != 2:
        raise ComparisonConfigError("data.joint.target_category_id must remain DEIM face category id 2")
    train = config.get("train", {})
    if train.get("production_mode") is not True:
        raise ComparisonConfigError("train.production_mode must be true")
    if int(train.get("epochs", 0)) < int(train.get("minimum_epochs", 100)):
        raise ComparisonConfigError("Production epochs must not be shorter than minimum_epochs")
    if float(train.get("fraction", 0.0)) != 1.0:
        raise ComparisonConfigError("Dataset subsampling is disabled in production mode")
    if not bool(train.get("amp", False)) or not bool(train.get("ema", {}).get("enabled", False)):
        raise ComparisonConfigError("Production training requires AMP and EMA")
    detector = config.get("detector", {})
    if str(detector.get("variant")) != "DEIMv2-L":
        raise ComparisonConfigError("All top-down inference must use the fixed DEIMv2-L detector")
    if str(detector.get("crop_geometry")) != "ellipse":
        raise ComparisonConfigError(
            "All top-down models must use ellipse-aligned DEIM crops"
        )
    augmentation = data.get("augmentation", {})
    if str(augmentation.get("roi_geometry", "")) != "long_axis_square":
        raise ComparisonConfigError(
            "Production top-down crops must use roi_geometry=long_axis_square"
        )
    if abs(float(augmentation.get("scale_factor", 0.0)) - 1.25) > 1.0e-8:
        raise ComparisonConfigError(
            "GT ellipse crop margin must remain scale_factor=1.25"
        )
    if not 0.0 < float(augmentation.get("scale_jitter", 0.0)) < 1.0:
        raise ComparisonConfigError(
            "Independent major/minor axis jitter must be between zero and one"
        )
    if float(augmentation.get("center_jitter", 0.0)) <= 0.0:
        raise ComparisonConfigError("GT ellipse center jitter must be positive")
    if float(augmentation.get("rotation_deg", 0.0)) <= 0.0:
        raise ComparisonConfigError("GT ellipse angle jitter must be positive")
    error_model = str(data.get("ellipse_error_model", "")).strip()
    calibration = data.get("crop_error_calibration")
    if not error_model or not isinstance(calibration, dict):
        raise ComparisonConfigError(
            "Production top-down training requires data.ellipse_error_model "
            "and data.crop_error_calibration"
        )
    if str(calibration.get("error_model", "")).strip() != error_model:
        raise ComparisonConfigError(
            "crop_error_calibration.error_model must equal data.ellipse_error_model"
        )
    if tuple(int(value) for value in calibration.get("category_ids", ())) != (2,):
        raise ComparisonConfigError(
            "Detector-error calibration must use the DEIM/DINO face category [2]"
        )
    if int(calibration.get("minimum_matched_samples", 0)) < 200:
        raise ComparisonConfigError(
            "Detector-error calibration requires at least 200 matched samples"
        )
    if int(calibration.get("area_bins", 0)) < 2:
        raise ComparisonConfigError(
            "Detector-error calibration must be stratified by face area"
        )
    quantiles = tuple(
        float(value) for value in calibration.get("robust_quantiles", ())
    )
    if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
        raise ComparisonConfigError("crop_error_calibration.robust_quantiles is invalid")
    severity = tuple(float(value) for value in calibration.get("severity_range", ()))
    if len(severity) != 2 or severity[0] < 0.0 or severity[1] < severity[0]:
        raise ComparisonConfigError("crop_error_calibration.severity_range is invalid")
    if calibration.get("independent_component_noise", False):
        raise ComparisonConfigError(
            "Independent ellipse-error noise would destroy measured residual correlation"
        )
    for name in ("dinov3_fpn_rtmcc", "dinov3_orformer_rtmcc"):
        model = models[name]
        backbone = model.get("backbone", {})
        if str(backbone.get("name", "")) != "dinov3_vits16":
            raise ComparisonConfigError(f"{name} must use DINOv3 ViT-S/16")
        if str(backbone.get("hub_model", "")) != "dinov3_vits16":
            raise ComparisonConfigError(f"{name}.backbone.hub_model must be dinov3_vits16")
        if int(backbone.get("embed_dim", 0)) != 384 or int(backbone.get("patch_size", 0)) != 16:
            raise ComparisonConfigError(f"{name} must use the 384-dimensional ViT-S/16 architecture")
        layers = tuple(int(value) for value in backbone.get("intermediate_layers", ()))
        if layers != (2, 5, 8, 11):
            raise ComparisonConfigError(f"{name} must use DINOv3-S blocks [2, 5, 8, 11]")
        size = int(model.get("input_size", 0))
        patch = int(backbone.get("patch_size", 16))
        if size < 256 or size % patch:
            raise ComparisonConfigError(f"{name}.input_size must be >=256 and divisible by patch_size")
        if not bool(backbone.get("require_pretrained", True)):
            raise ComparisonConfigError(f"{name} must require pretrained DINOv3 weights")
        if not backbone.get("weights"):
            raise ComparisonConfigError(f"{name} must specify a local DINOv3-S checkpoint")
    left = models["dinov3_fpn_rtmcc"]["backbone"]
    right = models["dinov3_orformer_rtmcc"]["backbone"]
    for key in ("name", "embed_dim", "patch_size", "weights", "intermediate_layers"):
        if left.get(key) != right.get(key):
            raise ComparisonConfigError(f"Models 2 and 3 must share the same DINOv3 initialization: {key}")
    if not bool(models["rtmpose_face5"].get("require_pretrained", True)):
        raise ComparisonConfigError("RTMPose-Face-5 must require pretrained weights")
