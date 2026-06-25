from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "dataset_yaml": "outputs_experiment/yolo_pose_dataset_ablation_20260625_all_zips/datasets/modified_100/face_pose.yaml",
        "data_root": "outputs_experiment/yolo_pose_dataset_ablation_20260625_all_zips/datasets/modified_100",
        "metadata_path": "outputs_experiment/yolo_pose_dataset_ablation_20260625_all_zips/datasets/_modified_all_train/metadata/conversion_metadata.json",
        "split": "test",
        "max_images": 300,
        "seed": 42,
    },
    "dino": {
        "model_name": "dinov2_vits14",
        "input_size": 518,
        "device": "auto",
        "use_cls_token": True,
        "use_patch_tokens": True,
        "layers": ["last"],
        "mock_features": False,
        "mock_feature_dim": 64,
    },
    "crops": {
        "use_face_crop": True,
        "use_head_crop": True,
        "use_eye_crop": True,
        "use_background_crop": True,
        "use_hard_negative_crop": True,
        "crop_padding_ratio": 0.15,
        "background_crops_per_image": 3,
        "hard_negative_crops_per_image": 1,
        "min_crop_size": 16,
        "max_background_iou": 0.05,
    },
    "analysis": {
        "knn_k": 5,
        "retrieval_topk": 10,
        "max_visualization_samples": 50,
        "similarity_map_samples": 30,
        "max_pair_samples": 200000,
        "seed": 42,
    },
    "output": {
        "output_dir": "outputs/phase1_dino_feature_analysis",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    config["_config_path"] = str(config_path)
    return config


def resolve_data_root(config: dict[str, Any]) -> Path:
    data = config.get("data", {})
    data_root = data.get("data_root")
    if data_root:
        return Path(data_root)
    dataset_yaml = data.get("dataset_yaml")
    if dataset_yaml:
        yaml_data = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
        yaml_root = yaml_data.get("path")
        if yaml_root:
            return Path(yaml_root)
    raise ValueError("Either data.data_root or data.dataset_yaml must be set.")


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("output", {}).get("output_dir", DEFAULT_CONFIG["output"]["output_dir"]))


def apply_cli_overrides(
    config: dict[str, Any],
    *,
    max_images: int | None = None,
    output_dir_override: str | Path | None = None,
    mock_features: bool | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if max_images is not None:
        config.setdefault("data", {})["max_images"] = max_images
    if output_dir_override is not None:
        config.setdefault("output", {})["output_dir"] = str(output_dir_override)
    if mock_features is not None:
        config.setdefault("dino", {})["mock_features"] = mock_features
    if split is not None:
        config.setdefault("data", {})["split"] = split
    return config
