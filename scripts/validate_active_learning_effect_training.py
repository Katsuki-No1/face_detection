#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keypoint_comparison1.landmark_detection.config import (  # noqa: E402
    KEYPOINT_TARGET_POLICY,
    load_landmark_config,
)
from src.config_utils import load_yaml_config  # noqa: E402


class ContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one arm of the active-learning data-effect experiment."
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/active_learning_effect/experiment.yaml",
    )
    parser.add_argument("--arm", choices=("normal400", "active357"), required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ContractError(f"{name}: expected={expected!r}, actual={actual!r}")


def coco_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    usable = sum(
        any(float(item) > 0 for item in annotation.get("keypoints", [])[2::3])
        for annotation in value.get("annotations", [])
    )
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "images": len(value.get("images", [])),
        "annotations": len(value.get("annotations", [])),
        "usable_keypoint_instances": int(usable),
    }


def normalized_sections(
    config: Mapping[str, Any],
    *,
    sections: tuple[str, ...],
) -> dict[str, Any]:
    return {name: config[name] for name in sections}


def validate_pair_fairness(experiment: Mapping[str, Any]) -> dict[str, Any]:
    detector_configs = {
        arm: load_yaml_config(resolve_path(raw["detector_config"]))
        for arm, raw in experiment["arms"].items()
    }
    query_configs = {
        arm: load_landmark_config(
            resolve_path(raw["query_config"]), project_root=PROJECT_ROOT
        ).raw
        for arm, raw in experiment["arms"].items()
    }
    left_detector = detector_configs["normal400"]
    right_detector = detector_configs["active357"]
    require_equal(
        "detector model/loss/augmentation",
        normalized_sections(
            left_detector, sections=("model", "loss", "augmentation")
        ),
        normalized_sections(
            right_detector, sections=("model", "loss", "augmentation")
        ),
    )
    left_train = dict(left_detector["train"])
    right_train = dict(right_detector["train"])
    left_train.pop("name", None)
    right_train.pop("name", None)
    require_equal("detector train recipe", left_train, right_train)

    left_query = query_configs["normal400"]
    right_query = query_configs["active357"]
    left_query_model = copy.deepcopy(left_query["model"])
    right_query_model = copy.deepcopy(right_query["model"])
    left_query_model["backbone"].pop("initialize_from_deim_checkpoint", None)
    right_query_model["backbone"].pop("initialize_from_deim_checkpoint", None)
    require_equal(
        "query model/loss/train recipe",
        {
            "model": left_query_model,
            "loss": left_query["loss"],
            "train": left_query["train"],
        },
        {
            "model": right_query_model,
            "loss": right_query["loss"],
            "train": right_query["train"],
        },
    )
    require_equal(
        "query augmentation",
        left_query["data"]["augmentation"],
        right_query["data"]["augmentation"],
    )
    for key in (
        "crop_geometry",
        "image_size",
        "score_threshold",
        "maximum_detections",
        "matching_iou_threshold",
    ):
        require_equal(
            f"query detector runtime {key}",
            left_query["detector"][key],
            right_query["detector"][key],
        )
    return {
        "detector_recipe_equal": True,
        "query_recipe_equal": True,
        "only_dataset_and_output_identity_differ": True,
    }


def validate_arm(
    experiment_path: Path,
    experiment: Mapping[str, Any],
    arm_name: str,
) -> dict[str, Any]:
    arm = experiment["arms"][arm_name]
    expected = arm["expected"]
    contract = experiment["comparison_contract"]
    latest = experiment["latest_model_contract"]

    detector_config_path = resolve_path(arm["detector_config"])
    query_config_path = resolve_path(arm["query_config"])
    detector_raw = yaml.safe_load(detector_config_path.read_text(encoding="utf-8"))
    query_raw = yaml.safe_load(query_config_path.read_text(encoding="utf-8"))
    require_equal(
        "detector extends latest source config",
        resolve_path(detector_config_path.parent / detector_raw["extends"]),
        resolve_path(latest["detector_source_config"]),
    )
    require_equal(
        "query extends latest source config",
        resolve_path(query_config_path.parent / query_raw["extends"]),
        resolve_path(latest["query_source_config"]),
    )

    detector = load_yaml_config(detector_config_path)
    query = load_landmark_config(
        query_config_path, project_root=PROJECT_ROOT
    ).raw
    require_equal("detector model.type", detector["model"]["type"], "dinov3_deimv2_ellipse")
    require_equal("detector verification variant", detector["verification"]["official_variant"], "DEIMv2-L")
    require_equal("detector backbone", detector["model"]["backbone"]["name"], "dinov3_vits16")
    require_equal("detector seed", int(detector["train"]["seed"]), int(contract["detector_seed"]))
    require_equal("ellipse border crop", float(detector["augmentation"]["ellipse_border_crop"]), 0.20)
    require_equal(
        "preserve outside ellipse center",
        detector["augmentation"]["preserve_outside_ellipse_center"],
        True,
    )
    require_equal(
        "detector public COCO initialization",
        detector["model"]["init_checkpoint"]["path"],
        "checkpoint/DEIMv2/deimv2_dinov3_l_coco.safetensors",
    )

    require_equal("query state head", query["model"]["state_head_mode"], latest["state_head_mode"])
    require_equal("query target policy", query["data"]["keypoint_target_policy"], KEYPOINT_TARGET_POLICY)
    require_equal("query seed", int(query["train"]["seed"]), int(contract["query_refinement_seed"]))
    require_equal("query warm-start disabled", query["train"]["refinement_initialize_from_query"], False)
    require_equal("query face threshold", float(query["detector"]["score_threshold"]), float(contract["face_threshold"]))
    require_equal("query default evaluation", query["evaluation"]["default_sets"], ["v2_val", "v2_test"])

    detector_output = resolve_path(arm["output_root"]) / "ellipse_deimv2_l"
    expected_detector_checkpoint = detector_output / "checkpoints/best_val_ellipse_map.pt"
    require_equal(
        "query detector checkpoint",
        Path(query["detector"]["checkpoint"]),
        expected_detector_checkpoint,
    )
    require_equal(
        "query DINO initialization",
        Path(query["model"]["backbone"]["initialize_from_deim_checkpoint"]),
        expected_detector_checkpoint,
    )
    require_equal(
        "query detector config",
        Path(query["detector"]["config"]),
        detector_config_path,
    )

    adapter_root = resolve_path(arm["adapter_root"])
    inventories = {
        split: coco_inventory(adapter_root / f"face_keypoints_{split}.json")
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        require_equal(
            f"{arm_name}.{split}.images",
            inventories[split]["images"],
            int(expected["images"][split]),
        )
        require_equal(
            f"{arm_name}.{split}.annotations",
            inventories[split]["annotations"],
            int(expected["annotations"][split]),
        )
        require_equal(
            f"{arm_name}.{split}.sha256",
            inventories[split]["sha256"],
            expected["coco_sha256"][split],
        )
    require_equal(
        f"{arm_name}.train.usable_keypoint_instances",
        inventories["train"]["usable_keypoint_instances"],
        int(expected["usable_keypoint_instances"]["train"]),
    )

    build_manifest_path = resolve_path(arm["dataset_build_manifest"])
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    require_equal("dataset build status", build_manifest["status"], "ok")
    require_equal("dataset build arm", build_manifest["arm"], arm_name)
    require_equal("base content overlap", int(build_manifest["base_content_overlap"]), 0)
    require_equal(
        "train-only split counts",
        build_manifest["split_counts"],
        {key: int(value) for key, value in expected["images"].items()},
    )
    require_equal(
        "unchanged val hash",
        inventories["val"]["sha256"],
        experiment["arms"]["normal400"]["expected"]["coco_sha256"]["val"],
    )
    require_equal(
        "unchanged test hash",
        inventories["test"]["sha256"],
        experiment["arms"]["normal400"]["expected"]["coco_sha256"]["test"],
    )

    return {
        "status": "passed",
        "arm": arm_name,
        "experiment_config": experiment_path.as_posix(),
        "experiment_config_sha256": sha256_file(experiment_path),
        "detector_config": detector_config_path.as_posix(),
        "detector_config_sha256": sha256_file(detector_config_path),
        "query_config": query_config_path.as_posix(),
        "query_config_sha256": sha256_file(query_config_path),
        "dataset_build_manifest": build_manifest_path.as_posix(),
        "dataset_build_manifest_sha256": sha256_file(build_manifest_path),
        "inventories": inventories,
    }


def validate_latest_references(experiment: Mapping[str, Any]) -> dict[str, Any]:
    latest = experiment["latest_model_contract"]
    detector_checkpoint = resolve_path(latest["detector_reference_checkpoint"])
    query_checkpoint = resolve_path(latest["corrected_query_reference_checkpoint"])
    require_equal(
        "detector reference checkpoint SHA-256",
        sha256_file(detector_checkpoint),
        latest["detector_reference_checkpoint_sha256"],
    )
    require_equal(
        "corrected query checkpoint SHA-256",
        sha256_file(query_checkpoint),
        latest["corrected_query_reference_checkpoint_sha256"],
    )
    checkpoint = torch.load(query_checkpoint, map_location="cpu", weights_only=False)
    require_equal("corrected query variant", checkpoint.get("variant"), "query_refinement")
    require_equal(
        "corrected query state head",
        checkpoint.get("state_head_mode"),
        latest["state_head_mode"],
    )
    require_equal(
        "corrected query target policy",
        checkpoint.get("keypoint_target_policy"),
        KEYPOINT_TARGET_POLICY,
    )
    source_files = (
        resolve_path(latest["detector_source_config"]),
        resolve_path(latest["query_source_config"]),
        PROJECT_ROOT / "src/models/dinov3_deimv2_ellipse/official.py",
        PROJECT_ROOT / "keypoint_comparison1/landmark_detection/model.py",
        PROJECT_ROOT / "keypoint_comparison1/landmark_detection/training.py",
    )
    return {
        "detector_reference_checkpoint": detector_checkpoint.as_posix(),
        "detector_reference_checkpoint_sha256": sha256_file(detector_checkpoint),
        "corrected_query_reference_checkpoint": query_checkpoint.as_posix(),
        "corrected_query_reference_checkpoint_sha256": sha256_file(query_checkpoint),
        "corrected_query_checkpoint_epoch": checkpoint.get("epoch"),
        "source_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in source_files
        },
    }


def git_provenance() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def main() -> int:
    args = parse_args()
    experiment_path = resolve_path(args.experiment_config)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    marker = resolve_path(args.success_marker)
    marker.unlink(missing_ok=True)
    require_equal("schema_version", int(experiment.get("schema_version", 0)), 1)
    payload = validate_arm(experiment_path, experiment, args.arm)
    payload["latest_model"] = validate_latest_references(experiment)
    payload["fairness"] = validate_pair_fairness(experiment)
    payload["git"] = git_provenance()
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
