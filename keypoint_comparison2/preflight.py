from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.data.ellipse_error_model import EllipseErrorSampler

from .config import ComparisonConfig, data_config_for_model
from .data import FaceCropDataset, JointFaceDataset
from .models import build_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_inventory(path: Path, image_root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    source_videos, hashes = set(), set()
    missing_images = 0
    for image in images:
        metadata = image.get("metadata", {})
        if metadata.get("source_video_id"):
            source_videos.add(str(metadata["source_video_id"]))
        if metadata.get("sha256"):
            hashes.add(str(metadata["sha256"]))
        relative = Path(str(image.get("file_name", "")))
        candidates = (relative,) if relative.is_absolute() else (image_root / relative, image_root / "images" / relative)
        if not any(candidate.is_file() for candidate in candidates):
            missing_images += 1
    usable = 0
    fully_labeled = 0
    valid_ellipses = 0
    invalid_ellipses = 0
    visibility_values = {"unlabeled": 0, "occluded": 0, "visible": 0}
    for annotation in annotations:
        keypoints = annotation.get("keypoints", ())
        values = [int(float(keypoints[index])) for index in range(2, min(15, len(keypoints)), 3)]
        values += [0] * (5 - len(values))
        for value in values[:5]:
            key = "visible" if value >= 2 else "occluded" if value == 1 else "unlabeled"
            visibility_values[key] += 1
        if any(value > 0 for value in values):
            usable += 1
        if all(value > 0 for value in values):
            fully_labeled += 1
        ellipse = annotation.get("ellipse")
        if ellipse is not None:
            if (
                isinstance(ellipse, list)
                and len(ellipse) == 5
                and float(ellipse[2]) > 0.0
                and float(ellipse[3]) > 0.0
            ):
                valid_ellipses += 1
            else:
                invalid_ellipses += 1
    categories = data.get("categories", [])
    keypoint_names = categories[0].get("keypoints", []) if categories else []
    return {
        "images": len(images),
        "annotations": len(annotations),
        "usable_keypoint_instances": usable,
        "fully_labeled_instances": fully_labeled,
        "visibility_values": visibility_values,
        "valid_ellipses": valid_ellipses,
        "invalid_ellipses": invalid_ellipses,
        "category_ids": sorted({int(row["category_id"]) for row in annotations}),
        "missing_image_files": missing_images,
        "keypoint_names": keypoint_names,
        "annotation_sha256": _sha256(path),
        "source_videos": source_videos,
        "sha256": hashes,
    }


def run_preflight(
    config: ComparisonConfig,
    *,
    models: list[str],
    device: torch.device,
    forward: bool = True,
) -> dict[str, Any]:
    raw = config.raw
    detector = raw["detector"]
    error_model_path = Path(raw["data"]["ellipse_error_model"])
    required = [
        Path(detector["config"]),
        Path(detector["checkpoint"]),
        error_model_path,
    ]
    for model_name in models:
        model_cfg = raw["models"][model_name]
        if model_name == "rtmpose_face5":
            required.append(Path(model_cfg["weights"]))
        elif model_name.startswith("dinov3_"):
            required.extend((Path(model_cfg["backbone"]["hub_repo"]), Path(model_cfg["backbone"]["weights"])))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required production assets are missing:\n- "
            + "\n- ".join(missing)
            + "\nSee keypoint_comparison2/README.md#事前学習済み重みと権限."
        )
    expected_cfg = raw["data"]["expected"]
    source_configs = {
        "rtmpose": raw["data"]["rtmpose"],
        "deim_dino": raw["data"]["dino"],
    }
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    private_sets: dict[str, dict[str, tuple[set[str], set[str]]]] = {}
    overlaps: dict[str, Any] = {}
    for source_name, split_cfg in source_configs.items():
        image_root = Path(split_cfg["image_root"])
        inventories[source_name] = {}
        private_sets[source_name] = {}
        for split in ("train", "val", "test"):
            path = Path(split_cfg["annotations"][split])
            inventory = _split_inventory(path, image_root)
            expected_images = int(expected_cfg[f"{split}_images"])
            if inventory["images"] != expected_images:
                raise RuntimeError(
                    f"{source_name}.{split} image count mismatch: "
                    f"{inventory['images']} != {expected_images}"
                )
            expected_annotations = int(expected_cfg[f"{split}_annotations"])
            if inventory["annotations"] != expected_annotations:
                raise RuntimeError(
                    f"{source_name}.{split} annotation count mismatch: "
                    f"{inventory['annotations']} != {expected_annotations}"
                )
            expected_usable = int(expected_cfg[f"{split}_usable_keypoint_instances"])
            if inventory["usable_keypoint_instances"] != expected_usable:
                raise RuntimeError(
                    f"{source_name}.{split} usable instance count mismatch: "
                    f"{inventory['usable_keypoint_instances']} != {expected_usable}"
                )
            expected_sha256 = str(expected_cfg["annotation_sha256"][source_name][split])
            if inventory["annotation_sha256"] != expected_sha256:
                raise RuntimeError(
                    f"{source_name}.{split} annotation SHA-256 mismatch: "
                    f"{inventory['annotation_sha256']} != {expected_sha256}"
                )
            expected_visibility = {
                key: int(value)
                for key, value in expected_cfg["visibility_counts"][split].items()
            }
            if inventory["visibility_values"] != expected_visibility:
                raise RuntimeError(
                    f"{source_name}.{split} visibility count mismatch: "
                    f"{inventory['visibility_values']} != {expected_visibility}"
                )
            expected_category_ids = sorted(int(value) for value in split_cfg["category_ids"])
            if inventory["category_ids"] != expected_category_ids:
                raise RuntimeError(
                    f"{source_name}.{split} category mismatch: "
                    f"{inventory['category_ids']} != {expected_category_ids}"
                )
            if (
                inventory["valid_ellipses"] != inventory["annotations"]
                or inventory["invalid_ellipses"]
            ):
                raise RuntimeError(
                    f"{source_name}.{split} must contain one valid DEIM ellipse per annotation"
                )
            if inventory["usable_keypoint_instances"] <= 0:
                raise RuntimeError(f"{source_name}.{split} has no usable keypoint annotations")
            if inventory["missing_image_files"]:
                raise RuntimeError(
                    f"{source_name}.{split} has {inventory['missing_image_files']} missing image files"
                )
            if tuple(inventory["keypoint_names"]) != tuple(raw["landmarks"]["names"]):
                raise RuntimeError(
                    f"{source_name}.{split} keypoint schema mismatch: "
                    f"{inventory['keypoint_names']} != {raw['landmarks']['names']}"
                )
            private_sets[source_name][split] = (
                inventory.pop("source_videos"),
                inventory.pop("sha256"),
            )
            inventories[source_name][split] = inventory
        source_overlaps: dict[str, Any] = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            video_overlap = private_sets[source_name][left][0] & private_sets[source_name][right][0]
            hash_overlap = private_sets[source_name][left][1] & private_sets[source_name][right][1]
            source_overlaps[f"{left}_{right}"] = {
                "source_video_overlap": sorted(video_overlap),
                "image_sha256_overlap": sorted(hash_overlap),
            }
            if video_overlap or hash_overlap:
                raise RuntimeError(
                    f"Dataset leakage detected in {source_name} between {left} and {right}"
                )
        overlaps[source_name] = source_overlaps
    calibration_cfg = raw["data"]["crop_error_calibration"]
    calibration_path = Path(calibration_cfg["annotation"])
    calibration_inventory = _split_inventory(
        calibration_path,
        Path(calibration_cfg["image_root"]),
    )
    expected_calibration = calibration_cfg["expected"]
    for key in ("images", "annotations"):
        if calibration_inventory[key] != int(expected_calibration[key]):
            raise RuntimeError(
                f"crop-error calibration {key} mismatch: "
                f"{calibration_inventory[key]} != {expected_calibration[key]}"
            )
    if (
        calibration_inventory["annotation_sha256"]
        != str(calibration_cfg["annotation_sha256"])
    ):
        raise RuntimeError("Crop-error calibration annotation SHA-256 mismatch")
    calibration_overlaps: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        video_overlap = (
            calibration_inventory["source_videos"]
            & private_sets["deim_dino"][split][0]
        )
        hash_overlap = (
            calibration_inventory["sha256"]
            & private_sets["deim_dino"][split][1]
        )
        calibration_overlaps[split] = {
            "source_video_overlap": sorted(video_overlap),
            "image_sha256_overlap": sorted(hash_overlap),
        }
        if video_overlap or hash_overlap:
            raise RuntimeError(
                "Detector-error calibration leaks into keypoint "
                f"{split}: videos={sorted(video_overlap)} hashes={len(hash_overlap)}"
            )
    error_sampler = EllipseErrorSampler.from_path(error_model_path)
    error_source = error_sampler.model["source"]
    if error_source.get("test_split_used") is not False:
        raise RuntimeError("Ellipse error calibration must never use the test split")
    if error_source.get("keypoint_selection_split_used") is not False:
        raise RuntimeError(
            "Ellipse error calibration must not use keypoint model-selection validation"
        )
    if error_source.get("calibration_annotation_sha256") != _sha256(calibration_path):
        raise RuntimeError("Ellipse error model was built from a different calibration set")
    if error_source.get("detector_checkpoint_sha256") != _sha256(
        Path(detector["checkpoint"])
    ):
        raise RuntimeError("Ellipse error model was built from a different DEIM checkpoint")
    for split in ("train", "val", "test"):
        left = inventories["rtmpose"][split]
        right = inventories["deim_dino"][split]
        for key in (
            "images",
            "annotations",
            "usable_keypoint_instances",
            "fully_labeled_instances",
            "visibility_values",
        ):
            if left[key] != right[key]:
                raise RuntimeError(f"Dedicated export parity mismatch for {split}.{key}")
        if private_sets["rtmpose"][split] != private_sets["deim_dino"][split]:
            raise RuntimeError(f"Dedicated export image identity mismatch for {split}")
    model_reports: dict[str, Any] = {}
    for model_name in models:
        model = build_model(model_name, raw).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        report: dict[str, Any] = {"parameters": parameters, "trainable_parameters": trainable}
        if hasattr(model, "pretrained_report"):
            report["pretrained"] = model.pretrained_report
        if forward:
            model.train()
            if model_name == "deimv2_joint":
                data_cfg = data_config_for_model(raw, model_name)
                dataset = JointFaceDataset(
                    annotation=data_cfg["annotations"]["train"],
                    image_root=data_cfg["image_root"],
                    image_size=int(raw["models"][model_name]["input_size"]),
                    category_ids=data_cfg["category_ids"],
                    target_category_id=int(data_cfg["target_category_id"]),
                    training=False,
                )
                image, target = dataset[0]
                moved = {key: value.to(device) for key, value in target.items()}
                output = model(image.unsqueeze(0).to(device), [moved])
            else:
                data_cfg = data_config_for_model(raw, model_name)
                dataset = FaceCropDataset(
                    annotation=data_cfg["annotations"]["train"],
                    image_root=data_cfg["image_root"],
                    input_size=int(raw["models"][model_name]["input_size"]),
                    training=False,
                    category_ids=data_cfg["category_ids"],
                )
                sample = dataset[0]
                targets = {
                    "coordinates": sample["coordinates"].unsqueeze(0).to(device),
                    "labeled": sample["labeled"].unsqueeze(0).to(device),
                    "visible": sample["visible"].unsqueeze(0).to(device),
                }
                output = model(sample["image"].unsqueeze(0).to(device), targets)
            loss = output["loss_total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name} smoke forward produced non-finite loss")
            loss.backward()
            report["forward_loss"] = float(loss.detach())
        model_reports[model_name] = report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    warnings = []
    if (
        "dinov3_orformer_rtmcc" in models
        and inventories["deim_dino"]["train"]["visibility_values"]["occluded"] == 0
    ):
        raise RuntimeError("ORFormer training requires natural visibility=1 landmark annotations")
    if "dinov3_orformer_rtmcc" in models:
        warnings.append(
            "Natural visibility=1 supervises landmark coordinates/visibility and stratified metrics; "
            "patch-level occlusion masks remain synthetic because point occlusion flags are not pixel masks."
        )
    report = {
        "dataset": inventories,
        "overlaps": overlaps,
        "training_crop_geometry": {
            "source": "ground_truth_ellipse",
            "roi_geometry": str(
                raw["data"]["augmentation"]["roi_geometry"]
            ),
            "resize": "isotropic_affine",
            "short_axis_surplus": "real_image_background",
            "padding": "image_boundary_only",
            "margin_scale": float(raw["data"]["augmentation"]["scale_factor"]),
            "train_geometry_jitter": "joint_empirical_ellipse_deim_residual",
            "validation_geometry_jitter": False,
            "error_model": str(error_model_path),
            "error_model_sha256": _sha256(error_model_path),
            "calibration": {
                key: value
                for key, value in calibration_inventory.items()
                if key not in {"source_videos", "sha256"}
            },
            "calibration_split_overlap": calibration_overlaps,
        },
        "models": model_reports,
        "warnings": warnings,
    }
    output = config.output_root / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
