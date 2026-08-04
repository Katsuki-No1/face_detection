from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data.coco_split_audit import audit_coco_splits

from .config import LandmarkExperimentConfig
from .dataset import CropAugmentation, TopDownFaceLandmarkDataset, collate_landmark_batch
from .detector import DeimFaceDetector
from .diagnostics import gradient_interference_matrix
from .evaluation import evaluate_deim_boxes, evaluate_gt_boxes, load_or_create_detector_predictions, sha256_file
from .losses import LandmarkLoss
from .model import VARIANTS, VITPOSE_VARIANTS, build_landmark_model
from .training import load_landmark_checkpoint, seed_everything, train_variant


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentCondition:
    condition: str
    bbox_mode: str
    variant: str
    purpose: str


EXPERIMENT_CONDITIONS: tuple[ExperimentCondition, ...] = (
    ExperimentCondition("A", "gt", "vitpose", "ViTPose-style head upper bound with oracle boxes"),
    ExperimentCondition("B", "gt", "query", "Landmark Query design effect with oracle boxes"),
    ExperimentCondition("C", "deim", "vitpose", "DEIM predicted-box impact on ViTPose-style"),
    ExperimentCondition("D", "deim", "query", "Deployment performance of Landmark Query"),
    ExperimentCondition("E", "deim", "query_refinement", "Added benefit from coordinate-conditioned staged refinement"),
    ExperimentCondition(
        "F",
        "gt",
        "vitpose_376",
        "Parameter-matched ViTPose-style head with oracle boxes",
    ),
    ExperimentCondition(
        "G",
        "deim",
        "vitpose_376",
        "Parameter-matched ViTPose-style deployment performance",
    ),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def _usable_annotations(coco: Mapping[str, Any]) -> int:
    total = 0
    for annotation in coco.get("annotations", []):
        keypoints = list(annotation.get("keypoints", []))
        if any(float(keypoints[index]) > 0 for index in range(2, len(keypoints), 3)):
            total += 1
    return total


def _annotation_inventory(
    path: Path,
    *,
    image_root: Path | None = None,
    verify_images: bool = False,
) -> dict[str, Any]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    source_videos = {
        str(image.get("metadata", {}).get("source_video_id", ""))
        for image in coco.get("images", [])
        if str(image.get("metadata", {}).get("source_video_id", ""))
    }
    image_hashes = {
        str(image.get("metadata", {}).get("sha256", "")).lower()
        for image in coco.get("images", [])
        if str(image.get("metadata", {}).get("sha256", ""))
    }
    valid_ellipses = sum(
        isinstance(annotation.get("ellipse"), list)
        and len(annotation["ellipse"]) == 5
        and float(annotation["ellipse"][2]) > 0.0
        and float(annotation["ellipse"][3]) > 0.0
        for annotation in coco.get("annotations", [])
    )
    visibility_counts = {"0": 0, "1": 0, "2": 0}
    for annotation in coco.get("annotations", []):
        keypoints = list(annotation.get("keypoints", []))
        for index in range(2, len(keypoints), 3):
            value = str(int(float(keypoints[index])))
            visibility_counts[value] = visibility_counts.get(value, 0) + 1
    verified_images = 0
    if verify_images:
        if image_root is None:
            raise ValueError("image_root is required when verify_images=True")
        actual_hashes: set[str] = set()
        for image in coco.get("images", []):
            relative = Path(str(image["file_name"]))
            if relative.is_absolute():
                raise RuntimeError(f"Absolute evaluation image path is forbidden: {relative}")
            image_path = image_root / relative
            if not image_path.is_file():
                raise FileNotFoundError(f"Evaluation image does not exist: {image_path}")
            actual = sha256_file(image_path)
            recorded = str(image.get("metadata", {}).get("sha256", "")).lower()
            if not recorded or actual != recorded:
                raise RuntimeError(
                    f"Evaluation image SHA mismatch: path={image_path}, recorded={recorded!r}, actual={actual}"
                )
            actual_hashes.add(actual)
            verified_images += 1
        image_hashes = actual_hashes
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "usable_instances": _usable_annotations(coco),
        "valid_ellipses": valid_ellipses,
        "visibility_counts": visibility_counts,
        "verified_images": verified_images,
        "source_video_ids": source_videos,
        "image_hashes": image_hashes,
    }


def _verify_expected(actual: Mapping[str, Any], expected: Mapping[str, Any], *, name: str) -> None:
    for key in ("sha256", "images", "annotations", "usable_instances"):
        if key in expected and actual[key] != expected[key]:
            raise RuntimeError(
                f"Dataset verification failed for {name}.{key}: expected={expected[key]!r}, actual={actual[key]!r}"
            )


def _parameter_report(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "head": sum(parameter.numel() for parameter in model.head.parameters()),
        "trainable_at_build": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _check_wandb(config: LandmarkExperimentConfig, *, required: bool) -> dict[str, Any]:
    value = config.raw.get("wandb", {})
    enabled = bool(value.get("enabled", False))
    status = {
        "enabled": enabled,
        "mode": str(value.get("mode", "online")),
        "entity": value.get("entity"),
        "project": value.get("project"),
        "group": value.get("group"),
    }
    if not enabled:
        if required:
            raise RuntimeError("W&B is required for production training but wandb.enabled is false")
        return status
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is enabled but the package is not installed") from error
    has_api_key = bool(getattr(wandb.api, "api_key", None))
    status["credentials_available"] = has_api_key
    if required and status["mode"] == "online" and not has_api_key:
        raise RuntimeError("W&B online mode is enabled but no API key is available; run `wandb login` first")
    return status


def preflight(
    config: LandmarkExperimentConfig,
    *,
    device: torch.device,
    check_detector: bool = True,
    require_wandb: bool = False,
) -> dict[str, Any]:
    """Validate data, real weights, all heads, gradients, and optional detector inference."""

    raw = config.raw
    seed_everything(int(raw["train"].get("seed", 0)))
    output_dir = config.output_root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = output_dir / "success.json"
    if success_marker.is_file():
        success_marker.unlink()
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "config_resolved.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    started = time.perf_counter()
    LOGGER.info("[preflight] verifying fixed datasets")
    inventories: dict[str, dict[str, Any]] = {}
    for split, annotation in raw["data"]["annotations"].items():
        inventory = _annotation_inventory(Path(annotation))
        _verify_expected(inventory, raw["data"].get("expected", {}).get(split, {}), name=split)
        if inventory["valid_ellipses"] != inventory["annotations"]:
            raise RuntimeError(
                f"{split} must contain one valid GT ellipse per annotation: "
                f"{inventory['valid_ellipses']} != {inventory['annotations']}"
            )
        inventories[split] = {key: value for key, value in inventory.items() if not isinstance(value, set)}

    split_audit = audit_coco_splits(
        raw["data"]["root"],
        annotation_paths=raw["data"]["annotations"],
        splits=tuple(raw["data"]["leakage_audit"].get("splits", ("train", "val", "test"))),
        expected_category_ids=raw["data"].get("category_ids", (1,)),
    )
    known_source_videos: set[str] = set()
    known_hashes: set[str] = set()
    for annotation in raw["data"]["annotations"].values():
        inventory = _annotation_inventory(Path(annotation))
        known_source_videos.update(inventory["source_video_ids"])
        known_hashes.update(inventory["image_hashes"])
    evaluation_inventory: dict[str, Any] = {}
    for name, definition in raw["data"].get("evaluation_sets", {}).items():
        if not bool(definition.get("enabled", True)):
            evaluation_inventory[name] = {
                "enabled": False,
                "disabled_reason": "disabled_by_experiment_config",
            }
            continue
        inventory = _annotation_inventory(
            Path(definition["annotation"]),
            image_root=Path(definition["root"]),
            verify_images=True,
        )
        expected_hash = definition.get("sha256")
        if expected_hash and inventory["sha256"] != expected_hash:
            raise RuntimeError(
                f"Evaluation-set SHA mismatch for {name}: expected={expected_hash}, actual={inventory['sha256']}"
            )
        source_overlap = sorted(inventory["source_video_ids"] & known_source_videos)
        image_overlap = sorted(inventory["image_hashes"] & known_hashes)
        # The configured v2 test is expected to be present in the known split set.
        external = bool(definition.get("must_be_disjoint_from_training", False))
        if external and (source_overlap or image_overlap) and bool(raw["data"]["leakage_audit"].get("fail_on_external_overlap", True)):
            raise RuntimeError(
                f"External evaluation leakage in {name}: source_videos={source_overlap[:5]}, image_hashes={image_overlap[:5]}"
            )
        evaluation_inventory[name] = {
            **{key: value for key, value in inventory.items() if not isinstance(value, set)},
            "source_video_overlap_count": len(source_overlap),
            "image_hash_overlap_count": len(image_overlap),
            "must_be_disjoint_from_training": external,
        }

    LOGGER.info("[preflight] checking DINOv3 initialization and A-G model graph")
    val_dataset = TopDownFaceLandmarkDataset(
        annotation_path=raw["data"]["annotations"]["val"],
        image_root=raw["data"]["root"],
        input_size=int(raw["data"]["input_size"]),
        training=False,
        augmentation=CropAugmentation(scale_factor=float(raw["data"]["augmentation"]["scale_factor"])),
        mean=raw["data"]["mean"],
        std=raw["data"]["std"],
        category_ids=raw["data"]["category_ids"],
        require_visible_keypoint=not bool(
            raw["data"].get("include_missing_keypoint_instances", False)
        ),
    )
    image, target = val_dataset[0]
    sample = image.unsqueeze(0).to(device)
    criterion = LandmarkLoss(raw.get("loss", {})).to(device)
    model_reports: dict[str, Any] = {}
    for variant in VARIANTS:
        LOGGER.info("[preflight:%s] real forward/backward", variant)
        model, initialization = build_landmark_model(raw, variant=variant)
        parameters = _parameter_report(model)
        model.to(device).train()
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(sample, return_attention=variant not in VITPOSE_VARIANTS)
            losses = criterion(output, [target])
        losses["loss_total"].backward()
        finite_gradients = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not finite_gradients:
            raise RuntimeError(f"Non-finite gradient in {variant} preflight")
        model_reports[variant] = {
            "parameters": parameters,
            "dino_initialization": initialization,
            "feature_map_shape": list(output["feature_map_shape"]),
            "coordinate_shape": list(output["coordinates"].shape),
            "loss_total": float(losses["loss_total"].detach().cpu()),
            "finite_gradients": finite_gradients,
        }
        del output, losses, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    detector_report: dict[str, Any] | None = None
    if check_detector:
        LOGGER.info("[preflight:detector] loading DEIMv2-L and running one real image")
        detector_cfg = raw["detector"]
        detector = DeimFaceDetector(
            config_path=detector_cfg["config"],
            checkpoint_path=detector_cfg["checkpoint"],
            device=str(device),
            image_size=int(detector_cfg["image_size"]),
            score_threshold=float(detector_cfg["score_threshold"]),
            maximum_detections=int(detector_cfg["maximum_detections"]),
            crop_geometry=str(detector_cfg["crop_geometry"]),
        )
        sample_path = Path(val_dataset.samples[0]["image_path"])
        detections = detector.predict(Image.open(sample_path).convert("RGB"))
        detector_report = {
            "config": str(detector.config_path),
            "checkpoint": str(detector.checkpoint_path),
            "checkpoint_sha256": sha256_file(detector.checkpoint_path),
            "sample_image": str(sample_path),
            "sample_detection_count": len(detections),
        }
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "status": "ok",
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "seed": int(raw["train"].get("seed", 0)),
        "state_head_mode": str(
            raw["model"].get("state_head_mode", "legacy_binary")
        ),
        "keypoint_target_policy": dict(
            raw["data"]["keypoint_target_policy"]
        ),
        "data": inventories,
        "split_leakage_audit": split_audit,
        "evaluation_sets": evaluation_inventory,
        "models": model_reports,
        "detector": detector_report,
        "wandb": _check_wandb(config, required=require_wandb),
        "conditions": [asdict(condition) for condition in EXPERIMENT_CONDITIONS],
    }
    _write_json(output_dir / "report.json", report)
    _write_json(
        success_marker,
        {
            "status": "success",
            "config": str(config.path),
            "config_sha256": sha256_file(config.path),
            "report": str(output_dir / "report.json"),
            "seed": report["seed"],
            "state_head_mode": report["state_head_mode"],
        },
    )
    LOGGER.info("[preflight] complete in %.1fs", report["elapsed_seconds"])
    return report


def train_variants(
    config: LandmarkExperimentConfig,
    *,
    variants: Sequence[str],
    device: torch.device,
    resume: bool,
    wandb_enabled: bool,
) -> list[dict[str, Any]]:
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown landmark variants: {unknown}")
    ordered = [variant for variant in VARIANTS if variant in variants]
    summaries: list[dict[str, Any]] = []
    for variant in ordered:
        last = config.output_root / variant / "checkpoints" / "last.pt"
        resume_from = last if resume and last.is_file() else None
        LOGGER.info("[train:%s] resume=%s", variant, resume_from or "no")
        summaries.append(
            train_variant(
                config,
                variant=variant,
                device=device,
                resume_from=resume_from,
                wandb_enabled=wandb_enabled,
            )
        )
    _write_json(config.output_root / "training_summary.json", summaries)
    return summaries


def _evaluation_dataset(config: LandmarkExperimentConfig, name: str) -> TopDownFaceLandmarkDataset:
    raw = config.raw
    definition = raw["data"]["evaluation_sets"][name]
    return TopDownFaceLandmarkDataset(
        annotation_path=definition["annotation"],
        image_root=definition["root"],
        input_size=int(raw["data"]["input_size"]),
        training=False,
        augmentation=CropAugmentation(scale_factor=float(raw["data"]["augmentation"]["scale_factor"])),
        mean=raw["data"]["mean"],
        std=raw["data"]["std"],
        category_ids=definition.get("category_ids", raw["data"].get("category_ids", (1,))),
        require_visible_keypoint=not bool(
            raw["data"].get("include_missing_keypoint_instances", False)
        ),
    )


def _summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [
        float(value)
        for row in rows
        if not bool(row.get("missed", False))
        for value in row.get("normalized_errors", [])
        if value is not None
    ]
    return {
        "instances": len(rows),
        "evaluated_keypoints": len(errors),
        "nme": float(np.mean(errors)) if errors else None,
        "pck_0.03": sum(value <= 0.03 for value in errors) / len(errors) if errors else None,
        "pck_0.05": sum(value <= 0.05 for value in errors) / len(errors) if errors else None,
        "pck_0.10": sum(value <= 0.10 for value in errors) / len(errors) if errors else None,
    }


def _paired_oracle_metrics(
    oracle_rows: Sequence[Mapping[str, Any]],
    predicted_box_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matched_ids = {
        (int(row["image_id"]), int(row["annotation_id"]))
        for row in predicted_box_rows
        if not bool(row.get("missed", False))
    }
    paired = [
        row
        for row in oracle_rows
        if (int(row["image_id"]), int(row["annotation_id"])) in matched_ids
    ]
    result = _summarize_prediction_rows(paired)
    result["matched_instance_ids"] = len(matched_ids)
    return result


def _interference_diagnostic(
    model: torch.nn.Module,
    dataset: TopDownFaceLandmarkDataset,
    criterion: LandmarkLoss,
    *,
    device: torch.device,
    batch_size: int,
    maximum_parameters: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_landmark_batch,
    )
    images, targets = next(iter(loader))
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(images.to(device), return_attention=True)
        return gradient_interference_matrix(
            model,
            criterion,
            output,
            targets,
            maximum_parameters=maximum_parameters,
        )


def evaluate_conditions(
    config: LandmarkExperimentConfig,
    *,
    evaluation_sets: Sequence[str],
    device: torch.device,
    force_detector_cache: bool = False,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, Any]:
    raw = config.raw
    available = {
        name: definition
        for name, definition in raw["data"].get(
            "evaluation_sets", {}
        ).items()
        if bool(definition.get("enabled", True))
    }
    evaluation_cfg = raw.get("evaluation", {})
    configured_common_root = evaluation_cfg.get("baseline_aligned_output_root")
    validation_success = (
        Path(str(configured_common_root)) / "val/baseline_aligned/success.json"
        if configured_common_root
        else None
    )
    if (
        "v2_test" in evaluation_sets
        and "v2_val" in available
        and "query_refinement" in variants
        and "v2_val" not in evaluation_sets
        and not (validation_success is not None and validation_success.is_file())
    ):
        # Keep already-launched queue commands compatible with the new common
        # contract: always produce validation before the fixed test report.
        evaluation_sets = ("v2_val", *evaluation_sets)
    unknown = sorted(set(evaluation_sets) - set(available))
    if unknown:
        raise ValueError(f"Unknown evaluation sets: {unknown}; available={sorted(available)}")
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(
            f"Unknown evaluation variants: {unknown_variants}; "
            f"available={list(VARIANTS)}"
        )
    selected_variants = tuple(
        variant for variant in VARIANTS if variant in set(variants)
    )
    if not selected_variants:
        raise ValueError("At least one evaluation variant is required")
    checkpoint_paths = {
        variant: config.output_root / variant / "checkpoints" / "best_nme.pt"
        for variant in selected_variants
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        if bool(
            raw.get("evaluation", {}).get(
                "defer_until_all_checkpoints",
                False,
            )
        ):
            result = {
                "status": "deferred",
                "reason": "not_all_head_checkpoints_are_available",
                "missing_checkpoints": missing,
            }
            LOGGER.warning(
                "Deferring evaluation until all %d selected heads are trained; "
                "missing=%s",
                len(selected_variants),
                missing,
            )
            _write_json(
                config.output_root / "evaluation" / "status.json",
                result,
            )
            return result
        raise FileNotFoundError(
            f"Cannot evaluate before all {len(selected_variants)} selected "
            "heads are trained. Missing: "
            + ", ".join(missing)
        )

    detector_cfg = raw["detector"]
    detector = DeimFaceDetector(
        config_path=detector_cfg["config"],
        checkpoint_path=detector_cfg["checkpoint"],
        device=str(device),
        image_size=int(detector_cfg["image_size"]),
        score_threshold=float(detector_cfg["score_threshold"]),
        maximum_detections=int(detector_cfg["maximum_detections"]),
        crop_geometry=str(detector_cfg["crop_geometry"]),
    )
    criterion = LandmarkLoss(raw.get("loss", {})).to(device)
    all_results: dict[str, Any] = {}
    for set_name in evaluation_sets:
        LOGGER.info("[evaluate:%s] preparing fixed test set", set_name)
        definition = available[set_name]
        dataset = _evaluation_dataset(config, set_name)
        set_root = config.output_root / "evaluation" / set_name
        detector_predictions = load_or_create_detector_predictions(
            detector=detector,
            annotation_path=Path(definition["annotation"]),
            image_root=Path(definition["root"]),
            cache_path=set_root / "cache" / "deimv2_l_predictions.json",
            force=force_detector_cache,
        )
        variant_results: dict[str, dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
        diagnostics: dict[str, Any] = {}
        for variant in selected_variants:
            LOGGER.info("[evaluate:%s:%s] loading %s", set_name, variant, checkpoint_paths[variant])
            model, initialization = build_landmark_model(raw, variant=variant)
            checkpoint_report = load_landmark_checkpoint(model, checkpoint_paths[variant], prefer_ema=True, strict=True)
            model.to(device).eval()
            gt_result = evaluate_gt_boxes(
                model,
                dataset,
                device=device,
                batch_size=int(evaluation_cfg.get("batch_size", 64)),
                num_workers=int(evaluation_cfg.get("num_workers", 8)),
                input_size=int(raw["data"]["input_size"]),
                attention_batches=int(evaluation_cfg.get("attention_batches", 4)),
            )
            deim_result = evaluate_deim_boxes(
                model,
                annotation_path=Path(definition["annotation"]),
                image_root=Path(definition["root"]),
                detector_predictions=detector_predictions,
                device=device,
                batch_size=int(evaluation_cfg.get("batch_size", 64)),
                input_size=int(raw["data"]["input_size"]),
                crop_scale_factor=float(raw["data"]["augmentation"]["scale_factor"]),
                mean=raw["data"]["mean"],
                std=raw["data"]["std"],
                matching_iou_threshold=float(detector_cfg.get("matching_iou_threshold", 0.50)),
                attention_batches=int(evaluation_cfg.get("attention_batches", 4)),
                category_ids=definition.get("category_ids", raw["data"].get("category_ids", (1,))),
                include_missing_keypoint_instances=bool(
                    raw["data"].get(
                        "include_missing_keypoint_instances",
                        False,
                    )
                ),
            )
            variant_results[variant] = {"gt": gt_result, "deim": deim_result}
            diagnostics[variant] = {
                "checkpoint": checkpoint_report,
                "dino_initialization": initialization,
                "gradient_interference": _interference_diagnostic(
                    model,
                    dataset,
                    criterion,
                    device=device,
                    batch_size=int(evaluation_cfg.get("interference_batch_size", 8)),
                    maximum_parameters=int(evaluation_cfg.get("interference_maximum_parameters", 2_000_000)),
                ),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        condition_results: dict[str, Any] = {}
        condition_rows: dict[str, list[dict[str, Any]]] = {}
        selected_conditions = tuple(
            condition
            for condition in EXPERIMENT_CONDITIONS
            if condition.variant in selected_variants
        )
        for condition in selected_conditions:
            metrics, rows = variant_results[condition.variant][condition.bbox_mode]
            metrics = dict(metrics)
            metrics["condition"] = asdict(condition)
            condition_results[condition.condition] = metrics
            condition_rows[condition.condition] = rows
            directory = set_root / f"{condition.condition}_{condition.variant}_{condition.bbox_mode}"
            _write_json(directory / "metrics.json", metrics)
            if bool(evaluation_cfg.get("save_predictions", True)):
                _write_jsonl(directory / "predictions.jsonl", rows)

        for predicted_condition, oracle_condition in {
            "C": "A",
            "D": "B",
            "G": "F",
        }.items():
            if {
                predicted_condition,
                oracle_condition,
            }.issubset(condition_rows):
                condition_results[predicted_condition][
                    "paired_oracle_reference"
                ] = _paired_oracle_metrics(
                    condition_rows[oracle_condition],
                    condition_rows[predicted_condition],
                )
        if "E" in condition_rows:
            condition_results["E"][
                "paired_oracle_reference"
            ] = _paired_oracle_metrics(
                variant_results["query_refinement"]["gt"][1],
                condition_rows["E"],
            )
        for condition in ("C", "D", "E", "G"):
            if condition not in condition_results:
                continue
            directory = set_root / (
                f"{condition}_{condition_results[condition]['condition']['variant']}_deim"
            )
            _write_json(directory / "metrics.json", condition_results[condition])
        _write_json(set_root / "diagnostics.json", diagnostics)
        comparison = (
            build_comparison(set_name, condition_results)
            if set(condition_results) == set("ABCDEFG")
            else build_subset_comparison(set_name, condition_results)
        )
        write_comparison(set_root, comparison)
        if (
            set_name in {"v2_val", "v2_test"}
            and "query_refinement" in selected_variants
        ):
            split = "val" if set_name == "v2_val" else "test"
            directory = set_root / "E_query_refinement_deim"
            configured_common_root = evaluation_cfg.get(
                "baseline_aligned_output_root"
            )
            common_output_dir = (
                Path(str(configured_common_root)) / split / "baseline_aligned"
                if configured_common_root
                else directory / "baseline_aligned"
            )
            command = [
                sys.executable,
                str(
                    config.project_root
                    / "tools"
                    / "evaluate_deim_query_baseline_aligned.py"
                ),
                "--head-annotation",
                str(
                    config.project_root
                    / "outputs"
                    / "dataset_cache"
                    / "face_pose_drive_v2_balanced_811"
                    / "annotations"
                    / "face_dino_v1"
                    / f"{split}.json"
                ),
                "--image-root",
                str(config.project_root / "outputs/dataset_cache/face_pose_current"),
                "--detector-cache",
                str(set_root / "cache" / "deimv2_l_predictions.json"),
                "--query-checkpoint",
                str(checkpoint_paths["query_refinement"]),
                "--output-dir",
                str(directory / "baseline_aligned_diagnostic"),
                "--common-output-dir",
                str(common_output_dir),
                "--split",
                split,
                "--architecture-id",
                str(
                    evaluation_cfg.get(
                        "architecture_id",
                        "deimv2_l_query_refinement_factorized_state",
                    )
                ),
                "--device",
                str(device),
                "--batch-size",
                str(int(evaluation_cfg.get("batch_size", 64))),
                "--face-threshold",
                str(float(detector_cfg["score_threshold"])),
                "--point-threshold",
                "0.70",
            ]
            if split == "val":
                command.extend(
                    [
                        "--ignored-head-id",
                        "1219",
                        "--ignored-head-id",
                        "4717",
                    ]
                )
            subprocess.run(command, cwd=config.project_root, check=True)
        all_results[set_name] = comparison

    _write_json(config.output_root / "evaluation" / "all_results.json", all_results)
    _write_json(
        config.output_root / "evaluation" / "status.json",
        {
            "status": "complete",
            "evaluation_sets": list(evaluation_sets),
            "checkpoints": {
                variant: str(path)
                for variant, path in checkpoint_paths.items()
            },
        },
    )
    return all_results


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_subset_comparison(
    set_name: str,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for condition in EXPERIMENT_CONDITIONS:
        if condition.condition not in results:
            continue
        metrics = results[condition.condition]
        rows.append(
            {
                **asdict(condition),
                "instances": metrics.get("instances"),
                "missed_instances": metrics.get("missed_instances"),
                "keypoint_coverage": metrics.get("keypoint_coverage"),
                "nme": metrics.get("nme"),
                "pck_0.03": metrics.get("pck_0.03"),
                "pck_0.05": metrics.get("pck_0.05"),
                "pck_0.10": metrics.get("pck_0.10"),
                "end_to_end_pck_0.05": metrics.get(
                    "end_to_end_pck_0.05"
                ),
                "end_to_end_pck_0.10": metrics.get(
                    "end_to_end_pck_0.10"
                ),
            }
        )
    deltas: dict[str, dict[str, float | None]] = {}
    for name, left_key, right_key in (
        ("B_minus_F_query_vs_vitpose376_gt", "B", "F"),
        ("D_minus_G_query_vs_vitpose376_deim", "D", "G"),
    ):
        if left_key not in results or right_key not in results:
            continue
        left = results[left_key]
        right = results[right_key]
        deltas[name] = {
            metric: _delta(left.get(metric), right.get(metric))
            for metric in (
                "nme",
                "pck_0.05",
                "pck_0.10",
                "end_to_end_pck_0.05",
            )
        }
    return {
        "evaluation_set": set_name,
        "conditions": rows,
        "deltas": deltas,
        "full_metrics": dict(results),
    }


def build_comparison(set_name: str, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for condition in EXPERIMENT_CONDITIONS:
        metrics = results[condition.condition]
        rows.append(
            {
                **asdict(condition),
                "instances": metrics.get("instances"),
                "missed_instances": metrics.get("missed_instances"),
                "keypoint_coverage": metrics.get("keypoint_coverage"),
                "nme": metrics.get("nme"),
                "pck_0.03": metrics.get("pck_0.03"),
                "pck_0.05": metrics.get("pck_0.05"),
                "pck_0.10": metrics.get("pck_0.10"),
                "end_to_end_pck_0.05": metrics.get("end_to_end_pck_0.05"),
                "end_to_end_pck_0.10": metrics.get("end_to_end_pck_0.10"),
            }
        )
    a, b, c, d, e, f, g = (results[key] for key in "ABCDEFG")
    deltas = {
        "B_minus_A_query_design_oracle_boxes": {
            metric: _delta(b.get(metric), a.get(metric)) for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "F_minus_A_vitpose_capacity_gain_oracle_boxes": {
            metric: _delta(f.get(metric), a.get(metric)) for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "B_minus_F_query_vs_parameter_matched_vitpose_oracle_boxes": {
            metric: _delta(b.get(metric), f.get(metric)) for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "C_minus_A_predicted_bbox_vitpose": {
            metric: _delta(c.get(metric), c.get("paired_oracle_reference", {}).get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "D_minus_B_predicted_bbox_query": {
            metric: _delta(d.get(metric), d.get("paired_oracle_reference", {}).get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "D_minus_C_query_vs_vitpose_deployment": {
            metric: _delta(d.get(metric), c.get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10", "end_to_end_pck_0.05")
        },
        "G_minus_C_vitpose_capacity_gain_deployment": {
            metric: _delta(g.get(metric), c.get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10", "end_to_end_pck_0.05")
        },
        "D_minus_G_query_vs_parameter_matched_vitpose_deployment": {
            metric: _delta(d.get(metric), g.get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10", "end_to_end_pck_0.05")
        },
        "G_minus_F_predicted_bbox_parameter_matched_vitpose": {
            metric: _delta(g.get(metric), g.get("paired_oracle_reference", {}).get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10")
        },
        "E_minus_D_refinement_gain": {
            metric: _delta(e.get(metric), d.get(metric))
            for metric in ("nme", "pck_0.05", "pck_0.10", "end_to_end_pck_0.05")
        },
    }
    return {"evaluation_set": set_name, "conditions": rows, "deltas": deltas, "full_metrics": dict(results)}


def write_comparison(output_dir: Path, comparison: Mapping[str, Any]) -> None:
    _write_json(output_dir / "comparison.json", comparison)
    rows = list(comparison["conditions"])
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Landmark comparison: {comparison['evaluation_set']}",
        "",
        "Lower NME is better; higher PCK is better. DEIM conditions also report end-to-end PCK, which counts detector misses.",
        "",
        "| Condition | bbox | head | NME | PCK@.05 | PCK@.10 | coverage | end-to-end PCK@.05 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def value(key: str) -> str:
            item = row.get(key)
            return "-" if item is None else f"{float(item):.6f}"

        lines.append(
            f"| {row['condition']} | {row['bbox_mode']} | {row['variant']} | {value('nme')} | "
            f"{value('pck_0.05')} | {value('pck_0.10')} | {value('keypoint_coverage')} | "
            f"{value('end_to_end_pck_0.05')} |"
        )
    lines.extend(["", "## Controlled deltas", ""])
    for name, values in comparison["deltas"].items():
        lines.append(f"- `{name}`: " + ", ".join(f"{key}={value:+.6f}" if value is not None else f"{key}=n/a" for key, value in values.items()))
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_existing(config: LandmarkExperimentConfig, *, evaluation_sets: Sequence[str]) -> dict[str, Any]:
    all_results: dict[str, Any] = {}
    for set_name in evaluation_sets:
        set_root = config.output_root / "evaluation" / set_name
        metrics: dict[str, Any] = {}
        for condition in EXPERIMENT_CONDITIONS:
            path = set_root / f"{condition.condition}_{condition.variant}_{condition.bbox_mode}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing condition result: {path}")
            metrics[condition.condition] = json.loads(path.read_text(encoding="utf-8"))
        comparison = build_comparison(set_name, metrics)
        write_comparison(set_root, comparison)
        all_results[set_name] = comparison
    _write_json(config.output_root / "evaluation" / "all_results.json", all_results)
    return all_results


__all__ = [
    "EXPERIMENT_CONDITIONS",
    "ExperimentCondition",
    "build_comparison",
    "build_subset_comparison",
    "compare_existing",
    "evaluate_conditions",
    "preflight",
    "train_variants",
]
