from __future__ import annotations

import csv
import json
import logging
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
from .model import VARIANTS, build_landmark_model
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

    LOGGER.info("[preflight] checking DINOv3 initialization and A-E model graph")
    val_dataset = TopDownFaceLandmarkDataset(
        annotation_path=raw["data"]["annotations"]["val"],
        image_root=raw["data"]["root"],
        input_size=int(raw["data"]["input_size"]),
        training=False,
        augmentation=CropAugmentation(scale_factor=float(raw["data"]["augmentation"]["scale_factor"])),
        mean=raw["data"]["mean"],
        std=raw["data"]["std"],
        category_ids=raw["data"]["category_ids"],
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
            output = model(sample, return_attention=variant != "vitpose")
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
        "data": inventories,
        "split_leakage_audit": split_audit,
        "evaluation_sets": evaluation_inventory,
        "models": model_reports,
        "detector": detector_report,
        "wandb": _check_wandb(config, required=require_wandb),
        "conditions": [asdict(condition) for condition in EXPERIMENT_CONDITIONS],
    }
    _write_json(output_dir / "report.json", report)
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
) -> dict[str, Any]:
    raw = config.raw
    available = raw["data"].get("evaluation_sets", {})
    unknown = sorted(set(evaluation_sets) - set(available))
    if unknown:
        raise ValueError(f"Unknown evaluation sets: {unknown}; available={sorted(available)}")
    checkpoint_paths = {
        variant: config.output_root / variant / "checkpoints" / "best_nme.pt" for variant in VARIANTS
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot evaluate before all three heads are trained. Missing: " + ", ".join(missing))

    detector_cfg = raw["detector"]
    detector = DeimFaceDetector(
        config_path=detector_cfg["config"],
        checkpoint_path=detector_cfg["checkpoint"],
        device=str(device),
        image_size=int(detector_cfg["image_size"]),
        score_threshold=float(detector_cfg["score_threshold"]),
        maximum_detections=int(detector_cfg["maximum_detections"]),
    )
    evaluation_cfg = raw.get("evaluation", {})
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
        for variant in VARIANTS:
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
        for condition in EXPERIMENT_CONDITIONS:
            metrics, rows = variant_results[condition.variant][condition.bbox_mode]
            metrics = dict(metrics)
            metrics["condition"] = asdict(condition)
            condition_results[condition.condition] = metrics
            condition_rows[condition.condition] = rows
            directory = set_root / f"{condition.condition}_{condition.variant}_{condition.bbox_mode}"
            _write_json(directory / "metrics.json", metrics)
            if bool(evaluation_cfg.get("save_predictions", True)):
                _write_jsonl(directory / "predictions.jsonl", rows)

        condition_results["C"]["paired_oracle_reference"] = _paired_oracle_metrics(
            condition_rows["A"], condition_rows["C"]
        )
        condition_results["D"]["paired_oracle_reference"] = _paired_oracle_metrics(
            condition_rows["B"], condition_rows["D"]
        )
        condition_results["E"]["paired_oracle_reference"] = _paired_oracle_metrics(
            variant_results["query_refinement"]["gt"][1], condition_rows["E"]
        )
        for condition in ("C", "D", "E"):
            directory = set_root / (
                f"{condition}_{condition_results[condition]['condition']['variant']}_deim"
            )
            _write_json(directory / "metrics.json", condition_results[condition])
        _write_json(set_root / "diagnostics.json", diagnostics)
        comparison = build_comparison(set_name, condition_results)
        write_comparison(set_root, comparison)
        all_results[set_name] = comparison

    _write_json(config.output_root / "evaluation" / "all_results.json", all_results)
    return all_results


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


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
    a, b, c, d, e = (results[key] for key in "ABCDE")
    deltas = {
        "B_minus_A_query_design_oracle_boxes": {
            metric: _delta(b.get(metric), a.get(metric)) for metric in ("nme", "pck_0.05", "pck_0.10")
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
    "compare_existing",
    "evaluate_conditions",
    "preflight",
    "train_variants",
]
