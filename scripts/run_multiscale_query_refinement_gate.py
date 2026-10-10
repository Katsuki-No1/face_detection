#!/usr/bin/env python3
"""Run gates for DINO intermediate-layer Query Refinement variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PYTHON = PROJECT_ROOT / ".venv/bin/python"
TRAIN_SCRIPT = PROJECT_ROOT / "keypoint_comparison1/scripts/run_landmark_experiments.py"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_multiscale_query_refinement.yaml"
)
ARCHITECTURE_ID = "deimv2_l_multiscale_query_refinement_seed42"
SELECTION_METRIC = "val/conditional_semantic_macro_f1_nme_0.10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("schema", "gpu_step", "overfit", "small_scale"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--architecture-id", default=ARCHITECTURE_ID)
    parser.add_argument(
        "--feature-contract",
        choices=(
            "local_baseline",
            "multiscale",
            "multilevel_semantic",
            "conv_kv",
            "rgb_refinement",
            "messenger_control",
            "messenger",
        ),
        default="multiscale",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_metric_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.is_file():
        raise RuntimeError(f"training did not create metrics: {path}")
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                if value in {None, ""}:
                    continue
                try:
                    number = float(value)
                except ValueError:
                    row[key] = value
                    continue
                if not math.isfinite(number):
                    raise RuntimeError(f"non-finite metric {key}={value} in {path}")
                row[key] = number
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")
    return rows


def gpu_state() -> tuple[float, float, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    total, used, temperature = (
        float(item.strip()) for item in completed.stdout.strip().split(",")
    )
    return total, used, temperature


def run_training(
    *,
    config: Path,
    output_root: Path,
    epochs: int,
    max_train_samples: int,
    max_val_samples: int,
    batch_size: int,
    accumulation: int,
    batch_plan: Path | None = None,
    fixed_augmentation_epoch: int | None = None,
    disable_early_stopping: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--config",
        str(config),
        "--stage",
        "train",
        "--variants",
        "query_refinement",
        "--device",
        "cuda:0",
        "--output-root",
        str(output_root),
        "--epochs",
        str(epochs),
        "--max-train-samples",
        str(max_train_samples),
        "--max-val-samples",
        str(max_val_samples),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--val-every",
        "1",
        "--validation-batch-size",
        str(min(32, max_val_samples)),
        "--no-resume",
        "--no-wandb",
    ]
    if batch_plan is not None:
        command.extend(("--batch-plan", str(batch_plan)))
    if fixed_augmentation_epoch is not None:
        command.extend(("--fixed-augmentation-epoch", str(fixed_augmentation_epoch)))
    if disable_early_stopping:
        command.append("--disable-early-stopping")

    log_path = output_root / "gate_command.log"
    peak_used = 0.0
    peak_temperature = 0.0
    total_memory = 0.0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            total_memory, used, temperature = gpu_state()
            peak_used = max(peak_used, used)
            peak_temperature = max(peak_temperature, temperature)
            time.sleep(0.25)
        returncode = int(process.returncode or 0)
    elapsed = time.perf_counter() - started
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(f"training gate failed ({returncode}):\n{tail}")

    metrics_path = output_root / "query_refinement/metrics.csv"
    rows = read_metric_rows(metrics_path)
    overflow_skips = sum(
        float(row.get("train/amp_overflow_skips", 0.0)) for row in rows
    )
    if overflow_skips:
        raise RuntimeError(f"training had {overflow_skips:g} AMP-overflow skips")
    if peak_used > 0.90 * total_memory:
        raise RuntimeError(
            f"peak VRAM exceeded 90%: {peak_used:.0f}/{total_memory:.0f} MiB"
        )
    if peak_temperature >= 83.0:
        raise RuntimeError(
            f"GPU temperature reached {peak_temperature:.0f} C (limit <83 C)"
        )
    training_seconds = sum(
        float(row.get("timing/train_seconds", 0.0)) for row in rows
    )
    validation_seconds = sum(
        float(row.get("timing/validation_seconds", 0.0)) for row in rows
    )
    return {
        "command": command,
        "output_root": str(output_root),
        "metrics": str(metrics_path),
        "elapsed_seconds": elapsed,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "training_samples_per_second": (
            epochs * max_train_samples / max(training_seconds, 1.0e-12)
        ),
        "peak_vram_mib": peak_used,
        "total_vram_mib": total_memory,
        "peak_temperature_c": peak_temperature,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps_minimum": epochs
        * math.ceil(max_train_samples / max(1, batch_size * accumulation)),
        "rows": rows,
    }


def metric_improvement(
    rows: list[dict[str, float | str]],
    metric: str,
    *,
    direction: str,
    minimum_delta: float,
) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if metric in row]
    if len(values) < 2:
        raise RuntimeError(f"{metric} requires at least two validation points")
    initial = values[0]
    best = min(values[1:]) if direction == "lower" else max(values[1:])
    improvement = initial - best if direction == "lower" else best - initial
    if improvement <= minimum_delta:
        raise RuntimeError(
            f"{metric} did not improve: initial={initial}, best={best}, delta={improvement}"
        )
    return {
        "metric": metric,
        "direction": direction,
        "values": values,
        "initial": initial,
        "best_after_initial": best,
        "improvement": improvement,
    }


def remove_checkpoints(root: Path) -> list[str]:
    removed: list[str] = []
    for checkpoint in root.rglob("*.pt"):
        removed.append(str(checkpoint))
        checkpoint.unlink()
    return removed


def retain_best_checkpoint(root: Path) -> dict[str, list[str]]:
    checkpoint_dir = root / "query_refinement/checkpoints"
    best = checkpoint_dir / "best_conditional_semantic_macro_f1.pt"
    if not best.is_file():
        raise RuntimeError(f"successful gate has no validation-selected best: {best}")
    removed: list[str] = []
    for checkpoint in checkpoint_dir.glob("*.pt"):
        if checkpoint != best:
            removed.append(str(checkpoint))
            checkpoint.unlink()
    for previous in root.rglob(".previous_*"):
        if previous.is_dir():
            removed.append(str(previous))
            shutil.rmtree(previous)
    return {"retained": [str(best)], "removed": removed}


def schema_gate(
    config: Path,
    work_dir: Path,
    *,
    feature_contract: str,
) -> dict[str, Any]:
    initialization_details: dict[str, Any] | None = None
    unit_test = {
        "local_baseline": "legacy_query_checkpoint_contract",
        "multiscale": "multiscale_query_refinement",
        "multilevel_semantic": "multilevel_semantic_query_refinement",
        "conv_kv": "conv_kv_query_refinement",
        "rgb_refinement": "rgb_cnn_refinement",
        "messenger_control": "messenger_recovery",
        "messenger": "messenger_recovery",
    }[feature_contract]
    if feature_contract in {"messenger_control", "messenger"}:
        test_command = [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(PROJECT_ROOT / "keypoint_comparison1/tests/test_messenger_recovery.py"),
        ]
    else:
        test_command = [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(PROJECT_ROOT / "keypoint_comparison1/tests/test_landmark_detection.py"),
            "-k",
            f"{unit_test} or legacy_query_checkpoint_contract",
        ]
    completed = subprocess.run(test_command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{feature_contract} Query Refinement unit tests failed"
        )
    from keypoint_comparison1.landmark_detection.config import load_landmark_config

    loaded = load_landmark_config(config, project_root=PROJECT_ROOT).raw
    selection = loaded["train"]["checkpoint_selection"]
    if feature_contract == "local_baseline":
        feature_definition = validate_local_baseline_contract(loaded)
        expected_definition = {"architecture": "local_baseline_query_refinement", "head_free": True}
        subprocess.run([
            str(PYTHON), str(PROJECT_ROOT / "tools/run_training_schema_gate.py"),
            "--kind", "keypoint_comparison1", "--config", str(config),
            "--success-marker", str(work_dir / "inventory_success.json"),
        ], cwd=PROJECT_ROOT, check=True)
    elif feature_contract == "multiscale":
        feature_definition = loaded["model"]["feature_pyramid"]
        expected_definition = {
            "enabled": True,
            "intermediate_layers": [5, 8, 11],
            "output_strides": [8, 16, 32],
            "out_channels": 384,
        }
    elif feature_contract == "multilevel_semantic":
        feature_definition = loaded["model"]["semantic_feature_levels"]
        expected_definition = {
            "enabled": True,
            "intermediate_layers": [5, 8, 11],
            "output_stride": 16,
            "fusion": "decoder_memory_concat",
        }
        if loaded["model"].get("feature_pyramid", {}).get("enabled", False):
            raise RuntimeError(
                "multi-level semantic contract must not enable feature_pyramid"
            )
    elif feature_contract == "conv_kv":
        feature_definition = loaded["model"]["kv_feature_pyramid"]
        expected_definition = {
            "enabled": True,
            "output_strides": [16, 32, 64],
            "semantic_fusion": "learned_per_scale",
        }
        semantic_definition = loaded["model"]["semantic_feature_levels"]
        if semantic_definition.get("intermediate_layers") != [5, 8, 11]:
            raise RuntimeError(
                f"unexpected Conv K/V semantic levels: {semantic_definition}"
            )
        cache = Path(loaded["data"]["validation_detector_crop_cache"])
        if not cache.is_file():
            raise RuntimeError(
                f"locked predicted-Face validation cache is missing: {cache}"
            )
        if float(loaded["data"]["validation_detector_matching_iou"]) != 0.50:
            raise RuntimeError("predicted-Face validation matching IoU must be 0.50")
        provenance = loaded["data"]["validation_detector_crop_provenance"]
        cache_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
        if cache_sha256 != provenance.get("cache_sha256"):
            raise RuntimeError(
                "predicted-Face validation cache SHA-256 changed: "
                f"actual={cache_sha256}, expected={provenance.get('cache_sha256')}"
            )
        cache_metadata = json.loads(cache.read_text(encoding="utf-8"))["metadata"]
        for metadata_key in (
            "detector_checkpoint_sha256",
            "detector_config_sha256",
            "score_threshold",
        ):
            if cache_metadata.get(metadata_key) != provenance.get(metadata_key):
                raise RuntimeError(
                    f"predicted-Face provenance mismatch for {metadata_key}: "
                    f"cache={cache_metadata.get(metadata_key)!r}, "
                    f"config={provenance.get(metadata_key)!r}"
                )
    elif feature_contract == "rgb_refinement":
        feature_definition = loaded["model"]["rgb_refinement"]
        expected_definition = {
            "enabled": True,
            "base_channels": 32,
            "projection_channels": 128,
            "feature_sources": ["stride8", "stride4"],
            "fusion": "refinement_only",
            "normalization": "group_norm",
            "initialize_from_deim_checkpoint": feature_definition.get(
                "initialize_from_deim_checkpoint"
            ),
            "initialize_from_deim_checkpoint_sha256": (
                "ffa0ae9fe63df3f7af7ff7b772b80717243dcf544b2c9fcdf944de8899ddd8b7"
            ),
            "strict_initialization": True,
            "freeze": False,
        }
        semantic_definition = loaded["model"]["semantic_feature_levels"]
        if semantic_definition != {
            "enabled": True,
            "intermediate_layers": [5, 8, 11],
            "output_stride": 16,
            "fusion": "decoder_memory_concat",
        }:
            raise RuntimeError(
                f"unexpected RGB refinement semantic levels: {semantic_definition}"
            )
        if int(loaded["data"]["input_size"]) != 256:
            raise RuntimeError("RGB refinement requires the locked 256x256 crop")
        expected_stages = [
            {
                "feature_source": "stride8",
                "radius_in_feature_cells": 2.0,
                "maximum_delta": 0.08,
                "samples_per_axis": 3,
            },
            {
                "feature_source": "stride4",
                "radius_in_feature_cells": 2.0,
                "maximum_delta": 0.04,
                "samples_per_axis": 3,
            },
        ]
        stages = loaded["model"]["variants"]["query_refinement"]["stages"]
        if stages != expected_stages:
            raise RuntimeError(f"unexpected RGB refinement stages: {stages}")
        if any(
            loaded["model"].get(name, {}).get("enabled", False)
            for name in ("feature_pyramid", "kv_feature_pyramid")
        ):
            raise RuntimeError(
                "RGB refinement must not enter the DINO decoder memory"
            )
        deim_checkpoint = Path(
            feature_definition["initialize_from_deim_checkpoint"]
        )
        actual_deim_sha256 = hashlib.sha256(deim_checkpoint.read_bytes()).hexdigest()
        if actual_deim_sha256 != feature_definition.get(
            "initialize_from_deim_checkpoint_sha256"
        ):
            raise RuntimeError(
                "RGB refinement DEIM initialization checkpoint SHA-256 changed"
            )
        finetune = loaded["train"]["finetune_checkpoint"]
        baseline_checkpoint = Path(finetune["path"])
        actual_baseline_sha256 = hashlib.sha256(
            baseline_checkpoint.read_bytes()
        ).hexdigest()
        if actual_baseline_sha256 != finetune.get("sha256"):
            raise RuntimeError("local-baseline warm-start checkpoint SHA-256 changed")
        cache = Path(loaded["data"]["validation_detector_crop_cache"])
        if not cache.is_file():
            raise RuntimeError(
                f"locked predicted-Face validation cache is missing: {cache}"
            )
        if float(loaded["data"]["validation_detector_matching_iou"]) != 0.50:
            raise RuntimeError("predicted-Face validation matching IoU must be 0.50")
        provenance = loaded["data"]["validation_detector_crop_provenance"]
        cache_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
        if cache_sha256 != provenance.get("cache_sha256"):
            raise RuntimeError(
                "predicted-Face validation cache SHA-256 changed: "
                f"actual={cache_sha256}, expected={provenance.get('cache_sha256')}"
            )
        cache_metadata = json.loads(cache.read_text(encoding="utf-8"))["metadata"]
        for metadata_key in (
            "detector_checkpoint_sha256",
            "detector_config_sha256",
            "score_threshold",
        ):
            if cache_metadata.get(metadata_key) != provenance.get(metadata_key):
                raise RuntimeError(
                    f"predicted-Face provenance mismatch for {metadata_key}: "
                    f"cache={cache_metadata.get(metadata_key)!r}, "
                    f"config={provenance.get(metadata_key)!r}"
                )
        from keypoint_comparison1.landmark_detection.config import (
            configured_keypoint_target_policy,
            configured_state_head_mode,
        )
        from keypoint_comparison1.landmark_detection.model import (
            build_landmark_model,
        )
        from keypoint_comparison1.landmark_detection.training import (
            initialize_from_finetune_checkpoint,
        )

        model, detector_initialization = build_landmark_model(
            loaded,
            variant="query_refinement",
        )
        finetune_initialization = initialize_from_finetune_checkpoint(
            model,
            finetune,
            variant="query_refinement",
            expected_target_policy=configured_keypoint_target_policy(loaded),
            expected_state_head_mode=configured_state_head_mode(loaded),
        )
        if finetune_initialization["unexpected"]:
            raise RuntimeError(
                "local-baseline warm start produced unexpected tensors"
            )
        rgb_initialization = dict(detector_initialization or {}).get(
            "rgb_refinement", {}
        )
        if len(rgb_initialization.get("loaded_tensors", ())) != 6:
            raise RuntimeError(
                "RGB refinement did not copy all six shape-matching STA tensors"
            )
        dino_loaded_tensors = dict(detector_initialization or {}).get(
            "loaded_tensors"
        )
        if not isinstance(dino_loaded_tensors, int) or dino_loaded_tensors <= 0:
            raise RuntimeError("DINO initialization did not load detector tensors")
        initialization_details = {
            "dino_loaded_tensors": dino_loaded_tensors,
            "rgb_sta_initialization": rgb_initialization,
            "local_baseline_warm_start": finetune_initialization,
        }
    else:
        from keypoint_comparison1.landmark_detection.config import (
            TWO_STAGE_FACE_KEYPOINT_PIPELINE,
            configured_evaluation_pipeline_stages,
            configured_keypoint_target_policy,
            configured_state_head_mode,
        )
        from keypoint_comparison1.landmark_detection.model import build_landmark_model
        from keypoint_comparison1.landmark_detection.training import (
            initialize_from_finetune_checkpoint,
        )

        feature_definition = loaded["model"]["messenger_recovery"]
        expected_definition = {
            "enabled": feature_contract == "messenger",
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
        }
        if configured_evaluation_pipeline_stages(loaded) != TWO_STAGE_FACE_KEYPOINT_PIPELINE:
            raise RuntimeError("Messenger experiment must remain HEAD-free two-stage")
        paired = loaded["data"]["augmentation"].get("paired_occlusion")
        if not isinstance(paired, dict) or float(paired.get("probability", 0.0)) != 0.30:
            raise RuntimeError("Messenger experiment requires paired occlusion at probability 0.30")
        if paired.get("kinds") != ["rectangle", "irregular", "boundary"]:
            raise RuntimeError(f"unexpected paired occlusion kinds: {paired}")
        if loaded["data"]["augmentation"].get("landmark_guided_occlusion") is not None:
            raise RuntimeError("legacy landmark occlusion must be disabled")
        if float(loaded["data"]["augmentation"].get("occlusion_probability", -1.0)) != 0.0:
            raise RuntimeError("legacy random occlusion must be disabled")
        finetune = loaded["train"]["finetune_checkpoint"]
        baseline_checkpoint = Path(finetune["path"])
        actual_baseline_sha256 = hashlib.sha256(
            baseline_checkpoint.read_bytes()
        ).hexdigest()
        if actual_baseline_sha256 != (
            "baf7a6d0b1df6d1089357b2f28ffc1b2da3aadbb6239bae2cb5085d36ef016e0"
        ):
            raise RuntimeError("local-baseline warm-start checkpoint SHA-256 changed")
        model, _ = build_landmark_model(
            loaded,
            variant="query_refinement",
            initialize_from_detector=False,
        )
        finetune_initialization = initialize_from_finetune_checkpoint(
            model,
            finetune,
            variant="query_refinement",
            expected_target_policy=configured_keypoint_target_policy(loaded),
            expected_state_head_mode=configured_state_head_mode(loaded),
        )
        expected_missing = (
            feature_contract == "messenger"
            and bool(finetune_initialization["missing"])
            and all(
                key.startswith("messenger_recovery.")
                for key in finetune_initialization["missing"]
            )
        )
        if feature_contract == "messenger" and not expected_missing:
            raise RuntimeError("Messenger warm-start did not isolate new recovery tensors")
        if feature_contract == "messenger_control" and (
            finetune_initialization["missing"] or finetune_initialization["unexpected"]
        ):
            raise RuntimeError("Control warm-start must be exactly checkpoint-compatible")
        initialization_details = {
            "local_baseline_warm_start": finetune_initialization,
            "paired_occlusion": paired,
            "pipeline_stages": list(TWO_STAGE_FACE_KEYPOINT_PIPELINE),
        }
    if feature_definition != expected_definition:
        raise RuntimeError(
            f"unexpected {feature_contract} feature contract: "
            f"{feature_definition}"
        )
    if selection != {
        "metric": "conditional_semantic_macro_f1_nme_0.10",
        "direction": "higher",
        "filename": "best_conditional_semantic_macro_f1.pt",
    }:
        raise RuntimeError(f"unexpected checkpoint-selection contract: {selection}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "test_command": test_command,
        "feature_contract": feature_contract,
        "feature_definition": feature_definition,
        "selection": selection,
        "initialization": initialization_details,
    }


def validate_local_baseline_contract(loaded: dict[str, Any]) -> dict[str, Any]:
    """Validate a new detector-bound run without enabling architecture variants."""
    from keypoint_comparison1.landmark_detection.config import (
        configured_evaluation_pipeline_stages, TWO_STAGE_FACE_KEYPOINT_PIPELINE,
    )
    if configured_evaluation_pipeline_stages(loaded) != TWO_STAGE_FACE_KEYPOINT_PIPELINE:
        raise RuntimeError("Local Query Refinement must remain HEAD-free")
    if any(loaded["model"].get(k, {}).get("enabled", False) for k in (
        "feature_pyramid", "semantic_feature_levels", "kv_feature_pyramid",
        "rgb_refinement", "messenger_recovery", "continuous_recovery")):
        raise RuntimeError("Unexpected architecture extension in local Query Refinement")
    checkpoint = Path(loaded["model"]["backbone"]["initialize_from_deim_checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != loaded["model"]["backbone"].get("initialize_from_deim_checkpoint_sha256"):
        raise RuntimeError("Detector backbone initialization hash mismatch")
    if checkpoint.resolve() != Path(loaded["detector"]["checkpoint"]).resolve():
        raise RuntimeError("Crop detector and backbone initialization must agree")
    cache = Path(loaded["data"]["validation_detector_crop_cache"])
    provenance = loaded["data"]["validation_detector_crop_provenance"]
    if hashlib.sha256(cache.read_bytes()).hexdigest() != provenance.get("cache_sha256"):
        raise RuntimeError("Predicted-Face validation cache hash mismatch")
    metadata = json.loads(cache.read_text())["metadata"]
    for key in ("detector_checkpoint_sha256", "detector_config_sha256", "score_threshold"):
        if metadata.get(key) != provenance.get(key):
            raise RuntimeError(f"Predicted-Face provenance mismatch: {key}")
    if (metadata.get("detector_checkpoint_sha256") != digest
            or metadata.get("detector_config_sha256") != hashlib.sha256(
                Path(loaded["detector"]["config"]).read_bytes()).hexdigest()
            or metadata.get("score_threshold") != loaded["detector"]["score_threshold"]
            or metadata.get("split") != "val"):
        raise RuntimeError("Validation crop cache belongs to a different detector/threshold")
    return {"architecture": "local_baseline_query_refinement", "head_free": True}


def gpu_step_gate(
    config: Path,
    work_dir: Path,
    *,
    architecture_id: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    effective_batch = 64
    for batch_size in (4, 8, 16):
        try:
            result = run_training(
                config=config,
                output_root=work_dir / f"batch_{batch_size}",
                epochs=1,
                max_train_samples=128,
                max_val_samples=64,
                batch_size=batch_size,
                accumulation=effective_batch // batch_size,
            )
            if int(result["optimizer_steps_minimum"]) < 2:
                raise RuntimeError("candidate did not exercise repeated optimizer steps")
            projected_seconds = (
                float(result["training_seconds"]) / 128.0 * 6529.0 * 40.0
                + float(result["validation_seconds"]) / 64.0 * 532.0 * 40.0
            )
            result["status"] = "safe"
            result["projected_production_seconds"] = projected_seconds
        except Exception as error:
            result = {
                "status": "unsafe",
                "batch_size": batch_size,
                "gradient_accumulation_steps": effective_batch // batch_size,
                "error": str(error),
            }
        attempts.append(result)
    safe = [attempt for attempt in attempts if attempt["status"] == "safe"]
    if not safe:
        raise RuntimeError("No safe Query Refinement microbatch candidate")
    fastest = min(safe, key=lambda item: float(item["projected_production_seconds"]))
    equivalent = [
        item
        for item in safe
        if float(item["projected_production_seconds"])
        <= float(fastest["projected_production_seconds"]) * 1.03
    ]
    selected = min(equivalent, key=lambda item: int(item["batch_size"]))
    plan = {
        "status": "passed",
        "architecture_id": architecture_id,
        "candidates": [4, 8, 16],
        "effective_batch_size": effective_batch,
        "selected_batch_size": int(selected["batch_size"]),
        "selected_gradient_accumulation_steps": int(
            selected["gradient_accumulation_steps"]
        ),
        "selection_rule": "shortest_projected_elapsed_within_3_percent_choose_smaller",
        "attempts": attempts,
    }
    write_json(work_dir / "batch_plan.json", plan)
    removed = remove_checkpoints(work_dir)
    return {"batch_plan": plan, "removed_checkpoints": removed}


def learning_gate(
    config: Path,
    work_dir: Path,
    *,
    small_scale: bool,
) -> dict[str, Any]:
    plan_path = work_dir.parent / "gpu_step/batch_plan.json"
    if small_scale:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("status") != "passed" or plan.get("candidates") != [4, 8, 16]:
            raise RuntimeError(f"GPU batch plan is invalid: {plan_path}")
        batch_size = int(plan["selected_batch_size"])
        accumulation = int(plan["selected_gradient_accumulation_steps"])
    else:
        batch_size = 1
        accumulation = 1
    result = run_training(
        config=config,
        output_root=work_dir,
        epochs=20 if small_scale else 12,
        max_train_samples=256 if small_scale else 4,
        max_val_samples=128 if small_scale else 16,
        batch_size=batch_size,
        accumulation=accumulation,
        batch_plan=plan_path if small_scale else None,
        fixed_augmentation_epoch=None if small_scale else 0,
        disable_early_stopping=small_scale,
    )
    assessment = metric_improvement(
        result["rows"],
        SELECTION_METRIC if small_scale else "train/loss_total",
        direction="higher" if small_scale else "lower",
        minimum_delta=1.0e-4 if small_scale else 0.0,
    )
    retention = retain_best_checkpoint(work_dir)
    write_json(work_dir / "checkpoint_retention.json", retention)
    return {"run": result, "assessment": assessment, "checkpoint_retention": retention}


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    work_dir = args.work_dir.resolve()
    marker = args.success_marker.resolve()
    marker.unlink(missing_ok=True)
    started = time.time()
    if args.stage == "schema":
        details = schema_gate(
            config,
            work_dir,
            feature_contract=args.feature_contract,
        )
    elif args.stage == "gpu_step":
        details = gpu_step_gate(
            config,
            work_dir,
            architecture_id=args.architecture_id,
        )
    elif args.stage == "overfit":
        details = learning_gate(config, work_dir, small_scale=False)
    else:
        details = learning_gate(config, work_dir, small_scale=True)
    payload = {
        "status": "passed",
        "architecture": args.architecture_id,
        "gate_stage": args.stage,
        "config": str(config),
        "elapsed_seconds": time.time() - started,
        "details": details,
    }
    write_json(work_dir / "report.json", payload)
    write_json(marker, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
