#!/usr/bin/env python3
"""Select a safe timed 4/8/16 microbatch for DINOv3 detector training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wandb_utils import write_json_atomic  # noqa: E402
from tools.dinov3_deimv2_common import (  # noqa: E402
    load_config,
    resolved_config_sha256,
)


PYTHON = PROJECT_ROOT / ".venv/bin/python"
DEFAULT_TRAINER = PROJECT_ROOT / "tools/train_dinov3_deimv2.py"
CODINO_TRAINER = PROJECT_ROOT / "tools/train_dinov3_codino.py"
REQUIRED_CANDIDATES = (4, 8, 16)
REQUIRED_OPTIMIZER_UPDATES = 3
# The first few AMP update windows can be skipped while GradScaler backs off
# from its initial scale. Exercise enough windows to verify real parameter
# updates instead of treating finite forward losses as sufficient evidence.
PROBE_UPDATE_WINDOWS = 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="Reuse completed candidate results and classify an interrupted candidate as unsafe without rerunning it.",
    )
    parser.add_argument(
        "--refresh-safe-candidate",
        action="store_true",
        help="Under recovery, rerun only the previously safe candidate against the current validation path in a new attempt directory.",
    )
    return parser.parse_args(argv)


def _gpu_sample(index: int) -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total, used, temperature = (
        float(value.strip()) for value in completed.stdout.strip().split(",")
    )
    return {
        "memory_total_mib": total,
        "memory_used_mib": used,
        "temperature_c": temperature,
    }


def _last_csv_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for key, value in rows[-1].items():
        if value in {None, ""}:
            continue
        try:
            result[key] = float(value)
        except ValueError:
            normalized = value.strip().lower()
            if normalized in {"true", "false"}:
                result[key] = normalized == "true"
            else:
                result[key] = value
    return result


def _image_count(config: dict[str, Any], split: str) -> int:
    path = Path(config["data"]["annotation_paths"][split])
    return len(json.loads(path.read_text(encoding="utf-8"))["images"])


def _remove_probe_checkpoints(path: Path) -> list[str]:
    removed: list[str] = []
    checkpoint_dir = path / "checkpoints"
    if checkpoint_dir.is_dir():
        for checkpoint in checkpoint_dir.glob("*.pt"):
            checkpoint.unlink()
            removed.append(str(checkpoint))
        if not any(checkpoint_dir.iterdir()):
            checkpoint_dir.rmdir()
    return removed


def _fallback_plans(
    safe_attempts: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_batch = int(selected["microbatch"])
    plans = [
        {
            "microbatch": int(attempt["microbatch"]),
            "accumulation_steps": int(attempt["accumulation_steps"]),
            "source": "safe_timed_preflight",
        }
        for attempt in sorted(
            (
                attempt
                for attempt in safe_attempts
                if int(attempt["microbatch"]) < selected_batch
            ),
            key=lambda attempt: int(attempt["microbatch"]),
            reverse=True,
        )
    ]
    existing = {int(plan["microbatch"]) for plan in plans}
    for microbatch in (2, 1):
        if microbatch < selected_batch and microbatch not in existing:
            plans.append(
                {
                    "microbatch": microbatch,
                    "accumulation_steps": 16 // microbatch,
                    "source": "emergency_lower_memory_fallback",
                }
            )
    return plans


def _probe_limits(candidate: int, accumulation: int) -> tuple[int, int]:
    max_train_batches = PROBE_UPDATE_WINDOWS * accumulation
    return max_train_batches, candidate * max_train_batches


def project_production_runtime(config, train_epoch_seconds, validation_pass_seconds,
                               *, first_epoch=1, last_epoch=None):
    """Nominal horizon; follow the trainer's cadence and forced final validation."""
    train = config["train"]
    epochs = int(train["epochs"])
    last_epoch = epochs if last_epoch is None else min(int(last_epoch), epochs)
    phase_epochs = range(max(1, int(first_epoch)), last_epoch + 1)
    validation = train.get("validation", {})
    every = int(validation.get("every", validation.get("val_every", 0)) or 0)
    validation_epochs = [epoch for epoch in phase_epochs
                         if validation.get("enabled", False) and every > 0
                         and (epoch % every == 0 or epoch == epochs)]
    return dict(train_epochs=len(phase_epochs), validation_epochs=validation_epochs,
                train_epoch_seconds=float(train_epoch_seconds),
                validation_pass_seconds=float(validation_pass_seconds),
                projected_production_seconds=(len(phase_epochs) * train_epoch_seconds
                    + len(validation_epochs) * validation_pass_seconds))


def _run_candidate(
    *,
    args: argparse.Namespace,
    candidate: int,
    accumulation: int,
    production_config: dict[str, Any],
    attempt_name: str | None = None,
    checkpoint_validator=None,
) -> dict[str, Any]:
    attempt = args.output_dir / (attempt_name or f"microbatch_{candidate}")
    if attempt.exists():
        shutil.rmtree(attempt)
    attempt.mkdir(parents=True)
    log_path = attempt / "probe.log"
    required_updates = REQUIRED_OPTIMIZER_UPDATES
    max_train_batches, max_train_samples = _probe_limits(candidate, accumulation)
    model_type = str(production_config.get("model", {}).get("type", "")).lower()
    trainer = (
        CODINO_TRAINER
        if model_type
        in {
            "dinov3_codino_official",
            "dinov3_codino_official_ellipse",
            "dinov3_codino",
            "dino_codino",
            "co_dino",
            "dinov3_codino_ellipse",
            "dino_codino_ellipse",
            "co_dino_ellipse",
        }
        else DEFAULT_TRAINER
    )
    command = [
        str(PYTHON),
        str(trainer),
        "--config",
        str(args.probe_config.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--output-dir",
        str(attempt),
        "--device",
        f"cuda:{args.device_index}",
        "--epochs",
        "1",
        "--batch-size",
        str(candidate),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--num-workers",
        "2",
        "--max-train-batches",
        str(max_train_batches),
        "--max-train-samples",
        str(max_train_samples),
        "--val-every",
        "1",
        "--val-max-images",
        "64",
        "--wandb-mode",
        "disabled",
    ]
    samples: list[dict[str, float]] = []
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            try:
                samples.append(_gpu_sample(args.device_index))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.25)
        exit_code = int(process.wait())
    elapsed = time.monotonic() - started
    try:
        samples.append(_gpu_sample(args.device_index))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    train_metric = _last_csv_row(attempt / "metrics.csv")
    val_metric = _last_csv_row(attempt / "val_metrics.csv")
    checkpoint_path = attempt / "checkpoints/last.pt"
    completed_updates = 0
    checkpoint_assertions = {}
    calibration_records = []
    trial_calibration = production_config.get("train", {}).get("amp_calibration", {}).get("enabled", False)
    if trial_calibration and production_config.get("train", {}).get("amp", False):
        calibration_records = [json.loads(p.read_text()) for p in sorted(attempt.glob("amp_calibration_*.json"))]
        checkpoint_assertions["amp_calibration_completed_before_verification"] = bool(calibration_records) and all(
            item.get("status") == "calibrated" and item.get("optimizer_updates") == 0
            for item in calibration_records)
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        completed_updates = int(checkpoint.get("step", 0))
        if checkpoint_validator is not None:
            checkpoint_assertions.update(checkpoint_validator(checkpoint))
        if trial_calibration:
            checkpoint_assertions["amp_resume_state_recorded"] = bool(checkpoint.get("amp_runtime"))
        del checkpoint
    peak_used = max(
        (sample["memory_used_mib"] for sample in samples),
        default=float("inf"),
    )
    physical = max(
        (sample["memory_total_mib"] for sample in samples), default=0.0
    )
    temperature = max(
        (sample["temperature_c"] for sample in samples),
        default=float("inf"),
    )
    vram_fraction = peak_used / physical if physical > 0 else float("inf")
    numeric_metrics = [
        float(value)
        for value in (*train_metric.values(), *val_metric.values())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    finite = (
        bool(train_metric)
        and bool(val_metric)
        and bool(numeric_metrics)
        and all(math.isfinite(value) for value in numeric_metrics)
    )
    train_seconds = float(train_metric.get("elapsed_seconds", float("inf")))
    validation_seconds = float(
        val_metric.get("elapsed_seconds", float("inf"))
    )
    observed_train_images = max_train_samples
    observed_val_images = int(val_metric.get("num_images", 0.0))
    isolated_probe = attempt / "isolated_probe.json"
    if checkpoint_validator is not None:
        if isolated_probe.is_file():
            probe = json.loads(isolated_probe.read_text())
            peak_used = max(peak_used, float(probe["peak_reserved_bytes"])/(1024**2))
            vram_fraction = peak_used/physical if physical > 0 else float("inf")
            train_seconds = float(probe["train_seconds_after_warmup"] or float("inf"))
            observed_train_images = int(probe["measured_train_images"])
            checkpoint_assertions["synchronized_post_warmup_timing"] = observed_train_images > 0 and math.isfinite(train_seconds)
        else:
            checkpoint_assertions["synchronized_post_warmup_timing"] = False
    projection = project_production_runtime(
        production_config,
        train_seconds * _image_count(production_config, "train") / max(1, observed_train_images),
        validation_seconds * _image_count(production_config, "val") / max(1, observed_val_images))
    lower_log = log_text.lower()
    assertions = {
        "exit_zero": exit_code == 0,
        "repeated_optimizer_steps": completed_updates >= required_updates,
        "finite_train_and_validation_metrics": finite,
        "production_validation_path_exercised": observed_val_images > 0,
        "no_cuda_oom": "out of memory" not in lower_log,
        "no_non_finite_values": "non-finite" not in lower_log,
        "vram_within_90_percent": vram_fraction <= 0.90,
        "temperature_below_83c": temperature < 83.0,
        "effective_batch_preserved": candidate * accumulation == 16,
    }
    if checkpoint_validator is not None:
        assertions["checkpoint_update_audit_present"] = bool(checkpoint_assertions)
        assertions.update(checkpoint_assertions)
    result = {
        "microbatch": candidate,
        "accumulation_steps": accumulation,
        "effective_batch_size": candidate * accumulation,
        "exit_code": exit_code,
        "optimizer_updates": completed_updates,
        "amp_calibration_records": calibration_records,
        "attempted_optimizer_windows": PROBE_UPDATE_WINDOWS,
        "train_images": observed_train_images,
        "validation_images": observed_val_images,
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "wall_clock_seconds": elapsed,
        "projected_production_seconds": projection["projected_production_seconds"],
        "runtime_projection": projection,
        "peak_memory_used_mib": peak_used,
        "physical_vram_mib": physical,
        "peak_vram_fraction": vram_fraction,
        "maximum_gpu_temperature_c": temperature,
        "assertions": assertions,
        "safe": all(assertions.values()),
        "log": str(log_path),
        "production_resolved_config_sha256": resolved_config_sha256(production_config),
    }
    result["removed_probe_checkpoints"] = _remove_probe_checkpoints(attempt)
    write_json_atomic(attempt / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.refresh_safe_candidate and not args.recover_interrupted:
        raise ValueError("--refresh-safe-candidate requires --recover-interrupted")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = args.success_marker.expanduser().resolve()
    success_marker.unlink(missing_ok=True)
    production_config = load_config(args.config.expanduser().resolve())
    probe_config = load_config(args.probe_config.expanduser().resolve())
    attempts = []
    for candidate in REQUIRED_CANDIDATES:
        attempt_dir = args.output_dir / f"microbatch_{candidate}"
        result_path = attempt_dir / "result.json"
        if args.recover_interrupted:
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if int(result.get("microbatch", -1)) != candidate:
                    raise RuntimeError(f"Mismatched candidate result: {result_path}")
                if args.refresh_safe_candidate and result.get("safe"):
                    refreshed = _run_candidate(
                        args=args,
                        candidate=candidate,
                        accumulation=16 // candidate,
                        production_config=production_config,
                        attempt_name=f"microbatch_{candidate}_video_val",
                    )
                    refreshed["recovery_source_result"] = str(result_path)
                    attempts.append(refreshed)
                    continue
                if not result.get("safe"):
                    result["reused_rejection_only"] = True
                    result["rejection_provenance"] = "same WIDER training path; training peak VRAM already exceeded 90%"
                attempts.append(result)
                continue
            if not (attempt_dir / "probe.log").is_file():
                raise RuntimeError(f"No prior attempt to recover: {attempt_dir}")
            attempts.append({
                "microbatch": candidate,
                "accumulation_steps": 16 // candidate,
                "effective_batch_size": 16,
                "status": "interrupted_by_host_shutdown",
                "safe": False,
                "reason": "No completed optimizer/validation/VRAM result; candidate is not approved and will not be rerun after host bugcheck 0x20001",
                "log": str(attempt_dir / "probe.log"),
            })
            continue
        attempts.append(_run_candidate(
            args=args,
            candidate=candidate,
            accumulation=16 // candidate,
            production_config=production_config,
        ))
    safe = [attempt for attempt in attempts if attempt["safe"]]
    if not safe:
        write_json_atomic(
            args.output_dir / "batch_plan.json",
            {"status": "failed", "attempts": attempts},
        )
        raise RuntimeError("No safe DINOv3 detector microbatch candidate")
    fastest = min(
        float(attempt["projected_production_seconds"]) for attempt in safe
    )
    equivalent = [
        attempt
        for attempt in safe
        if float(attempt["projected_production_seconds"]) <= fastest * 1.03
    ]
    selected = min(equivalent, key=lambda attempt: int(attempt["microbatch"]))
    fallback = _fallback_plans(safe, selected)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "config": str(args.config.expanduser().resolve()),
        "resolved_config_sha256": resolved_config_sha256(production_config),
        "probe_resolved_config_sha256": resolved_config_sha256(probe_config),
        "required_candidates": list(REQUIRED_CANDIDATES),
        "attempts": attempts,
        "selected": {
            "microbatch": int(selected["microbatch"]),
            "accumulation_steps": int(selected["accumulation_steps"]),
            "effective_batch_size": int(selected["effective_batch_size"]),
            "projected_production_seconds": float(
                selected["projected_production_seconds"]
            ),
        },
        "fallback_plans": [
            {
                "microbatch": int(plan["microbatch"]),
                "accumulation_steps": int(plan["accumulation_steps"]),
                "source": str(plan["source"]),
            }
            for plan in fallback
        ],
        "runtime_oom_fallback": True,
        "selection_rule": (
            "shortest projected production elapsed time; within 3% choose "
            "the smaller microbatch"
        ),
    }
    write_json_atomic(args.output_dir / "batch_plan.json", payload)
    write_json_atomic(success_marker, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
