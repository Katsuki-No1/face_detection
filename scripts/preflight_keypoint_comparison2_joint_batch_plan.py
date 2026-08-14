#!/usr/bin/env python3
"""Select a timed safe 4/8/16 plan for ellipse-local DEIMv2 Joint."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keypoint_comparison2.config import (  # noqa: E402
    load_config,
    resolved_config_sha256,
)
from src.wandb_utils import write_json_atomic  # noqa: E402


PYTHON = PROJECT_ROOT / ".venv/bin/python"
RUNNER = PROJECT_ROOT / "keypoint_comparison2/run.py"
REQUIRED_CANDIDATES = (4, 8, 16)
SELECTION_METRIC = "conditional_semantic_macro_f1_nme_0.10"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
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


def _annotation_images(path: str | Path) -> int:
    return len(
        json.loads(Path(path).read_text(encoding="utf-8"))["images"]
    )


def _remove_probe_checkpoints(path: Path) -> list[str]:
    removed: list[str] = []
    checkpoint_dir = path / "deimv2_joint/checkpoints"
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


def _run_candidate(
    *,
    args: argparse.Namespace,
    candidate: int,
    accumulation: int,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    attempt = args.output_dir / f"microbatch_{candidate}"
    if attempt.exists():
        shutil.rmtree(attempt)
    attempt.mkdir(parents=True)
    log_path = attempt / "probe.log"
    required_updates = 3
    train_samples = candidate * accumulation * required_updates
    validation_samples = 64
    command = [
        str(PYTHON),
        str(RUNNER),
        "--config",
        str(args.config.expanduser().resolve()),
        "--stage",
        "train",
        "--models",
        "deimv2_joint",
        "--device",
        f"cuda:{args.device_index}",
        "--output-root",
        str(attempt),
        "--epochs",
        "1",
        "--batch-size",
        str(candidate),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--num-workers",
        "0",
        "--max-train-samples",
        str(train_samples),
        "--max-val-samples",
        str(validation_samples),
        "--no-resume",
        "--no-wandb",
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
    metric = _last_csv_row(attempt / "deimv2_joint/metrics.csv")
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
        for value in metric.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    finite = (
        bool(metric)
        and bool(numeric_metrics)
        and all(math.isfinite(value) for value in numeric_metrics)
    )
    train_seconds = float(metric.get("train_seconds", float("inf")))
    validation_seconds = float(
        metric.get("validation_seconds", float("inf"))
    )
    joint_data = resolved["data"]["joint"]
    projected = int(resolved["train"]["epochs"]) * (
        train_seconds
        * _annotation_images(joint_data["annotations"]["train"])
        / max(1, train_samples)
        + validation_seconds
        * _annotation_images(joint_data["annotations"]["val"])
        / validation_samples
    )
    lower_log = log_text.lower()
    assertions = {
        "exit_zero": exit_code == 0,
        "repeated_optimizer_steps": int(metric.get("updates", 0.0))
        >= required_updates,
        "finite_train_and_validation_metrics": finite,
        "production_validation_path_exercised": float(
            metric.get("validation_seconds", 0.0)
        )
        > 0.0,
        "conditional_selection_population_nonempty": float(
            metric.get("conditional_eligible_faces", 0.0)
        )
        > 0.0,
        "selection_metric_present": SELECTION_METRIC in metric,
        "no_cuda_oom": "out of memory" not in lower_log,
        "no_non_finite_values": "non-finite" not in lower_log,
        "vram_within_90_percent": vram_fraction <= 0.90,
        "temperature_below_83c": temperature < 83.0,
        "effective_batch_preserved": candidate * accumulation == 16,
    }
    result = {
        "microbatch": candidate,
        "accumulation_steps": accumulation,
        "effective_batch_size": candidate * accumulation,
        "exit_code": exit_code,
        "optimizer_updates": int(metric.get("updates", 0.0)),
        "train_images": train_samples,
        "validation_images": validation_samples,
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "wall_clock_seconds": elapsed,
        "projected_production_seconds": projected,
        "selection_metric": metric.get(SELECTION_METRIC),
        "conditional_eligible_faces": metric.get(
            "conditional_eligible_faces"
        ),
        "peak_memory_used_mib": peak_used,
        "physical_vram_mib": physical,
        "peak_vram_fraction": vram_fraction,
        "maximum_gpu_temperature_c": temperature,
        "assertions": assertions,
        "safe": all(assertions.values()),
        "log": str(log_path),
    }
    result["removed_probe_checkpoints"] = _remove_probe_checkpoints(attempt)
    write_json_atomic(attempt / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = args.success_marker.expanduser().resolve()
    success_marker.unlink(missing_ok=True)
    config = load_config(args.config, project_root=PROJECT_ROOT)
    attempts = [
        _run_candidate(
            args=args,
            candidate=candidate,
            accumulation=16 // candidate,
            resolved=config.raw,
        )
        for candidate in REQUIRED_CANDIDATES
    ]
    safe = [attempt for attempt in attempts if attempt["safe"]]
    if not safe:
        write_json_atomic(
            args.output_dir / "batch_plan.json",
            {"status": "failed", "attempts": attempts},
        )
        raise RuntimeError("No safe ellipse-local Joint microbatch candidate")
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
        "resolved_config_sha256": resolved_config_sha256(config.raw),
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
