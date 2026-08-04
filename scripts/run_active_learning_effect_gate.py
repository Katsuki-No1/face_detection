#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both component gates for one data-effect experiment arm."
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/active_learning_effect/experiment.yaml",
    )
    parser.add_argument("--arm", choices=("normal400", "active357"), required=True)
    parser.add_argument(
        "--stage", choices=("gpu_step", "overfit", "small_scale"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--success-marker", type=Path, required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run(command: list[str]) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    result = {
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": time.time() - started,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Gate component failed with exit code {completed.returncode}: {command}"
        )
    return result


def query_command(
    *,
    query_config: Path,
    output_root: Path,
    baseline_detector: Path,
    device: str,
    epochs: int,
    train_samples: int,
    val_samples: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(PROJECT_ROOT / "keypoint_comparison1/scripts/run_landmark_experiments.py"),
        "--config",
        str(query_config),
        "--stage",
        "train",
        "--variants",
        "query_refinement",
        "--device",
        device,
        "--output-root",
        str(output_root),
        "--initialize-from-deim-checkpoint",
        str(baseline_detector),
        "--detector-checkpoint",
        str(baseline_detector),
        "--epochs",
        str(epochs),
        "--max-train-samples",
        str(train_samples),
        "--max-val-samples",
        str(val_samples),
        "--no-resume",
        "--no-wandb",
    ]


def improvement_gate_command(
    *,
    work_dir: Path,
    metrics: Path,
    metric: str,
    direction: str,
    minimum_points: int,
    minimum_relative_improvement: float,
    minimum_transition_fraction: float,
    command: list[str],
) -> list[str]:
    return [
        str(PYTHON),
        str(PROJECT_ROOT / "tools/run_small_scale_training_gate.py"),
        "--work-dir",
        str(work_dir),
        "--metrics",
        str(metrics),
        "--metric",
        metric,
        "--direction",
        direction,
        "--minimum-points",
        str(minimum_points),
        "--minimum-relative-improvement",
        str(minimum_relative_improvement),
        "--minimum-improving-transition-fraction",
        str(minimum_transition_fraction),
        "--success-marker",
        str(work_dir / "success.json"),
        "--report",
        str(work_dir / "report.json"),
        "--",
        *command,
    ]


def stage_commands(
    *,
    stage: str,
    arm_name: str,
    arm: dict[str, Any],
    latest: dict[str, Any],
    device: str,
) -> list[list[str]]:
    detector_config = resolve_path(arm["detector_config"])
    detector_gate_config = resolve_path(arm["detector_gate_config"])
    query_config = resolve_path(arm["query_config"])
    dataset_root = resolve_path(arm["dataset_root"])
    baseline_detector = resolve_path(latest["detector_reference_checkpoint"])
    gate_root = (
        PROJECT_ROOT
        / "outputs/training_gates"
        / f"active_learning_effect_{arm_name}"
        / stage
    )
    if stage == "gpu_step":
        query_output = gate_root / "query_preflight"
        return [
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/train_dino.py"),
                "--config",
                str(detector_config),
                "--dry-run",
                "--device",
                device,
                "--batch-size",
                "1",
            ],
            [
                str(PYTHON),
                str(
                    PROJECT_ROOT
                    / "keypoint_comparison1/scripts/run_landmark_experiments.py"
                ),
                "--config",
                str(query_config),
                "--stage",
                "preflight",
                "--device",
                device,
                "--skip-detector-preflight",
                "--output-root",
                str(query_output),
                "--initialize-from-deim-checkpoint",
                str(baseline_detector),
                "--detector-checkpoint",
                str(baseline_detector),
            ],
        ]
    if stage == "overfit":
        query_output = gate_root / "query_refinement"
        query_train = query_command(
            query_config=query_config,
            output_root=query_output,
            baseline_detector=baseline_detector,
            device=device,
            epochs=12,
            train_samples=4,
            val_samples=4,
        )
        return [
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/train_dino.py"),
                "--config",
                str(detector_config),
                "--overfit-smoke",
                "--device",
                device,
                "--max-samples",
                "4",
                "--steps",
                "64",
                "--batch-size",
                "4",
            ],
            improvement_gate_command(
                work_dir=query_output,
                metrics=query_output / "query_refinement/metrics.csv",
                metric="train/loss_total",
                direction="lower",
                minimum_points=8,
                minimum_relative_improvement=0.05,
                minimum_transition_fraction=0.40,
                command=query_train,
            ),
        ]
    detector_output = gate_root / "detector"
    detector_train = [
        str(PYTHON),
        str(PROJECT_ROOT / "tools/train_dinov3_deimv2.py"),
        "--config",
        str(detector_gate_config),
        "--data-root",
        str(dataset_root),
        "--output-dir",
        str(detector_output),
        "--device",
        device,
        "--max-train-batches",
        "32",
        "--val-every",
        "1",
        "--val-max-images",
        "128",
    ]
    query_output = gate_root / "query_refinement"
    query_train = query_command(
        query_config=query_config,
        output_root=query_output,
        baseline_detector=baseline_detector,
        device=device,
        epochs=12,
        train_samples=128,
        val_samples=128,
    )
    return [
        improvement_gate_command(
            work_dir=detector_output,
            metrics=detector_output / "val_metrics.csv",
            metric="ellipse_mAP",
            direction="higher",
            minimum_points=5,
            minimum_relative_improvement=0.001,
            minimum_transition_fraction=0.25,
            command=detector_train,
        ),
        improvement_gate_command(
            work_dir=query_output,
            metrics=query_output / "query_refinement/metrics.csv",
            metric="val/nme",
            direction="lower",
            minimum_points=5,
            minimum_relative_improvement=0.001,
            minimum_transition_fraction=0.25,
            command=query_train,
        ),
    ]


def main() -> int:
    args = parse_args()
    experiment_path = resolve_path(args.experiment_config)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    marker = resolve_path(args.success_marker)
    marker.unlink(missing_ok=True)
    commands = stage_commands(
        stage=args.stage,
        arm_name=args.arm,
        arm=experiment["arms"][args.arm],
        latest=experiment["latest_model_contract"],
        device=args.device,
    )
    started = time.time()
    results = [run(command) for command in commands]
    payload = {
        "status": "passed",
        "gate_stage": args.stage,
        "architecture": f"active_learning_effect_{args.arm}",
        "arm": args.arm,
        "experiment_config": experiment_path.as_posix(),
        "component_results": results,
        "elapsed_seconds": time.time() - started,
    }
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
