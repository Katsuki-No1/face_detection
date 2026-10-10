#!/usr/bin/env python3
"""Run a disposable short training and require measurable metric improvement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--direction", choices=("lower", "higher"), required=True)
    parser.add_argument("--minimum-points", type=int, default=3)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.005)
    parser.add_argument("--minimum-improving-transition-fraction", type=float, default=0.50)
    parser.add_argument(
        "--final-guard",
        action="append",
        nargs=3,
        default=[],
        metavar=("METRIC", "OPERATOR", "VALUE"),
        help=(
            "Require the final-window mean of another metric to satisfy "
            "OPERATOR ('ge' or 'le') VALUE; may be repeated."
        ),
    )
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--keep-checkpoint",
        type=Path,
        help=(
            "On success, keep this best checkpoint and remove sibling .pt "
            "gate checkpoints. Production retention is unaffected."
        ),
    )
    parser.add_argument(
        "--reuse-existing-metrics",
        action="store_true",
        help="Reassess an already completed run without launching training again.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _read_values(path: Path, metric: str) -> list[float]:
    if not path.is_file():
        raise RuntimeError(f"Small-scale training did not create metrics: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    try:
        values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Metric {metric!r} is not numeric in {path}") from error
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Metric {metric!r} contains non-finite values")
    return values


def assess_improvement(
    values: list[float],
    *,
    direction: str,
    window: int,
    minimum_points: int,
    minimum_relative_improvement: float,
    minimum_improving_transition_fraction: float,
) -> dict[str, Any]:
    if len(values) < minimum_points:
        raise RuntimeError(
            f"Need at least {minimum_points} metric points, got {len(values)}"
        )
    if window < 1 or window * 2 > len(values):
        raise ValueError("--window must fit in both the initial and final history")
    initial = sum(values[:window]) / window
    final = sum(values[-window:]) / window
    scale = max(abs(initial), 1.0e-12)
    signed_gain = initial - final if direction == "lower" else final - initial
    relative_improvement = signed_gain / scale
    improving = sum(
        (right < left if direction == "lower" else right > left)
        for left, right in zip(values, values[1:])
    )
    transition_fraction = improving / max(1, len(values) - 1)
    passed = (
        relative_improvement >= minimum_relative_improvement
        and transition_fraction >= minimum_improving_transition_fraction
    )
    return {
        "passed": passed,
        "points": len(values),
        "values": values,
        "initial_window_mean": initial,
        "final_window_mean": final,
        "relative_improvement": relative_improvement,
        "improving_transitions": improving,
        "improving_transition_fraction": transition_fraction,
        "minimum_relative_improvement": minimum_relative_improvement,
        "minimum_improving_transition_fraction": minimum_improving_transition_fraction,
    }


def assess_final_guards(
    path: Path,
    guards: list[list[str]],
    *,
    window: int,
) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for metric, operator, raw_threshold in guards:
        if operator not in {"ge", "le"}:
            raise ValueError(
                "--final-guard OPERATOR must be 'ge' or 'le', got "
                f"{operator!r}"
            )
        threshold = float(raw_threshold)
        values = _read_values(path, metric)
        if len(values) < window:
            raise RuntimeError(
                f"Guard metric {metric!r} needs {window} final points, "
                f"got {len(values)}"
            )
        final = sum(values[-window:]) / window
        passed = final >= threshold if operator == "ge" else final <= threshold
        assessments.append(
            {
                "passed": passed,
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "final_window_mean": final,
                "window": window,
            }
        )
    return assessments


def _archive_previous_attempt(work_dir: Path) -> Path | None:
    if not work_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived = work_dir.with_name(f"{work_dir.name}.previous_{stamp}")
    work_dir.rename(archived)
    return archived


def _remove_older_archives(work_dir: Path, *, keep: Path | None) -> None:
    candidates = sorted(
        work_dir.parent.glob(f"{work_dir.name}.previous_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if keep is not None and candidate == keep:
            continue
        shutil.rmtree(candidate)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _retain_only_checkpoint(path: Path) -> list[str]:
    keep = path.resolve()
    if not keep.is_file():
        raise RuntimeError(f"Requested gate checkpoint does not exist: {keep}")
    removed: list[str] = []
    for candidate in sorted(keep.parent.glob("*.pt")):
        resolved = candidate.resolve()
        if resolved == keep:
            continue
        resolved.unlink()
        removed.append(str(resolved))
    return removed


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.reuse_existing_metrics:
        raise ValueError("A training command is required after --")
    marker = args.success_marker.resolve()
    marker.unlink(missing_ok=True)
    archived = None
    started = time.time()
    if not args.reuse_existing_metrics:
        work_dir = args.work_dir.resolve()
        archived = _archive_previous_attempt(work_dir)
        # A failed retry keeps only its immediately preceding attempt. This
        # prevents repeated retries from accumulating unbounded archives.
        _remove_older_archives(work_dir, keep=archived)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    metrics_path = args.metrics.resolve()
    values = _read_values(metrics_path, args.metric)
    assessment = assess_improvement(
        values,
        direction=args.direction,
        window=int(args.window),
        minimum_points=int(args.minimum_points),
        minimum_relative_improvement=float(args.minimum_relative_improvement),
        minimum_improving_transition_fraction=float(
            args.minimum_improving_transition_fraction
        ),
    )
    guard_assessments = assess_final_guards(
        metrics_path,
        args.final_guard,
        window=int(args.window),
    )
    passed = assessment["passed"] and all(
        guard["passed"] for guard in guard_assessments
    )
    removed_checkpoints: list[str] = []
    if passed and args.keep_checkpoint is not None:
        removed_checkpoints = _retain_only_checkpoint(args.keep_checkpoint)
    report = {
        "status": "passed" if passed else "failed",
        "gate_stage": "small_scale",
        "metric": args.metric,
        "direction": args.direction,
        "metrics": str(args.metrics.resolve()),
        "command": command,
        "elapsed_seconds": time.time() - started,
        "archived_previous_attempt": str(archived) if archived else None,
        "assessment": assessment,
        "final_guards": guard_assessments,
        "retained_checkpoint": (
            str(args.keep_checkpoint.resolve())
            if args.keep_checkpoint is not None
            else None
        ),
        "removed_checkpoints": removed_checkpoints,
    }
    _write_json(args.report.resolve(), report)
    if not passed:
        raise RuntimeError(
            "Small-scale learning did not satisfy the required improvement "
            "and final metric guards; "
            f"see {args.report.resolve()}"
        )
    _write_json(marker, report)
    if archived is not None:
        # W&B and the gate report retain history; a successful retry no longer
        # needs a full local copy of its predecessor.
        shutil.rmtree(archived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
