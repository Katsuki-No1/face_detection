#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("normal400", "active357")
COLORS = {"normal400": "#4C78A8", "active357": "#E45756"}
METRICS = {
    "face_macro_f1": ("gt_head", "face", "macro_f1"),
    "ellipse_mean_iou": ("gt_head", "ellipse", "mean_iou"),
    "keypoint_macro_f1_0.05": ("gt_head", "keypoint", "macro_f1_0.05"),
    "keypoint_macro_f1_0.10": ("gt_head", "keypoint", "macro_f1_0.10"),
    "matched_point_nme_0.10": (
        "gt_head",
        "keypoint",
        "matched_point_nme_0.10",
    ),
    "state_macro_f1": ("gt_head", "state", "state_macro_f1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize normal400 versus active357 learning and evaluation."
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/active_learning_effect/experiment.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/active_learning_effect/comparison",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def numeric_series(
    rows: Iterable[dict[str, str]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        if row.get(x_key) in {None, ""} or row.get(y_key) in {None, ""}:
            continue
        x.append(float(row[x_key]))
        y.append(float(row[y_key]))
    return x, y


def nested_metric(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, (int, float)) else None


def evaluation_summary(path: Path) -> dict[str, float | None]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {name: nested_metric(value, metric_path) for name, metric_path in METRICS.items()}


def plot_learning_curves(
    histories: dict[str, dict[str, list[dict[str, str]]]], output: Path
) -> None:
    panels = (
        ("detector_train", "epoch", "loss_total", "DEIMv2-L train loss", "loss"),
        (
            "detector_val",
            "epoch",
            "ellipse_mAP",
            "DEIMv2-L validation ellipse mAP",
            "ellipse mAP",
        ),
        (
            "query",
            "epoch",
            "train/loss_total",
            "Query Refinement train loss",
            "loss",
        ),
        (
            "query",
            "epoch",
            "val/nme",
            "Query Refinement validation NME",
            "NME (lower is better)",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (history_key, x_key, y_key, title, ylabel) in zip(axes.flat, panels):
        for arm in ARMS:
            x, y = numeric_series(histories[arm][history_key], x_key, y_key)
            axis.plot(x, y, label=arm, color=COLORS[arm], linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Training-data effect: datasetv2 + normal400 vs datasetv2 + active357",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def comparison_rows(
    evaluations: dict[str, dict[str, dict[str, float | None]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for metric in METRICS:
            normal = evaluations["normal400"][split][metric]
            active = evaluations["active357"][split][metric]
            absolute = (
                active - normal
                if active is not None and normal is not None
                else None
            )
            relative = (
                absolute / abs(normal)
                if absolute is not None and normal not in {None, 0.0}
                else None
            )
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "normal400": normal,
                    "active357": active,
                    "active_minus_normal": absolute,
                    "relative_delta": relative,
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Active-learning data-effect comparison",
        "",
        "Both arms use the same corrected ellipse DEIMv2-L + factorized-state Query Refinement recipe. Validation is reported before the locked test; test results are not used for selection.",
        "",
        "| split | metric | normal400 | active357 | active - normal | relative |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def show(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.6f}"

    for row in rows:
        lines.append(
            f"| {row['split']} | {row['metric']} | {show(row['normal400'])} | "
            f"{show(row['active357'])} | {show(row['active_minus_normal'])} | "
            f"{show(row['relative_delta'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    experiment_path = resolve_path(args.experiment_config)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[str, dict[str, list[dict[str, str]]]] = {}
    evaluations: dict[str, dict[str, dict[str, float | None]]] = {}
    inputs: list[Path] = [experiment_path]
    for arm in ARMS:
        arm_root = resolve_path(experiment["arms"][arm]["output_root"])
        detector_root = arm_root / "ellipse_deimv2_l"
        query_root = arm_root / "query_refinement"
        history_paths = {
            "detector_train": detector_root / "metrics.csv",
            "detector_val": detector_root / "val_metrics.csv",
            "query": query_root / "metrics.csv",
        }
        histories[arm] = {
            name: read_csv(path) for name, path in history_paths.items()
        }
        inputs.extend(history_paths.values())
        evaluations[arm] = {}
        for split in ("val", "test"):
            metrics_path = arm_root / "evaluation" / split / "baseline_aligned/metrics.json"
            evaluations[arm][split] = evaluation_summary(metrics_path)
            inputs.append(metrics_path)

    plot_path = output_dir / "learning_curves.png"
    plot_learning_curves(histories, plot_path)
    rows = comparison_rows(evaluations)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment["experiment_id"],
        "selection_rule": "validation first; test is locked and never selects settings",
        "comparison": rows,
        "input_sha256": {path.as_posix(): sha256_file(path) for path in inputs},
        "learning_curves": plot_path.as_posix(),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "comparison.md", rows)
    success = {
        "status": "passed",
        "metrics": metrics_path.as_posix(),
        "metrics_sha256": sha256_file(metrics_path),
        "learning_curves": plot_path.as_posix(),
        "learning_curves_sha256": sha256_file(plot_path),
    }
    (output_dir / "success.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
