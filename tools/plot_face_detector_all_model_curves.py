from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay learning curves for DINO and YOLO face detectors.")
    parser.add_argument("--dino-run", action="append", default=[], type=parse_named_path, metavar="LABEL=DIR")
    parser.add_argument("--yolo-run", action="append", default=[], type=parse_named_path, metavar="LABEL=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        result = list(csv.DictReader(file))
    if not result:
        raise ValueError(f"No rows in {path}")
    return result


def f(row: dict[str, str], key: str) -> float:
    return float(row[key].strip())


def dino_series(run_dir: Path) -> dict[str, list[float]]:
    train = rows(run_dir / "metrics.csv")
    val = rows(run_dir / "val_metrics.csv")
    train_loss = [f(row, "loss_total") for row in train]
    return {
        "train_epoch": [f(row, "epoch") for row in train],
        "normalized_train_loss": [value / train_loss[0] for value in train_loss],
        "val_epoch": [f(row, "epoch") for row in val],
        "mAP": [f(row, "mAP") for row in val],
        "AP50": [f(row, "AP50") for row in val],
        "precision": [f(row, "precision") for row in val],
        "recall": [f(row, "recall") for row in val],
    }


def yolo_series(run_dir: Path) -> dict[str, list[float]]:
    result = rows(run_dir / "results.csv")
    loss_keys = [key for key in result[0] if key.startswith("train/") and key.endswith("_loss")]
    train_loss = [sum(f(row, key) for key in loss_keys) for row in result]
    return {
        "train_epoch": [f(row, "epoch") for row in result],
        "normalized_train_loss": [value / train_loss[0] for value in train_loss],
        "val_epoch": [f(row, "epoch") for row in result],
        "mAP": [f(row, "metrics/mAP50-95(B)") for row in result],
        "AP50": [f(row, "metrics/mAP50(B)") for row in result],
        "precision": [f(row, "metrics/precision(B)") for row in result],
        "recall": [f(row, "metrics/recall(B)") for row in result],
    }


def summarize(series: dict[str, list[float]]) -> dict[str, Any]:
    best_index = max(range(len(series["mAP"])), key=series["mAP"].__getitem__)
    return {
        "epochs_completed": int(series["train_epoch"][-1]),
        "best_val_epoch": int(series["val_epoch"][best_index]),
        "best_val_mAP50_95": series["mAP"][best_index],
        "AP50_at_best_mAP": series["AP50"][best_index],
        "precision_at_best_mAP": series["precision"][best_index],
        "recall_at_best_mAP": series["recall"][best_index],
        "final_normalized_train_loss": series["normalized_train_loss"][-1],
    }


def main() -> int:
    args = parse_args()
    if not args.dino_run and not args.yolo_run:
        raise ValueError("At least one --dino-run or --yolo-run is required")
    all_series: dict[str, dict[str, list[float]]] = {}
    for label, run_dir in args.dino_run:
        all_series[label] = dino_series(run_dir)
    for label, run_dir in args.yolo_run:
        all_series[label] = yolo_series(run_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = [
        ("normalized_train_loss", "Normalized train loss", "train_epoch"),
        ("mAP", "Validation bbox mAP50:95", "val_epoch"),
        ("AP50", "Validation bbox AP50", "val_epoch"),
        ("precision", "Validation bbox precision (native threshold)", "val_epoch"),
        ("recall", "Validation bbox recall (native threshold)", "val_epoch"),
    ]
    for axis, (key, title, epoch_key) in zip(axes.flat, panels):
        for label, series in all_series.items():
            marker = "o" if len(series[epoch_key]) <= 30 else None
            axis.plot(series[epoch_key], series[key], label=label, marker=marker, markersize=3)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "Loss is normalized by each model's epoch-1 value.\n"
        "Raw losses are not comparable across detector families.\n"
        "All validation panels use bbox metrics on the same val split.\n"
        "P/R use each trainer's validation threshold convention\n"
        "(DINO fixed at 0.30; YOLO model-native) and are contextual.",
        va="top",
        fontsize=11,
    )
    fig.suptitle("face_pose_drive_v1: all-model learning curves", fontsize=16)
    fig.savefig(args.output_dir / "all_models_learning_curves.png", dpi=180)
    plt.close(fig)

    summary = {label: summarize(series) for label, series in all_series.items()}
    (args.output_dir / "all_models_learning_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
