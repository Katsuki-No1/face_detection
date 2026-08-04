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
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare weak/medium/strong DINOv3+DEIMv2 augmentation learning curves."
    )
    parser.add_argument("--family", required=True, help="Augmentation family shown in the plot title.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="LABEL=DIR",
        help="Run directory. Repeat for weak, medium and strong.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8", newline="") as file:
        parsed: list[dict[str, float | str]] = []
        for raw in csv.DictReader(file):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value) if value not in (None, "") else ""
                except ValueError:
                    row[key] = value or ""
            parsed.append(row)
    if not parsed:
        raise ValueError(f"No rows found in {path}")
    return parsed


def values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def summarize(run_dir: Path) -> dict[str, Any]:
    train = read_rows(run_dir / "metrics.csv")
    val = read_rows(run_dir / "val_metrics.csv")
    best = max(val, key=lambda row: float(row["mAP"]))
    return {
        "run_dir": str(run_dir),
        "epochs_completed": int(float(train[-1]["epoch"])),
        "training_seconds": float(train[-1]["elapsed_seconds"]),
        "best_val_epoch": int(float(best["epoch"])),
        "best_val_mAP50_95": float(best["mAP"]),
        "AP50_at_best_mAP": float(best["AP50"]),
        "AP75_at_best_mAP": float(best["AP75"]),
        "fixed_f1_at_best_mAP": float(best["f1"]),
        "swept_f1_at_best_mAP": float(best["best_f1"]),
        "best_threshold_at_best_mAP": float(best["best_threshold"]),
        "final_val_mAP50_95": float(val[-1]["mAP"]),
        "peak_to_final_mAP_drop": float(best["mAP"]) - float(val[-1]["mAP"]),
    }


def main() -> int:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("At least two --run values are required")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique")

    train_runs = {label: read_rows(run_dir / "metrics.csv") for label, run_dir in args.run}
    val_runs = {label: read_rows(run_dir / "val_metrics.csv") for label, run_dir in args.run}
    colors = {label: color for label, color in zip(labels, ("#2878B5", "#E07A22", "#3A923A", "#8A55A2"))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = (
        (axes[0, 0], train_runs, "loss_total", "Train total loss"),
        (axes[0, 1], train_runs, "loss_cls", "Train classification loss"),
        (axes[0, 2], train_runs, "loss_giou", "Train GIoU loss"),
        (axes[1, 0], val_runs, "mAP", "Validation mAP50:95"),
        (axes[1, 1], val_runs, "AP50", "Validation AP50"),
        (axes[1, 2], val_runs, "best_f1", "Validation best F1"),
    )
    for axis, runs, key, title in panels:
        for label, rows in runs.items():
            axis.plot(
                values(rows, "epoch"),
                values(rows, key),
                color=colors[label],
                label=label,
                marker="o" if runs is val_runs else None,
                markersize=4,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    fig.suptitle(f"DINOv3 + DEIMv2: {args.family} augmentation strength", fontsize=16)
    plot_path = args.output_dir / f"{args.family}_augmentation_learning_curves.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    runs_summary = {label: summarize(run_dir) for label, run_dir in args.run}
    winner = max(runs_summary, key=lambda label: runs_summary[label]["best_val_mAP50_95"])
    summary = {
        "augmentation_family": args.family,
        "selection_metric": "best validation bbox mAP50:95",
        "selected_strength": winner,
        "runs": runs_summary,
    }
    summary_path = args.output_dir / f"{args.family}_augmentation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
