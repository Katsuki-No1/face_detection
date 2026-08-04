from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DINOv3+DEIMv2 freeze/unfreeze training and validation curves."
    )
    parser.add_argument("--freeze-run", type=Path, required=True)
    parser.add_argument("--unfreeze-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows: list[dict[str, float | str]] = []
        for raw in csv.DictReader(file):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value) if value not in (None, "") else ""
                except ValueError:
                    row[key] = value or ""
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def best_row(rows: list[dict[str, float | str]], key: str, *, maximize: bool) -> dict[str, float | str]:
    return (max if maximize else min)(rows, key=lambda row: float(row[key]))


def run_summary(run_dir: Path) -> dict[str, Any]:
    train = read_csv(run_dir / "metrics.csv")
    val = read_csv(run_dir / "val_metrics.csv")
    best_train = best_row(train, "loss_total", maximize=False)
    best_val = best_row(val, "mAP", maximize=True)
    final_train = train[-1]
    final_val = val[-1]
    return {
        "run_dir": str(run_dir),
        "epochs_completed": int(float(final_train["epoch"])),
        "best_train_loss": float(best_train["loss_total"]),
        "best_train_loss_epoch": int(float(best_train["epoch"])),
        "final_train_loss": float(final_train["loss_total"]),
        "best_val_mAP": float(best_val["mAP"]),
        "best_val_epoch": int(float(best_val["epoch"])),
        "best_val_AP50": float(best_val["AP50"]),
        "best_val_AP75": float(best_val["AP75"]),
        "best_val_fixed_f1": float(best_val["f1"]),
        "best_val_threshold_sweep_f1": float(best_val["best_f1"]),
        "best_val_threshold": float(best_val["best_threshold"]),
        "final_val_mAP": float(final_val["mAP"]),
        "mAP_drop_from_peak": float(best_val["mAP"]) - float(final_val["mAP"]),
        "training_seconds": float(final_train["elapsed_seconds"]),
    }


def plot_metric(
    axis: Any,
    runs: dict[str, tuple[list[dict[str, float | str]], str]],
    key: str,
    title: str,
    *,
    marker: str | None = None,
) -> None:
    for label, (rows, color) in runs.items():
        axis.plot(values(rows, "epoch"), values(rows, key), label=label, color=color, marker=marker, markersize=3)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.grid(alpha=0.25)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_train = read_csv(args.freeze_run / "metrics.csv")
    unfreeze_train = read_csv(args.unfreeze_run / "metrics.csv")
    freeze_val = read_csv(args.freeze_run / "val_metrics.csv")
    unfreeze_val = read_csv(args.unfreeze_run / "val_metrics.csv")
    train_runs = {"Freeze": (freeze_train, "#2878B5"), "Unfreeze": (unfreeze_train, "#E07A22")}
    val_runs = {"Freeze": (freeze_val, "#2878B5"), "Unfreeze": (unfreeze_val, "#E07A22")}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    plot_metric(axes[0, 0], train_runs, "loss_total", "Train total loss")
    plot_metric(axes[0, 1], train_runs, "loss_bbox", "Train bbox L1 loss")
    plot_metric(axes[0, 2], train_runs, "loss_giou", "Train GIoU loss")
    plot_metric(axes[1, 0], val_runs, "mAP", "Validation mAP50:95", marker="o")

    for label, (rows, color) in val_runs.items():
        epochs = values(rows, "epoch")
        axes[1, 1].plot(epochs, values(rows, "AP50"), color=color, label=f"{label} AP50")
        axes[1, 1].plot(epochs, values(rows, "AP75"), color=color, linestyle="--", label=f"{label} AP75")
        axes[1, 2].plot(epochs, values(rows, "best_f1"), color=color, label=f"{label} best F1")
        axes[1, 2].plot(epochs, values(rows, "f1"), color=color, linestyle="--", label=f"{label} F1@0.30")
    axes[1, 1].set_title("Validation AP50 / AP75")
    axes[1, 2].set_title("Validation F1")
    for axis in (axes[1, 1], axes[1, 2]):
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    fig.suptitle("DINOv3 + DEIMv2: frozen vs fully unfrozen backbone", fontsize=15)
    fig.savefig(args.output_dir / "freeze_unfreeze_learning_curves.png", dpi=180)
    plt.close(fig)

    summary = {
        "freeze": run_summary(args.freeze_run),
        "unfreeze": run_summary(args.unfreeze_run),
    }
    summary["unfreeze_minus_freeze"] = {
        "best_val_mAP": summary["unfreeze"]["best_val_mAP"] - summary["freeze"]["best_val_mAP"],
        "best_val_AP50": summary["unfreeze"]["best_val_AP50"] - summary["freeze"]["best_val_AP50"],
        "best_val_AP75": summary["unfreeze"]["best_val_AP75"] - summary["freeze"]["best_val_AP75"],
        "best_val_threshold_sweep_f1": (
            summary["unfreeze"]["best_val_threshold_sweep_f1"]
            - summary["freeze"]["best_val_threshold_sweep_f1"]
        ),
    }
    (args.output_dir / "learning_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
