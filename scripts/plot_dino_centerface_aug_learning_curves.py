from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DINO detector augmentation ablation learning curves.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/dino_centerface_ellipse_aug_ablation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_experiment/dino_centerface_ellipse_aug_ablation_learning_curves"))
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output_dir: Path, rows_by_experiment: dict[str, list[dict[str, Any]]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"matplotlib unavailable, skipping PNG plots: {error}")
        return []

    output_paths: list[str] = []
    metric_groups = [
        ("loss_total", "Total Loss"),
        ("loss_cls", "Classification Loss"),
        ("loss_center", "Center Loss"),
        ("loss_center_offset", "Center Offset Loss"),
        ("loss_ellipse", "Ellipse Loss"),
        ("loss_bbox", "BBox Loss"),
        ("loss_kpt", "Keypoint Loss"),
    ]
    for key, title in metric_groups:
        fig, ax = plt.subplots(figsize=(9, 5))
        for experiment, rows in rows_by_experiment.items():
            xs = []
            ys = []
            for row in rows:
                value = to_float(row, key)
                epoch = to_float(row, "epoch")
                if value is None or epoch is None:
                    continue
                xs.append(epoch)
                ys.append(value)
            if xs:
                ax.plot(xs, ys, label=experiment, linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = output_dir / f"{key}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        output_paths.append(str(output_path))
    return output_paths


def main() -> int:
    args = parse_args()
    rows_by_experiment: dict[str, list[dict[str, Any]]] = {}
    combined_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()) if args.run_root.exists() else []:
        metrics = read_metrics(run_dir / "metrics.csv")
        if not metrics:
            continue
        rows_by_experiment[run_dir.name] = metrics
        for row in metrics:
            combined_rows.append({"experiment": run_dir.name, **row})
        final = metrics[-1]
        best = min(metrics, key=lambda row: to_float(row, "loss_total") if to_float(row, "loss_total") is not None else float("inf"))
        summary_rows.append(
            {
                "experiment": run_dir.name,
                "epochs": len(metrics),
                "final_loss_total": final.get("loss_total", ""),
                "best_epoch": best.get("epoch", ""),
                "best_loss_total": best.get("loss_total", ""),
                "metrics_path": str(run_dir / "metrics.csv"),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "learning_curves_combined.csv", combined_rows)
    write_csv(args.output_dir / "learning_curve_summary.csv", summary_rows)
    plots = plot_curves(args.output_dir, rows_by_experiment)
    manifest = {
        "run_root": str(args.run_root),
        "output_dir": str(args.output_dir),
        "experiments": sorted(rows_by_experiment),
        "plots": plots,
        "combined_csv": str(args.output_dir / "learning_curves_combined.csv"),
        "summary_csv": str(args.output_dir / "learning_curve_summary.csv"),
    }
    (args.output_dir / "learning_curve_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
