import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODELS = [
    ("yolo11n", "yolo11n-pose.pt", "yolo11n_face_pose_baseline_v001_gpu_b32", None),
    ("yolo11s", "yolo11s-pose.pt", "yolo11s_face_pose_baseline_v001_gpu_b32", None),
    ("yolo11m", "yolo11m-pose.pt", "yolo11m_face_pose_baseline_v001_gpu_b16", 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare YOLO Pose GPU baselines.")
    parser.add_argument("--data", type=Path, default=Path("datasets/face_pose/face_pose.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--project", type=Path, default=Path("runs/face_pose"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_exoeriment/yolo_pose_baseline_v001_gpu_compare"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fallback-batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.01:0.95:0.01")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    process.elapsed_seconds = time.monotonic() - started  # type: ignore[attr-defined]
    return process


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def training_summary(run_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    rows = rows_from_csv(run_dir / "results.csv")
    epochs_completed = len(rows)
    result_time = numeric(rows[-1], "time") if rows else None
    measured_seconds = elapsed_seconds or result_time or 0.0
    epoch_time = measured_seconds / epochs_completed if epochs_completed else None
    warmup_excluded = None
    if len(rows) > 4:
        end_time = numeric(rows[-1], "time")
        start_time = numeric(rows[2], "time")
        if end_time is not None and start_time is not None:
            warmup_excluded = (end_time - start_time) / (len(rows) - 3)
    return {
        "epochs_completed": epochs_completed,
        "wall_time_seconds": measured_seconds,
        "average_epoch_seconds": epoch_time,
        "warmup_excluded_epoch_seconds": warmup_excluded,
        "last_box_map50": numeric(rows[-1], "metrics/mAP50(B)") if rows else None,
        "last_pose_map50": numeric(rows[-1], "metrics/mAP50(P)") if rows else None,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_curve_plot(run_dirs: list[tuple[str, Path]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric_candidates = [
        ("metrics/mAP50(B)", "bbox mAP50"),
        ("metrics/mAP50(P)", "pose mAP50"),
        ("val/box_loss", "val box loss"),
        ("val/pose_loss", "val pose loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()
    for axis, (column, title) in zip(axes_flat, metric_candidates):
        for model_name, run_dir in run_dirs:
            rows = rows_from_csv(run_dir / "results.csv")
            values = [numeric(row, column) for row in rows]
            series = [value for value in values if value is not None]
            if series:
                axis.plot(range(1, len(series) + 1), series, label=model_name)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    speed_summary: dict[str, Any] = {}
    run_dirs: list[tuple[str, Path]] = []

    for model_key, model_file, run_name, batch_override in MODELS:
        batch_used = batch_override or args.batch
        run_dir = args.project / run_name
        train_log = args.output_dir / f"train_{model_key}.log"
        train_command = [
            sys.executable,
            "scripts/train_yolo_pose.py",
            "--data",
            str(args.data),
            "--model",
            model_file,
            "--imgsz",
            str(args.imgsz),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(batch_used),
            "--patience",
            str(args.patience),
            "--project",
            str(args.project.resolve()),
            "--name",
            run_name,
            "--device",
            args.device,
        ]
        if (run_dir / "weights" / "best.pt").exists() and (run_dir / "results.csv").exists():
            train_result = subprocess.CompletedProcess(train_command, 0)
            train_result.elapsed_seconds = 0.0  # type: ignore[attr-defined]
        else:
            train_result = run_command(train_command, train_log)
            if train_result.returncode != 0 and "out of memory" in train_log.read_text(encoding="utf-8", errors="ignore").lower():
                batch_used = args.fallback_batch
                train_log = args.output_dir / f"train_{model_key}_batch{batch_used}.log"
                train_command[train_command.index("--batch") + 1] = str(batch_used)
                train_result = run_command(train_command, train_log)
            if train_result.returncode != 0:
                raise RuntimeError(f"Training failed for {model_key}. See {train_log}")

        if not (run_dir / "weights" / "best.pt").exists():
            raise RuntimeError(f"Missing best.pt for {model_key}: {run_dir}")
        if not (run_dir / "weights" / "last.pt").exists():
            raise RuntimeError(f"Missing last.pt for {model_key}: {run_dir}")
        if not (run_dir / "results.csv").exists():
            raise RuntimeError(f"Missing results.csv for {model_key}: {run_dir}")
        copy_if_exists(run_dir / "results.png", args.output_dir / f"training_curve_{model_key}.png")

        eval_dir = args.output_dir / f"_eval_{model_key}"
        val_log = args.output_dir / f"val_threshold_{model_key}.log"
        val_command = [
            sys.executable,
            "scripts/evaluate_yolo_pose_f1.py",
            "--weights",
            str(run_dir / "weights" / "best.pt"),
            "--data-root",
            str(args.data_root),
            "--split",
            "val",
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--iou-threshold",
            str(args.iou_threshold),
            "--prediction-conf-min",
            "0.001",
            "--thresholds",
            args.thresholds,
            "--output-dir",
            str(eval_dir),
            "--write-predictions",
        ]
        val_result = run_command(val_command, val_log)
        if val_result.returncode != 0:
            raise RuntimeError(f"Val threshold sweep failed for {model_key}. See {val_log}")
        best_val = read_json(eval_dir / "val_best_threshold.json")
        copy_if_exists(eval_dir / "val_threshold_sweep.csv", args.output_dir / f"val_threshold_sweep_{model_key}.csv")
        copy_if_exists(eval_dir / "val_best_threshold.json", args.output_dir / f"best_val_threshold_{model_key}.json")

        test_log = args.output_dir / f"test_metrics_{model_key}.log"
        test_command = [
            sys.executable,
            "scripts/evaluate_yolo_pose_f1.py",
            "--weights",
            str(run_dir / "weights" / "best.pt"),
            "--data-root",
            str(args.data_root),
            "--split",
            "test",
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--iou-threshold",
            str(args.iou_threshold),
            "--prediction-conf-min",
            "0.001",
            "--fixed-threshold",
            str(best_val["threshold"]),
            "--output-dir",
            str(eval_dir),
            "--write-predictions",
        ]
        test_result = run_command(test_command, test_log)
        if test_result.returncode != 0:
            raise RuntimeError(f"Test metrics failed for {model_key}. See {test_log}")
        test_metrics = read_json(eval_dir / "test_best_threshold.json")
        test_metrics.update({"model": model_key, "weights": str(run_dir / "weights" / "best.pt"), "best_val_threshold": best_val})
        (args.output_dir / f"test_metrics_{model_key}.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        jsonl_log = args.output_dir / f"test_predictions_{model_key}.log"
        jsonl_command = [
            sys.executable,
            "scripts/export_yolo_pose_jsonl.py",
            "--weights",
            str(run_dir / "weights" / "best.pt"),
            "--source",
            str(args.data_root / "images" / "test"),
            "--output",
            str(args.output_dir / f"test_predictions_{model_key}.jsonl"),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(best_val["threshold"]),
            "--device",
            args.device,
        ]
        jsonl_result = run_command(jsonl_command, jsonl_log)
        if jsonl_result.returncode != 0:
            raise RuntimeError(f"JSONL export failed for {model_key}. See {jsonl_log}")

        vis_log = args.output_dir / f"test_visualizations_{model_key}.log"
        vis_command = [
            sys.executable,
            "scripts/predict_yolo_pose.py",
            "--weights",
            str(run_dir / "weights" / "best.pt"),
            "--source",
            str(args.data_root / "images" / "test"),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(best_val["threshold"]),
            "--device",
            args.device,
            "--out-dir",
            str((args.output_dir / f"test_visualizations_{model_key}").resolve()),
        ]
        vis_result = run_command(vis_command, vis_log)
        if vis_result.returncode != 0:
            raise RuntimeError(f"Visualization failed for {model_key}. See {vis_log}")

        summary = training_summary(run_dir, float(getattr(train_result, "elapsed_seconds")))
        speed_summary[model_key] = summary
        run_dirs.append((model_key, run_dir))
        comparison_rows.append(
            {
                "model": model_key,
                "weights": str(run_dir / "weights" / "best.pt"),
                "batch": batch_used,
                "epochs_completed": summary["epochs_completed"],
                "wall_time_seconds": round(summary["wall_time_seconds"], 3),
                "average_epoch_seconds": round(summary["average_epoch_seconds"], 3) if summary["average_epoch_seconds"] else "",
                "val_best_threshold": best_val["threshold"],
                "val_f1": best_val["f1"],
                "val_precision": best_val["precision"],
                "val_recall": best_val["recall"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
            }
        )

    write_table(args.output_dir / "model_comparison.csv", comparison_rows)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "speed_summary.json").write_text(
        json.dumps(speed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_curve_plot(run_dirs, args.output_dir / "training_curves_comparison.png")
    print(json.dumps({"comparison": comparison_rows, "speed": speed_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
