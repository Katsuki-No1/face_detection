import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
DEFAULT_MODIFIED_ZIPS = [
    Path("data/output/2026_06_17_1/task1_mikity.zip"),
    Path("data/output/2026_06_17_2/task1_yk.zip"),
    Path("data/output/cvat_upload/test_kita.zip"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-evaluation YOLO Pose dataset variants, train old/modified ablations, "
            "and summarize metrics."
        )
    )
    parser.add_argument("--old-data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument(
        "--modified-zip",
        type=Path,
        action="append",
        default=[],
        help="Modified/current CVAT zip. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_experiment/yolo_pose_dataset_ablation"),
    )
    parser.add_argument("--project", type=Path, default=Path("runs/face_pose_dataset_ablation"))
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--old-existing-weights", type=Path, default=Path("runs/face_pose/yolo11s_face_pose_baseline_v001_gpu_b32/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fallback-batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.01:0.95:0.01")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-error-images", action="store_true")
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


def image_files(root: Path, split: str) -> list[Path]:
    split_root = root / "images" / split
    if not split_root.exists():
        return []
    return sorted(path for path in split_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_for_image(root: Path, split: str, image_path: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def write_dataset_yaml(root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: face",
            "kpt_shape: [5, 3]",
            "flip_idx: [1, 0, 2, 4, 3]",
            f"keypoints: [{', '.join(KEYPOINT_NAMES)}]",
            "",
        ]
    )
    (root / "face_pose.yaml").write_text(yaml_text, encoding="utf-8")


def ensure_split_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def copy_dataset(source: Path, destination: Path, skip_existing: bool) -> None:
    if destination.exists() and skip_existing:
        write_dataset_yaml(destination)
        return
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Remove it or use --skip-existing.")
    shutil.copytree(source, destination)
    write_dataset_yaml(destination)


def fixed_eval_names(old_root: Path) -> tuple[set[str], set[str]]:
    return (
        {path.name for path in image_files(old_root, "val")},
        {path.name for path in image_files(old_root, "test")},
    )


def convert_modified_dataset(args: argparse.Namespace, temp_root: Path) -> None:
    if temp_root.exists() and args.skip_existing:
        return
    if temp_root.exists():
        raise FileExistsError(f"{temp_root} already exists. Remove it or use --skip-existing.")
    zips = args.modified_zip or DEFAULT_MODIFIED_ZIPS
    missing = [path for path in zips if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing modified zip(s): {', '.join(str(path) for path in missing)}")
    command = [
        sys.executable,
        "scripts_YOLO/convert_annotations_to_yolo_pose.py",
        "--output-dir",
        str(temp_root),
        "--train-ratio",
        "1",
        "--val-ratio",
        "0",
        "--test-ratio",
        "0",
        "--clean",
    ]
    for path in zips:
        command.extend(["--annotation-zip", str(path)])
    result = run_command(command, args.output_dir / "logs" / "convert_modified_all.log")
    if result.returncode != 0:
        raise RuntimeError(f"Modified dataset conversion failed. See {args.output_dir / 'logs' / 'convert_modified_all.log'}")


def copy_sample(source_root: Path, source_split: str, destination_root: Path, destination_split: str, image_path: Path) -> None:
    destination_image = destination_root / "images" / destination_split / image_path.name
    destination_label = destination_root / "labels" / destination_split / f"{image_path.stem}.txt"
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_image)
    label_path = label_for_image(source_root, source_split, image_path)
    if label_path.exists():
        shutil.copy2(label_path, destination_label)
    else:
        destination_label.write_text("", encoding="utf-8")


def write_variant_metadata(root: Path, variant: str, source_root: Path, records: list[dict[str, Any]]) -> None:
    summary = {
        split: {
            "images": len(image_files(root, split)),
            "labels": len(list((root / "labels" / split).glob("*.txt"))),
        }
        for split in ["train", "val", "test"]
    }
    metadata = {
        "schema_version": "1.0",
        "variant": variant,
        "source_root": str(source_root),
        "summary": summary,
        "records": records,
    }
    (root / "metadata" / "variant_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_modified_100(source_root: Path, destination_root: Path, old_root: Path, skip_existing: bool) -> None:
    if destination_root.exists() and skip_existing:
        write_dataset_yaml(destination_root)
        return
    if destination_root.exists():
        raise FileExistsError(f"{destination_root} already exists. Remove it or use --skip-existing.")
    fixed_val, fixed_test = fixed_eval_names(old_root)
    ensure_split_dirs(destination_root)
    records = []
    for image_path in image_files(source_root, "train"):
        if image_path.name in fixed_val:
            split = "val"
        elif image_path.name in fixed_test:
            split = "test"
        else:
            split = "train"
        copy_sample(source_root, "train", destination_root, split, image_path)
        records.append({"image": image_path.name, "split": split})
    write_dataset_yaml(destination_root)
    write_variant_metadata(destination_root, "modified_100", source_root, records)


def build_fraction_dataset(
    source_root: Path,
    destination_root: Path,
    fraction: float,
    seed: int,
    skip_existing: bool,
) -> None:
    if destination_root.exists() and skip_existing:
        write_dataset_yaml(destination_root)
        return
    if destination_root.exists():
        raise FileExistsError(f"{destination_root} already exists. Remove it or use --skip-existing.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")
    ensure_split_dirs(destination_root)
    train_images = image_files(source_root, "train")
    sample_count = max(1, round(len(train_images) * fraction))
    rng = random.Random(seed)
    selected_names = set(rng.sample([path.name for path in train_images], sample_count))
    records = []
    for split in ["val", "test"]:
        for image_path in image_files(source_root, split):
            copy_sample(source_root, split, destination_root, split, image_path)
            records.append({"image": image_path.name, "split": split, "selected": True})
    for image_path in train_images:
        selected = image_path.name in selected_names
        if selected:
            copy_sample(source_root, "train", destination_root, "train", image_path)
        records.append({"image": image_path.name, "split": "train", "selected": selected})
    write_dataset_yaml(destination_root)
    write_variant_metadata(destination_root, f"modified_{int(fraction * 100)}", source_root, records)


def prepare_datasets(args: argparse.Namespace) -> dict[str, Path]:
    datasets_dir = args.output_dir / "datasets"
    old_root = datasets_dir / "old"
    modified_all_temp = datasets_dir / "_modified_all_train"
    modified_100 = datasets_dir / "modified_100"
    modified_50 = datasets_dir / "modified_50"
    copy_dataset(args.old_data_root, old_root, args.skip_existing)
    convert_modified_dataset(args, modified_all_temp)
    build_modified_100(modified_all_temp, modified_100, old_root, args.skip_existing)
    build_fraction_dataset(modified_100, modified_50, args.train_fraction, args.sample_seed, args.skip_existing)
    return {"old_repro": old_root, "modified_50": modified_50, "modified_100": modified_100}


def check_dataset(root: Path, args: argparse.Namespace, name: str) -> None:
    output_path = args.output_dir / "dataset_checks" / f"{name}.json"
    command = [
        sys.executable,
        "scripts_YOLO/check_yolo_pose_dataset.py",
        "--data-root",
        str(root),
        "--yaml",
        str(root / "face_pose.yaml"),
        "--output",
        str(output_path),
    ]
    result = run_command(command, args.output_dir / "logs" / f"check_dataset_{name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"Dataset check failed for {name}. See {output_path}")


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def training_summary(run_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    rows = rows_from_csv(run_dir / "results.csv")
    epochs_completed = len(rows)
    measured_seconds = elapsed_seconds
    if not measured_seconds and rows:
        measured_seconds = numeric(rows[-1].get("time")) or 0.0
    return {
        "epochs_completed": epochs_completed,
        "wall_time_seconds": measured_seconds,
        "average_epoch_seconds": measured_seconds / epochs_completed if epochs_completed else None,
    }


def train_variant(name: str, data_root: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    run_name = f"yolo11s_{name}"
    run_dir = args.project / run_name
    if args.skip_existing and (run_dir / "weights" / "best.pt").exists() and (run_dir / "results.csv").exists():
        return run_dir, training_summary(run_dir, 0.0)
    batch_used = args.batch
    command = [
        sys.executable,
        "scripts_YOLO/train_yolo_pose.py",
        "--data",
        str(data_root / "face_pose.yaml"),
        "--model",
        args.model,
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
    log_path = args.output_dir / "logs" / f"train_{name}.log"
    result = run_command(command, log_path)
    if result.returncode != 0 and "out of memory" in log_path.read_text(encoding="utf-8", errors="ignore").lower():
        batch_used = args.fallback_batch
        command[command.index("--batch") + 1] = str(batch_used)
        log_path = args.output_dir / "logs" / f"train_{name}_batch{batch_used}.log"
        result = run_command(command, log_path)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {name}. See {log_path}")
    if not (run_dir / "weights" / "best.pt").exists():
        raise RuntimeError(f"Missing best.pt for {name}: {run_dir}")
    summary = training_summary(run_dir, float(getattr(result, "elapsed_seconds", 0.0)))
    summary["batch"] = batch_used
    return run_dir, summary


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_variant(name: str, weights: Path, data_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    eval_dir = args.output_dir / "evaluations" / name
    threshold_dir = eval_dir / "threshold"
    threshold_command = [
        sys.executable,
        "scripts_YOLO/evaluate_yolo_pose_f1.py",
        "--weights",
        str(weights),
        "--data-root",
        str(data_root),
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
        str(threshold_dir),
    ]
    threshold_result = run_command(threshold_command, args.output_dir / "logs" / f"eval_threshold_{name}.log")
    if threshold_result.returncode != 0:
        raise RuntimeError(f"Threshold sweep failed for {name}. See {args.output_dir / 'logs' / f'eval_threshold_{name}.log'}")
    best_val = read_json(threshold_dir / "val_best_threshold.json")

    metrics_dir = eval_dir / "test"
    metrics_command = [
        sys.executable,
        "scripts_YOLO/evaluate_yolo_pose_metrics.py",
        "--weights",
        str(weights),
        "--data",
        str(data_root / "face_pose.yaml"),
        "--data-root",
        str(data_root),
        "--split",
        "test",
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--iou-threshold",
        str(args.iou_threshold),
        "--conf-threshold",
        str(best_val["threshold"]),
        "--prediction-conf-min",
        "0.001",
        "--output-dir",
        str(metrics_dir),
    ]
    if args.save_error_images:
        metrics_command.append("--save-error-images")
    metrics_result = run_command(metrics_command, args.output_dir / "logs" / f"eval_test_{name}.log")
    if metrics_result.returncode != 0:
        raise RuntimeError(f"Test metric evaluation failed for {name}. See {args.output_dir / 'logs' / f'eval_test_{name}.log'}")
    metrics = read_json(metrics_dir / "test_metrics_summary.json")
    metrics["best_val_threshold"] = best_val
    metrics["weights"] = str(weights)
    (eval_dir / "combined_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def comparison_row(name: str, data_root: Path, metrics: dict[str, Any], train_summary: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model": name,
        "data_root": str(data_root),
        "weights": metrics.get("weights", ""),
        "epochs_completed": "" if train_summary is None else train_summary.get("epochs_completed", ""),
        "wall_time_seconds": "" if train_summary is None else round(float(train_summary.get("wall_time_seconds") or 0.0), 3),
        "val_best_threshold": nested_get(metrics, ["best_val_threshold", "threshold"]),
        "test_box_map50_95": nested_get(metrics, ["map", "box", "map50_95"]),
        "test_box_map50": nested_get(metrics, ["map", "box", "map50"]),
        "test_box_precision": nested_get(metrics, ["map", "box", "precision"]),
        "test_box_recall": nested_get(metrics, ["map", "box", "recall"]),
        "test_pose_map50_95": nested_get(metrics, ["map", "pose", "map50_95"]),
        "test_pose_map50": nested_get(metrics, ["map", "pose", "map50"]),
        "test_pose_precision": nested_get(metrics, ["map", "pose", "precision"]),
        "test_pose_recall": nested_get(metrics, ["map", "pose", "recall"]),
        "fixed_precision": nested_get(metrics, ["fixed_threshold_detection", "precision"]),
        "fixed_recall": nested_get(metrics, ["fixed_threshold_detection", "recall"]),
        "fixed_f1": nested_get(metrics, ["fixed_threshold_detection", "f1"]),
        "fixed_tp": nested_get(metrics, ["fixed_threshold_detection", "tp"]),
        "fixed_fp": nested_get(metrics, ["fixed_threshold_detection", "fp"]),
        "fixed_fn": nested_get(metrics, ["fixed_threshold_detection", "fn"]),
        "keypoint_nme": nested_get(metrics, ["keypoints", "nme"]),
        "keypoint_pck_0.05": nested_get(metrics, ["keypoints", "pck_0.05"]),
        "keypoint_pck_0.10": nested_get(metrics, ["keypoints", "pck_0.10"]),
        "failures": nested_get(metrics, ["failure_categories", "total_failures"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def failure_comparison_rows(metrics_by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = set()
    for metrics in metrics_by_name.values():
        keys.update(nested_get(metrics, ["failure_categories", "by_failure_type_and_tag"]) or {})
        keys.update(f"failure_type:{key}" for key in (nested_get(metrics, ["failure_categories", "by_failure_type"]) or {}))
        keys.update(f"tag:{key}" for key in (nested_get(metrics, ["failure_categories", "by_tag"]) or {}))
    for key in sorted(keys):
        row = {"category": key}
        for name, metrics in metrics_by_name.items():
            if key.startswith("failure_type:"):
                value = (nested_get(metrics, ["failure_categories", "by_failure_type"]) or {}).get(key.split(":", 1)[1], 0)
            elif key.startswith("tag:"):
                value = (nested_get(metrics, ["failure_categories", "by_tag"]) or {}).get(key.split(":", 1)[1], 0)
            else:
                value = (nested_get(metrics, ["failure_categories", "by_failure_type_and_tag"]) or {}).get(key, 0)
            row[name] = value
        rows.append(row)
    return rows


def write_recommendations(
    path: Path,
    comparison_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    rows = comparison_rows
    old_row = next((row for row in rows if row["model"] == "old_repro"), None)
    full_row = next((row for row in rows if row["model"] == "modified_100"), None)
    lines = [
        "# Next Data Collection Targets",
        "",
        "Fixed test split is held constant across all rows. Lower failure counts are better.",
        "",
    ]
    if old_row and full_row:
        old_failures = float(old_row.get("failures") or 0.0)
        full_failures = float(full_row.get("failures") or 0.0)
        change = old_failures - full_failures
        lines.extend(
            [
                f"- Total failures changed from {old_failures:.0f} to {full_failures:.0f} ({change:+.0f}).",
                "- Prioritize categories that still have many modified_100 failures or did not drop from old_repro.",
                "",
            ]
        )
    old_name = "old_repro"
    full_name = "modified_100"
    scored_targets = []
    for row in failure_rows:
        old_count = float(row.get(old_name) or 0.0)
        full_count = float(row.get(full_name) or 0.0)
        if full_count <= 0:
            continue
        improvement = old_count - full_count
        improvement_rate = improvement / old_count if old_count else 0.0
        sluggish = old_count > 0 and improvement_rate < 0.2
        score = full_count + (5.0 if sluggish else 0.0)
        scored_targets.append((score, full_count, improvement_rate, row))
    scored_targets.sort(reverse=True, key=lambda item: (item[0], item[1], -item[2]))
    if scored_targets:
        lines.extend(["Recommended collection targets:", ""])
        for _, _, improvement_rate, row in scored_targets[:8]:
            old_count = float(row.get(old_name) or 0.0)
            full_count = float(row.get(full_name) or 0.0)
            half_count = row.get("modified_50", "")
            lines.append(
                "- "
                f"{row['category']}: modified_100={full_count:.0f}, "
                f"old_repro={old_count:.0f}, modified_50={half_count}, "
                f"old-to-100 improvement={improvement_rate:.1%}"
            )
        lines.append("")
    lines.append("See `failure_category_comparison.csv` for the exact category counts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)
    data_roots = prepare_datasets(args)
    for name, root in data_roots.items():
        check_dataset(root, args, name)
    if args.prepare_only:
        print(json.dumps({"datasets": {key: str(value) for key, value in data_roots.items()}}, ensure_ascii=False, indent=2))
        return 0

    train_summaries: dict[str, dict[str, Any] | None] = {}
    run_dirs: dict[str, Path] = {}
    if not args.eval_only:
        for name, root in data_roots.items():
            run_dir, train_summary = train_variant(name, root, args)
            run_dirs[name] = run_dir
            train_summaries[name] = train_summary
    else:
        for name in data_roots:
            run_dirs[name] = args.project / f"yolo11s_{name}"
            train_summaries[name] = training_summary(run_dirs[name], 0.0) if (run_dirs[name] / "results.csv").exists() else None

    metrics_by_name: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    if args.old_existing_weights and args.old_existing_weights.exists():
        metrics = evaluate_variant("old_existing", args.old_existing_weights, data_roots["old_repro"], args)
        metrics_by_name["old_existing"] = metrics
        comparison_rows.append(comparison_row("old_existing", data_roots["old_repro"], metrics, None))
    for name, root in data_roots.items():
        weights = run_dirs[name] / "weights" / "best.pt"
        if not weights.exists():
            raise RuntimeError(f"Missing weights for {name}: {weights}")
        metrics = evaluate_variant(name, weights, root, args)
        metrics_by_name[name] = metrics
        comparison_rows.append(comparison_row(name, root, metrics, train_summaries.get(name)))

    write_csv(args.output_dir / "model_comparison.csv", comparison_rows)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failure_rows = failure_comparison_rows(metrics_by_name)
    write_csv(args.output_dir / "failure_category_comparison.csv", failure_rows)
    write_recommendations(args.output_dir / "next_data_collection_targets.md", comparison_rows, failure_rows)
    print(json.dumps({"comparison": comparison_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
