import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_yolo11s_aug_inference_compare import (  # noqa: E402
    build_review_artifacts,
    command_to_shell,
    dataset_inventory,
    metric_attr,
    plain_value,
    read_json,
    write_csv,
    write_json,
    write_text,
)


BASE_AUGMENTATION = {
    "mosaic": 0.2,
    "close_mosaic": 20,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.25,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.25,
    "erasing": 0.0,
}


EXPERIMENTS = {
    "E0_current_control": {
        "purpose": "Existing phase1 control augmentation, re-run with seed 42.",
        "augmentation": BASE_AUGMENTATION,
    },
    "E1_photo_light": {
        "purpose": "Light photometric augmentation.",
        "augmentation": {**BASE_AUGMENTATION, "hsv_h": 0.01, "hsv_s": 0.30, "hsv_v": 0.30},
    },
    "E2_photo_mid": {
        "purpose": "Medium photometric augmentation.",
        "augmentation": {**BASE_AUGMENTATION, "hsv_h": 0.015, "hsv_s": 0.45, "hsv_v": 0.45},
    },
    "E3_geo_mid": {
        "purpose": "Medium geometric augmentation.",
        "augmentation": {
            **BASE_AUGMENTATION,
            "degrees": 10.0,
            "translate": 0.08,
            "scale": 0.35,
            "shear": 2.0,
            "perspective": 0.0002,
        },
    },
    "E4_mosaic_mid": {
        "purpose": "Medium mosaic augmentation.",
        "augmentation": {**BASE_AUGMENTATION, "mosaic": 0.4, "close_mosaic": 30},
    },
    "E5_combined_best": {
        "purpose": "Combined photometric, mosaic, and geometry augmentation.",
        "augmentation": {
            **BASE_AUGMENTATION,
            "mosaic": 0.4,
            "close_mosaic": 30,
            "degrees": 10.0,
            "translate": 0.08,
            "scale": 0.35,
            "shear": 2.0,
            "perspective": 0.0002,
            "hsv_h": 0.015,
            "hsv_s": 0.45,
            "hsv_v": 0.45,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO11s-pose augmentation ablations.")
    parser.add_argument("--data-root", type=Path, default=Path("outputs_experiment/yolo_pose_dataset_ablation_20260624_task2/datasets/modified_100"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("yolo11s-pose.pt"))
    parser.add_argument("--train-project", type=Path, default=Path("runs/pose_aug_ablation"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("outputs/aug_ablation_test_eval"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--keypoint-error-threshold", type=float, default=0.05)
    parser.add_argument("--max-error-images", type=int, default=80)
    parser.add_argument("--max-review-images", type=int, default=80)
    parser.add_argument("--yolo-bin", type=Path, default=Path(".venv/bin/yolo"))
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing runs in --train-project without training.")
    parser.add_argument("--allow-existing", action="store_true", help="Allow writing into existing train/eval directories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalize_args(args)
    validate_inputs(args)

    args.train_project.mkdir(parents=True, exist_ok=True)
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.eval_output_dir / "dataset_summary.json", dataset_inventory(args.data_root, args.data))
    write_experiment_plan(args)

    train_rows = []
    eval_command_rows = []
    summary_rows = []
    for name in args.experiments:
        experiment = EXPERIMENTS[name]
        run_dir = args.train_project / name
        eval_dir = args.eval_output_dir / name
        if not args.eval_only:
            train_rows.append(train_experiment(args, name, experiment, run_dir))
        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing trained best.pt for {name}: {weight_path}")
        eval_rows, summary = evaluate_experiment(args, name, weight_path, eval_dir)
        eval_command_rows.extend(eval_rows)
        summary_rows.append(summary)

    write_csv(args.eval_output_dir / "training_commands.csv", train_rows)
    write_csv(args.eval_output_dir / "evaluation_commands.csv", eval_command_rows)
    write_csv(args.eval_output_dir / "summary_metrics.csv", summary_rows)
    write_summary_md(args.eval_output_dir / "summary_metrics.md", summary_rows)
    write_ellipse_note(REPO_ROOT / "notes" / "ellipse_inference_note.md")
    write_json(args.eval_output_dir / "run_summary.json", {"training": train_rows, "evaluation": summary_rows})
    print(json.dumps({"summary_metrics": str(args.eval_output_dir / "summary_metrics.csv"), "rows": summary_rows}, ensure_ascii=False, indent=2))
    return 0


def normalize_args(args: argparse.Namespace) -> None:
    args.data_root = resolve(args.data_root)
    args.data = resolve(args.data) if args.data else args.data_root / "face_pose.yaml"
    args.model = resolve(args.model)
    args.train_project = resolve(args.train_project)
    args.eval_output_dir = resolve(args.eval_output_dir)
    args.yolo_bin = resolve(args.yolo_bin) if not args.yolo_bin.is_absolute() else args.yolo_bin


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Initial model weight not found: {args.model}")
    yolo_bin = args.yolo_bin if args.yolo_bin.exists() else None
    if yolo_bin is None:
        raise FileNotFoundError(f"YOLO CLI not found: {args.yolo_bin}")


def write_experiment_plan(args: argparse.Namespace) -> None:
    rows = []
    for name in args.experiments:
        item = EXPERIMENTS[name]
        rows.append(
            {
                "experiment": name,
                "purpose": item["purpose"],
                **item["augmentation"],
            }
        )
    write_csv(args.eval_output_dir / "experiment_plan.csv", rows)
    write_json(
        args.eval_output_dir / "experiment_plan.json",
        {
            "model": str(args.model),
            "data": str(args.data),
            "data_root": str(args.data_root),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "seed": args.seed,
            "device": args.device,
            "cutmix": 0,
            "albumentations_extra": "not added; no project-level integration was found",
            "experiments": EXPERIMENTS,
        },
    )


def train_experiment(args: argparse.Namespace, name: str, experiment: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if run_dir.exists() and not args.allow_existing:
        raise FileExistsError(f"Refusing to write into existing training directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = train_command(args, name, experiment["augmentation"])
    write_text(run_dir / "train_command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\ncd " + shlex.quote(str(REPO_ROOT)) + "\n" + command_to_shell(command) + "\n")
    write_json(
        run_dir / "augmentation_config.json",
        {
            "experiment": name,
            "purpose": experiment["purpose"],
            "augmentation": experiment["augmentation"],
            "model": str(args.model),
            "data": str(args.data),
            "seed": args.seed,
        },
    )
    write_text(run_dir / "used_dataset_yaml.txt", str(args.data) + "\n")
    row = run_command("train", name, command, run_dir / "train.log")
    status = {
        "args_yaml": (run_dir / "args.yaml").exists(),
        "results_csv": (run_dir / "results.csv").exists(),
        "results_png": (run_dir / "results.png").exists(),
        "best_pt": (run_dir / "weights" / "best.pt").exists(),
        "last_pt": (run_dir / "weights" / "last.pt").exists(),
    }
    write_json(run_dir / "train_artifact_status.json", status)
    if not all(status.values()):
        raise RuntimeError(f"Missing expected train artifact for {name}: {status}")
    return {
        **row,
        "run_dir": str(run_dir),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "last_pt": str(run_dir / "weights" / "last.pt"),
    }


def train_command(args: argparse.Namespace, name: str, aug: dict[str, Any]) -> list[str]:
    command = [
        str(args.yolo_bin),
        "pose",
        "train",
        f"model={args.model}",
        f"data={args.data}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"patience={args.patience}",
        f"device={args.device}",
        f"project={args.train_project}",
        f"name={name}",
        "exist_ok=True",
        "pretrained=True",
        "plots=True",
        "val=True",
        f"seed={args.seed}",
        "deterministic=True",
        "cache=False",
        "workers=8",
    ]
    for key, value in aug.items():
        command.append(f"{key}={value}")
    return command


def evaluate_experiment(args: argparse.Namespace, name: str, weight_path: Path, eval_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if eval_dir.exists() and not args.allow_existing:
        raise FileExistsError(f"Refusing to write into existing evaluation directory: {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_text(eval_dir / "weight_path.txt", str(weight_path) + "\n")
    write_text(eval_dir / "dataset_yaml.txt", str(args.data) + "\n")
    write_json(
        eval_dir / "eval_conditions.json",
        {
            "split": "test",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "match_iou_threshold": args.match_iou_threshold,
            "threshold_search": False,
        },
    )

    rows = []
    rows.append(
        run_command(
            "inference",
            name,
            [
                sys.executable,
                str(SCRIPT_DIR / "run_yolo11s_aug_inference_compare.py"),
                "predict-one",
                "--weights",
                str(weight_path),
                "--data-root",
                str(args.data_root),
                "--split",
                "test",
                "--output-dir",
                str(eval_dir),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--iou",
                str(args.nms_iou),
                "--device",
                args.device,
            ],
            eval_dir / "logs" / "inference.log",
        )
    )
    rows.append(
        run_command(
            "custom_eval",
            name,
            [
                sys.executable,
                str(SCRIPT_DIR / "evaluate_yolo_pose_metrics.py"),
                "--weights",
                str(weight_path),
                "--data",
                str(args.data),
                "--data-root",
                str(args.data_root),
                "--split",
                "test",
                "--imgsz",
                str(args.imgsz),
                "--device",
                args.device,
                "--conf-threshold",
                str(args.conf),
                "--prediction-conf-min",
                str(args.prediction_conf_min),
                "--iou-threshold",
                str(args.match_iou_threshold),
                "--keypoint-error-threshold",
                str(args.keypoint_error_threshold),
                "--output-dir",
                str(eval_dir / "evaluation"),
                "--save-error-images",
                "--max-error-images",
                str(args.max_error_images),
            ],
            eval_dir / "logs" / "custom_eval.log",
        )
    )
    rows.append(
        run_command(
            "yolo_val",
            name,
            [
                sys.executable,
                str(SCRIPT_DIR / "run_yolo11s_aug_inference_compare.py"),
                "yolo-val-one",
                "--weights",
                str(weight_path),
                "--data",
                str(args.data),
                "--split",
                "test",
                "--output-dir",
                str(eval_dir),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--iou",
                str(args.nms_iou),
                "--device",
                args.device,
            ],
            eval_dir / "logs" / "yolo_val.log",
        )
    )
    build_review_artifacts(eval_dir, args.max_review_images)
    return rows, summary_row(args, name, weight_path, eval_dir)


def run_command(kind: str, experiment: str, command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_command = command_to_shell(command)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shell_command}\n\n")
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.monotonic() - started
    row = {
        "kind": kind,
        "experiment": experiment,
        "command": shell_command,
        "log_path": str(log_path),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
    }
    if result.returncode != 0:
        raise RuntimeError(f"{kind} failed for {experiment}. See {log_path}")
    return row


def summary_row(args: argparse.Namespace, name: str, weight_path: Path, eval_dir: Path) -> dict[str, Any]:
    yolo_val = read_json(eval_dir / "yolo_val_metrics.json")
    custom = read_json(eval_dir / "evaluation" / "test_metrics_summary.json")
    prediction = read_json(eval_dir / "prediction_summary.json")
    box = yolo_val.get("box", {})
    pose = yolo_val.get("pose", {})
    speed = yolo_val.get("speed_ms", {}) or prediction.get("average_speed_ms", {})
    fixed = custom.get("fixed_threshold_detection", {})
    keypoints = custom.get("keypoints", {})
    return {
        "experiment": name,
        "weight_path": str(weight_path),
        "precision": box.get("precision"),
        "recall": box.get("recall"),
        "mAP50": box.get("map50"),
        "mAP50-95": box.get("map50_95"),
        "pose_precision": pose.get("precision"),
        "pose_recall": pose.get("recall"),
        "pose_mAP50": pose.get("map50"),
        "pose_mAP50-95": pose.get("map50_95"),
        "preprocess_time": speed.get("preprocess") if isinstance(speed, dict) else "",
        "inference_time": speed.get("inference") if isinstance(speed, dict) else "",
        "postprocess_time": speed.get("postprocess") if isinstance(speed, dict) else "",
        "notes": "fixed conf/iou; no threshold search",
        "fixed_precision": fixed.get("precision"),
        "fixed_recall": fixed.get("recall"),
        "fixed_f1": fixed.get("f1"),
        "tp": fixed.get("tp"),
        "fp": fixed.get("fp"),
        "fn": fixed.get("fn"),
        "nme": keypoints.get("nme"),
        "pck_0.05": keypoints.get("pck_0.05"),
        "pck_0.10": keypoints.get("pck_0.10"),
        "review_report": str(eval_dir / "review_report.html"),
    }


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "experiment",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "pose_precision",
        "pose_recall",
        "pose_mAP50",
        "pose_mAP50-95",
        "fixed_f1",
        "nme",
        "pck_0.05",
        "tp",
        "fp",
        "fn",
        "notes",
    ]
    lines = ["# YOLO11s Pose Augmentation Ablation Test Metrics", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
    write_text(path, "\n".join(lines) + "\n")


def write_ellipse_note(path: Path) -> None:
    lines = [
        "# Ellipse Inference Note",
        "",
        "This augmentation ablation keeps the standard YOLO-Pose bbox plus keypoints target.",
        "",
        "Ellipse input/output is not mixed into training or metrics. If ellipse inference is needed later, treat it as a separate postprocess or model-head experiment with its own matching metric.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
