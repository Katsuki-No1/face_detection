import argparse
import csv
import html
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_yolo_pose_metrics import image_paths, result_predictions  # noqa: E402


TARGET_EXPERIMENTS = [
    {
        "target": "E0_current_control",
        "augmentation": "control",
        "candidates": ["E0_current_control"],
        "mapping_note": "exact match",
    },
    {
        "target": "E1_photo_light",
        "augmentation": "photometric_light",
        "candidates": ["E1_photo_light", "E1_photometric_light", "E1_hsv_light", "photo_light"],
        "mapping_note": "not mapped to cutmix; cutmix is intentionally excluded",
    },
    {
        "target": "E2_photo_mid",
        "augmentation": "photometric_mid",
        "candidates": ["E2_photo_mid", "E2_photometric_mid", "E2_hsv_mid", "photo_mid"],
        "mapping_note": "not mapped to cutmix; cutmix is intentionally excluded",
    },
    {
        "target": "E3_geo_mid",
        "augmentation": "geometric_mid",
        "candidates": ["E3_geo_mid", "E3_viewpoint_light", "geo_mid", "viewpoint_light"],
        "mapping_note": "mapped to E3_viewpoint_light from existing phase1 run",
    },
    {
        "target": "E4_mosaic_mid",
        "augmentation": "mosaic_mid",
        "candidates": ["E4_mosaic_mid", "E2_mosaic_mid", "mosaic_mid"],
        "mapping_note": "mapped to E2_mosaic_mid from existing phase1 run",
    },
    {
        "target": "E5_combined_best",
        "augmentation": "combined_best",
        "candidates": ["E5_combined_best", "combined_best", "E5_combined"],
        "mapping_note": "requires an existing combined run; none is substituted",
    },
]


KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
FN_TYPES = {"missed_detection", "localization_error"}
REVIEW_CATEGORIES = {
    "false_negative_candidates": FN_TYPES,
    "false_positive_candidates": {"false_positive"},
    "keypoint_error_candidates": {"keypoint_error"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-threshold inference/evaluation for trained YOLO11s Pose augmentation models."
    )
    subparsers = parser.add_subparsers(dest="command")

    compare = subparsers.add_parser("compare", help="Run the full comparison.")
    add_common_compare_args(compare)

    predict = subparsers.add_parser("predict-one", help="Run prediction and export JSONL/CSV for one model.")
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--split", default="test", choices=["train", "val", "test"])
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--imgsz", type=int, required=True)
    predict.add_argument("--conf", type=float, required=True)
    predict.add_argument("--iou", type=float, required=True)
    predict.add_argument("--device", default="0")

    yolo_val = subparsers.add_parser("yolo-val-one", help="Run Ultralytics pose val for one model.")
    yolo_val.add_argument("--weights", type=Path, required=True)
    yolo_val.add_argument("--data", type=Path, required=True)
    yolo_val.add_argument("--split", default="test", choices=["train", "val", "test"])
    yolo_val.add_argument("--output-dir", type=Path, required=True)
    yolo_val.add_argument("--imgsz", type=int, required=True)
    yolo_val.add_argument("--conf", type=float, required=True)
    yolo_val.add_argument("--iou", type=float, required=True)
    yolo_val.add_argument("--device", default="0")

    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(2)
    return parser.parse_args()


def add_common_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs_experiment/yolo_pose_dataset_ablation_20260624_task2/datasets/modified_100"),
    )
    parser.add_argument("--data", type=Path, default=None, help="Defaults to <data-root>/face_pose.yaml.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/face_pose_yolo11s_aug_phase1_20260624_task2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference_aug_compare"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--keypoint-error-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-error-images", type=int, default=80)
    parser.add_argument("--max-review-images", type=int, default=80)
    parser.add_argument("--max-ellipse-samples", type=int, default=12)


def main() -> int:
    args = parse_args()
    if args.command == "predict-one":
        return predict_one(args)
    if args.command == "yolo-val-one":
        return yolo_val_one(args)
    if args.command == "compare":
        return compare(args)
    raise ValueError(f"Unknown command: {args.command}")


def compare(args: argparse.Namespace) -> int:
    args.data_root = resolve_repo_path(args.data_root)
    args.data = resolve_repo_path(args.data) if args.data else args.data_root / "face_pose.yaml"
    args.run_root = resolve_repo_path(args.run_root)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    dataset_summary = dataset_inventory(args.data_root, args.data)
    write_json(args.output_dir / "dataset_summary.json", dataset_summary)

    weight_rows = []
    command_rows = []
    model_rows = []
    for target in TARGET_EXPERIMENTS:
        model_dir = args.output_dir / target["target"]
        model_dir.mkdir(parents=True, exist_ok=True)
        actual_name, weight_path = find_weight(args.run_root, target["candidates"])
        status = "ok" if weight_path and weight_path.exists() else "missing"
        info = {
            "target_experiment": target["target"],
            "actual_experiment": actual_name or "",
            "augmentation": target["augmentation"],
            "status": status,
            "weight_path": str(weight_path) if weight_path else "",
            "dataset_yaml": str(args.data),
            "data_root": str(args.data_root),
            "split": "test",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "match_iou_threshold": args.match_iou_threshold,
            "prediction_conf_min_for_eval_map": args.prediction_conf_min,
            "mapping_note": target["mapping_note"],
            "candidate_names": target["candidates"],
        }
        write_json(model_dir / "model_info.json", info)
        write_text(model_dir / "weight_path.txt", str(weight_path or "") + "\n")
        write_text(model_dir / "dataset_yaml.txt", str(args.data) + "\n")
        write_text(model_dir / "conditions.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        weight_rows.append(info)

        if status != "ok":
            write_text(model_dir / "status.txt", "missing\n")
            model_rows.append(metric_missing_row(info))
            continue

        prediction_command = [
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
            str(model_dir),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.nms_iou),
            "--device",
            args.device,
        ]
        command_rows.append(run_logged_command("inference", target["target"], prediction_command, model_dir / "logs" / "inference.log"))

        eval_dir = model_dir / "evaluation"
        eval_command = [
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
            str(eval_dir),
            "--save-error-images",
            "--max-error-images",
            str(args.max_error_images),
        ]
        command_rows.append(run_logged_command("custom_eval", target["target"], eval_command, model_dir / "logs" / "custom_eval.log"))

        yolo_val_command = [
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
            str(model_dir),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.nms_iou),
            "--device",
            args.device,
        ]
        command_rows.append(run_logged_command("yolo_val", target["target"], yolo_val_command, model_dir / "logs" / "yolo_val.log"))

        build_review_artifacts(model_dir, args.max_review_images)
        build_ellipse_preview(model_dir, args.output_dir / "ellipse_postprocess_preview" / target["target"], args.max_ellipse_samples)
        model_rows.append(metric_row(info, model_dir))

    write_csv(args.output_dir / "weights_inventory.csv", weight_rows)
    write_csv(args.output_dir / "executed_commands.csv", command_rows)
    write_commands_md(args.output_dir / "executed_commands.md", command_rows)
    write_csv(args.output_dir / "summary_metrics.csv", model_rows)
    write_summary_md(args.output_dir / "summary_metrics.md", model_rows)
    write_ellipse_note(args.output_dir / "ellipse_inference_note.md")
    write_final_readme(args.output_dir / "README.md", dataset_summary, weight_rows, model_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "models": model_rows}, ensure_ascii=False, indent=2))
    return 0


def resolve_repo_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    return path if path.is_absolute() else REPO_ROOT / path


def find_weight(run_root: Path, candidates: list[str]) -> tuple[str | None, Path | None]:
    for candidate in candidates:
        weight = run_root / candidate / "weights" / "best.pt"
        if weight.exists():
            return candidate, weight.resolve()
    for candidate in candidates:
        for weight in sorted(run_root.glob(f"**/{candidate}/weights/best.pt")):
            if weight.exists():
                return candidate, weight.resolve()
    return None, None


def run_logged_command(kind: str, experiment: str, command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    shell_command = command_to_shell(command)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shell_command}\n\n")
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
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


def command_to_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def predict_one(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Use the repo .venv for this script.") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = image_paths(args.data_root, args.split)
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(args.output_dir),
        name="predictions",
        exist_ok=True,
        stream=True,
        verbose=True,
    )

    jsonl_path = args.output_dir / "predictions.jsonl"
    csv_path = args.output_dir / "predictions.csv"
    image_rows = []
    prediction_rows = []
    object_count = 0
    speed_totals: dict[str, float] = defaultdict(float)
    speed_counts: Counter[str] = Counter()
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for image_path, result in zip(images, results):
            predictions = result_predictions(result)
            speed = plain_value(getattr(result, "speed", {}))
            if isinstance(speed, dict):
                for key, value in speed.items():
                    if isinstance(value, (int, float)):
                        speed_totals[str(key)] += float(value)
                        speed_counts[str(key)] += 1
            record = {
                "image": str(image_path),
                "prediction_count": len(predictions),
                "speed_ms": speed,
                "objects": predictions,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            image_rows.append(
                {
                    "image": str(image_path),
                    "prediction_count": len(predictions),
                    "preprocess_ms": speed.get("preprocess") if isinstance(speed, dict) else "",
                    "inference_ms": speed.get("inference") if isinstance(speed, dict) else "",
                    "postprocess_ms": speed.get("postprocess") if isinstance(speed, dict) else "",
                }
            )
            for pred_index, prediction in enumerate(predictions):
                row: dict[str, Any] = {
                    "image": str(image_path),
                    "prediction_index": pred_index,
                    "score": prediction["score"],
                    "bbox_x1": prediction["bbox"][0],
                    "bbox_y1": prediction["bbox"][1],
                    "bbox_x2": prediction["bbox"][2],
                    "bbox_y2": prediction["bbox"][3],
                }
                for point in prediction.get("keypoints", []):
                    name = point.get("name", "")
                    row[f"{name}_x"] = point.get("x", "")
                    row[f"{name}_y"] = point.get("y", "")
                    row[f"{name}_score"] = point.get("score", "")
                prediction_rows.append(row)
                object_count += 1
    write_csv(csv_path, prediction_rows)
    write_csv(args.output_dir / "prediction_images.csv", image_rows)
    average_speed = {
        key: speed_totals[key] / speed_counts[key]
        for key in sorted(speed_totals)
        if speed_counts[key]
    }
    summary = {
        "weights": str(args.weights),
        "data_root": str(args.data_root),
        "split": args.split,
        "images": len(images),
        "predicted_objects": object_count,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "prediction_dir": str(args.output_dir / "predictions"),
        "predictions_jsonl": str(jsonl_path),
        "predictions_csv": str(csv_path),
        "average_speed_ms": average_speed,
    }
    write_json(args.output_dir / "prediction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def yolo_val_one(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Use the repo .venv for this script.") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        project=str(args.output_dir),
        name="yolo_val",
        exist_ok=True,
        verbose=True,
    )
    box = getattr(metrics, "box", None)
    pose = getattr(metrics, "pose", None)
    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "box": metric_group(box),
        "pose": metric_group(pose),
        "speed_ms": plain_value(getattr(metrics, "speed", {})),
        "results_dict": plain_value(getattr(metrics, "results_dict", {})),
        "save_dir": str(getattr(metrics, "save_dir", args.output_dir / "yolo_val")),
    }
    write_json(args.output_dir / "yolo_val_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def metric_group(value: Any) -> dict[str, float | None]:
    return {
        "precision": metric_attr(value, "mp"),
        "recall": metric_attr(value, "mr"),
        "map50": metric_attr(value, "map50"),
        "map50_95": metric_attr(value, "map"),
    }


def metric_attr(obj: Any, name: str) -> float | None:
    if obj is None or not hasattr(obj, name):
        return None
    try:
        return float(getattr(obj, name))
    except (TypeError, ValueError):
        return None


def plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def dataset_inventory(data_root: Path, data_yaml: Path) -> dict[str, Any]:
    split_rows = {}
    for split in ["train", "val", "test"]:
        images = image_paths(data_root, split)
        label_dir = data_root / "labels" / split
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        instances = 0
        positive_labels = 0
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            instances += len(lines)
            if lines:
                positive_labels += 1
        split_rows[split] = {
            "images": len(images),
            "labels": len(labels),
            "positive_labels": positive_labels,
            "instances": instances,
        }
    return {
        "data_root": str(data_root),
        "data_yaml": str(data_yaml),
        "splits": split_rows,
    }


def build_review_artifacts(model_dir: Path, max_review_images: int) -> None:
    failure_path = model_dir / "evaluation" / "test_failure_analysis.csv"
    failures = read_csv(failure_path)
    rows = []
    category_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    copied: set[tuple[str, str]] = set()
    for index, row in enumerate(failures):
        failure_type = row.get("failure_type", "")
        category = review_category(failure_type)
        if not category:
            continue
        image_path = Path(row.get("image", ""))
        source = review_source_image(model_dir, image_path)
        destination = ""
        key = (category, image_path.name)
        if source and source.exists() and key not in copied and category_counts[category] < max_review_images:
            destination_path = model_dir / category / f"{index:03d}_{image_path.name}"
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_path)
            copied.add(key)
            destination = str(destination_path)
            category_counts[category] += 1
        tags = row.get("tags", "")
        for tag in split_tags(tags):
            tag_counts[tag] += 1
        review_tags = requested_review_tags(tags)
        rows.append(
            {
                "category": category,
                "failure_type": failure_type,
                "image": str(image_path),
                "review_image": destination,
                "score": row.get("score", ""),
                "iou": row.get("iou", ""),
                "mean_keypoint_error": row.get("mean_keypoint_error", ""),
                "gt_index": row.get("gt_index", ""),
                "prediction_index": row.get("prediction_index", ""),
                "structural_tags": tags,
                "review_tags": "|".join(review_tags),
            }
        )
    write_csv(model_dir / "review_samples.csv", rows)
    write_csv(
        model_dir / "review_tag_summary.csv",
        [
            {"tag": tag, "failure_count": count}
            for tag, count in sorted(tag_counts.items())
        ],
    )
    write_review_html(model_dir / "review_report.html", rows)


def review_category(failure_type: str) -> str:
    for category, failure_types in REVIEW_CATEGORIES.items():
        if failure_type in failure_types:
            return category
    return ""


def review_source_image(model_dir: Path, image_path: Path) -> Path | None:
    candidates = [
        model_dir / "evaluation" / "error_images" / image_path.name,
        model_dir / "predictions" / image_path.name,
        image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def requested_review_tags(tags: str) -> list[str]:
    values = set(split_tags(tags))
    mapped = []
    if "small_face" in values:
        mapped.append("small")
    if "edge_face" in values:
        mapped.append("edge")
    return mapped


def split_tags(tags: str) -> list[str]:
    return [tag for tag in str(tags).split("|") if tag]


def write_review_html(path: Path, rows: list[dict[str, Any]]) -> None:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    sections = []
    for category in REVIEW_CATEGORIES:
        cards = []
        for row in by_category.get(category, []):
            review_image = row.get("review_image") or ""
            if review_image:
                rel_image = Path(review_image).resolve().relative_to(path.parent.resolve())
                image_html = f'<img src="{html.escape(str(rel_image))}" alt="{html.escape(Path(row["image"]).name)}">'
            else:
                image_html = "<div class=\"missing\">image not copied</div>"
            meta = " / ".join(
                part
                for part in [
                    row.get("failure_type", ""),
                    f'score {row.get("score")}' if row.get("score") else "",
                    f'iou {row.get("iou")}' if row.get("iou") else "",
                    row.get("review_tags", ""),
                ]
                if part
            )
            cards.append(
                "\n".join(
                    [
                        '<article class="card">',
                        image_html,
                        f'<div class="caption">{html.escape(Path(row["image"]).name)}</div>',
                        f'<div class="meta">{html.escape(meta)}</div>',
                        "</article>",
                    ]
                )
            )
        sections.append(
            "\n".join(
                [
                    f"<h2>{html.escape(category)}</h2>",
                    '<div class="grid">',
                    "\n".join(cards) if cards else '<div class="empty">no samples</div>',
                    "</div>",
                ]
            )
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLO Pose Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #171717; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 22px 0 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #d7d7d2; border-radius: 6px; padding: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    .caption {{ font-size: 12px; margin-top: 6px; word-break: break-all; }}
    .meta, .empty, .missing {{ font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <h1>YOLO Pose Review</h1>
  {"".join(sections)}
</body>
</html>
"""
    write_text(path, document)


def build_ellipse_preview(model_dir: Path, output_dir: Path, max_samples: int) -> None:
    rows = read_csv(model_dir / "predictions.csv")
    if not rows:
        return
    try:
        import cv2  # type: ignore
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    preview_rows = []
    used_images: set[str] = set()
    for row in rows:
        if written >= max_samples:
            break
        image_path = Path(row.get("image", ""))
        if not image_path.exists() or str(image_path) in used_images:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        bbox = [
            safe_float(row.get("bbox_x1")),
            safe_float(row.get("bbox_y1")),
            safe_float(row.get("bbox_x2")),
            safe_float(row.get("bbox_y2")),
        ]
        if any(value is None for value in bbox):
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox if value is not None]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        axis_x = max(1, int(round((x2 - x1) / 2.0)))
        axis_y = max(1, int(round((y2 - y1) / 2.0)))
        angle = keypoint_angle(row)
        cv2.ellipse(
            image,
            (int(round(cx)), int(round(cy))),
            (axis_x, axis_y),
            angle,
            0,
            360,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 0, 255),
            1,
        )
        output_path = output_dir / f"{image_path.stem}_ellipse_preview.jpg"
        cv2.imwrite(str(output_path), image)
        preview_rows.append(
            {
                "image": str(image_path),
                "preview": str(output_path),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "ellipse_cx": cx,
                "ellipse_cy": cy,
                "ellipse_axis_x": axis_x,
                "ellipse_axis_y": axis_y,
                "ellipse_angle_deg": angle,
                "source": "bbox axes; eye keypoints only set angle when both eyes are present",
            }
        )
        used_images.add(str(image_path))
        written += 1
    write_csv(output_dir / "ellipse_preview_index.csv", preview_rows)


def keypoint_angle(row: dict[str, str]) -> float:
    e1x = safe_float(row.get("e1_x"))
    e1y = safe_float(row.get("e1_y"))
    e2x = safe_float(row.get("e2_x"))
    e2y = safe_float(row.get("e2_y"))
    if None in (e1x, e1y, e2x, e2y):
        return 0.0
    if abs(float(e1x) - float(e2x)) < 1e-6 and abs(float(e1y) - float(e2y)) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(float(e2y) - float(e1y), float(e2x) - float(e1x)))


def metric_missing_row(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_experiment": info["target_experiment"],
        "actual_experiment": info.get("actual_experiment", ""),
        "augmentation": info.get("augmentation", ""),
        "status": "missing",
        "weight_path": info.get("weight_path", ""),
        "dataset_yaml": info.get("dataset_yaml", ""),
        "imgsz": info.get("imgsz", ""),
        "conf": info.get("conf", ""),
        "nms_iou": info.get("nms_iou", ""),
        "match_iou_threshold": info.get("match_iou_threshold", ""),
    }


def metric_row(info: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    yolo_val = read_json(model_dir / "yolo_val_metrics.json")
    custom = read_json(model_dir / "evaluation" / "test_metrics_summary.json")
    prediction_summary = read_json(model_dir / "prediction_summary.json")
    fixed = custom.get("fixed_threshold_detection", {})
    keypoints = custom.get("keypoints", {})
    box = yolo_val.get("box", {})
    pose = yolo_val.get("pose", {})
    speed = yolo_val.get("speed_ms", {}) or prediction_summary.get("average_speed_ms", {})
    return {
        "target_experiment": info["target_experiment"],
        "actual_experiment": info.get("actual_experiment", ""),
        "augmentation": info.get("augmentation", ""),
        "status": "ok",
        "weight_path": info.get("weight_path", ""),
        "dataset_yaml": info.get("dataset_yaml", ""),
        "imgsz": info.get("imgsz", ""),
        "conf": info.get("conf", ""),
        "nms_iou": info.get("nms_iou", ""),
        "match_iou_threshold": info.get("match_iou_threshold", ""),
        "test_images": custom.get("images", ""),
        "gt_objects": custom.get("ground_truth_objects", ""),
        "predicted_objects": custom.get("predicted_objects", ""),
        "yolo_box_precision": box.get("precision"),
        "yolo_box_recall": box.get("recall"),
        "yolo_box_map50": box.get("map50"),
        "yolo_box_map50_95": box.get("map50_95"),
        "yolo_pose_precision": pose.get("precision"),
        "yolo_pose_recall": pose.get("recall"),
        "yolo_pose_map50": pose.get("map50"),
        "yolo_pose_map50_95": pose.get("map50_95"),
        "fixed_precision": fixed.get("precision"),
        "fixed_recall": fixed.get("recall"),
        "fixed_f1": fixed.get("f1"),
        "tp": fixed.get("tp"),
        "fp": fixed.get("fp"),
        "fn": fixed.get("fn"),
        "mean_matched_iou": fixed.get("mean_matched_iou"),
        "nme": keypoints.get("nme"),
        "pck_0.05": keypoints.get("pck_0.05"),
        "pck_0.10": keypoints.get("pck_0.10"),
        "preprocess_ms": speed.get("preprocess") if isinstance(speed, dict) else "",
        "inference_ms": speed.get("inference") if isinstance(speed, dict) else "",
        "postprocess_ms": speed.get("postprocess") if isinstance(speed, dict) else "",
        "review_report": str(model_dir / "review_report.html"),
    }


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "target_experiment",
        "actual_experiment",
        "status",
        "fixed_precision",
        "fixed_recall",
        "fixed_f1",
        "yolo_box_map50",
        "yolo_box_map50_95",
        "yolo_pose_map50",
        "yolo_pose_map50_95",
        "nme",
        "pck_0.05",
        "tp",
        "fp",
        "fn",
    ]
    lines = ["# Summary Metrics", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
    write_text(path, "\n".join(lines) + "\n")


def write_final_readme(
    path: Path,
    dataset_summary: dict[str, Any],
    weight_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    ok_rows = [row for row in metric_rows if row.get("status") == "ok"]
    missing_rows = [row for row in metric_rows if row.get("status") == "missing"]
    control = next((row for row in ok_rows if row["target_experiment"] == "E0_current_control"), None)
    lines = [
        "# YOLO11s Pose Augmentation Inference Compare",
        "",
        "## Purpose",
        "",
        "Run fixed test-split inference and evaluation for trained YOLO11s-pose augmentation experiments. No training, package install, data download, split change, or threshold search was performed.",
        "",
        "## Dataset",
        "",
        f"- data root: `{dataset_summary['data_root']}`",
        f"- dataset yaml: `{dataset_summary['data_yaml']}`",
        f"- split counts: `{dataset_summary['splits']}`",
        "",
        "## Target Models",
        "",
        "| target | actual | status | weight | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in weight_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_cell(row.get("target_experiment")),
                    format_cell(row.get("actual_experiment")),
                    format_cell(row.get("status")),
                    format_cell(row.get("weight_path")),
                    format_cell(row.get("mapping_note")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Inference Conditions",
            "",
            "- split: `test`",
            "- imgsz: `960`",
            "- confidence threshold: `0.25`",
            "- NMS IoU: `0.7`",
            "- save image overlays: `true`",
            "- save txt/conf: `true`",
            "",
            "## Evaluation Conditions",
            "",
            "- Ultralytics `model.val` was run on the fixed test split with `conf=0.25` and `iou=0.7`.",
            "- The existing local evaluator was also run with fixed `conf=0.25`, low raw prediction collection at `0.001`, and GT matching IoU `0.5` for review CSVs.",
            "- No confidence or IoU threshold sweep was run.",
            "",
            "## Metrics Summary",
            "",
            "`summary_metrics.csv` and `summary_metrics.md` contain the complete table.",
            "",
        ]
    )
    if ok_rows:
        headers = ["target_experiment", "fixed_precision", "fixed_recall", "fixed_f1", "yolo_pose_map50_95", "nme", "pck_0.05", "tp", "fp", "fn"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in ok_rows:
            lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
        lines.append("")
    if missing_rows:
        lines.append("Missing models: " + ", ".join(row["target_experiment"] for row in missing_rows) + ".")
        lines.append("")
    lines.extend(["## Augmentation Trends", ""])
    if control:
        lines.extend(trend_lines(control, ok_rows))
    else:
        lines.append("- E0_current_control was not available, so E0 deltas were not computed.")
    lines.extend(["", "## Review Focus Summary", ""])
    lines.extend(review_focus_lines(ok_rows))
    lines.extend(
        [
            "",
            "## Next Augmentation Checks",
            "",
            "- Train and then evaluate the missing photometric light/mid runs if dark or low-quality misses remain important.",
            "- Keep mosaic-mid as a small-face candidate if precision does not drop on review images.",
            "- Keep geometric/viewpoint as an angle and pose-change candidate if keypoint quality remains stable.",
            "- Evaluate a combined run only after the single-factor runs exist under the same fixed split.",
            "",
            "## Ellipse Inference Decision",
            "",
            "Standard YOLO-Pose remains bbox plus keypoints. Ellipse input was not added. The optional preview in `ellipse_postprocess_preview/` draws postprocess-only bbox-derived ellipses and is not included in metrics.",
            "",
            "## Review Artifacts",
            "",
            "Per-model review files are under each model directory: `review_samples.csv`, `review_report.html`, `false_negative_candidates/`, `false_positive_candidates/`, and `keypoint_error_candidates/`.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def trend_lines(control: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        if row["target_experiment"] == control["target_experiment"]:
            continue
        precision_delta = nullable_delta(row.get("fixed_precision"), control.get("fixed_precision"))
        recall_delta = nullable_delta(row.get("fixed_recall"), control.get("fixed_recall"))
        nme_delta = nullable_delta(row.get("nme"), control.get("nme"))
        pck_delta = nullable_delta(row.get("pck_0.05"), control.get("pck_0.05"))
        lines.append(
            "- "
            + row["target_experiment"]
            + f": recall delta {format_signed(recall_delta)}, precision delta {format_signed(precision_delta)}, "
            + f"NME delta {format_signed(nme_delta)}, PCK@0.05 delta {format_signed(pck_delta)}."
        )
    if len(rows) <= 1:
        lines.append("- No non-control evaluated model was available for E0 comparison.")
    return lines


def review_focus_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "- Dataset-side manual tags for angle, dark, low_quality, occluded, and blur were not available in the fixed test split artifacts.",
        "- Review summaries therefore use automatic structural tags from the local evaluator: small_face, edge_face, crowded_image, missing_gt_keypoints, and very_sparse_keypoints.",
    ]
    tag_names = ["small_face", "edge_face", "crowded_image", "missing_gt_keypoints", "very_sparse_keypoints"]
    summaries = []
    for row in rows:
        report = row.get("review_report")
        if not report:
            continue
        tag_path = Path(str(report)).parent / "review_tag_summary.csv"
        counts = {item["tag"]: item["failure_count"] for item in read_csv(tag_path)}
        if not counts:
            continue
        summaries.append(
            row["target_experiment"]
            + ": "
            + ", ".join(f"{tag}={counts.get(tag, '0')}" for tag in tag_names)
        )
    if summaries:
        lines.append("- Structural failure counts: " + " | ".join(summaries) + ".")
    return lines


def write_commands_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Executed Commands", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['experiment']} {row['kind']}",
                "",
                "```bash",
                row["command"],
                "```",
                "",
                f"- log: `{row['log_path']}`",
                f"- returncode: `{row['returncode']}`",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def write_ellipse_note(path: Path) -> None:
    lines = [
        "# Ellipse Inference Note",
        "",
        "The main evaluation keeps the standard YOLO-Pose output format: bounding boxes and five keypoints.",
        "",
        "Ellipse annotations exist in the conversion metadata, but this run does not add ellipse input or train an ellipse head. The preview directory contains only postprocess samples:",
        "",
        "- bbox center and bbox half-size become the ellipse center and axes.",
        "- eye keypoints set a rough rotation angle when both eye points are present.",
        "- these previews are excluded from `summary_metrics.csv` and all adoption decisions.",
        "",
        "A true ellipse metric would need a separate annotation-to-prediction matching definition and should be handled as a separate experiment.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nullable_delta(value: Any, baseline: Any) -> float | None:
    current = safe_float(value)
    base = safe_float(baseline)
    if current is None or base is None:
        return None
    return current - base


def format_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    if "\n" in text:
        text = text.replace("\n", " ")
    return text.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
