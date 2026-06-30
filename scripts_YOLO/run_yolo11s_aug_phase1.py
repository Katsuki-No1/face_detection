import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
SPLITS = ["train", "val", "test"]
FN_FAILURE_TYPES = {"missed_detection", "localization_error"}
TAG_TARGETS = ["angle", "occlusion", "pixelated_blur", "small", "crowded", "edge"]
TAG_ALIASES = {
    "angle": {"angle", "profile", "side", "yaw", "pitch", "top", "overhead", "viewpoint"},
    "occlusion": {"occlusion", "occluded", "occluded_candidate", "mask", "partial"},
    "pixelated_blur": {"pixelated", "pixelation", "mosaic", "blur", "blur_dark", "lowres", "quality"},
    "small": {"small", "small_face"},
    "crowded": {"crowded", "crowded_image"},
    "edge": {"edge", "edge_face"},
}


@dataclass(frozen=True)
class AugExperiment:
    name: str
    purpose: str
    augmentation: dict[str, float]


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
    "E0_current_control": AugExperiment(
        "E0_current_control",
        "Control run for same data/model/training settings.",
        BASE_AUGMENTATION,
    ),
    "E1_cutmix_light": AugExperiment(
        "E1_cutmix_light",
        "Light CutMix check; do not treat as direct face-occlusion augmentation.",
        {**BASE_AUGMENTATION, "cutmix": 0.1},
    ),
    "E2_mosaic_mid": AugExperiment(
        "E2_mosaic_mid",
        "Stronger Mosaic check for small/crowded/edge/partial objects.",
        {**BASE_AUGMENTATION, "mosaic": 0.5, "close_mosaic": 40},
    ),
    "E3_viewpoint_light": AugExperiment(
        "E3_viewpoint_light",
        "Light roll/shear/perspective check; not a yaw/pitch/profile synthesis.",
        {**BASE_AUGMENTATION, "degrees": 7.0, "shear": 2.0, "perspective": 0.0002},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11s Pose train-time augmentation Phase 1 experiments."
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML. Defaults to <data-root>/face_pose.yaml.")
    parser.add_argument("--output-dir", type=Path, default=Path("output_experiment/05_yolo_aug_phase1"))
    parser.add_argument("--project", type=Path, default=None, help="YOLO run directory. Defaults to <output-dir>/runs.")
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.01:0.95:0.01")
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--keypoint-error-threshold", type=float, default=0.05)
    parser.add_argument("--yolo-bin", type=Path, default=Path(".venv/bin/yolo"))
    parser.add_argument("--manual-tags-csv", type=Path, default=None)
    parser.add_argument("--near-frame-window", type=int, default=3)
    parser.add_argument("--near-time-window", type=float, default=2.0)
    parser.add_argument("--max-visualizations", type=int, default=80)
    parser.add_argument("--max-error-images", type=int, default=80)
    parser.add_argument("--save-error-images", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=sorted(EXPERIMENTS),
        help="Subset of experiments to run/evaluate.",
    )
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for dataset audit.") from error
    return cv2


def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "labels" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() == ".txt")


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def canonical_tags(tokens: list[str]) -> set[str]:
    normalized = {normalize_token(token) for token in tokens if normalize_token(token)}
    tags = set()
    for canonical, aliases in TAG_ALIASES.items():
        if normalized & aliases:
            tags.add(canonical)
    return tags


def split_tag_text(value: str) -> list[str]:
    if not value:
        return []
    return [token for token in re.split(r"[|,;/\s]+", value) if token]


def load_manual_tag_index(path: Path | None) -> dict[str, set[str]]:
    rows = read_csv(path) if path else []
    index: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_value = row.get("image") or row.get("image_path") or row.get("path") or ""
        if not image_value:
            continue
        tokens: list[str] = []
        for key in ["tags", "tag", "manual_failure_type", "failure_type", "reason", "reasons", "auto_failure_type"]:
            tokens.extend(split_tag_text(row.get(key, "")))
        tags = canonical_tags(tokens)
        if tags:
            index[Path(image_value).name].update(tags)
            index[Path(image_value).stem].update(tags)
    return index


def tags_for_image(image_path: Path, manual_tag_index: dict[str, set[str]]) -> set[str]:
    return set(manual_tag_index.get(image_path.name, set())) | set(manual_tag_index.get(image_path.stem, set()))


def bbox_area_xyxy(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def structural_tags(box: list[float], width: int, height: int, image_gt_count: int) -> set[str]:
    area_ratio = bbox_area_xyxy(box) / max(1.0, float(width * height))
    tags = set()
    if area_ratio < 0.01:
        tags.add("small")
    if image_gt_count >= 3:
        tags.add("crowded")
    margin_x = width * 0.02
    margin_y = height * 0.02
    if box[0] <= margin_x or box[1] <= margin_y or box[2] >= width - margin_x or box[3] >= height - margin_y:
        tags.add("edge")
    return tags


def load_label_objects(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    objects = []
    if not label_path.exists():
        return objects
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(part) for part in line.split()]
        if len(values) < 5 + len(KEYPOINT_NAMES) * 3:
            raise ValueError(f"Invalid YOLO pose label at {label_path}:{line_number}")
        xc, yc, bw, bh = values[1:5]
        box_width = bw * width
        box_height = bh * height
        cx = xc * width
        cy = yc * height
        keypoints = []
        raw_keypoints = values[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            x_value = raw_keypoints[index * 3]
            y_value = raw_keypoints[index * 3 + 1]
            visibility = int(raw_keypoints[index * 3 + 2])
            keypoints.append(
                {
                    "name": name,
                    "x": x_value,
                    "y": y_value,
                    "visibility": visibility,
                }
            )
        objects.append(
            {
                "bbox": [
                    cx - box_width / 2.0,
                    cy - box_height / 2.0,
                    cx + box_width / 2.0,
                    cy + box_height / 2.0,
                ],
                "bbox_norm": [xc, yc, bw, bh],
                "keypoints": keypoints,
            }
        )
    return objects


def frame_info(path: Path) -> dict[str, Any]:
    stem = path.stem
    if "_frame_" not in stem:
        return {"family": stem, "frame_index": None, "time_sec": None}
    family, tail = stem.split("_frame_", 1)
    frame_match = re.match(r"(\d+)", tail)
    time_match = re.search(r"_(\d+(?:\.\d+)?)s$", tail)
    return {
        "family": family,
        "frame_index": int(frame_match.group(1)) if frame_match else None,
        "time_sec": float(time_match.group(1)) if time_match else None,
    }


def dataset_audit(
    data_root: Path,
    data_yaml: Path,
    manual_tag_index: dict[str, set[str]],
    near_frame_window: int,
    near_time_window: float,
) -> dict[str, Any]:
    cv2 = import_cv2()
    try:
        import yaml  # type: ignore
    except Exception as error:
        raise RuntimeError("PyYAML is required. Install requirements.txt or run with .venv/bin/python.") from error
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    images_by_split = {split: image_paths(data_root, split) for split in SPLITS}
    labels_by_split = {split: label_paths(data_root, split) for split in SPLITS}
    rows = []
    bbox_area_ratios: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    visibility_by_name: dict[str, Counter[str]] = {name: Counter() for name in KEYPOINT_NAMES}
    tag_image_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    tag_instance_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    image_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    frame_rows = []
    failures = []

    for split, images in images_by_split.items():
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels_by_split[split]}
        for stem in sorted(image_stems - label_stems):
            failures.append({"split": split, "stem": stem, "reason": "missing_label"})
        for stem in sorted(label_stems - image_stems):
            failures.append({"split": split, "stem": stem, "reason": "extra_label"})

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append({"split": split, "image": str(image_path), "reason": "image_read_failed"})
                continue
            height, width = image.shape[:2]
            objects = load_label_objects(label_path_for_image(data_root, split, image_path), width, height)
            image_tags = tags_for_image(image_path, manual_tag_index)
            if objects:
                for tag in image_tags:
                    tag_image_counts[split][tag] += 1
            else:
                tag_image_counts[split]["negative"] += 1

            for item in objects:
                box = item["bbox"]
                area_ratio = bbox_area_xyxy(box) / max(1.0, float(width * height))
                bbox_area_ratios.append(area_ratio)
                bbox_widths.append(float(item["bbox_norm"][2]))
                bbox_heights.append(float(item["bbox_norm"][3]))
                instance_tags = set(image_tags) | structural_tags(box, width, height, len(objects))
                for tag in instance_tags:
                    tag_instance_counts[split][tag] += 1
                for point in item["keypoints"]:
                    visibility_by_name[point["name"]][str(point["visibility"])] += 1

            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path_for_image(data_root, split, image_path)),
                    "width": width,
                    "height": height,
                    "objects": len(objects),
                    "positive": bool(objects),
                    "tags": "|".join(sorted(image_tags)),
                    "sha1": sha1_file(image_path),
                    **frame_info(image_path),
                }
            )
            image_hashes[rows[-1]["sha1"]].append({"split": split, "image": str(image_path)})
            frame_rows.append(rows[-1])

    duplicate_images = [
        {"sha1": digest, "items": items}
        for digest, items in sorted(image_hashes.items())
        if len({item["split"] for item in items}) > 1
    ]
    near_frame_pairs = find_near_frame_pairs(frame_rows, near_frame_window, near_time_window)
    flip_idx_ok = config.get("flip_idx") == [1, 0, 2, 4, 3]
    positive_images_by_split = {
        split: sum(1 for row in rows if row["split"] == split and row["objects"] > 0)
        for split in SPLITS
    }
    images_count_by_split = {split: len(images_by_split[split]) for split in SPLITS}
    report = {
        "data_root": str(data_root),
        "data_yaml": str(data_yaml),
        "status": "pass" if not failures and not duplicate_images and not near_frame_pairs and flip_idx_ok else "warn",
        "images_by_split": images_count_by_split,
        "labels_by_split": {split: len(labels_by_split[split]) for split in SPLITS},
        "positive_images_by_split": positive_images_by_split,
        "negative_images_by_split": {
            split: images_count_by_split[split] - positive_images_by_split[split]
            for split in SPLITS
        },
        "instances_by_split": {
            split: sum(int(row["objects"]) for row in rows if row["split"] == split)
            for split in SPLITS
        },
        "tag_image_counts": {split: dict(sorted(counter.items())) for split, counter in tag_image_counts.items()},
        "tag_instance_counts": {split: dict(sorted(counter.items())) for split, counter in tag_instance_counts.items()},
        "bbox_area_ratio": stats(bbox_area_ratios),
        "bbox_width_norm": stats(bbox_widths),
        "bbox_height_norm": stats(bbox_heights),
        "visibility_by_keypoint": {
            name: dict(sorted(counter.items())) for name, counter in visibility_by_name.items()
        },
        "yaml_kpt_shape": config.get("kpt_shape"),
        "yaml_flip_idx": config.get("flip_idx"),
        "expected_flip_idx": [1, 0, 2, 4, 3],
        "flip_idx_ok": flip_idx_ok,
        "split_duplicate_images": duplicate_images[:200],
        "near_frame_cross_split_pairs": near_frame_pairs[:200],
        "near_frame_cross_split_pair_count": len(near_frame_pairs),
        "failures": failures[:200],
        "failure_count": len(failures),
    }
    return {"report": report, "image_rows": rows}


def find_near_frame_pairs(
    rows: list[dict[str, Any]],
    near_frame_window: int,
    near_time_window: float,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family"):
            by_family[str(row["family"])].append(row)
    pairs = []
    for family, items in by_family.items():
        for index, first in enumerate(items):
            for second in items[index + 1 :]:
                if first["split"] == second["split"]:
                    continue
                frame_delta = None
                if first.get("frame_index") is not None and second.get("frame_index") is not None:
                    frame_delta = abs(int(first["frame_index"]) - int(second["frame_index"]))
                time_delta = None
                if first.get("time_sec") is not None and second.get("time_sec") is not None:
                    time_delta = abs(float(first["time_sec"]) - float(second["time_sec"]))
                is_near_frame = frame_delta is not None and frame_delta <= near_frame_window
                is_near_time = time_delta is not None and time_delta <= near_time_window
                if is_near_frame or is_near_time:
                    pairs.append(
                        {
                            "family": family,
                            "first_split": first["split"],
                            "first_image": first["image"],
                            "second_split": second["split"],
                            "second_image": second["image"],
                            "frame_delta": frame_delta,
                            "time_delta": time_delta,
                        }
                    )
    return pairs


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


def command_to_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def yolo_command(args: argparse.Namespace, experiment: AugExperiment, run_dir: Path) -> list[str]:
    yolo_bin = args.yolo_bin if args.yolo_bin.exists() else Path("yolo")
    command = [
        str(yolo_bin),
        "pose",
        "train",
        f"model={args.model}",
        f"data={args.data}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"patience={args.patience}",
        f"device={args.device}",
        f"project={run_dir.parent}",
        f"name={run_dir.name}",
        "exist_ok=True",
        "pretrained=True",
        "plots=True",
        "val=True",
        f"seed={args.seed}",
        "deterministic=True",
    ]
    for key, value in experiment.augmentation.items():
        command.append(f"{key}={value}")
    return command


def train_experiment(args: argparse.Namespace, experiment: AugExperiment, run_dir: Path) -> dict[str, Any]:
    weights = run_dir / "weights" / "best.pt"
    command = yolo_command(args, experiment, run_dir)
    write_json(
        args.output_dir / "configs" / f"{experiment.name}.json",
        {
            "name": experiment.name,
            "purpose": experiment.purpose,
            "data": str(args.data),
            "model": args.model,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
            "augmentation": experiment.augmentation,
            "command": command,
        },
    )
    command_path = args.output_dir / "commands" / f"train_{experiment.name}.sh"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\ncd "
        + shlex.quote(str(Path.cwd()))
        + "\n"
        + command_to_shell(command)
        + "\n",
        encoding="utf-8",
    )
    if args.skip_existing and weights.exists() and (run_dir / "results.csv").exists():
        return training_summary(run_dir, 0.0, skipped=True)
    result = run_command(command, args.output_dir / "logs" / f"train_{experiment.name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {experiment.name}. See {args.output_dir / 'logs' / f'train_{experiment.name}.log'}")
    if not weights.exists():
        raise RuntimeError(f"Training completed but best weights were not found: {weights}")
    return training_summary(run_dir, float(getattr(result, "elapsed_seconds", 0.0)), skipped=False)


def training_summary(run_dir: Path, elapsed_seconds: float, skipped: bool = False) -> dict[str, Any]:
    rows = read_csv(run_dir / "results.csv")
    epochs_completed = len(rows)
    return {
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
        "epochs_completed": epochs_completed,
        "wall_time_seconds": elapsed_seconds,
        "average_epoch_seconds": elapsed_seconds / epochs_completed if elapsed_seconds and epochs_completed else None,
        "skipped_existing": skipped,
    }


def evaluate_experiment(args: argparse.Namespace, experiment: AugExperiment, weights: Path) -> dict[str, Any]:
    eval_dir = args.output_dir / "evaluations" / experiment.name
    threshold_dir = eval_dir / "val_threshold"
    threshold_command = [
        sys.executable,
        "scripts_YOLO/evaluate_yolo_pose_f1.py",
        "--weights",
        str(weights),
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
        str(args.prediction_conf_min),
        "--thresholds",
        args.thresholds,
        "--output-dir",
        str(threshold_dir),
        "--write-predictions",
    ]
    result = run_command(threshold_command, args.output_dir / "logs" / f"threshold_val_{experiment.name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"Val threshold sweep failed for {experiment.name}. See logs/threshold_val_{experiment.name}.log")
    best_val = read_json(threshold_dir / "val_best_threshold.json")

    split_metrics = {}
    for split in ["val", "test"]:
        metrics_dir = eval_dir / split
        metrics_command = [
            sys.executable,
            "scripts_YOLO/evaluate_yolo_pose_metrics.py",
            "--weights",
            str(weights),
            "--data",
            str(args.data),
            "--data-root",
            str(args.data_root),
            "--split",
            split,
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--iou-threshold",
            str(args.iou_threshold),
            "--keypoint-error-threshold",
            str(args.keypoint_error_threshold),
            "--conf-threshold",
            str(best_val["threshold"]),
            "--prediction-conf-min",
            str(args.prediction_conf_min),
            "--output-dir",
            str(metrics_dir),
        ]
        if args.save_error_images:
            metrics_command.extend(["--save-error-images", "--max-error-images", str(args.max_error_images)])
        result = run_command(metrics_command, args.output_dir / "logs" / f"metrics_{split}_{experiment.name}.log")
        if result.returncode != 0:
            raise RuntimeError(f"{split} metrics failed for {experiment.name}. See logs/metrics_{split}_{experiment.name}.log")
        split_metrics[split] = read_json(metrics_dir / f"{split}_metrics_summary.json")

    raw_recall = {}
    for split in ["val", "test"]:
        raw_dir = eval_dir / f"{split}_raw_recall"
        raw_command = [
            sys.executable,
            "scripts_YOLO/evaluate_yolo_pose_f1.py",
            "--weights",
            str(weights),
            "--data-root",
            str(args.data_root),
            "--split",
            split,
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--iou-threshold",
            str(args.iou_threshold),
            "--prediction-conf-min",
            str(args.prediction_conf_min),
            "--fixed-threshold",
            str(args.prediction_conf_min),
            "--output-dir",
            str(raw_dir),
        ]
        result = run_command(raw_command, args.output_dir / "logs" / f"raw_recall_{split}_{experiment.name}.log")
        if result.returncode != 0:
            raise RuntimeError(f"{split} raw recall failed for {experiment.name}. See logs/raw_recall_{split}_{experiment.name}.log")
        raw_recall[split] = read_json(raw_dir / f"{split}_best_threshold.json")

    visualization_dir = eval_dir / "test_visualizations"
    visualize_command = [
        sys.executable,
        "scripts_YOLO/visualize_yolo_pose_predictions.py",
        "--weights",
        str(weights),
        "--data-root",
        str(args.data_root),
        "--split",
        "test",
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--conf-threshold",
        str(best_val["threshold"]),
        "--prediction-conf-min",
        str(args.prediction_conf_min),
        "--iou-threshold",
        str(args.iou_threshold),
        "--output-dir",
        str(visualization_dir),
        "--max-images",
        str(args.max_visualizations),
    ]
    result = run_command(visualize_command, args.output_dir / "logs" / f"visualize_test_{experiment.name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"Test visualization failed for {experiment.name}. See logs/visualize_test_{experiment.name}.log")

    combined = {
        "experiment": experiment.name,
        "purpose": experiment.purpose,
        "augmentation": experiment.augmentation,
        "weights": str(weights),
        "best_val_threshold": best_val,
        "metrics": split_metrics,
        "raw_recall": raw_recall,
        "visualization_dir": str(visualization_dir),
    }
    write_json(eval_dir / "combined_metrics.json", combined)
    return combined


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric_row(name: str, metrics: dict[str, Any], train_summary_row: dict[str, Any] | None) -> dict[str, Any]:
    test_metrics = metrics["metrics"]["test"]
    val_metrics = metrics["metrics"]["val"]
    return {
        "experiment": name,
        "epochs_completed": "" if train_summary_row is None else train_summary_row.get("epochs_completed", ""),
        "skipped_existing": "" if train_summary_row is None else train_summary_row.get("skipped_existing", ""),
        "val_best_threshold": nested_get(metrics, ["best_val_threshold", "threshold"]),
        "val_f1": nested_get(val_metrics, ["fixed_threshold_detection", "f1"]),
        "test_f1": nested_get(test_metrics, ["fixed_threshold_detection", "f1"]),
        "test_precision": nested_get(test_metrics, ["fixed_threshold_detection", "precision"]),
        "test_recall": nested_get(test_metrics, ["fixed_threshold_detection", "recall"]),
        "test_tp": nested_get(test_metrics, ["fixed_threshold_detection", "tp"]),
        "test_fp": nested_get(test_metrics, ["fixed_threshold_detection", "fp"]),
        "test_fn": nested_get(test_metrics, ["fixed_threshold_detection", "fn"]),
        "test_raw_recall_at_prediction_conf_min": nested_get(metrics, ["raw_recall", "test", "recall"]),
        "test_box_map50_95": nested_get(test_metrics, ["map", "box", "map50_95"]),
        "test_pose_map50_95": nested_get(test_metrics, ["map", "pose", "map50_95"]),
        "test_nme": nested_get(test_metrics, ["keypoints", "nme"]),
        "test_pck_0.05": nested_get(test_metrics, ["keypoints", "pck_0.05"]),
        "test_pck_0.10": nested_get(test_metrics, ["keypoints", "pck_0.10"]),
        "test_failures": nested_get(test_metrics, ["failure_categories", "total_failures"]),
    }


def canonical_tags_from_failure_row(row: dict[str, str]) -> set[str]:
    tokens = split_tag_text(row.get("tags", ""))
    return canonical_tags(tokens)


def tag_gt_counts_from_audit(audit_rows: list[dict[str, Any]], split: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in audit_rows:
        if row["split"] != split:
            continue
        object_count = int(row["objects"])
        if object_count <= 0:
            continue
        for tag in split_tag_text(str(row.get("tags", ""))):
            counter[tag] += object_count
    return counter


def tag_summary_for_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    audit_report: dict[str, Any],
) -> list[dict[str, Any]]:
    failure_path = args.output_dir / "evaluations" / experiment_name / "test" / "test_failure_analysis.csv"
    failures = read_csv(failure_path)
    structural_gt_counts = audit_report["report"].get("tag_instance_counts", {}).get("test", {})
    image_tags = {
        Path(row["image"]).name: set(split_tag_text(str(row.get("tags", ""))))
        for row in audit_report.get("image_rows", [])
        if row.get("split") == "test"
    }
    rows = []
    for tag in TAG_TARGETS:
        gt_total = int(structural_gt_counts.get(tag, 0) or 0)
        fn = 0
        for row in failures:
            if row.get("failure_type") not in FN_FAILURE_TYPES:
                continue
            row_tags = canonical_tags_from_failure_row(row)
            row_tags.update(image_tags.get(Path(row.get("image", "")).name, set()))
            if tag in row_tags:
                fn += 1
        recall = (gt_total - fn) / gt_total if gt_total else None
        rows.append(
            {
                "experiment": experiment_name,
                "tag": tag,
                "test_gt_with_tag": gt_total,
                "test_fn_with_tag": fn,
                "test_recall_with_tag": recall,
            }
        )
    return rows


def adoption_rows(comparison_rows: list[dict[str, Any]], tag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = next((row for row in comparison_rows if row["experiment"] == "E0_current_control"), None)
    if control is None:
        return []
    control_tags = {
        row["tag"]: row for row in tag_rows if row["experiment"] == "E0_current_control"
    }
    rows = []
    for row in comparison_rows:
        if row["experiment"] == "E0_current_control":
            continue
        f1_delta = nullable_delta(row.get("test_f1"), control.get("test_f1"))
        nme_delta = nullable_delta(row.get("test_nme"), control.get("test_nme"))
        pck_delta = nullable_delta(row.get("test_pck_0.05"), control.get("test_pck_0.05"))
        important_tag_fn_improved = False
        important_tag_regressed = False
        for tag_row in [item for item in tag_rows if item["experiment"] == row["experiment"]]:
            base = control_tags.get(tag_row["tag"])
            if not base:
                continue
            fn_delta = int(tag_row["test_fn_with_tag"]) - int(base["test_fn_with_tag"])
            if fn_delta <= -2:
                important_tag_fn_improved = True
            if fn_delta >= 3:
                important_tag_regressed = True
        adopted = (
            ((f1_delta is not None and f1_delta >= 0.02) or important_tag_fn_improved)
            and (nme_delta is None or nme_delta <= 0.005)
            and (pck_delta is None or pck_delta >= -0.03)
            and not important_tag_regressed
        )
        rows.append(
            {
                "experiment": row["experiment"],
                "test_f1_delta": f1_delta,
                "test_nme_delta": nme_delta,
                "test_pck_0.05_delta": pck_delta,
                "important_tag_fn_improved": important_tag_fn_improved,
                "important_tag_regressed": important_tag_regressed,
                "adoption_candidate": adopted,
            }
        )
    return rows


def nullable_delta(value: Any, baseline: Any) -> float | None:
    current = safe_float(value)
    base = safe_float(baseline)
    if current is None or base is None:
        return None
    return current - base


def write_report(
    path: Path,
    audit: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    adoption: list[dict[str, Any]],
) -> None:
    lines = [
        "# YOLO11s Pose Augmentation Phase 1 Report",
        "",
        "## Dataset Audit",
        "",
        f"- data root: `{audit['report']['data_root']}`",
        f"- status: `{audit['report']['status']}`",
        f"- images by split: `{audit['report']['images_by_split']}`",
        f"- instances by split: `{audit['report']['instances_by_split']}`",
        f"- split duplicate image groups: `{len(audit['report']['split_duplicate_images'])}`",
        f"- near-frame cross-split pairs: `{audit['report']['near_frame_cross_split_pair_count']}`",
        f"- flip_idx ok: `{audit['report']['flip_idx_ok']}`",
        "",
        "## Comparison",
        "",
    ]
    if comparison_rows:
        headers = ["experiment", "val_best_threshold", "test_f1", "test_recall", "test_fp", "test_fn", "test_nme", "test_pck_0.05"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in comparison_rows:
            lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
        lines.append("")
    lines.extend(["## Adoption Check", ""])
    if adoption:
        headers = ["experiment", "test_f1_delta", "test_nme_delta", "test_pck_0.05_delta", "adoption_candidate"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in adoption:
            lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
        lines.append("")
    lines.extend(
        [
            "## Phase 2 Direction",
            "",
            "- If occlusion remains dominant, test offline occlusion separately with keypoint visibility rules.",
            "- If pixelated/blur remains dominant, test pixelation/blur/JPEG degradation separately.",
            "- If angle remains dominant, prioritize real profile/top-view data or hard mining over 2D rotation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.data = (args.data or args.data_root / "face_pose.yaml").resolve()
    args.output_dir = args.output_dir.resolve()
    args.project = (args.project or args.output_dir / "runs").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    manual_tag_index = load_manual_tag_index(args.manual_tags_csv)
    audit = dataset_audit(
        args.data_root,
        args.data,
        manual_tag_index,
        args.near_frame_window,
        args.near_time_window,
    )
    write_json(args.output_dir / "phase0_dataset_audit.json", audit["report"])
    write_csv(args.output_dir / "phase0_dataset_images.csv", audit["image_rows"])
    if args.audit_only:
        print(json.dumps({"audit": audit["report"]}, ensure_ascii=False, indent=2))
        return 0

    train_summaries: dict[str, dict[str, Any]] = {}
    metrics_by_experiment: dict[str, dict[str, Any]] = {}
    for name in args.experiments:
        experiment = EXPERIMENTS[name]
        run_dir = args.project / name
        if not args.eval_only:
            train_summaries[name] = train_experiment(args, experiment, run_dir)
        else:
            train_summaries[name] = training_summary(run_dir, 0.0, skipped=True)
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            raise RuntimeError(f"Missing weights for evaluation: {weights}")
        metrics_by_experiment[name] = evaluate_experiment(args, experiment, weights)

    comparison = [
        metric_row(name, metrics_by_experiment[name], train_summaries.get(name))
        for name in args.experiments
    ]
    tag_rows = []
    for name in args.experiments:
        tag_rows.extend(tag_summary_for_experiment(args, name, audit))
    adoption = adoption_rows(comparison, tag_rows)

    write_csv(args.output_dir / "phase1_model_comparison.csv", comparison)
    write_json(args.output_dir / "phase1_model_comparison.json", comparison)
    write_csv(args.output_dir / "phase1_tag_fn_recall.csv", tag_rows)
    write_csv(args.output_dir / "phase1_adoption_check.csv", adoption)
    write_report(args.output_dir / "report.md", audit, comparison, adoption)
    print(json.dumps({"comparison": comparison, "adoption": adoption}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
