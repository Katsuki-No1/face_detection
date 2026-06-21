import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
CSV_COLUMNS = [
    "image_path",
    "label_path",
    "split",
    "gt_count",
    "pred_count",
    "matched_count",
    "fp_count",
    "fn_count",
    "mean_iou",
    "max_iou",
    "mean_nme",
    "max_nme",
    "pck005_pass",
    "auto_failure_type",
    "manual_failure_type",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1-3 artifacts for YOLO11s face pose improvement experiments."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--old-data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument(
        "--modified-data-root",
        type=Path,
        default=Path("outputs_experiment/yolo_pose_dataset_ablation/datasets/modified_100"),
        help="Converted all-current dataset with old val/test held fixed.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output_experiment"))
    parser.add_argument("--baseline-weights", type=Path, default=Path("runs/face_pose_dataset_ablation/yolo11s_old_repro/weights/best.pt"))
    parser.add_argument("--baseline-eval-root", type=Path, default=Path("output_experiment/00_repo_audit/baseline_eval_raw"))
    parser.add_argument("--fixed-conf-threshold", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-hard-ratio", type=float, default=0.4)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def image_files(root: Path, split: str) -> list[Path]:
    image_root = root / "images" / split
    if not image_root.exists():
        return []
    return sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_for_image(root: Path, split: str, image_path: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_label_instances(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_label(label_path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if not label_path.exists():
        return objects
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        expected_columns = 5 + len(KEYPOINT_NAMES) * 3
        if len(values) != expected_columns:
            raise ValueError(f"Invalid label column count at {label_path}:{line_number}: {len(values)} != {expected_columns}")
        keypoints = []
        raw = values[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            keypoints.append(
                {
                    "name": name,
                    "x": raw[index * 3],
                    "y": raw[index * 3 + 1],
                    "visibility": int(raw[index * 3 + 2]),
                }
            )
        objects.append(
            {
                "class_id": int(values[0]),
                "xc": values[1],
                "yc": values[2],
                "width": values[3],
                "height": values[4],
                "keypoints": keypoints,
            }
        )
    return objects


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for image-size checks and failure overlays.") from error
    return cv2


def image_size(cv2: Any, image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def label_tags(objects: list[dict[str, Any]], width: int, height: int) -> Counter[str]:
    tags: Counter[str] = Counter()
    for obj in objects:
        area_ratio = obj["width"] * obj["height"]
        if area_ratio < 0.01:
            tags["small"] += 1
        elif area_ratio < 0.05:
            tags["medium"] += 1
        else:
            tags["large"] += 1
        x1 = (obj["xc"] - obj["width"] / 2.0) * width
        y1 = (obj["yc"] - obj["height"] / 2.0) * height
        x2 = (obj["xc"] + obj["width"] / 2.0) * width
        y2 = (obj["yc"] + obj["height"] / 2.0) * height
        if x1 <= width * 0.02 or y1 <= height * 0.02 or x2 >= width * 0.98 or y2 >= height * 0.98:
            tags["edge"] += 1
        visible = sum(1 for point in obj["keypoints"] if point["visibility"] > 0)
        if visible < len(KEYPOINT_NAMES):
            tags["missing_keypoints"] += 1
        if visible <= 2:
            tags["sparse_keypoints"] += 1
    return tags


def hard_types_from_tags(tags: Counter[str], include_negative: bool = False) -> list[str]:
    hard_types = []
    if tags.get("small"):
        hard_types.append("small")
    if tags.get("edge"):
        hard_types.append("edge")
    if tags.get("missing_keypoints") or tags.get("sparse_keypoints"):
        hard_types.append("occluded_candidate")
    if include_negative and tags.get("negative"):
        hard_types.append("negative")
    return hard_types


def write_dataset_yaml(root: Path) -> None:
    config = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "face"},
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "keypoints": KEYPOINT_NAMES,
    }
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    (root / "data.yaml").write_text(text, encoding="utf-8")
    (root / "face_pose.yaml").write_text(text, encoding="utf-8")


def make_split_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def safe_remove_generated(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    allowed = (output_root / "02_datasets").resolve()
    if not str(resolved).startswith(str(allowed)):
        raise ValueError(f"Refusing to remove path outside generated datasets: {path}")
    if path.exists():
        shutil.rmtree(path)


def copy_sample(source_root: Path, source_split: str, destination_root: Path, destination_split: str, image_path: Path) -> bool:
    destination_image = destination_root / "images" / destination_split / image_path.name
    destination_label = destination_root / "labels" / destination_split / f"{image_path.stem}.txt"
    if destination_image.exists() or destination_label.exists():
        return False
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_image)
    source_label = label_for_image(source_root, source_split, image_path)
    if source_label.exists():
        shutil.copy2(source_label, destination_label)
    else:
        destination_label.write_text("", encoding="utf-8")
    return True


def copy_split(source_root: Path, source_split: str, destination_root: Path, destination_split: str) -> int:
    copied = 0
    for image_path in image_files(source_root, source_split):
        copied += int(copy_sample(source_root, source_split, destination_root, destination_split, image_path))
    return copied


def build_eval_hashes(old_root: Path) -> tuple[set[str], set[str], set[str]]:
    eval_names = {path.name for split in ["val", "test"] for path in image_files(old_root, split)}
    eval_stems = {Path(name).stem for name in eval_names}
    eval_hashes = {sha256_file(path) for split in ["val", "test"] for path in image_files(old_root, split)}
    return eval_names, eval_stems, eval_hashes


def train_candidate_rows(cv2: Any, modified_root: Path, old_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_names, eval_stems, eval_hashes = build_eval_hashes(old_root)
    old_train_names = {path.name for path in image_files(old_root, "train")}
    old_train_hashes = {sha256_file(path) for path in image_files(old_root, "train")}
    hard_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []

    for image_path in image_files(modified_root, "train"):
        file_hash = sha256_file(image_path)
        if image_path.name in eval_names or image_path.stem in eval_stems or file_hash in eval_hashes:
            continue
        if image_path.name in old_train_names or file_hash in old_train_hashes:
            continue
        label_path = label_for_image(modified_root, "train", image_path)
        objects = parse_label(label_path)
        width, height = image_size(cv2, image_path)
        tags = label_tags(objects, width, height)
        if not objects:
            tags["negative"] += 1
            negative_rows.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "source_split": "train",
                    "auto_failure_type": "negative",
                    "instances": 0,
                    "train_eligible": True,
                    "notes": "empty label candidate for FP control",
                }
            )
            continue
        hard_types = hard_types_from_tags(tags)
        if hard_types:
            hard_rows.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "source_split": "train",
                    "auto_failure_type": "|".join(hard_types),
                    "instances": len(objects),
                    "small_instances": tags.get("small", 0),
                    "edge_instances": tags.get("edge", 0),
                    "missing_keypoint_instances": tags.get("missing_keypoints", 0),
                    "train_eligible": True,
                    "notes": "selected by train-side structural hard proxy; fixed val/test excluded",
                }
            )
    return hard_rows, negative_rows


def read_failure_inputs(eval_root: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    data: dict[str, dict[str, list[dict[str, str]]]] = {}
    for split in ["val", "test"]:
        split_root = eval_root / split
        data[split] = {
            "failures": read_csv(split_root / f"{split}_failure_analysis.csv"),
            "matches": read_csv(split_root / f"{split}_matches.csv"),
            "keypoints": read_csv(split_root / f"{split}_keypoint_errors.csv"),
        }
    return data


def split_from_image_path(path_text: str) -> str:
    parts = Path(path_text).parts
    for split in ["train", "val", "test"]:
        if split in parts:
            return split
    return ""


def label_path_for_eval_image(old_root: Path, image_text: str) -> Path:
    image_path = Path(image_text)
    split = split_from_image_path(image_text)
    if split:
        return old_root / "labels" / split / f"{image_path.stem}.txt"
    return old_root / "labels" / "test" / f"{image_path.stem}.txt"


def auto_failure_type_for_image(failure_rows: list[dict[str, str]], match_rows: list[dict[str, str]]) -> str:
    types: set[str] = set()
    tags_text = "|".join(row.get("tags", "") for row in failure_rows + match_rows)
    if "small_face" in tags_text:
        types.add("small")
    if "edge_face" in tags_text:
        types.add("edge")
    if "false_positive" in [row.get("failure_type", "") for row in failure_rows]:
        types.add("fp")
    if any(row.get("failure_type", "") in {"missed_detection", "localization_error"} for row in failure_rows):
        types.add("fn")
    if any(row.get("failure_type", "") == "keypoint_error" for row in failure_rows):
        types.add("keypoint_ng")
    if "missing_gt_keypoints" in tags_text or "very_sparse_keypoints" in tags_text:
        types.add("occluded_candidate")
    return "|".join(sorted(types)) if types else "review"


def aggregate_failures(old_root: Path, eval_data: dict[str, dict[str, list[dict[str, str]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_data in eval_data.items():
        failures_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
        matches_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
        keypoints_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in split_data["failures"]:
            failures_by_image[row["image"]].append(row)
        for row in split_data["matches"]:
            matches_by_image[row["image"]].append(row)
        for row in split_data["keypoints"]:
            keypoints_by_image[row["image"]].append(row)

        for image_text in sorted(failures_by_image):
            label_path = label_path_for_eval_image(old_root, image_text)
            match_rows = matches_by_image.get(image_text, [])
            failure_rows = failures_by_image[image_text]
            ious = [value for value in (float_or_none(row.get("iou")) for row in match_rows + failure_rows) if value is not None]
            match_nmes = [value for value in (float_or_none(row.get("mean_keypoint_error")) for row in match_rows) if value is not None]
            failure_nmes = [value for value in (float_or_none(row.get("mean_keypoint_error")) for row in failure_rows) if value is not None]
            nmes = match_nmes + failure_nmes
            fp_count = sum(1 for row in failure_rows if row.get("failure_type") == "false_positive")
            fn_count = sum(1 for row in failure_rows if row.get("failure_type") in {"missed_detection", "localization_error"})
            matched_count = len(match_rows)
            gt_count = count_label_instances(label_path)
            pred_count = matched_count + fp_count
            pck005_pass = bool(nmes) and max(nmes) <= 0.05
            auto_failure_type = auto_failure_type_for_image(failure_rows, match_rows)
            rows.append(
                {
                    "image_path": image_text,
                    "label_path": str(label_path),
                    "split": split,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "matched_count": matched_count,
                    "fp_count": fp_count,
                    "fn_count": fn_count,
                    "mean_iou": sum(ious) / len(ious) if ious else "",
                    "max_iou": max(ious) if ious else "",
                    "mean_nme": sum(nmes) / len(nmes) if nmes else "",
                    "max_nme": max(nmes) if nmes else "",
                    "pck005_pass": pck005_pass,
                    "auto_failure_type": auto_failure_type,
                    "manual_failure_type": "",
                    "notes": "manual review needed for angle/mosaic/occluded/blur_dark",
                }
            )
    return rows


def copy_failure_visualizations(cv2: Any, output_root: Path, eval_root: Path, failure_rows: list[dict[str, Any]]) -> None:
    visualization_root = output_root / "01_failure_mining" / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for row in failure_rows:
        image_path = Path(str(row["image_path"]))
        source = eval_root / row["split"] / "error_images" / image_path.name
        if not source.exists():
            source = image_path
        image = cv2.imread(str(source))
        if image is None:
            continue
        overlay = [
            f'{row["split"]} {image_path.name}',
            f'type={row["auto_failure_type"]} fp={row["fp_count"]} fn={row["fn_count"]}',
            f'iou={row["mean_iou"]} nme={row["mean_nme"]} pck005={row["pck005_pass"]}',
        ]
        y = 22
        for text in overlay:
            cv2.putText(image, text[:180], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text[:180], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        output_path = visualization_root / f'{row["split"]}_{image_path.name}'
        cv2.imwrite(str(output_path), image)
        index_rows.append(
            {
                "image_path": row["image_path"],
                "visualization_path": str(output_path),
                "auto_failure_type": row["auto_failure_type"],
            }
        )
    write_csv(visualization_root / "visualization_index.csv", index_rows)


def failure_summary_rows(failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in failure_rows:
        for failure_type in str(row["auto_failure_type"]).split("|"):
            if failure_type:
                counter[failure_type] += 1
    return [{"failure_type": key, "images": value} for key, value in sorted(counter.items())]


def baseline_csv_rows(metrics_by_split: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in metrics_by_split.items():
        fixed = metrics["fixed_threshold_detection"]
        keypoints = metrics["keypoints"]
        rows.append(
            {
                "split": split,
                "images": metrics["images"],
                "gt_objects": metrics["ground_truth_objects"],
                "pred_objects": metrics["predicted_objects"],
                "box_map50_95": metrics["map"]["box"]["map50_95"],
                "box_precision": metrics["map"]["box"]["precision"],
                "box_recall": metrics["map"]["box"]["recall"],
                "pose_map50_95": metrics["map"]["pose"]["map50_95"],
                "fixed_f1": fixed["f1"],
                "fixed_conf_threshold": fixed["threshold"],
                "fixed_tp": fixed["tp"],
                "fixed_fp": fixed["fp"],
                "fixed_fn": fixed["fn"],
                "keypoint_nme": keypoints["nme"],
                "pck_005": keypoints["pck_0.05"],
                "failures": metrics["failure_categories"]["total_failures"],
            }
        )
    return rows


def get_git_status(repo_root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=repo_root, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def dataset_structure_report(old_root: Path, modified_root: Path, output_path: Path) -> None:
    config = yaml.safe_load((old_root / "face_pose.yaml").read_text(encoding="utf-8"))
    lines = [
        "# Dataset Structure",
        "",
        "## Fixed Evaluation Dataset",
        "",
        f"- root: `{old_root}`",
        f"- yaml: `{old_root / 'face_pose.yaml'}`",
        f"- kpt_shape: `{config.get('kpt_shape')}`",
        f"- keypoints: `{config.get('keypoints')}`",
        "",
        "| split | images | labels | instances | negatives |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        images = image_files(old_root, split)
        labels = list((old_root / "labels" / split).glob("*.txt")) if (old_root / "labels" / split).exists() else []
        instances = sum(count_label_instances(label) for label in labels)
        negatives = sum(1 for label in labels if count_label_instances(label) == 0)
        lines.append(f"| {split} | {len(images)} | {len(labels)} | {instances} | {negatives} |")
    lines.extend(
        [
            "",
            "## Train Candidate Source",
            "",
            f"- root: `{modified_root}`",
            "- fixed val/test image names and hashes are excluded from train additions.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repo_audit_report(repo_root: Path, output_path: Path, baseline_rows: list[dict[str, Any]]) -> None:
    git_status = get_git_status(repo_root)
    metric_checks = [
        ("Box mAP50-95", "present: evaluate_yolo_pose_metrics.py model.val box.map"),
        ("Box precision / recall", "present: evaluate_yolo_pose_metrics.py model.val box.mp/mr"),
        ("Pose mAP50-95", "present: evaluate_yolo_pose_metrics.py model.val pose.map"),
        ("fixed threshold F1", "present: fixed_threshold_detection"),
        ("keypoint NME", "present: keypoints.nme"),
        ("PCK@0.05", "present: keypoints.pck_0.05"),
        ("failures", "present: *_failure_analysis.csv and failure_categories"),
    ]
    lines = [
        "# Repo Audit",
        "",
        f"- repo root: `{repo_root}`",
        "- primary model: `yolo11s-pose.pt` / YOLO pose",
        "- fixed eval set: `datasets/face_pose` val=100, test=30",
        "- baseline weights: `runs/face_pose_dataset_ablation/yolo11s_old_repro/weights/best.pt`",
        "",
        "## Existing Scripts",
        "",
        "- `scripts_YOLO/train_yolo_pose.py`: YOLO pose training wrapper",
        "- `scripts_YOLO/evaluate_yolo_pose_metrics.py`: mAP, fixed F1, NME/PCK, failures",
        "- `scripts_YOLO/evaluate_yolo_pose_f1.py`: confidence threshold sweep",
        "- `scripts_YOLO/check_yolo_pose_dataset.py`: label/yaml consistency check",
        "- `scripts_YOLO/run_yolo_pose_dataset_ablation.py`: prior fixed-eval old/modified ablation runner",
        "",
        "## Metric Coverage",
        "",
        "| metric | status |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in metric_checks)
    lines.extend(["", "## Baseline Snapshot", "", "| split | box mAP50-95 | pose mAP50-95 | fixed F1 | NME | PCK@0.05 | failures |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in baseline_rows:
        lines.append(
            f"| {row['split']} | {float(row['box_map50_95']):.4f} | {float(row['pose_map50_95']):.4f} | "
            f"{float(row['fixed_f1']):.4f} | {float(row['keypoint_nme']):.4f} | {float(row['pck_005']):.4f} | {row['failures']} |"
        )
    lines.extend(["", "## Git Status At Audit Time", "", "```", git_status or "clean", "```", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataset_image_summary(cv2: Any, dataset_root: Path, split: str) -> dict[str, Any]:
    images = image_files(dataset_root, split)
    positive_images = 0
    negative_images = 0
    instances = 0
    visibility: Counter[str] = Counter()
    size_distribution: Counter[str] = Counter()
    edge_instances = 0
    label_failures: list[dict[str, Any]] = []

    for image_path in images:
        label_path = label_for_image(dataset_root, split, image_path)
        try:
            objects = parse_label(label_path)
        except Exception as error:
            label_failures.append({"image": str(image_path), "label": str(label_path), "error": str(error)})
            continue
        if objects:
            positive_images += 1
        else:
            negative_images += 1
        width, height = image_size(cv2, image_path)
        tags = label_tags(objects, width, height)
        instances += len(objects)
        edge_instances += tags.get("edge", 0)
        for obj in objects:
            area_ratio = obj["width"] * obj["height"]
            if area_ratio < 0.01:
                size_distribution["small"] += 1
            elif area_ratio < 0.05:
                size_distribution["medium"] += 1
            else:
                size_distribution["large"] += 1
            for point in obj["keypoints"]:
                visibility[str(point["visibility"])] += 1

    return {
        "images": len(images),
        "labels": len(list((dataset_root / "labels" / split).glob("*.txt"))),
        "positive_images": positive_images,
        "negative_images": negative_images,
        "instances": instances,
        "visibility_counts": dict(sorted(visibility.items())),
        "size_distribution": dict(sorted(size_distribution.items())),
        "edge_instances": edge_instances,
        "label_failures": label_failures[:200],
    }


def duplicate_report(dataset_root: Path) -> dict[str, Any]:
    by_name: defaultdict[str, list[str]] = defaultdict(list)
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        for image_path in image_files(dataset_root, split):
            rel = str(image_path.relative_to(dataset_root))
            by_name[image_path.name].append(rel)
            by_hash[sha256_file(image_path)].append(rel)
    duplicate_names = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    duplicate_hashes = {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}
    train_eval_hash_collisions = []
    train_hashes = {sha256_file(path): str(path.relative_to(dataset_root)) for path in image_files(dataset_root, "train")}
    for split in ["val", "test"]:
        for path in image_files(dataset_root, split):
            digest = sha256_file(path)
            if digest in train_hashes:
                train_eval_hash_collisions.append(
                    {"train": train_hashes[digest], "eval": str(path.relative_to(dataset_root)), "split": split}
                )
    return {
        "duplicate_name_count": len(duplicate_names),
        "duplicate_hash_count": len(duplicate_hashes),
        "train_eval_hash_collision_count": len(train_eval_hash_collisions),
        "duplicate_names": duplicate_names,
        "duplicate_hashes": duplicate_hashes,
        "train_eval_hash_collisions": train_eval_hash_collisions,
    }


def summarize_dataset(
    cv2: Any,
    dataset_root: Path,
    dataset_name: str,
    hard_rows: list[dict[str, Any]],
    failure_type_counts: Counter[str],
) -> dict[str, Any]:
    split_summary = {split: dataset_image_summary(cv2, dataset_root, split) for split in ["train", "val", "test"]}
    duplicates = duplicate_report(dataset_root)
    summary = {
        "dataset_name": dataset_name,
        "root": str(dataset_root),
        "data_yaml": str(dataset_root / "data.yaml"),
        "splits": split_summary,
        "hard_example_count": len(hard_rows),
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "duplicate_check": duplicates,
    }
    write_json(dataset_root / "summary.json", summary)
    lines = [
        f"# {dataset_name}",
        "",
        f"- root: `{dataset_root}`",
        f"- data yaml: `{dataset_root / 'data.yaml'}`",
        f"- hard examples: {len(hard_rows)}",
        f"- train/eval hash collisions: {duplicates['train_eval_hash_collision_count']}",
        "",
        "| split | images | positives | negatives | instances | edge boxes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in split_summary.items():
        lines.append(
            f"| {split} | {values['images']} | {values['positive_images']} | {values['negative_images']} | "
            f"{values['instances']} | {values['edge_instances']} |"
        )
    (dataset_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_dataset(
    output_root: Path,
    dataset_name: str,
    old_root: Path,
    source_rows: list[dict[str, Any]],
    modified_root: Path,
    overwrite: bool,
) -> tuple[Path, list[dict[str, Any]], Counter[str]]:
    dataset_root = output_root / "02_datasets" / dataset_name
    if overwrite:
        safe_remove_generated(dataset_root, output_root)
    if dataset_root.exists():
        raise FileExistsError(f"{dataset_root} already exists. Use --overwrite to replace generated datasets.")
    make_split_dirs(dataset_root)
    copied_hard_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()

    copy_split(old_root, "train", dataset_root, "train")
    copy_split(old_root, "val", dataset_root, "val")
    copy_split(old_root, "test", dataset_root, "test")

    for row in source_rows:
        image_path = Path(str(row["image_path"]))
        if copy_sample(modified_root, "train", dataset_root, "train", image_path):
            copied_hard_rows.append(row)
            for failure_type in str(row.get("auto_failure_type", "")).split("|"):
                if failure_type:
                    failure_counts[failure_type] += 1
    write_dataset_yaml(dataset_root)
    write_csv(dataset_root / "metadata" / "added_examples.csv", copied_hard_rows)
    return dataset_root, copied_hard_rows, failure_counts


def select_balanced_hard(rows: list[dict[str, Any]], normal_count: int, hard_ratio: float, seed: int) -> list[dict[str, Any]]:
    desired = int(round(normal_count * hard_ratio / max(1e-9, 1.0 - hard_ratio)))
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(desired, len(shuffled))]


def select_negative(rows: list[dict[str, Any]], normal_count: int, negative_ratio: float, seed: int) -> list[dict[str, Any]]:
    desired = int(round(normal_count * negative_ratio))
    rng = random.Random(seed + 17)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(desired, len(shuffled))]


def build_datasets(
    cv2: Any,
    output_root: Path,
    old_root: Path,
    modified_root: Path,
    hard_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    normal_count = len(image_files(old_root, "train"))
    variants = []
    selections = {
        "exp_yolo11s_base_repro": [],
        "exp_yolo11s_hard_mining_v1": hard_rows,
        "exp_yolo11s_hard_mining_balanced_v1": select_balanced_hard(hard_rows, normal_count, args.balanced_hard_ratio, args.seed),
        "exp_yolo11s_hard_aug_v1": select_balanced_hard(hard_rows, normal_count, args.balanced_hard_ratio, args.seed),
    }
    total_fp = sum(int(baseline_metrics[split]["fixed_threshold_detection"]["fp"]) for split in baseline_metrics)
    if total_fp > 0:
        selections["exp_yolo11s_negative_v1"] = select_negative(negative_rows, normal_count, args.negative_ratio, args.seed)
    else:
        selections["exp_yolo11s_negative_v1"] = []

    for name, rows in selections.items():
        dataset_root, copied_rows, failure_counts = build_dataset(output_root, name, old_root, rows, modified_root, args.overwrite)
        if name == "exp_yolo11s_hard_aug_v1":
            (dataset_root / "metadata" / "augmentation_intent.yaml").write_text(
                yaml.safe_dump(
                    {
                        "purpose": "Phase 5 augmentation target dataset; train/val/test split equals hard_mining_balanced_v1.",
                        "recommended_aug": "keypoint_safe or mosaic_like_degradation",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        summary = summarize_dataset(cv2, dataset_root, name, copied_rows, failure_counts)
        train = summary["splits"]["train"]
        variants.append(
            {
                "dataset_name": name,
                "data_yaml": str(dataset_root / "data.yaml"),
                "train_images": train["images"],
                "val_images": summary["splits"]["val"]["images"],
                "test_images": summary["splits"]["test"]["images"],
                "train_positive_images": train["positive_images"],
                "train_negative_images": train["negative_images"],
                "train_instances": train["instances"],
                "hard_examples_added": len(copied_rows),
                "failure_type_counts": json.dumps(dict(failure_counts), ensure_ascii=False),
                "train_eval_hash_collisions": summary["duplicate_check"]["train_eval_hash_collision_count"],
            }
        )
    write_csv(output_root / "02_datasets" / "dataset_compare.csv", variants)
    return variants


def write_command_scripts(output_root: Path, balanced_yaml: Path) -> None:
    scripts_dir = output_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_root / "03_yolo_imgsz"
    phase_script = scripts_dir / "run_phase123.sh"
    phase_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/katsuki/projects/face_detection",
                ".venv/bin/python scripts_YOLO/evaluate_yolo_pose_metrics.py --weights runs/face_pose_dataset_ablation/yolo11s_old_repro/weights/best.pt --data datasets/face_pose/face_pose.yaml --data-root datasets/face_pose --split val --imgsz 960 --device 0 --iou-threshold 0.5 --conf-threshold 0.51 --prediction-conf-min 0.001 --output-dir output_experiment/00_repo_audit/baseline_eval_raw/val --save-error-images",
                ".venv/bin/python scripts_YOLO/evaluate_yolo_pose_metrics.py --weights runs/face_pose_dataset_ablation/yolo11s_old_repro/weights/best.pt --data datasets/face_pose/face_pose.yaml --data-root datasets/face_pose --split test --imgsz 960 --device 0 --iou-threshold 0.5 --conf-threshold 0.51 --prediction-conf-min 0.001 --output-dir output_experiment/00_repo_audit/baseline_eval_raw/test --save-error-images",
                ".venv/bin/python scripts_YOLO/run_yolo11s_improvement_phase123.py --overwrite",
                "",
            ]
        ),
        encoding="utf-8",
    )
    full_script = scripts_dir / "run_yolo11s_hard_mining_balanced_imgsz960_full.sh"
    full_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/katsuki/projects/face_detection",
                ".venv/bin/python scripts_YOLO/train_yolo_pose.py \\",
                f"  --data {balanced_yaml} \\",
                "  --model yolo11s-pose.pt \\",
                "  --imgsz 960 \\",
                "  --epochs 200 \\",
                "  --batch auto \\",
                "  --patience 50 \\",
                f"  --project {project_dir.resolve()} \\",
                "  --name yolo11s_hard_mining_balanced_v1_imgsz960_full \\",
                "  --device 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for script in [phase_script, full_script]:
        script.chmod(0o755)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    old_root = resolve_repo_path(repo_root, args.old_data_root)
    modified_root = resolve_repo_path(repo_root, args.modified_data_root)
    output_root = resolve_repo_path(repo_root, args.output_root)
    eval_root = resolve_repo_path(repo_root, args.baseline_eval_root)
    cv2 = import_cv2()

    for required in [old_root, modified_root, eval_root / "val", eval_root / "test"]:
        if not required.exists():
            raise FileNotFoundError(required)

    audit_root = output_root / "00_repo_audit"
    failure_root = output_root / "01_failure_mining"
    datasets_root = output_root / "02_datasets"
    audit_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    metrics_by_split = {
        "val": read_json(eval_root / "val" / "val_metrics_summary.json"),
        "test": read_json(eval_root / "test" / "test_metrics_summary.json"),
    }
    baseline = {
        "schema_version": "1.0",
        "baseline": "old_repro",
        "weights": str(resolve_repo_path(repo_root, args.baseline_weights)),
        "fixed_conf_threshold": args.fixed_conf_threshold,
        "metrics": metrics_by_split,
    }
    write_json(audit_root / "baseline_eval.json", baseline)
    baseline_rows = baseline_csv_rows(metrics_by_split)
    write_csv(audit_root / "baseline_eval.csv", baseline_rows)
    repo_audit_report(repo_root, audit_root / "report.md", baseline_rows)
    dataset_structure_report(old_root, modified_root, audit_root / "dataset_structure.md")

    eval_data = read_failure_inputs(eval_root)
    failure_rows = aggregate_failures(old_root, eval_data)
    write_csv(failure_root / "failures.csv", failure_rows, CSV_COLUMNS)
    write_csv(failure_root / "failures_review.csv", failure_rows, CSV_COLUMNS)
    copy_failure_visualizations(cv2, failure_root.parent, eval_root, failure_rows)
    write_csv(failure_root / "summary_by_failure_type.csv", failure_summary_rows(failure_rows))

    hard_rows, negative_rows = train_candidate_rows(cv2, modified_root, old_root)
    hard_manifest_rows = hard_rows + negative_rows
    write_csv(failure_root / "hard_example_manifest.csv", hard_manifest_rows)

    dataset_rows = build_datasets(cv2, output_root, old_root, modified_root, hard_rows, negative_rows, metrics_by_split, args)
    balanced_yaml = output_root / "02_datasets" / "exp_yolo11s_hard_mining_balanced_v1" / "data.yaml"
    write_command_scripts(output_root, balanced_yaml)

    summary = {
        "baseline_rows": baseline_rows,
        "failure_images": len(failure_rows),
        "train_hard_candidates": len(hard_rows),
        "train_negative_candidates": len(negative_rows),
        "datasets": dataset_rows,
    }
    write_json(output_root / "02_datasets" / "phase123_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
