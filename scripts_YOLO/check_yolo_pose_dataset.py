import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a YOLO Pose dataset for face pose training.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--yaml", type=Path, default=Path("datasets/face_pose/face_pose.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs_exoeriment/yolo_pose_baseline_v001/pretrain_dataset_check.json"))
    return parser.parse_args()


def split_images(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_labels(data_root: Path, split: str) -> list[Path]:
    root = data_root / "labels" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() == ".txt")


def check_label(
    label_path: Path,
    expected_keypoints: int,
    class_count: int,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    objects = 0
    visibility_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    text = label_path.read_text(encoding="utf-8")
    if not text.strip():
        return objects, visibility_counts, failures
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        expected_columns = 5 + expected_keypoints * 3
        if len(parts) != expected_columns:
            failures.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "reason": "wrong_column_count",
                    "columns": len(parts),
                    "expected_columns": expected_columns,
                }
            )
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            failures.append({"file": str(label_path), "line": line_number, "reason": "non_numeric_value"})
            continue
        class_id = int(values[0])
        if class_id < 0 or class_id >= class_count or class_id != values[0]:
            failures.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "reason": "class_id_out_of_range",
                    "class_id": values[0],
                }
            )
        xc, yc, width, height = values[1:5]
        for name, value in [("xc", xc), ("yc", yc), ("width", width), ("height", height)]:
            if value < 0.0 or value > 1.0:
                failures.append(
                    {"file": str(label_path), "line": line_number, "reason": "bbox_value_out_of_range", "field": name, "value": value}
                )
        if width <= 0.0 or height <= 0.0:
            failures.append(
                {"file": str(label_path), "line": line_number, "reason": "bbox_non_positive", "width": width, "height": height}
            )
        keypoint_values = values[5:]
        for index in range(expected_keypoints):
            x_value = keypoint_values[index * 3]
            y_value = keypoint_values[index * 3 + 1]
            visibility = int(keypoint_values[index * 3 + 2])
            visibility_counts[str(visibility)] += 1
            if visibility not in {0, 1, 2}:
                failures.append(
                    {
                        "file": str(label_path),
                        "line": line_number,
                        "reason": "visibility_out_of_range",
                        "keypoint_index": index,
                        "visibility": visibility,
                    }
                )
            if visibility == 0:
                if x_value != 0.0 or y_value != 0.0:
                    failures.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "hidden_keypoint_has_coordinates",
                            "keypoint_index": index,
                            "x": x_value,
                            "y": y_value,
                        }
                    )
            elif x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0:
                failures.append(
                    {
                        "file": str(label_path),
                        "line": line_number,
                        "reason": "keypoint_value_out_of_range",
                        "keypoint_index": index,
                        "x": x_value,
                        "y": y_value,
                    }
                )
        objects += 1
    return objects, visibility_counts, failures


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    expected_keypoints = int(config["kpt_shape"][0])
    class_count = len(config["names"])
    splits = ["train", "val", "test"]
    failures: list[dict[str, Any]] = []
    images_by_split: dict[str, int] = {}
    labels_by_split: dict[str, int] = {}
    visibility_counts: Counter[str] = Counter()
    objects = 0
    empty_label_files = 0
    missing_labels = 0
    extra_labels = 0

    for split in splits:
        images = split_images(args.data_root, split)
        labels = split_labels(args.data_root, split)
        images_by_split[split] = len(images)
        labels_by_split[split] = len(labels)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            missing_labels += 1
            failures.append({"split": split, "stem": stem, "reason": "missing_label"})
        for stem in sorted(label_stems - image_stems):
            extra_labels += 1
            failures.append({"split": split, "stem": stem, "reason": "extra_label"})
        for label_path in labels:
            if not label_path.read_text(encoding="utf-8").strip():
                empty_label_files += 1
            object_count, split_visibility, split_failures = check_label(label_path, expected_keypoints, class_count)
            objects += object_count
            visibility_counts.update(split_visibility)
            failures.extend(split_failures)

    failure_counts = Counter(failure["reason"] for failure in failures)
    report = {
        "status": "pass" if not failures else "fail",
        "yaml_kpt_shape": config["kpt_shape"],
        "images": sum(images_by_split.values()),
        "labels": sum(labels_by_split.values()),
        "empty_label_files": empty_label_files,
        "objects": objects,
        "images_by_split": images_by_split,
        "labels_by_split": labels_by_split,
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "missing_labels": missing_labels,
        "extra_labels": extra_labels,
        "failure_counts": dict(sorted(failure_counts.items())),
        "failures": failures[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
