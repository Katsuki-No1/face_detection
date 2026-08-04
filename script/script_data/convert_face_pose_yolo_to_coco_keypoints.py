from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
SKELETON = ([1, 3], [2, 3], [3, 4], [3, 5])


class ConversionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert face_pose_drive-style YOLO Pose labels into COCO "
            "five-keypoint annotations for RTMPose."
        )
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def parse_label_line(
    line: str,
    *,
    path: Path,
    line_number: int,
    width: int,
    height: int,
) -> tuple[list[float], list[float], int, Counter[int]]:
    try:
        values = [float(value) for value in line.split()]
    except ValueError as error:
        raise ConversionError(f"{path}:{line_number}: non-numeric label") from error
    expected = 5 + len(KEYPOINT_NAMES) * 3
    if len(values) != expected:
        raise ConversionError(
            f"{path}:{line_number}: expected {expected} values, found {len(values)}"
        )
    if not values[0].is_integer() or int(values[0]) != 0:
        raise ConversionError(
            f"{path}:{line_number}: expected face class 0, found {values[0]}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ConversionError(f"{path}:{line_number}: non-finite value")

    cx, cy, box_width, box_height = values[1:5]
    x1 = max(0.0, min(float(width), (cx - box_width / 2.0) * width))
    y1 = max(0.0, min(float(height), (cy - box_height / 2.0) * height))
    x2 = max(0.0, min(float(width), (cx + box_width / 2.0) * width))
    y2 = max(0.0, min(float(height), (cy + box_height / 2.0) * height))
    bbox = [x1, y1, x2 - x1, y2 - y1]
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ConversionError(f"{path}:{line_number}: empty bbox after clipping")

    keypoints: list[float] = []
    visibility_counts: Counter[int] = Counter()
    num_keypoints = 0
    for index, _name in enumerate(KEYPOINT_NAMES):
        x_norm, y_norm, visibility_raw = values[5 + index * 3:8 + index * 3]
        if not visibility_raw.is_integer() or int(visibility_raw) not in {0, 1, 2}:
            raise ConversionError(
                f"{path}:{line_number}: invalid visibility {visibility_raw} "
                f"for keypoint {index}"
            )
        visibility = int(visibility_raw)
        visibility_counts[visibility] += 1
        if visibility == 0:
            keypoints.extend([0.0, 0.0, 0])
            continue
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise ConversionError(
                f"{path}:{line_number}: visible keypoint {index} is outside [0, 1]"
            )
        keypoints.extend([x_norm * width, y_norm * height, visibility])
        num_keypoints += 1
    return bbox, keypoints, num_keypoints, visibility_counts


def close_enough_bbox(left: list[float], right: list[float], tolerance: float = 0.05) -> bool:
    return len(left) == len(right) == 4 and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def convert_split(data_root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    detection = load_json(data_root / f"{split}.json")
    images = list(detection.get("images", []))
    detection_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in detection.get("annotations", []):
        detection_annotations[int(annotation["image_id"])].append(annotation)
    for rows in detection_annotations.values():
        rows.sort(key=lambda row: int(row["id"]))

    output_annotations: list[dict[str, Any]] = []
    visibility_counts: Counter[int] = Counter()
    annotation_id = 1
    negative_images = 0
    for image in images:
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        label_path = data_root / "labels" / split / f"{Path(image['file_name']).stem}.txt"
        if not label_path.is_file():
            raise ConversionError(f"Missing label file: {label_path}")
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_annotations = detection_annotations.get(image_id, [])
        if len(lines) != len(expected_annotations):
            raise ConversionError(
                f"{label_path}: {len(lines)} pose rows but detection COCO has "
                f"{len(expected_annotations)} annotations"
            )
        if not lines:
            negative_images += 1
        for row_index, (line, detection_annotation) in enumerate(
            zip(lines, expected_annotations), start=1
        ):
            bbox, keypoints, num_keypoints, row_visibility = parse_label_line(
                line,
                path=label_path,
                line_number=row_index,
                width=width,
                height=height,
            )
            if not close_enough_bbox(bbox, detection_annotation.get("bbox", [])):
                raise ConversionError(
                    f"{label_path}:{row_index}: bbox does not match detection COCO: "
                    f"pose={bbox}, detection={detection_annotation.get('bbox')}"
                )
            visibility_counts.update(row_visibility)
            output_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_keypoints,
                }
            )
            annotation_id += 1

    output = {
        "info": {
            "description": "face_pose_drive-style COCO five-keypoint dataset for RTMPose",
            "source_format": "YOLO Pose",
            "keypoint_order": list(KEYPOINT_NAMES),
            "flip_indices": [1, 0, 2, 4, 3],
        },
        "licenses": list(detection.get("licenses", [])),
        "images": images,
        "annotations": output_annotations,
        "categories": [
            {
                "id": 1,
                "name": "face",
                "supercategory": "person",
                "keypoints": list(KEYPOINT_NAMES),
                "skeleton": list(SKELETON),
            }
        ],
    }
    summary = {
        "images": len(images),
        "annotations": len(output_annotations),
        "negative_images": negative_images,
        "visibility_counts": {str(key): visibility_counts[key] for key in (0, 1, 2)},
    }
    return output, summary


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for split in SPLITS:
        converted, summary = convert_split(data_root, split)
        output_path = output_root / f"person_keypoints_{split}.json"
        output_path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[split] = summary

    overall = {
        "status": "ok",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "keypoint_names": list(KEYPOINT_NAMES),
        "flip_indices": [1, 0, 2, 4, 3],
        "splits": summaries,
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
