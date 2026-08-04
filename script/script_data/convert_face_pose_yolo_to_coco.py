from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.coco_split_audit import audit_coco_splits, sha256_file  # noqa: E402


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ConversionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a grouped YOLO Pose face dataset to COCO detection JSON."
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--category-id", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Could not read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return data


def metadata_index(path: Path) -> tuple[dict[str, dict[str, Any]], int, dict[str, Any]]:
    data = read_json(path)
    records = data.get("records", [])
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ConversionError(f"Invalid metadata record in {path}")
        keys: set[str] = set()
        for field in ("image_name", "output_image", "image_path"):
            value = record.get(field)
            if isinstance(value, str) and value:
                keys.update((value, Path(value).name, Path(value).stem))
        for key in keys:
            previous = index.get(key)
            if previous is not None and previous is not record:
                previous_identity = (previous.get("image_stem"), previous.get("image_name"))
                identity = (record.get("image_stem"), record.get("image_name"))
                if previous_identity != identity:
                    raise ConversionError(f"Ambiguous metadata key {key!r} in {path}")
            index[key] = record
    source_summary = data.get("summary", {})
    return index, len(records), source_summary if isinstance(source_summary, dict) else {}


def find_metadata(index: dict[str, dict[str, Any]], image_path: Path) -> dict[str, Any]:
    record = index.get(image_path.name) or index.get(image_path.stem)
    if record is None:
        raise ConversionError(f"No metadata record was found for {image_path}")
    return record


def parse_yolo_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    if not label_path.is_file():
        raise ConversionError(f"Missing YOLO label: {label_path}")
    boxes: list[list[float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise ConversionError(f"{label_path}:{line_number}: non-numeric YOLO label") from error
        if len(values) < 5:
            raise ConversionError(f"{label_path}:{line_number}: expected at least 5 values")
        class_id = values[0]
        if not class_id.is_integer() or int(class_id) != 0:
            raise ConversionError(
                f"{label_path}:{line_number}: expected YOLO face class 0, found {class_id}"
            )
        cx, cy, box_width, box_height = values[1:5]
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise ConversionError(f"{label_path}:{line_number}: bbox contains a non-finite value")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ConversionError(f"{label_path}:{line_number}: bbox has non-positive size")
        x1 = max(0.0, min(float(width), (cx - box_width / 2.0) * width))
        y1 = max(0.0, min(float(height), (cy - box_height / 2.0) * height))
        x2 = max(0.0, min(float(width), (cx + box_width / 2.0) * width))
        y2 = max(0.0, min(float(height), (cy + box_height / 2.0) * height))
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            raise ConversionError(f"{label_path}:{line_number}: bbox is empty after clipping")
        boxes.append([x1, y1, clipped_width, clipped_height])
    return boxes


def convert_dataset(
    data_root: Path,
    *,
    metadata_path: Path | None = None,
    category_id: int = 2,
) -> dict[str, Any]:
    root = data_root.resolve()
    metadata_path = (metadata_path or root / "metadata" / "conversion_metadata.json").resolve()
    index, metadata_record_count, source_metadata_summary = metadata_index(metadata_path)
    coco_by_split: dict[str, dict[str, Any]] = {}
    source_video_splits: dict[str, set[str]] = defaultdict(set)
    summary_splits: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ConversionError(f"Missing images/labels directory for split {split} under {root}")
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_stems = {path.stem for path in label_dir.glob("*.txt") if path.is_file()}
        image_stems = {path.stem for path in image_paths}
        if image_stems != label_stems:
            missing = sorted(image_stems - label_stems)[:3]
            extra = sorted(label_stems - image_stems)[:3]
            raise ConversionError(f"Image/label mismatch in {split}: missing={missing}, extra={extra}")

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        video_ids: set[str] = set()
        annotation_id = 1
        for image_id, image_path in enumerate(image_paths, start=1):
            record = find_metadata(index, image_path)
            metadata_split = str(record.get("split", "")).strip()
            if metadata_split != split:
                raise ConversionError(
                    f"Split mismatch for {image_path.name}: directory={split}, metadata={metadata_split!r}"
                )
            source_video_id = str(record.get("image_stem", "")).strip()
            if not source_video_id:
                raise ConversionError(f"Missing image_stem/source video id for {image_path.name}")
            source_video_splits[source_video_id].add(split)
            video_ids.add(source_video_id)
            with Image.open(image_path) as image:
                width, height = image.size
            digest = sha256_file(image_path)
            relative_path = image_path.relative_to(root).as_posix()
            images.append(
                {
                    "id": image_id,
                    "file_name": relative_path,
                    "width": int(width),
                    "height": int(height),
                    "metadata": {
                        "source_video_id": source_video_id,
                        "source_image": str(record.get("image_path", image_path)),
                        "annotation_kind": str(record.get("annotation_kind", "")),
                        "split": split,
                        "sha256": digest,
                    },
                }
            )
            boxes = parse_yolo_boxes(label_dir / f"{image_path.stem}.txt", width, height)
            for bbox in boxes:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        coco_by_split[split] = {
            "info": {
                "description": f"{root.name} COCO face detection dataset",
                "source_format": "YOLO Pose",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": int(category_id), "name": "face", "supercategory": "person"}],
        }
        summary_splits[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "source_videos": len(video_ids),
        }

    mixed_videos = {
        video_id: sorted(splits)
        for video_id, splits in source_video_splits.items()
        if len(splits) > 1
    }
    if mixed_videos:
        raise ConversionError(f"Source video ids cross dataset splits: {mixed_videos}")

    summary = {
        "schema_version": 1,
        "status": "ok",
        "data_root": str(root),
        "metadata": str(metadata_path),
        "metadata_records": metadata_record_count,
        "source_metadata_summary": source_metadata_summary,
        "source_excluded_faces": int(source_metadata_summary.get("excluded_faces", 0)),
        "conversion_errors": 0,
        "category_id": int(category_id),
        "splits": summary_splits,
        "totals": {
            "images": sum(item["images"] for item in summary_splits.values()),
            "annotations": sum(item["annotations"] for item in summary_splits.values()),
            "source_videos": len(source_video_splits),
        },
    }

    with tempfile.TemporaryDirectory(prefix=".coco_conversion_", dir=root) as directory:
        temporary_root = Path(directory)
        temporary_paths: dict[str, Path] = {}
        for split, coco in coco_by_split.items():
            path = temporary_root / f"{split}.json"
            path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_paths[split] = path
        audit = audit_coco_splits(
            root,
            annotation_paths=temporary_paths,
            expected_category_id=category_id,
        )
        summary["leakage_audit"] = audit
        summary_path = temporary_root / "coco_conversion_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for split in SPLITS:
            os.replace(temporary_paths[split], root / f"{split}.json")
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        os.replace(summary_path, root / "metadata" / "coco_conversion_summary.json")
    return summary


def main() -> int:
    args = parse_args()
    summary = convert_dataset(
        args.data_root,
        metadata_path=args.metadata,
        category_id=args.category_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
