from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment


SPLITS = ("train", "val", "test")
V2_EXPECTED_ANNOTATIONS = {"train": 6542, "val": 784, "test": 812}


class EllipseExportError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EllipseExportError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EllipseExportError(f"Expected a JSON object in {path}")
    return value


def _metadata_index(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in metadata.get("records", []):
        if not isinstance(record, dict):
            continue
        keys: set[str] = set()
        for field in ("output_image", "image_name", "image_path"):
            value = record.get(field)
            if isinstance(value, str) and value:
                keys.update((value, Path(value).name, Path(value).stem))
        for key in keys:
            previous = index.get(key)
            if previous is not None and previous is not record:
                raise EllipseExportError(f"Ambiguous metadata key: {key!r}")
            index[key] = record
    return index


def _record_for_image(index: Mapping[str, dict[str, Any]], file_name: str) -> dict[str, Any]:
    path = Path(file_name)
    record = index.get(file_name) or index.get(path.name) or index.get(path.stem)
    if record is None:
        raise EllipseExportError(f"No conversion metadata for image: {file_name}")
    return record


def _ellipse_from_object(item: Mapping[str, Any]) -> list[float]:
    face = item.get("face", {})
    face = face if isinstance(face, Mapping) else {}
    source = item.get("cvat_ellipse") or face.get("cvat_ellipse")
    if not isinstance(source, Mapping):
        raise EllipseExportError("Metadata object has no cvat_ellipse")
    try:
        cx = float(source["cx"])
        cy = float(source["cy"])
        first = float(source["rx"])
        second = float(source["ry"])
        theta = math.radians(float(source.get("rotation", source.get("angle", 0.0))))
    except (KeyError, TypeError, ValueError) as error:
        raise EllipseExportError(f"Malformed cvat_ellipse: {source}") from error
    if not all(math.isfinite(value) for value in (cx, cy, first, second, theta)):
        raise EllipseExportError(f"Non-finite cvat_ellipse: {source}")
    if first <= 0.0 or second <= 0.0:
        raise EllipseExportError(f"Non-positive ellipse radius: {source}")
    if second > first:
        first, second = second, first
        theta += math.pi * 0.5
    theta = (theta + math.pi * 0.5) % math.pi - math.pi * 0.5
    return [cx, cy, first, second, theta]


def _ellipse_aabb(ellipse: list[float], width: int, height: int) -> list[float]:
    cx, cy, major, minor, theta = ellipse
    half_width = math.sqrt((major * math.cos(theta)) ** 2 + (minor * math.sin(theta)) ** 2)
    half_height = math.sqrt((major * math.sin(theta)) ** 2 + (minor * math.cos(theta)) ** 2)
    return [
        max(0.0, min(float(width), cx - half_width)),
        max(0.0, min(float(height), cy - half_height)),
        max(0.0, min(float(width), cx + half_width)),
        max(0.0, min(float(height), cy + half_height)),
    ]


def _annotation_xyxy(annotation: Mapping[str, Any]) -> list[float]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise EllipseExportError(f"Invalid COCO bbox in annotation id={annotation.get('id')}")
    x, y, width, height = [float(value) for value in bbox]
    return [x, y, x + width, y + height]


def _box_iou(first: list[float], second: list[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0e-12, first_area + second_area - intersection)


def _match_ellipses(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    objects: list[Mapping[str, Any]],
    *,
    minimum_iou: float,
) -> list[list[float]]:
    if len(annotations) != len(objects):
        raise EllipseExportError(
            f"Object count mismatch for {image.get('file_name')}: "
            f"annotations={len(annotations)}, metadata_objects={len(objects)}"
        )
    if not annotations:
        return []
    ellipses = [_ellipse_from_object(item) for item in objects]
    source_boxes = [_ellipse_aabb(item, int(image["width"]), int(image["height"])) for item in ellipses]
    annotation_boxes = [_annotation_xyxy(item) for item in annotations]
    costs = [[1.0 - _box_iou(source_box, annotation_box) for annotation_box in annotation_boxes] for source_box in source_boxes]
    source_indices, annotation_indices = linear_sum_assignment(costs)
    by_annotation: list[list[float] | None] = [None] * len(annotations)
    for source_index, annotation_index in zip(source_indices.tolist(), annotation_indices.tolist()):
        overlap = 1.0 - float(costs[source_index][annotation_index])
        if overlap < minimum_iou:
            raise EllipseExportError(
                f"Ellipse/bbox mismatch for {image.get('file_name')}: "
                f"annotation_id={annotations[annotation_index].get('id')}, IoU={overlap:.9f}"
            )
        by_annotation[annotation_index] = ellipses[source_index]
    if any(value is None for value in by_annotation):
        raise EllipseExportError(f"Incomplete ellipse assignment for {image.get('file_name')}")
    return [value for value in by_annotation if value is not None]


def export_ellipse_coco(
    data_root: Path,
    *,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    expected_annotations: Mapping[str, int] | None = V2_EXPECTED_ANNOTATIONS,
    minimum_iou: float = 0.999,
) -> dict[str, Any]:
    root = data_root.resolve()
    metadata_path = (metadata_path or root / "metadata" / "conversion_metadata.json").resolve()
    output_dir = (output_dir or root / "ellipse_annotations").resolve()
    metadata = _read_json(metadata_path)
    index = _metadata_index(metadata)
    converted: dict[str, dict[str, Any]] = {}
    split_summary: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        source_path = root / f"{split}.json"
        source = _read_json(source_path)
        result = copy.deepcopy(source)
        images = {int(image["id"]): image for image in result.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
        for annotation in result.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        ellipse_count = 0
        negative_images = 0
        for image_id, image in images.items():
            annotations = annotations_by_image.get(image_id, [])
            record = _record_for_image(index, str(image.get("file_name", "")))
            if str(record.get("split", split)) != split:
                raise EllipseExportError(
                    f"Split mismatch for {image.get('file_name')}: COCO={split}, metadata={record.get('split')}"
                )
            raw_objects = record.get("objects", [])
            objects = [item for item in raw_objects if isinstance(item, Mapping)]
            assigned = _match_ellipses(image, annotations, objects, minimum_iou=minimum_iou)
            if not annotations:
                negative_images += 1
            for annotation, ellipse in zip(annotations, assigned):
                annotation["ellipse"] = ellipse
                ellipse_count += 1

        expected = expected_annotations.get(split) if expected_annotations is not None else None
        if expected is not None and ellipse_count != int(expected):
            raise EllipseExportError(
                f"Unexpected {split} ellipse count: expected={expected}, actual={ellipse_count}"
            )
        info = dict(result.get("info", {}))
        info.update(
            {
                "ellipse_schema_version": 1,
                "ellipse_format": "[cx, cy, semi_major, semi_minor, theta_rad]",
                "ellipse_angle_convention": "major-axis angle in [-pi/2, pi/2), modulo pi",
                "ellipse_source": str(metadata_path),
            }
        )
        result["info"] = info
        converted[split] = result
        split_summary[split] = {
            "images": len(images),
            "annotations": len(result.get("annotations", [])),
            "ellipses": ellipse_count,
            "negative_images": negative_images,
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "data_root": str(root),
        "metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "minimum_match_iou": float(minimum_iou),
        "splits": split_summary,
        "totals": {
            "images": sum(row["images"] for row in split_summary.values()),
            "annotations": sum(row["annotations"] for row in split_summary.values()),
            "ellipses": sum(row["ellipses"] for row in split_summary.values()),
        },
    }

    split_sha256: dict[str, str] = {}
    seen_image_digests: dict[str, str] = {}
    seen_source_videos: dict[str, str] = {}
    duplicate_digests: list[dict[str, str]] = []
    duplicate_source_videos: list[dict[str, str]] = []
    for split, data in converted.items():
        for image in data.get("images", []):
            metadata_row = image.get("metadata", {})
            metadata_row = metadata_row if isinstance(metadata_row, Mapping) else {}
            digest = str(metadata_row.get("sha256", ""))
            source_video = str(metadata_row.get("source_video_id", ""))
            if digest and digest in seen_image_digests and seen_image_digests[digest] != split:
                duplicate_digests.append({"sha256": digest, "first_split": seen_image_digests[digest], "split": split})
            elif digest:
                seen_image_digests[digest] = split
            if source_video and source_video in seen_source_videos and seen_source_videos[source_video] != split:
                duplicate_source_videos.append(
                    {"source_video_id": source_video, "first_split": seen_source_videos[source_video], "split": split}
                )
            elif source_video:
                seen_source_videos[source_video] = split
    if duplicate_digests or duplicate_source_videos:
        raise EllipseExportError(
            "Cross-split leakage detected while exporting ellipse annotations: "
            f"duplicate_images={len(duplicate_digests)}, duplicate_source_videos={len(duplicate_source_videos)}"
        )
    summary["leakage_audit"] = {
        "status": "ok",
        "duplicate_image_sha256": 0,
        "duplicate_source_video_ids": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ellipse_coco_", dir=output_dir.parent) as temp_directory:
        temp_root = Path(temp_directory)
        for split, data in converted.items():
            split_path = temp_root / f"{split}.json"
            split_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            split_sha256[split] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        summary["sha256"] = split_sha256
        (temp_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for split in SPLITS:
            os.replace(temp_root / f"{split}.json", output_dir / f"{split}.json")
        os.replace(temp_root / "summary.json", output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach original ellipse geometry to face-pose COCO annotations.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-iou", type=float, default=0.999)
    parser.add_argument("--skip-v2-count-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_ellipse_coco(
        args.data_root,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        expected_annotations=None if args.skip_v2_count_check else V2_EXPECTED_ANNOTATIONS,
        minimum_iou=float(args.minimum_iou),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
