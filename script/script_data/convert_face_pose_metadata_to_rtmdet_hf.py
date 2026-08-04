from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.coco_split_audit import audit_coco_splits, sha256_file  # noqa: E402


SPLITS = ("train", "val", "test")
CATEGORIES = (
    {"id": 1, "name": "head", "supercategory": "person"},
    {"id": 2, "name": "face", "supercategory": "person"},
)
CATEGORY_SOURCES = (("head", 1, "heads"), ("face", 2, "faces"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ConversionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert face_pose conversion metadata into a two-class RTMDet-HF "
            "COCO detection dataset."
        )
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Source face_pose dataset containing images and conversion metadata.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="New output directory. It must not already exist.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSON. Default: SOURCE_ROOT/metadata/conversion_metadata.json.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "reference"),
        default="hardlink",
        help=(
            "How to provide images. 'reference' writes annotations only and reuses "
            "SOURCE_ROOT/images without creating a second image tree."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default="rtmdet_hf_drive_v1",
        help="Version recorded in the conversion summary.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Could not read metadata {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ConversionError(f"Metadata must contain a records array: {path}")
    return data


def safe_relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConversionError(f"{field} must be a safe relative path: {value!r}")
    return path


def resolve_source_image(source_root: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    output_image = str(record.get("output_image", "")).strip()
    if output_image:
        relative = safe_relative_path(output_image, field="output_image")
        source_path = source_root / relative
        if source_path.is_file():
            return source_path, relative

    image_path_text = str(record.get("image_path", "")).strip()
    if image_path_text:
        image_path = Path(image_path_text)
        candidates = (
            [image_path]
            if image_path.is_absolute()
            else [PROJECT_ROOT / image_path, source_root / image_path]
        )
        for candidate in candidates:
            if candidate.is_file():
                split = str(record.get("split", "")).strip()
                relative = Path("images") / split / candidate.name
                return candidate, relative
    raise ConversionError(
        f"Could not resolve source image for {record.get('image_name')!r} "
        f"from output_image={output_image!r}, image_path={image_path_text!r}"
    )


def bbox_to_xyxy(shape: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = shape.get("bbox")
    if not isinstance(bbox, dict):
        raise ConversionError("Annotation shape is missing a bbox object")
    if {"xtl", "ytl", "xbr", "ybr"} <= set(bbox):
        values = (bbox["xtl"], bbox["ytl"], bbox["xbr"], bbox["ybr"])
    elif {"x", "y", "width", "height"} <= set(bbox):
        values = (
            bbox["x"],
            bbox["y"],
            float(bbox["x"]) + float(bbox["width"]),
            float(bbox["y"]) + float(bbox["height"]),
        )
    else:
        raise ConversionError(f"Unsupported bbox fields: {sorted(bbox)}")
    try:
        xyxy = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"BBox contains a non-numeric value: {values}") from error
    if not all(math.isfinite(value) for value in xyxy):
        raise ConversionError(f"BBox contains a non-finite value: {xyxy}")
    return xyxy  # type: ignore[return-value]


def clipped_xywh(
    shape: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = bbox_to_xyxy(shape)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    return [x1, y1, box_width, box_height]


def validate_shape_label(shape: dict[str, Any], expected: str) -> None:
    label = str(shape.get("label", "")).strip()
    if label and label.casefold() != expected.casefold():
        raise ConversionError(
            f"Expected {expected!r} in {expected} annotations, found label {label!r}"
        )


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.copy2(source, destination)
        return "copy_fallback"


def make_readme(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    image_note = (
        "Images are shared from the source dataset; this directory contains annotations only."
        if summary["image_mode_requested"] == "reference"
        else "Images are materialized from the source dataset without changing their bytes."
    )
    lines = [
        f"# {summary['dataset_version']}",
        "",
        "Two-class COCO Detection dataset generated from face_pose conversion metadata.",
        "",
        f"- Dataset version: `{summary['dataset_version']}`",
        f"- Source: `{summary['source_root']}`",
        f"- Images: {totals['images']}",
        f"- Head annotations: {totals['head_annotations']}",
        f"- Face annotations: {totals['face_annotations']}",
        "- Categories: `head=1`, `face=2`",
        "- Split policy: source train/val/test assignments are preserved",
        "",
        image_note,
        "See `conversion_summary.json` for counts, exclusions, hashes, and leakage audit results.",
        "",
    ]
    return "\n".join(lines)


def convert_dataset(
    source_root: Path,
    output_root: Path,
    *,
    metadata_path: Path | None = None,
    image_mode: str = "hardlink",
    dataset_version: str = "rtmdet_hf_drive_v1",
) -> dict[str, Any]:
    if image_mode not in {"hardlink", "copy", "reference"}:
        raise ValueError(f"Unsupported image_mode: {image_mode}")
    source = source_root.resolve()
    output = output_root.resolve()
    metadata = (metadata_path or source / "metadata" / "conversion_metadata.json").resolve()
    if output.exists():
        raise ConversionError(f"Output directory already exists; refusing to overwrite: {output}")
    if source == output:
        raise ConversionError("Output directory must not be the source directory")
    if source in output.parents and image_mode != "reference":
        raise ConversionError(
            "Output inside the source directory requires image_mode='reference'"
        )
    data = read_json(metadata)
    records = data["records"]
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}_", dir=output.parent) as temporary:
        staging_root = Path(temporary) / "dataset"
        staging_root.mkdir()
        seen_output_paths: set[str] = set()
        source_video_splits: dict[str, set[str]] = {}
        split_coco: dict[str, dict[str, Any]] = {}
        split_summary: dict[str, dict[str, int]] = {}
        materialization_counts: Counter[str] = Counter()

        for split in SPLITS:
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            counters: Counter[str] = Counter()
            annotation_id = 1
            split_records = [
                record
                for record in records
                if str(record.get("split", "")).strip() == split
            ]
            for image_id, record in enumerate(split_records, start=1):
                source_image, relative_image = resolve_source_image(source, record)
                if relative_image.parts[:2] != ("images", split):
                    raise ConversionError(
                        f"Image path does not match metadata split {split}: {relative_image}"
                    )
                output_key = relative_image.as_posix()
                if output_key in seen_output_paths:
                    raise ConversionError(f"Duplicate output image path: {relative_image}")
                seen_output_paths.add(output_key)
                if source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ConversionError(f"Unsupported image extension: {source_image}")
                with Image.open(source_image) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ConversionError(f"Invalid image dimensions: {source_image}")

                source_video_id = str(record.get("image_stem", "")).strip()
                if not source_video_id:
                    raise ConversionError(f"Missing image_stem/source video id: {source_image}")
                source_video_splits.setdefault(source_video_id, set()).add(split)
                digest = sha256_file(source_image)
                if image_mode == "reference":
                    effective_mode = "reference"
                else:
                    effective_mode = materialize_image(
                        source_image,
                        staging_root / relative_image,
                        image_mode,
                    )
                materialization_counts[effective_mode] += 1
                images.append(
                    {
                        "id": image_id,
                        "file_name": relative_image.as_posix(),
                        "width": int(width),
                        "height": int(height),
                        "metadata": {
                            "source_video_id": source_video_id,
                            "source_image": (
                                relative_image.as_posix()
                                if image_mode == "reference"
                                else str(source_image)
                            ),
                            "annotation_kind": str(record.get("annotation_kind", "")),
                            "split": split,
                            "sha256": digest,
                            "source_negative_sample": bool(record.get("negative_sample", False)),
                        },
                    }
                )
                counters["images"] += 1

                for category_name, category_id, record_key in CATEGORY_SOURCES:
                    source_shapes = record.get(record_key, [])
                    if not isinstance(source_shapes, list):
                        raise ConversionError(
                            f"{record_key} must be an array for image {source_image}"
                        )
                    valid_for_image = 0
                    for shape_index, shape in enumerate(source_shapes):
                        if not isinstance(shape, dict):
                            raise ConversionError(
                                f"Invalid {category_name} shape at index {shape_index}: {source_image}"
                            )
                        validate_shape_label(shape, category_name)
                        bbox = clipped_xywh(shape, width=width, height=height)
                        if bbox is None:
                            counters[f"excluded_{category_name}_annotations"] += 1
                            continue
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0,
                                "metadata": {
                                    "source_shape_index": shape_index,
                                    "source_label": str(shape.get("label", "")),
                                },
                            }
                        )
                        annotation_id += 1
                        valid_for_image += 1
                        counters[f"{category_name}_annotations"] += 1
                    if valid_for_image:
                        counters[f"images_with_{category_name}"] += 1

            split_coco[split] = {
                "info": {
                    "description": "RTMDet-HF head/face full-image detection dataset",
                    "dataset_version": dataset_version,
                    "source_format": "face_pose conversion metadata",
                },
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [dict(category) for category in CATEGORIES],
            }
            split_summary[split] = {
                key: int(counters.get(key, 0))
                for key in (
                    "images",
                    "images_with_head",
                    "images_with_face",
                    "head_annotations",
                    "face_annotations",
                    "excluded_head_annotations",
                    "excluded_face_annotations",
                )
            }

        unknown_split_records = [
            record
            for record in records
            if str(record.get("split", "")).strip() not in SPLITS
        ]
        if unknown_split_records:
            raise ConversionError(
                f"Metadata contains {len(unknown_split_records)} records without a valid split"
            )
        mixed_videos = {
            video_id: sorted(splits)
            for video_id, splits in source_video_splits.items()
            if len(splits) > 1
        }
        if mixed_videos:
            first_video, splits = next(iter(mixed_videos.items()))
            raise ConversionError(
                f"Source video ids cross dataset splits: {first_video!r} appears in {splits}"
            )

        for split, coco in split_coco.items():
            (staging_root / f"{split}.json").write_text(
                json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        annotation_paths = {
            split: staging_root / f"{split}.json"
            for split in SPLITS
        }
        audit_root = source if image_mode == "reference" else staging_root
        audit = audit_coco_splits(
            audit_root,
            annotation_paths=annotation_paths,
            expected_category_ids=(1, 2),
        )
        audit["data_root"] = str(source if image_mode == "reference" else output)
        audit["annotation_root"] = str(output)
        totals = {
            key: sum(values[key] for values in split_summary.values())
            for key in split_summary[SPLITS[0]]
        }
        summary = {
            "schema_version": 1,
            "status": "ok",
            "dataset_version": dataset_version,
            "source_root": str(source),
            "output_root": str(output),
            "source_metadata": str(metadata),
            "source_metadata_sha256": sha256_file(metadata),
            "source_records": len(records),
            "categories": [dict(category) for category in CATEGORIES],
            "image_mode_requested": image_mode,
            "image_materialization": dict(sorted(materialization_counts.items())),
            "splits": split_summary,
            "totals": totals,
            "leakage_audit": audit,
        }
        (staging_root / "conversion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_root / "README.md").write_text(make_readme(summary), encoding="utf-8")
        os.replace(staging_root, output)
    return summary


def main() -> int:
    args = parse_args()
    summary = convert_dataset(
        args.source_root,
        args.output_root,
        metadata_path=args.metadata,
        image_mode=args.image_mode,
        dataset_version=args.dataset_version,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
