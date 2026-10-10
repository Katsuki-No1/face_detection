from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("train", "val", "test")


class CocoSplitLeakageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CocoSplitLeakageError(f"Could not read COCO split {path}: {error}") from error
    if not isinstance(data, dict):
        raise CocoSplitLeakageError(f"COCO split must contain a JSON object: {path}")
    return data


def _cross_split_values(
    values: Mapping[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    return {
        value: items
        for value, items in values.items()
        if len({item["split"] for item in items}) > 1
    }


def audit_coco_splits(
    data_root: str | Path,
    *,
    annotation_paths: Mapping[str, str | Path] | None = None,
    image_roots: Mapping[str, str | Path] | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    expected_category_id: int | None = 2,
    expected_category_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    if expected_category_ids is not None:
        expected_ids = {int(category_id) for category_id in expected_category_ids}
    elif expected_category_id is not None:
        expected_ids = {int(expected_category_id)}
    else:
        raise ValueError("expected_category_id or expected_category_ids must be provided")
    if not expected_ids:
        raise ValueError("expected_category_ids must not be empty")
    paths = {
        split: Path(annotation_paths[split]) if annotation_paths is not None else root / f"{split}.json"
        for split in splits
    }
    source_videos: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_paths: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_summary: dict[str, dict[str, int]] = {}

    for split in splits:
        path = paths[split]
        coco = _load_coco(path)
        categories = coco.get("categories", [])
        category_ids = {
            int(category["id"])
            for category in categories
            if isinstance(category, dict) and "id" in category
        }
        if category_ids != expected_ids:
            raise CocoSplitLeakageError(
                f"{path} must define category ids {sorted(expected_ids)}; "
                f"found {sorted(category_ids)}"
            )

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise CocoSplitLeakageError(f"{path} has invalid images/annotations arrays")
        image_ids: set[int] = set()
        for image in images:
            if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
                raise CocoSplitLeakageError(f"{path} contains an invalid image record")
            image_id = int(image["id"])
            if image_id in image_ids:
                raise CocoSplitLeakageError(f"{path} contains duplicate image id {image_id}")
            image_ids.add(image_id)
            metadata = image.get("metadata", {})
            source_video_id = str(metadata.get("source_video_id", "")).strip()
            if not source_video_id:
                raise CocoSplitLeakageError(
                    f"{path}: image id {image_id} is missing metadata.source_video_id"
                )
            relative_path = Path(str(image["file_name"]))
            if relative_path.is_absolute():
                raise CocoSplitLeakageError(f"{path}: absolute image paths are not allowed: {relative_path}")
            split_root = (
                Path(image_roots[split])
                if image_roots is not None and split in image_roots
                else root
            )
            image_path = split_root / relative_path
            if not image_path.is_file():
                raise CocoSplitLeakageError(f"{path}: image file does not exist: {image_path}")
            digest = sha256_file(image_path)
            recorded_digest = str(metadata.get("sha256", "")).strip().lower()
            if not recorded_digest:
                raise CocoSplitLeakageError(f"{path}: image id {image_id} is missing metadata.sha256")
            if recorded_digest != digest:
                raise CocoSplitLeakageError(
                    f"{path}: SHA-256 changed for {relative_path}: recorded={recorded_digest}, actual={digest}"
                )
            item = {"split": split, "image": str(relative_path)}
            source_videos[source_video_id].append(item)
            image_hashes[digest].append(item)
            image_paths[str(image_path.resolve())].append(item)

        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise CocoSplitLeakageError(f"{path} contains an invalid annotation record")
            category_id = int(annotation.get("category_id", -1))
            if category_id not in expected_ids:
                raise CocoSplitLeakageError(
                    f"{path}: annotation {annotation.get('id')} has category id {category_id}, "
                    f"expected one of {sorted(expected_ids)}"
                )
            if int(annotation.get("image_id", -1)) not in image_ids:
                raise CocoSplitLeakageError(
                    f"{path}: annotation {annotation.get('id')} references an unknown image"
                )
        split_summary[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "source_videos": len(
                {
                    str(image.get("metadata", {}).get("source_video_id"))
                    for image in images
                }
            ),
        }

    overlaps = {
        "source_video_ids": _cross_split_values(source_videos),
        "sha256": _cross_split_values(image_hashes),
        "image_paths": _cross_split_values(image_paths),
    }
    for kind, values in overlaps.items():
        if values:
            first_value, first_items = next(iter(values.items()))
            raise CocoSplitLeakageError(
                f"Data leakage detected: {kind} value {first_value!r} appears across splits: "
                f"{first_items}"
            )

    return {
        "status": "ok",
        "data_root": str(root),
        "image_roots": {
            split: str(Path(image_roots[split])) if image_roots and split in image_roots else str(root)
            for split in splits
        },
        "expected_category_ids": sorted(expected_ids),
        "expected_category_id": next(iter(expected_ids)) if len(expected_ids) == 1 else None,
        "splits": split_summary,
        "totals": {
            "images": sum(item["images"] for item in split_summary.values()),
            "annotations": sum(item["annotations"] for item in split_summary.values()),
            "source_videos": len(source_videos),
        },
        "cross_split_overlap_counts": {kind: 0 for kind in overlaps},
    }


__all__ = ["CocoSplitLeakageError", "audit_coco_splits", "sha256_file"]
