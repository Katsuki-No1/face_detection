from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "val", "test")
SOURCE_KEYPOINT_NAMES = ("e1", "e2", "n", "m1", "m2")
TARGET_KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
FLIP_INDICES = (1, 0, 2, 4, 3)
SKELETON = ([1, 3], [2, 3], [3, 4], [3, 5])


class CanonicalAnnotationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalSource:
    dataset_root: Path
    metadata_path: Path
    metadata: dict[str, Any]
    records_by_split: dict[str, list[dict[str, Any]]]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalAnnotationError(f"Could not read canonical metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise CanonicalAnnotationError(f"Canonical metadata must be a JSON object: {path}")
    return value


def load_canonical_source(
    dataset_root: str | Path,
    metadata_path: str | Path | None = None,
) -> CanonicalSource:
    root = Path(dataset_root).expanduser().resolve()
    path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else root / "metadata" / "conversion_metadata.json"
    )
    metadata = _read_json(path)
    if tuple(metadata.get("keypoint_names", ())) != SOURCE_KEYPOINT_NAMES:
        raise CanonicalAnnotationError(
            f"Unexpected canonical keypoint order: {metadata.get('keypoint_names')}"
        )
    records = metadata.get("records")
    if not isinstance(records, list):
        raise CanonicalAnnotationError(f"Canonical metadata has no records list: {path}")
    records_by_split = {split: [] for split in SPLITS}
    seen_images: set[str] = set()
    source_video_splits: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise CanonicalAnnotationError(f"Invalid canonical record in {path}")
        split = str(record.get("split", ""))
        if split not in records_by_split:
            raise CanonicalAnnotationError(f"Invalid split {split!r} in {path}")
        output_image = str(record.get("output_image", ""))
        if not output_image or output_image in seen_images:
            raise CanonicalAnnotationError(f"Missing or duplicate output_image={output_image!r}")
        seen_images.add(output_image)
        expected_prefix = f"images/{split}/"
        if not output_image.startswith(expected_prefix):
            raise CanonicalAnnotationError(
                f"Image/split mismatch: split={split}, output_image={output_image}"
            )
        source_video = str(record.get("image_stem", "")).strip()
        if not source_video:
            raise CanonicalAnnotationError(f"Missing image_stem for {output_image}")
        source_video_splits.setdefault(source_video, set()).add(split)
        records_by_split[split].append(record)
    overlaps = {
        video: sorted(splits)
        for video, splits in source_video_splits.items()
        if len(splits) > 1
    }
    if overlaps:
        raise CanonicalAnnotationError(f"Source videos cross splits: {overlaps}")
    for values in records_by_split.values():
        values.sort(key=lambda record: str(record["output_image"]))
    return CanonicalSource(root, path, metadata, records_by_split)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_entry(
    source: CanonicalSource,
    record: dict[str, Any],
    *,
    image_id: int,
) -> tuple[dict[str, Any], int, int]:
    relative = Path(str(record["output_image"]))
    path = source.dataset_root / relative
    if not path.is_file():
        raise CanonicalAnnotationError(f"Missing canonical image: {path}")
    with Image.open(path) as image:
        width, height = image.size
    split = str(record["split"])
    entry = {
        "id": int(image_id),
        "file_name": relative.as_posix(),
        "width": int(width),
        "height": int(height),
        "metadata": {
            "source_video_id": str(record["image_stem"]),
            "source_image": str(record.get("image_path", "")),
            "annotation_kind": str(record.get("annotation_kind", "")),
            "split": split,
            "sha256": sha256_file(path),
        },
    }
    return entry, int(width), int(height)


def bbox_xywh(
    obj: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[float]:
    raw = obj.get("training_bbox")
    if not isinstance(raw, dict):
        raise CanonicalAnnotationError("Canonical object has no training_bbox")
    x1 = max(0.0, min(float(width), float(raw.get("xtl", 0.0))))
    y1 = max(0.0, min(float(height), float(raw.get("ytl", 0.0))))
    x2 = max(0.0, min(float(width), float(raw.get("xbr", 0.0))))
    y2 = max(0.0, min(float(height), float(raw.get("ybr", 0.0))))
    if x2 <= x1 or y2 <= y1:
        raise CanonicalAnnotationError(f"Invalid canonical bbox: {raw}")
    return [x1, y1, x2 - x1, y2 - y1]


def ellipse_params(obj: Mapping[str, Any]) -> list[float]:
    """Return the canonical DEIM ellipse target.

    The output intentionally matches the existing DEIMv2 ellipse dataset:
    [cx, cy, semi_major, semi_minor, major_axis_theta_rad], with the angle
    represented modulo pi in [-pi/2, pi/2).
    """

    face = obj.get("face", {})
    face = face if isinstance(face, Mapping) else {}
    raw = obj.get("cvat_ellipse") or face.get("cvat_ellipse")
    if not isinstance(raw, Mapping):
        raise CanonicalAnnotationError("Canonical face object has no cvat_ellipse")
    try:
        cx = float(raw["cx"])
        cy = float(raw["cy"])
        first = float(raw["rx"])
        second = float(raw["ry"])
        theta = math.radians(float(raw.get("rotation", raw.get("angle", 0.0))))
    except (KeyError, TypeError, ValueError) as error:
        raise CanonicalAnnotationError(f"Malformed canonical cvat_ellipse: {raw}") from error
    if not all(math.isfinite(value) for value in (cx, cy, first, second, theta)):
        raise CanonicalAnnotationError(f"Non-finite canonical cvat_ellipse: {raw}")
    if first <= 0.0 or second <= 0.0:
        raise CanonicalAnnotationError(f"Non-positive canonical ellipse radius: {raw}")
    if second > first:
        first, second = second, first
        theta += math.pi * 0.5
    theta = (theta + math.pi * 0.5) % math.pi - math.pi * 0.5
    return [cx, cy, first, second, theta]


def keypoint_payload(
    obj: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[list[float], list[int], list[bool]]:
    raw_keypoints = obj.get("keypoints")
    if not isinstance(raw_keypoints, dict):
        raise CanonicalAnnotationError("Canonical object has no keypoints mapping")
    values: list[float] = []
    visibility: list[int] = []
    occluded_flags: list[bool] = []
    for source_name in SOURCE_KEYPOINT_NAMES:
        point = raw_keypoints.get(source_name, {})
        if not isinstance(point, dict):
            point = {}
        present = int(point.get("v", 0)) > 0
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        inside = 0.0 <= x <= float(width) and 0.0 <= y <= float(height)
        if not present or not inside:
            values.extend((0.0, 0.0, 0))
            visibility.append(0)
            occluded_flags.append(False)
            continue
        is_occluded = bool(point.get("occluded", False))
        state = 1 if is_occluded else 2
        values.extend((x, y, state))
        visibility.append(state)
        occluded_flags.append(is_occluded)
    return values, visibility, occluded_flags


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
