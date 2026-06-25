from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .config import resolve_data_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def clip_box(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: list[float] | tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(first: list[float], second: list[float]) -> float:
    ix1 = max(first[0], second[0])
    iy1 = max(first[1], second[1])
    ix2 = min(first[2], second[2])
    iy2 = min(first[3], second[3])
    intersection = box_area([ix1, iy1, ix2, iy2])
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def expand_box(box: list[float], width: int, height: int, padding_ratio: float) -> list[float] | None:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def yolo_box_to_xyxy(values: list[float], width: int, height: int) -> list[float] | None:
    xc, yc, bw, bh = values
    box_width = bw * width
    box_height = bh * height
    cx = xc * width
    cy = yc * height
    return clip_box([cx - box_width / 2.0, cy - box_height / 2.0, cx + box_width / 2.0, cy + box_height / 2.0], width, height)


def dict_bbox_to_xyxy(bbox: dict[str, Any], width: int, height: int) -> list[float] | None:
    return clip_box(
        [
            float(bbox.get("xtl", 0.0)),
            float(bbox.get("ytl", 0.0)),
            float(bbox.get("xbr", 0.0)),
            float(bbox.get("ybr", 0.0)),
        ],
        width,
        height,
    )


def face_size_category(box: list[float], width: int, height: int) -> str:
    area_ratio = box_area(box) / float(max(1, width * height))
    if area_ratio < 0.0025:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio >= 0.15:
        return "large"
    return "medium"


def split_text(value: str) -> list[str]:
    return [item for item in re.split(r"[|,;]", value) if item]


def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def read_label_objects(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [float(part) for part in line.split()]
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        bbox = yolo_box_to_xyxy(parts[1:5], width, height)
        if bbox is None:
            continue
        keypoints = {}
        raw_keypoints = parts[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            offset = index * 3
            if offset + 2 >= len(raw_keypoints):
                keypoints[name] = {"x": 0.0, "y": 0.0, "v": 0}
                continue
            keypoints[name] = {
                "x": raw_keypoints[offset] * width,
                "y": raw_keypoints[offset + 1] * height,
                "v": int(raw_keypoints[offset + 2]),
            }
        objects.append({"bbox_xyxy": bbox, "keypoints": keypoints})
    return objects


def discover_metadata(data_root: Path, explicit_metadata: str | Path | None) -> Path | None:
    candidates = []
    if explicit_metadata:
        candidates.append(Path(explicit_metadata))
    candidates.extend(
        [
            data_root / "metadata" / "conversion_metadata.json",
            data_root.parent / "_modified_all_train" / "metadata" / "conversion_metadata.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metadata_index(data_root: Path, explicit_metadata: str | Path | None) -> dict[str, dict[str, Any]]:
    metadata_path = discover_metadata(data_root, explicit_metadata)
    if metadata_path is None:
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for record in data.get("records", []):
        keys = set()
        for field in ("image_name", "output_image", "image", "image_path"):
            value = record.get(field)
            if isinstance(value, str) and value:
                keys.add(value)
                keys.add(Path(value).name)
                keys.add(Path(value).stem)
        for key in keys:
            index[key] = record
    return index


def metadata_for_image(index: dict[str, dict[str, Any]], image_path: Path) -> dict[str, Any] | None:
    return index.get(image_path.name) or index.get(image_path.stem) or index.get(str(image_path))


def valid_metadata_objects(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    objects = []
    for item in record.get("objects", []):
        if "invalid_clipped_face_bbox" not in item.get("warnings", []):
            objects.append(item)
    return objects


def head_box_for_face(record: dict[str, Any] | None, face_box: list[float], width: int, height: int) -> list[float] | None:
    if not record:
        return None
    best_box = None
    best_score = 0.0
    face_cx = (face_box[0] + face_box[2]) / 2.0
    face_cy = (face_box[1] + face_box[3]) / 2.0
    for head in record.get("heads", []):
        bbox = head.get("bbox", {})
        candidate = dict_bbox_to_xyxy(bbox, width, height) if bbox else None
        if candidate is None:
            continue
        contains_center = candidate[0] <= face_cx <= candidate[2] and candidate[1] <= face_cy <= candidate[3]
        score = box_iou(candidate, face_box) + (0.25 if contains_center else 0.0)
        if score > best_score:
            best_score = score
            best_box = candidate
    return best_box


def eye_region_box(keypoints: dict[str, dict[str, Any]], face_box: list[float], width: int, height: int, padding_ratio: float) -> list[float] | None:
    eye_points = []
    for name in ("e1", "e2"):
        point = keypoints.get(name, {})
        if int(point.get("v", 0)) > 0:
            eye_points.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
    if not eye_points:
        return None

    if len(eye_points) == 1:
        face_w = face_box[2] - face_box[0]
        face_h = face_box[3] - face_box[1]
        half = max(face_w, face_h) * 0.18
        x, y = eye_points[0]
        base = [x - half, y - half, x + half, y + half]
    else:
        xs = [point[0] for point in eye_points]
        ys = [point[1] for point in eye_points]
        span = max(max(xs) - min(xs), max(4.0, (face_box[3] - face_box[1]) * 0.12))
        base = [min(xs) - span * 0.35, min(ys) - span * 0.8, max(xs) + span * 0.35, max(ys) + span * 0.8]
    return expand_box(base, width, height, padding_ratio)


def crop_brightness(image_path: Path, box: list[float]) -> float:
    with Image.open(image_path) as image:
        crop = image.convert("L").crop(tuple(round(value) for value in box))
        return float(ImageStat.Stat(crop).mean[0])


def object_tags(
    box: list[float],
    width: int,
    height: int,
    meta_object: dict[str, Any] | None,
    image_path: Path,
    keypoints: dict[str, dict[str, Any]],
) -> list[str]:
    tags = [face_size_category(box, width, height)]
    if crop_brightness(image_path, box) < 55.0:
        tags.append("dark")
    if meta_object:
        if meta_object.get("face", {}).get("occluded"):
            tags.append("occluded")
        if "face_bbox_clipped_to_image" in meta_object.get("warnings", []):
            tags.append("ellipse_oob")
        if any(point.get("occluded") for point in meta_object.get("keypoints", {}).values()):
            tags.append("keypoint_occluded")
    visible = [point for point in keypoints.values() if int(point.get("v", 0)) > 0]
    if len(visible) < len(KEYPOINT_NAMES):
        tags.append("incomplete_keypoints")
    eye_1 = keypoints.get("e1", {})
    eye_2 = keypoints.get("e2", {})
    if int(eye_1.get("v", 0)) > 0 and int(eye_2.get("v", 0)) > 0:
        angle = abs(math.degrees(math.atan2(float(eye_2["y"]) - float(eye_1["y"]), float(eye_2["x"]) - float(eye_1["x"]))))
        if 25.0 <= angle <= 155.0:
            tags.append("angled_eye_line")
    return sorted(set(tags))


def random_background_box(
    rng: random.Random,
    width: int,
    height: int,
    face_boxes: list[list[float]],
    min_crop_size: int,
    max_iou: float,
    prefer_face_like_size: bool,
) -> list[float] | None:
    if face_boxes and prefer_face_like_size:
        reference = rng.choice(face_boxes)
        ref_w = reference[2] - reference[0]
        ref_h = reference[3] - reference[1]
    elif face_boxes:
        reference = rng.choice(face_boxes)
        ref_w = reference[2] - reference[0]
        ref_h = reference[3] - reference[1]
    else:
        ref_w = width * 0.2
        ref_h = height * 0.2
    for _ in range(200):
        scale = rng.uniform(0.75, 1.35) if prefer_face_like_size else rng.uniform(0.6, 1.6)
        crop_w = int(max(min_crop_size, min(width, ref_w * scale)))
        crop_h = int(max(min_crop_size, min(height, ref_h * scale)))
        if crop_w >= width or crop_h >= height:
            crop_w = min(width, crop_w)
            crop_h = min(height, crop_h)
        x1 = rng.randint(0, max(0, width - crop_w))
        y1 = rng.randint(0, max(0, height - crop_h))
        candidate = [float(x1), float(y1), float(x1 + crop_w), float(y1 + crop_h)]
        if all(box_iou(candidate, face_box) <= max_iou for face_box in face_boxes):
            return candidate
    return None


def make_crop_record(
    *,
    image_path: Path,
    split: str,
    crop_type: str,
    label: str,
    box: list[float],
    width: int,
    height: int,
    object_index: int,
    crop_index: int,
    face_size: str,
    tags: list[str],
) -> dict[str, Any]:
    crop_id = safe_id(f"{split}_{image_path.stem}_{crop_type}_{object_index}_{crop_index}")
    return {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "crop_id": crop_id,
        "crop_type": crop_type,
        "label": label,
        "bbox_xyxy": [round(float(value), 4) for value in box],
        "original_width": width,
        "original_height": height,
        "face_size_category": face_size,
        "tags": sorted(set(tags)),
        "split": split,
        "object_index": object_index,
    }


def generate_phase1_crops(config: dict[str, Any], max_images: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_root = resolve_data_root(config)
    data_cfg = config.get("data", {})
    crop_cfg = config.get("crops", {})
    split = str(data_cfg.get("split", "test"))
    limit = max_images if max_images is not None else data_cfg.get("max_images")
    rng = random.Random(int(data_cfg.get("seed", 42)))
    metadata = metadata_index(data_root, data_cfg.get("metadata_path"))

    paths = image_paths(data_root, split)
    if limit is not None:
        paths = paths[: int(limit)]

    crop_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    padding = float(crop_cfg.get("crop_padding_ratio", 0.15))
    min_crop_size = int(crop_cfg.get("min_crop_size", 16))
    max_background_iou = float(crop_cfg.get("max_background_iou", 0.05))

    for image_path in paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = label_path_for_image(data_root, split, image_path)
        label_objects = read_label_objects(label_path, width, height)
        meta_record = metadata_for_image(metadata, image_path)
        meta_objects = valid_metadata_objects(meta_record)
        face_boxes = [item["bbox_xyxy"] for item in label_objects]
        image_records.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "split": split,
                "width": width,
                "height": height,
                "face_bboxes": face_boxes,
            }
        )

        for object_index, item in enumerate(label_objects):
            face_box = item["bbox_xyxy"]
            meta_object = meta_objects[object_index] if object_index < len(meta_objects) else None
            tags = object_tags(face_box, width, height, meta_object, image_path, item["keypoints"])
            size = face_size_category(face_box, width, height)
            if crop_cfg.get("use_face_crop", True):
                box = expand_box(face_box, width, height, padding) or face_box
                crop_records.append(
                    make_crop_record(
                        image_path=image_path,
                        split=split,
                        crop_type="face",
                        label="face_like",
                        box=box,
                        width=width,
                        height=height,
                        object_index=object_index,
                        crop_index=0,
                        face_size=size,
                        tags=tags,
                    )
                )

            if crop_cfg.get("use_head_crop", True):
                head_box = head_box_for_face(meta_record, face_box, width, height)
                if head_box is not None:
                    crop_records.append(
                        make_crop_record(
                            image_path=image_path,
                            split=split,
                            crop_type="head",
                            label="face_like",
                            box=head_box,
                            width=width,
                            height=height,
                            object_index=object_index,
                            crop_index=0,
                            face_size=size,
                            tags=[*tags, "head_context"],
                        )
                    )

            if crop_cfg.get("use_eye_crop", True):
                eye_box = eye_region_box(item["keypoints"], face_box, width, height, padding)
                if eye_box is not None:
                    crop_records.append(
                        make_crop_record(
                            image_path=image_path,
                            split=split,
                            crop_type="eye_region",
                            label="face_like",
                            box=eye_box,
                            width=width,
                            height=height,
                            object_index=object_index,
                            crop_index=0,
                            face_size=size,
                            tags=[*tags, "eye_region"],
                        )
                    )

        if crop_cfg.get("use_background_crop", True):
            for crop_index in range(int(crop_cfg.get("background_crops_per_image", 3))):
                box = random_background_box(rng, width, height, face_boxes, min_crop_size, max_background_iou, False)
                if box is None:
                    continue
                crop_records.append(
                    make_crop_record(
                        image_path=image_path,
                        split=split,
                        crop_type="background",
                        label="non_face",
                        box=box,
                        width=width,
                        height=height,
                        object_index=-1,
                        crop_index=crop_index,
                        face_size="none",
                        tags=["background"],
                    )
                )

        if crop_cfg.get("use_hard_negative_crop", True):
            for crop_index in range(int(crop_cfg.get("hard_negative_crops_per_image", 1))):
                box = random_background_box(rng, width, height, face_boxes, min_crop_size, max_background_iou, True)
                if box is None:
                    continue
                crop_records.append(
                    make_crop_record(
                        image_path=image_path,
                        split=split,
                        crop_type="hard_negative",
                        label="non_face",
                        box=box,
                        width=width,
                        height=height,
                        object_index=-1,
                        crop_index=crop_index,
                        face_size="none",
                        tags=["hard_negative", "todo_random_background_candidate"],
                    )
                )

    return crop_records, image_records


def crop_image(image_path: Path, box: list[float]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image.crop(tuple(round(value) for value in box))


def csv_ready_record(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    row["bbox_xyxy"] = json.dumps(record["bbox_xyxy"])
    row["tags"] = "|".join(record.get("tags", []))
    return row


def write_crop_metadata(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = [
        "image_id",
        "image_path",
        "crop_id",
        "crop_type",
        "label",
        "bbox_xyxy",
        "original_width",
        "original_height",
        "face_size_category",
        "tags",
        "split",
        "object_index",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_ready_record(record).get(key, "") for key in keys})


def save_crop_images(records: list[dict[str, Any]], crop_dir: Path) -> list[dict[str, Any]]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for record in records:
        crop = crop_image(Path(record["image_path"]), record["bbox_xyxy"])
        crop_path = crop_dir / f"{record['crop_id']}.jpg"
        crop.save(crop_path, quality=92)
        updated_record = dict(record)
        updated_record["crop_path"] = str(crop_path)
        updated.append(updated_record)
    return updated


def prepare_crops(config: dict[str, Any], output_dir: Path, max_images: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, image_records = generate_phase1_crops(config, max_images=max_images)
    crop_dir = output_dir / "crops"
    records = save_crop_images(records, crop_dir)
    (output_dir / "features").mkdir(parents=True, exist_ok=True)
    write_crop_metadata(output_dir / "crops" / "crop_metadata.csv", records)
    shutil.copy2(output_dir / "crops" / "crop_metadata.csv", output_dir / "features" / "crop_features_meta.csv")
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "image_records.json").write_text(
        json.dumps(image_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, image_records


def read_crop_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    records = []
    for row in rows:
        item = dict(row)
        item["bbox_xyxy"] = json.loads(item["bbox_xyxy"]) if item.get("bbox_xyxy") else []
        item["tags"] = split_text(item.get("tags", ""))
        item["original_width"] = int(float(item.get("original_width") or 0))
        item["original_height"] = int(float(item.get("original_height") or 0))
        item["object_index"] = int(float(item.get("object_index") or 0))
        records.append(item)
    return records
