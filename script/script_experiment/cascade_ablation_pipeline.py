from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.script_experiment.rtmdet_rtmpose_experiment import (  # noqa: E402
    assign_splits,
    bbox_area_xywh,
    bbox_dict_to_xyxy,
    iou_xywh,
    load_records,
    normalized_keypoints,
    repo_rel,
    xyxy_to_xywh,
)
from src.data.face_keypoint_dataset import clip_xyxy as clip_xyxy_or_none  # noqa: E402
from src.data.face_keypoint_dataset import ellipse_to_params, ellipse_to_xyxy  # noqa: E402


SPLITS = ("train", "val", "test")
DEFAULT_OUTPUT_DIR = Path("outputs_experiment/fdp_hfdp_cascade_ablation")
KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
SOURCE_KEYPOINT_NAMES = ("e1", "e2", "n", "m1", "m2")
DETECTION_CATEGORIES = [
    {"id": 1, "name": "head", "supercategory": "person"},
    {"id": 2, "name": "face", "supercategory": "person"},
]
FACE_CATEGORY = {
    "id": 1,
    "name": "face",
    "supercategory": "person",
    "keypoints": list(KEYPOINT_NAMES),
    "skeleton": [[1, 3], [2, 3], [3, 4], [3, 5]],
}
STAGE_DEFINITIONS = [
    {
        "alias": "FDP",
        "legacy_alias": "1stage",
        "source_id": "A",
        "pipeline": "image -> RTMDet-HF(face on full image) -> RTMPose(face)",
        "purpose": "Normal face detection followed by five-point keypoint estimation.",
        "intent": "Face Detection -> Pose baseline.",
        "detector_checkpoint_policy": "Use the full-image RTMDet-HF checkpoint for face detection on the original image.",
        "pose_checkpoint_policy": "Use the same RTMPose-face checkpoint as GTFP, HFDP and GTHFDP.",
    },
    {
        "alias": "HDP",
        "legacy_alias": "2stage",
        "source_id": "C",
        "pipeline": "image -> RTMDet-HF(head on full image) -> head crop -> RTMPose(head)",
        "purpose": "Check whether keypoints can be estimated directly from detected head crops.",
        "intent": "Head Detection -> Pose.",
        "detector_checkpoint_policy": "Use the same full-image RTMDet-HF checkpoint as FDP and HFDP.",
        "pose_checkpoint_policy": "Use RTMPose-head trained/evaluated with head bbox crops.",
    },
    {
        "alias": "GTFP",
        "legacy_alias": "2stage_GT",
        "source_id": "D",
        "pipeline": "GT face bbox -> RTMPose(face)",
        "purpose": "Upper bound of RTMPose-face keypoint estimation with perfect face localization.",
        "intent": "GT Face -> Pose.",
        "detector_checkpoint_policy": "No detector is used; face bbox is ground truth.",
        "pose_checkpoint_policy": "Use the same RTMPose-face checkpoint as FDP, HFDP and GTHFDP.",
    },
    {
        "alias": "HFDP",
        "legacy_alias": "3stage",
        "source_id": "B",
        "pipeline": "image -> RTMDet-HF(head on full image) -> head crop -> RTMDet-headcrop-face(face on crop) -> RTMPose(face)",
        "purpose": "Measure the benefit of adding head detection/cropping before face detection and keypoint estimation.",
        "intent": "Head Detection -> Face Detection -> Pose.",
        "detector_checkpoint_policy": "Use the full-image RTMDet-HF checkpoint for head detection, then a head-crop face detector checkpoint for crop-level face detection.",
        "pose_checkpoint_policy": "Use the same RTMPose-face checkpoint as FDP, GTFP and GTHFDP.",
    },
    {
        "alias": "GTHFDP",
        "legacy_alias": "3stage_GT",
        "source_id": "E",
        "pipeline": "GT head bbox -> head crop -> RTMDet-headcrop-face(face on crop) -> RTMPose(face)",
        "purpose": "Upper bound of HFDP when head localization is perfect.",
        "intent": "GT Head -> Face Detection -> Pose.",
        "detector_checkpoint_policy": "Use no head detector; use the head-crop face detector checkpoint for crop-level face detection.",
        "pose_checkpoint_policy": "Use the same RTMPose-face checkpoint as FDP, GTFP and HFDP.",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def stage_mapping_manifest(output_dir: Path) -> dict[str, Any]:
    return {
        "output_root": str(output_dir),
        "naming_policy": {
            "status": "use_path_abbreviations",
            "canonical_aliases": [row["alias"] for row in STAGE_DEFINITIONS],
            "interpretation": "Use short path abbreviations as experiment identifiers. This avoids ambiguity from counting stages and makes the detector/crop/pose path explicit.",
            "rationale": "The comparison is about whether adding head localization/cropping before face/keypoint estimation improves face and eye accuracy.",
        },
        "stage_aliases": STAGE_DEFINITIONS,
        "legacy_name_notes": {
            "keypoint_gt_bbox": "D / GTFP",
            "two_stage": "A / FDP in the legacy rtmdet_rtmpose_experiment.py naming; do not reuse it as HDP.",
        },
    }


def write_stage_mapping(output_dir: Path) -> None:
    manifest = stage_mapping_manifest(output_dir)
    write_json(output_dir / "stage_mapping.json", manifest)
    write_json(output_dir / "reports" / "stage_mapping.json", manifest)
    rows = [
        {
            "alias": row["alias"],
            "legacy_alias": row["legacy_alias"],
            "source_id": row["source_id"],
            "pipeline": row["pipeline"],
            "purpose": row["purpose"],
        }
        for row in STAGE_DEFINITIONS
    ]
    write_csv(output_dir / "stage_mapping.csv", rows)
    write_csv(output_dir / "reports" / "stage_mapping.csv", rows)
    for row in STAGE_DEFINITIONS:
        alias = str(row["alias"])
        for root_name in ("predictions", "metrics", "visualizations"):
            (output_dir / root_name / alias).mkdir(parents=True, exist_ok=True)


def clip_xyxy(raw: list[float], width: int, height: int) -> list[float] | None:
    return clip_xyxy_or_none([float(value) for value in raw], int(width), int(height))


def bbox_from_shape(shape: dict[str, Any]) -> tuple[list[float] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    ellipse_meta = shape.get("cvat_ellipse")
    if ellipse_meta:
        metadata["bbox_source"] = "ellipse_to_enclosing_bbox"
        metadata["cvat_ellipse"] = ellipse_meta
        metadata["ellipse_params"] = ellipse_to_params(ellipse_meta)
        return ellipse_to_xyxy(ellipse_meta), metadata
    bbox = shape.get("bbox", shape)
    if isinstance(bbox, dict):
        metadata["bbox_source"] = "bbox"
        return bbox_dict_to_xyxy(bbox), metadata
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        metadata["bbox_source"] = "xyxy_list"
        return [float(value) for value in bbox], metadata
    return None, metadata


def object_face_bbox(obj: dict[str, Any]) -> tuple[list[float] | None, dict[str, Any]]:
    ellipse_meta = obj.get("cvat_ellipse") or obj.get("face", {}).get("cvat_ellipse")
    if ellipse_meta:
        return ellipse_to_xyxy(ellipse_meta), {
            "bbox_source": "object_ellipse_to_enclosing_bbox",
            "cvat_ellipse": ellipse_meta,
            "ellipse_params": ellipse_to_params(ellipse_meta),
        }
    bbox = obj.get("training_bbox") or obj.get("raw_bbox") or obj.get("bbox") or obj.get("face", {}).get("bbox")
    if isinstance(bbox, dict):
        return bbox_dict_to_xyxy(bbox), {"bbox_source": "object_bbox"}
    return None, {}


def link_or_copy_image(source: Path, dataset_root: Path, split: str) -> str:
    dest = dataset_root / "images" / split / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            dest.symlink_to(source.resolve())
        except OSError:
            image = Image.open(source).convert("RGB")
            image.save(dest)
    return f"images/{split}/{dest.name}"


def coco_image(record: dict[str, Any], dataset_root: Path) -> dict[str, Any] | None:
    image_abs = Path(record.get("_image_abs", ""))
    if not image_abs.exists():
        return None
    split = str(record["_split"])
    return {
        "id": int(record["_image_id"]),
        "file_name": link_or_copy_image(image_abs, dataset_root, split),
        "width": int(record["_width"]),
        "height": int(record["_height"]),
        "metadata": {
            "source_image": repo_rel(image_abs, REPO_ROOT),
            "annotation_kind": record.get("annotation_kind", ""),
            "image_stem": record.get("image_stem", ""),
            "negative_sample": bool(record.get("negative_sample", False)),
        },
    }


def add_detection_annotation(
    annotations: list[dict[str, Any]],
    *,
    ann_id: int,
    image_id: int,
    category_id: int,
    xyxy: list[float],
    width: int,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    clipped = clip_xyxy(xyxy, width, height)
    if clipped is None:
        return ann_id
    bbox = xyxy_to_xywh(clipped)
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return ann_id
    annotations.append(
        {
            "id": ann_id,
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": bbox,
            "area": bbox_area_xywh(bbox),
            "iscrowd": 0,
            "metadata": metadata or {},
        }
    )
    return ann_id + 1


def convert_mmdet_hf(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    dataset_root = output_dir / "data" / "mmdet_hf"
    summary: dict[str, Any] = {"dataset": "mmdet_hf", "splits": {}}
    for split in SPLITS:
        images = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        class_counts = Counter()
        for record in records:
            if record["_split"] != split:
                continue
            image = coco_image(record, dataset_root)
            if image is None:
                continue
            images.append(image)
            if record.get("negative_sample", False):
                continue
            width, height = int(record["_width"]), int(record["_height"])
            for head_index, head in enumerate(record.get("heads", [])):
                raw, meta = bbox_from_shape(head)
                if raw is None:
                    continue
                ann_id_before = ann_id
                ann_id = add_detection_annotation(
                    annotations,
                    ann_id=ann_id,
                    image_id=record["_image_id"],
                    category_id=1,
                    xyxy=raw,
                    width=width,
                    height=height,
                    metadata={**meta, "source_shape_index": head_index, "label": "head"},
                )
                if ann_id != ann_id_before:
                    class_counts["head"] += 1
            for face_index, face in enumerate(record.get("faces", [])):
                raw, meta = bbox_from_shape(face)
                if raw is None:
                    continue
                ann_id_before = ann_id
                ann_id = add_detection_annotation(
                    annotations,
                    ann_id=ann_id,
                    image_id=record["_image_id"],
                    category_id=2,
                    xyxy=raw,
                    width=width,
                    height=height,
                    metadata={**meta, "source_shape_index": face_index, "label": "face"},
                )
                if ann_id != ann_id_before:
                    class_counts["face"] += 1
        coco = {
            "info": {"description": "RTMDet-HF full-image head/face detection dataset", "date_created": now_text()},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": DETECTION_CATEGORIES,
        }
        write_json(dataset_root / f"{split}.json", coco)
        summary["splits"][split] = {"images": len(images), "annotations": len(annotations), "class_counts": dict(class_counts)}
    return summary


def keypoint_annotation(
    *,
    ann_id: int,
    image_id: int,
    bbox_xyxy: list[float],
    width: int,
    height: int,
    obj: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    clipped = clip_xyxy(bbox_xyxy, width, height)
    if clipped is None:
        return None
    bbox = xyxy_to_xywh(clipped)
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return None
    keypoints, kpt_meta = normalized_keypoints(obj)
    num_keypoints = sum(1 for index in range(2, len(keypoints), 3) if int(keypoints[index]) > 0)
    return {
        "id": ann_id,
        "image_id": int(image_id),
        "category_id": 1,
        "bbox": bbox,
        "area": bbox_area_xywh(bbox),
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "metadata": {**metadata, **kpt_meta},
    }


def head_boxes_for_record(record: dict[str, Any]) -> list[tuple[int, list[float]]]:
    boxes = []
    width, height = int(record["_width"]), int(record["_height"])
    for index, head in enumerate(record.get("heads", [])):
        raw, _meta = bbox_from_shape(head)
        if raw is None:
            continue
        clipped = clip_xyxy(raw, width, height)
        if clipped is not None:
            boxes.append((index, clipped))
    return boxes


def match_head_for_object(record: dict[str, Any], obj: dict[str, Any]) -> tuple[int | None, list[float] | None, str]:
    face_xyxy, _meta = object_face_bbox(obj)
    if face_xyxy is None:
        return None, None, "missing_face_bbox"
    width, height = int(record["_width"]), int(record["_height"])
    face_xyxy = clip_xyxy(face_xyxy, width, height) or face_xyxy
    face_xywh = xyxy_to_xywh(face_xyxy)
    cx = face_xyxy[0] + (face_xyxy[2] - face_xyxy[0]) * 0.5
    cy = face_xyxy[1] + (face_xyxy[3] - face_xyxy[1]) * 0.5
    candidates = head_boxes_for_record(record)
    if not candidates:
        return None, None, "missing_head_bbox"
    containing = [
        (index, box)
        for index, box in candidates
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
    ]
    if containing:
        index, box = min(containing, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
        return index, box, "face_center_inside_head"
    scored = [(iou_xywh(face_xywh, xyxy_to_xywh(box)), index, box) for index, box in candidates]
    scored.sort(reverse=True)
    score, index, box = scored[0]
    if score > 0.0:
        return index, box, "best_iou_head"
    return None, None, "no_overlapping_head"


def convert_mmpose_dataset(records: list[dict[str, Any]], output_dir: Path, *, bbox_source: str) -> dict[str, Any]:
    assert bbox_source in {"face", "head"}
    dataset_root = output_dir / "data" / f"mmpose_{bbox_source}"
    summary: dict[str, Any] = {"dataset": f"mmpose_{bbox_source}", "splits": {}, "skipped": Counter()}
    for split in SPLITS:
        images = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        visible_counts = Counter()
        for record in records:
            if record["_split"] != split:
                continue
            image = coco_image(record, dataset_root)
            if image is None:
                continue
            images.append(image)
            width, height = int(record["_width"]), int(record["_height"])
            for obj_index, obj in enumerate(record.get("objects", [])):
                face_xyxy, face_meta = object_face_bbox(obj)
                if face_xyxy is None:
                    summary["skipped"]["missing_face_bbox"] += 1
                    continue
                if bbox_source == "face":
                    bbox_xyxy = face_xyxy
                    source_meta = {**face_meta, "bbox_role": "gt_face_bbox", "source_object_index": obj_index}
                else:
                    head_index, head_xyxy, policy = match_head_for_object(record, obj)
                    if head_xyxy is None:
                        summary["skipped"][policy] += 1
                        continue
                    bbox_xyxy = head_xyxy
                    source_meta = {
                        **face_meta,
                        "bbox_role": "gt_head_bbox",
                        "matched_head_index": head_index,
                        "head_match_policy": policy,
                        "source_object_index": obj_index,
                        "face_bbox_xyxy": face_xyxy,
                    }
                ann = keypoint_annotation(
                    ann_id=ann_id,
                    image_id=record["_image_id"],
                    bbox_xyxy=bbox_xyxy,
                    width=width,
                    height=height,
                    obj=obj,
                    metadata=source_meta,
                )
                if ann is None:
                    summary["skipped"]["invalid_bbox"] += 1
                    continue
                annotations.append(ann)
                for offset, name in enumerate(KEYPOINT_NAMES):
                    visible_counts[f"{name}_v{int(ann['keypoints'][offset * 3 + 2])}"] += 1
                ann_id += 1
        coco = {
            "info": {
                "description": f"RTMPose {bbox_source} top-down five-keypoint dataset",
                "date_created": now_text(),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [FACE_CATEGORY],
        }
        write_json(dataset_root / f"{split}.json", coco)
        summary["splits"][split] = {
            "images": len(images),
            "annotations": len(annotations),
            "keypoint_visibility_counts": dict(visible_counts),
        }
    summary["skipped"] = dict(summary["skipped"])
    return summary


def expand_box(xyxy: list[float], width: int, height: int, margin: float) -> list[int] | None:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0.0 or bh <= 0.0:
        return None
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    new_w = bw * margin
    new_h = bh * margin
    left = max(0, int(math.floor(cx - new_w * 0.5)))
    top = max(0, int(math.floor(cy - new_h * 0.5)))
    right = min(int(width), int(math.ceil(cx + new_w * 0.5)))
    bottom = min(int(height), int(math.ceil(cy + new_h * 0.5)))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def intersection_xyxy(a: list[float], b: list[float]) -> list[float] | None:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def convert_headcrop_detector(records: list[dict[str, Any]], output_dir: Path, *, margin: float = 1.20) -> dict[str, Any]:
    dataset_root = output_dir / "data" / "mmdet_headcrop_face"
    summary: dict[str, Any] = {"dataset": "mmdet_headcrop_face", "margin": margin, "splits": {}}
    for split in SPLITS:
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        image_id = 1
        negative_crops = 0
        for record in records:
            if record["_split"] != split:
                continue
            image_abs = Path(record.get("_image_abs", ""))
            if not image_abs.exists():
                continue
            width, height = int(record["_width"]), int(record["_height"])
            head_boxes = head_boxes_for_record(record)
            if not head_boxes:
                continue
            source_image = Image.open(image_abs).convert("RGB")
            face_boxes = []
            if not record.get("negative_sample", False):
                for face_index, face in enumerate(record.get("faces", [])):
                    raw, meta = bbox_from_shape(face)
                    if raw is None:
                        continue
                    clipped = clip_xyxy(raw, width, height)
                    if clipped is not None:
                        face_boxes.append((face_index, clipped, meta))
            for head_index, head_xyxy in head_boxes:
                crop_xyxy = expand_box(head_xyxy, width, height, margin)
                if crop_xyxy is None:
                    continue
                crop_left, crop_top, crop_right, crop_bottom = crop_xyxy
                crop = source_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                crop_name = f"{int(record['_image_id']):06d}_head{head_index:02d}_{Path(image_abs).stem}.jpg"
                crop_rel = f"images/{split}/{crop_name}"
                crop_path = dataset_root / crop_rel
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=95)
                crop_w, crop_h = crop.size
                images.append(
                    {
                        "id": image_id,
                        "file_name": crop_rel,
                        "width": crop_w,
                        "height": crop_h,
                        "metadata": {
                            "source_image_id": int(record["_image_id"]),
                            "source_image": repo_rel(image_abs, REPO_ROOT),
                            "head_index": head_index,
                            "head_bbox_xyxy": head_xyxy,
                            "crop_xyxy": crop_xyxy,
                            "margin": margin,
                        },
                    }
                )
                crop_annotations_before = len(annotations)
                for face_index, face_xyxy, meta in face_boxes:
                    overlap = intersection_xyxy(face_xyxy, crop_xyxy)
                    if overlap is None:
                        continue
                    crop_face_xyxy = [
                        overlap[0] - crop_left,
                        overlap[1] - crop_top,
                        overlap[2] - crop_left,
                        overlap[3] - crop_top,
                    ]
                    ann_id = add_detection_annotation(
                        annotations,
                        ann_id=ann_id,
                        image_id=image_id,
                        category_id=2,
                        xyxy=crop_face_xyxy,
                        width=crop_w,
                        height=crop_h,
                        metadata={
                            **meta,
                            "source_face_index": face_index,
                            "source_face_bbox_xyxy": face_xyxy,
                            "crop_xyxy": crop_xyxy,
                        },
                    )
                if len(annotations) == crop_annotations_before:
                    negative_crops += 1
                image_id += 1
        coco = {
            "info": {
                "description": "Head-crop face detection dataset for optional RTMDet-HF augmentation",
                "date_created": now_text(),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": DETECTION_CATEGORIES,
        }
        write_json(dataset_root / f"{split}.json", coco)
        summary["splits"][split] = {
            "images": len(images),
            "annotations": len(annotations),
            "negative_crops": negative_crops,
        }
    return summary


def draw_box(draw: ImageDraw.ImageDraw, bbox: list[float], color: tuple[int, int, int], text: str = "") -> None:
    x, y, w, h = [float(value) for value in bbox]
    draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
    if text:
        draw.text((x + 3, max(0.0, y + 3)), text, fill=color)


def draw_keypoints(draw: ImageDraw.ImageDraw, keypoints: list[float]) -> None:
    colors = [(255, 80, 80), (80, 160, 255), (255, 220, 0), (255, 140, 0), (160, 80, 255)]
    for index, name in enumerate(SOURCE_KEYPOINT_NAMES):
        x = float(keypoints[index * 3])
        y = float(keypoints[index * 3 + 1])
        v = int(keypoints[index * 3 + 2])
        if v <= 0:
            continue
        color = colors[index]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x + 5, y + 5), name, fill=color)


def render_dataset_overlays(dataset_root: Path, vis_dir: Path, *, max_images: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for split in SPLITS:
        json_path = dataset_root / f"{split}.json"
        if not json_path.exists():
            continue
        coco = read_json(json_path)
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)
        images = list(coco.get("images", []))
        rng.shuffle(images)
        for image in images[:max_images]:
            image_path = dataset_root / image["file_name"]
            if not image_path.exists():
                continue
            canvas = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for ann in anns_by_image.get(int(image["id"]), []):
                if "keypoints" in ann:
                    draw_box(draw, ann["bbox"], (40, 220, 80), "bbox")
                    draw_keypoints(draw, ann["keypoints"])
                else:
                    label = "head" if int(ann["category_id"]) == 1 else "face"
                    color = (40, 220, 80) if label == "head" else (80, 160, 255)
                    draw_box(draw, ann["bbox"], color, label)
            out_path = vis_dir / split / f"{Path(image['file_name']).stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_path, quality=95)
            rows.append(
                {
                    "split": split,
                    "source": str(image_path),
                    "overlay": str(out_path),
                    "width": image["width"],
                    "height": image["height"],
                    "annotation_count": len(anns_by_image.get(int(image["id"]), [])),
                }
            )
    write_csv(vis_dir / "visualization_index.csv", rows)
    return rows


def visualize_converted_data(output_dir: Path, *, max_images: int = 50, seed: int = 42) -> dict[str, int]:
    specs = [
        ("mmdet_hf", "check_mmdet_hf"),
        ("mmpose_face", "check_mmpose_face"),
        ("mmpose_head", "check_mmpose_head"),
        ("mmdet_headcrop_face", "check_headcrop"),
    ]
    counts = {}
    for dataset_name, vis_name in specs:
        rows = render_dataset_overlays(
            output_dir / "data" / dataset_name,
            output_dir / "vis" / vis_name,
            max_images=max_images,
            seed=seed,
        )
        counts[dataset_name] = len(rows)
    return counts


def mmpose_meta_config() -> str:
    return """dataset_info = dict(
    dataset_name='face5',
    paper_info=dict(author='local', title='Face five-point dataset', container='local', year='2026', homepage=''),
    keypoint_info={
        0: dict(name='left_eye', id=0, color=[255, 80, 80], type='upper', swap='right_eye'),
        1: dict(name='right_eye', id=1, color=[80, 160, 255], type='upper', swap='left_eye'),
        2: dict(name='nose_tip', id=2, color=[255, 220, 0], type='upper', swap=''),
        3: dict(name='left_mouth_corner', id=3, color=[255, 140, 0], type='upper', swap='right_mouth_corner'),
        4: dict(name='right_mouth_corner', id=4, color=[160, 80, 255], type='upper', swap='left_mouth_corner'),
    },
    skeleton_info={
        0: dict(link=('left_eye', 'nose_tip'), id=0, color=[255, 220, 0]),
        1: dict(link=('right_eye', 'nose_tip'), id=1, color=[255, 220, 0]),
        2: dict(link=('nose_tip', 'left_mouth_corner'), id=2, color=[255, 140, 0]),
        3: dict(link=('nose_tip', 'right_mouth_corner'), id=3, color=[160, 80, 255]),
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.025, 0.025, 0.026, 0.035, 0.035])
"""


def generate_mmdet_config(output_dir: Path) -> None:
    data_root = (output_dir / "data" / "mmdet_hf").resolve().as_posix() + "/"
    work_dir = (output_dir / "work_dirs" / "mmdet" / "rtmdet_hf_full").resolve().as_posix()
    cspnext_checkpoint = (
        REPO_ROOT
        / "output_experiment"
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "cspnext-s_imagenet_600e.pth"
    ).resolve().as_posix()
    text = f"""custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)
default_scope = 'mmdet'
_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = '{data_root}'
work_dir = '{work_dir}'
classes = ('head', 'face')
metainfo = dict(classes=classes, palette=[(40, 220, 80), (80, 160, 255)])
num_classes = 2
img_scale = (960, 960)
seed = 42
randomness = dict(seed=seed, deterministic=False)

model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
    backbone=dict(init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='{cspnext_checkpoint}')),
    bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.75, 1.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_cfg = dict(max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json', metric='bbox')
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox')
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/bbox_mAP_50', rule='greater', max_keep_ckpts=2),
    logger=dict(interval=50))
"""
    write_text(output_dir / "configs" / "mmdet" / "rtmdet_hf_full.py", text)

    smoke_text = text.replace("img_scale = (960, 960)", "img_scale = (320, 320)")
    smoke_text = smoke_text.replace("train_cfg = dict(max_epochs=100, val_interval=1)", "train_cfg = dict(max_epochs=1, val_interval=1)")
    smoke_text = smoke_text.replace("batch_size=8", "batch_size=1").replace("batch_size=4", "batch_size=1")
    smoke_text = smoke_text.replace("num_workers=4", "num_workers=0").replace("num_workers=2", "num_workers=0")
    smoke_text = smoke_text.replace("persistent_workers=True", "persistent_workers=False")
    smoke_text = smoke_text.replace("rtmdet_hf_full", "rtmdet_hf_cpu_smoke")
    write_text(output_dir / "configs" / "mmdet" / "rtmdet_hf_cpu_smoke.py", smoke_text)


def generate_mmdet_headcrop_face_config(output_dir: Path) -> None:
    data_root = (output_dir / "data" / "mmdet_headcrop_face").resolve().as_posix() + "/"
    work_dir = (output_dir / "work_dirs" / "mmdet" / "rtmdet_headcrop_face").resolve().as_posix()
    cspnext_checkpoint = (
        REPO_ROOT
        / "output_experiment"
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "cspnext-s_imagenet_600e.pth"
    ).resolve().as_posix()
    text = f"""custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)
default_scope = 'mmdet'
_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = '{data_root}'
work_dir = '{work_dir}'
classes = ('head', 'face')
metainfo = dict(classes=classes, palette=[(40, 220, 80), (80, 160, 255)])
num_classes = 2
img_scale = (320, 320)
seed = 42
randomness = dict(seed=seed, deterministic=False)

model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
    backbone=dict(init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='{cspnext_checkpoint}')),
    bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.75, 1.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_cfg = dict(max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json', metric='bbox')
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox')
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/bbox_mAP_50', rule='greater', max_keep_ckpts=2),
    logger=dict(interval=50))
"""
    write_text(output_dir / "configs" / "mmdet" / "rtmdet_headcrop_face.py", text)

    smoke_text = text.replace("train_cfg = dict(max_epochs=100, val_interval=1)", "train_cfg = dict(max_epochs=1, val_interval=1)")
    smoke_text = smoke_text.replace("batch_size=8", "batch_size=1").replace("batch_size=4", "batch_size=1")
    smoke_text = smoke_text.replace("num_workers=4", "num_workers=0").replace("num_workers=2", "num_workers=0")
    smoke_text = smoke_text.replace("persistent_workers=True", "persistent_workers=False")
    smoke_text = smoke_text.replace("rtmdet_headcrop_face", "rtmdet_headcrop_face_cpu_smoke")
    write_text(output_dir / "configs" / "mmdet" / "rtmdet_headcrop_face_cpu_smoke.py", smoke_text)


def generate_mmpose_config(output_dir: Path, *, bbox_source: str) -> None:
    assert bbox_source in {"face", "head"}
    data_root = (output_dir / "data" / f"mmpose_{bbox_source}").resolve().as_posix() + "/"
    work_dir = (output_dir / "work_dirs" / "mmpose" / f"rtmpose_{bbox_source}").resolve().as_posix()
    meta_file = (output_dir / "configs" / "mmpose" / "face5_dataset_meta.py").resolve().as_posix()
    cspnext_checkpoint = (
        REPO_ROOT
        / "output_experiment"
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "cspnext-s_imagenet_600e-ea671761.pth"
    ).resolve().as_posix()
    text = f"""_base_ = ['mmpose::_base_/default_runtime.py']
custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)
default_scope = 'mmpose'

num_keypoints = 5
input_size = (256, 256)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

max_epochs = 100
base_lr = 4e-3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=42, deterministic=False)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.005, begin=30, end=max_epochs, T_max=max_epochs - 30, by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=512)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='{cspnext_checkpoint}')),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0., drop_path=0., act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        loss=dict(type='KLDiscretLoss', use_target_weight=True, beta=10., label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '{data_root}'
metainfo = dict(from_file='{meta_file}')
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform', scale_factor=[0.75, 1.25], rotate_factor=20),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = dict(type='NME', norm_mode='use_norm_item', norm_item='bbox_size')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less', max_keep_ckpts=2, interval=1), logger=dict(interval=50))
work_dir = '{work_dir}'
"""
    write_text(output_dir / "configs" / "mmpose" / f"rtmpose_{bbox_source}.py", text)
    smoke_text = text.replace("batch_size=32", "batch_size=1").replace("batch_size=16", "batch_size=1")
    smoke_text = smoke_text.replace("num_workers=4", "num_workers=0").replace("num_workers=2", "num_workers=0")
    smoke_text = smoke_text.replace("persistent_workers=True", "persistent_workers=False")
    smoke_text = smoke_text.replace("max_epochs = 100", "max_epochs = 1")
    smoke_text = smoke_text.replace(f"rtmpose_{bbox_source}", f"rtmpose_{bbox_source}_cpu_smoke")
    write_text(output_dir / "configs" / "mmpose" / f"rtmpose_{bbox_source}_cpu_smoke.py", smoke_text)


def generate_configs(output_dir: Path) -> None:
    write_text(output_dir / "configs" / "mmpose" / "face5_dataset_meta.py", mmpose_meta_config())
    generate_mmdet_config(output_dir)
    generate_mmdet_headcrop_face_config(output_dir)
    generate_mmpose_config(output_dir, bbox_source="face")
    generate_mmpose_config(output_dir, bbox_source="head")


def validate_coco_dataset(dataset_root: Path, split: str, *, keypoints: bool) -> dict[str, Any]:
    coco = read_json(dataset_root / f"{split}.json")
    images = {int(image["id"]): image for image in coco.get("images", [])}
    issues: list[str] = []
    class_counts = Counter()
    for image in images.values():
        path = dataset_root / image["file_name"]
        if not path.exists():
            issues.append(f"missing image: {path}")
            continue
        with Image.open(path) as opened:
            if (opened.width, opened.height) != (int(image["width"]), int(image["height"])):
                issues.append(f"size mismatch: {path} json={(image['width'], image['height'])} actual={opened.size}")
    for ann in coco.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if image is None:
            issues.append(f"annotation {ann.get('id')} references missing image {ann.get('image_id')}")
            continue
        x, y, w, h = [float(value) for value in ann.get("bbox", [0, 0, 0, 0])]
        values = [x, y, w, h]
        if not all(math.isfinite(value) for value in values):
            issues.append(f"annotation {ann.get('id')} bbox has non-finite values")
        if w <= 0.0 or h <= 0.0:
            issues.append(f"annotation {ann.get('id')} bbox has non-positive size")
        if x < -1e-3 or y < -1e-3 or x + w > float(image["width"]) + 1e-3 or y + h > float(image["height"]) + 1e-3:
            issues.append(f"annotation {ann.get('id')} bbox outside image")
        class_counts[str(ann.get("category_id"))] += 1
        if keypoints:
            raw = ann.get("keypoints", [])
            if len(raw) != len(KEYPOINT_NAMES) * 3:
                issues.append(f"annotation {ann.get('id')} keypoint length is {len(raw)}")
                continue
            for index in range(len(KEYPOINT_NAMES)):
                px, py, vis = float(raw[index * 3]), float(raw[index * 3 + 1]), int(raw[index * 3 + 2])
                if not all(math.isfinite(value) for value in [px, py]):
                    issues.append(f"annotation {ann.get('id')} keypoint {index} has non-finite values")
                if vis > 0 and (px < -1e-3 or py < -1e-3 or px > float(image["width"]) + 1e-3 or py > float(image["height"]) + 1e-3):
                    issues.append(f"annotation {ann.get('id')} visible keypoint {index} outside image")
    return {
        "split": split,
        "images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "class_counts": dict(class_counts),
        "issues": issues[:50],
        "issue_count": len(issues),
    }


def smoke_openmmlab_dataset(config_path: Path, *, scope: str) -> dict[str, Any]:
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope

        cfg = Config.fromfile(config_path.as_posix())
        init_default_scope(scope)
        if scope == "mmdet":
            import mmdet.datasets  # noqa: F401
            from mmdet.registry import DATASETS
        else:
            import mmpose.datasets  # noqa: F401
            from mmpose.registry import DATASETS
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        sample = dataset[0] if len(dataset) else None
        sample_type = type(sample).__name__ if sample is not None else "empty"
        return {"config": str(config_path), "scope": scope, "status": "ok", "dataset_len": len(dataset), "sample_type": sample_type}
    except Exception as exc:
        return {"config": str(config_path), "scope": scope, "status": "failed", "error": repr(exc)}


def run_smoke(output_dir: Path) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    dataset_checks = []
    for name, keypoints in [
        ("mmdet_hf", False),
        ("mmpose_face", True),
        ("mmpose_head", True),
        ("mmdet_headcrop_face", False),
    ]:
        root = output_dir / "data" / name
        for split in SPLITS:
            dataset_checks.append(validate_coco_dataset(root, split, keypoints=keypoints))
    config_checks = [
        smoke_openmmlab_dataset(output_dir / "configs" / "mmdet" / "rtmdet_hf_cpu_smoke.py", scope="mmdet"),
        smoke_openmmlab_dataset(output_dir / "configs" / "mmdet" / "rtmdet_headcrop_face_cpu_smoke.py", scope="mmdet"),
        smoke_openmmlab_dataset(output_dir / "configs" / "mmpose" / "rtmpose_face_cpu_smoke.py", scope="mmpose"),
        smoke_openmmlab_dataset(output_dir / "configs" / "mmpose" / "rtmpose_head_cpu_smoke.py", scope="mmpose"),
    ]
    result = {
        "created_at": now_text(),
        "device": "cpu",
        "training_executed": False,
        "dataset_checks": dataset_checks,
        "config_dataset_build_checks": config_checks,
        "status": "ok"
        if all(row["issue_count"] == 0 for row in dataset_checks) and all(row["status"] == "ok" for row in config_checks)
        else "failed",
    }
    write_json(output_dir / "reports" / "smoke_status.json", result)
    return result


def write_reports(output_dir: Path, summaries: list[dict[str, Any]], split_summary: dict[str, Any], vis_counts: dict[str, int]) -> None:
    write_json(output_dir / "reports" / "conversion_summary.json", {"split_summary": split_summary, "datasets": summaries, "visualizations": vis_counts})
    output_dir_arg = repo_rel(output_dir, REPO_ROOT)
    dino_output_arg = "outputs/dino_scrfd_no_aug_cpu_overlay/test_best_loss"
    lines = [
        "# Cascade Ablation RTMDet + RTMPose",
        "",
        "このディレクトリは、FDP系列の多段階推論比較に向けたデータ変換・config・smoke結果をまとめる場所です。",
        "今回の実行では学習は行っていません。CPU smoke はデータとOpenMMLab dataset buildまでです。",
        "",
        "## Canonical Experiment IDs",
        "",
        "| Alias | Legacy | Source | Pipeline |",
        "|---|---|---|---|",
    ]
    for row in STAGE_DEFINITIONS:
        lines.append(f"| `{row['alias']}` | `{row['legacy_alias']}` | {row['source_id']} | {row['pipeline']} |")
    lines.extend(
        [
            "",
            "既存 `rtmdet_rtmpose_experiment.py` の `two_stage` は、この命名では `FDP` 相当です。`HDP` と混同しないよう、最終比較ではFDP系列の出力名に正規化します。",
        "",
        "## 生成ファイル",
        "",
        "- `data/mmdet_hf/{train,val,test}.json`: full image の head/face RTMDet 用COCO。",
        "- `data/mmpose_face/{train,val,test}.json`: GT face bbox を使う RTMPose-face 用COCO keypoint。",
        "- `data/mmpose_head/{train,val,test}.json`: GT head bbox を使う RTMPose-head 用COCO keypoint。",
        "- `data/mmdet_headcrop_face/{train,val,test}.json`: 1.20倍 head crop 内 face detection 用COCO。",
        "- `configs/mmdet/rtmdet_hf_full.py`: GPU本学習用 RTMDet-s head/face config。",
        "- `configs/mmdet/rtmdet_headcrop_face.py`: GPU本学習用 head crop 内 face detector config。",
        "- `configs/mmpose/rtmpose_face.py`: GPU本学習用 RTMPose-face config。",
        "- `configs/mmpose/rtmpose_head.py`: GPU本学習用 RTMPose-head config。",
        "- `configs/*/*_cpu_smoke.py`: CPU smoke 用の軽量config。",
        "- `stage_mapping.json` / `stage_mapping.csv`: FDP系列のcanonical実験ID定義。",
        "- `vis/check_*`: 変換後annotationのoverlay。",
        "- `reports/smoke_status.json`: CPU smoke結果。",
        "",
        "## 実行コマンド",
        "",
        "```bash",
        f".venv/bin/python script/script_experiment/cascade_ablation_pipeline.py prepare --output-dir {output_dir_arg}",
        f".venv/bin/python script/script_experiment/cascade_ablation_pipeline.py smoke --output-dir {output_dir_arg}",
        f".venv/bin/python script/script_experiment/cascade_ablation_pipeline.py verify-dino-overlay --output-dir {output_dir_arg} --dino-output {dino_output_arg}",
        "```",
        "",
        "## GPU本学習コマンド",
        "",
        "GPUが空いたら以下を順に実行します。HFDP/GTHFDPでは full-image head detector と head-crop face detector を分けて使い、FDP/GTFP/HFDP/GTHFDPでは同じ `rtmpose_face` checkpointを使います。",
        "",
        "```bash",
        f"CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --nproc_per_node=1 .venv/lib/python3.12/site-packages/mmdet/.mim/tools/train.py {output_dir_arg}/configs/mmdet/rtmdet_hf_full.py",
        f"CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --nproc_per_node=1 .venv/lib/python3.12/site-packages/mmdet/.mim/tools/train.py {output_dir_arg}/configs/mmdet/rtmdet_headcrop_face.py",
        f"CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --nproc_per_node=1 .venv/lib/python3.12/site-packages/mmpose/.mim/tools/train.py {output_dir_arg}/configs/mmpose/rtmpose_face.py",
        f"CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --nproc_per_node=1 .venv/lib/python3.12/site-packages/mmpose/.mim/tools/train.py {output_dir_arg}/configs/mmpose/rtmpose_head.py",
        "```",
        "",
        "## Split Summary",
        "",
        f"- Method: `{split_summary.get('method')}`",
        f"- Images by split: `{split_summary.get('images_by_split')}`",
        "",
        "## Dataset Counts",
        "",
    ]
    )
    for summary in summaries:
        lines.append(f"### {summary['dataset']}")
        for split, values in summary.get("splits", {}).items():
            lines.append(f"- {split}: {values}")
        if summary.get("skipped"):
            lines.append(f"- skipped: {summary['skipped']}")
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- 既存metadataの train/val/test split を維持する。splitがなければ既存ヘルパーのgroup splitを使うが、今回のmetadataでは既存splitを再利用した。",
            "- `e1/e2` と `m1/m2` は既存実装に合わせ、両点がある場合は画像x座標でleft/rightへ正規化する。",
            "- RTMPose-head の bbox は、face bbox中心を含む最小head bboxを優先し、なければface bboxとのIoUが最大のhead bboxを使う。",
            "- HDPはhead crop入力の別checkpointなので、FDP/GTFP/HFDP/GTHFDPとの厳密な同一pose checkpoint比較ではなく補助実験として扱う。",
            "- flipは初期configでは無効化した。左右flipを入れる場合は `left_eye/right_eye` と `left_mouth_corner/right_mouth_corner` のflip_idxを目視確認してから有効化する。",
            "- CPU smokeでは学習ループ・optimizer step・checkpoint作成は行わない。",
        ]
    )
    write_text(output_dir / "README.md", "\n".join(lines))


def prepare(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    records, _metadata, metadata_path = load_records(REPO_ROOT, args.metadata)
    records, split_summary = assign_splits(records)
    write_stage_mapping(output_dir)
    summaries = [
        convert_mmdet_hf(records, output_dir),
        convert_mmpose_dataset(records, output_dir, bbox_source="face"),
        convert_mmpose_dataset(records, output_dir, bbox_source="head"),
        convert_headcrop_detector(records, output_dir, margin=args.crop_margin),
    ]
    generate_configs(output_dir)
    vis_counts = visualize_converted_data(output_dir, max_images=args.max_visualizations, seed=args.seed)
    write_reports(output_dir, summaries, split_summary, vis_counts)
    write_json(
        output_dir / "reports" / "prepare_status.json",
        {
            "status": "ok",
            "created_at": now_text(),
            "metadata": str(metadata_path),
            "output_dir": str(output_dir),
            "training_executed": False,
        },
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "training_executed": False}, ensure_ascii=False, indent=2))
    return 0


def verify_dino_overlay(args: argparse.Namespace) -> int:
    index_path = args.dino_output / "visualization_index.csv"
    rows: list[dict[str, Any]] = []
    mismatches = []
    if not index_path.exists():
        raise FileNotFoundError(f"visualization_index.csv not found: {index_path}")
    with index_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            image_path = Path(row.get("image", row.get("image_path", "")))
            overlay_path = Path(row.get("visualization", row.get("overlay", "")))
            if not image_path.exists() or not overlay_path.exists():
                rows.append({"image": str(image_path), "overlay": str(overlay_path), "status": "missing_file"})
                mismatches.append(str(overlay_path))
                continue
            with Image.open(image_path) as image, Image.open(overlay_path) as overlay:
                match = image.size == overlay.size
                item = {
                    "image": str(image_path),
                    "overlay": str(overlay_path),
                    "image_size": image.size,
                    "overlay_size": overlay.size,
                    "match": match,
                }
                rows.append(item)
                if not match:
                    mismatches.append(str(overlay_path))
    result = {
        "created_at": now_text(),
        "dino_output": str(args.dino_output),
        "checked": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "status": "ok" if not mismatches else "failed",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "reports" / "dino_overlay_size_check.json", result)
    write_csv(args.output_dir / "reports" / "dino_overlay_size_check.csv", rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


def smoke(args: argparse.Namespace) -> int:
    result = run_smoke(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RTMDet + RTMPose cascade ablation data/configs and run CPU smoke checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Convert datasets, generate configs, and render annotation overlays.")
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--metadata", type=Path, default=Path("datasets/face_pose/metadata/conversion_metadata.json"))
    prepare_parser.add_argument("--crop-margin", type=float, default=1.20)
    prepare_parser.add_argument("--max-visualizations", type=int, default=50)
    prepare_parser.add_argument("--seed", type=int, default=42)

    smoke_parser = subparsers.add_parser("smoke", help="Run CPU-only dataset/config smoke checks without training.")
    smoke_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    dino_parser = subparsers.add_parser("verify-dino-overlay", help="Verify DINO overlay files are saved at original image size.")
    dino_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    dino_parser.add_argument("--dino-output", type=Path, default=Path("outputs/dino_scrfd_no_aug_cpu_overlay/test_best_loss"))

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "verify-dino-overlay":
        return verify_dino_overlay(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
