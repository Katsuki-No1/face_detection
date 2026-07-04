from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
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
HEAD_CATEGORY = {"id": 1, "name": "head", "supercategory": "person"}
FACE_CATEGORY = {
    "id": 1,
    "name": "face",
    "supercategory": "person",
    "keypoints": list(KEYPOINT_NAMES),
    "skeleton": [[1, 3], [2, 3], [3, 4], [3, 5]],
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def area_xywh(box: list[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def clip_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def intersection_xyxy(a: list[float], b: list[float]) -> list[float] | None:
    return clip_xyxy(
        [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])],
        width=int(max(a[2], b[2])),
        height=int(max(a[3], b[3])),
    )


def transform_keypoints_to_crop(
    raw: list[float],
    *,
    crop_left: float,
    crop_top: float,
    crop_width: int,
    crop_height: int,
) -> tuple[list[float], int]:
    transformed: list[float] = []
    visible_count = 0
    for point_index in range(len(KEYPOINT_NAMES)):
        offset = point_index * 3
        if offset + 2 >= len(raw) or int(raw[offset + 2]) <= 0:
            transformed.extend([0.0, 0.0, 0])
            continue
        x = float(raw[offset]) - float(crop_left)
        y = float(raw[offset + 1]) - float(crop_top)
        if 0.0 <= x <= float(crop_width) and 0.0 <= y <= float(crop_height):
            transformed.extend([x, y, int(raw[offset + 2])])
            visible_count += 1
        else:
            transformed.extend([0.0, 0.0, 0])
    return transformed, visible_count


def prepare_head_detector_dataset(source_root: Path, output_root: Path, splits: tuple[str, ...]) -> dict[str, Any]:
    dataset_root = output_root / "data" / "dino_head_detector"
    summary: dict[str, Any] = {"dataset": "dino_head_detector", "splits": {}}
    for split in splits:
        source = read_json(source_root / "mmdet_hf" / f"{split}.json")
        annotations = []
        ann_id = 1
        for ann in source.get("annotations", []):
            if int(ann.get("category_id", 0)) != 1:
                continue
            updated = dict(ann)
            updated["id"] = ann_id
            updated["category_id"] = 1
            updated["metadata"] = {
                **dict(ann.get("metadata") or {}),
                "source_annotation_id": ann.get("id"),
                "source_dataset": "mmdet_hf",
                "source_category_id": ann.get("category_id"),
            }
            annotations.append(updated)
            ann_id += 1
        coco = {
            "info": {
                "description": "DINO-SCRFD single-class head detector dataset derived from mmdet_hf.",
                "date_created": now_text(),
                "image_root": str((source_root / "mmdet_hf").resolve()),
            },
            "licenses": source.get("licenses", []),
            "images": source.get("images", []),
            "annotations": annotations,
            "categories": [HEAD_CATEGORY],
        }
        write_json(dataset_root / f"{split}.json", coco)
        images_with_ann = {int(ann["image_id"]) for ann in annotations}
        summary["splits"][split] = {
            "images": len(coco["images"]),
            "annotations": len(annotations),
            "positive_images": len(images_with_ann),
            "negative_images": len(coco["images"]) - len(images_with_ann),
            "image_root": str((source_root / "mmdet_hf").resolve()),
        }
    return summary


def prepare_headcrop_face_keypoint_dataset(source_root: Path, output_root: Path, splits: tuple[str, ...]) -> dict[str, Any]:
    dataset_root = output_root / "data" / "dino_headcrop_face_keypoint"
    summary: dict[str, Any] = {"dataset": "dino_headcrop_face_keypoint", "splits": {}}
    for split in splits:
        crop_coco = read_json(source_root / "mmdet_headcrop_face" / f"{split}.json")
        face_coco = read_json(source_root / "mmpose_face" / f"{split}.json")
        face_by_source_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in face_coco.get("annotations", []):
            face_by_source_image[int(ann["image_id"])].append(ann)

        annotations: list[dict[str, Any]] = []
        ann_id = 1
        positive_images = set()
        skipped = defaultdict(int)
        for image in crop_coco.get("images", []):
            metadata = dict(image.get("metadata") or {})
            source_image_id = metadata.get("source_image_id")
            crop_xyxy = metadata.get("crop_xyxy")
            if source_image_id is None or not isinstance(crop_xyxy, list) or len(crop_xyxy) != 4:
                skipped["crop_metadata_missing"] += 1
                continue
            crop_left, crop_top, crop_right, crop_bottom = [float(value) for value in crop_xyxy]
            crop_width = int(image["width"])
            crop_height = int(image["height"])
            crop_box_in_source = [crop_left, crop_top, crop_right, crop_bottom]
            for source_ann in face_by_source_image.get(int(source_image_id), []):
                source_face_xyxy = xywh_to_xyxy(source_ann["bbox"])
                overlap = intersection_xyxy(source_face_xyxy, crop_box_in_source)
                if overlap is None:
                    continue
                crop_face_xyxy = [
                    overlap[0] - crop_left,
                    overlap[1] - crop_top,
                    overlap[2] - crop_left,
                    overlap[3] - crop_top,
                ]
                clipped = clip_xyxy(crop_face_xyxy, crop_width, crop_height)
                if clipped is None:
                    skipped["invalid_crop_face_bbox"] += 1
                    continue
                bbox = xyxy_to_xywh(clipped)
                keypoints, num_keypoints = transform_keypoints_to_crop(
                    source_ann.get("keypoints", []),
                    crop_left=crop_left,
                    crop_top=crop_top,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": int(image["id"]),
                        "category_id": 1,
                        "bbox": bbox,
                        "area": area_xywh(bbox),
                        "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": num_keypoints,
                        "metadata": {
                            "source_dataset": "mmpose_face",
                            "source_face_annotation_id": source_ann.get("id"),
                            "source_face_image_id": source_ann.get("image_id"),
                            "source_face_bbox_xyxy": source_face_xyxy,
                            "crop_xyxy": crop_xyxy,
                            "source_face_metadata": source_ann.get("metadata", {}),
                        },
                    }
                )
                positive_images.add(int(image["id"]))
                ann_id += 1
        coco = {
            "info": {
                "description": "DINO-SCRFD head-crop face bbox and five-keypoint dataset.",
                "date_created": now_text(),
                "image_root": str((source_root / "mmdet_headcrop_face").resolve()),
            },
            "licenses": crop_coco.get("licenses", []),
            "images": crop_coco.get("images", []),
            "annotations": annotations,
            "categories": [FACE_CATEGORY],
        }
        write_json(dataset_root / f"{split}.json", coco)
        summary["splits"][split] = {
            "images": len(coco["images"]),
            "annotations": len(annotations),
            "positive_images": len(positive_images),
            "negative_images": len(coco["images"]) - len(positive_images),
            "visible_keypoints": sum(int(ann.get("num_keypoints", 0)) for ann in annotations),
            "image_root": str((source_root / "mmdet_headcrop_face").resolve()),
            "skipped": dict(skipped),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DINOv3 HFDP datasets from existing FDP COCO data.")
    parser.add_argument("--source-root", type=Path, default=Path("outsput/experience/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outsput/experience/dino_hfdp"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = tuple(args.splits)
    source_root = args.source_root.resolve()
    output_root = args.output_dir.resolve()
    required = [
        source_root / "mmdet_hf" / "train.json",
        source_root / "mmdet_headcrop_face" / "train.json",
        source_root / "mmpose_face" / "train.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing FDP source datasets: {missing}")
    summaries = [
        prepare_head_detector_dataset(source_root, output_root, splits),
        prepare_headcrop_face_keypoint_dataset(source_root, output_root, splits),
    ]
    manifest = {
        "created_at": now_text(),
        "source_root": str(source_root),
        "output_dir": str(output_root),
        "splits": list(splits),
        "datasets": summaries,
    }
    write_json(output_root / "data" / "dino_hfdp_data_summary.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
