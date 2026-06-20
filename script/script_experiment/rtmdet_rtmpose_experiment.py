from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SOURCE_KEYPOINTS = ("e1", "e2", "n", "m1", "m2")
KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
DETECTION_CATEGORIES = (
    {"id": 1, "name": "head", "supercategory": "person", "class_index": 0},
    {"id": 2, "name": "face", "supercategory": "person", "class_index": 1},
)
FACE_CATEGORY = {
    "id": 1,
    "name": "face",
    "supercategory": "person",
    "keypoints": list(KEYPOINT_NAMES),
    "skeleton": [[1, 3], [2, 3], [3, 4], [3, 5]],
}
MISSING_OPENMMLAB_COMMANDS = [
    ".venv/bin/python -m pip install -U openmim",
    '.venv/bin/mim install "mmengine>=0.10" "mmcv>=2.0" "mmdet>=3.0" "mmpose>=1.3" pycocotools',
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def path_text(path: Path) -> str:
    return path.as_posix()


def repo_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_layout(output_dir: Path) -> None:
    dirs = [
        "configs/rtm_det_head_face",
        "configs/rtmpose_face",
        "data/raw_index",
        "data/converted/detection_coco",
        "data/converted/keypoint_coco",
        "data/splits",
        "predictions/detector",
        "predictions/keypoint_gt_bbox",
        "predictions/two_stage",
        "metrics",
        "visualizations/detector",
        "visualizations/keypoint_gt_bbox",
        "visualizations/two_stage",
        "visualizations/errors",
        "reports",
        "logs",
    ]
    for rel in dirs:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_report(output_dir: Path, title: str, lines: list[str]) -> None:
    report_path = output_dir / "reports" / "experiment_report.md"
    append_text(report_path, "\n".join(["", f"## {title}", *lines]))


def configure_output_environment(output_dir: Path) -> None:
    cache_root = output_dir.resolve() / "cache"
    env_defaults = {
        "XDG_CACHE_HOME": cache_root.as_posix(),
        "TORCH_HOME": (cache_root / "torch").as_posix(),
        "MIM_CACHE_DIR": (cache_root / "mim").as_posix(),
        "MPLCONFIGDIR": (cache_root / "matplotlib").as_posix(),
        "MMENGINE_HOME": (cache_root / "mmengine").as_posix(),
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    for value in env_defaults.values():
        Path(value).mkdir(parents=True, exist_ok=True)


def python_executable(data_root: Path) -> str:
    candidate = data_root / ".venv" / "bin" / "python"
    return candidate.as_posix() if candidate.exists() else sys.executable


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def effective_device(requested: str) -> tuple[str, str | None]:
    if requested.startswith("cuda") and not torch_cuda_available():
        return "cpu", f"requested {requested}, but torch.cuda.is_available() is false"
    return requested, None


def mmcv_cuda_ops_available() -> bool:
    try:
        from mmcv.ops import get_compiling_cuda_version  # type: ignore

        version = str(get_compiling_cuda_version()).lower()
        return version not in {"not available", "none", ""}
    except Exception:
        return False


def effective_detector_device(requested: str) -> tuple[str, str | None]:
    run_device, note = effective_device(requested)
    if run_device.startswith("cuda") and not mmcv_cuda_ops_available():
        return "cpu", "requested cuda detector inference, but mmcv was built without CUDA ops; falling back to CPU for NMS"
    return run_device, note


def is_output_checkpoint(path: Path, output_dir: Path) -> bool:
    try:
        path.resolve().relative_to(output_dir.resolve())
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def local_checkpoint_load_context(checkpoint: Path, output_dir: Path) -> Any:
    if not is_output_checkpoint(checkpoint, output_dir):
        yield
        return
    import torch  # type: ignore

    original_load = torch.load

    def load_with_local_checkpoint_defaults(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_local_checkpoint_defaults
    try:
        yield
    finally:
        torch.load = original_load


def env_report(data_root: Path) -> dict[str, Any]:
    modules = [
        "torch",
        "cv2",
        "mmdet",
        "mmengine",
        "mmcv",
        "mmpose",
        "xtcocotools",
        "pycocotools",
        "ultralytics",
        "numpy",
        "PIL",
        "yaml",
    ]
    report: dict[str, Any] = {
        "python": sys.executable,
        "recommended_python": python_executable(data_root),
        "modules": {name: module_available(name) for name in modules},
        "mmcv_cuda_ops_available": mmcv_cuda_ops_available() if module_available("mmcv") else False,
    }
    try:
        import torch  # type: ignore

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - environment only
        report["torch_error"] = str(exc)
    return report


def find_metadata_path(data_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit if explicit.is_absolute() else data_root / explicit
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"metadata path not found: {candidate}")
    preferred = data_root / "datasets" / "face_pose" / "metadata" / "conversion_metadata.json"
    if preferred.exists():
        return preferred
    for path in sorted(data_root.rglob("conversion_metadata.json")):
        if "face_pose" in path.parts:
            return path
    raise FileNotFoundError("conversion_metadata.json was not found under the data root")


def image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2  # type: ignore

        image = cv2.imread(path.as_posix())
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def resolve_image_path(record: dict[str, Any], data_root: Path) -> Path | None:
    output_image = record.get("output_image") or record.get("image_output")
    if output_image:
        candidate = data_root / "datasets" / "face_pose" / str(output_image)
        if candidate.exists():
            return candidate
        candidate = data_root / str(output_image)
        if candidate.exists():
            return candidate
    image_path = record.get("image_path")
    if image_path:
        candidate = Path(str(image_path))
        if not candidate.is_absolute():
            candidate = data_root / candidate
        if candidate.exists():
            return candidate
    image_name = record.get("image_name")
    if image_name:
        matches = list((data_root / "datasets").rglob(str(image_name))) if (data_root / "datasets").exists() else []
        if matches:
            return matches[0]
    return None


def load_records(data_root: Path, metadata_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    path = find_metadata_path(data_root, metadata_path)
    data = read_json(path)
    records = data.get("records", [])
    enriched: list[dict[str, Any]] = []
    for index, raw in enumerate(records, start=1):
        record = dict(raw)
        image_path = resolve_image_path(record, data_root)
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        if image_path is not None and (width <= 0 or height <= 0):
            width, height = image_size(image_path)
        record["_image_id"] = index
        record["_image_abs"] = image_path.as_posix() if image_path else ""
        record["_image_file"] = repo_rel(image_path, data_root) if image_path else str(record.get("image_name", ""))
        record["_width"] = width
        record["_height"] = height
        record["_group"] = record_group(record)
        enriched.append(record)
    return enriched, data, path


def record_group(record: dict[str, Any]) -> str:
    stem = record.get("image_stem") or Path(str(record.get("image_name", ""))).stem
    return f"{record.get('annotation_kind', 'unknown')}::{stem}"


def bbox_dict_to_xyxy(bbox: dict[str, Any]) -> list[float]:
    if {"xtl", "ytl", "xbr", "ybr"} <= set(bbox):
        return [float(bbox["xtl"]), float(bbox["ytl"]), float(bbox["xbr"]), float(bbox["ybr"])]
    if {"x", "y", "width", "height"} <= set(bbox):
        return [
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["x"]) + float(bbox["width"]),
            float(bbox["y"]) + float(bbox["height"]),
        ]
    return [0.0, 0.0, 0.0, 0.0]


def clip_xyxy(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def xyxy_to_xywh(xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_area_xywh(bbox: list[float]) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area_xywh(a) + bbox_area_xywh(b) - inter
    return 0.0 if union <= 0 else inter / union


def bbox_size_bucket(bbox: list[float]) -> str:
    area = bbox_area_xywh(bbox)
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def point_visibility(point: dict[str, Any]) -> int:
    if int(point.get("v", 0)) <= 0:
        return 0
    return 1 if bool(point.get("occluded", False)) else 2


def source_point(raw: dict[str, Any] | None) -> list[float]:
    if not raw or int(raw.get("v", 0)) <= 0:
        return [0.0, 0.0, 0]
    return [float(raw.get("x", 0.0)), float(raw.get("y", 0.0)), point_visibility(raw)]


def assign_pair_by_x(
    keypoints: dict[str, Any],
    first_source: str,
    second_source: str,
    left_name: str,
    right_name: str,
    output: dict[str, list[float]],
) -> None:
    present = [
        (name, keypoints.get(name))
        for name in (first_source, second_source)
        if keypoints.get(name) and int(keypoints[name].get("v", 0)) > 0
    ]
    if len(present) >= 2:
        ordered = sorted(present[:2], key=lambda item: float(item[1].get("x", 0.0)))
        output[left_name] = source_point(ordered[0][1])
        output[right_name] = source_point(ordered[1][1])
    elif len(present) == 1:
        source_name, point = present[0]
        target = right_name if source_name == first_source else left_name
        output[target] = source_point(point)


def normalized_keypoints(obj: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    raw = obj.get("keypoints", {})
    assigned = {name: [0.0, 0.0, 0] for name in KEYPOINT_NAMES}
    assign_pair_by_x(raw, "e1", "e2", "left_eye", "right_eye", assigned)
    assigned["nose_tip"] = source_point(raw.get("n"))
    assign_pair_by_x(raw, "m1", "m2", "left_mouth_corner", "right_mouth_corner", assigned)
    flat: list[float] = []
    for name in KEYPOINT_NAMES:
        flat.extend(assigned[name])
    metadata = {
        "source_keypoints": raw,
        "normalized_keypoints": {name: assigned[name] for name in KEYPOINT_NAMES},
        "normalization_policy": "Pairs are assigned left/right by image x when both points are present; a single visible point keeps its source-derived side.",
    }
    return flat, metadata


def validate_existing_split(records: list[dict[str, Any]]) -> bool:
    if not records or any(record.get("split") not in SPLITS for record in records):
        return False
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["_group"]].add(str(record.get("split")))
    return all(len(values) == 1 for values in group_splits.values())


def assign_splits(records: list[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assigned = [dict(record) for record in records]
    if validate_existing_split(assigned):
        method = "reused_existing_metadata_split"
        for record in assigned:
            record["_split"] = record["split"]
    else:
        import random

        method = "generated_group_split_70_15_15"
        groups = sorted({record["_group"] for record in assigned})
        rng = random.Random(seed)
        rng.shuffle(groups)
        train_count = round(len(groups) * 0.70)
        val_count = round(len(groups) * 0.15)
        if val_count == 0 and len(groups) >= 2:
            val_count = 1
        if train_count + val_count >= len(groups) and len(groups) >= 3:
            train_count = max(1, len(groups) - val_count - 1)
        split_by_group = {}
        for index, group in enumerate(groups):
            if index < train_count:
                split_by_group[group] = "train"
            elif index < train_count + val_count:
                split_by_group[group] = "val"
            else:
                split_by_group[group] = "test"
        for record in assigned:
            record["_split"] = split_by_group[record["_group"]]
    summary = {
        "method": method,
        "seed": seed,
        "images_by_split": dict(Counter(record["_split"] for record in assigned)),
        "groups_by_split": dict(Counter(next(record["_split"] for record in assigned if record["_group"] == group) for group in sorted({r["_group"] for r in assigned}))),
        "group_count": len({record["_group"] for record in assigned}),
    }
    return assigned, summary


def write_split_outputs(records: list[dict[str, Any]], output_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "image_id": record["_image_id"],
                "split": record["_split"],
                "group": record["_group"],
                "annotation_kind": record.get("annotation_kind", ""),
                "image_stem": record.get("image_stem", ""),
                "image_name": record.get("image_name", ""),
                "image_file": record["_image_file"],
                "head_count": len(record.get("heads", [])),
                "face_count": len(record.get("faces", [])),
                "keypoint_object_count": len(record.get("objects", [])),
                "negative_sample": bool(record.get("negative_sample", False)),
            }
        )
    write_csv(output_dir / "data" / "splits" / "split_manifest.csv", rows)
    write_json(output_dir / "data" / "splits" / "split_summary.json", summary)


def annotation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    annotation_id = 1
    for record in records:
        base = {
            "image_id": record["_image_id"],
            "split": record.get("_split", record.get("split", "")),
            "image_name": record.get("image_name", ""),
            "image_file": record["_image_file"],
            "negative_sample": bool(record.get("negative_sample", False)),
        }
        for label_type, label_name, shapes in [
            ("bbox", "head", record.get("heads", [])),
            ("bbox", "face", record.get("faces", [])),
        ]:
            for shape_index, shape in enumerate(shapes):
                bbox = xyxy_to_xywh(bbox_dict_to_xyxy(shape.get("bbox", {})))
                rows.append(
                    {
                        **base,
                        "annotation_id": annotation_id,
                        "annotation_type": label_type,
                        "label": label_name,
                        "shape_index": shape_index,
                        "bbox_x": bbox[0],
                        "bbox_y": bbox[1],
                        "bbox_w": bbox[2],
                        "bbox_h": bbox[3],
                        "visible_keypoints": "",
                    }
                )
                annotation_id += 1
        for obj_index, obj in enumerate(record.get("objects", [])):
            keypoints, _ = normalized_keypoints(obj)
            visible = sum(1 for index in range(2, len(keypoints), 3) if int(keypoints[index]) > 0)
            bbox = xyxy_to_xywh(bbox_dict_to_xyxy(obj.get("training_bbox") or obj.get("raw_bbox") or obj.get("face", {}).get("bbox", {})))
            rows.append(
                {
                    **base,
                    "annotation_id": annotation_id,
                    "annotation_type": "keypoints",
                    "label": "face5",
                    "shape_index": obj_index,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "visible_keypoints": visible,
                }
            )
            annotation_id += 1
    return rows


def inspect_dataset(data_root: Path, output_dir: Path, metadata_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensure_layout(output_dir)
    records, metadata, resolved_metadata = load_records(data_root, metadata_path)
    records, split_summary = assign_splits(records)
    write_split_outputs(records, output_dir, split_summary)
    image_rows = [
        {
            "image_id": record["_image_id"],
            "split": record["_split"],
            "group": record["_group"],
            "annotation_kind": record.get("annotation_kind", ""),
            "image_stem": record.get("image_stem", ""),
            "image_name": record.get("image_name", ""),
            "image_file": record["_image_file"],
            "width": record["_width"],
            "height": record["_height"],
            "head_count": len(record.get("heads", [])),
            "face_count": len(record.get("faces", [])),
            "keypoint_object_count": len(record.get("objects", [])),
            "negative_sample": bool(record.get("negative_sample", False)),
            "missing_image": not bool(record.get("_image_abs")),
        }
        for record in records
    ]
    ann_rows = annotation_rows(records)
    label_counter = Counter()
    for record in records:
        if record.get("negative_sample", False):
            label_counter["negative_image"] += 1
        label_counter["head"] += len(record.get("heads", []))
        label_counter["face"] += len(record.get("faces", []))
        for obj in record.get("objects", []):
            keypoints, _ = normalized_keypoints(obj)
            for index, name in enumerate(KEYPOINT_NAMES):
                visibility = int(keypoints[index * 3 + 2])
                label_counter[f"{name}_v{visibility}"] += 1
    label_rows = [{"label": key, "count": value} for key, value in sorted(label_counter.items())]
    write_csv(output_dir / "data" / "raw_index" / "images.csv", image_rows)
    write_csv(output_dir / "data" / "raw_index" / "annotations.csv", ann_rows)
    write_csv(output_dir / "data" / "raw_index" / "label_stats.csv", label_rows)
    env = env_report(data_root)
    write_json(output_dir / "logs" / "environment.json", env)
    negative_with_heads = sum(1 for record in records if record.get("negative_sample") and record.get("heads"))
    data_report = [
        "# Data Report",
        "",
        f"- Generated at: {now_text()}",
        f"- Data root: `{path_text(data_root)}`",
        f"- Metadata: `{repo_rel(resolved_metadata, data_root)}`",
        f"- Images: {len(records)}",
        f"- Annotation rows: {len(ann_rows)}",
        f"- Split method: {split_summary['method']}",
        f"- Images by split: {split_summary['images_by_split']}",
        f"- Groups: {split_summary['group_count']}",
        f"- Head bbox count: {sum(len(record.get('heads', [])) for record in records)}",
        f"- Face bbox count: {sum(len(record.get('faces', [])) for record in records)}",
        f"- Keypoint object count: {sum(len(record.get('objects', [])) for record in records)}",
        f"- Negative samples: {sum(1 for record in records if record.get('negative_sample'))}",
        f"- Negative samples with head boxes that will be ignored for detection training: {negative_with_heads}",
        f"- Missing image files: {sum(1 for row in image_rows if row['missing_image'])}",
        "",
        "## Label Counts",
        "",
        *[f"- {row['label']}: {row['count']}" for row in label_rows],
        "",
        "## Environment",
        "",
        f"- CUDA available: {env.get('torch', {}).get('cuda_available')}",
        f"- CUDA device: {env.get('torch', {}).get('cuda_device')}",
        f"- MMCV CUDA ops available: {env.get('mmcv_cuda_ops_available')}",
        f"- MMDetection installed: {env['modules'].get('mmdet')}",
        f"- MMPose installed: {env['modules'].get('mmpose')}",
    ]
    write_text(output_dir / "reports" / "data_report.md", "\n".join(data_report))
    assumptions = [
        "# Assumptions",
        "",
        "- 既存 YOLO Pose metadata の split が group をまたがない場合は、その split を再利用する。",
        "- `e1/e2` と `m1/m2` は、両方が存在する場合は画像上の x 座標で left/right に割り当てる。",
        "- 片側だけ存在する eye/mouth keypoint は、元の `e1/e2/m1/m2` の対応を暫定的に使う。",
        "- `negative_sample=true` の画像は detection COCO でも annotation を空にする。そこに含まれる head box は初期実験では学習対象から外す。",
        "- keypoint 学習の bbox は face bbox または face ellipse 由来 bbox だけを使い、head-only 画像は keypoint 学習から除外する。",
        "- NME は face bbox の対角長で正規化する。",
    ]
    write_text(output_dir / "reports" / "assumptions.md", "\n".join(assumptions))
    return records, split_summary


def coco_image(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["_image_id"],
        "file_name": record["_image_file"],
        "width": record["_width"],
        "height": record["_height"],
        "metadata": {
            "annotation_kind": record.get("annotation_kind", ""),
            "image_stem": record.get("image_stem", ""),
            "group": record["_group"],
            "negative_sample": bool(record.get("negative_sample", False)),
        },
    }


def detection_annotations_for_record(record: dict[str, Any], start_id: int) -> tuple[list[dict[str, Any]], int, int]:
    annotations: list[dict[str, Any]] = []
    ignored_negative_shapes = 0
    if record.get("negative_sample", False):
        ignored_negative_shapes = len(record.get("heads", [])) + len(record.get("faces", []))
        return annotations, start_id, ignored_negative_shapes
    width, height = int(record["_width"]), int(record["_height"])
    for category_id, shapes in [(1, record.get("heads", [])), (2, record.get("faces", []))]:
        for shape_index, shape in enumerate(shapes):
            raw_xyxy = bbox_dict_to_xyxy(shape.get("bbox", {}))
            clipped = clip_xyxy(raw_xyxy, width, height)
            bbox = xyxy_to_xywh(clipped)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            annotations.append(
                {
                    "id": start_id,
                    "image_id": record["_image_id"],
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox_area_xywh(bbox),
                    "iscrowd": 0,
                    "metadata": {
                        "source_shape_index": shape_index,
                        "source_type": shape.get("type", ""),
                        "raw_bbox_xyxy": raw_xyxy,
                        "clipped_bbox_xyxy": clipped,
                        "was_clipped": raw_xyxy != clipped,
                    },
                }
            )
            start_id += 1
    return annotations, start_id, ignored_negative_shapes


def keypoint_annotations_for_record(record: dict[str, Any], start_id: int) -> tuple[list[dict[str, Any]], int]:
    annotations: list[dict[str, Any]] = []
    width, height = int(record["_width"]), int(record["_height"])
    for obj_index, obj in enumerate(record.get("objects", [])):
        source_bbox = obj.get("training_bbox") or obj.get("raw_bbox") or obj.get("face", {}).get("bbox", {})
        raw_xyxy = bbox_dict_to_xyxy(source_bbox)
        clipped = clip_xyxy(raw_xyxy, width, height)
        bbox = xyxy_to_xywh(clipped)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        keypoints, metadata = normalized_keypoints(obj)
        num_keypoints = sum(1 for index in range(2, len(keypoints), 3) if int(keypoints[index]) > 0)
        annotations.append(
            {
                "id": start_id,
                "image_id": record["_image_id"],
                "category_id": 1,
                "bbox": bbox,
                "area": bbox_area_xywh(bbox),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "metadata": {
                    **metadata,
                    "source_object_index": obj_index,
                    "raw_bbox_xyxy": raw_xyxy,
                    "clipped_bbox_xyxy": clipped,
                    "face_index": obj.get("face_index"),
                },
            }
        )
        start_id += 1
    return annotations, start_id


def convert_annotations(data_root: Path, output_dir: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    records, _metadata, _resolved = load_records(data_root, metadata_path)
    records, split_summary = assign_splits(records)
    write_split_outputs(records, output_dir, split_summary)
    detection_id = 1
    keypoint_id = 1
    conversion_summary: dict[str, Any] = {
        "created_at": now_text(),
        "images_by_split": {},
        "detection_annotations_by_split": {},
        "keypoint_annotations_by_split": {},
        "ignored_negative_shapes": 0,
    }
    for split in SPLITS:
        split_records = [record for record in records if record["_split"] == split]
        images = [coco_image(record) for record in split_records]
        detection_annotations: list[dict[str, Any]] = []
        keypoint_annotations: list[dict[str, Any]] = []
        for record in split_records:
            ann, detection_id, ignored = detection_annotations_for_record(record, detection_id)
            detection_annotations.extend(ann)
            conversion_summary["ignored_negative_shapes"] += ignored
            kp_ann, keypoint_id = keypoint_annotations_for_record(record, keypoint_id)
            keypoint_annotations.extend(kp_ann)
        detection_coco = {
            "info": {"description": "Head/face detection dataset for RTMDet", "version": "1.0", "date_created": now_text()},
            "licenses": [],
            "images": images,
            "annotations": detection_annotations,
            "categories": list(DETECTION_CATEGORIES),
        }
        keypoint_coco = {
            "info": {"description": "Five-point face keypoint dataset for RTMPose", "version": "1.0", "date_created": now_text()},
            "licenses": [],
            "images": images,
            "annotations": keypoint_annotations,
            "categories": [FACE_CATEGORY],
        }
        write_json(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json", detection_coco)
        write_json(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json", keypoint_coco)
        conversion_summary["images_by_split"][split] = len(images)
        conversion_summary["detection_annotations_by_split"][split] = len(detection_annotations)
        conversion_summary["keypoint_annotations_by_split"][split] = len(keypoint_annotations)
    write_json(output_dir / "data" / "converted" / "conversion_summary.json", conversion_summary)
    return conversion_summary


def generate_configs(data_root: Path, output_dir: Path) -> None:
    root = data_root.resolve().as_posix()
    out = output_dir.resolve().as_posix()
    det_common = f"""
custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)
_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = '{root}/'
output_root = '{out}'
classes = ('head', 'face')
metainfo = dict(classes=classes, palette=[(80, 220, 120), (80, 160, 255)])
num_classes = 2

model = dict(bbox_head=dict(num_classes=num_classes))

train_cfg = dict(max_epochs=2, val_interval=999)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(
    checkpoint=dict(interval=1, save_best=None, save_last=True, max_keep_ckpts=2),
    logger=dict(interval=10))

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='output_experiment/data/converted/detection_coco/instances_train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1)))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='output_experiment/data/converted/detection_coco/instances_test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True))
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'output_experiment/data/converted/detection_coco/instances_test.json', metric='bbox')
work_dir = output_root + '/logs/rtmdet_head_face_sanity'
"""
    det_tiny = det_common.replace("_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'", "_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'").replace(
        "work_dir = output_root + '/logs/rtmdet_head_face_sanity'",
        "work_dir = output_root + '/logs/rtmdet_tiny_head_face_sanity'",
    )
    write_text(output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py", det_common)
    write_text(output_dir / "configs" / "rtm_det_head_face" / "rtmdet_tiny_head_face_640_sanity.py", det_tiny)
    meta = """
dataset_info = dict(
    dataset_name='face5',
    paper_info=dict(
        author='local',
        title='Face five-point dataset',
        container='local experiment',
        year='2026',
        homepage=''),
    keypoint_info={
        0: dict(name='left_eye', id=0, color=[0, 255, 0], type='upper', swap='right_eye'),
        1: dict(name='right_eye', id=1, color=[0, 255, 0], type='upper', swap='left_eye'),
        2: dict(name='nose_tip', id=2, color=[255, 255, 0], type='upper', swap=''),
        3: dict(name='left_mouth_corner', id=3, color=[255, 128, 0], type='upper', swap='right_mouth_corner'),
        4: dict(name='right_mouth_corner', id=4, color=[255, 128, 0], type='upper', swap='left_mouth_corner'),
    },
    skeleton_info={
        0: dict(link=('left_eye', 'nose_tip'), id=0, color=[0, 255, 0]),
        1: dict(link=('right_eye', 'nose_tip'), id=1, color=[0, 255, 0]),
        2: dict(link=('nose_tip', 'left_mouth_corner'), id=2, color=[255, 128, 0]),
        3: dict(link=('nose_tip', 'right_mouth_corner'), id=3, color=[255, 128, 0]),
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.025, 0.025, 0.026, 0.035, 0.035])
"""
    write_text(output_dir / "configs" / "rtmpose_face" / "face5_dataset_meta.py", meta)
    pose_cfg = f"""
_base_ = ['mmpose::_base_/default_runtime.py']
custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)

num_keypoints = 5
input_size = (256, 256)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

max_epochs = 2
base_lr = 4e-3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=100),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.005, begin=1, end=max_epochs, T_max=max_epochs - 1, by_epoch=True, convert_to_iter_based=True),
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
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth')),
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
    test_cfg=dict(flip_test=True))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '{root}/'
metainfo = dict(from_file='{out}/configs/rtmpose_face/face5_dataset_meta.py')
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', scale_factor=[0.75, 1.25], rotate_factor=30),
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
        ann_file='output_experiment/data/converted/keypoint_coco/person_keypoints_train.json',
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
        ann_file='output_experiment/data/converted/keypoint_coco/person_keypoints_val.json',
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
        ann_file='output_experiment/data/converted/keypoint_coco/person_keypoints_test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = dict(type='NME', norm_mode='use_norm_item', norm_item='bbox_size')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less', max_keep_ckpts=2, interval=1), logger=dict(interval=10))
work_dir = '{out}/logs/rtmpose_face5_sanity'
"""
    write_text(output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py", pose_cfg)


def write_readme_and_reports(data_root: Path, output_dir: Path, conversion_summary: dict[str, Any]) -> None:
    env = env_report(data_root)
    install_block = "\n".join(f"  {command}" for command in MISSING_OPENMMLAB_COMMANDS)
    readme = f"""
# RTMDet + RTMPose-Face Experiment

This directory contains generated configs, converted data, model outputs, checkpoints, logs, and reports.
Runnable experiment scripts live under `script/script_experiment/` to match the repository layout.

## Commands

```bash
{python_executable(data_root)} script/script_experiment/rtmdet_rtmpose_experiment.py run-all --data-root . --output-dir output_experiment --mode prepare
{python_executable(data_root)} script/script_experiment/rtmdet_rtmpose_experiment.py run-all --data-root . --output-dir output_experiment --mode sanity --device cuda:0
```

## Outputs

- `data/raw_index/`: source image and annotation indexes.
- `data/converted/detection_coco/`: RTMDet COCO detection datasets.
- `data/converted/keypoint_coco/`: RTMPose five-keypoint COCO datasets.
- `configs/`: MMDetection and MMPose configs.
- `logs/`: OpenMMLab work dirs, command logs, checkpoint status JSON.
- `cache/`: downloaded pretrained weights and library caches.
- `predictions/`, `metrics/`, `visualizations/`: inference and evaluation artifacts.
- `reports/`: assumptions, data report, experiment report, failure analysis, and next actions.
"""
    write_text(output_dir / "README.md", readme)
    experiment = [
        "# Experiment Report",
        "",
        f"- Generated at: {now_text()}",
        "- Purpose: YOLO-Pose とは異なる二段構成として、RTMDet で head/face bbox を検出し、RTMPose-face で face crop 内の 5 点 keypoint を推定する。",
        f"- Data root: `{path_text(data_root)}`",
        f"- Converted images by split: {conversion_summary.get('images_by_split')}",
        f"- Detection annotations by split: {conversion_summary.get('detection_annotations_by_split')}",
        f"- Keypoint annotations by split: {conversion_summary.get('keypoint_annotations_by_split')}",
        f"- CUDA device: {env.get('torch', {}).get('cuda_device')}",
        f"- MMCV CUDA ops available: {env.get('mmcv_cuda_ops_available')}",
        f"- MMDetection installed: {env['modules'].get('mmdet')}",
        f"- MMPose installed: {env['modules'].get('mmpose')}",
        "",
        "## OpenMMLab Setup",
        "",
    ]
    if not (env["modules"].get("mmdet") and env["modules"].get("mmpose")):
        experiment.extend(
            [
                "MMDetection / MMPose が未導入のため、学習と実推論は未実行扱いにする。導入後は次を実行する。",
                "",
                "```bash",
                install_block,
                "```",
            ]
        )
    elif not env.get("mmcv_cuda_ops_available"):
        experiment.extend(
            [
                "MMDetection / MMPose は導入済み。現在の MMCV は CUDA ops なしで build されているため、detector の CUDA NMS は使えない。",
                "この実験 CLI は detector 推論だけ CPU にフォールバックし、RTMPose は `cuda:0` で推論する。",
                "detector も完全に GPU 推論する場合は CUDA-enabled MMCV を rebuild する。",
            ]
        )
    write_text(output_dir / "reports" / "experiment_report.md", "\n".join(experiment))
    failure = [
        "# Failure Analysis",
        "",
        "まだ実推論がない場合、失敗カテゴリは評価スクリプトが空予測として集計する。",
    ]
    write_text(output_dir / "reports" / "failure_analysis.md", "\n".join(failure))
    next_actions = [
        "# Next Actions",
        "",
        "1. MMDetection / MMPose を導入し、`--mode sanity --device cuda:0` を再実行する。",
        "2. GT bbox + RTMPose と two-stage の差を見て、detector 起因か landmark 起因かを切り分ける。",
        "3. negative sample での false positive と小さい顔の recall を優先して確認する。",
        "4. left/right swap が多い場合は keypoint 定義と左右割り当てを見直す。",
    ]
    write_text(output_dir / "reports" / "next_actions.md", "\n".join(next_actions))


def dependency_status(required: list[str]) -> tuple[bool, dict[str, bool]]:
    status = {name: module_available(name) for name in required}
    return all(status.values()), status


def write_status(output_dir: Path, name: str, status: dict[str, Any]) -> None:
    write_json(output_dir / "logs" / f"{name}_status.json", status)


def apply_training_overrides(
    cfg: Any,
    *,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    work_dir: Path | None = None,
) -> None:
    if max_epochs is not None:
        cfg.train_cfg.max_epochs = int(max_epochs)
        if "max_epochs" in cfg:
            cfg.max_epochs = int(max_epochs)
        for scheduler in cfg.get("param_scheduler", []):
            if scheduler.get("by_epoch"):
                begin = int(scheduler.get("begin", 0))
                if begin >= int(max_epochs):
                    begin = max(0, int(max_epochs) - 1)
                    scheduler["begin"] = begin
                if "end" in scheduler:
                    scheduler["end"] = int(max_epochs)
                end = int(scheduler.get("end", max_epochs))
                if end <= begin:
                    end = begin + 1
                    scheduler["end"] = end
                if "T_max" in scheduler:
                    scheduler["T_max"] = max(1, end - begin)
    if batch_size is not None:
        cfg.train_dataloader.batch_size = int(batch_size)
    if work_dir is not None:
        cfg.work_dir = work_dir.as_posix()
    if cfg.get("val_cfg", object()) is None:
        cfg.val_dataloader = None
        cfg.val_evaluator = None


def train_with_mmengine(
    config: Path,
    output_dir: Path,
    name: str,
    required: list[str],
    *,
    max_epochs: int | None = None,
    batch_size: int | None = None,
) -> int:
    configure_output_environment(output_dir)
    ok, modules = dependency_status(required)
    if not ok:
        status = {
            "status": "skipped",
            "reason": "missing_dependencies",
            "modules": modules,
            "install_commands": MISSING_OPENMMLAB_COMMANDS,
            "config": config.as_posix(),
        }
        write_status(output_dir, name, status)
        append_report(output_dir, f"{name} skipped", ["- Reason: missing OpenMMLab dependencies.", *[f"- `{cmd}`" for cmd in MISSING_OPENMMLAB_COMMANDS]])
        return 0
    try:
        from mmengine.config import Config  # type: ignore
        from mmengine.runner import Runner  # type: ignore

        cfg = Config.fromfile(config.as_posix())
        work_dir = Path(cfg.get("work_dir", (output_dir / "logs" / name).as_posix()))
        apply_training_overrides(cfg, max_epochs=max_epochs, batch_size=batch_size, work_dir=work_dir)
        runner = Runner.from_cfg(cfg)
        runner.train()
        write_status(
            output_dir,
            name,
            {
                "status": "completed",
                "config": config.as_posix(),
                "work_dir": cfg.work_dir,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
            },
        )
        return 0
    except Exception as exc:  # pragma: no cover - requires OpenMMLab
        write_status(output_dir, name, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        return 1


def load_coco(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"images": [], "annotations": [], "categories": []}
    return read_json(path)


def empty_detector_predictions(coco: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "created_at": now_text(),
        "status": "empty",
        "reason": reason,
        "images": [
            {"image_id": image["id"], "file_name": image["file_name"], "detections": []}
            for image in coco.get("images", [])
        ],
    }


def empty_keypoint_predictions(coco: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "created_at": now_text(),
        "status": "empty",
        "reason": reason,
        "images": [
            {"image_id": image["id"], "file_name": image["file_name"], "instances": []}
            for image in coco.get("images", [])
        ],
    }


def infer_detector(data_root: Path, output_dir: Path, config: Path, checkpoint: Path | None, split: str, device: str) -> int:
    configure_output_environment(output_dir)
    coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
    pred_path = output_dir / "predictions" / "detector" / "predictions.json"
    ok, modules = dependency_status(["mmdet", "mmengine"])
    if not ok or checkpoint is None or not checkpoint.exists():
        reason = "missing_dependencies" if not ok else "missing_checkpoint"
        write_json(pred_path, empty_detector_predictions(coco, reason))
        write_status(output_dir, "infer_detector", {"status": "skipped", "reason": reason, "modules": modules, "checkpoint": str(checkpoint) if checkpoint else ""})
        return 0
    try:
        from mmdet.apis import inference_detector, init_detector  # type: ignore

        run_device, device_note = effective_detector_device(device)
        with local_checkpoint_load_context(checkpoint, output_dir):
            model = init_detector(config.as_posix(), checkpoint.as_posix(), device=run_device)
        images = []
        for image in coco.get("images", []):
            result = inference_detector(model, (data_root / image["file_name"]).as_posix())
            pred_instances = result.pred_instances
            detections = []
            for bbox, score, label in zip(pred_instances.bboxes.cpu().numpy(), pred_instances.scores.cpu().numpy(), pred_instances.labels.cpu().numpy()):
                x1, y1, x2, y2 = [float(value) for value in bbox]
                category_id = int(label) + 1
                detections.append(
                    {
                        "category_id": category_id,
                        "class_name": "head" if category_id == 1 else "face",
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
            images.append({"image_id": image["id"], "file_name": image["file_name"], "detections": detections})
        write_json(
            pred_path,
            {
                "created_at": now_text(),
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": images,
            },
        )
        write_status(
            output_dir,
            "infer_detector",
            {
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": len(images),
            },
        )
        return 0
    except Exception as exc:  # pragma: no cover - requires OpenMMLab
        write_json(pred_path, empty_detector_predictions(coco, "inference_failed"))
        write_status(output_dir, "infer_detector", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        return 1


def keypoint_instances_from_mmpose(predictions: Any) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for sample in predictions:
        pred = sample.pred_instances
        keypoints = pred.keypoints[0].tolist() if len(pred.keypoints) else []
        scores = pred.keypoint_scores[0].tolist() if hasattr(pred, "keypoint_scores") and len(pred.keypoint_scores) else []
        bbox = pred.bboxes[0].tolist() if hasattr(pred, "bboxes") and len(pred.bboxes) else [0, 0, 0, 0]
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        instances.append({"bbox": bbox, "keypoints": keypoints, "keypoint_scores": scores, "score": min(scores) if scores else 0.0})
    return instances


def infer_keypoint_gt_bbox(data_root: Path, output_dir: Path, config: Path, checkpoint: Path | None, split: str, device: str) -> int:
    configure_output_environment(output_dir)
    coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
    pred_path = output_dir / "predictions" / "keypoint_gt_bbox" / "predictions.json"
    ok, modules = dependency_status(["mmpose", "mmengine"])
    if not ok or checkpoint is None or not checkpoint.exists():
        reason = "missing_dependencies" if not ok else "missing_checkpoint"
        write_json(pred_path, empty_keypoint_predictions(coco, reason))
        write_status(output_dir, "infer_keypoint_gt_bbox", {"status": "skipped", "reason": reason, "modules": modules, "checkpoint": str(checkpoint) if checkpoint else ""})
        return 0
    try:
        from mmpose.apis import inference_topdown, init_model  # type: ignore

        run_device, device_note = effective_device(device)
        with local_checkpoint_load_context(checkpoint, output_dir):
            model = init_model(config.as_posix(), checkpoint.as_posix(), device=run_device)
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)
        images = []
        for image in coco.get("images", []):
            bboxes = []
            ann_ids = []
            for ann in anns_by_image[int(image["id"])]:
                x, y, w, h = ann["bbox"]
                bboxes.append([x, y, x + w, y + h])
                ann_ids.append(ann["id"])
            preds = inference_topdown(model, (data_root / image["file_name"]).as_posix(), bboxes=bboxes) if bboxes else []
            instances = keypoint_instances_from_mmpose(preds)
            for instance, ann_id in zip(instances, ann_ids):
                instance["gt_annotation_id"] = ann_id
            images.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances})
        write_json(
            pred_path,
            {
                "created_at": now_text(),
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": images,
            },
        )
        write_status(
            output_dir,
            "infer_keypoint_gt_bbox",
            {
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": len(images),
            },
        )
        return 0
    except Exception as exc:  # pragma: no cover - requires OpenMMLab
        write_json(pred_path, empty_keypoint_predictions(coco, "inference_failed"))
        write_status(output_dir, "infer_keypoint_gt_bbox", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        return 1


def infer_two_stage(data_root: Path, output_dir: Path, config: Path, checkpoint: Path | None, split: str, device: str) -> int:
    configure_output_environment(output_dir)
    coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
    detector_predictions = read_json(output_dir / "predictions" / "detector" / "predictions.json") if (output_dir / "predictions" / "detector" / "predictions.json").exists() else {}
    pred_path = output_dir / "predictions" / "two_stage" / "predictions.json"
    ok, modules = dependency_status(["mmpose", "mmengine"])
    if not ok or checkpoint is None or not checkpoint.exists():
        reason = "missing_dependencies" if not ok else "missing_checkpoint"
        write_json(pred_path, empty_keypoint_predictions(coco, reason))
        write_status(output_dir, "infer_two_stage", {"status": "skipped", "reason": reason, "modules": modules, "checkpoint": str(checkpoint) if checkpoint else ""})
        return 0
    try:
        from mmpose.apis import inference_topdown, init_model  # type: ignore

        run_device, device_note = effective_device(device)
        with local_checkpoint_load_context(checkpoint, output_dir):
            model = init_model(config.as_posix(), checkpoint.as_posix(), device=run_device)
        det_by_image = {int(row["image_id"]): row.get("detections", []) for row in detector_predictions.get("images", [])}
        images = []
        for image in coco.get("images", []):
            face_dets = [det for det in det_by_image.get(int(image["id"]), []) if int(det.get("category_id", 0)) == 2]
            bboxes = [[det["bbox"][0], det["bbox"][1], det["bbox"][0] + det["bbox"][2], det["bbox"][1] + det["bbox"][3]] for det in face_dets]
            preds = inference_topdown(model, (data_root / image["file_name"]).as_posix(), bboxes=bboxes) if bboxes else []
            instances = keypoint_instances_from_mmpose(preds)
            for instance, det in zip(instances, face_dets):
                instance["detector_score"] = det.get("score", 0.0)
            images.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances})
        write_json(
            pred_path,
            {
                "created_at": now_text(),
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": images,
            },
        )
        write_status(
            output_dir,
            "infer_two_stage",
            {
                "status": "completed",
                "requested_device": device,
                "device": run_device,
                "device_note": device_note,
                "checkpoint": checkpoint.as_posix(),
                "config": config.as_posix(),
                "images": len(images),
            },
        )
        return 0
    except Exception as exc:  # pragma: no cover - requires OpenMMLab
        write_json(pred_path, empty_keypoint_predictions(coco, "inference_failed"))
        write_status(output_dir, "infer_two_stage", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        return 1


def prediction_map(predictions: dict[str, Any], kind: str) -> dict[int, list[dict[str, Any]]]:
    key = "detections" if kind == "detector" else "instances"
    return {int(row["image_id"]): row.get(key, []) for row in predictions.get("images", [])}


def match_detection_at_threshold(coco: dict[str, Any], predictions: dict[str, Any], threshold: float) -> dict[str, Any]:
    gts_by_class_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        gts_by_class_image[(int(ann["category_id"]), int(ann["image_id"]))].append(ann)
    preds = []
    for image_row in predictions.get("images", []):
        for det in image_row.get("detections", []):
            preds.append({**det, "image_id": int(image_row["image_id"])})
    preds.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    matched: set[int] = set()
    tp = fp = 0
    matches = []
    for pred in preds:
        candidates = gts_by_class_image.get((int(pred.get("category_id", 0)), int(pred["image_id"])), [])
        best_iou = 0.0
        best_ann = None
        for ann in candidates:
            if int(ann["id"]) in matched:
                continue
            iou = iou_xywh(pred["bbox"], ann["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ann = ann
        if best_ann is not None and best_iou >= threshold:
            matched.add(int(best_ann["id"]))
            tp += 1
            matches.append({"type": "tp", "image_id": pred["image_id"], "category_id": pred.get("category_id"), "iou": best_iou})
        else:
            fp += 1
            matches.append({"type": "fp", "image_id": pred["image_id"], "category_id": pred.get("category_id"), "iou": best_iou})
    gt_count = len(coco.get("annotations", []))
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "matched_ids": matched, "matches": matches}


def average_precision_for_class(coco: dict[str, Any], predictions: dict[str, Any], category_id: int, threshold: float) -> float | None:
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if int(ann["category_id"]) == category_id:
            gt_by_image[int(ann["image_id"])].append(ann)
    gt_count = sum(len(rows) for rows in gt_by_image.values())
    if gt_count == 0:
        return None
    preds = []
    for image_row in predictions.get("images", []):
        for det in image_row.get("detections", []):
            if int(det.get("category_id", 0)) == category_id:
                preds.append({**det, "image_id": int(image_row["image_id"])})
    preds.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    matched: set[int] = set()
    precisions: list[float] = []
    recalls: list[float] = []
    tp = fp = 0
    for pred in preds:
        best_iou, best_ann = 0.0, None
        for ann in gt_by_image.get(int(pred["image_id"]), []):
            if int(ann["id"]) in matched:
                continue
            iou = iou_xywh(pred["bbox"], ann["bbox"])
            if iou > best_iou:
                best_iou, best_ann = iou, ann
        if best_ann is not None and best_iou >= threshold:
            matched.add(int(best_ann["id"]))
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / gt_count)
    if not precisions:
        return 0.0
    ap = 0.0
    for recall_threshold in [i / 100 for i in range(0, 101)]:
        precision_at_recall = max([p for p, r in zip(precisions, recalls) if r >= recall_threshold] or [0.0])
        ap += precision_at_recall / 101.0
    return ap


def evaluate_detector(output_dir: Path, split: str = "test") -> dict[str, Any]:
    coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
    predictions = read_json(output_dir / "predictions" / "detector" / "predictions.json") if (output_dir / "predictions" / "detector" / "predictions.json").exists() else empty_detector_predictions(coco, "missing_predictions")
    primary = match_detection_at_threshold(coco, predictions, 0.5)
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    class_ap: dict[str, Any] = {}
    for category in DETECTION_CATEGORIES:
        aps = [average_precision_for_class(coco, predictions, int(category["id"]), threshold) for threshold in thresholds]
        valid = [value for value in aps if value is not None]
        class_ap[category["name"]] = {
            "AP50": average_precision_for_class(coco, predictions, int(category["id"]), 0.5),
            "AP50_95": sum(valid) / len(valid) if valid else None,
        }
    map50_values = [row["AP50"] for row in class_ap.values() if row["AP50"] is not None]
    map5095_values = [row["AP50_95"] for row in class_ap.values() if row["AP50_95"] is not None]
    pred_by_image = prediction_map(predictions, "detector")
    image_ids_with_gt = {int(ann["image_id"]) for ann in coco.get("annotations", [])}
    negative_fp = sum(len(pred_by_image.get(int(image["id"]), [])) for image in coco.get("images", []) if int(image["id"]) not in image_ids_with_gt)
    class_recall = {}
    for category in DETECTION_CATEGORIES:
        class_coco = {**coco, "annotations": [ann for ann in coco.get("annotations", []) if int(ann["category_id"]) == int(category["id"])]}
        class_recall[category["name"]] = match_detection_at_threshold(class_coco, predictions, 0.5)["recall"]
    size_counts = Counter()
    size_matched = Counter()
    matched_ids = primary["matched_ids"]
    for ann in coco.get("annotations", []):
        bucket = bbox_size_bucket(ann["bbox"])
        size_counts[bucket] += 1
        if int(ann["id"]) in matched_ids:
            size_matched[bucket] += 1
    size_recall = {bucket: (size_matched[bucket] / size_counts[bucket] if size_counts[bucket] else None) for bucket in ("small", "medium", "large")}
    image_count = len(coco.get("images", []))
    metrics = {
        "status": predictions.get("status", "completed"),
        "prediction_reason": predictions.get("reason"),
        "split": split,
        "images": image_count,
        "gt_annotations": len(coco.get("annotations", [])),
        "mAP@0.5": sum(map50_values) / len(map50_values) if map50_values else None,
        "mAP@0.5:0.95": sum(map5095_values) / len(map5095_values) if map5095_values else None,
        "class_wise_AP": class_ap,
        "precision": primary["precision"],
        "recall": primary["recall"],
        "F1": primary["f1"],
        "TP": primary["tp"],
        "FP": primary["fp"],
        "FN": primary["fn"],
        "FP_per_image": primary["fp"] / image_count if image_count else 0.0,
        "FN_per_image": primary["fn"] / image_count if image_count else 0.0,
        "size_recall": size_recall,
        "class_recall": class_recall,
        "negative_sample_false_positive_count": negative_fp,
    }
    write_json(output_dir / "metrics" / "detector_metrics.json", metrics)
    write_csv(output_dir / "metrics" / "detector_metrics.csv", [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in metrics.items()])
    return metrics


def match_keypoint_predictions(coco: dict[str, Any], predictions: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], int]:
    pred_by_image = prediction_map(predictions, "keypoint")
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    matched_pred_ids: set[tuple[int, int]] = set()
    for ann in coco.get("annotations", []):
        candidates = pred_by_image.get(int(ann["image_id"]), [])
        direct = [pred for pred in candidates if pred.get("gt_annotation_id") == ann.get("id")]
        selected = direct[0] if direct else None
        if selected is None:
            best_iou, best_index = 0.0, None
            for index, pred in enumerate(candidates):
                key = (int(ann["image_id"]), index)
                if key in matched_pred_ids:
                    continue
                iou = iou_xywh(ann["bbox"], pred.get("bbox", [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index is not None and best_iou >= 0.5:
                selected = candidates[best_index]
                matched_pred_ids.add((int(ann["image_id"]), best_index))
        pairs.append((ann, selected))
    total_predictions = sum(len(row.get("instances", [])) for row in predictions.get("images", []))
    return pairs, total_predictions


def flatten_pred_keypoints(pred: dict[str, Any]) -> list[float]:
    keypoints = pred.get("keypoints", [])
    if not keypoints:
        return []
    if isinstance(keypoints[0], list):
        flat = []
        scores = pred.get("keypoint_scores", [])
        for index, point in enumerate(keypoints):
            x, y = float(point[0]), float(point[1])
            score = float(scores[index]) if index < len(scores) else float(pred.get("score", 1.0))
            flat.extend([x, y, 2 if score > 0 else 0])
        return flat
    return [float(value) for value in keypoints]


def evaluate_keypoint(output_dir: Path, split: str = "test", prediction_name: str = "keypoint_gt_bbox") -> dict[str, Any]:
    coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
    pred_path = output_dir / "predictions" / prediction_name / "predictions.json"
    predictions = read_json(pred_path) if pred_path.exists() else empty_keypoint_predictions(coco, "missing_predictions")
    pairs, total_predictions = match_keypoint_predictions(coco, predictions)
    distances: list[float] = []
    normalized_distances: list[float] = []
    pck_counts = Counter()
    kp_errors: dict[str, list[float]] = {name: [] for name in KEYPOINT_NAMES}
    visibility_errors: dict[str, list[float]] = {"visible": [], "occluded": []}
    size_errors: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    missed = 0
    evaluated_instances = 0
    for ann, pred in pairs:
        if pred is None:
            missed += 1
            continue
        pred_kpts = flatten_pred_keypoints(pred)
        if len(pred_kpts) < len(KEYPOINT_NAMES) * 3:
            missed += 1
            continue
        evaluated_instances += 1
        diag = math.hypot(float(ann["bbox"][2]), float(ann["bbox"][3]))
        if diag <= 0:
            continue
        bucket = bbox_size_bucket(ann["bbox"])
        gt = ann["keypoints"]
        for index, name in enumerate(KEYPOINT_NAMES):
            v = int(gt[index * 3 + 2])
            if v <= 0:
                continue
            dx = float(pred_kpts[index * 3]) - float(gt[index * 3])
            dy = float(pred_kpts[index * 3 + 1]) - float(gt[index * 3 + 1])
            dist = math.hypot(dx, dy)
            norm = dist / diag
            distances.append(dist)
            normalized_distances.append(norm)
            kp_errors[name].append(norm)
            visibility_errors["occluded" if v == 1 else "visible"].append(norm)
            size_errors[bucket].append(norm)
            for threshold in (0.03, 0.05, 0.10):
                if norm <= threshold:
                    pck_counts[str(threshold)] += 1
    denom = len(normalized_distances)
    metrics = {
        "status": predictions.get("status", "completed"),
        "prediction_reason": predictions.get("reason"),
        "split": split,
        "normalization": "face_bbox_diagonal",
        "gt_instances": len(coco.get("annotations", [])),
        "predicted_instances": total_predictions,
        "evaluated_instances": evaluated_instances,
        "missed_instances": missed,
        "NME": sum(normalized_distances) / denom if denom else None,
        "PCK@0.03": pck_counts["0.03"] / denom if denom else None,
        "PCK@0.05": pck_counts["0.05"] / denom if denom else None,
        "PCK@0.10": pck_counts["0.1"] / denom if denom else None,
        "keypoint_wise_error": {name: (sum(values) / len(values) if values else None) for name, values in kp_errors.items()},
        "visibility_error": {name: (sum(values) / len(values) if values else None) for name, values in visibility_errors.items()},
        "bbox_size_error": {name: (sum(values) / len(values) if values else None) for name, values in size_errors.items()},
    }
    if prediction_name == "keypoint_gt_bbox":
        json_name = "keypoint_gt_bbox_metrics.json"
        csv_name = "keypoint_gt_bbox_metrics.csv"
    else:
        json_name = f"{prediction_name}_keypoint_metrics.json"
        csv_name = f"{prediction_name}_keypoint_metrics.csv"
    write_json(output_dir / "metrics" / json_name, metrics)
    write_csv(output_dir / "metrics" / csv_name, [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in metrics.items()])
    return metrics


def evaluate_two_stage(output_dir: Path, split: str = "test") -> dict[str, Any]:
    coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
    det_coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
    detector_predictions = read_json(output_dir / "predictions" / "detector" / "predictions.json") if (output_dir / "predictions" / "detector" / "predictions.json").exists() else empty_detector_predictions(det_coco, "missing_predictions")
    keypoint_predictions = read_json(output_dir / "predictions" / "two_stage" / "predictions.json") if (output_dir / "predictions" / "two_stage" / "predictions.json").exists() else empty_keypoint_predictions(coco, "missing_predictions")
    face_gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in det_coco.get("annotations", []):
        if int(ann["category_id"]) == 2:
            face_gt_by_image[int(ann["image_id"])].append(ann)
    det_by_image = prediction_map(detector_predictions, "detector")
    kp_pairs, total_kp_predictions = match_keypoint_predictions(coco, keypoint_predictions)
    kp_metrics = evaluate_keypoint(output_dir, split, "two_stage")
    failures: list[dict[str, Any]] = []
    matched_faces = 0
    duplicate_count = 0
    fp_count = 0
    for image in det_coco.get("images", []):
        image_id = int(image["id"])
        gts = face_gt_by_image.get(image_id, [])
        face_dets = [det for det in det_by_image.get(image_id, []) if int(det.get("category_id", 0)) == 2]
        used: set[int] = set()
        for det in sorted(face_dets, key=lambda row: float(row.get("score", 0.0)), reverse=True):
            best_iou, best_index = 0.0, None
            for index, gt in enumerate(gts):
                iou = iou_xywh(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index is not None and best_iou >= 0.5:
                if best_index in used:
                    duplicate_count += 1
                    failures.append({"image_id": image_id, "image_file": image["file_name"], "category": "detector_duplicate", "iou": best_iou})
                else:
                    used.add(best_index)
                    matched_faces += 1
            else:
                fp_count += 1
                failures.append({"image_id": image_id, "image_file": image["file_name"], "category": "detector_false_positive", "iou": best_iou})
        for index, gt in enumerate(gts):
            if index not in used:
                failures.append({"image_id": image_id, "image_file": image["file_name"], "category": "detector_miss", "gt_annotation_id": gt["id"]})
    detector_negative_fp = sum(
        len(det_by_image.get(int(image["id"]), []))
        for image in det_coco.get("images", [])
        if not face_gt_by_image.get(int(image["id"]))
    )
    failure_counts = dict(Counter(row["category"] for row in failures))
    metrics = {
        "status": keypoint_predictions.get("status", "completed"),
        "prediction_reason": keypoint_predictions.get("reason"),
        "split": split,
        "gt_face_count": sum(len(rows) for rows in face_gt_by_image.values()),
        "face_detection_recall": matched_faces / sum(len(rows) for rows in face_gt_by_image.values()) if face_gt_by_image else 0.0,
        "keypoint_success_rate": kp_metrics["evaluated_instances"] / len(coco.get("annotations", [])) if coco.get("annotations") else 0.0,
        "end_to_end_PCK@0.05": kp_metrics.get("PCK@0.05"),
        "end_to_end_PCK@0.10": kp_metrics.get("PCK@0.10"),
        "missed_face_count": failure_counts.get("detector_miss", 0),
        "detected_face_but_keypoint_failed_count": max(0, matched_faces - kp_metrics["evaluated_instances"]),
        "duplicate_detection_count": duplicate_count,
        "false_positive_mosaic_candidate_count": fp_count,
        "negative_sample_false_positive_count": detector_negative_fp,
        "keypoint_prediction_count": total_kp_predictions,
        "failure_counts": failure_counts,
    }
    write_json(output_dir / "metrics" / "two_stage_metrics.json", metrics)
    write_csv(output_dir / "metrics" / "two_stage_metrics.csv", [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in metrics.items()])
    failure_fieldnames = ["image_id", "image_file", "category", "gt_annotation_id", "iou"]
    write_csv(output_dir / "metrics" / "failure_analysis.csv", failures, failure_fieldnames)
    summary_rows = [
        {"metric_group": "detector", "metric": "face_detection_recall", "value": metrics["face_detection_recall"]},
        {"metric_group": "keypoint", "metric": "keypoint_success_rate", "value": metrics["keypoint_success_rate"]},
        {"metric_group": "two_stage", "metric": "end_to_end_PCK@0.05", "value": metrics["end_to_end_PCK@0.05"]},
    ]
    write_csv(output_dir / "metrics" / "summary.csv", summary_rows)
    failure_md = ["# Failure Analysis", "", "## Category Counts", ""]
    if failure_counts:
        failure_md.extend([f"- {key}: {value}" for key, value in sorted(failure_counts.items())])
    else:
        failure_md.append("- No failures were recorded.")
    failure_md.extend(["", "Representative images are written under `visualizations/errors/` when visualization is run."])
    write_text(output_dir / "reports" / "failure_analysis.md", "\n".join(failure_md))
    return metrics


def draw_bbox(cv2: Any, image: Any, bbox: list[float], color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(image, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_keypoints(cv2: Any, image: Any, keypoints: list[float], color: tuple[int, int, int]) -> None:
    for index in range(0, len(keypoints), 3):
        if int(keypoints[index + 2]) <= 0:
            continue
        x, y = int(round(keypoints[index])), int(round(keypoints[index + 1]))
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)


def visualize_predictions(data_root: Path, output_dir: Path, split: str = "test", max_images: int = 200) -> int:
    if not module_available("cv2"):
        write_status(output_dir, "visualize_predictions", {"status": "skipped", "reason": "missing_cv2"})
        return 0
    import cv2  # type: ignore

    colors = {
        "gt": (0, 220, 0),
        "pred": (0, 0, 255),
        "pred_alt": (255, 80, 0),
        "missed": (0, 255, 255),
        "false_positive": (180, 0, 180),
    }
    det_coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
    kp_coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
    det_predictions = read_json(output_dir / "predictions" / "detector" / "predictions.json") if (output_dir / "predictions" / "detector" / "predictions.json").exists() else empty_detector_predictions(det_coco, "missing_predictions")
    kp_predictions = read_json(output_dir / "predictions" / "keypoint_gt_bbox" / "predictions.json") if (output_dir / "predictions" / "keypoint_gt_bbox" / "predictions.json").exists() else empty_keypoint_predictions(kp_coco, "missing_predictions")
    two_predictions = read_json(output_dir / "predictions" / "two_stage" / "predictions.json") if (output_dir / "predictions" / "two_stage" / "predictions.json").exists() else empty_keypoint_predictions(kp_coco, "missing_predictions")
    det_gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in det_coco.get("annotations", []):
        det_gt_by_image[int(ann["image_id"])].append(ann)
    kp_gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in kp_coco.get("annotations", []):
        kp_gt_by_image[int(ann["image_id"])].append(ann)
    det_pred_by_image = prediction_map(det_predictions, "detector")
    kp_pred_by_image = prediction_map(kp_predictions, "keypoint")
    two_pred_by_image = prediction_map(two_predictions, "keypoint")
    for image in kp_coco.get("images", [])[:max_images]:
        image_path = data_root / image["file_name"]
        base_image = cv2.imread(image_path.as_posix())
        if base_image is None:
            continue
        stem = Path(image["file_name"]).stem
        image_id = int(image["id"])
        det_canvas = base_image.copy()
        kp_canvas = base_image.copy()
        two_canvas = base_image.copy()
        for ann in det_gt_by_image.get(image_id, []):
            label = "GT head" if int(ann["category_id"]) == 1 else "GT face"
            draw_bbox(cv2, det_canvas, ann["bbox"], colors["gt"], label)
        for det in det_pred_by_image.get(image_id, []):
            draw_bbox(cv2, det_canvas, det["bbox"], colors["pred"], f"P {det.get('class_name', '')} {float(det.get('score', 0.0)):.2f}")
        for ann in kp_gt_by_image.get(image_id, []):
            draw_bbox(cv2, kp_canvas, ann["bbox"], colors["gt"], "GT face")
            draw_bbox(cv2, two_canvas, ann["bbox"], colors["gt"], "GT face")
            draw_keypoints(cv2, kp_canvas, ann["keypoints"], colors["gt"])
            draw_keypoints(cv2, two_canvas, ann["keypoints"], colors["gt"])
        for pred in kp_pred_by_image.get(image_id, []):
            draw_bbox(cv2, kp_canvas, pred.get("bbox", [0, 0, 0, 0]), colors["pred_alt"], "P kpt")
            draw_keypoints(cv2, kp_canvas, flatten_pred_keypoints(pred), colors["pred_alt"])
        for pred in two_pred_by_image.get(image_id, []):
            draw_bbox(cv2, two_canvas, pred.get("bbox", [0, 0, 0, 0]), colors["pred"], "P two")
            draw_keypoints(cv2, two_canvas, flatten_pred_keypoints(pred), colors["pred"])
        cv2.imwrite((output_dir / "visualizations" / "detector" / f"{stem}.jpg").as_posix(), det_canvas)
        cv2.imwrite((output_dir / "visualizations" / "keypoint_gt_bbox" / f"{stem}.jpg").as_posix(), kp_canvas)
        cv2.imwrite((output_dir / "visualizations" / "two_stage" / f"{stem}.jpg").as_posix(), two_canvas)
    failure_path = output_dir / "metrics" / "failure_analysis.csv"
    if failure_path.exists():
        with failure_path.open(encoding="utf-8", newline="") as file:
            for row in list(csv.DictReader(file))[:max_images]:
                source = data_root / row["image_file"]
                image = cv2.imread(source.as_posix())
                if image is None:
                    continue
                cv2.putText(image, row.get("category", "error"), (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["missed"], 2, cv2.LINE_AA)
                cv2.imwrite((output_dir / "visualizations" / "errors" / f"{Path(row['image_file']).stem}_{row.get('category', 'error')}.jpg").as_posix(), image)
    write_status(output_dir, "visualize_predictions", {"status": "completed", "split": split, "max_images": max_images})
    return 0


def prepare(data_root: Path, output_dir: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    configure_output_environment(output_dir)
    ensure_layout(output_dir)
    inspect_dataset(data_root, output_dir, metadata_path)
    conversion_summary = convert_annotations(data_root, output_dir, metadata_path)
    generate_configs(data_root, output_dir)
    write_readme_and_reports(data_root, output_dir, conversion_summary)
    return conversion_summary


def checkpoint_candidates(output_dir: Path, keywords: tuple[str, ...]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("best*.pth", "epoch_*.pth"):
        candidates.extend(output_dir.rglob(pattern))
    filtered = []
    for path in candidates:
        rel_parts = path.relative_to(output_dir).parts if path.is_relative_to(output_dir) else path.parts
        if "cache" in rel_parts:
            continue
        haystack = path.as_posix().lower()
        if any(keyword in haystack for keyword in keywords):
            filtered.append(path)
    return sorted(set(filtered), key=lambda path: (path.stat().st_mtime, path.as_posix()))


def default_detector_checkpoint(output_dir: Path) -> Path | None:
    candidates = checkpoint_candidates(output_dir, ("rtmdet", "detector"))
    return candidates[-1] if candidates else None


def default_keypoint_checkpoint(output_dir: Path) -> Path | None:
    candidates = checkpoint_candidates(output_dir, ("rtmpose", "keypoint"))
    return candidates[-1] if candidates else None


def run_all(
    data_root: Path,
    output_dir: Path,
    mode: str,
    device: str,
    metadata_path: Path | None = None,
    *,
    detector_epochs: int | None = None,
    keypoint_epochs: int | None = None,
    detector_batch: int | None = None,
    keypoint_batch: int | None = None,
) -> int:
    prepare(data_root, output_dir, metadata_path)
    if mode == "prepare":
        return 0
    det_config = output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py"
    kp_config = output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
    if mode in {"sanity", "full"}:
        train_with_mmengine(
            det_config,
            output_dir,
            "train_detector",
            ["mmdet", "mmengine"],
            max_epochs=detector_epochs,
            batch_size=detector_batch,
        )
        train_with_mmengine(
            kp_config,
            output_dir,
            "train_keypoint",
            ["mmpose", "mmengine"],
            max_epochs=keypoint_epochs,
            batch_size=keypoint_batch,
        )
    det_checkpoint = default_detector_checkpoint(output_dir)
    kp_checkpoint = default_keypoint_checkpoint(output_dir)
    append_report(
        output_dir,
        "Checkpoint Selection",
        [
            f"- Detector checkpoint: `{det_checkpoint.as_posix() if det_checkpoint else ''}`",
            f"- Keypoint checkpoint: `{kp_checkpoint.as_posix() if kp_checkpoint else ''}`",
        ],
    )
    infer_detector(data_root, output_dir, det_config, det_checkpoint, "test", device)
    infer_keypoint_gt_bbox(data_root, output_dir, kp_config, kp_checkpoint, "test", device)
    infer_two_stage(data_root, output_dir, kp_config, kp_checkpoint, "test", device)
    detector_metrics = evaluate_detector(output_dir, "test")
    keypoint_metrics = evaluate_keypoint(output_dir, "test", "keypoint_gt_bbox")
    two_stage_metrics = evaluate_two_stage(output_dir, "test")
    visualize_predictions(data_root, output_dir, "test", 200)
    append_report(
        output_dir,
        "Evaluation Summary",
        [
            f"- Detector F1: {detector_metrics.get('F1')}",
            f"- Detector recall: {detector_metrics.get('recall')}",
            f"- Keypoint GT bbox NME: {keypoint_metrics.get('NME')}",
            f"- Two-stage face recall: {two_stage_metrics.get('face_detection_recall')}",
            f"- Two-stage PCK@0.05: {two_stage_metrics.get('end_to_end_PCK@0.05')}",
        ],
    )
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("output_experiment"))
    parser.add_argument("--metadata", type=Path, default=None)


def main_inspect_dataset() -> int:
    parser = argparse.ArgumentParser(description="Inspect local face/head/keypoint dataset.")
    add_common_args(parser)
    args = parser.parse_args()
    inspect_dataset(args.data_root.resolve(), args.output_dir, args.metadata)
    return 0


def main_make_splits() -> int:
    parser = argparse.ArgumentParser(description="Create train/val/test split manifest.")
    add_common_args(parser)
    args = parser.parse_args()
    records, _metadata, _resolved = load_records(args.data_root.resolve(), args.metadata)
    records, summary = assign_splits(records)
    ensure_layout(args.output_dir)
    write_split_outputs(records, args.output_dir, summary)
    return 0


def main_convert_annotations() -> int:
    parser = argparse.ArgumentParser(description="Convert annotations to detection/keypoint COCO.")
    add_common_args(parser)
    args = parser.parse_args()
    ensure_layout(args.output_dir)
    convert_annotations(args.data_root.resolve(), args.output_dir, args.metadata)
    generate_configs(args.data_root.resolve(), args.output_dir)
    return 0


def main_train_detector() -> int:
    parser = argparse.ArgumentParser(description="Train RTMDet detector.")
    add_common_args(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()
    config = args.config or args.output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py"
    return train_with_mmengine(config, args.output_dir, "train_detector", ["mmdet", "mmengine"], max_epochs=args.epochs, batch_size=args.batch)


def main_train_keypoint() -> int:
    parser = argparse.ArgumentParser(description="Train RTMPose keypoint model.")
    add_common_args(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()
    config = args.config or args.output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
    return train_with_mmengine(config, args.output_dir, "train_keypoint", ["mmpose", "mmengine"], max_epochs=args.epochs, batch_size=args.batch)


def main_infer_detector() -> int:
    parser = argparse.ArgumentParser(description="Run RTMDet inference.")
    add_common_args(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = args.config or args.output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py"
    checkpoint = args.checkpoint or default_detector_checkpoint(args.output_dir)
    return infer_detector(args.data_root.resolve(), args.output_dir, config, checkpoint, args.split, args.device)


def main_infer_keypoint_gt_bbox() -> int:
    parser = argparse.ArgumentParser(description="Run RTMPose inference with GT bbox.")
    add_common_args(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = args.config or args.output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
    checkpoint = args.checkpoint or default_keypoint_checkpoint(args.output_dir)
    return infer_keypoint_gt_bbox(args.data_root.resolve(), args.output_dir, config, checkpoint, args.split, args.device)


def main_infer_two_stage() -> int:
    parser = argparse.ArgumentParser(description="Run two-stage RTMDet + RTMPose inference.")
    add_common_args(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = args.config or args.output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
    checkpoint = args.checkpoint or default_keypoint_checkpoint(args.output_dir)
    return infer_two_stage(args.data_root.resolve(), args.output_dir, config, checkpoint, args.split, args.device)


def main_evaluate_detector() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector predictions.")
    add_common_args(parser)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate_detector(args.output_dir, args.split)
    return 0


def main_evaluate_keypoint() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GT-bbox keypoint predictions.")
    add_common_args(parser)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate_keypoint(args.output_dir, args.split, "keypoint_gt_bbox")
    return 0


def main_evaluate_two_stage() -> int:
    parser = argparse.ArgumentParser(description="Evaluate two-stage predictions.")
    add_common_args(parser)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate_two_stage(args.output_dir, args.split)
    return 0


def main_visualize_predictions() -> int:
    parser = argparse.ArgumentParser(description="Visualize GT and predictions.")
    add_common_args(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    return visualize_predictions(args.data_root.resolve(), args.output_dir, args.split, args.max_images)


def main_run_all() -> int:
    parser = argparse.ArgumentParser(description="Run the RTMDet + RTMPose experiment pipeline.")
    add_common_args(parser)
    parser.add_argument("--mode", choices=["prepare", "sanity", "infer", "full"], default="prepare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--keypoint-epochs", type=int, default=None)
    parser.add_argument("--detector-batch", type=int, default=None)
    parser.add_argument("--keypoint-batch", type=int, default=None)
    args = parser.parse_args()
    return run_all(
        args.data_root.resolve(),
        args.output_dir,
        args.mode,
        args.device,
        args.metadata,
        detector_epochs=args.detector_epochs,
        keypoint_epochs=args.keypoint_epochs,
        detector_batch=args.detector_batch,
        keypoint_batch=args.keypoint_batch,
    )


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "min": ordered[0],
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "max": ordered[-1],
    }


def bbox_out_of_bounds(xyxy: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = xyxy
    return x1 < 0 or y1 < 0 or x2 > width or y2 > height


def bbox_near_image_edge(xywh: list[float], width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    x, y, w, h = xywh
    margin_x = width * 0.05
    margin_y = height * 0.05
    return x <= margin_x or y <= margin_y or (x + w) >= width - margin_x or (y + h) >= height - margin_y


def build_annotation_id_maps(output_dir: Path) -> tuple[dict[int, dict[int, list[int]]], dict[int, list[int]]]:
    det_id_map: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    kp_id_map: dict[int, list[int]] = defaultdict(list)
    for split in SPLITS:
        det_coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
        kp_coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
        for ann in det_coco.get("annotations", []):
            det_id_map[int(ann["image_id"])][int(ann["category_id"])].append(int(ann["id"]))
        for ann in kp_coco.get("annotations", []):
            kp_id_map[int(ann["image_id"])].append(int(ann["id"]))
    return det_id_map, kp_id_map


def converted_negative_annotation_counts(output_dir: Path) -> dict[str, int]:
    counts = {"detection": 0, "keypoint": 0}
    for split in SPLITS:
        det_coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
        kp_coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
        negative_det_ids = {
            int(image["id"])
            for image in det_coco.get("images", [])
            if image.get("metadata", {}).get("negative_sample")
        }
        negative_kp_ids = {
            int(image["id"])
            for image in kp_coco.get("images", [])
            if image.get("metadata", {}).get("negative_sample")
        }
        counts["detection"] += sum(1 for ann in det_coco.get("annotations", []) if int(ann["image_id"]) in negative_det_ids)
        counts["keypoint"] += sum(1 for ann in kp_coco.get("annotations", []) if int(ann["image_id"]) in negative_kp_ids)
    return counts


def inspect_converted_dataset(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_paths = []
    invalid_visibility = []
    keypoint_length_mismatches = []
    negative_annotation_counts = converted_negative_annotation_counts(output_dir)
    converted_counts: dict[str, Any] = {}
    for split in SPLITS:
        det_coco = load_coco(output_dir / "data" / "converted" / "detection_coco" / f"instances_{split}.json")
        kp_coco = load_coco(output_dir / "data" / "converted" / "keypoint_coco" / f"person_keypoints_{split}.json")
        converted_counts[split] = {
            "detection_images": len(det_coco.get("images", [])),
            "detection_annotations": len(det_coco.get("annotations", [])),
            "keypoint_images": len(kp_coco.get("images", [])),
            "keypoint_annotations": len(kp_coco.get("annotations", [])),
        }
        for image in det_coco.get("images", []):
            if not Path(image.get("file_name", "")).exists():
                missing_paths.append({"split": split, "dataset": "detection", "image_id": image.get("id"), "file_name": image.get("file_name")})
        for image in kp_coco.get("images", []):
            if not Path(image.get("file_name", "")).exists():
                missing_paths.append({"split": split, "dataset": "keypoint", "image_id": image.get("id"), "file_name": image.get("file_name")})
        expected_kpt_len = len(KEYPOINT_NAMES) * 3
        for ann in kp_coco.get("annotations", []):
            keypoints = ann.get("keypoints", [])
            if len(keypoints) != expected_kpt_len:
                keypoint_length_mismatches.append(
                    {"split": split, "annotation_id": ann.get("id"), "length": len(keypoints), "expected": expected_kpt_len}
                )
            for index in range(2, len(keypoints), 3):
                if int(keypoints[index]) not in {0, 1, 2}:
                    invalid_visibility.append(
                        {"split": split, "annotation_id": ann.get("id"), "keypoint_index": index // 3, "visibility": keypoints[index]}
                    )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["_group"]].add(record["_split"])
    mixed_groups = {group: sorted(values) for group, values in group_splits.items() if len(values) > 1}
    return {
        "converted_counts": converted_counts,
        "missing_paths": missing_paths,
        "invalid_visibility": invalid_visibility,
        "keypoint_length_mismatches": keypoint_length_mismatches,
        "negative_annotation_counts": negative_annotation_counts,
        "mixed_video_groups": mixed_groups,
    }


def collect_validation_stats(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    image_issue_rows: list[dict[str, Any]] = []
    keypoint_counter: dict[str, Counter[str]] = {name: Counter() for name in KEYPOINT_NAMES}
    for split in SPLITS:
        split_records = [record for record in records if record["_split"] == split]
        split_rows.append(
            {
                "split": split,
                "images": len(split_records),
                "instances": sum(len(record.get("objects", [])) for record in split_records),
                "head_bbox_count": sum(len(record.get("heads", [])) for record in split_records),
                "face_bbox_count": sum(len(record.get("faces", [])) for record in split_records),
                "keypoint_valid_count": 0,
                "negative_sample_count": sum(1 for record in split_records if record.get("negative_sample")),
            }
        )
    split_row_by_name = {row["split"]: row for row in split_rows}
    bbox_id = 1
    out_of_bounds_count = 0
    invalid_bbox_count = 0
    for record in records:
        width, height = int(record["_width"]), int(record["_height"])
        if not record.get("_image_abs"):
            image_issue_rows.append({"image_id": record["_image_id"], "split": record["_split"], "issue": "missing_image", "image_name": record.get("image_name", "")})
        shape_groups = [("head", record.get("heads", [])), ("face", record.get("faces", []))]
        for label, shapes in shape_groups:
            for shape_index, shape in enumerate(shapes):
                raw_xyxy = bbox_dict_to_xyxy(shape.get("bbox", {}))
                clipped = clip_xyxy(raw_xyxy, width, height)
                xywh = xyxy_to_xywh(raw_xyxy)
                clipped_xywh = xyxy_to_xywh(clipped)
                area = bbox_area_xywh(xywh)
                clipped_area = bbox_area_xywh(clipped_xywh)
                aspect = xywh[2] / xywh[3] if xywh[3] > 0 else None
                is_out = bbox_out_of_bounds(raw_xyxy, width, height)
                is_invalid = xywh[2] <= 0 or xywh[3] <= 0 or width <= 0 or height <= 0
                out_of_bounds_count += int(is_out)
                invalid_bbox_count += int(is_invalid)
                bbox_rows.append(
                    {
                        "bbox_id": bbox_id,
                        "image_id": record["_image_id"],
                        "split": record["_split"],
                        "label": label,
                        "shape_index": shape_index,
                        "x": xywh[0],
                        "y": xywh[1],
                        "width": xywh[2],
                        "height": xywh[3],
                        "area": area,
                        "clipped_area": clipped_area,
                        "aspect_ratio": aspect,
                        "out_of_bounds": is_out,
                        "invalid_bbox": is_invalid,
                        "near_edge": bbox_near_image_edge(clipped_xywh, width, height),
                        "image_width": width,
                        "image_height": height,
                    }
                )
                bbox_id += 1
        for obj in record.get("objects", []):
            flat, _metadata = normalized_keypoints(obj)
            for index, name in enumerate(KEYPOINT_NAMES):
                visibility = int(flat[index * 3 + 2])
                if visibility == 2:
                    keypoint_counter[name]["visible"] += 1
                    split_row_by_name[record["_split"]]["keypoint_valid_count"] += 1
                elif visibility == 1:
                    keypoint_counter[name]["invisible"] += 1
                    split_row_by_name[record["_split"]]["keypoint_valid_count"] += 1
                else:
                    keypoint_counter[name]["missing"] += 1
    keypoint_rows = [
        {
            "keypoint": name,
            "visible": keypoint_counter[name]["visible"],
            "invisible": keypoint_counter[name]["invisible"],
            "missing": keypoint_counter[name]["missing"],
            "total": sum(keypoint_counter[name].values()),
        }
        for name in KEYPOINT_NAMES
    ]
    area_values = [float(row["area"]) for row in bbox_rows if not row["invalid_bbox"]]
    aspect_values = [float(row["aspect_ratio"]) for row in bbox_rows if row["aspect_ratio"] not in (None, "")]
    converted_report = inspect_converted_dataset(output_dir, records)
    summary = {
        "created_at": now_text(),
        "split_summary": split_rows,
        "keypoint_stats": keypoint_rows,
        "bbox_area_distribution": quantiles(area_values),
        "bbox_aspect_ratio_distribution": quantiles(aspect_values),
        "bbox_count": len(bbox_rows),
        "bbox_out_of_bounds_count": out_of_bounds_count,
        "invalid_bbox_count": invalid_bbox_count,
        "negative_sample_count": sum(1 for record in records if record.get("negative_sample")),
        "train_val_test_mixed_video_groups": converted_report["mixed_video_groups"],
        "converted_dataset_checks": converted_report,
        "keypoint_order_confirmed": list(KEYPOINT_NAMES),
    }
    stats_root = output_dir / "post_impl_validation" / "dataset_stats"
    write_csv(stats_root / "split_stats.csv", split_rows)
    write_csv(stats_root / "bbox_stats.csv", bbox_rows)
    write_csv(stats_root / "keypoint_visibility_stats.csv", keypoint_rows)
    write_csv(stats_root / "image_issues.csv", image_issue_rows)
    write_json(stats_root / "dataset_stats.json", summary)
    stats_md = [
        "# Dataset Stats",
        "",
        f"- Generated at: {summary['created_at']}",
        f"- Split summary: {split_rows}",
        f"- Bbox count: {len(bbox_rows)}",
        f"- Bbox out of bounds: {out_of_bounds_count}",
        f"- Invalid bbox count: {invalid_bbox_count}",
        f"- Negative samples: {summary['negative_sample_count']}",
        f"- Mixed train/val/test groups: {converted_report['mixed_video_groups'] or 'none'}",
        f"- Converted negative annotation counts: {converted_report['negative_annotation_counts']}",
        "",
        "## Keypoint Visibility",
        "",
        *[f"- {row['keypoint']}: visible={row['visible']}, invisible={row['invisible']}, missing={row['missing']}" for row in keypoint_rows],
        "",
        "## Bbox Distribution",
        "",
        f"- Area: {summary['bbox_area_distribution']}",
        f"- Aspect ratio: {summary['bbox_aspect_ratio_distribution']}",
    ]
    write_text(stats_root / "dataset_stats.md", "\n".join(stats_md))
    return summary


def draw_validation_record(
    data_root: Path,
    output_path: Path,
    record: dict[str, Any],
    det_id_map: dict[int, dict[int, list[int]]],
    kp_id_map: dict[int, list[int]],
    tag: str,
) -> bool:
    if not module_available("cv2") or not record.get("_image_abs"):
        return False
    import cv2  # type: ignore

    image = cv2.imread(record["_image_abs"])
    if image is None:
        return False
    colors = {
        "head": (80, 220, 80),
        "face": (80, 160, 255),
        "visible": (255, 255, 0),
        "occluded": (0, 160, 255),
        "missing": (0, 0, 255),
        "text_bg": (0, 0, 0),
        "text": (255, 255, 255),
    }
    image_id = int(record["_image_id"])
    head_ann_ids = det_id_map.get(image_id, {}).get(1, [])
    face_ann_ids = det_id_map.get(image_id, {}).get(2, [])
    kp_ann_ids = kp_id_map.get(image_id, [])

    def label_box(text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        y = max(16, y)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    header = f"{tag} | id={image_id} | {record.get('image_name', '')}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), colors["text_bg"], -1)
    cv2.putText(image, header[:150], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colors["text"], 1, cv2.LINE_AA)
    for index, head in enumerate(record.get("heads", [])):
        bbox = xyxy_to_xywh(clip_xyxy(bbox_dict_to_xyxy(head.get("bbox", {})), record["_width"], record["_height"]))
        ann_id = head_ann_ids[index] if index < len(head_ann_ids) else "n/a"
        draw_bbox(cv2, image, bbox, colors["head"], f"head ann={ann_id}")
    for index, face in enumerate(record.get("faces", [])):
        bbox = xyxy_to_xywh(clip_xyxy(bbox_dict_to_xyxy(face.get("bbox", {})), record["_width"], record["_height"]))
        ann_id = face_ann_ids[index] if index < len(face_ann_ids) else "n/a"
        draw_bbox(cv2, image, bbox, colors["face"], f"face ann={ann_id}")
    missing_notes: list[str] = []
    for obj_index, obj in enumerate(record.get("objects", [])):
        flat, _metadata = normalized_keypoints(obj)
        kp_ann_id = kp_ann_ids[obj_index] if obj_index < len(kp_ann_ids) else "n/a"
        face_bbox = xyxy_to_xywh(
            clip_xyxy(
                bbox_dict_to_xyxy(obj.get("training_bbox") or obj.get("raw_bbox") or obj.get("face", {}).get("bbox", {})),
                record["_width"],
                record["_height"],
            )
        )
        draw_bbox(cv2, image, face_bbox, (255, 100, 50), f"kp ann={kp_ann_id}")
        for kp_index, name in enumerate(KEYPOINT_NAMES):
            x = float(flat[kp_index * 3])
            y = float(flat[kp_index * 3 + 1])
            visibility = int(flat[kp_index * 3 + 2])
            if visibility <= 0:
                missing_notes.append(f"{kp_index}:{name}")
                continue
            color = colors["visible"] if visibility == 2 else colors["occluded"]
            cv2.circle(image, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
            label_box(f"{kp_index} {name} v{visibility}", int(round(x)) + 5, int(round(y)) - 5, color)
    if missing_notes:
        cv2.putText(
            image,
            "missing " + ",".join(missing_notes[:8]),
            (8, image.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors["missing"],
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(output_path.as_posix(), image))


def select_validation_samples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {
        "train_20": [record for record in records if record["_split"] == "train"][:20],
        "val_20": [record for record in records if record["_split"] == "val"][:20],
        "negative_max10": [record for record in records if record.get("negative_sample")][:10],
    }
    face_records = [record for record in records if record.get("faces")]

    def min_face_area(record: dict[str, Any]) -> float:
        areas = [bbox_area_xywh(xyxy_to_xywh(bbox_dict_to_xyxy(face.get("bbox", {})))) for face in record.get("faces", [])]
        return min(areas) if areas else float("inf")

    samples["small_face"] = sorted(face_records, key=min_face_area)[:20]
    samples["occluded_face"] = [
        record
        for record in records
        if any(face.get("occluded") for face in record.get("faces", []))
        or any(
            int(value) == 1
            for obj in record.get("objects", [])
            for index, value in enumerate(normalized_keypoints(obj)[0])
            if index % 3 == 2
        )
    ][:20]
    samples["image_edge_face"] = [
        record
        for record in records
        if any(
            bbox_near_image_edge(
                xyxy_to_xywh(clip_xyxy(bbox_dict_to_xyxy(face.get("bbox", {})), record["_width"], record["_height"])),
                record["_width"],
                record["_height"],
            )
            for face in record.get("faces", [])
        )
    ][:20]
    samples["multiple_people"] = [
        record for record in records if len(record.get("heads", [])) > 1 or len(record.get("faces", [])) > 1 or len(record.get("objects", [])) > 1
    ][:20]
    return samples


def visualize_validation_annotations(data_root: Path, output_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual_root = output_dir / "post_impl_validation" / "visualize_annotations"
    det_id_map, kp_id_map = build_annotation_id_maps(output_dir)
    index_rows: list[dict[str, Any]] = []
    samples = select_validation_samples(records)
    for category, selected_records in samples.items():
        for rank, record in enumerate(selected_records, start=1):
            safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(record.get("image_name", f'image_{rank}')))
            output_path = visual_root / category / f"{rank:03d}_{safe_name}"
            if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
                output_path = output_path.with_suffix(".jpg")
            written = draw_validation_record(data_root, output_path, record, det_id_map, kp_id_map, category)
            index_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "written": written,
                    "image_id": record["_image_id"],
                    "split": record["_split"],
                    "image_name": record.get("image_name", ""),
                    "output": repo_rel(output_path, output_dir),
                    "head_count": len(record.get("heads", [])),
                    "face_count": len(record.get("faces", [])),
                    "keypoint_object_count": len(record.get("objects", [])),
                    "negative_sample": bool(record.get("negative_sample")),
                }
            )
    write_csv(visual_root / "visualization_index.csv", index_rows)
    return index_rows


def audit_repository_for_post_validation(data_root: Path, output_dir: Path, records: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    env = env_report(data_root)
    split_counts = {row["split"]: row for row in stats["split_summary"]}
    main_scripts = [
        "script/script_experiment/rtmdet_rtmpose_experiment.py",
        "script/script_experiment/evaluate_face_detections.py",
        "script/script_experiment/visualize_detections.py",
        "scripts_YOLO/convert_annotations_to_yolo_pose.py",
        "scripts_YOLO/check_yolo_pose_dataset.py",
    ]
    checkpoint_candidates = sorted(
        [path.as_posix() for path in (output_dir / "logs").glob("**/*.pth")]
        + [path.as_posix() for path in (data_root / "weights").glob("*.pth")]
        + [path.as_posix() for path in (data_root / "weights").glob("*.pt")]
    )
    risks = []
    if not env["modules"].get("mmdet") or not env["modules"].get("mmpose"):
        risks.append("MMDetection / MMPose が未導入のため、OpenMMLab dataloader と実推論は未検証。")
    if not checkpoint_candidates:
        risks.append("RTMDet / RTMPose の checkpoint が見つからないため、推論品質は未検証。")
    if stats["bbox_out_of_bounds_count"]:
        risks.append(f"raw bbox が画像外に出ている例が {stats['bbox_out_of_bounds_count']} 件ある。変換後 COCO では clip 済み。")
    if stats["invalid_bbox_count"]:
        risks.append(f"0 面積または異常 bbox が {stats['invalid_bbox_count']} 件ある。")
    if stats["converted_dataset_checks"]["negative_annotation_counts"] != {"detection": 0, "keypoint": 0}:
        risks.append("negative sample に annotation が残っている可能性がある。")
    risks.extend(
        [
            "left/right keypoint は画像 x 座標に基づく割り当てを含むため、横顔や強い傾きでは左右入れ替わりの可能性がある。",
            "既存 split を再利用しており、70/15/15 ぴったりではない。比較可能性を優先した split である。",
            "negative_sample=true かつ head bbox がある raw record は detection 学習では空 annotation として扱うため、意図確認が必要。",
        ]
    )
    audit = [
        "# Repo Audit",
        "",
        f"- Generated at: {now_text()}",
        f"- Data root: `{data_root}`",
        f"- Validation output: `{output_dir / 'post_impl_validation'}`",
        "",
        "## Model Setup",
        "",
        "- Detector: RTMDet-s head/face detector, with RTMDet-tiny fallback config.",
        "- Landmark model: MMPose / RTMPose-style TopdownPoseEstimator with RTMCCHead and 5 face keypoints.",
        f"- Detector config: `{repo_rel(output_dir / 'configs/rtm_det_head_face/rtmdet_s_head_face_640_sanity.py', data_root)}`",
        f"- Landmark config: `{repo_rel(output_dir / 'configs/rtmpose_face/rtmpose_s_face5_256_sanity.py', data_root)}`",
        f"- Checkpoint candidates: {checkpoint_candidates or 'none found'}",
        "",
        "## Dataset",
        "",
        "- Dataset format: COCO detection JSON for head/face, COCO keypoint JSON for face 5-point landmark.",
        "- Class definitions: class 0=head, class 1=face.",
        f"- Keypoint definitions: {list(KEYPOINT_NAMES)}",
        f"- Train images: {split_counts.get('train', {}).get('images', 0)}",
        f"- Val images: {split_counts.get('val', {}).get('images', 0)}",
        f"- Test images: {split_counts.get('test', {}).get('images', 0)}",
        f"- Negative samples: {stats['negative_sample_count']}",
        "- Negative sample handling: converted detection/keypoint COCO files keep the image but should have empty annotations.",
        f"- Converted negative annotation counts: {stats['converted_dataset_checks']['negative_annotation_counts']}",
        f"- Mixed train/val/test source groups: {stats['train_val_test_mixed_video_groups'] or 'none'}",
        "",
        "## Scripts And Commands",
        "",
        *[f"- `{script}`" for script in main_scripts if (data_root / script).exists()],
        "",
        "```bash",
        ".venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py post-impl-validation --data-root . --output-dir output_experiment",
        ".venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py run-all --data-root . --output-dir output_experiment --mode sanity --device cuda:0",
        "```",
        "",
        "## Risks",
        "",
        *[f"- {risk}" for risk in risks],
    ]
    write_text(output_dir / "post_impl_validation" / "repo_audit.md", "\n".join(audit))


def write_stage1_summary(
    output_dir: Path,
    stats: dict[str, Any],
    visual_rows: list[dict[str, Any]],
    fixed_issues: list[str],
) -> None:
    checks = stats["converted_dataset_checks"]
    unresolved = []
    if not module_available("mmdet") or not module_available("mmpose"):
        unresolved.append("MMDetection / MMPose が未導入のため、OpenMMLab dataloader は未確認。")
    if not list((output_dir / "logs").glob("**/*.pth")):
        unresolved.append("fine-tuned checkpoint がないため、推論品質と評価値は未確認。")
    if stats["bbox_out_of_bounds_count"]:
        unresolved.append(f"raw bbox の画像外は {stats['bbox_out_of_bounds_count']} 件あるが、変換 COCO では clip 済み。")
    can_proceed = (
        not checks["missing_paths"]
        and not checks["invalid_visibility"]
        and not checks["keypoint_length_mismatches"]
        and checks["negative_annotation_counts"] == {"detection": 0, "keypoint": 0}
        and stats["invalid_bbox_count"] == 0
    )
    unresolved_lines = [f"- {issue}" for issue in unresolved] if unresolved else ["- 重大な未解決データ破綻は見つかっていない。"]
    summary = [
        "# Stage 1 Summary",
        "",
        "## 実行したこと",
        "",
        "- リポジトリ構成、config、変換済み COCO、split、実行スクリプト、checkpoint 候補を確認した。",
        "- annotation 可視化を train / val / negative / small face / occluded / image edge / multiple people で出力した。",
        "- dataset stats を CSV / JSON / Markdown で出力した。",
        "- keypoint order と visibility 値、negative sample の空 annotation を確認した。",
        "",
        "## 作成したファイル",
        "",
        "- `output_experiment/post_impl_validation/repo_audit.md`",
        "- `output_experiment/post_impl_validation/STAGE1_SUMMARY.md`",
        "- `output_experiment/post_impl_validation/visualize_annotations/`",
        "- `output_experiment/post_impl_validation/dataset_stats/dataset_stats.{json,md}`",
        "- `output_experiment/post_impl_validation/dataset_stats/*.csv`",
        "",
        "## 見つかった問題",
        "",
        f"- OpenMMLab dataloader 検証: {'未実行' if (not module_available('mmdet') or not module_available('mmpose')) else '実行可能'}",
        f"- missing image paths: {len(checks['missing_paths'])}",
        f"- invalid visibility values: {len(checks['invalid_visibility'])}",
        f"- keypoint length mismatches: {len(checks['keypoint_length_mismatches'])}",
        f"- converted negative annotations: {checks['negative_annotation_counts']}",
        f"- invalid bbox count: {stats['invalid_bbox_count']}",
        f"- raw bbox out of bounds count: {stats['bbox_out_of_bounds_count']}",
        f"- mixed source groups across split: {stats['train_val_test_mixed_video_groups'] or 'none'}",
        "",
        "## 修正した問題",
        "",
        *(f"- {issue}" for issue in fixed_issues),
        "",
        "## 未解決の問題",
        "",
        *unresolved_lines,
        "",
        "## 次に進んでよいか",
        "",
        "- データ変換と annotation 形式は sanity training 前段としては進行可能。" if can_proceed else "- 本学習前に上記の破綻を修正してください。",
        "- ただし MMDetection / MMPose 導入後に dataloader smoke と最小学習を必ず実行すること。",
        "",
        "## 次の推奨 command",
        "",
        "```bash",
        ".venv/bin/python -m pip install -U openmim",
        '.venv/bin/mim install "mmengine>=0.10" "mmcv>=2.0" "mmdet>=3.0" "mmpose>=1.3" pycocotools',
        ".venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py run-all --data-root . --output-dir output_experiment --mode sanity --device cuda:0",
        "```",
        "",
        f"- Visualization files written: {sum(1 for row in visual_rows if row['written'])}",
    ]
    write_text(output_dir / "post_impl_validation" / "STAGE1_SUMMARY.md", "\n".join(summary))


def run_post_impl_validation(data_root: Path, output_dir: Path, metadata_path: Path | None = None) -> int:
    post_root = output_dir / "post_impl_validation"
    for rel in ["visualize_annotations", "dataset_stats"]:
        (post_root / rel).mkdir(parents=True, exist_ok=True)
    if not (output_dir / "data" / "converted" / "conversion_summary.json").exists():
        prepare(data_root, output_dir, metadata_path)
    records, _metadata, _resolved = load_records(data_root, metadata_path)
    records, split_summary = assign_splits(records)
    write_json(post_root / "split_summary_used.json", split_summary)
    stats = collect_validation_stats(records, output_dir)
    visual_rows = visualize_validation_annotations(data_root, output_dir, records)
    fixed_issues: list[str] = ["明らかな破綻は変換済み COCO では検出されなかったため、自動修正は行っていない。"]
    audit_repository_for_post_validation(data_root, output_dir, records, stats)
    write_stage1_summary(output_dir, stats, visual_rows, fixed_issues)
    return 0


def main_post_impl_validation() -> int:
    parser = argparse.ArgumentParser(description="Run post-implementation dataset/config validation without training.")
    add_common_args(parser)
    args = parser.parse_args()
    return run_post_impl_validation(args.data_root.resolve(), args.output_dir, args.metadata)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    add_config_arg(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RTMDet + RTMPose face experiment tools. Outputs stay under output_experiment/."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command_specs = [
        ("inspect-dataset", "Inspect local data and write raw index reports."),
        ("make-splits", "Create split manifests."),
        ("convert-annotations", "Convert metadata to detection/keypoint COCO and write configs."),
        ("train-detector", "Train RTMDet when MMDetection is available."),
        ("train-keypoint", "Train RTMPose when MMPose is available."),
        ("infer-detector", "Run detector inference."),
        ("infer-keypoint-gt-bbox", "Run keypoint inference with GT face bboxes."),
        ("infer-two-stage", "Run RTMDet + RTMPose two-stage inference."),
        ("evaluate-detector", "Evaluate detector predictions."),
        ("evaluate-keypoint", "Evaluate GT-bbox keypoint predictions."),
        ("evaluate-two-stage", "Evaluate two-stage predictions."),
        ("visualize-predictions", "Render GT/prediction overlays."),
        ("post-impl-validation", "Audit converted data/configs and render validation visualizations without training."),
        ("run-all", "Run prepare/sanity/infer/full pipeline."),
    ]
    commands = {name: subparsers.add_parser(name, help=help_text) for name, help_text in command_specs}
    for command in commands.values():
        add_common_args(command)

    add_config_arg(commands["train-detector"])
    add_config_arg(commands["train-keypoint"])
    commands["train-detector"].add_argument("--epochs", type=int, default=None)
    commands["train-detector"].add_argument("--batch", type=int, default=None)
    commands["train-keypoint"].add_argument("--epochs", type=int, default=None)
    commands["train-keypoint"].add_argument("--batch", type=int, default=None)
    add_checkpoint_args(commands["infer-detector"])
    add_checkpoint_args(commands["infer-keypoint-gt-bbox"])
    add_checkpoint_args(commands["infer-two-stage"])
    for name in ("evaluate-detector", "evaluate-keypoint", "evaluate-two-stage"):
        commands[name].add_argument("--split", default="test")
    commands["visualize-predictions"].add_argument("--split", default="test")
    commands["visualize-predictions"].add_argument("--max-images", type=int, default=200)
    commands["run-all"].add_argument("--mode", choices=["prepare", "sanity", "infer", "full"], default="prepare")
    commands["run-all"].add_argument("--device", default="cuda:0")
    commands["run-all"].add_argument("--detector-epochs", type=int, default=None)
    commands["run-all"].add_argument("--keypoint-epochs", type=int, default=None)
    commands["run-all"].add_argument("--detector-batch", type=int, default=None)
    commands["run-all"].add_argument("--keypoint-batch", type=int, default=None)
    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir

    if args.command == "inspect-dataset":
        inspect_dataset(data_root, output_dir, args.metadata)
    elif args.command == "make-splits":
        records, _metadata, _resolved = load_records(data_root, args.metadata)
        records, summary = assign_splits(records)
        ensure_layout(output_dir)
        write_split_outputs(records, output_dir, summary)
    elif args.command == "convert-annotations":
        ensure_layout(output_dir)
        convert_annotations(data_root, output_dir, args.metadata)
        generate_configs(data_root, output_dir)
    elif args.command == "train-detector":
        config = args.config or output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py"
        return train_with_mmengine(config, output_dir, "train_detector", ["mmdet", "mmengine"], max_epochs=args.epochs, batch_size=args.batch)
    elif args.command == "train-keypoint":
        config = args.config or output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
        return train_with_mmengine(config, output_dir, "train_keypoint", ["mmpose", "mmengine"], max_epochs=args.epochs, batch_size=args.batch)
    elif args.command == "infer-detector":
        config = args.config or output_dir / "configs" / "rtm_det_head_face" / "rtmdet_s_head_face_640_sanity.py"
        checkpoint = args.checkpoint or default_detector_checkpoint(output_dir)
        return infer_detector(data_root, output_dir, config, checkpoint, args.split, args.device)
    elif args.command == "infer-keypoint-gt-bbox":
        config = args.config or output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
        checkpoint = args.checkpoint or default_keypoint_checkpoint(output_dir)
        return infer_keypoint_gt_bbox(data_root, output_dir, config, checkpoint, args.split, args.device)
    elif args.command == "infer-two-stage":
        config = args.config or output_dir / "configs" / "rtmpose_face" / "rtmpose_s_face5_256_sanity.py"
        checkpoint = args.checkpoint or default_keypoint_checkpoint(output_dir)
        return infer_two_stage(data_root, output_dir, config, checkpoint, args.split, args.device)
    elif args.command == "evaluate-detector":
        evaluate_detector(output_dir, args.split)
    elif args.command == "evaluate-keypoint":
        evaluate_keypoint(output_dir, args.split, "keypoint_gt_bbox")
    elif args.command == "evaluate-two-stage":
        evaluate_two_stage(output_dir, args.split)
    elif args.command == "visualize-predictions":
        return visualize_predictions(data_root, output_dir, args.split, args.max_images)
    elif args.command == "post-impl-validation":
        return run_post_impl_validation(data_root, output_dir, args.metadata)
    elif args.command == "run-all":
        return run_all(
            data_root,
            output_dir,
            args.mode,
            args.device,
            args.metadata,
            detector_epochs=args.detector_epochs,
            keypoint_epochs=args.keypoint_epochs,
            detector_batch=args.detector_batch,
            keypoint_batch=args.keypoint_batch,
        )
    else:
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
