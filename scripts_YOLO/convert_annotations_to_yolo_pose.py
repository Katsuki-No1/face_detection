import argparse
import json
import math
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
FRAME_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$")
FRAME_PREFIX_PATTERNS = ("__frame_", "_frame_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CVAT face annotations into a YOLO Pose dataset."
    )
    parser.add_argument(
        "--ground-truth-json",
        type=Path,
        action="append",
        default=[],
        help="Existing converted ground-truth JSON. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--annotation-zip",
        type=Path,
        action="append",
        default=[],
        help="CVAT zip containing annotations.xml. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--annotation-xml",
        type=Path,
        action="append",
        default=[],
        help="CVAT annotations.xml file. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        help="Directory to search by image filename when image_path is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/face_pose"),
        help="YOLO dataset output directory. Default: datasets/face_pose.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-kind-contains",
        action="append",
        default=[],
        help="If annotation_kind contains this text, force the record into test.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image-record limit for smoke conversion.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing images/ labels/ metadata before writing output.",
    )
    return parser.parse_args()


def float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.get(name)
    return default if value is None else float(value)


def int_attr(element: ET.Element, name: str, default: int = 0) -> int:
    value = element.get(name)
    return default if value is None else int(value)


def parse_points(points_text: str) -> list[dict[str, float]]:
    points = []
    for pair in points_text.split(";"):
        if not pair:
            continue
        x_text, y_text = pair.split(",", 1)
        points.append({"x": float(x_text), "y": float(y_text)})
    return points


def frame_time_seconds(image_name: str) -> float | None:
    match = FRAME_TIME_RE.search(image_name)
    return None if match is None else float(match.group(1))


def infer_image_stem(image_name: str) -> str:
    for pattern in FRAME_PREFIX_PATTERNS:
        if pattern in image_name:
            return image_name.split(pattern, 1)[0]
    return Path(image_name).stem


def common_shape_fields(element: ET.Element) -> dict[str, Any]:
    return {
        "label": element.get("label", ""),
        "source": element.get("source", ""),
        "occluded": bool(int_attr(element, "occluded", 0)),
        "z_order": int_attr(element, "z_order", 0),
    }


def bbox_from_xyxy(xtl: float, ytl: float, xbr: float, ybr: float) -> dict[str, float]:
    return {
        "xtl": xtl,
        "ytl": ytl,
        "xbr": xbr,
        "ybr": ybr,
        "width": xbr - xtl,
        "height": ybr - ytl,
    }


def rotated_ellipse_bbox(cx: float, cy: float, rx: float, ry: float, rotation: float) -> dict[str, float]:
    theta = math.radians(rotation)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    half_width = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
    half_height = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
    return bbox_from_xyxy(cx - half_width, cy - half_height, cx + half_width, cy + half_height)


def convert_box(element: ET.Element) -> dict[str, Any]:
    return {
        **common_shape_fields(element),
        "type": "box",
        "bbox": bbox_from_xyxy(
            float_attr(element, "xtl"),
            float_attr(element, "ytl"),
            float_attr(element, "xbr"),
            float_attr(element, "ybr"),
        ),
    }


def convert_ellipse(element: ET.Element) -> dict[str, Any]:
    cx = float_attr(element, "cx")
    cy = float_attr(element, "cy")
    rx = float_attr(element, "rx")
    ry = float_attr(element, "ry")
    rotation = float_attr(element, "rotation", 0.0)
    return {
        **common_shape_fields(element),
        "type": "ellipse",
        "cvat_ellipse": {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "rotation": rotation},
        "center": {"x": cx, "y": cy},
        "radius": {"x": rx, "y": ry},
        "rotation": rotation,
        "bbox": rotated_ellipse_bbox(cx, cy, rx, ry, rotation),
    }


def convert_points(element: ET.Element) -> dict[str, Any]:
    return {
        **common_shape_fields(element),
        "type": "points",
        "points": parse_points(element.get("points", "")),
    }


def convert_skeleton(element: ET.Element) -> dict[str, Any]:
    points = []
    for point in element.findall("points"):
        parsed = parse_points(point.get("points", ""))
        if not parsed:
            continue
        points.append(
            {
                "label": point.get("label", ""),
                "x": parsed[0]["x"],
                "y": parsed[0]["y"],
                "outside": bool(int_attr(point, "outside", 0)),
                "occluded": bool(int_attr(point, "occluded", 0)),
            }
        )
    return {**common_shape_fields(element), "type": "skeleton", "points": points}


def convert_cvat_image(
    image: ET.Element,
    annotation_kind: str,
    annotation_xml: str,
) -> dict[str, Any]:
    image_name = image.get("name", "")
    record = {
        "annotation_kind": annotation_kind,
        "annotation_mode": "cvat",
        "annotation_xml": annotation_xml,
        "image_stem": infer_image_stem(image_name),
        "image_name": image_name,
        "image_path": "",
        "width": int_attr(image, "width", 0),
        "height": int_attr(image, "height", 0),
        "time_seconds": frame_time_seconds(image_name),
        "heads": [],
        "faces": [],
        "simple_landmarks": [],
        "skeleton_landmarks": [],
    }
    for shape in image:
        if shape.tag == "box":
            record["heads"].append(convert_box(shape))
        elif shape.tag == "ellipse":
            record["faces"].append(convert_ellipse(shape))
        elif shape.tag == "points":
            record["simple_landmarks"].append(convert_points(shape))
        elif shape.tag == "skeleton":
            record["skeleton_landmarks"].append(convert_skeleton(shape))
    return record


def load_records_from_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    records = data.get("records", [])
    for record in records:
        record.setdefault("annotation_kind", path.stem)
    return records


def load_records_from_xml_text(xml_text: str, annotation_kind: str, annotation_xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    return [
        convert_cvat_image(image, annotation_kind=annotation_kind, annotation_xml=annotation_xml)
        for image in root.findall(".//image")
    ]


def load_records_from_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith("annotations.xml")]
        if not xml_names:
            raise ValueError(f"{path} does not contain annotations.xml")
        xml_text = archive.read(xml_names[0]).decode("utf-8")
    return load_records_from_xml_text(xml_text, annotation_kind=path.stem, annotation_xml=str(path))


def load_records_from_xml(path: Path) -> list[dict[str, Any]]:
    return load_records_from_xml_text(
        path.read_text(encoding="utf-8"),
        annotation_kind=path.parent.name,
        annotation_xml=str(path),
    )


def build_image_index(image_roots: list[Path]) -> dict[str, Path]:
    index = {}
    for root in image_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
    return index


def resolve_image_path(record: dict[str, Any], image_index: dict[str, Path]) -> Path | None:
    raw_image_path = record.get("image_path", "")
    image_path = Path(raw_image_path) if raw_image_path else None
    if image_path is not None and image_path.is_file():
        return image_path
    return image_index.get(record.get("image_name", ""))


def point_in_bbox(point: dict[str, float], bbox: dict[str, float]) -> bool:
    return bbox["xtl"] <= point["x"] <= bbox["xbr"] and bbox["ytl"] <= point["y"] <= bbox["ybr"]


def bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return ((bbox["xtl"] + bbox["xbr"]) / 2.0, (bbox["ytl"] + bbox["ybr"]) / 2.0)


def point_distance_to_bbox_center(point: dict[str, float], bbox: dict[str, float]) -> float:
    cx, cy = bbox_center(bbox)
    return math.hypot(point["x"] - cx, point["y"] - cy)


def point_belongs_to_face(
    point: dict[str, float],
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> bool:
    containing_indices = [
        index for index, bbox in enumerate(face_bboxes) if point_in_bbox(point, bbox)
    ]
    if face_index not in containing_indices:
        return False
    nearest_index = min(
        containing_indices,
        key=lambda index: (point_distance_to_bbox_center(point, face_bboxes[index]), index),
    )
    return nearest_index == face_index


def normalize_label(label: str) -> str:
    return label.strip().lower()


def skeleton_label_to_keypoint(label: str) -> str | None:
    normalized = normalize_label(label)
    return {
        "e1": "e1",
        "eye_1": "e1",
        "right_eye": "e1",
        "e2": "e2",
        "eye_2": "e2",
        "left_eye": "e2",
        "n": "n",
        "nose": "n",
        "m1": "m1",
        "mouth_1": "m1",
        "right_mouth": "m1",
        "m2": "m2",
        "mouth_2": "m2",
        "left_mouth": "m2",
    }.get(normalized)


def empty_keypoints() -> dict[str, dict[str, Any]]:
    return {name: {"x": 0.0, "y": 0.0, "v": 0, "occluded": None} for name in KEYPOINT_NAMES}


def assign_from_skeletons(
    record: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> dict[str, dict[str, Any]]:
    keypoints = empty_keypoints()
    face_bbox = face_bboxes[face_index]
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in KEYPOINT_NAMES}
    for skeleton in record.get("skeleton_landmarks", []):
        for point in skeleton.get("points", []):
            if point.get("outside", False):
                continue
            key = skeleton_label_to_keypoint(point.get("label", ""))
            if key is None:
                continue
            if point_belongs_to_face(point, face_bboxes, face_index):
                candidates[key].append(point)
    for key, points in candidates.items():
        if not points:
            continue
        selected = min(points, key=lambda point: point_distance_to_bbox_center(point, face_bbox))
        keypoints[key] = {
            "x": selected["x"],
            "y": selected["y"],
            "v": 2,
            "occluded": bool(selected.get("occluded", False)),
        }
    return keypoints


def simple_points_for_label(
    record: dict[str, Any],
    label: str,
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> list[dict[str, Any]]:
    candidates = []
    face_bbox = face_bboxes[face_index]
    for shape in record.get("simple_landmarks", []):
        if normalize_label(shape.get("label", "")) != label:
            continue
        for point in shape.get("points", []):
            if point_belongs_to_face(point, face_bboxes, face_index):
                candidates.append(
                    {
                        "x": point["x"],
                        "y": point["y"],
                        "occluded": bool(shape.get("occluded", False)),
                    }
                )
    candidates.sort(key=lambda point: (point_distance_to_bbox_center(point, face_bbox), point["x"]))
    return candidates


def assign_from_simple_points(
    record: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> dict[str, dict[str, Any]]:
    keypoints = empty_keypoints()
    face_bbox = face_bboxes[face_index]
    eyes = sorted(
        simple_points_for_label(record, "eye", face_bboxes, face_index)[:2],
        key=lambda point: point["x"],
    )
    mouths = sorted(
        simple_points_for_label(record, "mouth", face_bboxes, face_index)[:2],
        key=lambda point: point["x"],
    )
    noses = simple_points_for_label(record, "nose", face_bboxes, face_index)[:1]

    for key, point in zip(["e1", "e2"], eyes):
        keypoints[key] = {"x": point["x"], "y": point["y"], "v": 2, "occluded": point["occluded"]}
    for key, point in zip(["m1", "m2"], mouths):
        keypoints[key] = {"x": point["x"], "y": point["y"], "v": 2, "occluded": point["occluded"]}
    if noses:
        point = noses[0]
        keypoints["n"] = {"x": point["x"], "y": point["y"], "v": 2, "occluded": point["occluded"]}
    return keypoints


def assign_keypoints(
    record: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> dict[str, dict[str, Any]]:
    skeleton_keypoints = assign_from_skeletons(record, face_bboxes, face_index)
    if any(point["v"] for point in skeleton_keypoints.values()):
        return skeleton_keypoints
    return assign_from_simple_points(record, face_bboxes, face_index)


def normalize_bbox(bbox: dict[str, float], width: int, height: int) -> list[float]:
    xc = ((bbox["xtl"] + bbox["xbr"]) / 2.0) / width
    yc = ((bbox["ytl"] + bbox["ybr"]) / 2.0) / height
    bw = (bbox["xbr"] - bbox["xtl"]) / width
    bh = (bbox["ybr"] - bbox["ytl"]) / height
    return [xc, yc, bw, bh]


def clip_bbox_to_image(bbox: dict[str, float], width: int, height: int) -> dict[str, float]:
    return bbox_from_xyxy(
        max(0.0, min(float(width), bbox["xtl"])),
        max(0.0, min(float(height), bbox["ytl"])),
        max(0.0, min(float(width), bbox["xbr"])),
        max(0.0, min(float(height), bbox["ybr"])),
    )


def bboxes_differ(first: dict[str, float], second: dict[str, float]) -> bool:
    return any(abs(first[key] - second[key]) > 1e-6 for key in ["xtl", "ytl", "xbr", "ybr"])


def out_of_bounds(value: float) -> bool:
    return value < 0.0 or value > 1.0


def point_inside_image(point: dict[str, Any], width: int, height: int) -> bool:
    return 0.0 <= point["x"] <= float(width) and 0.0 <= point["y"] <= float(height)


def yolo_line_for_face(
    record: dict[str, Any],
    face: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
) -> tuple[str | None, dict[str, Any]]:
    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    metadata = {
        "warnings": [],
        "keypoints": {},
        "face": face,
        "cvat_ellipse": face.get("cvat_ellipse"),
    }
    if width <= 0 or height <= 0:
        metadata["warnings"].append("invalid_image_size")
        return None, metadata
    bbox = face["bbox"]
    clipped_bbox = clip_bbox_to_image(bbox, width, height)
    metadata["raw_bbox"] = bbox
    metadata["training_bbox"] = clipped_bbox
    if bboxes_differ(bbox, clipped_bbox):
        metadata["warnings"].append("face_bbox_clipped_to_image")
    bbox_values = normalize_bbox(clipped_bbox, width, height)
    if bbox_values[2] <= 0.0 or bbox_values[3] <= 0.0:
        metadata["warnings"].append("invalid_clipped_face_bbox")
        return None, metadata

    keypoints = assign_keypoints(record, face_bboxes, face_index)
    values: list[float | int] = [0, *bbox_values]
    for name in KEYPOINT_NAMES:
        point = keypoints[name]
        metadata["keypoints"][name] = point
        if point["v"] == 0:
            values.extend([0.0, 0.0, 0])
            continue
        if not point_inside_image(point, width, height):
            metadata["warnings"].append(f"{name}_outside_image")
            values.extend([0.0, 0.0, 0])
            continue
        nx = point["x"] / width
        ny = point["y"] / height
        values.extend([nx, ny, 2])

    return " ".join(format_yolo_value(value) for value in values), metadata


def format_yolo_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}".rstrip("0").rstrip(".")


def split_group(record: dict[str, Any]) -> str:
    annotation_kind = record.get("annotation_kind", "")
    image_stem = record.get("image_stem", "")
    if image_stem:
        return f"{annotation_kind}::{image_stem}"
    return f"{annotation_kind}::{infer_image_stem(record.get('image_name', ''))}"


def build_split_map(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test_kind_contains: list[str],
) -> dict[str, str]:
    forced_test = {
        split_group(record)
        for record in records
        if any(marker in record.get("annotation_kind", "") for marker in test_kind_contains)
    }
    groups = sorted({split_group(record) for record in records} - forced_test)
    rng = random.Random(seed)
    rng.shuffle(groups)
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive number")
    train_count = round(len(groups) * train_ratio / total_ratio)
    val_count = round(len(groups) * val_ratio / total_ratio)
    test_count = len(groups) - train_count - val_count
    if val_ratio > 0 and val_count == 0 and len(groups) >= 2:
        val_count = 1
        if test_count > 0:
            test_count -= 1
        else:
            train_count = max(1, train_count - 1)
    if test_ratio > 0 and test_count == 0 and len(groups) >= 3 and not forced_test:
        test_count = 1
        train_count = max(1, train_count - 1)
    train_count = max(0, min(train_count, len(groups)))
    val_count = max(0, min(val_count, len(groups) - train_count))
    train_cut = train_count
    val_cut = train_cut + val_count
    split_map = {group: "train" for group in groups[:train_cut]}
    split_map.update({group: "val" for group in groups[train_cut:val_cut]})
    split_map.update({group: "test" for group in groups[val_cut:]})
    split_map.update({group: "test" for group in forced_test})
    return split_map


def safe_output_name(record: dict[str, Any], image_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.get("annotation_kind", "ann")).strip("_")
    return f"{stem}__{image_path.name}"


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: face",
            "kpt_shape: [5, 3]",
            "flip_idx: [1, 0, 2, 4, 3]",
            f"keypoints: [{', '.join(KEYPOINT_NAMES)}]",
            "",
        ]
    )
    (output_dir / "face_pose.yaml").write_text(yaml_text, encoding="utf-8")


def prepare_output(output_dir: Path, clean: bool) -> None:
    if clean:
        for name in ["images", "labels", "metadata"]:
            path = output_dir / name
            if path.exists():
                shutil.rmtree(path)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def write_yolo_sample(
    output_dir: Path,
    split: str,
    record: dict[str, Any],
    image_path: Path,
    lines: list[str],
) -> tuple[str, str]:
    output_name = safe_output_name(record, image_path)
    shutil.copy2(image_path, output_dir / "images" / split / output_name)
    label_name = f"{Path(output_name).stem}.txt"
    label_text = "\n".join(lines)
    if label_text:
        label_text += "\n"
    (output_dir / "labels" / split / label_name).write_text(label_text, encoding="utf-8")
    return str(Path("images") / split / output_name), str(Path("labels") / split / label_name)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in args.ground_truth_json:
        records.extend(load_records_from_json(path))
    for path in args.annotation_zip:
        records.extend(load_records_from_zip(path))
    for path in args.annotation_xml:
        records.extend(load_records_from_xml(path))
    if args.max_images is not None:
        records = records[: args.max_images]

    source_image_roots = [path.parent for path in [*args.annotation_zip, *args.annotation_xml]]
    configured_image_roots = args.image_root or [
        Path("data/output"),
        Path("experiments/scrfd_mediapipe_baseline/images"),
        Path("output/cvat_upload"),
    ]
    image_roots = list(dict.fromkeys([*source_image_roots, *configured_image_roots]))
    image_index = build_image_index(image_roots)
    split_map = build_split_map(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        test_kind_contains=args.test_kind_contains,
    )
    prepare_output(args.output_dir, clean=args.clean)

    summary = {
        "records": len(records),
        "written_images": 0,
        "negative_images": 0,
        "written_objects": 0,
        "skipped_records": 0,
        "skipped_faces": 0,
        "splits": {"train": 0, "val": 0, "test": 0},
    }
    metadata_records = []

    for record in records:
        split = split_map.get(split_group(record), "train")
        image_path = resolve_image_path(record, image_index)
        record_meta = {
            "annotation_kind": record.get("annotation_kind", ""),
            "image_stem": record.get("image_stem", ""),
            "image_name": record.get("image_name", ""),
            "image_path": str(image_path) if image_path else record.get("image_path", ""),
            "split": split,
            "heads": record.get("heads", []),
            "faces": record.get("faces", []),
            "objects": [],
            "skipped": [],
        }
        if image_path is None:
            record_meta["skipped"].append("image_not_found")
            summary["skipped_records"] += 1
            metadata_records.append(record_meta)
            continue
        if not record.get("faces"):
            output_image, output_label = write_yolo_sample(
                args.output_dir,
                split,
                record,
                image_path,
                lines=[],
            )
            record_meta["negative_sample"] = True
            record_meta["warnings"] = ["no_face_ellipse_negative_sample"]
            record_meta["output_image"] = output_image
            record_meta["output_label"] = output_label
            summary["written_images"] += 1
            summary["negative_images"] += 1
            summary["splits"][split] += 1
            metadata_records.append(record_meta)
            continue

        lines = []
        face_bboxes = [face["bbox"] for face in record.get("faces", [])]
        for face_index, face in enumerate(record.get("faces", [])):
            line, object_meta = yolo_line_for_face(record, face, face_bboxes, face_index)
            object_meta["face_index"] = face_index
            if line is None:
                summary["skipped_faces"] += 1
                record_meta["objects"].append(object_meta)
                continue
            lines.append(line)
            record_meta["objects"].append(object_meta)

        if not lines:
            record_meta["skipped"].append("no_valid_face_objects")
            summary["skipped_records"] += 1
            metadata_records.append(record_meta)
            continue

        output_image, output_label = write_yolo_sample(args.output_dir, split, record, image_path, lines)
        record_meta["negative_sample"] = False
        record_meta["output_image"] = output_image
        record_meta["output_label"] = output_label
        summary["written_images"] += 1
        summary["written_objects"] += len(lines)
        summary["splits"][split] += 1
        metadata_records.append(record_meta)

    write_dataset_yaml(args.output_dir)
    metadata = {
        "schema_version": "1.0",
        "keypoint_names": KEYPOINT_NAMES,
        "bbox_source": (
            "Rotation-aware Face ellipse bbox from cx/cy/rx/ry/rotation, "
            "then clipped to image bounds for YOLO labels"
        ),
        "visibility_policy": "0 for missing or outside-image keypoints, 2 for present in-image keypoints",
        "summary": summary,
        "records": metadata_records,
    }
    metadata_path = args.output_dir / "metadata" / "conversion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "metadata": str(metadata_path), "yaml": str(args.output_dir / "face_pose.yaml")}


def main() -> int:
    args = parse_args()
    result = convert(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
