import argparse
import json
import math
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
FRAME_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$")
FRAME_PREFIX_PATTERNS = ("__frame_", "_frame_")
SPLITS = ("train", "val", "test")
SPLIT_FEATURE_WEIGHTS = {
    "groups": 1.0,
    "images": 8.0,
    "face_positive_images": 2.0,
    "face_negative_images": 2.0,
    "head_positive_images": 2.0,
    "head_negative_images": 2.0,
    "head_instances": 5.0,
    "face_instances": 4.0,
    "visible_keypoints": 3.0,
    "complete_keypoint_faces": 2.0,
    "incomplete_keypoint_faces": 2.0,
    "occluded_keypoints": 1.0,
    "head_count_0": 1.5,
    "head_count_1": 1.5,
    "head_count_2": 1.5,
    "head_count_3plus": 1.5,
    "face_count_0": 1.0,
    "face_count_1": 1.0,
    "face_count_2": 1.0,
    "face_count_3plus": 1.0,
    "head_area_tiny": 1.5,
    "head_area_small": 1.5,
    "head_area_medium": 1.5,
    "head_area_large": 1.5,
    "face_area_tiny": 1.0,
    "face_area_small": 1.0,
    "face_area_medium": 1.0,
    "face_area_large": 1.0,
    **{f"visible_keypoint_{name}": 1.5 for name in KEYPOINT_NAMES},
    **{f"missing_keypoint_{name}": 1.0 for name in KEYPOINT_NAMES},
    **{f"keypoint_count_{count}": 1.0 for count in range(len(KEYPOINT_NAMES) + 1)},
}


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
        "--annotation-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory searched recursively for CVAT zip files. Can be supplied multiple "
            "times; discovered files are combined with --annotation-zip."
        ),
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
        "--split-strategy",
        choices=("balanced", "random-groups"),
        default="balanced",
        help=(
            "balanced jointly matches image, Head, Face, and keypoint distributions; "
            "random-groups preserves the legacy behavior. Default: balanced."
        ),
    )
    parser.add_argument(
        "--split-search-restarts",
        type=int,
        default=256,
        help="Deterministic randomized restarts for balanced group assignment. Default: 256.",
    )
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


def discover_annotation_zips(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Annotation root does not exist: {root}")
        paths.extend(path for path in root.rglob("*.zip") if path.is_file())
    return sorted(set(paths))


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
        metadata["exclusion_reason"] = "invalid_clipped_face_bbox"
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
    image_stem = str(record.get("image_stem", "")).strip()
    if image_stem:
        return image_stem
    inferred = infer_image_stem(str(record.get("image_name", ""))).strip()
    if inferred:
        return inferred
    annotation_kind = str(record.get("annotation_kind", "")).strip()
    return f"unknown::{annotation_kind}"


def _ratio_map(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    return {split: value / total for split, value in ratios.items()}


def _count_bin(prefix: str, count: int) -> str:
    suffix = str(count) if count < 3 else "3plus"
    return f"{prefix}_count_{suffix}"


def _area_bin(shape: dict[str, Any], width: int, height: int) -> str:
    bbox = shape.get("bbox", {})
    area = max(0.0, float(bbox.get("width", 0.0))) * max(
        0.0, float(bbox.get("height", 0.0))
    )
    image_area = max(1.0, float(width * height))
    ratio = area / image_area
    if ratio < 0.0025:
        return "tiny"
    if ratio < 0.01:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


def record_split_features(record: dict[str, Any]) -> Counter[str]:
    features: Counter[str] = Counter(images=1)
    heads = list(record.get("heads", []))
    faces = list(record.get("faces", []))
    head_count = len(heads)
    face_count = len(faces)
    features["head_positive_images" if heads else "head_negative_images"] += 1
    features["face_positive_images" if faces else "face_negative_images"] += 1
    features["head_instances"] += head_count
    features["face_instances"] += face_count
    features[_count_bin("head", head_count)] += 1
    features[_count_bin("face", face_count)] += 1

    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    for shape in heads:
        features[f"head_area_{_area_bin(shape, width, height)}"] += 1
    for shape in faces:
        features[f"face_area_{_area_bin(shape, width, height)}"] += 1

    if faces:
        face_bboxes = [face["bbox"] for face in faces]
        for face_index in range(len(faces)):
            keypoints = assign_keypoints(record, face_bboxes, face_index)
            visible = []
            for name, point in keypoints.items():
                if point.get("v", 0) > 0 and point_inside_image(point, width, height):
                    visible.append(point)
                    features[f"visible_keypoint_{name}"] += 1
                else:
                    features[f"missing_keypoint_{name}"] += 1
            features["visible_keypoints"] += len(visible)
            features[f"keypoint_count_{len(visible)}"] += 1
            features["occluded_keypoints"] += sum(
                bool(point.get("occluded", False)) for point in visible
            )
            if len(visible) == len(KEYPOINT_NAMES):
                features["complete_keypoint_faces"] += 1
            else:
                features["incomplete_keypoint_faces"] += 1
    return features


def group_split_features(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[split_group(record)].update(record_split_features(record))
    for features in grouped.values():
        features["groups"] = 1
    return dict(grouped)


def split_objective(
    actual: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
) -> float:
    score = 0.0
    for feature, weight in SPLIT_FEATURE_WEIGHTS.items():
        total = float(totals.get(feature, 0.0))
        if total <= 0:
            continue
        for split in SPLITS:
            target_fraction = ratios[split]
            if target_fraction <= 0:
                score += weight * (float(actual[split].get(feature, 0.0)) / total) ** 2
                continue
            actual_fraction = float(actual[split].get(feature, 0.0)) / total
            relative_error = (actual_fraction - target_fraction) / target_fraction
            score += weight * relative_error * relative_error
    return score


def _assignment_counters(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
) -> dict[str, Counter[str]]:
    actual = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        actual[split].update(group_features[group])
    return actual


def _greedy_balanced_assignment(
    groups: list[str],
    forced_test: set[str],
    group_features: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, Counter[str]], float]:
    assignment = {group: "test" for group in forced_test}
    actual = _assignment_counters(assignment, group_features)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda group: (
            -group_features[group]["images"],
            rng.random(),
        )
    )
    enabled_splits = [split for split in SPLITS if ratios[split] > 0]
    for group in shuffled:
        choices = []
        for split in enabled_splits:
            actual[split].update(group_features[group])
            score = split_objective(actual, totals, ratios)
            actual[split].subtract(group_features[group])
            choices.append((score, rng.random(), split))
        _, _, selected = min(choices)
        assignment[group] = selected
        actual[selected].update(group_features[group])
    return assignment, actual, split_objective(actual, totals, ratios)


def _improve_balanced_assignment(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
    forced_test: set[str],
    score: float,
) -> tuple[dict[str, str], dict[str, Counter[str]], float]:
    actual = _assignment_counters(assignment, group_features)
    movable = sorted(set(assignment) - forced_test)
    enabled_splits = [split for split in SPLITS if ratios[split] > 0]

    for _ in range(20):
        changed = False
        group_counts = Counter(assignment.values())
        for group in movable:
            source = assignment[group]
            if group_counts[source] <= 1:
                continue
            best = (score, source)
            for destination in enabled_splits:
                if destination == source:
                    continue
                actual[source].subtract(group_features[group])
                actual[destination].update(group_features[group])
                candidate = split_objective(actual, totals, ratios)
                actual[destination].subtract(group_features[group])
                actual[source].update(group_features[group])
                if candidate < best[0] - 1e-12:
                    best = (candidate, destination)
            if best[1] != source:
                actual[source].subtract(group_features[group])
                actual[best[1]].update(group_features[group])
                assignment[group] = best[1]
                group_counts[source] -= 1
                group_counts[best[1]] += 1
                score = best[0]
                changed = True
        if not changed:
            break

    for _ in range(20):
        best_swap: tuple[float, str, str] | None = None
        for left_index, left in enumerate(movable):
            left_split = assignment[left]
            for right in movable[left_index + 1 :]:
                right_split = assignment[right]
                if left_split == right_split:
                    continue
                actual[left_split].subtract(group_features[left])
                actual[right_split].subtract(group_features[right])
                actual[left_split].update(group_features[right])
                actual[right_split].update(group_features[left])
                candidate = split_objective(actual, totals, ratios)
                actual[right_split].subtract(group_features[left])
                actual[left_split].subtract(group_features[right])
                actual[right_split].update(group_features[right])
                actual[left_split].update(group_features[left])
                if candidate < score - 1e-12 and (
                    best_swap is None or candidate < best_swap[0]
                ):
                    best_swap = (candidate, left, right)
        if best_swap is None:
            break
        score, left, right = best_swap
        left_split = assignment[left]
        right_split = assignment[right]
        actual[left_split].subtract(group_features[left])
        actual[right_split].subtract(group_features[right])
        actual[left_split].update(group_features[right])
        actual[right_split].update(group_features[left])
        assignment[left], assignment[right] = right_split, left_split
    return assignment, actual, score


def _legacy_random_group_assignment(
    groups: list[str],
    forced_test: set[str],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * ratios["train"])
    val_count = round(len(shuffled) * ratios["val"])
    test_count = len(shuffled) - train_count - val_count
    if ratios["val"] > 0 and val_count == 0 and len(shuffled) >= 2:
        val_count = 1
        if test_count > 0:
            test_count -= 1
        else:
            train_count = max(1, train_count - 1)
    if ratios["test"] > 0 and test_count == 0 and len(shuffled) >= 3 and not forced_test:
        test_count = 1
        train_count = max(1, train_count - 1)
    train_count = max(0, min(train_count, len(shuffled)))
    val_count = max(0, min(val_count, len(shuffled) - train_count))
    train_cut = train_count
    val_cut = train_cut + val_count
    assignment = {group: "train" for group in shuffled[:train_cut]}
    assignment.update({group: "val" for group in shuffled[train_cut:val_cut]})
    assignment.update({group: "test" for group in shuffled[val_cut:]})
    assignment.update({group: "test" for group in forced_test})
    return assignment


def build_split_map(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test_kind_contains: list[str],
    strategy: str = "balanced",
    search_restarts: int = 256,
) -> dict[str, str]:
    ratios = _ratio_map(train_ratio, val_ratio, test_ratio)
    forced_test = {
        split_group(record)
        for record in records
        if any(marker in record.get("annotation_kind", "") for marker in test_kind_contains)
    }
    groups = sorted({split_group(record) for record in records} - forced_test)
    if strategy == "random-groups":
        return _legacy_random_group_assignment(groups, forced_test, seed, ratios)
    if strategy != "balanced":
        raise ValueError(f"Unsupported split strategy: {strategy}")
    if search_restarts <= 0:
        raise ValueError("split search restarts must be positive")

    group_features = group_split_features(records)
    totals = Counter()
    for features in group_features.values():
        totals.update(features)
    rng = random.Random(seed)
    best: tuple[float, dict[str, str]] | None = None
    for _ in range(search_restarts):
        assignment, _, score = _greedy_balanced_assignment(
            groups,
            forced_test,
            group_features,
            totals,
            ratios,
            rng,
        )
        if best is None or score < best[0]:
            best = (score, assignment)
    assert best is not None
    assignment, _, _ = _improve_balanced_assignment(
        dict(best[1]),
        group_features,
        totals,
        ratios,
        forced_test,
        best[0],
    )
    return assignment


def split_balance_report(
    records: list[dict[str, Any]],
    split_map: dict[str, str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strategy: str,
    seed: int,
    test_kind_contains: list[str],
) -> dict[str, Any]:
    ratios = _ratio_map(train_ratio, val_ratio, test_ratio)
    group_features = group_split_features(records)
    actual = _assignment_counters(split_map, group_features)
    totals = Counter()
    for features in group_features.values():
        totals.update(features)
    split_rows: dict[str, Any] = {}
    for split in SPLITS:
        counts = {feature: int(actual[split].get(feature, 0)) for feature in SPLIT_FEATURE_WEIGHTS}
        proportions = {
            feature: (
                float(counts[feature]) / float(totals[feature]) if totals.get(feature, 0) else None
            )
            for feature in SPLIT_FEATURE_WEIGHTS
        }
        split_rows[split] = {
            "target_fraction": ratios[split],
            "counts": counts,
            "fractions_of_dataset": proportions,
        }
    forced_test = sorted(
        group
        for group in split_map
        if any(
            split_group(record) == group
            and any(marker in str(record.get("annotation_kind", "")) for marker in test_kind_contains)
            for record in records
        )
    )
    group_concentration: dict[str, Any] = {}
    for feature in SPLIT_FEATURE_WEIGHTS:
        total = int(totals.get(feature, 0))
        ranked = sorted(
            (
                (group, int(features.get(feature, 0)))
                for group, features in group_features.items()
                if features.get(feature, 0) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        group_concentration[feature] = {
            "nonzero_groups": len(ranked),
            "largest_group_fraction": (float(ranked[0][1]) / total if ranked and total else None),
            "top_two_group_fraction": (
                float(sum(count for _, count in ranked[:2])) / total if ranked and total else None
            ),
            "top_groups": [
                {"group": group, "count": count}
                for group, count in ranked[:5]
            ],
        }
    return {
        "strategy": strategy,
        "seed": seed,
        "group_key": "image_stem (annotation_kind is deliberately excluded)",
        "ratios": ratios,
        "objective": split_objective(actual, totals, ratios),
        "feature_weights": SPLIT_FEATURE_WEIGHTS,
        "totals": {feature: int(totals.get(feature, 0)) for feature in SPLIT_FEATURE_WEIGHTS},
        "group_concentration": group_concentration,
        "forced_test_markers": list(test_kind_contains),
        "forced_test_groups": forced_test,
        "splits": split_rows,
    }


def safe_output_name(record: dict[str, Any], image_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.get("annotation_kind", "ann")).strip("_")
    return f"{stem}__{image_path.name}"


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_lines = [
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
    yaml_text = "\n".join(yaml_lines)
    portable_text = "\n".join(["path: .", *yaml_lines[1:]])
    (output_dir / "face_pose.yaml").write_text(yaml_text, encoding="utf-8")
    (output_dir / "face_pose_drive.yaml").write_text(portable_text, encoding="utf-8")


def write_dataset_manifest(output_dir: Path, summary: dict[str, Any]) -> None:
    split_counts = summary["splits"]
    text = "\n".join(
        [
            f"dataset_version: {output_dir.name}",
            "layout: unified-single-image-root",
            "image_root: .",
            "images:",
            "  train: images/train",
            "  val: images/val",
            "  test: images/test",
            "image_counts:",
            f"  train: {split_counts['train']}",
            f"  val: {split_counts['val']}",
            f"  test: {split_counts['test']}",
            "annotations:",
            "  yolo_pose:",
            "    train: labels/train",
            "    val: labels/val",
            "    test: labels/test",
            "    config: face_pose_drive.yaml",
            "    classes: [face]",
            "    keypoints: [e1, e2, n, m1, m2]",
            "  yolo_pose_coco_eval:",
            "    train: train.json",
            "    val: val.json",
            "    test: test.json",
            "    categories: {face: 2}",
            "    source: yolo_pose",
            "  rtmpose_keypoints:",
            "    train: annotations/rtmpose/person_keypoints_train.json",
            "    val: annotations/rtmpose/person_keypoints_val.json",
            "    test: annotations/rtmpose/person_keypoints_test.json",
            "    categories: {face: 1}",
            "    keypoints: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]",
            "    source: canonical_cvat_metadata",
            "  deim_dino:",
            "    train: annotations/deim_dino/face_keypoints_train.json",
            "    val: annotations/deim_dino/face_keypoints_val.json",
            "    test: annotations/deim_dino/face_keypoints_test.json",
            "    categories: {face: 2}",
            "    keypoints: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]",
            "    ellipse: [cx, cy, semi_major, semi_minor, theta_rad]",
            "    source: canonical_cvat_metadata",
            "  coco_head_face:",
            "    train: annotations/rtmdet_hf/train.json",
            "    val: annotations/rtmdet_hf/val.json",
            "    test: annotations/rtmdet_hf/test.json",
            "    categories: {head: 1, face: 2}",
            "metadata:",
            "  conversion: metadata/conversion_metadata.json",
            "  split_balance: metadata/split_balance.json",
            "drive_upload_roots:",
            f"  - {output_dir.name}",
            "",
        ]
    )
    (output_dir / "dataset_manifest.yaml").write_text(text, encoding="utf-8")


def prepare_output(output_dir: Path, clean: bool) -> None:
    if clean:
        for name in ["images", "labels", "metadata", "annotations"]:
            path = output_dir / name
            if path.exists():
                shutil.rmtree(path)
        for name in ["train.json", "val.json", "test.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()
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


def write_dataset_readme(
    output_dir: Path,
    summary: dict[str, Any],
    balance: dict[str, Any],
) -> None:
    ratios = balance["ratios"]
    target_text = " / ".join(
        f"{split} {ratios[split] * 100:.1f}%" for split in SPLITS
    )
    split_lines = []
    for split in SPLITS:
        counts = balance["splits"][split]["counts"]
        split_lines.append(
            f"| {split} | {counts['groups']} | {summary['splits'][split]} | "
            f"{counts['head_instances']} | {counts['face_instances']} | "
            f"{counts['visible_keypoints']} | {counts['complete_keypoint_faces']} | "
            f"{counts['incomplete_keypoint_faces']} |"
        )
    text = "\n".join(
        [
            f"# {output_dir.name}",
            "",
            "Unified training dataset with one shared image tree and task-specific annotations.",
            "Upload this directory alone; no separate Head/Face image dataset is required.",
            "",
            "## Split policy",
            "",
            f"- Strategy: `{balance['strategy']}`",
            f"- Group key: `{balance['group_key']}`",
            f"- Seed: `{balance['seed']}`",
            f"- Target: {target_text}",
            "- Balanced signals: images, positive/negative images, Head/Face counts and size "
            "bins, visible/occluded keypoints, and complete/incomplete five-point faces",
            "- Source-video leakage is prevented before distribution optimization",
            "",
            "| split | videos | images | source Heads | source Faces | visible keypoints | complete 5-point faces | incomplete faces |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *split_lines,
            "",
            "The complete targets, counts, per-split fractions, feature weights, and objective "
            "are stored in `metadata/split_balance.json`.",
            "Rare signals concentrated in one source video cannot always match 80/10/10 "
            "without leakage; `group_concentration` records that constraint explicitly.",
            "",
            "## Preserved behavior from v1",
            "",
            "- Face bbox uses the rotation-aware CVAT ellipse envelope and is clipped to the image.",
            "- Images without Face ellipses remain valid negative samples.",
            "- Keypoints are `e1`, `e2`, `n`, `m1`, `m2`; direct model exports use visibility 0/1/2 for missing/occluded/visible.",
            "- Horizontal flip order is `[1, 0, 2, 4, 3]`.",
            "",
            "## Files",
            "",
            "- `images/{train,val,test}`: the single shared image tree",
            "- `labels/{train,val,test}`: YOLO Pose Face bbox + five-keypoint labels",
            "- `face_pose.yaml`: repository-relative training configuration",
            "- `face_pose_drive.yaml`: portable configuration for use from this directory",
            "- `dataset_manifest.yaml`: all task-specific paths in one portable manifest",
            "- `metadata/conversion_metadata.json`: complete source and conversion metadata",
            "- `metadata/split_balance.json`: split distribution audit",
            "- `{train,val,test}.json`: YOLO Pose evaluation COCO annotations (not used to train other models)",
            "- `annotations/rtmpose/person_keypoints_{train,val,test}.json`: RTMPose専用5点annotation",
            "- `annotations/deim_dino/face_keypoints_{train,val,test}.json`: DEIM/DINO専用bbox + ellipse + 5点annotation",
            "- `annotations/rtmdet_hf/{train,val,test}.json`: COCO Head + Face annotations",
            "",
            "## Training entry points",
            "",
            "- YOLO Pose: `face_pose_drive.yaml`",
            "- RTMPose: `configs/rtmpose/rtmpose_s_face_pose_drive_v2_256.py`",
            "- DINOv3/DEIMv2: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full.yaml`",
            "- RTMDet Head+Face: `configs/rtmdet/rtmdet_s_hf_drive_v2_960.py`",
            "",
            "## Rebuild the unified dataset",
            "",
            "```bash",
            ".venv/bin/python scripts_YOLO/convert_annotations_to_yolo_pose.py \\",
            "  --annotation-root data/output \\",
            "  --image-root data/output \\",
            f"  --output-dir {output_dir.as_posix()} \\",
            "  --seed 42 --split-strategy balanced --split-search-restarts 256 --clean",
            ".venv/bin/python script/script_data/export_face_keypoints_rtmpose.py \\",
            f"  {output_dir.as_posix()}",
            ".venv/bin/python script/script_data/export_face_keypoints_deim_dino.py \\",
            f"  {output_dir.as_posix()}",
            ".venv/bin/python script/script_data/convert_face_pose_metadata_to_rtmdet_hf.py \\",
            f"  {output_dir.as_posix()} {output_dir.as_posix()}/annotations/rtmdet_hf \\",
            "  --dataset-version face_pose_drive_v2_head_face --image-mode reference",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def convert(args: argparse.Namespace) -> dict[str, Any]:
    discovered_zips = discover_annotation_zips(list(getattr(args, "annotation_root", [])))
    annotation_zips = list(
        dict.fromkeys([*getattr(args, "annotation_zip", []), *discovered_zips])
    )
    records: list[dict[str, Any]] = []
    for path in args.ground_truth_json:
        records.extend(load_records_from_json(path))
    for path in annotation_zips:
        records.extend(load_records_from_zip(path))
    for path in args.annotation_xml:
        records.extend(load_records_from_xml(path))
    if not records:
        raise ValueError("No annotation records were loaded")
    if args.max_images is not None:
        records = records[: args.max_images]

    source_image_roots = [path.parent for path in [*annotation_zips, *args.annotation_xml]]
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
        strategy=getattr(args, "split_strategy", "balanced"),
        search_restarts=getattr(args, "split_search_restarts", 256),
    )
    balance = split_balance_report(
        records,
        split_map,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        strategy=getattr(args, "split_strategy", "balanced"),
        seed=args.seed,
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
        "excluded_faces": 0,
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
            "excluded_faces": [],
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
                summary["excluded_faces"] += 1
                record_meta["excluded_faces"].append(object_meta)
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
    write_dataset_manifest(args.output_dir, summary)
    metadata = {
        "schema_version": "2.0",
        "keypoint_names": KEYPOINT_NAMES,
        "bbox_source": (
            "Rotation-aware Face ellipse bbox from cx/cy/rx/ry/rotation, "
            "then clipped to image bounds; model-specific exporters consume this canonical target"
        ),
        "visibility_policy": (
            "Canonical records preserve CVAT occluded; model exports map missing/outside=0, "
            "occluded=1, visible=2. YOLO Pose labels use their own visibility encoding."
        ),
        "source": {
            "ground_truth_json": [str(path) for path in args.ground_truth_json],
            "annotation_zip": [str(path) for path in annotation_zips],
            "annotation_xml": [str(path) for path in args.annotation_xml],
            "annotation_root": [str(path) for path in getattr(args, "annotation_root", [])],
        },
        "split_balance": balance,
        "summary": summary,
        "records": metadata_records,
    }
    metadata_path = args.output_dir / "metadata" / "conversion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    balance_path = args.output_dir / "metadata" / "split_balance.json"
    balance_path.write_text(json.dumps(balance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_dataset_readme(args.output_dir, summary, balance)
    return {
        "summary": summary,
        "split_balance_objective": balance["objective"],
        "metadata": str(metadata_path),
        "split_balance": str(balance_path),
        "yaml": str(args.output_dir / "face_pose.yaml"),
    }


def main() -> int:
    args = parse_args()
    result = convert(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
