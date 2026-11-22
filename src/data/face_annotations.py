"""CVAT face annotation parsing, ellipse geometry and semantic point assignment.

Coordinates stay in original-image pixels. Source occluded/outside flags and
legacy e1/e2/n/m1/m2 slot ordering are preserved; no model runtime is required.
New assignments apply the versioned unique-ellipse up/down side policy. Use
legacy_image_x explicitly when reproducing historical label conversion.
"""

import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from src.data.keypoint_side_assignment import POLICY, V1_POLICY, V2_POLICY, LEGACY_POLICY, side_permutation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
FRAME_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$")
FRAME_PREFIX_PATTERNS = ("__frame_", "_frame_")


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
    *, side_policy: str = POLICY,
) -> dict[str, dict[str, Any]]:
    return assign_keypoints_with_audit(record, face_bboxes, face_index, side_policy=side_policy)[0]


def assign_keypoints_with_audit(
    record: dict[str, Any],
    face_bboxes: list[dict[str, float]],
    face_index: int,
    *, side_policy: str = POLICY,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    skeleton_keypoints = assign_from_skeletons(record, face_bboxes, face_index)
    if any(point["v"] for point in skeleton_keypoints.values()):
        keypoints = skeleton_keypoints
    else:
        keypoints = assign_from_simple_points(record, face_bboxes, face_index)
    if side_policy not in (POLICY, V1_POLICY, V2_POLICY, LEGACY_POLICY):
        raise ValueError(f"Unknown keypoint side policy: {side_policy}")
    if side_policy == LEGACY_POLICY:
        return keypoints, dict(policy=side_policy, status="legacy_unchanged", permutation=list(range(5)))
    ellipses = []
    for face in record.get("faces", []):
        e = face.get("cvat_ellipse")
        if e is None:
            return keypoints, dict(policy=side_policy, status="missing_ellipse_geometry", permutation=list(range(5)))
        ellipses.append([e["cx"], e["cy"], e["rx"], e["ry"], math.radians(e["rotation"])])
    if len(ellipses) != len(face_bboxes):
        return keypoints, dict(policy=side_policy, status="ellipse_inventory_mismatch", permutation=list(range(5)))
    points = [[keypoints[k]["x"], keypoints[k]["y"]] for k in KEYPOINT_NAMES]
    # Keep annotation state attached to its physical point during a permutation.
    states = [0 if keypoints[k]["v"] == 0 else 1 if keypoints[k]["occluded"] else 2 for k in KEYPOINT_NAMES]
    # Match native-export state0 handling: outside-image annotations must not
    # provide orientation evidence that the final supervised GT does not have.
    width, height = record.get("width", 0), record.get("height", 0)
    if width > 0 and height > 0:
        states = [s if point_inside_image(keypoints[k], width, height) else 0
                  for k,s in zip(KEYPOINT_NAMES, states)]
    permutation, audit = side_permutation(points, states, ellipses, face_index, policy=side_policy)
    return {k: keypoints[KEYPOINT_NAMES[permutation[i]]].copy() for i,k in enumerate(KEYPOINT_NAMES)}, audit


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


def safe_output_name(record: dict[str, Any], image_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.get("annotation_kind", "ann")).strip("_")
    return f"{stem}__{image_path.name}"
