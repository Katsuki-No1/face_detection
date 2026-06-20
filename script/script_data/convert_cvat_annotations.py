import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ANNOTATION_PREFIXES = ("an_simple_", "an_skeleton_")
FRAME_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML annotations into machine-readable ground truth JSON."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline"),
        help="Experiment directory that contains images/ and annotations/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/ground_truth.json"),
        help="Path to write ground truth JSON.",
    )
    return parser.parse_args()


def infer_annotation_mode(annotation_dir: Path) -> str:
    name = annotation_dir.name
    if name.startswith("an_simple_"):
        return "simple"
    if name.startswith("an_skeleton_"):
        return "skeleton"
    return "unknown"


def infer_image_stem(annotation_dir: Path) -> str:
    name = annotation_dir.name
    for prefix in ANNOTATION_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def frame_time_seconds(image_name: str) -> float | None:
    match = FRAME_TIME_RE.search(image_name)
    if match is None:
        return None
    return float(match.group(1))


def float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.get(name)
    if value is None:
        return default
    return float(value)


def int_attr(element: ET.Element, name: str, default: int = 0) -> int:
    value = element.get(name)
    if value is None:
        return default
    return int(value)


def parse_points(points_text: str) -> list[dict[str, float]]:
    points = []
    for pair in points_text.split(";"):
        if not pair:
            continue
        x_text, y_text = pair.split(",", 1)
        points.append({"x": float(x_text), "y": float(y_text)})
    return points


def common_shape_fields(element: ET.Element) -> dict[str, Any]:
    return {
        "label": element.get("label", ""),
        "source": element.get("source", ""),
        "occluded": bool(int_attr(element, "occluded", 0)),
        "z_order": int_attr(element, "z_order", 0),
    }


def convert_box(element: ET.Element) -> dict[str, Any]:
    xtl = float_attr(element, "xtl")
    ytl = float_attr(element, "ytl")
    xbr = float_attr(element, "xbr")
    ybr = float_attr(element, "ybr")
    return {
        **common_shape_fields(element),
        "type": "box",
        "bbox": {
            "xtl": xtl,
            "ytl": ytl,
            "xbr": xbr,
            "ybr": ybr,
            "width": xbr - xtl,
            "height": ybr - ytl,
        },
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
    return {
        **common_shape_fields(element),
        "type": "skeleton",
        "points": points,
    }


def convert_image(
    image: ET.Element,
    annotation_xml: Path,
    annotation_kind: str,
    annotation_mode: str,
    image_stem: str,
    image_path: Path,
) -> dict[str, Any]:
    boxes = []
    ellipses = []
    simple_points = []
    skeletons = []

    for shape in image:
        if shape.tag == "box":
            boxes.append(convert_box(shape))
        elif shape.tag == "ellipse":
            ellipses.append(convert_ellipse(shape))
        elif shape.tag == "points":
            simple_points.append(convert_points(shape))
        elif shape.tag == "skeleton":
            skeletons.append(convert_skeleton(shape))

    image_name = image.get("name", "")
    return {
        "annotation_kind": annotation_kind,
        "annotation_mode": annotation_mode,
        "annotation_xml": str(annotation_xml),
        "image_stem": image_stem,
        "image_name": image_name,
        "image_path": str(image_path),
        "width": int_attr(image, "width", 0),
        "height": int_attr(image, "height", 0),
        "time_seconds": frame_time_seconds(image_name),
        "heads": boxes,
        "faces": ellipses,
        "simple_landmarks": simple_points,
        "skeleton_landmarks": skeletons,
    }


def convert_annotations(experiment_dir: Path) -> dict[str, Any]:
    image_root = experiment_dir / "images"
    annotation_root = experiment_dir / "annotations"
    records = []
    summary: dict[str, Any] = {
        "annotation_sets": 0,
        "images": 0,
        "heads": 0,
        "faces": 0,
        "simple_landmarks": 0,
        "skeleton_landmarks": 0,
    }

    for annotation_xml in sorted(annotation_root.glob("*/annotations.xml")):
        annotation_kind = annotation_xml.parent.name
        annotation_mode = infer_annotation_mode(annotation_xml.parent)
        image_stem = infer_image_stem(annotation_xml.parent)
        tree = ET.parse(annotation_xml)
        summary["annotation_sets"] += 1

        for image in tree.getroot().findall(".//image"):
            image_name = image.get("name", "")
            record = convert_image(
                image=image,
                annotation_xml=annotation_xml,
                annotation_kind=annotation_kind,
                annotation_mode=annotation_mode,
                image_stem=image_stem,
                image_path=image_root / image_stem / image_name,
            )
            summary["images"] += 1
            summary["heads"] += len(record["heads"])
            summary["faces"] += len(record["faces"])
            summary["simple_landmarks"] += sum(
                len(points["points"]) for points in record["simple_landmarks"]
            )
            summary["skeleton_landmarks"] += sum(
                len(skeleton["points"]) for skeleton in record["skeleton_landmarks"]
            )
            records.append(record)

    return {
        "schema_version": "1.0",
        "description": "Ground truth converted from CVAT XML annotations.",
        "evaluation_policy": {
            "primary_target_region": "head",
            "secondary_target_region": "face_bbox",
            "comparison_unit": "face",
            "primary_metric_priority": ["recall", "precision", "f1", "mean_iou"],
            "notes": (
                "Use CVAT Head boxes as the primary ground truth for mosaic coverage. "
                "Keep Face ellipse-derived bounding boxes as secondary regions for future "
                "tighter face-area evaluation."
            ),
        },
        "summary": summary,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    ground_truth = convert_annotations(args.experiment_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(ground_truth, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"output: {args.output}")
    for key, value in ground_truth["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
