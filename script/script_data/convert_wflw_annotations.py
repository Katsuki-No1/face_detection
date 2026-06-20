import argparse
import json
import sys
from pathlib import Path
from typing import Any


LANDMARK_COUNT = 98
COORDINATE_COUNT = LANDMARK_COUNT * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WFLW 98-point annotations into the project ground truth JSON schema."
    )
    parser.add_argument(
        "--wflw-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/WFLW"),
        help="Directory containing WFLW_images and WFLW_annotations.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="WFLW split to convert. Default: test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/wflw_ground_truth_test.json"),
        help="Output ground truth JSON path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError(
            "OpenCV is required to read image dimensions. Run with /tmp/fd-env/bin/python."
        ) from error
    return cv2


def annotation_file(wflw_dir: Path, split: str) -> Path:
    return (
        wflw_dir
        / "WFLW_annotations"
        / "list_98pt_rect_attr_train_test"
        / f"list_98pt_rect_attr_{split}.txt"
    )


def parse_line(line: str) -> dict[str, Any]:
    parts = line.strip().split()
    if len(parts) < COORDINATE_COUNT + 4 + 6 + 1:
        raise ValueError(f"Unexpected WFLW row length: {len(parts)}")

    coordinates = [float(value) for value in parts[:COORDINATE_COUNT]]
    landmarks = [
        {"label": f"pt_{index:02d}", "x": coordinates[index * 2], "y": coordinates[index * 2 + 1]}
        for index in range(LANDMARK_COUNT)
    ]
    rect_start = COORDINATE_COUNT
    x_min, y_min, x_max, y_max = [float(value) for value in parts[rect_start : rect_start + 4]]
    attr_start = rect_start + 4
    attrs = [int(value) for value in parts[attr_start : attr_start + 6]]
    image_name = parts[attr_start + 6]

    return {
        "landmarks": landmarks,
        "bbox": {
            "xtl": x_min,
            "ytl": y_min,
            "xbr": x_max,
            "ybr": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
        },
        "attributes": {
            "pose": attrs[0],
            "expression": attrs[1],
            "illumination": attrs[2],
            "makeup": attrs[3],
            "occlusion": attrs[4],
            "blur": attrs[5],
        },
        "image_name": image_name,
    }


def landmark_shape(landmarks: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "label": "wflw_98pt",
        "source": "wflw",
        "occluded": False,
        "z_order": 0,
        "type": "points",
        "points": landmarks,
    }


def read_image_size(image_path: Path, cv2: Any) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read WFLW image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def convert(args: argparse.Namespace) -> dict[str, Any]:
    cv2 = import_cv2()
    images_root = args.wflw_dir / "WFLW_images"
    rows = []
    missing = 0

    for line_number, line in enumerate(annotation_file(args.wflw_dir, args.split).open(encoding="utf-8"), start=1):
        parsed = parse_line(line)
        image_path = images_root / parsed["image_name"]
        if not image_path.is_file():
            missing += 1
            continue

        width, height = read_image_size(image_path, cv2)
        rows.append(
            {
                "annotation_kind": f"wflw_{args.split}",
                "annotation_mode": "wflw",
                "annotation_xml": "",
                "image_stem": str(Path(parsed["image_name"]).parent),
                "image_name": Path(parsed["image_name"]).name,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "time_seconds": None,
                "attributes": parsed["attributes"],
                "heads": [
                    {
                        "label": "WFLW_rect",
                        "source": "wflw",
                        "occluded": False,
                        "z_order": 0,
                        "type": "box",
                        "bbox": parsed["bbox"],
                    }
                ],
                "faces": [
                    {
                        "label": "WFLW_rect",
                        "source": "wflw",
                        "occluded": False,
                        "z_order": 0,
                        "type": "box",
                        "bbox": parsed["bbox"],
                    }
                ],
                "simple_landmarks": [landmark_shape(parsed["landmarks"])],
                "skeleton_landmarks": [],
            }
        )
        if args.max_records is not None and len(rows) >= args.max_records:
            break

    return {
        "schema_version": "1.0",
        "description": "WFLW ground truth converted into the project evaluation schema.",
        "evaluation_policy": {
            "primary_target_region": "head",
            "secondary_target_region": "face_bbox",
            "comparison_unit": "face",
            "primary_metric_priority": ["recall", "precision", "f1", "mean_iou"],
            "notes": (
                "WFLW detection rectangles are stored in heads/faces so the same evaluation "
                "scripts can be used. They are face rectangles, not CVAT Head boxes."
            ),
        },
        "summary": {
            "annotation_sets": 1,
            "images": len(rows),
            "heads": len(rows),
            "faces": len(rows),
            "simple_landmarks": len(rows) * LANDMARK_COUNT,
            "skeleton_landmarks": 0,
            "missing_images": missing,
        },
        "records": rows,
    }


def main() -> int:
    args = parse_args()
    converted = convert(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"output: {args.output}")
    for key, value in converted["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
