import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ANNOTATION_PREFIXES = ("an_simple_", "an_skeleton_")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that CVAT annotation XML files are paired with image files."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline"),
        help="Experiment directory that contains images/ and annotations/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV path to write one row per annotated image.",
    )
    return parser.parse_args()


def infer_image_stem(annotation_dir: Path) -> str:
    name = annotation_dir.name
    for prefix in ANNOTATION_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def count_shapes(image_element: ET.Element) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in image_element:
        counts[child.tag] = counts.get(child.tag, 0) + 1
    return counts


def validate(experiment_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    image_root = experiment_dir / "images"
    annotation_root = experiment_dir / "annotations"
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    xml_files = sorted(annotation_root.glob("*/annotations.xml"))
    if not xml_files:
        errors.append(f"No annotations.xml files found under {annotation_root}")
        return rows, errors

    for xml_path in xml_files:
        annotation_kind = xml_path.parent.name
        image_stem = infer_image_stem(xml_path.parent)
        image_dir = image_root / image_stem

        if not image_dir.is_dir():
            errors.append(f"{annotation_kind}: missing image directory: {image_dir}")
            continue

        image_files = {
            path.name: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        root = ET.parse(xml_path).getroot()
        annotated_images = root.findall(".//image")
        annotated_names = [image.get("name", "") for image in annotated_images]

        missing = [name for name in annotated_names if name not in image_files]
        extra = sorted(set(image_files) - set(annotated_names))
        if missing:
            errors.append(f"{annotation_kind}: missing {len(missing)} image(s): {missing[:5]}")
        if extra:
            errors.append(f"{annotation_kind}: extra {len(extra)} image(s): {extra[:5]}")

        for image in annotated_images:
            name = image.get("name", "")
            shape_counts = count_shapes(image)
            rows.append(
                {
                    "annotation_kind": annotation_kind,
                    "annotation_xml": str(xml_path),
                    "image_stem": image_stem,
                    "image_name": name,
                    "image_path": str(image_dir / name),
                    "width": image.get("width", ""),
                    "height": image.get("height", ""),
                    "box_count": str(shape_counts.get("box", 0)),
                    "ellipse_count": str(shape_counts.get("ellipse", 0)),
                    "points_count": str(shape_counts.get("points", 0)),
                    "skeleton_count": str(shape_counts.get("skeleton", 0)),
                }
            )

    return rows, errors


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_kind",
        "annotation_xml",
        "image_stem",
        "image_name",
        "image_path",
        "width",
        "height",
        "box_count",
        "ellipse_count",
        "points_count",
        "skeleton_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows, errors = validate(args.experiment_dir)

    if args.manifest is not None:
        write_manifest(args.manifest, rows)
        print(f"manifest: {args.manifest}")

    annotation_count = len({row["annotation_kind"] for row in rows})
    print(f"annotation sets: {annotation_count}")
    print(f"annotated image rows: {len(rows)}")

    if errors:
        print("status: incomplete")
        for error in errors:
            print(f"- {error}")
        return 1

    print("status: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
