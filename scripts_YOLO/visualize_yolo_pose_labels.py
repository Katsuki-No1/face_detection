import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
COLORS = {
    "bbox": (0, 220, 0),
    "keypoint": (0, 255, 255),
    "missing": (80, 80, 80),
    "head": (0, 0, 255),
    "face": (255, 0, 255),
    "text_bg": (20, 20, 20),
    "text": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO Pose labels for face keypoints.")
    parser.add_argument("--data", type=Path, default=Path("datasets/face_pose/face_pose.yaml"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional conversion_metadata.json to overlay original Head/Face annotations.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/label_check"))
    parser.add_argument("--show-missing", action="store_true")
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required. Install requirements.txt before visualization.") from error
    return cv2


def parse_dataset_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    dataset_root = Path(data.get("path", path.parent.as_posix()))
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    data["path"] = str(dataset_root)
    return data


def load_metadata(path: Path | None, dataset_root: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    indexed = {}
    for record in metadata.get("records", []):
        output_image = record.get("output_image")
        if output_image:
            indexed[str((dataset_root / output_image).resolve())] = record
    return indexed


def image_label_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    pairs = []
    for split_name in splits:
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        for image_path in sorted(image_dir.glob("*")):
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
    return pairs


def draw_label(image: Any, text: str, x: int, y: int, cv2: Any, scale: float = 0.45) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, x)
    y = max(height + baseline + 2, y)
    cv2.rectangle(
        image,
        (x, y - height - baseline - 4),
        (x + width + 6, y + 2),
        COLORS["text_bg"],
        -1,
    )
    cv2.putText(image, text, (x + 3, y - baseline - 1), font, scale, COLORS["text"], thickness)


def denormalize_bbox(values: list[float], width: int, height: int) -> dict[str, float]:
    xc, yc, bw, bh = values
    box_width = bw * width
    box_height = bh * height
    center_x = xc * width
    center_y = yc * height
    return {
        "xtl": center_x - box_width / 2.0,
        "ytl": center_y - box_height / 2.0,
        "xbr": center_x + box_width / 2.0,
        "ybr": center_y + box_height / 2.0,
    }


def draw_box(image: Any, bbox: dict[str, float], color: tuple[int, int, int], cv2: Any, thickness: int = 2) -> None:
    cv2.rectangle(
        image,
        (int(round(bbox["xtl"])), int(round(bbox["ytl"]))),
        (int(round(bbox["xbr"])), int(round(bbox["ybr"]))),
        color,
        thickness,
    )


def draw_metadata(image: Any, record: dict[str, Any], cv2: Any) -> None:
    for index, head in enumerate(record.get("heads", [])):
        draw_box(image, head["bbox"], COLORS["head"], cv2, thickness=1)
        draw_label(image, f"Head {index}", int(head["bbox"]["xtl"]), int(head["bbox"]["ytl"]) - 3, cv2)
    for index, face in enumerate(record.get("faces", [])):
        draw_box(image, face["bbox"], COLORS["face"], cv2, thickness=1)
        center = face.get("center")
        radius = face.get("radius")
        if center and radius:
            cv2.ellipse(
                image,
                (int(round(center["x"])), int(round(center["y"]))),
                (int(round(radius["x"])), int(round(radius["y"]))),
                float(face.get("rotation", 0.0)),
                0,
                360,
                COLORS["face"],
                1,
            )
        draw_label(image, f"Face {index}", int(face["bbox"]["xtl"]), int(face["bbox"]["ytl"]) - 3, cv2)


def visualize_pair(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    metadata: dict[str, dict[str, Any]],
    show_missing: bool,
    cv2: Any,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    meta = metadata.get(str(image_path.resolve()))
    if meta:
        draw_metadata(image, meta, cv2)

    for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        bbox = denormalize_bbox(values[1:5], width, height)
        draw_box(image, bbox, COLORS["bbox"], cv2, thickness=2)
        draw_label(image, f"face {object_index}", int(bbox["xtl"]), int(bbox["ytl"]) - 5, cv2)
        keypoint_values = values[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            x_norm, y_norm, visibility = keypoint_values[index * 3 : index * 3 + 3]
            if visibility == 0 and not show_missing:
                continue
            x = int(round(x_norm * width))
            y = int(round(y_norm * height))
            color = COLORS["keypoint"] if visibility else COLORS["missing"]
            cv2.circle(image, (x, y), 4, color, -1)
            draw_label(image, name if visibility else f"{name}:0", x + 5, y - 4, cv2, scale=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> int:
    args = parse_args()
    cv2 = import_cv2()
    dataset = parse_dataset_yaml(args.data)
    dataset_root = Path(dataset["path"])
    metadata = load_metadata(args.metadata, dataset_root)
    pairs = image_label_pairs(dataset_root, args.split)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    selected = pairs[: args.samples]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, label_path) in enumerate(selected):
        output_path = args.out_dir / f"{index:04d}_{image_path.name}"
        visualize_pair(image_path, label_path, output_path, metadata, args.show_missing, cv2)
    print(f"wrote {len(selected)} files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
