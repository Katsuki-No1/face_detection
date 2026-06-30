import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
SPLITS = ["train", "val", "test"]
DEFAULT_BALANCE_CLASSES = [
    "negative",
    "small_face",
    "edge_face",
    "ellipse_oob",
    "occluded_face",
    "occluded_keypoint",
    "incomplete_keypoints",
    "multi_face",
    "crowded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 dataset audit for YOLO Pose face datasets. This does not run training."
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML. Defaults to <data-root>/face_pose.yaml.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional conversion_metadata.json path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_experiment/phase0_dataset_audit"),
        help="Directory for audit JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--small-face-area-threshold",
        type=float,
        default=0.01,
        help="Small-face threshold as bbox area / image area. Default: 0.01.",
    )
    parser.add_argument(
        "--tiny-face-area-threshold",
        type=float,
        default=0.0025,
        help="Tiny-face threshold as bbox area / image area. Default: 0.0025.",
    )
    parser.add_argument(
        "--large-face-area-threshold",
        type=float,
        default=0.15,
        help="Large-face threshold as bbox area / image area. Default: 0.15.",
    )
    parser.add_argument(
        "--edge-margin-ratio",
        type=float,
        default=0.02,
        help="BBox edge class threshold as image width/height ratio. Default: 0.02.",
    )
    parser.add_argument(
        "--balance-classes",
        default=",".join(DEFAULT_BALANCE_CLASSES),
        help="Comma-separated image classes used for train sampling weights.",
    )
    parser.add_argument(
        "--max-sample-weight",
        type=float,
        default=5.0,
        help="Cap for inverse-frequency sample weight. Default: 5.0.",
    )
    parser.add_argument(
        "--max-sample-repeat",
        type=int,
        default=4,
        help="Cap for repeated entries in balanced train image list. Default: 4.",
    )
    parser.add_argument(
        "--near-frame-window",
        type=int,
        default=3,
        help="Cross-split near-frame threshold by frame index. Default: 3.",
    )
    parser.add_argument(
        "--near-time-window",
        type=float,
        default=2.0,
        help="Cross-split near-frame threshold by seconds. Default: 2.0.",
    )
    parser.add_argument(
        "--max-listed-issues",
        type=int,
        default=200,
        help="Maximum detailed issues/pairs embedded in JSON report. Full rows are written to CSV.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def import_dependencies() -> tuple[Any, Any]:
    missing = []
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
        missing.append("PyYAML")
    if missing:
        raise RuntimeError("Missing package(s): " + ", ".join(missing) + ". Install requirements.txt first.")
    return cv2, yaml


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "labels" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() == ".txt")


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def frame_info(path: Path) -> dict[str, Any]:
    stem = path.stem
    if "_frame_" not in stem:
        return {"family": stem, "frame_index": None, "time_sec": None}
    family, tail = stem.split("_frame_", 1)
    frame_match = re.match(r"(\d+)", tail)
    time_match = re.search(r"_(\d+(?:\.\d+)?)s$", tail)
    return {
        "family": family,
        "frame_index": int(frame_match.group(1)) if frame_match else None,
        "time_sec": float(time_match.group(1)) if time_match else None,
    }


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def split_text(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[|,;]", value) if item.strip()]


def class_text(classes: set[str]) -> str:
    return "|".join(sorted(classes))


def possible_metadata_keys(record: dict[str, Any]) -> set[str]:
    keys = set()
    for field in ("image_name", "output_image", "image", "image_path"):
        value = record.get(field)
        if isinstance(value, str) and value:
            keys.add(value)
            keys.add(Path(value).name)
    return keys


def discover_metadata(data_root: Path, explicit_metadata: Path | None) -> tuple[Path | None, Path | None]:
    if explicit_metadata is not None:
        return explicit_metadata, None

    direct = data_root / "metadata" / "conversion_metadata.json"
    if direct.exists():
        return direct, None

    variant = data_root / "metadata" / "variant_metadata.json"
    if variant.exists():
        variant_data = read_json(variant)
        source_root_value = variant_data.get("source_root")
        if isinstance(source_root_value, str):
            candidates = [Path(source_root_value), data_root.parent / Path(source_root_value).name]
            for source_root in candidates:
                metadata = source_root / "metadata" / "conversion_metadata.json"
                if metadata.exists():
                    return metadata, variant

    sibling = data_root.parent / "_modified_all_train" / "metadata" / "conversion_metadata.json"
    if sibling.exists():
        return sibling, variant if variant.exists() else None
    return None, variant if variant.exists() else None


def variant_split_index(variant_path: Path | None) -> dict[str, str]:
    if variant_path is None or not variant_path.exists():
        return {}
    data = read_json(variant_path)
    index = {}
    for record in data.get("records", []):
        image = record.get("image")
        split = record.get("split")
        if isinstance(image, str) and isinstance(split, str):
            index[image] = split
            index[Path(image).name] = split
    return index


def metadata_index(metadata_path: Path | None, variant_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if metadata_path is None or not metadata_path.exists():
        return {}, {"metadata_path": None, "variant_metadata_path": str(variant_path) if variant_path else None}

    data = read_json(metadata_path)
    split_index = variant_split_index(variant_path)
    index: dict[str, dict[str, Any]] = {}
    for source_record in data.get("records", []):
        record = dict(source_record)
        keys = possible_metadata_keys(record)
        dataset_split = None
        for key in keys:
            if key in split_index:
                dataset_split = split_index[key]
                break
        if dataset_split is None:
            dataset_split = str(record.get("split", ""))
        record["_dataset_split"] = dataset_split
        for key in keys:
            index[key] = record
    return index, {
        "metadata_path": str(metadata_path),
        "variant_metadata_path": str(variant_path) if variant_path else None,
        "metadata_records": len(data.get("records", [])),
        "metadata_summary": data.get("summary", {}),
    }


def read_image_size(path: Path, cv2: Any) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def load_label_objects(
    label_path: Path,
    width: int,
    height: int,
    keypoint_names: list[str],
    failures: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    objects = []
    if not label_path.exists():
        return objects
    expected_values = 5 + len(keypoint_names) * 3
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values = [float(part) for part in line.split()]
        except ValueError:
            failures.append({"split": split, "label": str(label_path), "line": line_number, "reason": "non_numeric_label"})
            continue
        if len(values) != expected_values:
            failures.append(
                {
                    "split": split,
                    "label": str(label_path),
                    "line": line_number,
                    "reason": "unexpected_value_count",
                    "expected": expected_values,
                    "actual": len(values),
                }
            )
            continue
        class_id = int(values[0])
        xc, yc, bw, bh = values[1:5]
        bbox_norm = [xc, yc, bw, bh]
        if class_id != 0:
            failures.append({"split": split, "label": str(label_path), "line": line_number, "reason": "unexpected_class_id", "class_id": class_id})
        if any(value < 0.0 or value > 1.0 for value in bbox_norm):
            failures.append({"split": split, "label": str(label_path), "line": line_number, "reason": "bbox_norm_out_of_range", "bbox": bbox_norm})
        if bw <= 0.0 or bh <= 0.0:
            failures.append({"split": split, "label": str(label_path), "line": line_number, "reason": "bbox_non_positive", "bbox": bbox_norm})

        box_width = bw * width
        box_height = bh * height
        cx = xc * width
        cy = yc * height
        keypoints = []
        raw_keypoints = values[5:]
        for index, name in enumerate(keypoint_names):
            x_value = raw_keypoints[index * 3]
            y_value = raw_keypoints[index * 3 + 1]
            visibility = int(raw_keypoints[index * 3 + 2])
            if visibility not in {0, 1, 2}:
                failures.append(
                    {
                        "split": split,
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "visibility_out_of_range",
                        "keypoint": name,
                        "visibility": visibility,
                    }
                )
            if visibility > 0 and (x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0):
                failures.append(
                    {
                        "split": split,
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "visible_keypoint_out_of_range",
                        "keypoint": name,
                        "x": x_value,
                        "y": y_value,
                    }
                )
            keypoints.append({"name": name, "x": x_value, "y": y_value, "visibility": visibility})
        objects.append(
            {
                "bbox_norm": bbox_norm,
                "bbox_xyxy": [
                    cx - box_width / 2.0,
                    cy - box_height / 2.0,
                    cx + box_width / 2.0,
                    cy + box_height / 2.0,
                ],
                "bbox_width_px": box_width,
                "bbox_height_px": box_height,
                "bbox_area_ratio": bw * bh,
                "keypoints": keypoints,
            }
        )
    return objects


def metadata_for_image(index: dict[str, dict[str, Any]], image_path: Path) -> dict[str, Any] | None:
    return index.get(image_path.name) or index.get(str(image_path)) or index.get(image_path.stem)


def bbox_outside_image(bbox: dict[str, Any], width: int, height: int) -> bool:
    return (
        float(bbox.get("xtl", 0.0)) < 0.0
        or float(bbox.get("ytl", 0.0)) < 0.0
        or float(bbox.get("xbr", 0.0)) > float(width)
        or float(bbox.get("ybr", 0.0)) > float(height)
    )


def bbox_near_image_edge(box: list[float], width: int, height: int, margin_ratio: float) -> bool:
    margin_x = float(width) * margin_ratio
    margin_y = float(height) * margin_ratio
    return box[0] <= margin_x or box[1] <= margin_y or box[2] >= width - margin_x or box[3] >= height - margin_y


def split_valid_and_excluded_metadata_objects(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_faces = list(record.get("excluded_faces", []))
    valid_objects = []
    for item in record.get("objects", []):
        if "invalid_clipped_face_bbox" in item.get("warnings", []):
            excluded_faces.append(item)
        else:
            valid_objects.append(item)
    return valid_objects, excluded_faces


def object_class_set(
    item: dict[str, Any],
    width: int,
    height: int,
    keypoint_count: int,
    meta_object: dict[str, Any] | None,
    args: argparse.Namespace,
) -> set[str]:
    classes = {"face"}
    area_ratio = float(item["bbox_area_ratio"])
    if area_ratio < args.tiny_face_area_threshold:
        classes.add("tiny_face")
    if area_ratio < args.small_face_area_threshold:
        classes.add("small_face")
    elif area_ratio >= args.large_face_area_threshold:
        classes.add("large_face")
    else:
        classes.add("medium_face")

    if bbox_near_image_edge(item["bbox_xyxy"], width, height, args.edge_margin_ratio):
        classes.add("edge_face")

    visible_keypoints = sum(1 for point in item["keypoints"] if int(point["visibility"]) > 0)
    if visible_keypoints < keypoint_count:
        classes.add("incomplete_keypoints")
    else:
        classes.add("complete_keypoints")

    if isinstance(meta_object, dict):
        if meta_object.get("face", {}).get("occluded"):
            classes.add("occluded_face")
        if "face_bbox_clipped_to_image" in meta_object.get("warnings", []):
            classes.add("ellipse_oob")
        keypoints = meta_object.get("keypoints", {})
        if any(point.get("occluded") for point in keypoints.values()):
            classes.add("occluded_keypoint")
    return classes


def metadata_image_stats(record: dict[str, Any] | None, width: int, height: int) -> dict[str, Any]:
    if record is None:
        return {
            "metadata_found": False,
            "metadata_faces": 0,
            "metadata_face_occluded": 0,
            "metadata_ellipse_out_of_image": 0,
            "metadata_objects": 0,
            "metadata_excluded_faces": 0,
            "metadata_objects_with_occluded_keypoint": 0,
            "metadata_occluded_keypoints": 0,
            "metadata_face_bbox_clipped_objects": 0,
        }

    faces = record.get("faces", [])
    objects, excluded_faces = split_valid_and_excluded_metadata_objects(record)
    ellipse_out = 0
    for face in faces:
        bbox = face.get("bbox", {})
        if bbox and bbox_outside_image(bbox, width, height):
            ellipse_out += 1

    objects_with_occluded_keypoint = 0
    occluded_keypoints = 0
    face_bbox_clipped_objects = 0
    for item in objects:
        keypoints = item.get("keypoints", {})
        keypoint_occluded_here = 0
        for point in keypoints.values():
            if point.get("occluded"):
                occluded_keypoints += 1
                keypoint_occluded_here += 1
        if keypoint_occluded_here:
            objects_with_occluded_keypoint += 1
        if "face_bbox_clipped_to_image" in item.get("warnings", []):
            face_bbox_clipped_objects += 1

    return {
        "metadata_found": True,
        "metadata_faces": len(faces),
        "metadata_face_occluded": sum(1 for face in faces if face.get("occluded")),
        "metadata_ellipse_out_of_image": ellipse_out,
        "metadata_objects": len(objects),
        "metadata_excluded_faces": len(excluded_faces),
        "metadata_objects_with_occluded_keypoint": objects_with_occluded_keypoint,
        "metadata_occluded_keypoints": occluded_keypoints,
        "metadata_face_bbox_clipped_objects": face_bbox_clipped_objects,
    }


def find_near_frame_pairs(
    image_rows: list[dict[str, Any]],
    near_frame_window: int,
    near_time_window: float,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        if row.get("family"):
            by_family[str(row["family"])].append(row)

    pairs = []
    for family, rows in by_family.items():
        ordered = sorted(rows, key=lambda row: (row.get("frame_index") is None, row.get("frame_index") or -1, row.get("time_sec") or -1.0))
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if first["split"] == second["split"]:
                    continue
                frame_delta = None
                if first.get("frame_index") is not None and second.get("frame_index") is not None:
                    frame_delta = abs(int(first["frame_index"]) - int(second["frame_index"]))
                time_delta = None
                if first.get("time_sec") is not None and second.get("time_sec") is not None:
                    time_delta = abs(float(first["time_sec"]) - float(second["time_sec"]))
                is_near_frame = frame_delta is not None and frame_delta <= near_frame_window
                is_near_time = time_delta is not None and time_delta <= near_time_window
                if is_near_frame or is_near_time:
                    pairs.append(
                        {
                            "family": family,
                            "first_split": first["split"],
                            "first_image": first["image"],
                            "second_split": second["split"],
                            "second_image": second["image"],
                            "frame_delta": frame_delta,
                            "time_delta": time_delta,
                        }
                    )
    return pairs


def split_pair_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for pair in pairs:
        key = "/".join(sorted([pair["first_split"], pair["second_split"]]))
        counts[key] += 1
    return dict(sorted(counts.items()))


def train_sampling_weights(
    image_rows: list[dict[str, Any]],
    balance_classes: list[str],
    max_sample_weight: float,
    max_sample_repeat: int,
) -> list[dict[str, Any]]:
    train_rows = [row for row in image_rows if row["split"] == "train"]
    class_counts = Counter()
    balance_class_set = set(balance_classes)
    for row in train_rows:
        classes = set(split_text(str(row.get("classes", "")))) & balance_class_set
        for class_name in classes:
            class_counts[class_name] += 1

    nonzero_counts = [class_counts[class_name] for class_name in balance_classes if class_counts[class_name] > 0]
    target_count = max(nonzero_counts) if nonzero_counts else 1
    sampling_rows = []
    for row in train_rows:
        classes = set(split_text(str(row.get("classes", "")))) & balance_class_set
        if classes:
            raw_weight = max(target_count / class_counts[class_name] for class_name in classes if class_counts[class_name] > 0)
        else:
            raw_weight = 1.0
        sample_weight = min(max_sample_weight, max(1.0, raw_weight))
        repeat_count = max(1, min(max_sample_repeat, int(math.ceil(sample_weight))))
        sampling_rows.append(
            {
                "image": row["image"],
                "label": row["label"],
                "classes": row.get("classes", ""),
                "objects": row.get("objects", 0),
                "sample_weight": round(sample_weight, 6),
                "repeat_count": repeat_count,
            }
        )
    return sampling_rows


def write_balanced_train_files(
    output_dir: Path,
    data_root: Path,
    keypoint_names: list[str],
    sampling_rows: list[dict[str, Any]],
) -> None:
    train_list = output_dir / "phase0_balanced_train_images.txt"
    lines = []
    for row in sampling_rows:
        for _ in range(int(row["repeat_count"])):
            lines.append(str(Path(str(row["image"])).resolve()))
    train_list.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_text = "\n".join(
        [
            f"path: {data_root.as_posix()}",
            f"train: {train_list.resolve().as_posix()}",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: face",
            "kpt_shape: [5, 3]",
            "flip_idx: [1, 0, 2, 4, 3]",
            f"keypoints: [{', '.join(keypoint_names)}]",
            "",
        ]
    )
    (output_dir / "face_pose_class_balanced.yaml").write_text(yaml_text, encoding="utf-8")


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cv2, yaml = import_dependencies()
    data_root = args.data_root.resolve()
    data_yaml = (args.data or data_root / "face_pose.yaml").resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    keypoint_names = config.get("keypoints") or DEFAULT_KEYPOINT_NAMES
    metadata_path, variant_path = discover_metadata(data_root, args.metadata)
    metadata_records, metadata_summary = metadata_index(metadata_path, variant_path)

    image_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    bbox_area_ratios: list[float] = []
    bbox_width_norms: list[float] = []
    bbox_height_norms: list[float] = []
    bbox_width_px: list[float] = []
    bbox_height_px: list[float] = []
    visibility_counts: Counter[str] = Counter()
    visibility_by_keypoint: dict[str, Counter[str]] = {name: Counter() for name in keypoint_names}
    image_class_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    instance_class_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    image_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)

    split_counters: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    metadata_counters: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    all_failures: list[dict[str, Any]] = []

    for split in SPLITS:
        images = image_paths(data_root, split)
        labels = label_paths(data_root, split)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        for stem in sorted(image_stems - label_stems):
            all_failures.append({"split": split, "stem": stem, "reason": "missing_label"})
        for stem in sorted(label_stems - image_stems):
            all_failures.append({"split": split, "stem": stem, "reason": "extra_label"})

        for image_path in images:
            size = read_image_size(image_path, cv2)
            if size is None:
                all_failures.append({"split": split, "image": str(image_path), "reason": "image_read_failed"})
                continue
            width, height = size
            label_path = label_path_for_image(data_root, split, image_path)
            objects = load_label_objects(label_path, width, height, keypoint_names, all_failures, split)
            meta = metadata_for_image(metadata_records, image_path)
            valid_meta_objects: list[dict[str, Any]] = []
            if meta is not None:
                valid_meta_objects, _ = split_valid_and_excluded_metadata_objects(meta)
            meta_stats = metadata_image_stats(meta, width, height)
            digest = sha1_file(image_path)
            image_hashes[digest].append({"split": split, "image": str(image_path)})

            small_instances = 0
            image_classes: set[str] = {"positive"} if objects else {"negative"}
            if len(objects) >= 2:
                image_classes.add("multi_face")
            if len(objects) >= 3:
                image_classes.add("crowded")

            for object_index, item in enumerate(objects):
                bbox_area_ratios.append(float(item["bbox_area_ratio"]))
                bbox_width_norms.append(float(item["bbox_norm"][2]))
                bbox_height_norms.append(float(item["bbox_norm"][3]))
                bbox_width_px.append(float(item["bbox_width_px"]))
                bbox_height_px.append(float(item["bbox_height_px"]))

                visible_keypoints = 0
                for point in item["keypoints"]:
                    visibility = str(point["visibility"])
                    visibility_counts[visibility] += 1
                    visibility_by_keypoint[point["name"]][visibility] += 1
                    visible_keypoints += int(point["visibility"] > 0)

                meta_object = None
                if object_index < len(valid_meta_objects):
                    meta_object = valid_meta_objects[object_index]
                object_classes = object_class_set(item, width, height, len(keypoint_names), meta_object, args)
                image_classes.update(object_classes - {"face", "complete_keypoints", "medium_face", "large_face"})
                is_small = "small_face" in object_classes
                small_instances += int(is_small)
                meta_keypoints = meta_object.get("keypoints", {}) if isinstance(meta_object, dict) else {}
                metadata_keypoint_occluded_count = sum(1 for point in meta_keypoints.values() if point.get("occluded"))
                metadata_face_occluded = False
                if isinstance(meta_object, dict):
                    metadata_face_occluded = bool(meta_object.get("face", {}).get("occluded"))
                for class_name in object_classes:
                    instance_class_counts[split][class_name] += 1
                instance_rows.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "label": str(label_path),
                        "object_index": object_index,
                        "bbox_xc_norm": item["bbox_norm"][0],
                        "bbox_yc_norm": item["bbox_norm"][1],
                        "bbox_width_norm": item["bbox_norm"][2],
                        "bbox_height_norm": item["bbox_norm"][3],
                        "bbox_width_px": item["bbox_width_px"],
                        "bbox_height_px": item["bbox_height_px"],
                        "bbox_area_ratio": item["bbox_area_ratio"],
                        "small_face": is_small,
                        "visible_keypoints": visible_keypoints,
                        "classes": class_text(object_classes),
                        "metadata_found": meta is not None,
                        "metadata_face_occluded": metadata_face_occluded,
                        "metadata_keypoint_occluded_count": metadata_keypoint_occluded_count,
                        "metadata_face_bbox_clipped_to_image": bool(
                            isinstance(meta_object, dict) and "face_bbox_clipped_to_image" in meta_object.get("warnings", [])
                        ),
                    }
                )

            if objects and image_classes.isdisjoint(
                {
                    "small_face",
                    "tiny_face",
                    "edge_face",
                    "ellipse_oob",
                    "occluded_face",
                    "occluded_keypoint",
                    "incomplete_keypoints",
                    "multi_face",
                    "crowded",
                }
            ):
                image_classes.add("normal_face")
            for class_name in image_classes:
                image_class_counts[split][class_name] += 1

            row = {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "width": width,
                "height": height,
                "objects": len(objects),
                "positive": bool(objects),
                "negative": not objects,
                "small_instances": small_instances,
                "small_instance_ratio": ratio(small_instances, len(objects)),
                "classes": class_text(image_classes),
                "sha1": digest,
                **frame_info(image_path),
                **meta_stats,
            }
            image_rows.append(row)
            if meta_stats["metadata_ellipse_out_of_image"]:
                issue_rows.append(
                    {
                        "severity": "warn",
                        "reason": "ellipse_out_of_image",
                        "split": split,
                        "image": str(image_path),
                        "count": meta_stats["metadata_ellipse_out_of_image"],
                    }
                )
            if meta is not None and len(objects) != meta_stats["metadata_objects"]:
                issue_rows.append(
                    {
                        "severity": "warn",
                        "reason": "metadata_object_count_differs_from_yolo_instances",
                        "split": split,
                        "image": str(image_path),
                        "yolo_instances": len(objects),
                        "metadata_objects": meta_stats["metadata_objects"],
                    }
                )

            split_counters[split]["images"] += 1
            split_counters[split]["positive_images"] += int(bool(objects))
            split_counters[split]["negative_images"] += int(not objects)
            split_counters[split]["instances"] += len(objects)
            split_counters[split]["small_instances"] += small_instances

            for key, value in meta_stats.items():
                if isinstance(value, bool):
                    metadata_counters[split][key] += int(value)
                elif isinstance(value, int):
                    metadata_counters[split][key] += value

    duplicate_images = [
        {"sha1": digest, "items": items}
        for digest, items in sorted(image_hashes.items())
        if len({item["split"] for item in items}) > 1
    ]
    near_frame_pairs = find_near_frame_pairs(image_rows, args.near_frame_window, args.near_time_window)

    for failure in all_failures:
        issue_rows.append({**failure, "severity": "error"})
    for duplicate in duplicate_images:
        issue_rows.append({"severity": "warn", "reason": "duplicate_image_across_splits", "sha1": duplicate["sha1"], "items": json.dumps(duplicate["items"], ensure_ascii=False)})
    for pair in near_frame_pairs:
        issue_rows.append({**pair, "severity": "warn", "reason": "near_frame_across_splits"})

    split_summary = {}
    for split in SPLITS:
        counter = split_counters[split]
        meta_counter = metadata_counters[split]
        split_summary[split] = {
            "images": counter["images"],
            "positive_images": counter["positive_images"],
            "negative_images": counter["negative_images"],
            "positive_image_ratio": ratio(counter["positive_images"], counter["images"]),
            "negative_image_ratio": ratio(counter["negative_images"], counter["images"]),
            "instances": counter["instances"],
            "small_instances": counter["small_instances"],
            "small_instance_ratio": ratio(counter["small_instances"], counter["instances"]),
            "metadata_found_images": meta_counter["metadata_found"],
            "metadata_faces": meta_counter["metadata_faces"],
            "metadata_face_occluded": meta_counter["metadata_face_occluded"],
            "metadata_face_occluded_ratio": ratio(meta_counter["metadata_face_occluded"], meta_counter["metadata_faces"]),
            "metadata_ellipse_out_of_image": meta_counter["metadata_ellipse_out_of_image"],
            "metadata_ellipse_out_of_image_ratio": ratio(meta_counter["metadata_ellipse_out_of_image"], meta_counter["metadata_faces"]),
            "metadata_objects": meta_counter["metadata_objects"],
            "metadata_excluded_faces": meta_counter["metadata_excluded_faces"],
            "metadata_objects_with_occluded_keypoint": meta_counter["metadata_objects_with_occluded_keypoint"],
            "metadata_objects_with_occluded_keypoint_ratio": ratio(
                meta_counter["metadata_objects_with_occluded_keypoint"], meta_counter["metadata_objects"]
            ),
            "metadata_occluded_keypoints": meta_counter["metadata_occluded_keypoints"],
            "metadata_face_bbox_clipped_objects": meta_counter["metadata_face_bbox_clipped_objects"],
        }

    total_images = sum(summary["images"] for summary in split_summary.values())
    total_instances = sum(summary["instances"] for summary in split_summary.values())
    total_faces = sum(summary["metadata_faces"] for summary in split_summary.values())
    total_metadata_objects = sum(summary["metadata_objects"] for summary in split_summary.values())
    balance_classes = split_text(args.balance_classes)
    train_sampling_rows = train_sampling_weights(
        image_rows,
        balance_classes=balance_classes,
        max_sample_weight=args.max_sample_weight,
        max_sample_repeat=args.max_sample_repeat,
    )
    checks = {
        "no_training_run": True,
        "failure_count": len(all_failures),
        "duplicate_image_cross_split_count": len(duplicate_images),
        "near_frame_cross_split_pair_count": len(near_frame_pairs),
        "near_frame_cross_split_pair_counts_by_split_pair": split_pair_counts(near_frame_pairs),
        "ellipse_out_of_image_count": sum(summary["metadata_ellipse_out_of_image"] for summary in split_summary.values()),
        "metadata_object_mismatch_image_count": sum(
            1 for row in image_rows if row["metadata_found"] and row["objects"] != row["metadata_objects"]
        ),
        "metadata_available": bool(metadata_records),
        "balance_classes": balance_classes,
        "balanced_train_unique_images": len(train_sampling_rows),
        "balanced_train_repeated_entries": sum(int(row["repeat_count"]) for row in train_sampling_rows),
        "yaml_kpt_shape": config.get("kpt_shape"),
        "yaml_flip_idx": config.get("flip_idx"),
        "expected_flip_idx": [1, 0, 2, 4, 3],
        "flip_idx_ok": config.get("flip_idx") == [1, 0, 2, 4, 3],
    }
    status = "pass"
    if (
        all_failures
        or duplicate_images
        or near_frame_pairs
        or not checks["flip_idx_ok"]
        or checks["ellipse_out_of_image_count"] > 0
        or checks["metadata_object_mismatch_image_count"] > 0
    ):
        status = "warn"

    report = {
        "schema_version": "1.0",
        "phase": "Phase 0: Dataset audit",
        "status": status,
        "data_root": str(data_root),
        "data_yaml": str(data_yaml),
        "keypoint_names": keypoint_names,
        "small_face_area_threshold": args.small_face_area_threshold,
        "near_frame_window": args.near_frame_window,
        "near_time_window": args.near_time_window,
        "metadata": metadata_summary,
        "checks": checks,
        "summary": {
            "images": total_images,
            "instances": total_instances,
            "positive_images": sum(summary["positive_images"] for summary in split_summary.values()),
            "negative_images": sum(summary["negative_images"] for summary in split_summary.values()),
            "small_instances": sum(summary["small_instances"] for summary in split_summary.values()),
            "small_instance_ratio": ratio(sum(summary["small_instances"] for summary in split_summary.values()), total_instances),
            "metadata_faces": total_faces,
            "metadata_face_occluded": sum(summary["metadata_face_occluded"] for summary in split_summary.values()),
            "metadata_face_occluded_ratio": ratio(sum(summary["metadata_face_occluded"] for summary in split_summary.values()), total_faces),
            "metadata_ellipse_out_of_image": sum(summary["metadata_ellipse_out_of_image"] for summary in split_summary.values()),
            "metadata_ellipse_out_of_image_ratio": ratio(
                sum(summary["metadata_ellipse_out_of_image"] for summary in split_summary.values()), total_faces
            ),
            "metadata_objects": total_metadata_objects,
            "metadata_excluded_faces": sum(summary["metadata_excluded_faces"] for summary in split_summary.values()),
            "metadata_objects_with_occluded_keypoint": sum(
                summary["metadata_objects_with_occluded_keypoint"] for summary in split_summary.values()
            ),
            "metadata_objects_with_occluded_keypoint_ratio": ratio(
                sum(summary["metadata_objects_with_occluded_keypoint"] for summary in split_summary.values()), total_metadata_objects
            ),
        },
        "split_summary": split_summary,
        "bbox_size_distribution": {
            "area_ratio": stats(bbox_area_ratios),
            "width_norm": stats(bbox_width_norms),
            "height_norm": stats(bbox_height_norms),
            "width_px": stats(bbox_width_px),
            "height_px": stats(bbox_height_px),
        },
        "keypoint_visibility_distribution": {
            "overall": dict(sorted(visibility_counts.items())),
            "by_keypoint": {name: dict(sorted(counter.items())) for name, counter in visibility_by_keypoint.items()},
        },
        "image_class_counts_by_split": {
            split: dict(sorted(counter.items())) for split, counter in image_class_counts.items()
        },
        "instance_class_counts_by_split": {
            split: dict(sorted(counter.items())) for split, counter in instance_class_counts.items()
        },
        "duplicate_images_across_splits": duplicate_images[: args.max_listed_issues],
        "near_frame_cross_split_pairs": near_frame_pairs[: args.max_listed_issues],
        "failures": all_failures[: args.max_listed_issues],
    }

    return {
        "report": report,
        "image_rows": image_rows,
        "instance_rows": instance_rows,
        "issue_rows": issue_rows,
        "near_frame_pairs": near_frame_pairs,
        "train_sampling_rows": train_sampling_rows,
    }


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 0 Dataset Audit",
        "",
        "This audit does not run training.",
        "",
        f"- status: `{report['status']}`",
        f"- data root: `{report['data_root']}`",
        f"- data yaml: `{report['data_yaml']}`",
        f"- metadata: `{report['metadata'].get('metadata_path')}`",
        f"- small face threshold: bbox area ratio < `{report['small_face_area_threshold']}`",
        "",
        "## Summary",
        "",
        "| item | value |",
        "| --- | ---: |",
    ]
    summary = report["summary"]
    for key in [
        "images",
        "instances",
        "positive_images",
        "negative_images",
        "small_instances",
        "metadata_faces",
        "metadata_face_occluded",
        "metadata_ellipse_out_of_image",
        "metadata_objects_with_occluded_keypoint",
        "metadata_excluded_faces",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            f"| small_instance_ratio | {pct(summary['small_instance_ratio'])} |",
            f"| metadata_face_occluded_ratio | {pct(summary['metadata_face_occluded_ratio'])} |",
            f"| metadata_ellipse_out_of_image_ratio | {pct(summary['metadata_ellipse_out_of_image_ratio'])} |",
            f"| metadata_objects_with_occluded_keypoint_ratio | {pct(summary['metadata_objects_with_occluded_keypoint_ratio'])} |",
            "",
            "## Split Distribution",
            "",
            "| split | images | positive | negative | instances | small inst. | face occluded | ellipse OOB | excluded faces | near leak |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    pair_counts = report["checks"]["near_frame_cross_split_pair_counts_by_split_pair"]
    for split, row in report["split_summary"].items():
        near_count = sum(value for key, value in pair_counts.items() if split in key.split("/"))
        lines.append(
            f"| {split} | {row['images']} | {row['positive_images']} | {row['negative_images']} | "
            f"{row['instances']} | {row['small_instances']} | {row['metadata_face_occluded']} | "
            f"{row['metadata_ellipse_out_of_image']} | {row['metadata_excluded_faces']} | {near_count} |"
        )

    bbox = report["bbox_size_distribution"]["area_ratio"]
    lines.extend(
        [
            "",
            "## Image Classes",
            "",
            "| class | train | val | test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    image_classes = sorted(
        {
            class_name
            for split_counts in report["image_class_counts_by_split"].values()
            for class_name in split_counts
        }
    )
    for class_name in image_classes:
        counts = report["image_class_counts_by_split"]
        lines.append(
            f"| {class_name} | {counts.get('train', {}).get(class_name, 0)} | "
            f"{counts.get('val', {}).get(class_name, 0)} | {counts.get('test', {}).get(class_name, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Instance Classes",
            "",
            "| class | train | val | test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    instance_classes = sorted(
        {
            class_name
            for split_counts in report["instance_class_counts_by_split"].values()
            for class_name in split_counts
        }
    )
    for class_name in instance_classes:
        counts = report["instance_class_counts_by_split"]
        lines.append(
            f"| {class_name} | {counts.get('train', {}).get(class_name, 0)} | "
            f"{counts.get('val', {}).get(class_name, 0)} | {counts.get('test', {}).get(class_name, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Balanced Training Files",
            "",
            f"- balance classes: `{report['checks']['balance_classes']}`",
            f"- unique train images: `{report['checks']['balanced_train_unique_images']}`",
            f"- repeated train entries: `{report['checks']['balanced_train_repeated_entries']}`",
            "- outputs: `phase0_train_sampling_weights.csv`, `phase0_balanced_train_images.txt`, `face_pose_class_balanced.yaml`",
            "",
            "## Bbox Area Ratio",
            "",
            "| count | min | p10 | p50 | p90 | max | mean |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {bbox.get('count', 0)} | {bbox.get('min', 0):.6f} | {bbox.get('p10', 0):.6f} | "
                f"{bbox.get('p50', 0):.6f} | {bbox.get('p90', 0):.6f} | {bbox.get('max', 0):.6f} | "
                f"{bbox.get('mean', 0):.6f} |"
            ),
            "",
            "## Keypoint Visibility",
            "",
            "| keypoint | v0 | v1 | v2 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, counts in report["keypoint_visibility_distribution"]["by_keypoint"].items():
        lines.append(f"| {name} | {counts.get('0', 0)} | {counts.get('1', 0)} | {counts.get('2', 0)} |")

    checks = report["checks"]
    lines.extend(
        [
            "",
            "## Leakage / Structure Checks",
            "",
            f"- failures: `{checks['failure_count']}`",
            f"- duplicate images across splits: `{checks['duplicate_image_cross_split_count']}`",
            f"- near-frame cross-split pairs: `{checks['near_frame_cross_split_pair_count']}`",
            f"- near-frame pair counts by split pair: `{checks['near_frame_cross_split_pair_counts_by_split_pair']}`",
            f"- ellipse out of image count: `{checks['ellipse_out_of_image_count']}`",
            f"- metadata/object mismatch images: `{checks['metadata_object_mismatch_image_count']}`",
            f"- flip_idx ok: `{checks['flip_idx_ok']}`",
            "",
            "## Notes",
            "",
            "- `metadata_face_occluded` and `metadata_ellipse_out_of_image` require conversion metadata.",
            "- `ellipse OOB` means the raw ellipse-derived bbox extends outside the image before YOLO clipping.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result = audit_dataset(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "phase0_dataset_audit.json", result["report"])
    write_csv(args.output_dir / "phase0_dataset_images.csv", result["image_rows"])
    write_csv(args.output_dir / "phase0_dataset_instances.csv", result["instance_rows"])
    write_csv(args.output_dir / "phase0_dataset_issues.csv", result["issue_rows"])
    write_csv(args.output_dir / "phase0_near_frame_pairs.csv", result["near_frame_pairs"])
    write_csv(args.output_dir / "phase0_train_sampling_weights.csv", result["train_sampling_rows"])
    write_balanced_train_files(
        args.output_dir,
        Path(result["report"]["data_root"]),
        result["report"].get("keypoint_names", DEFAULT_KEYPOINT_NAMES),
        result["train_sampling_rows"],
    )
    write_text(args.output_dir / "phase0_dataset_audit.md", make_markdown(result["report"]))
    print(json.dumps({"audit": result["report"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
