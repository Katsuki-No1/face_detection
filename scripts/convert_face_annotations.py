"""Build the shared face dataset and its compatibility label/metadata files."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.face_annotations import (
    KEYPOINT_NAMES, build_image_index, discover_annotation_zips,
    load_records_from_json, load_records_from_xml, load_records_from_zip,
    resolve_image_path, safe_output_name,
)
from src.data.face_pose_labels import normalized_pose_line_for_face
from src.data.keypoint_side_assignment import POLICY, V1_POLICY, V2_POLICY, LEGACY_POLICY
from src.data.face_pose_split import SPLITS, build_split_map, split_balance_report, split_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CVAT face annotations into the shared face dataset."
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
        help="Shared dataset output directory. Default: datasets/face_pose.",
    )
    parser.add_argument("--keypoint-side-policy", choices=(POLICY, V1_POLICY, V2_POLICY, LEGACY_POLICY), default=POLICY,
                        help="Unique-ellipse rotating-axis correction with nose fallback; v1/v2/legacy reproduce historical policies")
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


def write_pose_sample(
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
            "- RTMPose: `configs/rtmpose/rtmpose_s_face_pose_drive_v2_256.py`",
            "- DINOv3/DEIMv2: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full.yaml`",
            "- RTMDet Head+Face: `configs/rtmdet/rtmdet_s_hf_drive_v2_960.py`",
            "",
            "## Rebuild the unified dataset",
            "",
            "```bash",
            ".venv/bin/python scripts/convert_face_annotations.py \\",
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
            output_image, output_label = write_pose_sample(
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
            line, object_meta = normalized_pose_line_for_face(record, face, face_bboxes, face_index,
                                                             side_policy=args.keypoint_side_policy)
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

        output_image, output_label = write_pose_sample(args.output_dir, split, record, image_path, lines)
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
