#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/datasets/default_face_pose.yaml"
SPLITS = ("train", "val", "test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DatasetPreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _absolute_without_resolving_symlinks(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return Path(os.path.abspath(candidate))


def load_contract(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetPreparationError(f"Could not load dataset contract {path}: {error}") from error
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise DatasetPreparationError("Dataset contract schema_version must be 1")
    contract = raw.get("default_dataset")
    if not isinstance(contract, dict):
        raise DatasetPreparationError("default_dataset mapping is required")
    return contract


def validate_archive(archive: Path, contract: dict[str, Any]) -> None:
    if not archive.is_file():
        raise DatasetPreparationError(f"Dataset archive does not exist: {archive}")
    expected_sha = str(contract.get("archive_sha256", "")).lower()
    actual_sha = sha256_file(archive)
    if actual_sha != expected_sha:
        raise DatasetPreparationError(
            f"Dataset archive SHA-256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    expected_root = str(contract["archive_root"]).rstrip("/") + "/"
    destination = _resolve(contract["materialized_root"]).parent
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise DatasetPreparationError(f"Corrupt member in dataset archive: {corrupt}")
        members = bundle.infolist()
        if not members:
            raise DatasetPreparationError("Dataset archive is empty")
        for member in members:
            name = member.filename
            if name in {"SHA256SUMS", "EXPORT_README_JA.md", "export_metadata.json"}:
                continue
            if not name.startswith(expected_root):
                raise DatasetPreparationError(f"Unexpected archive root for member: {name}")
            target = (destination / name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise DatasetPreparationError(f"Unsafe archive member path: {name}")


def extract_archive(archive: Path, contract: dict[str, Any]) -> Path:
    root = _resolve(contract["materialized_root"])
    if root.exists():
        if not root.is_dir():
            raise DatasetPreparationError(f"Materialized dataset path is not a directory: {root}")
        return root
    destination = root.parent
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    if not root.is_dir():
        raise DatasetPreparationError(f"Archive did not create expected dataset root: {root}")
    return root


def verify_export_checksums(archive: Path, dataset_root: Path) -> int:
    with zipfile.ZipFile(archive) as bundle:
        try:
            checksum_text = bundle.read("SHA256SUMS").decode("utf-8")
        except (KeyError, UnicodeDecodeError) as error:
            raise DatasetPreparationError("Archive has no valid SHA256SUMS") from error
    verified = 0
    expected_prefix = dataset_root.name + "/"
    for line in checksum_text.splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise DatasetPreparationError(f"Malformed SHA256SUMS line: {line!r}") from error
        relative = relative.lstrip("*")
        if not relative.startswith(expected_prefix):
            raise DatasetPreparationError(f"Checksum entry is outside dataset root: {relative}")
        local_relative = relative[len(expected_prefix):]
        path = dataset_root / local_relative
        if not path.is_file():
            raise DatasetPreparationError(f"Archive member is missing after extraction: {path}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise DatasetPreparationError(
                f"Extracted file checksum mismatch: {path}; expected={expected}, actual={actual}"
            )
        verified += 1
    return verified


def make_cache_read_only(dataset_root: Path) -> None:
    paths = sorted(dataset_root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        mode = path.stat().st_mode & 0o7777
        path.chmod(mode & ~0o222)
    root_mode = dataset_root.stat().st_mode & 0o7777
    dataset_root.chmod(root_mode & ~0o222)


def verify_cache_read_only(dataset_root: Path) -> None:
    for path in (dataset_root, *dataset_root.rglob("*")):
        if path.stat().st_mode & 0o222:
            raise DatasetPreparationError(f"Dataset cache is writable: {path}")


def ensure_deim_annotations(
    dataset_root: Path,
    output_root: Path,
    *,
    regenerate: bool,
) -> dict[str, Any]:
    required = [output_root / f"face_keypoints_{split}.json" for split in SPLITS]
    if regenerate or not all(path.is_file() for path in required):
        from script.script_data.export_face_keypoints_deim_dino import export_deim_dino

        return export_deim_dino(dataset_root, output_root=output_root)
    summary_path = output_root / "conversion_summary.json"
    if not summary_path.is_file():
        raise DatasetPreparationError(f"Missing DEIM conversion summary: {summary_path}")
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"Invalid DEIM conversion summary: {summary_path}")
    return value


def ensure_rtmpose_annotations(
    dataset_root: Path,
    output_root: Path,
    *,
    regenerate: bool,
) -> dict[str, Any]:
    required = [output_root / f"person_keypoints_{split}.json" for split in SPLITS]
    if regenerate or not all(path.is_file() for path in required):
        from script.script_data.export_face_keypoints_rtmpose import export_rtmpose

        return export_rtmpose(dataset_root, output_root=output_root)
    summary_path = output_root / "conversion_summary.json"
    if not summary_path.is_file():
        raise DatasetPreparationError(f"Missing RTMPose conversion summary: {summary_path}")
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"Invalid RTMPose conversion summary: {summary_path}")
    return value


def write_local_yolo_yaml(adapter_root: Path, canonical_root: Path) -> Path:
    adapter_root.mkdir(parents=True, exist_ok=True)
    path = adapter_root / "face_pose.yaml"
    payload = {
        "path": str(canonical_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "face"},
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "keypoints": ["e1", "e2", "n", "m1", "m2"],
    }
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def activate_canonical_root(dataset_root: Path, canonical_root: Path) -> None:
    canonical_root.parent.mkdir(parents=True, exist_ok=True)
    if canonical_root.exists() and not canonical_root.is_symlink():
        raise DatasetPreparationError(
            f"Canonical dataset root exists and is not a symlink; refusing to replace it: {canonical_root}"
        )
    with tempfile.TemporaryDirectory(
        prefix=".face_pose_dataset_switch_",
        dir=canonical_root.parent,
    ) as temp_dir:
        temporary_link = Path(temp_dir) / canonical_root.name
        temporary_link.symlink_to(dataset_root, target_is_directory=True)
        os.replace(temporary_link, canonical_root)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetPreparationError(f"Could not read annotation JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"Annotation JSON must be an object: {path}")
    return value


def audit_coco(
    dataset_root: Path,
    annotation_paths: dict[str, str | Path],
    expected_images: dict[str, int],
    expected_annotations: dict[str, int],
    *,
    require_ellipses: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    summaries: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    all_file_names: set[str] = set()
    for split in SPLITS:
        configured_path = Path(annotation_paths[split])
        path = (
            configured_path
            if configured_path.is_absolute()
            else dataset_root / configured_path
        )
        data = _load_json(path)
        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise DatasetPreparationError(f"COCO images/annotations must be lists: {path}")
        if len(images) != int(expected_images[split]):
            raise DatasetPreparationError(
                f"Unexpected {split} image count in {path}: {len(images)}"
            )
        if len(annotations) != int(expected_annotations[split]):
            raise DatasetPreparationError(
                f"Unexpected {split} annotation count in {path}: {len(annotations)}"
            )
        image_ids = [int(row["id"]) for row in images]
        annotation_ids = [int(row["id"]) for row in annotations]
        if len(set(image_ids)) != len(image_ids) or len(set(annotation_ids)) != len(annotation_ids):
            raise DatasetPreparationError(f"Duplicate COCO IDs in {path}")
        image_id_set = set(image_ids)
        if any(int(row["image_id"]) not in image_id_set for row in annotations):
            raise DatasetPreparationError(f"Annotation references a missing image in {path}")
        split_names: set[str] = set()
        for image in images:
            file_name = str(image.get("file_name", ""))
            if not file_name.startswith(f"images/{split}/"):
                raise DatasetPreparationError(
                    f"Image path does not match split={split}: {file_name}"
                )
            if file_name in split_names or file_name in all_file_names:
                raise DatasetPreparationError(f"Duplicate image path across splits: {file_name}")
            split_names.add(file_name)
            if not (dataset_root / file_name).is_file():
                raise DatasetPreparationError(f"COCO image does not exist: {file_name}")
        all_file_names.update(split_names)
        if require_ellipses:
            for annotation in annotations:
                ellipse = annotation.get("ellipse")
                if not isinstance(ellipse, list) or len(ellipse) != 5:
                    raise DatasetPreparationError(
                        f"Missing ellipse in {path}, annotation={annotation.get('id')}"
                    )
                values = [float(value) for value in ellipse]
                if not all(math.isfinite(value) for value in values):
                    raise DatasetPreparationError(f"Non-finite ellipse in {path}")
                if values[2] < values[3] or values[3] <= 0.0:
                    raise DatasetPreparationError(f"Invalid ellipse axes in {path}: {ellipse}")
        categories = Counter(int(row["category_id"]) for row in annotations)
        summaries[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "category_annotations": {str(key): value for key, value in sorted(categories.items())},
        }
        hashes[split] = sha256_file(path)
    return summaries, hashes


def audit_yolo(dataset_root: Path, expected_images: dict[str, int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        labels = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")
        if len(images) != int(expected_images[split]) or len(labels) != len(images):
            raise DatasetPreparationError(
                f"YOLO {split} image/label mismatch: images={len(images)}, labels={len(labels)}"
            )
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise DatasetPreparationError(f"YOLO {split} image/label stems do not match")
        instances = 0
        for label in labels:
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 20:
                    raise DatasetPreparationError(
                        f"YOLO Pose row must contain 20 fields: {label}:{line_number}"
                    )
                numeric = [float(value) for value in values]
                if int(numeric[0]) != 0 or any(not math.isfinite(value) for value in numeric):
                    raise DatasetPreparationError(f"Invalid YOLO Pose row: {label}:{line_number}")
                if any(value < 0.0 or value > 1.0 for value in numeric[1:5]):
                    raise DatasetPreparationError(f"YOLO box is outside [0,1]: {label}:{line_number}")
                instances += 1
        result[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": instances,
        }
    return result


def audit_dataset(
    dataset_root: Path,
    canonical_root: Path,
    adapter_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected = contract["expected"]
    annotations = contract["annotations"]
    image_counts = {split: int(expected["images"][split]) for split in SPLITS}
    face_counts = {split: int(expected["face_annotations"][split]) for split in SPLITS}
    head_counts = {split: int(expected["head_annotations"][split]) for split in SPLITS}
    head_face_counts = {
        split: int(expected["head_face_annotations"][split]) for split in SPLITS
    }
    deim, deim_hashes = audit_coco(
        dataset_root,
        {
            split: adapter_root / annotations["deim_dino"][split]
            for split in SPLITS
        },
        image_counts,
        face_counts,
        require_ellipses=True,
    )
    face_dino, face_dino_hashes = audit_coco(
        dataset_root,
        annotations["face_dino"],
        image_counts,
        head_counts,
        # Head-only records without an associated face intentionally carry a
        # zero ellipse. This set is valid for head detection, not as the
        # direct-ellipse training source.
        require_ellipses=False,
    )
    rtmpose, rtmpose_hashes = audit_coco(
        dataset_root,
        {
            split: adapter_root / annotations["rtmpose"][split]
            for split in SPLITS
        },
        image_counts,
        face_counts,
    )
    rtmdet, rtmdet_hashes = audit_coco(
        dataset_root,
        annotations["rtmdet_hf"],
        image_counts,
        head_face_counts,
    )
    local_yaml = adapter_root / "face_pose.yaml"
    yolo_config = yaml.safe_load(local_yaml.read_text(encoding="utf-8"))
    if Path(str(yolo_config.get("path", ""))).resolve() != canonical_root.resolve():
        raise DatasetPreparationError(f"YOLO dataset root is not canonical: {local_yaml}")
    return {
        "status": "ok",
        "dataset_version": str(contract["version"]),
        "dataset_root": str(dataset_root),
        "canonical_root": str(canonical_root),
        "adapter_root": str(adapter_root),
        "yolo": audit_yolo(dataset_root, image_counts),
        "deim_dino": {"splits": deim, "sha256": deim_hashes},
        "face_dino": {"splits": face_dino, "sha256": face_dino_hashes},
        "rtmpose": {"splits": rtmpose, "sha256": rtmpose_hashes},
        "rtmdet_hf": {"splits": rtmdet, "sha256": rtmdet_hashes},
    }


def prepare(
    config_path: Path,
    *,
    verify_only: bool = False,
    regenerate_deim: bool = False,
    activate: bool = True,
) -> dict[str, Any]:
    contract = load_contract(config_path)
    archive = _resolve(contract["archive"])
    dataset_root = _resolve(contract["materialized_root"])
    canonical_root = _absolute_without_resolving_symlinks(contract["canonical_root"])
    adapter_root = _resolve(contract["adapter_root"])
    validate_archive(archive, contract)
    if verify_only and not dataset_root.is_dir():
        raise DatasetPreparationError(
            f"Dataset has not been materialized; rerun without --verify-only: {dataset_root}"
        )
    if not verify_only:
        extract_archive(archive, contract)
    verified_files = verify_export_checksums(archive, dataset_root)
    if verify_only:
        verify_cache_read_only(dataset_root)
    else:
        make_cache_read_only(dataset_root)
    if verify_only:
        missing_adapters = [
            adapter_root / f"face_keypoints_{split}.json"
            for split in SPLITS
            if not (adapter_root / f"face_keypoints_{split}.json").is_file()
        ]
        missing_adapters.extend(
            adapter_root / "rtmpose" / f"person_keypoints_{split}.json"
            for split in SPLITS
            if not (
                adapter_root / "rtmpose" / f"person_keypoints_{split}.json"
            ).is_file()
        )
        if missing_adapters:
            raise DatasetPreparationError(
                f"Missing DEIM adapters during verify-only: {missing_adapters}"
            )
    ensure_deim_annotations(
        dataset_root,
        adapter_root,
        regenerate=regenerate_deim and not verify_only,
    )
    ensure_rtmpose_annotations(
        dataset_root,
        adapter_root / "rtmpose",
        regenerate=regenerate_deim and not verify_only,
    )
    if activate and not verify_only:
        activate_canonical_root(dataset_root, canonical_root)
    elif not canonical_root.exists():
        if verify_only:
            raise DatasetPreparationError(
                f"Canonical dataset root is not active: {canonical_root}"
            )
        canonical_root = dataset_root
    local_yolo_yaml = adapter_root / "face_pose.yaml"
    if verify_only:
        if not local_yolo_yaml.is_file():
            raise DatasetPreparationError(f"Missing YOLO adapter: {local_yolo_yaml}")
    else:
        write_local_yolo_yaml(adapter_root, canonical_root)
    report = audit_dataset(
        dataset_root,
        canonical_root,
        adapter_root,
        contract,
    )
    report["archive"] = str(archive)
    report["archive_sha256"] = sha256_file(archive)
    report["archive_files_verified"] = verified_files
    report["dataset_cache_read_only"] = True
    report_path = adapter_root / "dataset_activation_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize, validate, and activate the repository's default face-pose dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--regenerate-adapters",
        "--regenerate-deim",
        dest="regenerate_deim",
        action="store_true",
    )
    parser.add_argument("--no-activate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    report = prepare(
        config,
        verify_only=bool(args.verify_only),
        regenerate_deim=bool(args.regenerate_deim),
        activate=not bool(args.no_activate),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
