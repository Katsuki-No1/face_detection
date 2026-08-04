#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/datasets/active_learning_effect.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.script_data.export_face_keypoints_deim_dino import (  # noqa: E402
    export_deim_dino,
)
from scripts_YOLO.convert_annotations_to_yolo_pose import (  # noqa: E402
    load_records_from_zip,
    yolo_line_for_face,
)


class DatasetBuildError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable-base training views for the normal400 versus "
            "active-learning357 comparison."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the two configured derived dataset/adapter directories.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise DatasetBuildError("Dataset comparison config must use schema_version: 1")
    for section in ("base", "normal", "active", "arms"):
        if not isinstance(raw.get(section), dict):
            raise DatasetBuildError(f"Missing mapping: {section}")
    raw["_config_path"] = config_path.as_posix()
    raw["_config_sha256"] = sha256_file(config_path)
    return raw


def _stable_annotation_kind(pool_name: str, zip_path: Path) -> str:
    relative = zip_path.relative_to(PROJECT_ROOT)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.with_suffix("").as_posix())
    return f"{pool_name}__{token}"


def _record_image_path(zip_path: Path, image_name: str) -> Path:
    relative = Path(image_name)
    candidates = (zip_path.parent / relative, zip_path.parent / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise DatasetBuildError(
        f"Image referenced by {zip_path} does not exist: {image_name}"
    )


def load_pool(pool_name: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    zip_values = raw.get("annotation_zips")
    if not isinstance(zip_values, list) or not zip_values:
        raise DatasetBuildError(f"{pool_name}.annotation_zips must be non-empty")
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for value in zip_values:
        zip_path = resolve_path(value)
        if not zip_path.is_file():
            raise DatasetBuildError(f"Missing annotation zip: {zip_path}")
        annotation_kind = _stable_annotation_kind(pool_name, zip_path)
        for record in load_records_from_zip(zip_path):
            row = deepcopy(record)
            image_name = str(row.get("image_name", ""))
            if not image_name or image_name in seen_names:
                raise DatasetBuildError(
                    f"Duplicate or empty {pool_name} image name: {image_name!r}"
                )
            seen_names.add(image_name)
            image_path = _record_image_path(zip_path, image_name)
            row["annotation_kind"] = annotation_kind
            row["annotation_xml"] = f"{zip_path.as_posix()}::annotations.xml"
            row["image_path"] = image_path.as_posix()
            row["source_annotation_zip"] = zip_path.as_posix()
            row["content_sha256"] = sha256_file(image_path)
            records.append(row)
    records.sort(
        key=lambda row: (
            str(row.get("source_annotation_zip", "")),
            str(row.get("image_name", "")),
        )
    )
    expected = int(raw["expected_pool_frames"])
    if len(records) != expected:
        raise DatasetBuildError(
            f"Unexpected {pool_name} pool size: expected={expected}, actual={len(records)}"
        )
    hashes = [str(row["content_sha256"]) for row in records]
    if len(set(hashes)) != len(hashes):
        raise DatasetBuildError(f"{pool_name} contains byte-identical frame duplicates")
    return records


def select_uniform_records(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(records):
        raise DatasetBuildError(
            f"Invalid sample count {count} for pool of {len(records)} records"
        )
    indices = sorted(random.Random(seed).sample(range(len(records)), count))
    return [deepcopy(records[index]) for index in indices]


def training_eligibility(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether the standard converter can supervise this frame."""
    faces = list(record.get("faces", []))
    if not faces:
        return True, []
    face_bboxes = [face["bbox"] for face in faces]
    reasons: list[str] = []
    for face_index, face in enumerate(faces):
        line, object_metadata = yolo_line_for_face(
            record,
            face,
            face_bboxes,
            face_index,
        )
        if line is not None:
            return True, []
        reason = str(object_metadata.get("exclusion_reason", "")).strip()
        if reason:
            reasons.append(reason)
        else:
            reasons.extend(str(item) for item in object_metadata.get("warnings", []))
    return False, sorted(set(reasons or ["no_valid_face_objects"]))


def filter_training_eligible_records(
    pool_name: str,
    records: list[dict[str, Any]],
    raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        is_eligible, reasons = training_eligibility(record)
        if is_eligible:
            eligible.append(record)
            continue
        excluded.append(
            {
                "image_name": str(record.get("image_name", "")),
                "image_stem": str(record.get("image_stem", "")),
                "image_path": str(record.get("image_path", "")),
                "source_annotation_zip": str(
                    record.get("source_annotation_zip", "")
                ),
                "content_sha256": str(record.get("content_sha256", "")),
                "reasons": reasons,
            }
        )
    expected_eligible = int(raw["expected_eligible_frames"])
    expected_excluded = int(raw["expected_excluded_frames"])
    if len(eligible) != expected_eligible or len(excluded) != expected_excluded:
        raise DatasetBuildError(
            f"Unexpected {pool_name} converter eligibility: "
            f"expected eligible/excluded={expected_eligible}/{expected_excluded}, "
            f"actual={len(eligible)}/{len(excluded)}"
        )
    return eligible, excluded


def selected_records(
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    normal_pool = load_pool("normal", config["normal"])
    active_pool = load_pool("active", config["active"])
    normal_hashes = {str(row["content_sha256"]) for row in normal_pool}
    active_hashes = {str(row["content_sha256"]) for row in active_pool}
    if normal_hashes & active_hashes:
        raise DatasetBuildError("Normal and active-learning pools overlap by SHA-256")
    normal_eligible, normal_excluded = filter_training_eligible_records(
        "normal", normal_pool, config["normal"]
    )
    active_eligible, active_excluded = filter_training_eligible_records(
        "active", active_pool, config["active"]
    )
    seed = int(config["selection_seed"])
    normal = select_uniform_records(
        normal_eligible,
        count=int(config["normal"]["selected_frames"]),
        seed=seed,
    )
    active = [deepcopy(row) for row in active_eligible]
    if len(active) != int(config["active"]["selected_frames"]):
        raise DatasetBuildError("Active-learning selection must include its full pool")
    audit = {
        "selection_seed": seed,
        "normal_pool_frames": len(normal_pool),
        "normal_eligible_frames": len(normal_eligible),
        "normal_excluded_frames": len(normal_excluded),
        "normal_exclusions": normal_excluded,
        "normal_selected_frames": len(normal),
        "active_pool_frames": len(active_pool),
        "active_eligible_frames": len(active_eligible),
        "active_excluded_frames": len(active_excluded),
        "active_exclusions": active_excluded,
        "active_selected_frames": len(active),
        "normal_selected_groups": len({str(row["image_stem"]) for row in normal}),
        "active_selected_groups": len({str(row["image_stem"]) for row in active}),
        "normal_active_content_overlap": 0,
    }
    return {"normal400": normal, "active357": active}, audit


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialize_base_tree(base_root: Path, staging: Path) -> Counter[str]:
    modes: Counter[str] = Counter()
    for tree_name in ("images", "labels"):
        tree = base_root / tree_name
        if not tree.is_dir():
            raise DatasetBuildError(f"Missing base dataset tree: {tree}")
        for source in sorted(path for path in tree.rglob("*") if path.is_file()):
            relative = source.relative_to(base_root)
            modes[_link_or_copy(source, staging / relative)] += 1
    return modes


def _extra_metadata_record(
    record: dict[str, Any],
    *,
    staging: Path,
) -> dict[str, Any]:
    source = Path(str(record["image_path"]))
    content_sha256 = str(record.get("content_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
        raise DatasetBuildError(f"Invalid or missing image SHA-256: {source}")
    suffix = source.suffix.lower() or ".jpg"
    output_name = f"extra__{content_sha256[:24]}{suffix}"
    output_image = Path("images/train") / output_name
    output_label = Path("labels/train") / f"{Path(output_name).stem}.txt"
    image_target = staging / output_image
    if image_target.exists():
        raise DatasetBuildError(f"Derived image collision: {output_image}")

    metadata = {
        "annotation_kind": record.get("annotation_kind", ""),
        "image_stem": record.get("image_stem", ""),
        "image_name": record.get("image_name", ""),
        "image_path": source.as_posix(),
        "split": "train",
        "heads": record.get("heads", []),
        "faces": record.get("faces", []),
        "objects": [],
        "excluded_faces": [],
        "skipped": [],
        "source_annotation_zip": record.get("source_annotation_zip", ""),
        "content_sha256": record.get("content_sha256", ""),
        "selection_source": record.get("selection_source", ""),
    }
    lines: list[str] = []
    faces = list(record.get("faces", []))
    if faces:
        face_bboxes = [face["bbox"] for face in faces]
        for face_index, face in enumerate(faces):
            line, object_metadata = yolo_line_for_face(
                record,
                face,
                face_bboxes,
                face_index,
            )
            object_metadata["face_index"] = face_index
            if line is None:
                metadata["excluded_faces"].append(object_metadata)
                continue
            lines.append(line)
            metadata["objects"].append(object_metadata)
        if not lines:
            raise DatasetBuildError(
                f"Selected positive frame has no valid Face objects: {source}"
            )
        metadata["negative_sample"] = False
    else:
        metadata["negative_sample"] = True
        metadata["warnings"] = ["no_face_ellipse_negative_sample"]

    _link_or_copy(source, image_target)
    label_target = staging / output_label
    label_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    metadata["output_image"] = output_image.as_posix()
    metadata["output_label"] = output_label.as_posix()
    return metadata


def _split_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("split", "")) for record in records)
    return {split: int(counts[split]) for split in ("train", "val", "test")}


def _write_selection_manifest(
    path: Path,
    selected: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "image_name",
                "image_stem",
                "image_path",
                "source_annotation_zip",
                "content_sha256",
                "has_face",
            ),
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "image_name": row["image_name"],
                    "image_stem": row["image_stem"],
                    "image_path": row["image_path"],
                    "source_annotation_zip": row["source_annotation_zip"],
                    "content_sha256": row["content_sha256"],
                    "has_face": bool(row.get("faces")),
                }
            )


def _remove_configured_target(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise DatasetBuildError(
            f"Derived output already exists; use --force to replace this exact target: {path}"
        )
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def build_arm(
    *,
    arm_name: str,
    arm_config: dict[str, Any],
    selected: list[dict[str, Any]],
    config: dict[str, Any],
    base_metadata: dict[str, Any],
    base_root: Path,
    base_hashes: set[str],
    force: bool,
) -> dict[str, Any]:
    dataset_root = resolve_path(arm_config["dataset_root"])
    adapter_root = resolve_path(arm_config["adapter_root"])
    _remove_configured_target(dataset_root, force=force)
    _remove_configured_target(adapter_root, force=force)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    adapter_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{arm_name}.", dir=dataset_root.parent))
    adapter_staging = Path(
        tempfile.mkdtemp(prefix=f".{arm_name}.", dir=adapter_root.parent)
    )
    try:
        materialization = _materialize_base_tree(base_root, staging)
        extra_hashes = {str(row["content_sha256"]) for row in selected}
        overlap = base_hashes & extra_hashes
        if overlap:
            raise DatasetBuildError(
                f"{arm_name} additions overlap datasetv2 by {len(overlap)} content hashes"
            )
        tagged = []
        for row in selected:
            copy = deepcopy(row)
            copy["selection_source"] = arm_name
            tagged.append(copy)
        extra_metadata = [
            _extra_metadata_record(row, staging=staging)
            for row in tagged
        ]
        records = [deepcopy(row) for row in base_metadata["records"]]
        records.extend(extra_metadata)
        expected_base = dict(config["base"]["expected_images"])
        expected_counts = {
            "train": int(expected_base["train"]) + len(selected),
            "val": int(expected_base["val"]),
            "test": int(expected_base["test"]),
        }
        actual_counts = _split_counts(records)
        if actual_counts != expected_counts:
            raise DatasetBuildError(
                f"{arm_name} split mismatch: expected={expected_counts}, actual={actual_counts}"
            )
        if any(str(row.get("split")) != "train" for row in extra_metadata):
            raise DatasetBuildError(f"{arm_name} has an added record outside train")

        metadata = {
            "schema_version": "2.0",
            "dataset_version": arm_config["dataset_version"],
            "keypoint_names": base_metadata["keypoint_names"],
            "bbox_source": base_metadata["bbox_source"],
            "visibility_policy": base_metadata["visibility_policy"],
            "source": {
                "base_dataset": config["base"]["dataset_version"],
                "base_metadata": str(config["base"]["metadata"]),
                "comparison_config": config["_config_path"],
                "comparison_config_sha256": config["_config_sha256"],
                "selection_seed": int(config["selection_seed"]),
                "arm": arm_name,
                "annotation_zips": sorted(
                    {str(row["source_annotation_zip"]) for row in selected}
                ),
                "annotation_zip_sha256": {
                    str(path): sha256_file(Path(path))
                    for path in sorted(
                        {str(row["source_annotation_zip"]) for row in selected}
                    )
                },
            },
            "split_policy": {
                "base": "locked face_pose_drive_v2_balanced_811 split",
                "additions": "train_only",
                "val_test_changed": False,
            },
            "summary": {
                "records": len(records),
                "written_images": len(records),
                "negative_images": sum(
                    bool(row.get("negative_sample", False)) for row in records
                ),
                "written_objects": sum(len(row.get("objects", [])) for row in records),
                "splits": actual_counts,
                "base_train_images": int(expected_base["train"]),
                "added_train_images": len(selected),
                "added_positive_images": sum(bool(row.get("objects")) for row in extra_metadata),
                "added_negative_images": sum(not bool(row.get("objects")) for row in extra_metadata),
                "added_groups": len({str(row["image_stem"]) for row in extra_metadata}),
            },
            "records": records,
        }
        metadata_dir = staging / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "conversion_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_selection_manifest(metadata_dir / "selected_additions.csv", tagged)
        (staging / "dataset_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "dataset_version": arm_config["dataset_version"],
                    "base_dataset": config["base"]["dataset_version"],
                    "split_policy": "locked-base-with-train-only-additions",
                    "image_counts": actual_counts,
                    "added_train_images": len(selected),
                    "metadata": "metadata/conversion_metadata.json",
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        os.replace(staging, dataset_root)
        export_summary = export_deim_dino(
            dataset_root,
            output_root=adapter_staging,
        )
        os.replace(adapter_staging, adapter_root)
        annotation_hashes = {
            split: sha256_file(adapter_root / f"face_keypoints_{split}.json")
            for split in ("train", "val", "test")
        }
        build_manifest = {
            "schema_version": 1,
            "status": "ok",
            "arm": arm_name,
            "dataset_version": arm_config["dataset_version"],
            "dataset_root": dataset_root.as_posix(),
            "adapter_root": adapter_root.as_posix(),
            "selection_seed": int(config["selection_seed"]),
            "selection_fingerprint": sha256_json(
                [str(row["content_sha256"]) for row in tagged]
            ),
            "split_counts": actual_counts,
            "base_split_counts": expected_base,
            "added_train_images": len(selected),
            "added_groups": metadata["summary"]["added_groups"],
            "added_positive_images": metadata["summary"]["added_positive_images"],
            "added_negative_images": metadata["summary"]["added_negative_images"],
            "base_content_overlap": 0,
            "materialization": dict(materialization),
            "annotation_hashes": annotation_hashes,
            "export_summary": export_summary,
        }
        (dataset_root / "build_manifest.json").write_text(
            json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return build_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if adapter_staging.exists():
            shutil.rmtree(adapter_staging)
        raise


def build(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    base_root = resolve_path(config["base"]["dataset_root"])
    base_metadata_path = resolve_path(config["base"]["metadata"])
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    base_records = base_metadata.get("records")
    if not isinstance(base_records, list):
        raise DatasetBuildError("Base metadata records are missing")
    if len(base_records) != int(config["base"]["expected_records"]):
        raise DatasetBuildError(
            f"Base record count mismatch: {len(base_records)}"
        )
    expected_base_counts = {
        key: int(value) for key, value in config["base"]["expected_images"].items()
    }
    if _split_counts(base_records) != expected_base_counts:
        raise DatasetBuildError("Base split inventory does not match the locked contract")
    base_hashes = {
        sha256_file(base_root / str(row["output_image"]))
        for row in base_records
    }
    if len(base_hashes) != len(base_records):
        raise DatasetBuildError("Base dataset contains duplicate image content")

    selections, selection_audit = selected_records(config)
    manifests = {}
    for arm_name in ("normal400", "active357"):
        manifests[arm_name] = build_arm(
            arm_name=arm_name,
            arm_config=config["arms"][arm_name],
            selected=selections[arm_name],
            config=config,
            base_metadata=base_metadata,
            base_root=base_root,
            base_hashes=base_hashes,
            force=force,
        )
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": config["experiment_id"],
        "config": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "selection": selection_audit,
        "arms": manifests,
    }
    summary_path = (
        resolve_path(config["arms"]["normal400"]["dataset_root"]).parent
        / "comparison_dataset_summary.json"
    )
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    result = build(args.config, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
