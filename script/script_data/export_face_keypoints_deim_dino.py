#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .face_keypoint_export_common import (
        FLIP_INDICES,
        SKELETON,
        SPLITS,
        TARGET_KEYPOINT_NAMES,
        bbox_xywh,
        ellipse_params,
        image_entry,
        keypoint_payload,
        load_canonical_source,
        sha256_file,
        write_json_atomic,
    )
except ImportError:
    from face_keypoint_export_common import (
        FLIP_INDICES,
        SKELETON,
        SPLITS,
        TARGET_KEYPOINT_NAMES,
        bbox_xywh,
        ellipse_params,
        image_entry,
        keypoint_payload,
        load_canonical_source,
        sha256_file,
        write_json_atomic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DEIM/DINO face keypoints directly from canonical CVAT metadata."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Canonical conversion_metadata.json; defaults below dataset_root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <dataset_root>/annotations/deim_dino",
    )
    return parser.parse_args()


def export_deim_dino(
    dataset_root: Path,
    *,
    metadata: Path | None = None,
    output_root: Path | None = None,
) -> dict:
    source = load_canonical_source(dataset_root, metadata)
    destination = (
        output_root.expanduser().resolve()
        if output_root is not None
        else source.dataset_root / "annotations" / "deim_dino"
    )
    summaries = {}
    for split in SPLITS:
        images = []
        annotations = []
        visibility_counts: Counter[int] = Counter()
        face_occluded = 0
        keypoint_occluded_instances = 0
        ellipse_count = 0
        annotation_id = 1
        for image_id, record in enumerate(source.records_by_split[split], start=1):
            image, width, height = image_entry(source, record, image_id=image_id)
            images.append(image)
            for obj in record.get("objects", []):
                bbox = bbox_xywh(obj, width=width, height=height)
                ellipse = ellipse_params(obj)
                keypoints, visibility, occluded_flags = keypoint_payload(
                    obj,
                    width=width,
                    height=height,
                )
                visibility_counts.update(visibility)
                is_face_occluded = bool(obj.get("face", {}).get("occluded", False))
                face_occluded += int(is_face_occluded)
                keypoint_occluded_instances += int(any(occluded_flags))
                ellipse_count += 1
                occluded_names = [
                    name
                    for name, flag in zip(TARGET_KEYPOINT_NAMES, occluded_flags)
                    if flag
                ]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        # DEIMv2 face checkpoints use category id 2.
                        "category_id": 2,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        # Required by the existing DEIMv2 ellipse dataset
                        # loader/loss/evaluator. Extra keypoints are ignored by
                        # pure face-detection training and consumed by DINO and
                        # joint face-keypoint training.
                        "ellipse": ellipse,
                        "keypoints": keypoints,
                        "num_keypoints": sum(state > 0 for state in visibility),
                        "attributes": {
                            "face_occluded": is_face_occluded,
                            "keypoint_occluded": occluded_flags,
                            "occluded_keypoint_names": occluded_names,
                            "occluded_keypoint_count": len(occluded_names),
                        },
                    }
                )
                annotation_id += 1
        output = {
            "info": {
                "description": "DEIM/DINO face-keypoint dataset exported from canonical CVAT metadata",
                "source_format": "canonical_cvat_metadata",
                "keypoint_order": list(TARGET_KEYPOINT_NAMES),
                "flip_indices": list(FLIP_INDICES),
                "visibility_policy": {
                    "0": "missing or outside image",
                    "1": "annotated but occluded",
                    "2": "annotated and visible",
                },
                "ellipse_schema_version": 1,
                "ellipse_format": "[cx, cy, semi_major, semi_minor, theta_rad]",
                "ellipse_angle_convention": "major-axis angle in [-pi/2, pi/2), modulo pi",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 2,
                    "name": "face",
                    "supercategory": "person",
                    "keypoints": list(TARGET_KEYPOINT_NAMES),
                    "skeleton": list(SKELETON),
                }
            ],
        }
        path = destination / f"face_keypoints_{split}.json"
        write_json_atomic(path, output)
        summaries[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "usable_keypoint_instances": sum(
                int(annotation["num_keypoints"] > 0) for annotation in annotations
            ),
            "face_occluded_instances": face_occluded,
            "keypoint_occluded_instances": keypoint_occluded_instances,
            "ellipses": ellipse_count,
            "visibility_counts": {
                str(value): visibility_counts[value] for value in (0, 1, 2)
            },
            "sha256": sha256_file(path),
        }
    summary = {
        "schema_version": 1,
        "status": "ok",
        "exporter": "deim_dino",
        "source": str(source.metadata_path),
        "output_root": str(destination),
        "category_id": 2,
        "splits": summaries,
    }
    write_json_atomic(destination / "conversion_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = export_deim_dino(
        args.dataset_root,
        metadata=args.metadata,
        output_root=args.output_root,
    )
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
