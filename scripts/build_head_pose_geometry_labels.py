#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.head_pose_geometry.crops import HeadCropBuilder
from src.head_pose_geometry.heads import HeadDetection
from src.head_pose_geometry.labels import (
    DERIVED_LABEL_SCHEMA_VERSION,
    derive_instance_labels,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive deterministic 2D geometry labels from face_dino_v1."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-instances", type=int)
    return parser.parse_args()


def image_path(root: Path, file_name: str) -> Path:
    value = Path(file_name)
    for candidate in (root / value, root / "images" / value):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(file_name)


def audit_split_leakage(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    videos = {
        split: {
            str(image.get("source_video_id", ""))
            for image in payload["images"]
            if image.get("source_video_id")
        }
        for split, payload in payloads.items()
    }
    overlap: dict[str, list[str]] = {}
    names = sorted(videos)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            common = sorted(videos[first] & videos[second])
            overlap[f"{first}__{second}"] = common
            if common:
                raise ValueError(
                    f"Source-video leakage between {first} and {second}: {common[:10]}"
                )
    return {
        "source_video_counts": {split: len(values) for split, values in videos.items()},
        "overlap": overlap,
    }


def main() -> int:
    args = parse_args()
    payloads = {
        split: json.loads(
            (args.annotation_dir / f"{split}.json").read_text(encoding="utf-8")
        )
        for split in args.splits
    }
    leakage = audit_split_leakage(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = HeadCropBuilder(output_size=args.crop_size)
    summary: dict[str, Any] = {"leakage_audit": leakage, "splits": {}}
    for split, payload in payloads.items():
        images = {int(row["id"]): row for row in payload["images"]}
        records = []
        statistics: defaultdict[str, int] = defaultdict(int)
        rows = sorted(payload["annotations"], key=lambda item: int(item["id"]))
        if args.max_instances is not None:
            rows = rows[: args.max_instances]
        for row in rows:
            if int(row.get("category_id", -1)) != 1 or not bool(
                row.get("attribute_valid", False)
            ):
                statistics["skipped_without_face_attributes"] += 1
                continue
            image = images[int(row["image_id"])]
            height, width = int(image["height"]), int(image["width"])
            x, y, box_width, box_height = (float(value) for value in row["bbox"])
            detection = HeadDetection(
                box_xyxy=(x, y, x + box_width, y + box_height),
                confidence=1.0,
                instance_id=int(row["id"]),
                source="ground_truth",
            )
            # RGB values do not affect the valid mask; avoid decoding thousands
            # of images during label conversion.
            canvas = torch.zeros((3, 1, 1), dtype=torch.float32).expand(
                3,
                height,
                width,
            )
            crop = builder.build(canvas, [detection], training=False)
            points = [
                [float(point[0]), float(point[1]), int(state)]
                for point, state in zip(
                    row["keypoints"],
                    row["keypoint_states"],
                    strict=True,
                )
            ]
            derived = derive_instance_labels(
                ellipse=row["ellipse"],
                head_box_xyxy=detection.box_xyxy,
                image_size=(height, width),
                keypoints_xyv=points,
                valid_mask=crop.valid_mask[0, 0].bool(),
            )
            records.append(
                {
                    "image_id": int(row["image_id"]),
                    "annotation_id": int(row["id"]),
                    "source_video_id": str(image.get("source_video_id", "")),
                    "source_head_index": int(row.get("source_head_index", -1)),
                    **derived,
                }
            )
            statistics["instances"] += 1
            statistics["truncated"] += int(any(derived["truncation"].values()))
            statistics["zero_visible_keypoints"] += int(
                derived["number_visible_keypoints"] == 0
            )
        source = args.annotation_dir / f"{split}.json"
        output = {
            "schema_version": DERIVED_LABEL_SCHEMA_VERSION,
            "split": split,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "crop": {
                "output_size": args.crop_size,
                "context_scale": builder.context_scale,
            },
            "statistics": dict(statistics),
            "records": records,
        }
        destination = args.output_dir / f"{split}.json"
        destination.write_text(
            json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["splits"][split] = {
            **dict(statistics),
            "output": str(destination),
            "sha256": sha256_file(destination),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
