#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.head_pose_geometry.heads import HeadDetection, serialize_head_cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert external COCO HEAD predictions to cache schema v1."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--detector-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--minimum-score", type=float, default=0.001)
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help=(
            "Validation-selected threshold JSON from "
            "select_head_codino_threshold.py. Overrides --minimum-score."
        ),
    )
    parser.add_argument("--maximum-per-image", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    minimum_score = float(args.minimum_score)
    threshold_metadata: dict[str, Any] | None = None
    if args.threshold_file is not None:
        threshold_metadata = json.loads(
            args.threshold_file.read_text(encoding="utf-8")
        )
        if str(threshold_metadata.get("selection_split")) != "val":
            raise ValueError("HEAD threshold must be selected on validation")
        minimum_score = float(threshold_metadata["selected_threshold"])
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    images: dict[int, dict[str, Any]] = {
        int(row["id"]): row for row in annotations["images"]
    }
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    grouped: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for order, row in enumerate(predictions):
        if (
            int(row.get("category_id", -1)) != args.category_id
            or float(row.get("score", 0.0)) < minimum_score
        ):
            continue
        image_id = int(row["image_id"])
        if image_id not in images:
            raise ValueError(f"Prediction references unknown image_id={image_id}")
        grouped[image_id].append((-float(row["score"]), order, row))
    entries: dict[str, list[HeadDetection]] = {}
    for image_id, image in sorted(images.items()):
        detections = []
        for _, order, row in sorted(grouped.get(image_id, ()))[
            : args.maximum_per_image
        ]:
            x, y, width, height = (float(value) for value in row["bbox"])
            detections.append(
                HeadDetection(
                    box_xyxy=(x, y, x + width, y + height),
                    confidence=float(row["score"]),
                    instance_id=order,
                    valid_image_region_xyxy=(
                        0.0,
                        0.0,
                        float(image["width"]),
                        float(image["height"]),
                    ),
                    source=args.detector_name,
                ).validated((int(image["height"]), int(image["width"])))
            )
        entries[str(image_id)] = detections
    payload = serialize_head_cache(
        entries,
        metadata={
            "split": args.split,
            "detector_name": args.detector_name,
            "detector_checkpoint": str(args.checkpoint),
            "detector_checkpoint_sha256": sha256_file(args.checkpoint),
            "annotation": str(args.annotations),
            "annotation_sha256": sha256_file(args.annotations),
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "minimum_score": minimum_score,
            "threshold_file": (
                str(args.threshold_file)
                if args.threshold_file is not None
                else None
            ),
            "threshold_file_sha256": (
                sha256_file(args.threshold_file)
                if args.threshold_file is not None
                else None
            ),
            "threshold_selection": threshold_metadata,
            "maximum_per_image": args.maximum_per_image,
            "coordinate_system": "original_image_xyxy_pixels",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "images": len(entries),
                "detections": sum(len(rows) for rows in entries.values()),
                "sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
