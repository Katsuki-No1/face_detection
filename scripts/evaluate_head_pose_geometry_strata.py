#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.head_pose_geometry.labels import ellipse_visible_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify cooperative real FACE/keypoint evaluation rows."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-subsets", type=Path)
    return parser.parse_args()


def xywh_to_xyxy(values: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in values)
    return np.asarray((x, y, x + width, y + height), dtype=np.float64)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    upper = np.maximum(left[:2], right[:2])
    lower = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(0.0, lower - upper)))
    areas = [
        float(np.prod(np.maximum(0.0, box[2:] - box[:2]))) for box in (left, right)
    ]
    return intersection / max(areas[0] + areas[1] - intersection, 1.0e-12)


def keypoints_xyv(row: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(row.get("keypoints", ()), dtype=np.float64)
    if raw.size == 10:
        states = np.asarray(row.get("keypoint_states", (0,) * 5))
        return np.concatenate((raw.reshape(5, 2), states.reshape(5, 1)), axis=1)
    return raw.reshape(5, 3)


def bin_name(value: float, edges: tuple[float, ...]) -> str:
    for lower, upper in pairwise(edges):
        if lower <= value < upper:
            return f"[{lower:.2g},{upper:.2g})"
    return f"[{edges[-2]:.2g},{edges[-1]:.2g}]"


def base_strata(
    row: dict[str, Any],
    image: dict[str, Any],
) -> list[str]:
    width, height = int(image["width"]), int(image["height"])
    visible = ellipse_visible_geometry(
        row["ellipse"],
        width=width,
        height=height,
    )
    states = keypoints_xyv(row)[:, 2]
    visible_count = int((states > 1).sum())
    head_box = xywh_to_xyxy(row.get("head_bbox", row["bbox"]))
    area_fraction = (
        max(0.0, head_box[2] - head_box[0])
        * max(0.0, head_box[3] - head_box[1])
        / max(1.0, width * height)
    )
    size = (
        "small"
        if area_fraction < 0.01
        else ("medium" if area_fraction < 0.05 else "large")
    )
    truncation = [name for name, enabled in visible["truncation"].items() if enabled]
    return [
        "overall",
        "visible_ratio/"
        + bin_name(
            float(visible["visible_area_ratio"]),
            (0.0, 0.25, 0.5, 0.75, 1.000001),
        ),
        "truncation/"
        + (
            "none"
            if not truncation
            else ("multiple" if len(truncation) > 1 else truncation[0])
        ),
        "visible_keypoints/"
        + (
            "0"
            if visible_count == 0
            else (
                "1-2" if visible_count <= 2 else ("3-4" if visible_count <= 4 else "5")
            )
        ),
        f"head_size/{size}",
    ]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = defaultdict(list)
    count = len(records)
    for record in records:
        for key, value in record.items():
            if key in {"strata"} or value is None:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[key].append(float(value))
    return {
        "instances": count,
        **{
            key: {
                "mean": float(np.mean(items)),
                "median": float(np.median(items)),
                "count": len(items),
            }
            for key, items in sorted(values.items())
        },
    }


def main() -> int:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in coco["annotations"]:
        if (
            bool(row.get("attribute_valid", "ellipse" in row))
            and len(row.get("ellipse", ())) == 5
        ):
            annotations[int(row["image_id"])].append(row)
    for rows in annotations.values():
        rows.sort(key=lambda row: int(row.get("id", 0)))
    manual = (
        json.loads(args.manual_subsets.read_text(encoding="utf-8"))
        if args.manual_subsets is not None
        else {}
    )
    records: list[dict[str, Any]] = []
    with args.predictions_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            evaluation = json.loads(line)
            image_id = int(evaluation["image_id"])
            prediction = evaluation["predictions"]
            face = prediction["refined_face"]
            pose = prediction["refined_pose"]
            face_match = {
                int(key): int(value)
                for key, value in evaluation["matches"]["refined_face"][
                    "matched"
                ].items()
            }
            pose_match = {
                int(key): int(value)
                for key, value in evaluation["matches"]["refined_pose"][
                    "matched"
                ].items()
            }
            geometry = prediction.get("head_pose_geometry", {})
            geometry_boxes = [
                np.asarray(box, dtype=np.float64)
                for box in geometry.get("head_boxes", ())
            ]
            for index, annotation in enumerate(annotations.get(image_id, ())):
                ellipse = np.asarray(annotation["ellipse"], dtype=np.float64)
                head = xywh_to_xyxy(annotation.get("head_bbox", annotation["bbox"]))
                diagonal = max(
                    1.0,
                    math.hypot(head[2] - head[0], head[3] - head[1]),
                )
                record: dict[str, Any] = {
                    "face_matched": float(index in face_match),
                    "pose_matched": float(index in pose_match),
                    "strata": base_strata(annotation, images[image_id]),
                }
                manual_key = f"{image_id}:{annotation.get('id', index)}"
                record["strata"].extend(
                    f"manual/{name}" for name in manual.get(manual_key, ())
                )
                if index in face_match:
                    prediction_index = face_match[index]
                    predicted = np.asarray(
                        face["ellipses"][prediction_index],
                        dtype=np.float64,
                    )
                    record["ellipse_center_error_head_diag"] = float(
                        np.linalg.norm(predicted[:2] - ellipse[:2]) / diagonal
                    )
                    record["ellipse_log_axis_error"] = float(
                        np.abs(
                            np.log(
                                np.maximum(predicted[2:4], 1.0e-8)
                                / np.maximum(ellipse[2:4], 1.0e-8)
                            )
                        ).mean()
                    )
                    record["ellipse_periodic_angle_error"] = float(
                        1.0 - math.cos(2.0 * (predicted[4] - ellipse[4]))
                    )
                    record["direct_uncertainty"] = 1.0 - float(
                        face["scores"][prediction_index]
                    )
                if index in pose_match:
                    prediction_index = pose_match[index]
                    predicted_points = np.asarray(
                        pose["keypoints"][prediction_index],
                        dtype=np.float64,
                    )
                    points = keypoints_xyv(annotation)
                    labeled = points[:, 2] > 0
                    if labeled.any():
                        record["keypoint_nme_head_diag"] = float(
                            np.linalg.norm(
                                predicted_points[labeled] - points[labeled, :2],
                                axis=-1,
                            ).mean()
                            / diagonal
                        )
                if geometry_boxes:
                    geometry_index = max(
                        range(len(geometry_boxes)),
                        key=lambda value: iou(head, geometry_boxes[value]),
                    )
                    uncertainties = geometry.get("geometry_uncertainty", ())
                    if geometry_index < len(uncertainties):
                        record["geometry_uncertainty"] = float(
                            uncertainties[geometry_index]
                        )
                records.append(record)

    # Dataset-derived quartiles keep uncertainty strata comparable within one
    # frozen evaluation run and do not tune a test threshold.
    for field in ("direct_uncertainty", "geometry_uncertainty"):
        available = [float(record[field]) for record in records if field in record]
        if not available:
            continue
        edges = np.quantile(available, (0.0, 0.25, 0.5, 0.75, 1.0))
        edges[-1] += 1.0e-12
        for record in records:
            if field in record:
                quartile = (
                    int(np.searchsorted(edges[1:], record[field], side="right")) + 1
                )
                record["strata"].append(f"{field}/Q{min(quartile, 4)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for stratum in record["strata"]:
            grouped[stratum].append(record)
    output = {
        "schema_version": 1,
        "annotations": str(args.annotations),
        "predictions": str(args.predictions_jsonl),
        "strata": {name: summarize(rows) for name, rows in sorted(grouped.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
