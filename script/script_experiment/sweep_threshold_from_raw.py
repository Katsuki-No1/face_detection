import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from evaluate_face_detections import bbox_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize candidate counts from merged_candidates.jsonl across score thresholds."
    )
    parser.add_argument("--raw", type=Path, required=True, help="merged_candidates.jsonl from dump_detector_stages.py.")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path.")
    parser.add_argument(
        "--thresholds",
        default="0.01,0.05,0.10,0.15,0.20,0.30,0.50",
        help="Comma-separated score thresholds. Default: 0.01,0.05,0.10,0.15,0.20,0.30,0.50.",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.4,
        help="IoU threshold for duplicate candidate estimation. Default: 0.4.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return records


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("All thresholds must be between 0 and 1.")
    return thresholds


def bbox_from_detection(detection: dict[str, Any]) -> dict[str, float]:
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    return {"xtl": float(x1), "ytl": float(y1), "xbr": float(x2), "ybr": float(y2)}


def duplicate_count(detections: list[dict[str, Any]], threshold: float) -> int:
    count = 0
    for first_index, first in enumerate(detections):
        for second in detections[first_index + 1 :]:
            if bbox_iou(bbox_from_detection(first), bbox_from_detection(second)) >= threshold:
                count += 1
    return count


def build_rows(records: list[dict[str, Any]], thresholds: list[float], duplicate_iou: float) -> list[dict[str, Any]]:
    rows = []
    image_count = len(records)
    all_candidates = sum(len(record.get("detections", [])) for record in records)
    for threshold in thresholds:
        per_image_counts = []
        duplicate_candidates = 0
        for record in records:
            detections = [
                detection
                for detection in record.get("detections", [])
                if detection.get("score") is not None and float(detection["score"]) >= threshold
            ]
            per_image_counts.append(len(detections))
            duplicate_candidates += duplicate_count(detections, duplicate_iou)
        final_count = sum(per_image_counts)
        rows.append(
            {
                "threshold": threshold,
                "total_candidates": all_candidates,
                "final_candidates_after_threshold": final_count,
                "avg_candidates_per_image": final_count / image_count if image_count else 0.0,
                "images_with_no_detection": sum(1 for count in per_image_counts if count == 0),
                "duplicate_candidates_estimated": duplicate_candidates,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not args.raw.is_file():
        raise FileNotFoundError(f"Raw JSONL not found: {args.raw}")
    if not 0.0 <= args.duplicate_iou <= 1.0:
        raise ValueError("--duplicate-iou must be between 0 and 1.")

    records = load_jsonl(args.raw)
    thresholds = parse_thresholds(args.thresholds)
    rows = build_rows(records, thresholds, args.duplicate_iou)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "threshold",
            "total_candidates",
            "final_candidates_after_threshold",
            "avg_candidates_per_image",
            "images_with_no_detection",
            "duplicate_candidates_estimated",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
