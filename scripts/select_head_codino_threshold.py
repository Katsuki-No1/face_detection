#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(values: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in values)
    return np.asarray((x, y, x + width, y + height), dtype=np.float64)


def iou_matrix(
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
) -> np.ndarray:
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)
    left = np.stack(ground_truth)
    right = np.stack(predictions)
    upper = np.maximum(left[:, None, :2], right[None, :, :2])
    lower = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.maximum(0.0, lower - upper).prod(axis=-1)
    left_area = np.maximum(0.0, left[:, 2:] - left[:, :2]).prod(axis=-1)
    right_area = np.maximum(0.0, right[:, 2:] - right[:, :2]).prod(axis=-1)
    return intersection / np.maximum(
        left_area[:, None] + right_area[None, :] - intersection,
        1.0e-12,
    )


def metrics_at_threshold(
    ground_truth: dict[int, list[np.ndarray]],
    predictions: dict[int, list[tuple[float, np.ndarray]]],
    *,
    threshold: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    image_ids = sorted(set(ground_truth) | set(predictions))
    for image_id in image_ids:
        gt = ground_truth.get(image_id, [])
        predicted = [
            box
            for score, box in predictions.get(image_id, [])
            if score >= threshold
        ]
        overlaps = iou_matrix(gt, predicted)
        matched = 0
        if overlaps.size:
            gt_indices, prediction_indices = linear_sum_assignment(1.0 - overlaps)
            matched = sum(
                overlaps[gt_index, prediction_index] >= iou_threshold
                for gt_index, prediction_index in zip(
                    gt_indices,
                    prediction_indices,
                    strict=True,
                )
            )
        true_positive += int(matched)
        false_positive += len(predicted) - int(matched)
        false_negative += len(gt) - int(matched)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Co-DINO HEAD threshold on validation only. The default "
            "maximizes recall subject to a precision floor."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations: dict[str, Any] = json.loads(
        args.annotations.read_text(encoding="utf-8")
    )
    raw_predictions: list[dict[str, Any]] = json.loads(
        args.predictions.read_text(encoding="utf-8")
    )
    ground_truth: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in annotations.get("annotations", []):
        if int(row.get("category_id", -1)) == args.category_id:
            ground_truth[int(row["image_id"])].append(xywh_to_xyxy(row["bbox"]))
    predictions: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
    scores = {0.0, 1.0}
    for row in raw_predictions:
        if int(row.get("category_id", -1)) != args.category_id:
            continue
        score = float(row["score"])
        predictions[int(row["image_id"])].append(
            (score, xywh_to_xyxy(row["bbox"]))
        )
        scores.add(score)
    rows = [
        metrics_at_threshold(
            ground_truth,
            predictions,
            threshold=threshold,
            iou_threshold=float(args.iou_threshold),
        )
        for threshold in sorted(scores)
    ]
    eligible = [
        row
        for row in rows
        if float(row["precision"]) >= float(args.minimum_precision)
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["recall"]),
                float(row["precision"]),
                -float(row["threshold"]),
            ),
        )
        rule = "maximum_recall_subject_to_precision_floor"
    else:
        selected = max(
            rows,
            key=lambda row: (
                float(row["f1"]),
                float(row["recall"]),
                float(row["precision"]),
            ),
        )
        rule = "fallback_maximum_f1_no_threshold_met_precision_floor"
    payload = {
        "schema_version": 1,
        "selection_split": "val",
        "selection_rule": rule,
        "minimum_precision": float(args.minimum_precision),
        "iou_threshold": float(args.iou_threshold),
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "annotation": str(args.annotations),
        "annotation_sha256": sha256_file(args.annotations),
        "predictions": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
        "candidate_count": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
