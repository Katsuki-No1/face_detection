"""One-to-one FACE bbox evaluation with explicit duplicate accounting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "face-bbox-iou50-duplication-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ix1 = max(float(first[0]), float(second[0]))
    iy1 = max(float(first[1]), float(second[1]))
    ix2 = min(float(first[2]), float(second[2]))
    iy2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(first) + _area(second) - intersection
    return intersection / max(union, 1.0e-12)


def load_face_ground_truth(
    annotation_path: Path,
    *,
    category_ids: Iterable[int] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    allowed = None if category_ids is None else {int(value) for value in category_ids}
    images = {int(row["id"]): [] for row in coco.get("images", [])}
    for annotation in coco.get("annotations", []):
        if int(annotation.get("iscrowd", 0)):
            continue
        category_id = int(annotation.get("category_id", -1))
        if allowed is not None and category_id not in allowed:
            continue
        images.setdefault(int(annotation["image_id"]), []).append(
            {
                "annotation_id": int(annotation.get("id", -1)),
                "category_id": category_id,
                "bbox_xyxy": _xywh_to_xyxy(annotation["bbox"]),
            }
        )
    return images, len(coco.get("images", []))


def _confidence_distribution(values: Sequence[float]) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(array, bins=edges)
    if array.size == 0:
        summary = {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    else:
        quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
        summary = {
            "count": int(array.size),
            "min": float(array.min()),
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }
    return {
        **summary,
        "histogram": {
            "bin_edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
        },
    }


def threshold_metrics(
    ground_truth_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float = 0.50,
    num_images: int | None = None,
) -> dict[str, Any]:
    """Match predictions by descending confidence and classify duplicate FPs.

    A prediction is a duplicate FP when it cannot claim an unmatched GT but has
    IoU at least ``iou_threshold`` with a GT already claimed by a higher-score
    prediction. This keeps duplicate accounting consistent with the primary
    one-to-one detection metric.
    """

    selected = [
        dict(row)
        for row in predictions
        if float(row.get("score", 0.0)) >= float(score_threshold)
    ]
    selected.sort(
        key=lambda row: (
            -float(row.get("score", 0.0)),
            int(row.get("image_id", -1)),
        )
    )
    matched: dict[int, set[int]] = defaultdict(set)
    duplicated_gt: set[tuple[int, int]] = set()
    duplicate_images: set[int] = set()
    duplicate_confidences: list[float] = []
    tp = 0
    fp = 0
    for prediction in selected:
        image_id = int(prediction["image_id"])
        gt_rows = ground_truth_by_image.get(image_id, ())
        overlaps = [
            bbox_iou(prediction["bbox_xyxy"], gt["bbox_xyxy"])
            for gt in gt_rows
        ]
        available = [
            (overlap, index)
            for index, overlap in enumerate(overlaps)
            if index not in matched[image_id] and overlap >= iou_threshold
        ]
        if available:
            _overlap, gt_index = max(available)
            matched[image_id].add(gt_index)
            tp += 1
            continue

        fp += 1
        claimed = [
            (overlap, index)
            for index, overlap in enumerate(overlaps)
            if index in matched[image_id] and overlap >= iou_threshold
        ]
        if claimed:
            _overlap, gt_index = max(claimed)
            duplicated_gt.add((image_id, gt_index))
            duplicate_images.add(image_id)
            duplicate_confidences.append(float(prediction["score"]))

    total_gt = sum(len(rows) for rows in ground_truth_by_image.values())
    total_images = int(num_images) if num_images is not None else len(ground_truth_by_image)
    fn = max(0, total_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    duplicate_count = len(duplicate_confidences)
    prediction_count = len(selected)
    return {
        "threshold": float(score_threshold),
        "matching_iou": float(iou_threshold),
        "prediction_count": prediction_count,
        "gt_count": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "duplicated_gt_count": len(duplicated_gt),
        "duplicate_extra_prediction_count": duplicate_count,
        "images_with_duplicates": len(duplicate_images),
        "duplicate_prediction_rate": (
            duplicate_count / prediction_count if prediction_count else 0.0
        ),
        "duplicated_gt_rate": len(duplicated_gt) / total_gt if total_gt else 0.0,
        "duplicate_image_rate": (
            len(duplicate_images) / total_images if total_images else 0.0
        ),
        "duplicate_free_image_rate": (
            (total_images - len(duplicate_images)) / total_images
            if total_images
            else 0.0
        ),
        "duplicate_confidence_distribution": _confidence_distribution(
            duplicate_confidences
        ),
        "duplicate_confidences": duplicate_confidences,
    }


def select_validation_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    strict_precision_minimum: float = 0.90,
) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in rows
        if float(row["precision"]) > float(strict_precision_minimum)
    ]
    if not candidates:
        raise RuntimeError(
            "No validation threshold satisfies strict bbox precision "
            f"> {strict_precision_minimum:.6f}"
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            -float(row["duplicate_prediction_rate"]),
            float(row["threshold"]),
        ),
    )


def _public_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "duplicate_confidences"}


def write_evaluation(
    *,
    annotation_path: Path,
    predictions_path: Path,
    output_dir: Path,
    split: str,
    alpha: float,
    category_ids: Sequence[int],
    common_threshold: float = 0.33,
    threshold_floor_exclusive: float = 0.33,
    strict_precision_minimum: float = 0.90,
    iou_threshold: float = 0.50,
    validation_metrics_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth, num_images = load_face_ground_truth(
        annotation_path, category_ids=category_ids
    )
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    allowed = {int(value) for value in category_ids}
    predictions = [
        row
        for row in predictions
        if int(row.get("category_id", -1)) in allowed
    ]
    thresholds = [
        round(float(value), 2)
        for value in np.arange(0.01, 0.96, 0.01)
        if float(value) > float(threshold_floor_exclusive) + 1.0e-9
    ]
    threshold_rows = [
        threshold_metrics(
            ground_truth,
            predictions,
            score_threshold=threshold,
            iou_threshold=iou_threshold,
            num_images=num_images,
        )
        for threshold in thresholds
    ]
    common = threshold_metrics(
        ground_truth,
        predictions,
        score_threshold=common_threshold,
        iou_threshold=iou_threshold,
        num_images=num_images,
    )
    if split == "val":
        selected = select_validation_threshold(
            threshold_rows,
            strict_precision_minimum=strict_precision_minimum,
        )
        threshold_source = "selected_on_validation"
    else:
        if validation_metrics_path is None:
            raise ValueError("test evaluation requires validation_metrics_path")
        validation = json.loads(validation_metrics_path.read_text(encoding="utf-8"))
        selected_threshold = float(validation["validation_selected"]["threshold"])
        selected = threshold_metrics(
            ground_truth,
            predictions,
            score_threshold=selected_threshold,
            iou_threshold=iou_threshold,
            num_images=num_images,
        )
        threshold_source = str(validation_metrics_path.resolve())

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "mal_alpha": float(alpha),
        "annotation": str(annotation_path.resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": _sha256(predictions_path),
        "category_ids": [int(value) for value in category_ids],
        "matching": {
            "type": "score_descending_one_to_one_bbox",
            "iou_threshold": float(iou_threshold),
            "unmatched_predictions": "false_positive",
            "duplicate_definition": (
                "unmatched prediction with bbox IoU at least threshold to a GT "
                "already claimed by a higher-confidence prediction"
            ),
        },
        "threshold_selection": {
            "population": "GT FACE bboxes",
            "candidate_floor_exclusive": float(threshold_floor_exclusive),
            "candidate_step": 0.01,
            "criterion": (
                "maximize bbox-IoU-0.50 recall subject to strict precision > "
                f"{strict_precision_minimum:.2f}; ties: precision, lower "
                "duplicate prediction rate, higher threshold"
            ),
            "source": threshold_source,
        },
        "common_0_33": _public_metrics(common),
        "validation_selected": _public_metrics(selected),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "threshold_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "prediction_count",
            "duplicated_gt_count",
            "duplicate_extra_prediction_count",
            "duplicate_prediction_rate",
            "duplicated_gt_rate",
            "images_with_duplicates",
            "duplicate_image_rate",
            "duplicate_free_image_rate",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in threshold_rows)
    duplicate_payload = {
        "common_0_33": {
            "values": common["duplicate_confidences"],
            "distribution": common["duplicate_confidence_distribution"],
        },
        "validation_selected": {
            "threshold": selected["threshold"],
            "values": selected["duplicate_confidences"],
            "distribution": selected["duplicate_confidence_distribution"],
        },
    }
    (output_dir / "duplicate_confidences.json").write_text(
        json.dumps(duplicate_payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    success = {
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "mal_alpha": float(alpha),
        "inventory_verified": num_images > 0 and sum(map(len, ground_truth.values())) > 0,
        "metrics": str((output_dir / "metrics.json").resolve()),
    }
    (output_dir / "success.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics

