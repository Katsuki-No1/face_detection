from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RTMPose five-point predictions on GT face boxes or on "
            "RTMDet boxes from eval_rtmdet_with_dinov3_deimv2_metrics.py."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-predictions", type=Path, default=None)
    parser.add_argument("--detector-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1e-9)


def prediction_keypoints(sample: Any) -> tuple[list[list[float]], list[float]]:
    instances = sample.pred_instances
    if not len(instances.keypoints):
        return [], []
    points_value = instances.keypoints[0]
    if hasattr(points_value, "detach"):
        points_value = points_value.detach().cpu()
    points = points_value.tolist()
    if hasattr(instances, "keypoint_scores") and len(instances.keypoint_scores):
        scores_value = instances.keypoint_scores[0]
        if hasattr(scores_value, "detach"):
            scores_value = scores_value.detach().cpu()
        scores = scores_value.tolist()
    else:
        scores = [1.0] * len(points)
    return points, [float(value) for value in scores]


def select_input_boxes(
    image_id: int,
    gt_annotations: list[dict[str, Any]],
    detector_rows: dict[int, list[dict[str, Any]]] | None,
    detector_threshold: float | None,
) -> list[dict[str, Any]]:
    if detector_rows is None:
        return [
            {
                "bbox_xyxy": xywh_to_xyxy(annotation["bbox"]),
                "gt_annotation_id": int(annotation["id"]),
                "detector_score": None,
            }
            for annotation in gt_annotations
        ]
    threshold = float(detector_threshold or 0.0)
    return [
        {
            "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
            "gt_annotation_id": None,
            "detector_score": float(row["score"]),
        }
        for row in detector_rows.get(image_id, [])
        if float(row["score"]) >= threshold
    ]


def match_predictions(
    gt_annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    direct_gt_boxes: bool,
    iou_threshold: float,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    if direct_gt_boxes:
        by_annotation = {
            int(row["gt_annotation_id"]): row
            for row in predictions
            if row.get("gt_annotation_id") is not None
        }
        return [
            (annotation, by_annotation.get(int(annotation["id"])))
            for annotation in gt_annotations
        ]

    matched: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for annotation in gt_annotations:
        gt_box = xywh_to_xyxy(annotation["bbox"])
        best_index = -1
        best_iou = 0.0
        for index, prediction in enumerate(predictions):
            if index in matched:
                continue
            overlap = iou(gt_box, prediction["bbox_xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            pairs.append((annotation, predictions[best_index]))
        else:
            pairs.append((annotation, None))
    return pairs


def calculate_metrics(
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    predicted_instances: int,
) -> dict[str, Any]:
    normalized_distances: list[float] = []
    per_keypoint: dict[str, list[float]] = {name: [] for name in KEYPOINT_NAMES}
    pck_counts: Counter[str] = Counter()
    missed = 0
    evaluated = 0
    gt_visible_keypoints = 0
    for annotation, prediction in pairs:
        gt_keypoints = annotation["keypoints"]
        visible_keypoints = sum(
            int(gt_keypoints[index * 3 + 2]) > 0
            for index in range(len(KEYPOINT_NAMES))
        )
        gt_visible_keypoints += visible_keypoints
        if prediction is None or len(prediction.get("keypoints", [])) < len(KEYPOINT_NAMES):
            missed += 1
            continue
        evaluated += 1
        _x, _y, width, height = [float(value) for value in annotation["bbox"]]
        diagonal = math.hypot(width, height)
        if diagonal <= 0:
            continue
        for index, name in enumerate(KEYPOINT_NAMES):
            visibility = int(gt_keypoints[index * 3 + 2])
            if visibility <= 0:
                continue
            pred_x, pred_y = prediction["keypoints"][index][:2]
            gt_x = float(gt_keypoints[index * 3])
            gt_y = float(gt_keypoints[index * 3 + 1])
            normalized = math.hypot(float(pred_x) - gt_x, float(pred_y) - gt_y) / diagonal
            normalized_distances.append(normalized)
            per_keypoint[name].append(normalized)
            for threshold in (0.03, 0.05, 0.10):
                if normalized <= threshold:
                    pck_counts[str(threshold)] += 1

    denominator = len(normalized_distances)
    return {
        "normalization": "face_bbox_diagonal",
        "gt_instances": len(pairs),
        "predicted_instances": predicted_instances,
        "evaluated_instances": evaluated,
        "missed_instances": missed,
        "gt_visible_keypoints": gt_visible_keypoints,
        "evaluated_keypoints": denominator,
        "keypoint_coverage": (
            denominator / gt_visible_keypoints if gt_visible_keypoints else None
        ),
        "NME": sum(normalized_distances) / denominator if denominator else None,
        "PCK@0.03": pck_counts["0.03"] / denominator if denominator else None,
        "PCK@0.05": pck_counts["0.05"] / denominator if denominator else None,
        "PCK@0.10": pck_counts["0.1"] / denominator if denominator else None,
        "end_to_end_PCK@0.03": (
            pck_counts["0.03"] / gt_visible_keypoints
            if gt_visible_keypoints
            else None
        ),
        "end_to_end_PCK@0.05": (
            pck_counts["0.05"] / gt_visible_keypoints
            if gt_visible_keypoints
            else None
        ),
        "end_to_end_PCK@0.10": (
            pck_counts["0.1"] / gt_visible_keypoints
            if gt_visible_keypoints
            else None
        ),
        "keypoint_wise_NME": {
            name: (sum(values) / len(values) if values else None)
            for name, values in per_keypoint.items()
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coco = json.loads(args.annotation.read_text(encoding="utf-8"))
    images = list(coco.get("images", []))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    detector_rows: dict[int, list[dict[str, Any]]] | None = None
    if args.detector_predictions is not None:
        rows = json.loads(args.detector_predictions.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Detector predictions must be the flat JSON list from the RTMDet evaluator")
        detector_rows = defaultdict(list)
        for row in rows:
            detector_rows[int(row["image_id"])].append(row)
        if args.detector_threshold is None:
            raise ValueError("--detector-threshold is required with --detector-predictions")

    from mmpose.apis import inference_topdown, init_model  # type: ignore

    model = init_model(
        args.config.as_posix(), args.checkpoint.as_posix(), device=args.device
    )
    all_predictions: list[dict[str, Any]] = []
    all_pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    started = time.perf_counter()
    for image in images:
        image_id = int(image["id"])
        gt_annotations = annotations_by_image.get(image_id, [])
        input_rows = select_input_boxes(
            image_id,
            gt_annotations,
            detector_rows,
            args.detector_threshold,
        )
        boxes = [row["bbox_xyxy"] for row in input_rows]
        image_path = args.image_root / str(image["file_name"])
        samples = (
            inference_topdown(model, image_path.as_posix(), bboxes=boxes)
            if boxes
            else []
        )
        image_predictions: list[dict[str, Any]] = []
        for input_row, sample in zip(input_rows, samples):
            keypoints, scores = prediction_keypoints(sample)
            row = {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "bbox_xyxy": input_row["bbox_xyxy"],
                "gt_annotation_id": input_row["gt_annotation_id"],
                "detector_score": input_row["detector_score"],
                "keypoints": keypoints,
                "keypoint_scores": scores,
            }
            image_predictions.append(row)
            all_predictions.append(row)
        all_pairs.extend(
            match_predictions(
                gt_annotations,
                image_predictions,
                direct_gt_boxes=detector_rows is None,
                iou_threshold=args.iou_threshold,
            )
        )
    elapsed = time.perf_counter() - started

    metrics = calculate_metrics(all_pairs, predicted_instances=len(all_predictions))
    metrics.update(
        {
            "mode": "gt_bbox" if detector_rows is None else "rtmdet_two_stage",
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "annotation": str(args.annotation),
            "num_images": len(images),
            "detector_predictions": (
                str(args.detector_predictions) if args.detector_predictions else None
            ),
            "detector_threshold": args.detector_threshold,
            "matching_iou_threshold": args.iou_threshold,
            "inference_elapsed_sec": elapsed,
            "inference_images_per_sec": len(images) / elapsed if elapsed > 0 else 0.0,
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "predictions.json").write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
