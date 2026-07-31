from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


LANDMARK_NAMES = ("left_eye", "right_eye", "nose_tip", "left_mouth_corner", "right_mouth_corner")


def face_size_category(bbox_xyxy: Sequence[float]) -> str:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


class LandmarkMetricAccumulator:
    def __init__(
        self,
        landmark_names: Sequence[str] = LANDMARK_NAMES,
        *,
        visibility_threshold: float = 0.5,
    ) -> None:
        self.names = tuple(landmark_names)
        self.visibility_threshold = float(visibility_threshold)
        self.rows: list[dict[str, Any]] = []
        self.missed_instances = 0
        self.detector_false_positives = 0

    def add_missed(
        self,
        *,
        gt_keypoints: np.ndarray,
        labeled: np.ndarray,
        visible: np.ndarray,
        bbox_xyxy: Sequence[float],
        image_id: int,
        annotation_id: int,
        source_video_id: str = "",
        file_name: str = "",
    ) -> None:
        labeled_mask = np.asarray(labeled, dtype=bool)
        visible_mask = np.asarray(visible, dtype=bool)
        if bool((visible_mask & ~labeled_mask).any()):
            raise ValueError("visible keypoints must be a subset of labeled keypoints")
        self.missed_instances += 1
        self.rows.append(
            {
                "image_id": int(image_id),
                "annotation_id": int(annotation_id),
                "source_video_id": str(source_video_id),
                "file_name": str(file_name),
                "bbox_xyxy": [float(value) for value in bbox_xyxy],
                "face_size": face_size_category(bbox_xyxy),
                "labeled": labeled_mask.tolist(),
                "visible": visible_mask.tolist(),
                "gt_keypoints": np.asarray(gt_keypoints, dtype=np.float32).tolist(),
                "predicted_keypoints": None,
                "landmark_scores": None,
                "presence_scores": None,
                "state_probabilities": None,
                "normalized_errors": [None] * len(self.names),
                "detector_iou": None,
                "detector_bbox_xyxy": None,
                "detector_score": None,
                "missed": True,
            }
        )

    def add_prediction(
        self,
        *,
        predicted_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        labeled: np.ndarray,
        visible: np.ndarray,
        bbox_xyxy: Sequence[float],
        image_id: int,
        annotation_id: int,
        source_video_id: str = "",
        file_name: str = "",
        landmark_scores: np.ndarray | None = None,
        presence_scores: np.ndarray | None = None,
        state_probabilities: np.ndarray | None = None,
        detector_iou: float | None = None,
        detector_bbox_xyxy: Sequence[float] | None = None,
        detector_score: float | None = None,
    ) -> None:
        predicted = np.asarray(predicted_keypoints, dtype=np.float32).reshape(len(self.names), 2)
        gt = np.asarray(gt_keypoints, dtype=np.float32).reshape(len(self.names), 2)
        labeled_mask = np.asarray(labeled, dtype=bool).reshape(len(self.names))
        visible_mask = np.asarray(visible, dtype=bool).reshape(len(self.names))
        if bool((visible_mask & ~labeled_mask).any()):
            raise ValueError("visible keypoints must be a subset of labeled keypoints")
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        diagonal = max(1e-6, math.hypot(x2 - x1, y2 - y1))
        distances = np.linalg.norm(predicted - gt, axis=1) / diagonal
        normalized_errors: list[float | None] = [
            float(value) if labeled_mask[index] else None
            for index, value in enumerate(distances)
        ]
        self.rows.append(
            {
                "image_id": int(image_id),
                "annotation_id": int(annotation_id),
                "source_video_id": str(source_video_id),
                "file_name": str(file_name),
                "bbox_xyxy": [x1, y1, x2, y2],
                "face_size": face_size_category(bbox_xyxy),
                "labeled": labeled_mask.tolist(),
                "visible": visible_mask.tolist(),
                "gt_keypoints": gt.tolist(),
                "predicted_keypoints": predicted.tolist(),
                "landmark_scores": None if landmark_scores is None else np.asarray(landmark_scores).tolist(),
                "presence_scores": (
                    None
                    if presence_scores is None
                    else np.asarray(presence_scores).tolist()
                ),
                "state_probabilities": (
                    None
                    if state_probabilities is None
                    else np.asarray(state_probabilities).tolist()
                ),
                "normalized_errors": normalized_errors,
                "detector_iou": None if detector_iou is None else float(detector_iou),
                "detector_bbox_xyxy": (
                    None if detector_bbox_xyxy is None else [float(value) for value in detector_bbox_xyxy]
                ),
                "detector_score": None if detector_score is None else float(detector_score),
                "missed": False,
            }
        )

    def _summarize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labeled_total = sum(
            sum(bool(value) for value in row["labeled"])
            for row in rows
        )
        visible_total = sum(
            sum(bool(value) for value in row["visible"])
            for row in rows
        )
        evaluated_values: list[float] = []
        natural_visible_values: list[float] = []
        natural_occluded_values: list[float] = []
        per_landmark: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for index, value in enumerate(row["normalized_errors"]):
                if value is None:
                    continue
                evaluated_values.append(float(value))
                per_landmark[self.names[index]].append(float(value))
                if bool(row["visible"][index]):
                    natural_visible_values.append(float(value))
                else:
                    natural_occluded_values.append(float(value))
        evaluated = len(evaluated_values)
        result: dict[str, Any] = {
            "instances": len(rows),
            "missed_instances": sum(bool(row["missed"]) for row in rows),
            "labeled_keypoints": labeled_total,
            "visible_keypoints": visible_total,
            "occluded_keypoints": labeled_total - visible_total,
            "evaluated_keypoints": evaluated,
            "keypoint_coverage": evaluated / labeled_total if labeled_total else None,
            "nme": float(np.mean(evaluated_values)) if evaluated_values else None,
        }
        for threshold in (0.03, 0.05, 0.10):
            count = sum(value <= threshold for value in evaluated_values)
            result[f"pck_{threshold:.2f}"] = count / evaluated if evaluated else None
            result[f"end_to_end_pck_{threshold:.2f}"] = (
                count / labeled_total if labeled_total else None
            )
            for prefix, values in (
                ("natural_visible", natural_visible_values),
                ("natural_occluded", natural_occluded_values),
            ):
                result[f"{prefix}_pck_{threshold:.2f}"] = (
                    sum(value <= threshold for value in values) / len(values)
                    if values
                    else None
                )
        result.update(
            {
                "natural_visible_points": len(natural_visible_values),
                "natural_visible_nme": (
                    float(np.mean(natural_visible_values))
                    if natural_visible_values
                    else None
                ),
                "natural_occluded_points": len(natural_occluded_values),
                "natural_occluded_nme": (
                    float(np.mean(natural_occluded_values))
                    if natural_occluded_values
                    else None
                ),
            }
        )
        visible_targets: list[bool] = []
        visible_predictions: list[bool] = []
        for row in rows:
            scores = row.get("landmark_scores")
            if scores is None:
                continue
            for labeled, visible, score in zip(
                row["labeled"],
                row["visible"],
                scores,
                strict=True,
            ):
                if not bool(labeled):
                    continue
                visible_targets.append(bool(visible))
                visible_predictions.append(
                    float(score) >= self.visibility_threshold
                )
        if visible_targets:
            targets = np.asarray(visible_targets, dtype=bool)
            predictions = np.asarray(visible_predictions, dtype=bool)
            true_positive = int((predictions & targets).sum())
            false_positive = int((predictions & ~targets).sum())
            false_negative = int((~predictions & targets).sum())
            true_negative = int((~predictions & ~targets).sum())
            precision = true_positive / max(
                1,
                true_positive + false_positive,
            )
            recall = true_positive / max(
                1,
                true_positive + false_negative,
            )
            visible_f1 = (
                2.0 * precision * recall
                / max(1.0e-8, precision + recall)
            )
            occluded_precision = true_negative / max(
                1,
                true_negative + false_negative,
            )
            occluded_recall = true_negative / max(
                1,
                true_negative + false_positive,
            )
            occluded_f1 = (
                2.0 * occluded_precision * occluded_recall
                / max(1.0e-8, occluded_precision + occluded_recall)
            )
            result.update(
                {
                    "visibility_threshold": self.visibility_threshold,
                    "visibility_evaluated_keypoints": len(visible_targets),
                    "visibility_true_positive": true_positive,
                    "visibility_false_positive": false_positive,
                    "visibility_false_negative": false_negative,
                    "visibility_true_negative": true_negative,
                    "visibility_precision": precision,
                    "visibility_recall": recall,
                    "visibility_f1": visible_f1,
                    "occluded_precision": occluded_precision,
                    "occluded_recall": occluded_recall,
                    "occluded_f1": occluded_f1,
                    "visibility_occluded_macro_f1": (
                        visible_f1 + occluded_f1
                    )
                    / 2.0,
                    "visibility_confusion_matrix": {
                        "labels": ["visible", "occluded"],
                        "rows": "ground_truth",
                        "columns": "prediction",
                        "values": [
                            [true_positive, false_negative],
                            [false_positive, true_negative],
                        ],
                    },
                }
            )
        presence_targets: list[bool] = []
        presence_predictions: list[bool] = []
        state_targets: list[int] = []
        state_predictions: list[int] = []
        for row in rows:
            presence_scores = row.get("presence_scores")
            state_probabilities = row.get("state_probabilities")
            if presence_scores is None:
                continue
            for labeled, visible, score, probabilities in zip(
                row["labeled"],
                row["visible"],
                presence_scores,
                (
                    state_probabilities
                    if state_probabilities is not None
                    else [None] * len(presence_scores)
                ),
                strict=True,
            ):
                presence_targets.append(bool(labeled))
                presence_predictions.append(
                    float(score) >= self.visibility_threshold
                )
                if probabilities is not None:
                    state_targets.append(
                        2 if bool(visible) else 1 if bool(labeled) else 0
                    )
                    state_predictions.append(
                        int(np.asarray(probabilities).argmax())
                    )
        if presence_targets:
            targets = np.asarray(presence_targets, dtype=bool)
            predictions = np.asarray(presence_predictions, dtype=bool)
            true_positive = int((predictions & targets).sum())
            false_positive = int((predictions & ~targets).sum())
            false_negative = int((~predictions & targets).sum())
            true_negative = int((~predictions & ~targets).sum())
            present_precision = true_positive / max(
                1,
                true_positive + false_positive,
            )
            present_recall = true_positive / max(
                1,
                true_positive + false_negative,
            )
            present_f1 = (
                2.0 * present_precision * present_recall
                / max(1.0e-8, present_precision + present_recall)
            )
            missing_precision = true_negative / max(
                1,
                true_negative + false_negative,
            )
            missing_recall = true_negative / max(
                1,
                true_negative + false_positive,
            )
            missing_f1 = (
                2.0 * missing_precision * missing_recall
                / max(1.0e-8, missing_precision + missing_recall)
            )
            result.update(
                {
                    "presence_threshold": self.visibility_threshold,
                    "presence_evaluated_keypoints": len(presence_targets),
                    "presence_true_positive": true_positive,
                    "presence_false_positive": false_positive,
                    "presence_false_negative": false_negative,
                    "presence_true_negative": true_negative,
                    "presence_precision": present_precision,
                    "presence_recall": present_recall,
                    "presence_f1": present_f1,
                    "missing_precision": missing_precision,
                    "missing_recall": missing_recall,
                    "missing_f1": missing_f1,
                    "presence_missing_macro_f1": (
                        present_f1 + missing_f1
                    )
                    / 2.0,
                    "presence_confusion_matrix": {
                        "labels": ["present", "missing"],
                        "rows": "ground_truth",
                        "columns": "prediction",
                        "values": [
                            [true_positive, false_negative],
                            [false_positive, true_negative],
                        ],
                    },
                }
            )
        if state_targets:
            labels = ("missing", "occluded", "visible")
            confusion = np.zeros((3, 3), dtype=np.int64)
            for target, prediction in zip(
                state_targets,
                state_predictions,
                strict=True,
            ):
                confusion[target, prediction] += 1
            per_class: dict[str, dict[str, float | int]] = {}
            f1_values: list[float] = []
            for index, name in enumerate(labels):
                true_positive = int(confusion[index, index])
                false_positive = int(confusion[:, index].sum()) - true_positive
                false_negative = int(confusion[index, :].sum()) - true_positive
                precision = true_positive / max(
                    1,
                    true_positive + false_positive,
                )
                recall = true_positive / max(
                    1,
                    true_positive + false_negative,
                )
                f1 = (
                    2.0 * precision * recall
                    / max(1.0e-8, precision + recall)
                )
                f1_values.append(f1)
                per_class[name] = {
                    "support": int(confusion[index, :].sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            result.update(
                {
                    "state_evaluated_keypoints": len(state_targets),
                    "state_accuracy": float(
                        np.trace(confusion) / max(1, confusion.sum())
                    ),
                    "state_macro_f1": float(np.mean(f1_values)),
                    "state_per_class": per_class,
                    "state_confusion_matrix": {
                        "labels": list(labels),
                        "rows": "ground_truth",
                        "columns": "prediction",
                        "values": confusion.tolist(),
                    },
                }
            )
        result["per_landmark"] = {
            name: {
                "count": len(per_landmark[name]),
                "nme": float(np.mean(per_landmark[name])) if per_landmark[name] else None,
                "pck_0.05": (
                    sum(value <= 0.05 for value in per_landmark[name]) / len(per_landmark[name])
                    if per_landmark[name]
                    else None
                ),
            }
            for name in self.names
        }
        return result

    def compute(self) -> dict[str, Any]:
        summary = self._summarize_rows(self.rows)
        summary["detector_false_positives"] = int(self.detector_false_positives)
        detector_tp = len(self.rows) - self.missed_instances
        detector_gt = len(self.rows)
        detector_precision = detector_tp / (detector_tp + self.detector_false_positives) if detector_tp + self.detector_false_positives else None
        detector_recall = detector_tp / detector_gt if detector_gt else None
        detector_f1 = (
            2.0 * detector_precision * detector_recall / (detector_precision + detector_recall)
            if detector_precision is not None and detector_recall is not None and detector_precision + detector_recall > 0
            else None
        )
        summary["detector_operating_point"] = {
            "true_positives": detector_tp,
            "false_positives": int(self.detector_false_positives),
            "false_negatives": int(self.missed_instances),
            "precision": detector_precision,
            "recall": detector_recall,
            "f1": detector_f1,
        }
        summary["by_face_size"] = {
            name: self._summarize_rows([row for row in self.rows if row["face_size"] == name])
            for name in ("small", "medium", "large")
        }
        videos = sorted({str(row["source_video_id"]) for row in self.rows if row["source_video_id"]})
        summary["by_source_video"] = {
            video: self._summarize_rows([row for row in self.rows if row["source_video_id"] == video])
            for video in videos
        }
        error_matrix = np.full((len(self.rows), len(self.names)), np.nan, dtype=np.float64)
        for row_index, row in enumerate(self.rows):
            for landmark_index, value in enumerate(row["normalized_errors"]):
                if value is not None:
                    error_matrix[row_index, landmark_index] = float(value)
        correlation = np.eye(len(self.names), dtype=np.float64)
        pair_counts = np.zeros_like(correlation, dtype=np.int64)
        for left in range(len(self.names)):
            for right in range(len(self.names)):
                valid = np.isfinite(error_matrix[:, left]) & np.isfinite(error_matrix[:, right])
                pair_counts[left, right] = int(valid.sum())
                if valid.sum() >= 3 and np.std(error_matrix[valid, left]) > 0 and np.std(error_matrix[valid, right]) > 0:
                    correlation[left, right] = float(np.corrcoef(error_matrix[valid, left], error_matrix[valid, right])[0, 1])
                elif left != right:
                    correlation[left, right] = float("nan")
        summary["landmark_error_correlation"] = {
            "landmark_names": list(self.names),
            "matrix": [[None if not np.isfinite(value) else float(value) for value in row] for row in correlation],
            "pair_counts": pair_counts.tolist(),
        }
        return summary


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1e-9)


def match_detector_boxes(
    ground_truth: Sequence[Sequence[float]],
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(ground_truth):
        for detection_index, detection in enumerate(detections):
            overlap = iou_xyxy(gt_box, detection["bbox_xyxy"])
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, detection_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_detection: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, gt_index, detection_index in candidates:
        if gt_index in used_gt or detection_index in used_detection:
            continue
        used_gt.add(gt_index)
        used_detection.add(detection_index)
        matches.append((gt_index, detection_index, overlap))
    missed = [index for index in range(len(ground_truth)) if index not in used_gt]
    false_positives = [index for index in range(len(detections)) if index not in used_detection]
    return sorted(matches), missed, false_positives
