"""Model-independent helpers for the legacy DINO/SCRFD text-label evaluator.

Preserves xyxy IoU and greedy one-to-one matching, including tie ordering.
object_scale uses sqrt(max(1, bbox area)); this legacy convention is NOT the
current baseline's GT-Head-normalized NME contract.
"""

import math
from pathlib import Path
from typing import Any

from src.data.face_annotations import IMAGE_EXTENSIONS, KEYPOINT_NAMES

def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = bbox_area([x1, y1, x2, y2])
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    objects = []
    if not label_path.exists():
        return objects
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 5 + len(KEYPOINT_NAMES) * 3:
            raise ValueError(f"Invalid YOLO pose label at {label_path}:{line_number}")
        xc, yc, bw, bh = values[1:5]
        box_width = bw * width
        box_height = bh * height
        cx = xc * width
        cy = yc * height
        keypoints = []
        raw_keypoints = values[5:]
        for index, name in enumerate(KEYPOINT_NAMES):
            x_value = raw_keypoints[index * 3]
            y_value = raw_keypoints[index * 3 + 1]
            visibility = int(raw_keypoints[index * 3 + 2])
            keypoints.append(
                {
                    "name": name,
                    "x": x_value * width if visibility else 0.0,
                    "y": y_value * height if visibility else 0.0,
                    "visibility": visibility,
                }
            )
        objects.append(
            {
                "bbox": [
                    cx - box_width / 2.0,
                    cy - box_height / 2.0,
                    cx + box_width / 2.0,
                    cy + box_height / 2.0,
                ],
                "keypoints": keypoints,
            }
        )
    return objects


def greedy_matches(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            iou = bbox_iou(gt["bbox"], prediction["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def object_scale(gt: dict[str, Any]) -> float:
    box = gt["bbox"]
    return math.sqrt(max(1.0, bbox_area(box)))


def visible_keypoints(gt: dict[str, Any]) -> list[dict[str, Any]]:
    return [point for point in gt["keypoints"] if int(point["visibility"]) > 0]
