from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Sequence


BBox = Sequence[float]


def bbox_iou(a: BBox, b: BBox) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(a: BBox, b: BBox) -> float:
    center_a = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    center_b = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    diagonal = max(1.0, sqrt(max(1.0, a[2] - a[0]) ** 2 + max(1.0, a[3] - a[1]) ** 2))
    return hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) / diagonal


def bbox_area_ratio(a: BBox, b: BBox) -> float:
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return area_b / area_a if area_a > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class BoxMatch:
    left_index: int
    right_index: int
    iou: float
    center_distance: float
    area_ratio: float


def match_bboxes(
    left: Sequence[BBox],
    right: Sequence[BBox],
    *,
    iou_threshold: float = 0.30,
    center_distance_threshold: float = 0.25,
    min_area_ratio: float = 0.33,
    max_area_ratio: float = 3.0,
) -> tuple[list[BoxMatch], list[int], list[int]]:
    candidates: list[tuple[float, BoxMatch]] = []
    for left_index, box_a in enumerate(left):
        for right_index, box_b in enumerate(right):
            iou = bbox_iou(box_a, box_b)
            distance = normalized_center_distance(box_a, box_b)
            ratio = bbox_area_ratio(box_a, box_b)
            if iou >= iou_threshold or (
                distance <= center_distance_threshold and min_area_ratio <= ratio <= max_area_ratio
            ):
                quality = iou + max(0.0, 1.0 - distance) * 0.35 - abs(ratio - 1.0) * 0.05
                candidates.append((quality, BoxMatch(left_index, right_index, iou, distance, ratio)))
    matches: list[BoxMatch] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, match in sorted(candidates, key=lambda item: item[0], reverse=True):
        if match.left_index in used_left or match.right_index in used_right:
            continue
        matches.append(match)
        used_left.add(match.left_index)
        used_right.add(match.right_index)
    return matches, [i for i in range(len(left)) if i not in used_left], [i for i in range(len(right)) if i not in used_right]
