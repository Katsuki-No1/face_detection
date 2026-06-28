from __future__ import annotations

from typing import Any


DEFAULT_CLASS_NAMES = ("head", "face")
DEFAULT_KEYPOINT_NAMES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
SOURCE_YOLO_KEYPOINT_NAMES = ("e1", "e2", "n", "m1", "m2")
FLIP_INDEX = (1, 0, 2, 4, 3)


def yolo_xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = box
    box_w = bw * width
    box_h = bh * height
    cx = xc * width
    cy = yc * height
    return [cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0]


def parse_yolo_pose_label_line(
    line: str,
    *,
    width: int,
    height: int,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    keypoint_names: tuple[str, ...] = DEFAULT_KEYPOINT_NAMES,
    target_class_name: str = "face",
) -> dict[str, Any]:
    """Converts one existing YOLO Pose label row into a training target dict.

    Existing labels use class id 0 for face and keypoint order e1/e2/n/m1/m2.
    The experimental detector reserves class ids for ("head", "face"), so this
    adapter maps the YOLO row to the configured target_class_name by default.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got width={width}, height={height}")
    values = [float(value) for value in line.split()]
    if len(values) < 5:
        raise ValueError("YOLO pose label line must contain class id and bbox values")
    if target_class_name not in class_names:
        raise ValueError(f"target_class_name={target_class_name!r} is not in class_names={class_names}")

    raw_keypoints = values[5:]
    expected_keypoint_values = len(keypoint_names) * 3
    if raw_keypoints and len(raw_keypoints) < expected_keypoint_values:
        raise ValueError(
            "YOLO pose label has too few keypoint values: "
            f"got={len(raw_keypoints)}, expected={expected_keypoint_values}"
        )

    keypoints: list[list[float]] = []
    for index in range(len(keypoint_names)):
        offset = index * 3
        if offset + 2 >= len(raw_keypoints):
            keypoints.append([0.0, 0.0, 0.0])
            continue
        visibility = float(raw_keypoints[offset + 2])
        if visibility <= 0:
            keypoints.append([0.0, 0.0, 0.0])
            continue
        keypoints.append([raw_keypoints[offset] * width, raw_keypoints[offset + 1] * height, visibility])

    return {
        "boxes": [yolo_xywhn_to_xyxy(values[1:5], width, height)],
        "labels": [class_names.index(target_class_name)],
        "keypoints": [keypoints],
        "source_class_id": int(values[0]),
        "source_keypoint_names": SOURCE_YOLO_KEYPOINT_NAMES,
        "keypoint_names": keypoint_names,
    }


def dummy_face_target(width: int = 448, height: int = 448) -> dict[str, Any]:
    line = "0 0.5 0.5 0.25 0.30 0.45 0.45 2 0.55 0.45 2 0.50 0.52 2 0.46 0.60 2 0.54 0.60 2"
    return parse_yolo_pose_label_line(line, width=width, height=height)
