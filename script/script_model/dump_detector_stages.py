import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "script_experiment"))
sys.path.insert(0, str(SCRIPT_ROOT / "script_model"))

from evaluate_face_detections import bbox_area, bbox_iou
from run_scrfd_mediapipe import (
    IMAGE_EXTENSIONS,
    decode_scrfd_outputs,
    resize_with_padding,
    unpad_bbox,
    unpad_keypoints,
)


SCRFD_COLOR = (255, 0, 0)
MEDIAPIPE_COLOR = (0, 180, 0)
MERGED_COLOR = (0, 220, 220)
FINAL_COLOR = (0, 0, 255)
REMOVED_COLOR = (180, 180, 180)
TEXT_BG = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump SCRFD, MediaPipe, merge, and postprocess detector stages."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write debug outputs.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/scrfd/det_10g.onnx"),
        help="SCRFD ONNX model path. Default: models/scrfd/det_10g.onnx.",
    )
    parser.add_argument("--input-size", type=int, default=640, help="Square SCRFD input size. Default: 640.")
    parser.add_argument(
        "--score-min",
        type=float,
        default=0.01,
        help="Lowest SCRFD score to dump before postprocessing. Default: 0.01.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Postprocess score threshold applied after merge. Default: 0.5.",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.4, help="Primary NMS IoU threshold. Default: 0.4.")
    parser.add_argument(
        "--mediapipe-min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe Face Detection min_detection_confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--merge-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold used to match SCRFD and MediaPipe candidates. Default: 0.3.",
    )
    parser.add_argument(
        "--merge-center-distance-ratio",
        type=float,
        default=0.35,
        help="Fallback center-distance threshold as a ratio of the smaller box size. Default: 0.35.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=0.0,
        help="Optional minimum sqrt(width*height) after primary NMS. Default: 0.",
    )
    parser.add_argument(
        "--post-nms-iou",
        type=float,
        default=None,
        help="Optional second-pass NMS IoU threshold after size filtering.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Optional maximum detections per image after sorting by score.",
    )
    parser.add_argument("--recursive", action="store_true", help="Read image directories recursively.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--save-overlays", action="store_true", help="Write per-stage overlay images.")
    parser.add_argument("--print-every", type=int, default=10, help="Progress print interval. Default: 10.")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any]:
    missing = []
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        mp = None
        missing.append("mediapipe")
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")
    if missing:
        raise RuntimeError("Missing Python package(s): " + ", ".join(missing))
    return cv2, mp, np


def iter_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_index_from_name(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def time_sec_from_name(path: Path) -> float | None:
    match = re.search(r"_(\d+(?:\.\d+)?)s(?:_|$)", path.stem)
    return float(match.group(1)) if match else None


def bbox_xyxy(bbox: dict[str, float]) -> list[float]:
    return [float(bbox["xtl"]), float(bbox["ytl"]), float(bbox["xbr"]), float(bbox["ybr"])]


def bbox_from_xyxy(values: list[float]) -> dict[str, float]:
    xtl, ytl, xbr, ybr = values
    return {
        "xtl": float(xtl),
        "ytl": float(ytl),
        "xbr": float(xbr),
        "ybr": float(ybr),
        "width": max(0.0, float(xbr) - float(xtl)),
        "height": max(0.0, float(ybr) - float(ytl)),
    }


def point_list_to_dict(points: list[dict[str, float]]) -> dict[str, list[float]]:
    return {point["label"]: [float(point["x"]), float(point["y"])] for point in points}


def detection_bbox(detection: dict[str, Any]) -> dict[str, float]:
    return bbox_from_xyxy(detection["bbox_xyxy"])


def box_size(detection: dict[str, Any]) -> float:
    bbox = detection_bbox(detection)
    return math.sqrt(max(0.0, bbox["width"]) * max(0.0, bbox["height"]))


def center_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_bbox = detection_bbox(first)
    second_bbox = detection_bbox(second)
    first_center = ((first_bbox["xtl"] + first_bbox["xbr"]) / 2.0, (first_bbox["ytl"] + first_bbox["ybr"]) / 2.0)
    second_center = (
        (second_bbox["xtl"] + second_bbox["xbr"]) / 2.0,
        (second_bbox["ytl"] + second_bbox["ybr"]) / 2.0,
    )
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def base_record(image_path: Path, image: Any) -> dict[str, Any]:
    height, width = image.shape[:2]
    return {
        "image_name": image_path.name,
        "frame_index": frame_index_from_name(image_path),
        "time_sec": time_sec_from_name(image_path),
        "image_width": int(width),
        "image_height": int(height),
    }


def jsonl_record(image_path: Path, image: Any, model: str, detections: list[dict[str, Any]]) -> dict[str, Any]:
    record = base_record(image_path, image)
    record["model"] = model
    record["detections"] = detections
    return record


def dump_scrfd(
    image: Any,
    net: Any,
    args: argparse.Namespace,
    cv2: Any,
    np: Any,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    model_input, scale, pad_x, pad_y = resize_with_padding(image, args.input_size, cv2, np)
    blob = cv2.dnn.blobFromImage(
        model_input,
        scalefactor=1.0 / 128.0,
        size=(args.input_size, args.input_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    bboxes, scores, keypoints = decode_scrfd_outputs(outputs, args.input_size, args.score_min, np)

    detections = []
    for index, bbox in enumerate(bboxes):
        unpadded = unpad_bbox(bbox, scale, pad_x, pad_y, width, height)
        landmarks = unpad_keypoints(keypoints[index], scale, pad_x, pad_y) if keypoints[index] else []
        det_id = f"scrfd_{index}"
        detections.append(
            {
                "det_id": det_id,
                "source": "scrfd",
                "score": float(scores[index]),
                "bbox_xyxy": bbox_xyxy(unpadded),
                "bbox_area": bbox_area(unpadded),
                "landmarks": point_list_to_dict(landmarks),
                "raw_extra": {
                    "input_size": args.input_size,
                    "scale": float(scale),
                    "pad_x": int(pad_x),
                    "pad_y": int(pad_y),
                    "model_bbox_xyxy": [float(value) for value in bbox],
                },
            }
        )
    return detections


def dump_mediapipe(image: Any, face_detector: Any, cv2: Any) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)
    detections = []
    if not result.detections:
        return detections

    keypoint_labels = [
        "right_eye",
        "left_eye",
        "nose_tip",
        "mouth_center",
        "right_ear_tragion",
        "left_ear_tragion",
    ]
    for index, detection in enumerate(result.detections):
        relative_bbox = detection.location_data.relative_bounding_box
        x1 = max(0.0, min(float(width), relative_bbox.xmin * width))
        y1 = max(0.0, min(float(height), relative_bbox.ymin * height))
        x2 = max(0.0, min(float(width), (relative_bbox.xmin + relative_bbox.width) * width))
        y2 = max(0.0, min(float(height), (relative_bbox.ymin + relative_bbox.height) * height))
        bbox = bbox_from_xyxy([x1, y1, x2, y2])
        normalized_landmarks = []
        landmarks = {}
        for label, keypoint in zip(keypoint_labels, detection.location_data.relative_keypoints):
            normalized_landmarks.append({"label": label, "x": float(keypoint.x), "y": float(keypoint.y)})
            landmarks[label] = [float(keypoint.x) * width, float(keypoint.y) * height]
        detections.append(
            {
                "det_id": f"mediapipe_{index}",
                "source": "mediapipe",
                "score": float(detection.score[0]) if detection.score else None,
                "bbox_xyxy": bbox_xyxy(bbox),
                "bbox_area": bbox_area(bbox),
                "landmarks": landmarks,
                "raw_extra": {
                    "normalized_bbox": {
                        "xmin": float(relative_bbox.xmin),
                        "ymin": float(relative_bbox.ymin),
                        "width": float(relative_bbox.width),
                        "height": float(relative_bbox.height),
                    },
                    "normalized_landmarks": normalized_landmarks,
                },
            }
        )
    return detections


def merge_detections(
    scrfd: list[dict[str, Any]],
    mediapipe: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    merged = []
    matched_mediapipe = set()

    for scrfd_det in sorted(scrfd, key=lambda item: item.get("score") or 0.0, reverse=True):
        best = None
        best_key = (-1.0, -1.0)
        scrfd_size = max(1.0, min(detection_bbox(scrfd_det)["width"], detection_bbox(scrfd_det)["height"]))
        for mp_index, mp_det in enumerate(mediapipe):
            if mp_index in matched_mediapipe:
                continue
            iou = bbox_iou(detection_bbox(scrfd_det), detection_bbox(mp_det))
            distance = center_distance(scrfd_det, mp_det)
            max_distance = args.merge_center_distance_ratio * scrfd_size
            if iou < args.merge_iou_threshold and distance > max_distance:
                continue
            key = (iou, -distance)
            if key > best_key:
                best = (mp_index, mp_det, iou, distance)
                best_key = key

        det_id = f"merged_{len(merged)}"
        if best is None:
            merged.append(
                {
                    "det_id": det_id,
                    "sources": ["scrfd"],
                    "source_det_ids": [scrfd_det["det_id"]],
                    "score": scrfd_det["score"],
                    "bbox_xyxy": scrfd_det["bbox_xyxy"],
                    "landmarks": scrfd_det["landmarks"],
                    "merge_info": {
                        "matched": False,
                        "match_iou": None,
                        "center_distance": None,
                        "selected_bbox_from": "scrfd",
                        "selected_score_from": "scrfd",
                        "selected_landmarks_from": "scrfd",
                    },
                }
            )
            continue

        mp_index, mp_det, iou, distance = best
        matched_mediapipe.add(mp_index)
        merged.append(
            {
                "det_id": det_id,
                "sources": ["scrfd", "mediapipe"],
                "source_det_ids": [scrfd_det["det_id"], mp_det["det_id"]],
                "score": scrfd_det["score"],
                "bbox_xyxy": scrfd_det["bbox_xyxy"],
                "landmarks": scrfd_det["landmarks"],
                "merge_info": {
                    "matched": True,
                    "match_iou": iou,
                    "center_distance": distance,
                    "selected_bbox_from": "scrfd",
                    "selected_score_from": "scrfd",
                    "selected_landmarks_from": "scrfd",
                },
            }
        )

    for mp_index, mp_det in enumerate(mediapipe):
        if mp_index in matched_mediapipe:
            continue
        merged.append(
            {
                "det_id": f"merged_{len(merged)}",
                "sources": ["mediapipe"],
                "source_det_ids": [mp_det["det_id"]],
                "score": mp_det["score"],
                "bbox_xyxy": mp_det["bbox_xyxy"],
                "landmarks": mp_det["landmarks"],
                "merge_info": {
                    "matched": False,
                    "match_iou": None,
                    "center_distance": None,
                    "selected_bbox_from": "mediapipe",
                    "selected_score_from": "mediapipe",
                    "selected_landmarks_from": "mediapipe",
                },
            }
        )
    return merged


def removal_from_detection(
    detection: dict[str, Any],
    removed_by: str,
    reason: str,
    kept: dict[str, Any] | None = None,
    iou_with_kept: float | None = None,
) -> dict[str, Any]:
    removed = {
        "det_id": detection["det_id"],
        "source_det_ids": detection.get("source_det_ids", []),
        "sources": detection.get("sources", []),
        "score": detection.get("score"),
        "bbox_xyxy": detection.get("bbox_xyxy"),
        "removed_by": removed_by,
        "reason": reason,
        "kept_det_id": kept.get("det_id") if kept else None,
        "iou_with_kept": iou_with_kept,
    }
    return removed


def nms_with_reasons(
    detections: list[dict[str, Any]],
    threshold: float,
    removed_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    removed = []
    for detection in sorted(detections, key=lambda item: item.get("score") or 0.0, reverse=True):
        matched_kept = None
        matched_iou = 0.0
        for kept_detection in kept:
            iou = bbox_iou(detection_bbox(detection), detection_bbox(kept_detection))
            if iou >= threshold and iou > matched_iou:
                matched_kept = kept_detection
                matched_iou = iou
        if matched_kept is None:
            kept.append(detection)
        else:
            removed.append(
                removal_from_detection(
                    detection,
                    removed_by=removed_by,
                    reason="overlap_with_higher_score_detection",
                    kept=matched_kept,
                    iou_with_kept=matched_iou,
                )
            )
    return kept, removed


def postprocess(
    detections: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    removed = []
    score_kept = []
    for detection in detections:
        score = detection.get("score")
        if score is None or float(score) < args.score_threshold:
            removed.append(
                removal_from_detection(
                    detection,
                    removed_by="score_threshold",
                    reason="score_below_threshold",
                )
            )
        else:
            score_kept.append(detection)

    nms_kept, nms_removed = nms_with_reasons(score_kept, args.nms_threshold, "nms")
    removed.extend(nms_removed)

    size_kept = []
    for detection in nms_kept:
        if args.min_box_size > 0.0 and box_size(detection) < args.min_box_size:
            removed.append(
                removal_from_detection(
                    detection,
                    removed_by="noise_filter",
                    reason="box_size_below_min_box_size",
                )
            )
        else:
            size_kept.append(detection)

    if args.post_nms_iou is not None:
        post_kept, post_removed = nms_with_reasons(size_kept, args.post_nms_iou, "post_nms")
        removed.extend(post_removed)
    else:
        post_kept = size_kept

    sorted_kept = sorted(post_kept, key=lambda item: item.get("score") or 0.0, reverse=True)
    if args.max_detections is not None and len(sorted_kept) > args.max_detections:
        for detection in sorted_kept[args.max_detections :]:
            removed.append(
                removal_from_detection(
                    detection,
                    removed_by="max_detections",
                    reason="lower_score_after_max_detections_limit",
                )
            )
        sorted_kept = sorted_kept[: args.max_detections]

    final = []
    for index, detection in enumerate(sorted_kept):
        final.append(
            {
                "det_id": f"final_{index}",
                "original_det_ids": detection.get("source_det_ids", []) + [detection["det_id"]],
                "score": detection.get("score"),
                "bbox_xyxy": detection.get("bbox_xyxy"),
                "landmarks": detection.get("landmarks", {}),
                "postprocess_info": {
                    "nms_kept": True,
                    "noise_removed": False,
                    "smoothed": False,
                    "gap_filled": False,
                    "track_id": None,
                },
            }
        )
    return final, removed


def safe_stem(image_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._") or "image"


def draw_label(image: Any, text: str, x: int, y: int, cv2: Any) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(height + baseline + 2, y)
    cv2.rectangle(image, (x, y - height - baseline - 4), (x + width + 6, y + 2), TEXT_BG, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 1), font, scale, TEXT_COLOR, thickness)


def draw_detections(
    image: Any,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int],
    cv2: Any,
    removed: bool = False,
    show_labels: bool = True,
) -> None:
    for detection in detections:
        bbox = detection_bbox(detection)
        x1 = int(round(bbox["xtl"]))
        y1 = int(round(bbox["ytl"]))
        x2 = int(round(bbox["xbr"]))
        y2 = int(round(bbox["ybr"]))
        thickness = 1 if removed else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            label = f"{detection.get('det_id')} {float(detection.get('score') or 0.0):.3f}"
            draw_label(image, label, x1, y1 - 4, cv2)


def write_overlay(
    output_dir: Path,
    image_path: Path,
    image: Any,
    stage: str,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int],
    cv2: Any,
    removed: list[dict[str, Any]] | None = None,
) -> None:
    overlay = image.copy()
    draw_detections(overlay, detections, color, cv2)
    if removed:
        removed_as_detections = [
            {
                "det_id": item["det_id"],
                "score": item.get("score"),
                "bbox_xyxy": item["bbox_xyxy"],
            }
            for item in removed
            if item.get("bbox_xyxy")
        ]
        draw_detections(overlay, removed_as_detections, REMOVED_COLOR, cv2, removed=True, show_labels=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{safe_stem(image_path)}_{stage}.jpg"), overlay)


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def max_score(detections: list[dict[str, Any]]) -> float | None:
    scores = [float(item["score"]) for item in detections if item.get("score") is not None]
    return max(scores) if scores else None


def removed_count(removed: list[dict[str, Any]], name: str) -> int:
    return sum(1 for item in removed if item.get("removed_by") == name)


def duplicate_candidate_count(detections: list[dict[str, Any]], threshold: float) -> int:
    count = 0
    for first_index, first in enumerate(detections):
        for second in detections[first_index + 1 :]:
            if bbox_iou(detection_bbox(first), detection_bbox(second)) >= threshold:
                count += 1
    return count


def summary_row(
    image_path: Path,
    scrfd: list[dict[str, Any]],
    mediapipe: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    final: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "image_name": image_path.name,
        "frame_index": frame_index_from_name(image_path),
        "time_sec": time_sec_from_name(image_path),
        "scrfd_count": len(scrfd),
        "mediapipe_count": len(mediapipe),
        "merged_count": len(merged),
        "final_count": len(final),
        "removed_by_nms": removed_count(removed, "nms") + removed_count(removed, "post_nms"),
        "removed_by_noise": removed_count(removed, "noise_filter"),
        "removed_by_score_threshold": removed_count(removed, "score_threshold"),
        "max_scrfd_score": max_score(scrfd),
        "max_mediapipe_score": max_score(mediapipe),
        "max_final_score": max_score(final),
        "has_detection_final": bool(final),
        "duplicate_candidate_count": duplicate_candidate_count(merged, args.nms_threshold),
        "low_score_candidate_count": sum(
            1 for item in merged if item.get("score") is None or float(item["score"]) < args.score_threshold
        ),
        "small_area_removed_count": removed_count(removed, "noise_filter"),
        "large_area_removed_count": 0,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"SCRFD model not found: {args.model}")
    for name in ["score_min", "score_threshold", "nms_threshold", "merge_iou_threshold"]:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.post_nms_iou is not None and not 0.0 <= args.post_nms_iou <= 1.0:
        raise ValueError("--post-nms-iou must be between 0 and 1.")
    if args.max_detections is not None and args.max_detections <= 0:
        raise ValueError("--max-detections must be positive.")
    if args.min_box_size < 0.0:
        raise ValueError("--min-box-size must be non-negative.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    cv2, mp, np = import_dependencies()

    image_paths = iter_images(args.input, args.recursive)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output / "overlays"
    net = cv2.dnn.readNetFromONNX(str(args.model))
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=args.mediapipe_min_detection_confidence,
    )

    summary_rows = []
    skipped_rows = []
    handles = {
        "raw_scrfd": (args.output / "raw_scrfd.jsonl").open("w", encoding="utf-8"),
        "raw_mediapipe": (args.output / "raw_mediapipe.jsonl").open("w", encoding="utf-8"),
        "merged": (args.output / "merged_candidates.jsonl").open("w", encoding="utf-8"),
        "postprocessed": (args.output / "postprocessed.jsonl").open("w", encoding="utf-8"),
        "removed": (args.output / "removed_detections.jsonl").open("w", encoding="utf-8"),
    }

    try:
        for index, image_path in enumerate(image_paths, start=1):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError("failed_to_read_image")

                scrfd = dump_scrfd(image, net, args, cv2, np)
                mediapipe = dump_mediapipe(image, face_detector, cv2)
                merged = merge_detections(scrfd, mediapipe, args)
                final, removed = postprocess(merged, args)

                write_jsonl_line(handles["raw_scrfd"], jsonl_record(image_path, image, "scrfd", scrfd))
                write_jsonl_line(handles["raw_mediapipe"], jsonl_record(image_path, image, "mediapipe", mediapipe))
                write_jsonl_line(handles["merged"], {**base_record(image_path, image), "detections": merged})
                write_jsonl_line(handles["postprocessed"], {**base_record(image_path, image), "detections": final})
                write_jsonl_line(handles["removed"], {**base_record(image_path, image), "removed": removed})
                summary_rows.append(summary_row(image_path, scrfd, mediapipe, merged, final, removed, args))

                if args.save_overlays:
                    write_overlay(overlay_dir, image_path, image, "scrfd", scrfd, SCRFD_COLOR, cv2)
                    write_overlay(overlay_dir, image_path, image, "mediapipe", mediapipe, MEDIAPIPE_COLOR, cv2)
                    write_overlay(overlay_dir, image_path, image, "merged", merged, MERGED_COLOR, cv2)
                    write_overlay(overlay_dir, image_path, image, "postprocessed", final, FINAL_COLOR, cv2, removed=removed)

                if args.print_every and (index == 1 or index % args.print_every == 0 or index == len(image_paths)):
                    print(f"{index}/{len(image_paths)} {image_path.name}: final={len(final)} removed={len(removed)}")
            except Exception as error:
                skipped_rows.append({"image_path": str(image_path), "error": str(error)})
                print(f"skipped {image_path}: {error}", file=sys.stderr)
    finally:
        for handle in handles.values():
            handle.close()
        face_detector.close()

    summary_fields = [
        "image_name",
        "frame_index",
        "time_sec",
        "scrfd_count",
        "mediapipe_count",
        "merged_count",
        "final_count",
        "removed_by_nms",
        "removed_by_noise",
        "removed_by_score_threshold",
        "max_scrfd_score",
        "max_mediapipe_score",
        "max_final_score",
        "has_detection_final",
        "duplicate_candidate_count",
        "low_score_candidate_count",
        "small_area_removed_count",
        "large_area_removed_count",
    ]
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.output / "skipped.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "error"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"output: {args.output}")
    print(f"images: {len(summary_rows)} processed, {len(skipped_rows)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
