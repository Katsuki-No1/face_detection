from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from script.script_experiment.rtmdet_rtmpose_experiment import (  # noqa: E402
    average_precision_for_class,
    bbox_size_bucket,
    iou_xywh,
    keypoint_instances_from_mmpose,
    local_checkpoint_load_context,
    match_detection_at_threshold,
    xyxy_to_xywh,
)


KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
STAGES = ("FDP", "HDP", "GTFP", "HFDP", "GTHFDP")
DEFAULT_OUTPUT_DIR = Path("outputs_experiment/fdp_hfdp_cascade_ablation")
DETECTION_CATEGORIES = (
    {"id": 1, "name": "head"},
    {"id": 2, "name": "face"},
)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def newest_checkpoint(work_dir: Path, patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        candidates = list(work_dir.glob(pattern))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"checkpoint not found under {work_dir}")


def default_checkpoints(output_dir: Path) -> dict[str, Path]:
    return {
        "detector": newest_checkpoint(output_dir / "work_dirs" / "mmdet" / "rtmdet_hf_full", ("best*.pth", "epoch_100.pth")),
        "crop_face_detector": newest_checkpoint(output_dir / "work_dirs" / "mmdet" / "rtmdet_headcrop_face", ("best*.pth", "epoch_100.pth")),
        "pose_face": newest_checkpoint(output_dir / "work_dirs" / "mmpose" / "rtmpose_face", ("best*.pth", "epoch_100.pth")),
        "pose_head": newest_checkpoint(output_dir / "work_dirs" / "mmpose" / "rtmpose_head", ("best*.pth", "epoch_100.pth")),
    }


def detect_image(det_model: Any, image: Any) -> list[dict[str, Any]]:
    from mmengine.registry import init_default_scope  # type: ignore
    from mmdet.apis import inference_detector  # type: ignore

    init_default_scope("mmdet")
    result = inference_detector(det_model, image)
    pred = result.pred_instances
    detections: list[dict[str, Any]] = []
    for bbox, score, label in zip(pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        category_id = int(label) + 1
        detections.append(
            {
                "category_id": category_id,
                "class_name": "head" if category_id == 1 else "face",
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            }
        )
    return detections


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def clip_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_xyxy(box: list[float], width: int, height: int, margin: float) -> list[float] | None:
    if margin <= 0.0:
        raise ValueError(f"crop margin must be positive: {margin}")
    clipped = clip_xyxy(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w * 0.5
    cy = y1 + box_h * 0.5
    new_w = box_w * margin
    new_h = box_h * margin
    expanded = [cx - new_w * 0.5, cy - new_h * 0.5, cx + new_w * 0.5, cy + new_h * 0.5]
    return clip_xyxy(expanded, width, height)


def select_detections(detections: list[dict[str, Any]], category_id: int, score_thr: float, topk: int) -> list[dict[str, Any]]:
    rows = [det for det in detections if int(det.get("category_id", 0)) == category_id and float(det.get("score", 0.0)) >= score_thr]
    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return rows[:topk]


def run_pose(pose_model: Any, image_path: Path, boxes_xyxy: list[list[float]]) -> list[dict[str, Any]]:
    from mmengine.registry import init_default_scope  # type: ignore
    from mmpose.apis import inference_topdown  # type: ignore

    if not boxes_xyxy:
        return []
    init_default_scope("mmpose")
    preds = inference_topdown(pose_model, image_path.as_posix(), bboxes=boxes_xyxy)
    return keypoint_instances_from_mmpose(preds)


def run_full_detector(output_dir: Path, config: Path, checkpoint: Path, device: str, split: str) -> tuple[dict[str, Any], float]:
    from mmdet.apis import init_detector  # type: ignore

    coco = read_json(output_dir / "data" / "mmdet_hf" / f"{split}.json")
    dataset_root = output_dir / "data" / "mmdet_hf"
    started = time.perf_counter()
    with local_checkpoint_load_context(checkpoint, output_dir):
        det_model = init_detector(config.as_posix(), checkpoint.as_posix(), device=device)
    rows = []
    for image in coco.get("images", []):
        image_path = dataset_root / image["file_name"]
        detections = detect_image(det_model, image_path.as_posix())
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "detections": detections})
    seconds = time.perf_counter() - started
    predictions = {
        "created_at": now_text(),
        "status": "completed",
        "config": config.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "device": device,
        "split": split,
        "images": rows,
    }
    write_json(output_dir / "predictions" / "detector" / "predictions.json", predictions)
    return predictions, seconds


def detections_by_image(predictions: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(row["image_id"]): row.get("detections", []) for row in predictions.get("images", [])}


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        out[int(ann["image_id"])].append(ann)
    return out


def pose_from_detection_stage(
    output_dir: Path,
    stage: str,
    pose_model: Any,
    gt_coco: dict[str, Any],
    detector_predictions: dict[str, Any],
    *,
    category_id: int,
    score_thr: float,
    topk: int,
    split: str,
) -> tuple[dict[str, Any], float]:
    dataset_root = output_dir / "data" / "mmdet_hf"
    det_by_image = detections_by_image(detector_predictions)
    started = time.perf_counter()
    rows = []
    for image in gt_coco.get("images", []):
        detections = select_detections(det_by_image.get(int(image["id"]), []), category_id, score_thr, topk)
        boxes = [xywh_to_xyxy(det["bbox"]) for det in detections]
        instances = run_pose(pose_model, dataset_root / image["file_name"], boxes)
        for instance, det in zip(instances, detections):
            instance["detector_score"] = det.get("score", 0.0)
            instance["detector_category_id"] = det.get("category_id")
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances})
    seconds = time.perf_counter() - started
    predictions = {"created_at": now_text(), "status": "completed", "stage": stage, "split": split, "images": rows}
    write_json(output_dir / "predictions" / stage / "predictions.json", predictions)
    return predictions, seconds


def pose_from_gt_bbox_stage(
    output_dir: Path,
    stage: str,
    pose_model: Any,
    gt_coco: dict[str, Any],
    *,
    split: str,
) -> tuple[dict[str, Any], float]:
    dataset_root = output_dir / "data" / "mmdet_hf"
    ann_by_image = annotations_by_image(gt_coco)
    started = time.perf_counter()
    rows = []
    for image in gt_coco.get("images", []):
        anns = ann_by_image.get(int(image["id"]), [])
        boxes = [xywh_to_xyxy(ann["bbox"]) for ann in anns]
        instances = run_pose(pose_model, dataset_root / image["file_name"], boxes)
        for instance, ann in zip(instances, anns):
            instance["gt_annotation_id"] = ann["id"]
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances})
    seconds = time.perf_counter() - started
    predictions = {"created_at": now_text(), "status": "completed", "stage": stage, "split": split, "images": rows}
    write_json(output_dir / "predictions" / stage / "predictions.json", predictions)
    return predictions, seconds


def crop_face_detection_stage(
    output_dir: Path,
    stage: str,
    det_model: Any,
    pose_model: Any,
    gt_face_coco: dict[str, Any],
    heads_by_image: dict[int, list[dict[str, Any]]],
    *,
    face_score_thr: float,
    topk: int,
    crop_margin: float,
    split: str,
) -> tuple[dict[str, Any], float]:
    import cv2  # type: ignore

    dataset_root = output_dir / "data" / "mmdet_hf"
    started = time.perf_counter()
    rows = []
    for image in gt_face_coco.get("images", []):
        image_path = dataset_root / image["file_name"]
        full = cv2.imread(image_path.as_posix())
        if full is None:
            rows.append({"image_id": image["id"], "file_name": image["file_name"], "instances": []})
            continue
        height, width = full.shape[:2]
        face_dets: list[dict[str, Any]] = []
        for head_index, head in enumerate(heads_by_image.get(int(image["id"]), [])):
            crop_xyxy = expand_xyxy(xywh_to_xyxy(head["bbox"]), width, height, crop_margin)
            if crop_xyxy is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in crop_xyxy]
            crop = full[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_dets = select_detections(detect_image(det_model, crop), 2, face_score_thr, topk)
            for det in crop_dets:
                bx1, by1, bx2, by2 = xywh_to_xyxy(det["bbox"])
                mapped = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                clipped = clip_xyxy(mapped, width, height)
                if clipped is None:
                    continue
                face_dets.append(
                    {
                        **det,
                        "bbox": xyxy_to_xywh(clipped),
                        "source_head_index": head_index,
                        "source_head_score": head.get("score"),
                        "crop_xyxy": [x1, y1, x2, y2],
                    }
                )
        face_dets.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        face_dets = face_dets[:topk]
        boxes = [xywh_to_xyxy(det["bbox"]) for det in face_dets]
        instances = run_pose(pose_model, image_path, boxes)
        for instance, det in zip(instances, face_dets):
            instance["detector_score"] = det.get("score", 0.0)
            instance["source_head_score"] = det.get("source_head_score")
            instance["crop_xyxy"] = det.get("crop_xyxy")
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances, "face_detections": face_dets})
    seconds = time.perf_counter() - started
    predictions = {
        "created_at": now_text(),
        "status": "completed",
        "stage": stage,
        "split": split,
        "crop_margin": crop_margin,
        "images": rows,
    }
    write_json(output_dir / "predictions" / stage / "predictions.json", predictions)
    return predictions, seconds


def flatten_pred_keypoints(pred: dict[str, Any]) -> list[float]:
    keypoints = pred.get("keypoints", [])
    if not keypoints:
        return []
    if isinstance(keypoints[0], list):
        flat: list[float] = []
        scores = pred.get("keypoint_scores", [])
        for index, point in enumerate(keypoints):
            x, y = float(point[0]), float(point[1])
            score = float(scores[index]) if index < len(scores) else float(pred.get("score", 1.0))
            flat.extend([x, y, 2 if score > 0 else 0])
        return flat
    return [float(value) for value in keypoints]


def keypoint_prediction_map(predictions: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(row["image_id"]): row.get("instances", []) for row in predictions.get("images", [])}


def match_keypoints(coco: dict[str, Any], predictions: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], int]:
    pred_by_image = keypoint_prediction_map(predictions)
    matched_pred_ids: set[tuple[int, int]] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        candidates = pred_by_image.get(image_id, [])
        direct = [pred for pred in candidates if pred.get("gt_annotation_id") == ann.get("id")]
        selected = direct[0] if direct else None
        if selected is None:
            best_iou = 0.0
            best_index: int | None = None
            for index, pred in enumerate(candidates):
                key = (image_id, index)
                if key in matched_pred_ids:
                    continue
                iou = iou_xywh(ann["bbox"], pred.get("bbox", [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                selected = candidates[best_index]
                matched_pred_ids.add((image_id, best_index))
        pairs.append((ann, selected))
    total_predictions = sum(len(row.get("instances", [])) for row in predictions.get("images", []))
    return pairs, total_predictions


def evaluate_keypoints(coco: dict[str, Any], predictions: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    pairs, total_predictions = match_keypoints(coco, predictions)
    distances: list[float] = []
    normalized_distances: list[float] = []
    pck_counts = Counter()
    kp_errors: dict[str, list[float]] = {name: [] for name in KEYPOINT_NAMES}
    size_errors: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    missed = 0
    evaluated_instances = 0
    for ann, pred in pairs:
        if pred is None:
            missed += 1
            continue
        pred_kpts = flatten_pred_keypoints(pred)
        if len(pred_kpts) < len(KEYPOINT_NAMES) * 3:
            missed += 1
            continue
        diag = math.hypot(float(ann["bbox"][2]), float(ann["bbox"][3]))
        if diag <= 0:
            missed += 1
            continue
        evaluated_instances += 1
        bucket = bbox_size_bucket(ann["bbox"])
        gt = ann["keypoints"]
        for index, name in enumerate(KEYPOINT_NAMES):
            if int(gt[index * 3 + 2]) <= 0:
                continue
            dx = float(pred_kpts[index * 3]) - float(gt[index * 3])
            dy = float(pred_kpts[index * 3 + 1]) - float(gt[index * 3 + 1])
            dist = math.hypot(dx, dy)
            norm = dist / diag
            distances.append(dist)
            normalized_distances.append(norm)
            kp_errors[name].append(norm)
            size_errors[bucket].append(norm)
            for threshold in (0.03, 0.05, 0.10):
                if norm <= threshold:
                    pck_counts[str(threshold)] += 1
    denom = len(normalized_distances)
    metrics = {
        "stage": stage,
        "status": predictions.get("status", "completed"),
        "normalization": "matched_gt_bbox_diagonal",
        "gt_instances": len(coco.get("annotations", [])),
        "predicted_instances": total_predictions,
        "evaluated_instances": evaluated_instances,
        "missed_instances": missed,
        "NME": sum(normalized_distances) / denom if denom else None,
        "PCK@0.03": pck_counts["0.03"] / denom if denom else None,
        "PCK@0.05": pck_counts["0.05"] / denom if denom else None,
        "PCK@0.10": pck_counts["0.1"] / denom if denom else None,
        "keypoint_wise_error": {name: (sum(values) / len(values) if values else None) for name, values in kp_errors.items()},
        "bbox_size_error": {name: (sum(values) / len(values) if values else None) for name, values in size_errors.items()},
    }
    write_json(output_dir / "metrics" / stage / "keypoint_metrics.json", metrics)
    return metrics


def evaluate_detector(
    output_dir: Path,
    detector_predictions: dict[str, Any],
    split: str,
    score_thresholds: dict[int, float],
) -> dict[str, Any]:
    coco = read_json(output_dir / "data" / "mmdet_hf" / f"{split}.json")
    raw_primary = match_detection_at_threshold(coco, detector_predictions, 0.5)
    thresholded_predictions = filter_detector_predictions(detector_predictions, score_thresholds)
    primary = match_detection_at_threshold(coco, thresholded_predictions, 0.5)
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    class_ap: dict[str, Any] = {}
    for category in DETECTION_CATEGORIES:
        aps = [average_precision_for_class(coco, detector_predictions, int(category["id"]), threshold) for threshold in thresholds]
        valid = [value for value in aps if value is not None]
        class_ap[category["name"]] = {
            "AP50": average_precision_for_class(coco, detector_predictions, int(category["id"]), 0.5),
            "AP50_95": sum(valid) / len(valid) if valid else None,
        }
    class_recall = {}
    class_precision = {}
    for category in DETECTION_CATEGORIES:
        class_coco = {**coco, "annotations": [ann for ann in coco.get("annotations", []) if int(ann["category_id"]) == int(category["id"])]}
        class_predictions = filter_detector_category(thresholded_predictions, int(category["id"]))
        class_match = match_detection_at_threshold(class_coco, class_predictions, 0.5)
        class_recall[category["name"]] = class_match["recall"]
        class_precision[category["name"]] = class_match["precision"]
    metrics = {
        "split": split,
        "score_thresholds": {"head": score_thresholds.get(1, 0.0), "face": score_thresholds.get(2, 0.0)},
        "mAP@0.5": sum(v["AP50"] for v in class_ap.values() if v["AP50"] is not None) / max(1, sum(1 for v in class_ap.values() if v["AP50"] is not None)),
        "mAP@0.5:0.95": sum(v["AP50_95"] for v in class_ap.values() if v["AP50_95"] is not None) / max(1, sum(1 for v in class_ap.values() if v["AP50_95"] is not None)),
        "precision": primary["precision"],
        "recall": primary["recall"],
        "F1": primary["f1"],
        "TP": primary["tp"],
        "FP": primary["fp"],
        "FN": primary["fn"],
        "raw_unthresholded_precision": raw_primary["precision"],
        "raw_unthresholded_recall": raw_primary["recall"],
        "raw_unthresholded_F1": raw_primary["f1"],
        "class_wise_AP": class_ap,
        "class_precision": class_precision,
        "class_recall": class_recall,
    }
    write_json(output_dir / "metrics" / "detector" / "metrics.json", metrics)
    return metrics


def gt_heads_from_coco(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            {"bbox": ann["bbox"], "score": 1.0, "gt_annotation_id": ann["id"], "category_id": 1, "class_name": "head"}
            for ann in anns
        ]
        for image_id, anns in annotations_by_image(coco).items()
    }


def detected_heads_from_predictions(predictions: dict[str, Any], score_thr: float, topk: int) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: select_detections(detections, 1, score_thr, topk)
        for image_id, detections in detections_by_image(predictions).items()
    }


def filter_detector_predictions(predictions: dict[str, Any], score_thresholds: dict[int, float]) -> dict[str, Any]:
    rows = []
    for image_row in predictions.get("images", []):
        detections = []
        for det in image_row.get("detections", []):
            category_id = int(det.get("category_id", 0))
            threshold = float(score_thresholds.get(category_id, 0.0))
            if float(det.get("score", 0.0)) >= threshold:
                detections.append(det)
        rows.append({**image_row, "detections": detections})
    return {**predictions, "images": rows}


def filter_detector_category(predictions: dict[str, Any], category_id: int) -> dict[str, Any]:
    rows = []
    for image_row in predictions.get("images", []):
        detections = [det for det in image_row.get("detections", []) if int(det.get("category_id", 0)) == category_id]
        rows.append({**image_row, "detections": detections})
    return {**predictions, "images": rows}


def latest_work_dir_log(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob("*/20*.log"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def training_runtime_rows(output_dir: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    if run_id:
        suffix = f"_{run_id}"
        specs = [
            ("rtmdet_hf_full", latest_work_dir_log(output_dir / "work_dirs" / "mmdet" / f"rtmdet_hf_full{suffix}"), None),
            (
                "rtmdet_headcrop_face_full100_640",
                latest_work_dir_log(output_dir / "work_dirs" / "mmdet" / f"rtmdet_headcrop_face_full100_640{suffix}"),
                None,
            ),
            ("rtmpose_face", latest_work_dir_log(output_dir / "work_dirs" / "mmpose" / f"rtmpose_face{suffix}"), None),
            ("rtmpose_head", latest_work_dir_log(output_dir / "work_dirs" / "mmpose" / f"rtmpose_head{suffix}"), None),
        ]
    else:
        specs = [
            ("rtmdet_hf_full", output_dir / "logs" / "train_detector_console.log", None),
            ("rtmdet_headcrop_face", output_dir / "logs" / "train_headcrop_face_detector_console.log", None),
            ("rtmpose_face", output_dir / "logs" / "train_rtmpose_face_console.log", None),
            ("rtmpose_head", output_dir / "logs" / "train_rtmpose_head_console.log", output_dir / "logs" / "train_rtmpose_head_time.txt"),
        ]
    rows: list[dict[str, Any]] = []
    for name, log_path, time_path in specs:
        row: dict[str, Any] = {"name": name, "kind": "train"}
        if log_path and log_path.exists():
            row["log_path"] = log_path.as_posix()
            text = log_path.read_text(errors="ignore")
            times = re.findall(r"^((?:\d{4}/)?\d\d/\d\d \d\d:\d\d:\d\d)", text, flags=re.M)
            if times:
                row["log_start"] = times[0]
                row["log_end"] = times[-1]
                try:
                    if len(times[0]) == len("YYYY/MM/DD HH:MM:SS"):
                        started = datetime.strptime(times[0], "%Y/%m/%d %H:%M:%S")
                        ended = datetime.strptime(times[-1], "%Y/%m/%d %H:%M:%S")
                    else:
                        year = datetime.now().year
                        started = datetime.strptime(f"{year}/{times[0]}", "%Y/%m/%d %H:%M:%S")
                        ended = datetime.strptime(f"{year}/{times[-1]}", "%Y/%m/%d %H:%M:%S")
                    if ended >= started:
                        row["elapsed_seconds"] = round((ended - started).total_seconds(), 3)
                        row["elapsed_source"] = "console_log_timestamps"
                except ValueError:
                    pass
        if time_path and time_path.exists():
            for line in time_path.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    row[key] = value
            if "elapsed_seconds" in row:
                row["elapsed_source"] = "usr_bin_time"
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    crop_face_detector_config = (
        args.crop_face_detector_config.resolve()
        if args.crop_face_detector_config
        else output_dir / "configs" / "mmdet" / "rtmdet_headcrop_face.py"
    )
    checkpoints = default_checkpoints(output_dir)
    if args.detector_checkpoint:
        checkpoints["detector"] = args.detector_checkpoint.resolve()
    if args.crop_face_detector_checkpoint:
        checkpoints["crop_face_detector"] = args.crop_face_detector_checkpoint.resolve()
    if args.pose_face_checkpoint:
        checkpoints["pose_face"] = args.pose_face_checkpoint.resolve()
    if args.pose_head_checkpoint:
        checkpoints["pose_head"] = args.pose_head_checkpoint.resolve()

    from mmdet.apis import init_detector  # type: ignore
    from mmpose.apis import init_model  # type: ignore

    detector_predictions, detector_seconds = run_full_detector(
        output_dir,
        output_dir / "configs" / "mmdet" / "rtmdet_hf_full.py",
        checkpoints["detector"],
        args.device,
        args.split,
    )
    detector_metrics = evaluate_detector(
        output_dir,
        detector_predictions,
        args.split,
        {1: args.head_score_thr, 2: args.face_score_thr},
    )

    with local_checkpoint_load_context(checkpoints["crop_face_detector"], output_dir):
        crop_face_det_model = init_detector(
            crop_face_detector_config.as_posix(),
            checkpoints["crop_face_detector"].as_posix(),
            device=args.device,
        )
    with local_checkpoint_load_context(checkpoints["pose_face"], output_dir):
        pose_face = init_model(
            (output_dir / "configs" / "mmpose" / "rtmpose_face.py").as_posix(),
            checkpoints["pose_face"].as_posix(),
            device=args.device,
        )
    with local_checkpoint_load_context(checkpoints["pose_head"], output_dir):
        pose_head = init_model(
            (output_dir / "configs" / "mmpose" / "rtmpose_head.py").as_posix(),
            checkpoints["pose_head"].as_posix(),
            device=args.device,
        )

    face_coco = read_json(output_dir / "data" / "mmpose_face" / f"{args.split}.json")
    head_coco = read_json(output_dir / "data" / "mmpose_head" / f"{args.split}.json")

    stage_predictions: dict[str, dict[str, Any]] = {}
    stage_seconds: dict[str, float] = {}
    stage_predictions["FDP"], stage_seconds["FDP_pose_only"] = pose_from_detection_stage(
        output_dir,
        "FDP",
        pose_face,
        face_coco,
        detector_predictions,
        category_id=2,
        score_thr=args.face_score_thr,
        topk=args.topk,
        split=args.split,
    )
    stage_predictions["HDP"], stage_seconds["HDP_pose_only"] = pose_from_detection_stage(
        output_dir,
        "HDP",
        pose_head,
        head_coco,
        detector_predictions,
        category_id=1,
        score_thr=args.head_score_thr,
        topk=args.topk,
        split=args.split,
    )
    stage_predictions["GTFP"], stage_seconds["GTFP"] = pose_from_gt_bbox_stage(output_dir, "GTFP", pose_face, face_coco, split=args.split)
    stage_predictions["HFDP"], stage_seconds["HFDP_crop_face_pose_only"] = crop_face_detection_stage(
        output_dir,
        "HFDP",
        crop_face_det_model,
        pose_face,
        face_coco,
        detected_heads_from_predictions(detector_predictions, args.head_score_thr, args.topk),
        face_score_thr=args.face_score_thr,
        topk=args.topk,
        crop_margin=args.crop_margin,
        split=args.split,
    )
    stage_predictions["GTHFDP"], stage_seconds["GTHFDP_crop_face_pose_only"] = crop_face_detection_stage(
        output_dir,
        "GTHFDP",
        crop_face_det_model,
        pose_face,
        face_coco,
        gt_heads_from_coco(head_coco),
        face_score_thr=args.face_score_thr,
        topk=args.topk,
        crop_margin=args.crop_margin,
        split=args.split,
    )

    metrics_rows: list[dict[str, Any]] = []
    for stage, predictions in stage_predictions.items():
        gt = head_coco if stage == "HDP" else face_coco
        metrics = evaluate_keypoints(gt, predictions, stage, output_dir)
        metrics_rows.append(
            {
                "stage": stage,
                "NME": metrics.get("NME"),
                "PCK@0.03": metrics.get("PCK@0.03"),
                "PCK@0.05": metrics.get("PCK@0.05"),
                "PCK@0.10": metrics.get("PCK@0.10"),
                "gt_instances": metrics.get("gt_instances"),
                "predicted_instances": metrics.get("predicted_instances"),
                "evaluated_instances": metrics.get("evaluated_instances"),
                "missed_instances": metrics.get("missed_instances"),
            }
        )
    write_csv(output_dir / "metrics" / "fdp_stage_keypoint_summary.csv", metrics_rows)

    runtime_rows = training_runtime_rows(output_dir, args.run_id)
    inference_rows = [
        {"name": "full_detector_shared", "kind": "inference", "elapsed_seconds": detector_seconds, "images": len(detector_predictions.get("images", []))},
        {"name": "FDP_pose_only", "kind": "inference", "elapsed_seconds": stage_seconds["FDP_pose_only"], "images": len(face_coco.get("images", []))},
        {"name": "FDP_estimated_end_to_end", "kind": "inference", "elapsed_seconds": detector_seconds + stage_seconds["FDP_pose_only"], "images": len(face_coco.get("images", []))},
        {"name": "HDP_pose_only", "kind": "inference", "elapsed_seconds": stage_seconds["HDP_pose_only"], "images": len(head_coco.get("images", []))},
        {"name": "HDP_estimated_end_to_end", "kind": "inference", "elapsed_seconds": detector_seconds + stage_seconds["HDP_pose_only"], "images": len(head_coco.get("images", []))},
        {"name": "GTFP", "kind": "inference", "elapsed_seconds": stage_seconds["GTFP"], "images": len(face_coco.get("images", []))},
        {"name": "HFDP_estimated_end_to_end", "kind": "inference", "elapsed_seconds": detector_seconds + stage_seconds["HFDP_crop_face_pose_only"], "images": len(face_coco.get("images", []))},
        {"name": "GTHFDP", "kind": "inference", "elapsed_seconds": stage_seconds["GTHFDP_crop_face_pose_only"], "images": len(face_coco.get("images", []))},
    ]
    runtime_rows.extend(inference_rows)
    write_csv(output_dir / "reports" / "runtime_summary.csv", runtime_rows)
    write_json(
        output_dir / "reports" / "runtime_summary.json",
        {
            "created_at": now_text(),
            "run_id": args.run_id,
            "split": args.split,
            "device": args.device,
            "score_thresholds": {"face": args.face_score_thr, "head": args.head_score_thr},
            "crop_margin": args.crop_margin,
            "configs": {
                "crop_face_detector": crop_face_detector_config.as_posix(),
            },
            "checkpoints": {key: value.as_posix() for key, value in checkpoints.items()},
            "detector_metrics": detector_metrics,
            "runtime": runtime_rows,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FDP-family RTMDet/RTMPose inference and evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--face-score-thr", type=float, default=0.3)
    parser.add_argument("--head-score-thr", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-face-detector-config", type=Path, default=None)
    parser.add_argument("--crop-face-detector-checkpoint", type=Path, default=None)
    parser.add_argument("--pose-face-checkpoint", type=Path, default=None)
    parser.add_argument("--pose-head-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-margin", type=float, default=1.20)
    parser.add_argument("--run-id", default=None, help="Optional run suffix used to report training runtime from matching work_dirs.")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
