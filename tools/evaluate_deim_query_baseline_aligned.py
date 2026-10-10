#!/usr/bin/env python3
"""Evaluate ellipse-DEIM + Query Refinement on the baseline GT-Head contract.

The baseline report evaluates face presence, ellipse geometry, and fixed facial
landmarks from ground-truth Head instances.  This tool keeps the candidate
detector's native full-image input, associates each face prediction to one GT
Head ROI, and then computes the downstream metrics with the Head ROI side as
the normalization scale.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison1.landmark_detection.dataset import image_to_tensor  # noqa: E402
from keypoint_comparison1.landmark_detection.geometry import (  # noqa: E402
    build_crop_transform,
    normalized_crop_to_original,
    warp_crop,
)
from keypoint_comparison1.landmark_detection.model import build_landmark_model  # noqa: E402
from src.data.face_crop_cache import (  # noqa: E402
    canonicalize_ellipse,
    ellipse_aligned_rectangle,
)
from src.evaluation.ellipse_detection import ellipse_iou  # noqa: E402
from src.evaluation.baseline_aligned import (  # noqa: E402
    _state_metrics as _common_state_metrics,
    evaluate_normalized_predictions,
)


BASELINE_TEST = {
    "face_macro_f1": 0.812,
    "face_present_recall": 0.969,
    "no_face_recall": 0.769,
    "false_face_rate": 0.231,
    "ellipse_mean_iou": 0.723,
    "ellipse_median_iou": 0.787,
    "ellipse_p10_iou": 0.450,
    "ellipse_iou_ge_0.50": 0.871,
    "ellipse_iou_ge_0.75": 0.595,
    "ellipse_center_nme": 0.0527,
    "ellipse_major_relative_error": 0.127,
    "ellipse_minor_relative_error": 0.176,
    "ellipse_angle_mae_deg": 9.32,
    "point_macro_f1_0.05": 0.835,
    "point_macro_f1_0.10": 0.896,
    "matched_point_nme_0.10": 0.0190,
    "visible_point_nme_0.10": 0.0174,
    "occluded_point_nme_0.10": 0.0312,
}

SEMANTIC_SLOTS = {
    "eye": (0, 1),
    "nose": (2,),
    "mouth": (3, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline-aligned GT Head evaluation for direct ellipse-DEIMv2-L "
            "plus DINOv3 Query Refinement."
        )
    )
    parser.add_argument(
        "--head-annotation",
        type=Path,
        default=REPO_ROOT
        / "outputs/dataset_cache/face_pose_current/annotations/face_dino_v1/test.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPO_ROOT / "outputs/dataset_cache/face_pose_current",
    )
    parser.add_argument(
        "--detector-cache",
        type=Path,
        default=REPO_ROOT
        / "outputs/evaluations/deimv2_l_query_refinement_baseline_aligned_v2_test"
        / "cache/deimv2_l_raw_predictions.json",
    )
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs/landmark_detection"
        / (
            "dinov3_vits16_face_pose_drive_v2_balanced_811_"
            "gt_ellipse_long_axis_square_coco_visibility"
        )
        / "query_refinement/checkpoints/best_nme.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / (
            "outputs/evaluations/"
            "deimv2_l_query_refinement_factorized_state_"
            "baseline_aligned_v2_test"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.33,
        help="Val-locked DEIM face threshold.",
    )
    parser.add_argument(
        "--point-threshold",
        type=float,
        default=0.70,
        help=(
            "Keypoint presence threshold. Factorized checkpoints apply it to "
            "P(present); legacy checkpoints apply it to point existence."
        ),
    )
    parser.add_argument("--head-roi-scale", type=float, default=1.20)
    parser.add_argument("--ellipse-crop-scale", type=float, default=1.25)
    parser.add_argument("--angle-ratio-threshold", type=float, default=1.05)
    parser.add_argument(
        "--ignored-head-id",
        action="append",
        type=int,
        default=[4096],
        help="Ambiguous Head excluded from face supervision (repeatable).",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--architecture-id", default="deimv2_l_query_refinement")
    parser.add_argument(
        "--common-output-dir",
        type=Path,
        help="Write the shared all-architecture metric table here.",
    )
    parser.add_argument(
        "--head-predictions",
        type=Path,
        help="COCO-format predicted HEAD boxes for deployment-style evaluation.",
    )
    parser.add_argument("--head-detector-config", type=Path)
    parser.add_argument("--head-detector-checkpoint", type=Path)
    parser.add_argument(
        "--pipeline-contract",
        choices=(
            "two_stage_face_keypoint",
            "three_stage_head_face_keypoint",
        ),
        default="two_stage_face_keypoint",
        help=(
            "Exact deployed cascade. Two-stage evaluation starts from the "
            "full-image FACE detector and leaves HEAD-only metrics unsupported."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_head_predictions(
    path: Path,
    *,
    image_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("HEAD predictions must be a COCO prediction list")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload:
        if int(row.get("category_id", -1)) != 1:
            continue
        image_id = int(row["image_id"])
        if image_id not in image_ids:
            raise RuntimeError(
                f"HEAD prediction references unknown image_id={image_id}"
            )
        x, y, width, height = (float(value) for value in row["bbox"])
        grouped[image_id].append(
            {
                "bbox_xyxy": [x, y, x + width, y + height],
                "score": float(row["score"]),
            }
        )
    for image_id in image_ids:
        grouped[image_id].sort(key=lambda row: float(row["score"]), reverse=True)
        grouped[image_id] = grouped[image_id][:100]
    return dict(grouped)


def _cuda_device_index(device: torch.device) -> int | None:
    """Return a concrete CUDA index accepted by CUDA memory-stat APIs."""
    if device.type != "cuda":
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(
            f"CUDA device index {index} is outside the visible range [0, {count})"
        )
    return int(index)


def _prepare_cuda_device(device: torch.device) -> int | None:
    index = _cuda_device_index(device)
    if index is None:
        return None
    torch.cuda.set_device(index)
    # Passing the concrete integer avoids a PyTorch/driver-dependent
    # ``Invalid device argument`` seen with torch.device("cuda:0").
    torch.cuda.reset_peak_memory_stats(index)
    return index


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
            )


def _resolve_image(root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (
        (relative,)
        if relative.is_absolute()
        else (root / relative, root / "images" / relative)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image was not found below {root}: {file_name}")


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def _ellipse_to_xyxy(ellipse: Sequence[float]) -> list[float]:
    cx, cy, major, minor, theta = map(float, ellipse)
    cosine, sine = math.cos(theta), math.sin(theta)
    extent_x = math.sqrt((major * cosine) ** 2 + (minor * sine) ** 2)
    extent_y = math.sqrt((major * sine) ** 2 + (minor * cosine) ** 2)
    return [cx - extent_x, cy - extent_y, cx + extent_x, cy + extent_y]


def _expanded_head_square(
    box_xywh: Sequence[float],
    *,
    scale: float,
) -> tuple[list[float], float]:
    x, y, width, height = (float(value) for value in box_xywh)
    side = max(width, height) * float(scale)
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    half = side * 0.5
    return [center_x - half, center_y - half, center_x + half, center_y + half], side


def _center_in_box(center: Sequence[float], box: Sequence[float]) -> bool:
    return (
        float(box[0]) <= float(center[0]) <= float(box[2])
        and float(box[1]) <= float(center[1]) <= float(box[3])
    )


def associate_predictions_to_heads(
    heads: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    head_roi_scale: float,
) -> dict[int, int]:
    """Greedily assign high-score detections whose ellipse center is in a Head.

    DEIM remains a full-image face detector.  GT Heads are used only as
    instance anchors, and every face prediction can be consumed at most once.
    """

    pairs: list[tuple[float, float, int, int]] = []
    for head_index, head in enumerate(heads):
        roi, side = _expanded_head_square(
            head["bbox"],
            scale=head_roi_scale,
        )
        center_x = (roi[0] + roi[2]) * 0.5
        center_y = (roi[1] + roi[3]) * 0.5
        for detection_index, detection in enumerate(detections):
            ellipse = detection.get("ellipse")
            if not isinstance(ellipse, (list, tuple)) or len(ellipse) != 5:
                continue
            if not _center_in_box(ellipse[:2], roi):
                continue
            distance = math.hypot(
                float(ellipse[0]) - center_x,
                float(ellipse[1]) - center_y,
            ) / max(side, 1.0e-6)
            pairs.append(
                (
                    float(detection.get("score", 0.0)),
                    -distance,
                    head_index,
                    detection_index,
                )
            )
    pairs.sort(reverse=True)
    assigned_heads: set[int] = set()
    assigned_detections: set[int] = set()
    assignments: dict[int, int] = {}
    for _score, _negative_distance, head_index, detection_index in pairs:
        if head_index in assigned_heads or detection_index in assigned_detections:
            continue
        assignments[head_index] = detection_index
        assigned_heads.add(head_index)
        assigned_detections.add(detection_index)
    return assignments


def _cache_predictions_by_file_name(
    cache_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    metadata = dict(cache.get("metadata", {}))
    annotation_path = Path(str(metadata.get("annotation", "")))
    if not annotation_path.is_file():
        raise FileNotFoundError(
            "Detector cache annotation is required to recover filename keys: "
            f"{annotation_path}"
        )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    file_name_by_id = {
        int(image["id"]): str(image["file_name"])
        for image in annotation.get("images", [])
    }
    rows_by_file_name: dict[str, list[dict[str, Any]]] = {}
    for raw_image_id, rows in cache.get("predictions", {}).items():
        image_id = int(raw_image_id)
        if image_id not in file_name_by_id:
            raise RuntimeError(f"Detector cache contains unknown image id: {image_id}")
        rows_by_file_name[file_name_by_id[image_id]] = [
            dict(row) for row in rows
        ]
    return metadata, rows_by_file_name


def _load_query_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("config"), dict
    ):
        raise RuntimeError(f"Query checkpoint has no embedded config: {checkpoint_path}")
    config = dict(checkpoint["config"])
    model, _initialization = build_landmark_model(
        config,
        variant="query_refinement",
        initialize_from_detector=False,
    )
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"Query checkpoint has no model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Query checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    state_head_mode = checkpoint.get("state_head_mode")
    factorized_state = state_head_mode == "factorized_presence_visibility"
    provenance = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "variant": checkpoint.get("variant"),
        "state_head_mode": state_head_mode,
        "factorized_state": factorized_state,
        "checkpoint_keypoint_target_policy": checkpoint.get(
            "keypoint_target_policy"
        ),
        "embedded_visibility_contract": (
            "factorized P(present) and P(visible|present)"
            if factorized_state
            else (
                "legacy COCO v>0 point-existence target; "
                "no occluded/visible state"
            )
        ),
    }
    del checkpoint, state
    gc.collect()
    return model.to(device).eval(), config, provenance


def infer_query_predictions(
    *,
    model: torch.nn.Module,
    device: torch.device,
    work_items: Sequence[dict[str, Any]],
    image_root: Path,
    batch_size: int,
    input_size: int,
    crop_scale: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> dict[int, dict[str, Any]]:
    from keypoint_comparison1.landmark_detection.face_overlap import crop_face_context, face_context_kwargs
    results: dict[int, dict[str, Any]] = {}
    image_cache: dict[str, np.ndarray] = {}
    batch_images: list[torch.Tensor] = []
    batch_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch_images:
            return
        images = torch.stack(batch_images).to(device)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda",
        ):
            output = model(images, **face_context_kwargs(model, batch_metadata, device))
        coordinates = output["decoded_coordinates"].detach().float().cpu().numpy()
        point_logits = output.get(
            "presence_logits",
            output["visibility_logits"],
        )
        scores = (
            torch.sigmoid(point_logits)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        state_probabilities = (
            output["state_probabilities"].detach().float().cpu().numpy()
            if "state_probabilities" in output
            else [None] * len(batch_metadata)
        )
        for points, point_scores, state, metadata in zip(
            coordinates,
            scores,
            state_probabilities,
            batch_metadata,
            strict=True,
        ):
            results[int(metadata["annotation_id"])] = {
                "keypoints": normalized_crop_to_original(
                    points,
                    metadata["transform"],
                ).tolist(),
                "point_scores": point_scores.tolist(),
                "state_probabilities": (
                    None if state is None else state.tolist()
                ),
            }
        batch_images.clear()
        batch_metadata.clear()

    for item in work_items:
        file_name = str(item["file_name"])
        image = image_cache.get(file_name)
        if image is None:
            image = np.asarray(
                Image.open(_resolve_image(image_root, file_name)).convert("RGB")
            )
            image_cache = {file_name: image}
        crop_box, crop_rotation = ellipse_aligned_rectangle(item["ellipse"])
        transform = build_crop_transform(
            crop_box,
            output_size=input_size,
            scale_factor=crop_scale,
            rotation_deg=crop_rotation,
            warp_bbox_aspect=False,
        )
        crop = warp_crop(image, transform)
        batch_images.append(image_to_tensor(crop, mean=mean, std=std))
        batch_metadata.append(
            {
                "annotation_id": int(item["annotation_id"]),
                "transform": transform,
                **(crop_face_context(item, transform) if getattr(model, "face_overlap_enabled", False) else {}),
            }
        )
        if len(batch_images) >= batch_size:
            flush()
    flush()
    return results


def _binary_metrics(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _face_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supervised = [row for row in rows if row["face_target"] != "ignored"]
    true_positive = sum(
        row["face_target"] == "positive" and row["face_predicted"]
        for row in supervised
    )
    false_positive = sum(
        row["face_target"] == "negative" and row["face_predicted"]
        for row in supervised
    )
    false_negative = sum(
        row["face_target"] == "positive" and not row["face_predicted"]
        for row in supervised
    )
    true_negative = sum(
        row["face_target"] == "negative" and not row["face_predicted"]
        for row in supervised
    )
    positive = _binary_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    negative = _binary_metrics(
        true_positive=true_negative,
        false_positive=false_negative,
        false_negative=false_positive,
    )
    return {
        "supervised_heads": len(supervised),
        "positive_heads": true_positive + false_negative,
        "negative_heads": true_negative + false_positive,
        "ignored_heads": len(rows) - len(supervised),
        "threshold": rows[0]["face_threshold"] if rows else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": positive["precision"],
        "face_present_recall": positive["recall"],
        "face_present_f1": positive["f1"],
        "no_face_recall": negative["recall"],
        "no_face_f1": negative["f1"],
        "false_face_rate": false_positive / max(1, false_positive + true_negative),
        "macro_f1": (positive["f1"] + negative["f1"]) * 0.5,
    }


def _ellipse_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    angle_ratio_threshold: float,
) -> dict[str, Any]:
    positive = [row for row in rows if row["face_target"] == "positive"]
    overlaps: list[float] = []
    center_errors: list[float] = []
    major_errors: list[float] = []
    minor_errors: list[float] = []
    angle_errors: list[float] = []
    associated = 0
    for row in positive:
        predicted = row.get("predicted_ellipse")
        target = row["gt_ellipse"]
        if predicted is None:
            overlaps.append(0.0)
            continue
        associated += 1
        overlap = ellipse_iou(predicted, target)
        overlaps.append(overlap)
        side = float(row["head_roi_side"])
        center_errors.append(
            math.hypot(
                float(predicted[0]) - float(target[0]),
                float(predicted[1]) - float(target[1]),
            )
            / max(side, 1.0e-6)
        )
        major_errors.append(
            abs(float(predicted[2]) - float(target[2]))
            / max(float(target[2]), 1.0e-6)
        )
        minor_errors.append(
            abs(float(predicted[3]) - float(target[3]))
            / max(float(target[3]), 1.0e-6)
        )
        if (
            float(target[2]) / max(float(target[3]), 1.0e-6)
            >= angle_ratio_threshold
        ):
            difference = abs((float(predicted[4]) - float(target[4])) % math.pi)
            difference = min(difference, math.pi - difference)
            angle_errors.append(math.degrees(difference))
    return {
        "positive_heads": len(positive),
        "associated_predictions": associated,
        "association_coverage": associated / max(1, len(positive)),
        "iou_unassociated_policy": "zero",
        "iou_implementation": "96-point ellipse polygons",
        "mean_iou": float(np.mean(overlaps)) if overlaps else None,
        "median_iou": float(np.median(overlaps)) if overlaps else None,
        "p10_iou": float(np.quantile(overlaps, 0.10)) if overlaps else None,
        "iou_ge_0.50": (
            sum(value >= 0.50 for value in overlaps) / len(overlaps)
            if overlaps
            else None
        ),
        "iou_ge_0.75": (
            sum(value >= 0.75 for value in overlaps) / len(overlaps)
            if overlaps
            else None
        ),
        "center_nme": (
            float(np.mean(center_errors)) if center_errors else None
        ),
        "major_relative_error": (
            float(np.mean(major_errors)) if major_errors else None
        ),
        "minor_relative_error": (
            float(np.mean(minor_errors)) if minor_errors else None
        ),
        "angle_ratio_threshold": angle_ratio_threshold,
        "angle_samples": len(angle_errors),
        "angle_mae_deg": (
            float(np.mean(angle_errors)) if angle_errors else None
        ),
    }


def _point_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    point_threshold: float,
    factorized_state: bool,
) -> dict[str, Any]:
    thresholds = (0.05, 0.10)
    semantic_counts = {
        name: {
            threshold: {"pred": 0, "gt": 0, "tp": 0}
            for threshold in thresholds
        }
        for name in SEMANTIC_SLOTS
    }
    matched_at_010: list[float] = []
    visible_at_010: list[float] = []
    occluded_at_010: list[float] = []
    present_points = 0
    predicted_points = 0
    evaluated_heads = 0
    for row in rows:
        if row["face_target"] == "ignored":
            continue
        states = [int(value) for value in row["gt_keypoint_states"]]
        points = row.get("predicted_keypoints")
        scores = row.get("point_scores")
        face_predicted = bool(row["face_predicted"])
        if points is not None:
            evaluated_heads += 1
        for semantic, slots in SEMANTIC_SLOTS.items():
            for slot in slots:
                target_present = states[slot] > 0
                prediction_present = (
                    face_predicted
                    and points is not None
                    and scores is not None
                    and float(scores[slot]) >= point_threshold
                )
                present_points += int(target_present)
                predicted_points += int(prediction_present)
                normalized_error = None
                if target_present and points is not None:
                    normalized_error = (
                        math.hypot(
                            float(points[slot][0])
                            - float(row["gt_keypoints"][slot][0]),
                            float(points[slot][1])
                            - float(row["gt_keypoints"][slot][1]),
                        )
                        / max(float(row["head_roi_side"]), 1.0e-6)
                    )
                for threshold in thresholds:
                    counts = semantic_counts[semantic][threshold]
                    counts["gt"] += int(target_present)
                    counts["pred"] += int(prediction_present)
                    if (
                        target_present
                        and prediction_present
                        and normalized_error is not None
                        and normalized_error <= threshold
                    ):
                        counts["tp"] += 1
                if (
                    target_present
                    and prediction_present
                    and normalized_error is not None
                    and normalized_error <= 0.10
                ):
                    matched_at_010.append(normalized_error)
                    if states[slot] == 2:
                        visible_at_010.append(normalized_error)
                    elif states[slot] == 1:
                        occluded_at_010.append(normalized_error)

    per_semantic: dict[str, Any] = {}
    macro: dict[str, float] = {}
    for threshold in thresholds:
        f1_values: list[float] = []
        for semantic in SEMANTIC_SLOTS:
            counts = semantic_counts[semantic][threshold]
            metrics = _binary_metrics(
                true_positive=counts["tp"],
                false_positive=counts["pred"] - counts["tp"],
                false_negative=counts["gt"] - counts["tp"],
            )
            per_semantic.setdefault(semantic, {})[f"{threshold:.2f}"] = {
                **counts,
                **metrics,
            }
            f1_values.append(metrics["f1"])
        macro[f"{threshold:.2f}"] = float(np.mean(f1_values))
    return {
        "point_threshold": point_threshold,
        "point_score_contract": (
            "factorized P(present)"
            if factorized_state
            else "legacy Query sigmoid trained as COCO v>0 point existence"
        ),
        "face_gate": True,
        "heads_with_query_output": evaluated_heads,
        "gt_present_points": present_points,
        "predicted_points": predicted_points,
        "macro_f1_0.05": macro["0.05"],
        "macro_f1_0.10": macro["0.10"],
        "matched_point_nme_0.10": (
            float(np.mean(matched_at_010)) if matched_at_010 else None
        ),
        "visible_point_nme_0.10": (
            float(np.mean(visible_at_010)) if visible_at_010 else None
        ),
        "occluded_point_nme_0.10": (
            float(np.mean(occluded_at_010)) if occluded_at_010 else None
        ),
        "matched_points_0.10": len(matched_at_010),
        "visible_matched_points_0.10": len(visible_at_010),
        "occluded_matched_points_0.10": len(occluded_at_010),
        "per_semantic": per_semantic,
    }


def _difference(candidate: float | None, baseline: float) -> float | None:
    return None if candidate is None else float(candidate) - float(baseline)


def _comparison_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    face = metrics["face"]
    ellipse = metrics["ellipse"]
    point = metrics["keypoint"]
    definitions = (
        ("Face macro F1", "higher", BASELINE_TEST["face_macro_f1"], face["macro_f1"]),
        (
            "Face-present recall",
            "higher",
            BASELINE_TEST["face_present_recall"],
            face["face_present_recall"],
        ),
        (
            "No-face recall",
            "higher",
            BASELINE_TEST["no_face_recall"],
            face["no_face_recall"],
        ),
        (
            "False-face rate",
            "lower",
            BASELINE_TEST["false_face_rate"],
            face["false_face_rate"],
        ),
        (
            "Ellipse mean IoU",
            "higher",
            BASELINE_TEST["ellipse_mean_iou"],
            ellipse["mean_iou"],
        ),
        (
            "Ellipse median IoU",
            "higher",
            BASELINE_TEST["ellipse_median_iou"],
            ellipse["median_iou"],
        ),
        (
            "Ellipse P10 IoU",
            "higher",
            BASELINE_TEST["ellipse_p10_iou"],
            ellipse["p10_iou"],
        ),
        (
            "Ellipse IoU >= .50",
            "higher",
            BASELINE_TEST["ellipse_iou_ge_0.50"],
            ellipse["iou_ge_0.50"],
        ),
        (
            "Ellipse IoU >= .75",
            "higher",
            BASELINE_TEST["ellipse_iou_ge_0.75"],
            ellipse["iou_ge_0.75"],
        ),
        (
            "Ellipse center NME",
            "lower",
            BASELINE_TEST["ellipse_center_nme"],
            ellipse["center_nme"],
        ),
        (
            "Major radius relative error",
            "lower",
            BASELINE_TEST["ellipse_major_relative_error"],
            ellipse["major_relative_error"],
        ),
        (
            "Minor radius relative error",
            "lower",
            BASELINE_TEST["ellipse_minor_relative_error"],
            ellipse["minor_relative_error"],
        ),
        (
            "Angle error",
            "lower",
            BASELINE_TEST["ellipse_angle_mae_deg"],
            ellipse["angle_mae_deg"],
        ),
        (
            "Point macro F1 @ NME .05",
            "higher",
            BASELINE_TEST["point_macro_f1_0.05"],
            point["macro_f1_0.05"],
        ),
        (
            "Point macro F1 @ NME .10",
            "higher",
            BASELINE_TEST["point_macro_f1_0.10"],
            point["macro_f1_0.10"],
        ),
        (
            "Matched point NME @ .10",
            "lower",
            BASELINE_TEST["matched_point_nme_0.10"],
            point["matched_point_nme_0.10"],
        ),
        (
            "Visible point NME @ .10",
            "lower",
            BASELINE_TEST["visible_point_nme_0.10"],
            point["visible_point_nme_0.10"],
        ),
        (
            "Occluded point NME @ .10",
            "lower",
            BASELINE_TEST["occluded_point_nme_0.10"],
            point["occluded_point_nme_0.10"],
        ),
    )
    return [
        {
            "metric": metric,
            "direction": direction,
            "baseline": baseline,
            "candidate": candidate,
            "candidate_minus_baseline": _difference(candidate, baseline),
        }
        for metric, direction, baseline, candidate in definitions
    ]


def _format_value(value: float | None, *, degrees: bool = False) -> str:
    if value is None:
        return "—"
    if degrees:
        return f"{value:.2f}°"
    return f"{value:.4f}"


def _write_markdown(
    path: Path,
    *,
    metrics: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
) -> None:
    query_provenance = metrics["provenance"]["query_checkpoint"]
    factorized_state = bool(query_provenance.get("factorized_state", False))
    point_contract = (
        "factorized P(present)"
        if factorized_state
        else "legacy Query point-existence sigmoid"
    )
    state_contract = (
        "The selected Query checkpoint uses factorized P(present) and "
        "P(visible|present); missing/occluded/visible state metrics are reported."
        if factorized_state
        else (
            "The selected legacy Query checkpoint predicts point existence "
            "(COCO v>0), not occluded vs visible; state metrics are unavailable."
        )
    )
    lines = [
        "# Baseline-aligned Test comparison",
        "",
        "GT Head ROI を instance anchor とし、Head 検出は評価していない。",
        "candidate は全画面 DEIM 推論を維持し、予測 ellipse 中心を 1.20× GT Head ROI に対応付けた。",
        "",
        "| 指標 | 向き | Baseline | DEIMv2-L ellipse + Query Refinement | 差 (candidate - baseline) |",
        "|---|:---:|---:|---:|---:|",
    ]
    for row in comparison:
        degrees = row["metric"] == "Angle error"
        lines.append(
            "| {metric} | {direction} | {baseline} | {candidate} | {difference} |".format(
                metric=row["metric"],
                direction="↑" if row["direction"] == "higher" else "↓",
                baseline=_format_value(row["baseline"], degrees=degrees),
                candidate=_format_value(row["candidate"], degrees=degrees),
                difference=_format_value(
                    row["candidate_minus_baseline"],
                    degrees=degrees,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Evaluation contract",
            "",
            f"- Test: {metrics['dataset']['images']} images / "
            f"{metrics['dataset']['heads']} Heads / "
            f"{metrics['dataset']['face_positive']} face-positive / "
            f"{metrics['dataset']['no_face']} no-face / "
            f"{metrics['dataset']['ignored_ambiguous']} ignored ambiguous",
            f"- Face threshold: {metrics['thresholds']['face']:.2f} "
            "(candidate Val-locked)",
            f"- Point threshold: {metrics['thresholds']['point']:.2f} "
            f"({point_contract})",
            "- Point/center NME denominator: 1.20 × max(GT Head width, height)",
            f"- Ellipse association coverage: "
            f"{metrics['ellipse']['association_coverage']:.4f}",
            "- Ellipse IoU uses the repository's 96-point polygon implementation; "
            "the baseline report describes rasterized region IoU, so very small "
            "numeric differences from raster resolution remain possible.",
            f"- {state_contract}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    evaluation_started = time.perf_counter()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device is unavailable: {args.device}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    cuda_device_index = _prepare_cuda_device(device)
    head_coco = json.loads(args.head_annotation.read_text(encoding="utf-8"))
    images = list(head_coco.get("images", []))
    if args.max_images is not None:
        images = images[: int(args.max_images)]
    selected_image_ids = {int(image["id"]) for image in images}
    head_predictions_by_image = (
        _load_head_predictions(
            args.head_predictions,
            image_ids=selected_image_ids,
        )
        if args.head_predictions is not None
        else {image_id: [] for image_id in selected_image_ids}
    )
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in head_coco.get("annotations", []):
        if int(annotation["image_id"]) in selected_image_ids:
            annotations_by_image[int(annotation["image_id"])].append(
                dict(annotation)
            )
    cache_metadata, predictions_by_file_name = _cache_predictions_by_file_name(
        args.detector_cache
    )
    ignored_head_ids = {int(value) for value in args.ignored_head_id}
    rows: list[dict[str, Any]] = []
    query_work: list[dict[str, Any]] = []
    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        heads = annotations_by_image.get(image_id, [])
        detections = predictions_by_file_name.get(file_name, [])
        assignments = associate_predictions_to_heads(
            heads,
            detections,
            head_roi_scale=args.head_roi_scale,
        )
        for head_index, head in enumerate(heads):
            annotation_id = int(head["id"])
            attribute_valid = bool(head.get("attribute_valid", False))
            face_target = (
                "positive"
                if attribute_valid
                else (
                    "ignored"
                    if annotation_id in ignored_head_ids
                    else "negative"
                )
            )
            detection_index = assignments.get(head_index)
            detection = (
                detections[detection_index]
                if detection_index is not None
                else None
            )
            score = (
                float(detection.get("score", 0.0))
                if detection is not None
                else 0.0
            )
            predicted_ellipse = None
            if detection is not None:
                raw_ellipse = detection.get("ellipse")
                if isinstance(raw_ellipse, (list, tuple)) and len(raw_ellipse) == 5:
                    predicted_ellipse = canonicalize_ellipse(raw_ellipse)
            _roi, roi_side = _expanded_head_square(
                head["bbox"],
                scale=args.head_roi_scale,
            )
            row = {
                "image_id": image_id,
                "file_name": file_name,
                "annotation_id": annotation_id,
                "source_head_index": head.get("source_head_index"),
                "face_target": face_target,
                "face_threshold": float(args.face_threshold),
                "face_score": score,
                "face_predicted": bool(
                    predicted_ellipse is not None and score >= args.face_threshold
                ),
                "head_bbox_xyxy": _xywh_to_xyxy(head["bbox"]),
                "head_roi_side": roi_side,
                "gt_ellipse": (
                    canonicalize_ellipse(head["ellipse"])
                    if attribute_valid
                    else None
                ),
                "predicted_ellipse": predicted_ellipse,
                "gt_keypoints": [
                    [float(point[0]), float(point[1])]
                    for point in head.get("keypoints", ())
                ],
                "gt_keypoint_states": [
                    int(value) for value in head.get("keypoint_states", ())
                ],
                "predicted_keypoints": None,
                "point_scores": None,
            }
            rows.append(row)
            if row["face_predicted"]:
                query_work.append(
                    {
                        "annotation_id": annotation_id,
                        "file_name": file_name,
                        "ellipse": predicted_ellipse,
                        **({key: detection[key] for key in ("face_query_index", "face_query_features", "detector_bbox_xyxy")}
                           if detection is not None and "face_query_features" in detection else {}),
                    }
                )

    expected_inventory = None
    if args.max_images is None:
        expected_inventory = (
            {
                "images": 530,
                "heads": 882,
                "face_positive": 834,
                "no_face": 46,
                "ignored_ambiguous": 2,
            }
            if args.split == "val"
            else {
                "images": 528,
                "heads": 806,
                "face_positive": 766,
                "no_face": 39,
                "ignored_ambiguous": 1,
            }
        )
        actual_inventory = {
            "images": len(images),
            "heads": len(rows),
            "face_positive": sum(
                row["face_target"] == "positive" for row in rows
            ),
            "no_face": sum(row["face_target"] == "negative" for row in rows),
            "ignored_ambiguous": sum(
                row["face_target"] == "ignored" for row in rows
            ),
        }
        if actual_inventory != expected_inventory:
            raise RuntimeError(
                "Baseline Test inventory mismatch: "
                f"expected={expected_inventory}, actual={actual_inventory}"
            )

    model, query_config, query_provenance = _load_query_model(
        args.query_checkpoint,
        device=device,
    )
    factorized_state = bool(query_provenance["factorized_state"])
    data_config = query_config["data"]
    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)
    inference_started = time.perf_counter()
    query_predictions = infer_query_predictions(
        model=model,
        device=device,
        work_items=query_work,
        image_root=args.image_root,
        batch_size=args.batch_size,
        input_size=int(data_config["input_size"]),
        crop_scale=float(args.ellipse_crop_scale),
        mean=data_config["mean"],
        std=data_config["std"],
    )
    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)
    inference_seconds = time.perf_counter() - inference_started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for row in rows:
        prediction = query_predictions.get(int(row["annotation_id"]))
        if prediction is not None:
            row["predicted_keypoints"] = prediction["keypoints"]
            row["point_scores"] = prediction["point_scores"]
            row["state_probabilities"] = prediction["state_probabilities"]

    dataset = {
        "images": len(images),
        "heads": len(rows),
        "face_positive": sum(row["face_target"] == "positive" for row in rows),
        "no_face": sum(row["face_target"] == "negative" for row in rows),
        "ignored_ambiguous": sum(
            row["face_target"] == "ignored" for row in rows
        ),
    }
    metrics = {
        "status": "ok",
        "evaluation_contract": "baseline GT Head ROI downstream diagnostic",
        "dataset": dataset,
        "thresholds": {
            "face": float(args.face_threshold),
            "point": float(args.point_threshold),
            "head_roi_scale": float(args.head_roi_scale),
            "ellipse_crop_scale": float(args.ellipse_crop_scale),
            "selection": (
                "face=DEIM Val locked; "
                + (
                    "point=factorized presence threshold fixed at 0.70; "
                    if factorized_state
                    else "point=legacy checkpoint fixed default; "
                )
                + "Test was not used for threshold selection"
            ),
        },
        "association": {
            "method": (
                "one-to-one descending-score assignment when predicted ellipse "
                "center is inside the 1.20x GT Head square"
            ),
            "gt_face_geometry_used_for_selection": False,
        },
        "face": _face_metrics(rows),
        "ellipse": _ellipse_metrics(
            rows,
            angle_ratio_threshold=float(args.angle_ratio_threshold),
        ),
        "keypoint": _point_metrics(
            rows,
            point_threshold=float(args.point_threshold),
            factorized_state=factorized_state,
        ),
        "state_metrics": _common_state_metrics(
            rows,
            supported=factorized_state,
            point_threshold=float(args.point_threshold),
        ),
        "provenance": {
            "head_annotation": str(args.head_annotation),
            "head_annotation_sha256": sha256_file(args.head_annotation),
            "detector_cache": str(args.detector_cache),
            "detector_cache_metadata": cache_metadata,
            "query_checkpoint": query_provenance,
            "ignored_head_ids": sorted(ignored_head_ids),
            "seed": None,
            "deterministic_evaluation": True,
        },
    }
    comparison = _comparison_rows(metrics)
    metrics["comparison"] = comparison
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "metrics.json", metrics)
    _write_jsonl(args.output_dir / "predictions.jsonl", rows)
    _write_markdown(
        args.output_dir / "comparison.md",
        metrics=metrics,
        comparison=comparison,
    )
    if args.common_output_dir is not None and args.max_images is None:
        common_predictions: dict[int, dict[str, Any]] = {}
        rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_image[int(row["image_id"])].append(row)
        for image in images:
            image_id = int(image["id"])
            detections = predictions_by_file_name.get(str(image["file_name"]), [])
            faces = []
            for detection in detections:
                raw_ellipse = detection.get("ellipse")
                if not isinstance(raw_ellipse, (list, tuple)) or len(raw_ellipse) != 5:
                    continue
                ellipse = canonicalize_ellipse(raw_ellipse)
                faces.append(
                    {
                        "bbox_xyxy": list(
                            map(
                                float,
                                detection.get(
                                    "bbox_xyxy",
                                    _ellipse_to_xyxy(ellipse),
                                ),
                            )
                        ),
                        "ellipse": ellipse,
                        "score": float(detection.get("score", 0.0)),
                    }
                )
            poses = []
            for row in rows_by_image.get(image_id, []):
                points = row.get("predicted_keypoints")
                predicted_ellipse = row.get("predicted_ellipse")
                if points is None or predicted_ellipse is None:
                    continue
                poses.append(
                    {
                        "bbox_xyxy": _ellipse_to_xyxy(predicted_ellipse),
                        "score": float(row.get("face_score", 0.0)),
                        "keypoints": points,
                        "point_scores": row.get("point_scores"),
                        "state_probabilities": row.get("state_probabilities"),
                    }
                )
            common_predictions[image_id] = {
                "heads": head_predictions_by_image.get(image_id, []),
                "faces": faces,
                "poses": poses,
            }
        uses_head_stage = (
            args.pipeline_contract == "three_stage_head_face_keypoint"
        )
        head_arguments = (
            args.head_predictions,
            args.head_detector_config,
            args.head_detector_checkpoint,
        )
        if uses_head_stage and any(value is None for value in head_arguments):
            raise RuntimeError(
                "The three-stage pipeline requires HEAD predictions, config, "
                "and checkpoint provenance"
            )
        if not uses_head_stage and any(value is not None for value in head_arguments):
            raise RuntimeError(
                "The two-stage FACE/keypoint pipeline must not receive an "
                "external HEAD detector"
            )

        head_runtime: Mapping[str, Any] = {}
        head_provenance: dict[str, Any] = {
            "head_stage": {
                "supported": False,
                "unsupported_reason": "declared_pipeline_has_no_head_stage",
            }
        }
        if uses_head_stage:
            assert args.head_predictions is not None
            assert args.head_detector_config is not None
            assert args.head_detector_checkpoint is not None
            head_metrics_path = args.head_predictions.with_name(
                "predictions_metrics.json"
            )
            head_metrics_payload = json.loads(
                head_metrics_path.read_text(encoding="utf-8")
            )
            if str(head_metrics_payload.get("split")) != args.split:
                raise RuntimeError("HEAD prediction metrics split mismatch")
            if int(head_metrics_payload.get("category_id", -1)) != 1:
                raise RuntimeError("HEAD prediction metrics must use category_id=1")
            if int(head_metrics_payload.get("max_dets", -1)) != 100:
                raise RuntimeError("HEAD prediction metrics must use max_dets=100")
            head_checkpoint_sha256 = sha256_file(args.head_detector_checkpoint)
            if (
                head_metrics_payload.get("checkpoint_sha256")
                != head_checkpoint_sha256
            ):
                raise RuntimeError(
                    "HEAD prediction checkpoint provenance mismatch"
                )
            head_runtime = head_metrics_payload.get("runtime", {})
            head_provenance = {
                "head_predictions": str(args.head_predictions),
                "head_predictions_sha256": sha256_file(args.head_predictions),
                "head_predictions_metrics": str(head_metrics_path),
                "head_predictions_metrics_sha256": sha256_file(
                    head_metrics_path
                ),
                "head_detector_config": str(args.head_detector_config),
                "head_detector_config_sha256": sha256_file(
                    args.head_detector_config
                ),
                "head_detector_checkpoint": str(
                    args.head_detector_checkpoint
                ),
                "head_detector_checkpoint_sha256": head_checkpoint_sha256,
            }
        attribute_elapsed = time.perf_counter() - evaluation_started
        head_elapsed = float(head_runtime.get("total_seconds", 0.0))
        total_elapsed = attribute_elapsed + head_elapsed
        head_inference = float(head_runtime.get("model_inference_seconds", 0.0))
        combined_inference = inference_seconds + head_inference
        continuous_local_contract = None
        if query_config.get("model", {}).get("continuous_recovery", {}).get("enabled", False):
            from keypoint_comparison1.landmark_detection.config import load_landmark_config

            baseline_config = load_landmark_config(
                REPO_ROOT / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_messenger_control.yaml"
            ).raw
            detector_fields = ("config", "checkpoint", "image_size", "maximum_detections", "crop_geometry")
            if any(query_config["detector"].get(key) != baseline_config["detector"].get(key) for key in detector_fields):
                raise RuntimeError("Continuous local FACE comparison requires the unchanged local-baseline detector")
            continuous_local_contract = {
                "checkpoint_sha256": sha256_file(Path(query_config["detector"]["checkpoint"])),
                "config_sha256": sha256_file(Path(query_config["detector"]["config"])),
                "prediction_cap": int(query_config["detector"]["maximum_detections"]),
            }
            if cache_metadata.get("detector_checkpoint_sha256") != continuous_local_contract["checkpoint_sha256"]:
                raise RuntimeError("FACE cache checkpoint hash does not match the local baseline")
        evaluate_normalized_predictions(
            annotation_path=args.head_annotation,
            predictions_by_image=common_predictions,
            split=args.split,
            architecture_id=args.architecture_id,
            comparison_stage=(
                "head_plus_deim_face_plus_query_refinement"
                if uses_head_stage
                else "full_image_deim_face_plus_query_refinement"
            ),
            output_dir=args.common_output_dir,
            provenance={
                "evaluator_path": str(
                    REPO_ROOT / "src/evaluation/baseline_aligned.py"
                ),
                "evaluator_sha256": sha256_file(
                    REPO_ROOT / "src/evaluation/baseline_aligned.py"
                ),
                "driver_path": str(Path(__file__).resolve()),
                "driver_sha256": sha256_file(Path(__file__).resolve()),
                "detector_cache": str(args.detector_cache),
                "detector_cache_metadata": cache_metadata,
                "query_checkpoint": query_provenance,
                "pipeline_contract": args.pipeline_contract,
                **head_provenance,
                "diagnostic_metrics": str(args.output_dir / "metrics.json"),
                "seed": query_config.get("train", {}).get("seed"),
                **({"continuous_local_face_contract": continuous_local_contract} if continuous_local_contract else {}),
            },
            runtime={
                "total_seconds": total_elapsed,
                "images_per_second": len(images)
                / max(total_elapsed, 1.0e-12),
                "model_inference_seconds": combined_inference,
                "model_inference_milliseconds_per_image": (
                    1000.0 * combined_inference / max(1, len(images))
                ),
                "peak_vram_mib": (
                    max(
                        torch.cuda.max_memory_allocated(cuda_device_index)
                        / (1024.0**2),
                        float(head_runtime.get("peak_vram_mib", 0.0)),
                    )
                    if cuda_device_index is not None
                    else float(head_runtime.get("peak_vram_mib", 0.0))
                ),
                "batch_size": int(args.batch_size),
                "input_size": int(data_config["input_size"]),
            },
            validation_selected_point_threshold=float(args.point_threshold),
            face_score_candidate_floor=float(
                cache_metadata["score_threshold"]
            ),
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
