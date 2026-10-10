from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.face_crop_cache import crop_geometry_from_detection

from .dataset import TopDownFaceLandmarkDataset, _parse_keypoints, collate_landmark_batch, image_to_tensor, xywh_to_xyxy
from .detector import DeimFaceDetector
from .face_overlap import CONTEXT_SCHEMA, crop_face_context, face_context_kwargs
from .diagnostics import attention_diagnostics
from .geometry import build_crop_transform, normalized_crop_to_original, warp_crop
from .metrics import LandmarkMetricAccumulator, iou_xyxy, match_detector_boxes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _target_numpy(target: dict[str, Any], key: str) -> np.ndarray:
    value = target[key]
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _dataset_attribute(dataset: Any, name: str, default: Any = None) -> Any:
    """Read metadata from a dataset, including through nested Subset wrappers."""
    current = dataset
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "dataset", None)
    return default


def predictions_to_original(
    normalized_coordinates: torch.Tensor,
    inverse_affine: torch.Tensor,
    *,
    input_size: int,
) -> np.ndarray:
    coordinates = normalized_coordinates.detach().float().cpu().numpy()
    matrices = inverse_affine.detach().float().cpu().numpy()
    result = []
    for points, matrix in zip(coordinates, matrices, strict=True):
        crop_points = points * float(input_size - 1)
        homogeneous = np.concatenate([crop_points, np.ones((crop_points.shape[0], 1), dtype=np.float32)], axis=1)
        result.append(homogeneous @ matrix.T)
    return np.asarray(result, dtype=np.float32)


def _merge_attention(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    self_values = np.asarray([item["self_attention_layer_mean"] for item in summaries], dtype=np.float64)
    cross_values = np.asarray([item["cross_attention_normalized_entropy"] for item in summaries], dtype=np.float64)
    return {
        "self_attention_layer_mean": self_values.mean(axis=0).tolist(),
        "cross_attention_normalized_entropy": cross_values.mean(axis=0).tolist(),
        "diagnostic_batches": len(summaries),
    }


def _state_probabilities(
    output: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray]:
    visibility = (
        output["visibility_logits"].sigmoid().detach().float().cpu().numpy()
    )
    if "presence_logits" not in output:
        return None, visibility
    presence = (
        output["presence_logits"].sigmoid().detach().float().cpu().numpy()
    )
    return presence, visibility


@torch.inference_mode()
def evaluate_gt_boxes(
    model: torch.nn.Module,
    dataset: TopDownFaceLandmarkDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    input_size: int,
    attention_batches: int = 4,
    recovery_mode: str = "recovered",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_landmark_batch,
    )
    accumulator = LandmarkMetricAccumulator()
    attention_rows: list[dict[str, Any]] = []
    model.eval()
    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(
                images,
                return_attention=batch_index < attention_batches,
                recovery_mode=recovery_mode,
                **face_context_kwargs(model, targets, device),
            )
        decoded = output["decoded_coordinates"]
        inverse = torch.stack([target["inverse_affine"] for target in targets])
        original_predictions = predictions_to_original(decoded, inverse, input_size=input_size)
        presence_scores, visibility_scores = _state_probabilities(output)
        state_probabilities = (
            output["state_probabilities"].detach().float().cpu().numpy()
            if "state_probabilities" in output
            else [None] * len(targets)
        )
        presence_rows = (
            presence_scores
            if presence_scores is not None
            else [None] * len(targets)
        )
        for prediction, presence, visibility, state, target in zip(
            original_predictions,
            presence_rows,
            visibility_scores,
            state_probabilities,
            targets,
            strict=True,
        ):
            accumulator.add_prediction(
                predicted_keypoints=prediction,
                gt_keypoints=_target_numpy(target, "original_keypoints"),
                labeled=_target_numpy(target, "original_labeled"),
                visible=_target_numpy(target, "original_visible"),
                bbox_xyxy=_target_numpy(target, "bbox_xyxy"),
                image_id=int(target["image_id"]),
                annotation_id=int(target["annotation_id"]),
                source_video_id=str(target.get("source_video_id", "")),
                file_name=str(target.get("file_name", "")),
                landmark_scores=visibility,
                presence_scores=presence,
                state_probabilities=state,
            )
        diagnostic = attention_diagnostics(output)
        if diagnostic is not None:
            attention_rows.append(diagnostic)
    metrics = accumulator.compute()
    detector_crop_report = _dataset_attribute(dataset, "detector_crop_report")
    metrics.update(
        {
            "bbox_mode": (
                "predicted_face_crop"
                if detector_crop_report is not None
                else "gt"
            ),
            "detector_crop_report": detector_crop_report,
            "attention": _merge_attention(attention_rows),
            "recovery_mode": recovery_mode,
        }
    )
    return metrics, accumulator.rows


def load_or_create_detector_predictions(
    *,
    detector: DeimFaceDetector,
    annotation_path: Path,
    image_root: Path,
    cache_path: Path,
    force: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    annotation_hash = sha256_file(annotation_path)
    checkpoint_hash = sha256_file(detector.checkpoint_path)
    config_hash = sha256_file(detector.config_path)
    expected_metadata = {
        "annotation_sha256": annotation_hash,
        "detector_config_sha256": config_hash,
        "detector_checkpoint_sha256": checkpoint_hash,
        "score_threshold": detector.score_threshold,
        "image_size": detector.image_size,
        "maximum_detections": detector.maximum_detections,
        "crop_geometry": getattr(detector, "crop_geometry", "bbox"),
    }
    if getattr(detector, "return_face_query_features", False):
        expected_metadata["face_context_schema"] = CONTEXT_SCHEMA
    if cache_path.is_file() and not force:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = value.get("metadata", {})
        if all(metadata.get(key) == expected for key, expected in expected_metadata.items()):
            return {int(key): list(rows) for key, rows in value["predictions"].items()}
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    predictions: dict[int, list[dict[str, Any]]] = {}
    for image_info in coco.get("images", []):
        image_id = int(image_info["id"])
        image_path = image_root / str(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        predictions[image_id] = detector.predict(image)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            **expected_metadata,
            "annotation": str(annotation_path),
            "annotation_sha256": annotation_hash,
            "detector_config": str(detector.config_path),
            "detector_config_sha256": config_hash,
            "detector_checkpoint": str(detector.checkpoint_path),
            "detector_checkpoint_sha256": checkpoint_hash,
            "score_threshold": detector.score_threshold,
            "image_size": detector.image_size,
            "maximum_detections": detector.maximum_detections,
            "crop_geometry": getattr(detector, "crop_geometry", "bbox"),
        },
        "predictions": {str(key): value for key, value in predictions.items()},
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return predictions


def _flush_predicted_batch(
    *,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    images: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    accumulator: LandmarkMetricAccumulator,
    return_attention: bool,
) -> dict[str, Any] | None:
    if not images:
        return None
    batch = torch.stack(images).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(batch, return_attention=return_attention,
                       **face_context_kwargs(model, metadata, device))
    coordinates = output["decoded_coordinates"].detach().float().cpu().numpy()
    presence_scores, visibility_scores = _state_probabilities(output)
    state_probabilities = (
        output["state_probabilities"].detach().float().cpu().numpy()
        if "state_probabilities" in output
        else [None] * len(metadata)
    )
    presence_rows = (
        presence_scores
        if presence_scores is not None
        else [None] * len(metadata)
    )
    for prediction, presence, visibility, state, item in zip(
        coordinates,
        presence_rows,
        visibility_scores,
        state_probabilities,
        metadata,
        strict=True,
    ):
        original = normalized_crop_to_original(prediction, item["transform"])
        accumulator.add_prediction(
            predicted_keypoints=original,
            gt_keypoints=item["gt_keypoints"],
            labeled=item["labeled"],
            visible=item["visible"],
            bbox_xyxy=item["gt_bbox"],
            image_id=item["image_id"],
            annotation_id=item["annotation_id"],
            source_video_id=item["source_video_id"],
            file_name=item["file_name"],
            landmark_scores=visibility,
            presence_scores=presence,
            state_probabilities=state,
            detector_iou=item["detector_iou"],
            detector_bbox_xyxy=item["detector_bbox_xyxy"],
            detector_score=item["detector_score"],
        )
    return attention_diagnostics(output)


def evaluate_deim_boxes(
    model: torch.nn.Module,
    *,
    annotation_path: Path,
    image_root: Path,
    detector_predictions: dict[int, list[dict[str, Any]]],
    device: torch.device,
    batch_size: int,
    input_size: int,
    crop_scale_factor: float,
    mean: Sequence[float],
    std: Sequence[float],
    matching_iou_threshold: float = 0.50,
    attention_batches: int = 4,
    category_ids: Sequence[int] | None = None,
    include_missing_keypoint_instances: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    allowed_categories = {int(value) for value in category_ids} if category_ids is not None else None
    for annotation in coco.get("annotations", []):
        if allowed_categories is not None and int(annotation.get("category_id", -1)) not in allowed_categories:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    accumulator = LandmarkMetricAccumulator()
    model.eval()
    crop_images: list[torch.Tensor] = []
    crop_metadata: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    flushed_batches = 0

    def flush() -> None:
        nonlocal flushed_batches
        diagnostic = _flush_predicted_batch(
            model=model,
            device=device,
            input_size=input_size,
            images=crop_images,
            metadata=crop_metadata,
            accumulator=accumulator,
            return_attention=flushed_batches < attention_batches,
        )
        if diagnostic is not None:
            attention_rows.append(diagnostic)
        crop_images.clear()
        crop_metadata.clear()
        flushed_batches += 1

    for image_info in coco.get("images", []):
        image_id = int(image_info["id"])
        all_annotations = annotations_by_image.get(image_id, [])
        annotations: list[dict[str, Any]] = []
        ignored_boxes: list[list[float]] = []
        for annotation in all_annotations:
            _keypoints, annotation_labeled, _annotation_visible = (
                _parse_keypoints(annotation.get("keypoints", []))
            )
            if (
                include_missing_keypoint_instances
                or bool(annotation_labeled.any())
            ):
                annotations.append(annotation)
            else:
                ignored_boxes.append(xywh_to_xyxy(annotation["bbox"]))
        gt_boxes = [xywh_to_xyxy(annotation["bbox"]) for annotation in annotations]
        detections = detector_predictions.get(image_id, [])
        matches, missed, false_positives = match_detector_boxes(
            gt_boxes,
            detections,
            iou_threshold=matching_iou_threshold,
        )
        # In legacy mode, faces without labeled landmarks remain ignored
        # regions. The factorized state-head protocol includes them as
        # missing-state supervision and evaluation instances.
        false_positives = [
            index
            for index in false_positives
            if not any(
                iou_xyxy(detections[index]["bbox_xyxy"], ignored) >= matching_iou_threshold
                for ignored in ignored_boxes
            )
        ]
        accumulator.detector_false_positives += len(false_positives)
        source_video_id = str(image_info.get("metadata", {}).get("source_video_id", ""))
        for gt_index in missed:
            annotation = annotations[gt_index]
            gt_keypoints, labeled, visible = _parse_keypoints(
                annotation.get("keypoints", [])
            )
            accumulator.add_missed(
                gt_keypoints=gt_keypoints,
                labeled=labeled,
                visible=visible,
                bbox_xyxy=gt_boxes[gt_index],
                image_id=image_id,
                annotation_id=int(annotation.get("id", gt_index)),
                source_video_id=source_video_id,
                file_name=str(image_info["file_name"]),
            )
        if not matches:
            continue
        image = np.asarray(Image.open(image_root / str(image_info["file_name"])).convert("RGB"))
        for gt_index, detection_index, overlap in matches:
            annotation = annotations[gt_index]
            detection_box = detections[detection_index]["bbox_xyxy"]
            crop_box, crop_rotation = crop_geometry_from_detection(
                detections[detection_index],
                require_ellipse=(
                    detections[detection_index].get("crop_geometry")
                    == "ellipse"
                ),
            )
            transform = build_crop_transform(
                crop_box,
                output_size=input_size,
                scale_factor=crop_scale_factor,
                rotation_deg=crop_rotation,
                warp_bbox_aspect=False,
            )
            crop = warp_crop(image, transform)
            gt_keypoints, labeled, visible = _parse_keypoints(
                annotation.get("keypoints", [])
            )
            crop_images.append(image_to_tensor(crop, mean=mean, std=std))
            crop_metadata.append(
                {
                    "transform": transform,
                    "gt_keypoints": gt_keypoints,
                    "labeled": labeled,
                    "visible": visible,
                    "gt_bbox": gt_boxes[gt_index],
                    "image_id": image_id,
                    "annotation_id": int(annotation.get("id", gt_index)),
                    "source_video_id": source_video_id,
                    "file_name": str(image_info["file_name"]),
                    "detector_iou": overlap,
                    "detector_bbox_xyxy": [float(value) for value in detection_box],
                    "detector_score": float(detections[detection_index].get("score", 0.0)),
                    **(crop_face_context(detections[detection_index], transform)
                       if getattr(model, "face_overlap_enabled", False) else {}),
                }
            )
            if len(crop_images) >= batch_size:
                flush()
    flush()
    metrics = accumulator.compute()
    matched_ious = [row.get("detector_iou") for row in accumulator.rows if row.get("detector_iou") is not None]
    metrics.update(
        {
            "bbox_mode": "deim",
            "matching_iou_threshold": matching_iou_threshold,
            "mean_matched_detector_iou": float(np.mean(matched_ious)) if matched_ious else None,
            "attention": _merge_attention(attention_rows),
        }
    )
    return metrics, accumulator.rows
