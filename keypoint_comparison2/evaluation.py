from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .config import ComparisonConfig
from .inference import (
    DEIMv2LFaceDetector,
    JointFaceKeypointPipeline,
    TopDownFaceKeypointPipeline,
)
from .metrics import KeypointMetrics
from .training import best_checkpoint_path, load_trained_model
from src.wandb_utils import sync_evaluation_to_wandb


def _xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray((x, y, x + width, y + height), dtype=np.float32)


def _iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(-1)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0, None).prod(-1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0, None).prod(-1)
    return intersection / np.clip(left_area[:, None] + right_area[None, :] - intersection, 1e-8, None)


def _match(
    gt_boxes: list[np.ndarray],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not gt_boxes or not predictions:
        return [], list(range(len(gt_boxes))), list(range(len(predictions)))
    gt = np.stack(gt_boxes)
    predicted = np.asarray([row["bbox_xyxy"] for row in predictions], dtype=np.float32)
    overlaps = _iou(gt, predicted)
    gt_indices, prediction_indices = linear_sum_assignment(1.0 - overlaps)
    matched = [
        (int(gt_index), int(prediction_index))
        for gt_index, prediction_index in zip(gt_indices, prediction_indices)
        if overlaps[gt_index, prediction_index] >= threshold
    ]
    matched_gt = {left for left, _ in matched}
    matched_predictions = {right for _, right in matched}
    return (
        matched,
        [index for index in range(len(gt_boxes)) if index not in matched_gt],
        [index for index in range(len(predictions)) if index not in matched_predictions],
    )


def evaluate_model(
    config: ComparisonConfig,
    model_name: str,
    *,
    device: torch.device,
    checkpoint: Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError(f"Evaluation split must be val or test, got {split!r}")
    elapsed_started = time.perf_counter()
    raw = config.raw
    model = load_trained_model(config, model_name, device=device, checkpoint=checkpoint)
    if model_name == "deimv2_joint":
        pipeline: Any = JointFaceKeypointPipeline(
            model,
            input_size=int(raw["models"][model_name]["input_size"]),
            device=device,
            score_threshold=float(raw["detector"]["score_threshold"]),
            visibility_threshold=float(raw["evaluation"]["visibility_threshold"]),
        )
    else:
        detector = DEIMv2LFaceDetector(raw["detector"], device=device)
        pipeline = TopDownFaceKeypointPipeline(
            detector,
            model,
            input_size=int(raw["models"][model_name]["input_size"]),
            crop_scale=float(raw["data"]["augmentation"]["scale_factor"]),
            batch_size=int(raw["evaluation"]["batch_size"]),
            device=device,
            flip_test=bool(raw["models"][model_name].get("test_time_flip", True)),
        )
    annotation_path = (
        config.project_root
        / "outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations"
        / "face_dino_v1"
        / f"{split}.json"
    )
    image_root = (
        config.project_root
        / "outputs/dataset_cache/face_pose_drive_v2_balanced_811"
    )
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    keypoint_metrics = KeypointMetrics(tuple(float(value) for value in raw["evaluation"]["pck_thresholds"]))
    natural_visible_metrics = KeypointMetrics(
        tuple(float(value) for value in raw["evaluation"]["pck_thresholds"])
    )
    natural_occluded_metrics = KeypointMetrics(
        tuple(float(value) for value in raw["evaluation"]["pck_thresholds"])
    )
    true_positive = false_positive = false_negative = 0
    matched_visible_points = total_visible_points = 0
    output_dir = config.output_root / model_name / "evaluation" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / "predictions.jsonl"
    inference_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with prediction_file.open("w", encoding="utf-8") as handle:
        for image_info in coco.get("images", []):
            path = image_root / str(image_info["file_name"])
            if not path.is_file():
                path = image_root / "images" / str(image_info["file_name"])
            image = Image.open(path).convert("RGB")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            predictions = pipeline(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started
            ground_truth = annotations_by_image.get(int(image_info["id"]), [])
            boxes = [_xywh_to_xyxy(row["bbox"]) for row in ground_truth]
            matched, missed, extra = _match(
                boxes,
                predictions,
                float(raw["detector"]["matching_iou_threshold"]),
            )
            true_positive += len(matched)
            false_negative += len(missed)
            false_positive += len(extra)
            for gt_index, prediction_index in matched:
                annotation = ground_truth[gt_index]
                raw_points = np.asarray(
                    annotation.get("keypoints", [[0.0, 0.0]] * 5),
                    dtype=np.float32,
                )
                if raw_points.shape == (5, 2):
                    states = np.asarray(
                        annotation.get("keypoint_states", [0] * 5),
                        dtype=np.int64,
                    )
                    target_values = raw_points
                else:
                    values = raw_points.reshape(5, 3)
                    target_values = values[:, :2]
                    states = values[:, 2].astype(np.int64)
                labeled = torch.from_numpy(states > 0).unsqueeze(0)
                visible = torch.from_numpy(states > 1).unsqueeze(0)
                target = torch.from_numpy(target_values).unsqueeze(0)
                prediction = torch.tensor(predictions[prediction_index]["keypoints"]).unsqueeze(0)
                visibility_probability = torch.tensor(predictions[prediction_index]["visibility"]).unsqueeze(0)
                box = boxes[gt_index]
                normalization = torch.tensor([max(box[2] - box[0], box[3] - box[1])])
                keypoint_metrics.update(
                    prediction,
                    target,
                    labeled,
                    visibility_probability=visibility_probability,
                    visibility_target=visible,
                    normalization=normalization,
                )
                natural_visible_metrics.update(
                    prediction,
                    target,
                    visible,
                    normalization=normalization,
                )
                natural_occluded_metrics.update(
                    prediction,
                    target,
                    labeled & ~visible,
                    normalization=normalization,
                )
                matched_visible_points += int(labeled.sum())
            total_visible_points += sum(
                int(
                    (
                        np.asarray(row.get("keypoint_states", [0] * 5)) > 0
                        if np.asarray(row.get("keypoints", [])).shape == (5, 2)
                        else np.asarray(row.get("keypoints", [0] * 15))
                        .reshape(5, 3)[:, 2]
                        > 0
                    ).sum()
                )
                for row in ground_truth
            )
            row = {
                "image_id": int(image_info["id"]),
                "file_name": str(image_info["file_name"]),
                "predictions": predictions,
                "matched": matched,
                "missed_gt": missed,
                "false_positive_predictions": extra,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metrics: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "checkpoint": str(checkpoint or best_checkpoint_path(config, model_name)),
        "detector_precision": precision,
        "detector_recall": recall,
        "detector_f1": 2 * precision * recall / max(1e-8, precision + recall),
        "keypoint_coverage": matched_visible_points / max(1, total_visible_points),
        **keypoint_metrics.compute(),
    }
    metrics.update(
        {
            f"natural_visible_{key}": value
            for key, value in natural_visible_metrics.compute().items()
        }
    )
    metrics.update(
        {
            f"natural_occluded_{key}": value
            for key, value in natural_occluded_metrics.compute().items()
        }
    )
    for threshold in raw["evaluation"]["pck_thresholds"]:
        name = f"pck_{float(threshold):.2f}"
        metrics[f"end_to_end_{name}"] = metrics.get(name, float("nan")) * metrics["keypoint_coverage"]
    elapsed = time.perf_counter() - elapsed_started
    image_count = len(coco.get("images", []))
    metrics["runtime"] = {
        "total_seconds": elapsed,
        "images_per_second": image_count / max(elapsed, 1.0e-12),
        "milliseconds_per_image": elapsed * 1000.0 / max(1, image_count),
        "model_inference_seconds": inference_seconds,
        "model_inference_milliseconds_per_image": (
            inference_seconds * 1000.0 / max(1, image_count)
        ),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "batch_size": 1,
        "input_size": int(raw["models"][model_name]["input_size"]),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    baseline_dir = output_dir / "baseline_aligned"
    checkpoint_path = checkpoint or best_checkpoint_path(config, model_name)
    subprocess.run(
        [
            sys.executable,
            str(config.project_root / "tools/evaluate_baseline_aligned_predictions.py"),
            "--format",
            "keypoint_comparison2",
            "--predictions",
            str(prediction_file),
            "--annotations",
            str(annotation_path),
            "--split",
            split,
            "--architecture-id",
            f"keypoint_comparison2_{model_name}",
            "--comparison-stage",
            "joint",
            "--output-dir",
            str(baseline_dir),
            "--source-metrics",
            str(metrics_path),
            "--config",
            str(config.path),
            "--checkpoint",
            str(checkpoint_path),
            "--seed",
            str(int(raw.get("train", {}).get("seed", 42))),
        ],
        cwd=config.project_root,
        check=True,
    )
    wandb_cfg = raw.get("train", {}).get("wandb", {})
    if isinstance(wandb_cfg, dict) and bool(wandb_cfg.get("enabled", False)):
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is required to sync keypoint comparison evaluation"
            ) from error
        sync_evaluation_to_wandb(
            wandb,
            state_path=config.output_root / model_name / "wandb_run.json",
            split=split,
            metric_sources={
                "deployment": metrics_path,
                "baseline_aligned": baseline_dir / "metrics.json",
            },
            artifact_files=(
                prediction_file,
                baseline_dir / "comparison_table.md",
                baseline_dir / "success.json",
            ),
            headline={
                "detector_f1": metrics["detector_f1"],
                "keypoint_coverage": metrics["keypoint_coverage"],
                "nme": metrics.get("nme", float("nan")),
                "pck_0_05": metrics.get("pck_0.05", float("nan")),
                "pck_0_10": metrics.get("pck_0.10", float("nan")),
            },
            report_path=output_dir / "wandb_evaluation_sync.json",
        )
    return metrics
