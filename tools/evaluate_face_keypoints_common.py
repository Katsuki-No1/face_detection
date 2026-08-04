from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison1.landmark_detection.config import (  # noqa: E402
    KEYPOINT_TARGET_POLICY,
    load_landmark_config,
)
from keypoint_comparison1.landmark_detection.model import (  # noqa: E402
    build_landmark_model,
)
from keypoint_comparison2.common_evaluation import (  # noqa: E402
    CommonKeypointMetrics,
    bbox_iou,
    greedy_bbox_matches,
    normalized_to_original,
)
from keypoint_comparison2.config import (  # noqa: E402
    data_config_for_model,
    load_config as load_comparison2_config,
)
from keypoint_comparison2.data import (  # noqa: E402
    CropAugmentation,
    FaceCropDataset,
    collate_crops,
)
from keypoint_comparison2.training import (  # noqa: E402
    best_checkpoint_path,
    load_trained_model,
)
from scripts_YOLO.evaluate_yolo_pose_metrics import (  # noqa: E402
    result_predictions as yolo_result_predictions,
)
from scripts_YOLO.yolo26_pose.inference import (  # noqa: E402
    iter_batched_image_predictions,
)


LANDMARK_MODELS = (
    "vitpose_376",
    "query",
    "query_refinement",
)
TOPDOWN_MODELS = (
    "rtmpose_face5",
    "dinov3_fpn_rtmcc",
    "dinov3_orformer_rtmcc",
    *LANDMARK_MODELS,
)
ALL_MODELS = (*TOPDOWN_MODELS, "yolo26m_pose")
DISPLAY_NAMES = {
    "rtmpose_face5": "RTMPose Face5",
    "dinov3_fpn_rtmcc": "DINOv3-FPN-RTMCC",
    "dinov3_orformer_rtmcc": "DINOv3-ORFormer-RTMCC",
    "vitpose_376": "DINOv3 ViTPose-376",
    "query": "DINOv3 Query",
    "query_refinement": "DINOv3 Query Refinement",
    "yolo26m_pose": "YOLO26m Pose (legacy v2)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate face five-keypoint models with one GT-bbox-diagonal "
            "metric contract."
        )
    )
    parser.add_argument(
        "--comparison2-config",
        type=Path,
        default=REPO_ROOT / "keypoint_comparison2/configs/production.yaml",
    )
    parser.add_argument(
        "--query-config",
        type=Path,
        default=(
            REPO_ROOT
            / "keypoint_comparison1/configs/"
            "dinov3_vits16_face_pose_drive_v2.yaml"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*ALL_MODELS, "all"),
        default=["all"],
    )
    parser.add_argument(
        "--comparison1-output-root",
        type=Path,
        default=None,
        help=(
            "Override the checkpoint/training-summary root for ViTPose-376 "
            "and Query-family models."
        ),
    )
    parser.add_argument(
        "--reuse-results-from",
        type=Path,
        default=None,
        help=(
            "Reuse matching model rows from an existing comparison.json; "
            "protocol compatibility is checked before reuse."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("val", "test", "external_v1_v2"),
        default="val",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/yolo/yolo26_pose/face_pose_drive_v2/"
            "yolo26m_pose_full/weights/best.pt"
        ),
    )
    parser.add_argument("--yolo-input-size", type=int, default=960)
    parser.add_argument("--yolo-score-threshold", type=float, default=0.38)
    parser.add_argument("--matching-iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/benchmarks/"
            "face_keypoint_common_gt_bbox_diagonal_val_20260728"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(image_root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (
        (relative,)
        if relative.is_absolute()
        else (image_root / relative, image_root / "images" / relative)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve {file_name!r} below {image_root}"
    )


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def model_parameters(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reusable_results(
    source: Path | None,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if source is None:
        return {}
    comparison_path = (
        source / "comparison.json" if source.is_dir() else source
    ).resolve()
    payload = read_json(comparison_path)
    source_protocol = payload.get("protocol")
    if not isinstance(source_protocol, Mapping):
        raise TypeError(
            f"Reusable comparison has no protocol mapping: {comparison_path}"
        )
    compatibility_keys = (
        "split",
        "annotation_sha256",
        "coordinate_mask",
        "visibility_target",
        "visibility_evaluation_mask",
        "normalization",
        "pck_thresholds",
        "topdown_crop",
        "test_time_flip",
        "inference_precision",
    )
    mismatches = [
        key
        for key in compatibility_keys
        if source_protocol.get(key) != protocol.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Reusable comparison protocol mismatch for "
            f"{mismatches}: {comparison_path}"
        )
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise TypeError(
            f"Reusable comparison has no results list: {comparison_path}"
        )
    return {
        str(row["model_key"]): {
            **row,
            "reused_from": str(comparison_path),
        }
        for row in rows
        if isinstance(row, Mapping) and row.get("model_key")
    }


def common_augmentation(raw: Mapping[str, Any]) -> CropAugmentation:
    values = dict(raw)
    for key in ("occlusion_area", "occlusion_count"):
        if key in values:
            values[key] = tuple(values[key])
    return CropAugmentation(**values)


def load_landmark_model(
    raw: Mapping[str, Any],
    *,
    variant: str,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, _ = build_landmark_model(
        raw,
        variant=variant,
        initialize_from_detector=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("ema", state.get("model"))
    if not isinstance(weights, Mapping):
        raise TypeError(f"Landmark checkpoint has no state dict: {checkpoint}")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    summary_path = output_root / variant / "training_summary.json"
    summary = read_json(summary_path)
    return model, {
        "checkpoint_epoch": int(state["epoch"]),
        "completed_epoch": int(summary["last_epoch"]),
        "training_dataset": str(raw["data"]["dataset_version"]),
        "keypoint_visibility_training_contract": (
            "factorized_presence_and_conditional_visibility"
            if state.get("keypoint_target_policy")
            == KEYPOINT_TARGET_POLICY
            else "legacy_v_gt_0_collapsed"
        ),
        "state_head_mode": state.get("state_head_mode", "legacy_binary"),
    }


def load_topdown_model(
    model_name: str,
    *,
    comparison2: Any,
    query: Any,
    comparison1_output_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int, Path, dict[str, Any]]:
    if model_name in LANDMARK_MODELS:
        checkpoint = (
            comparison1_output_root
            / model_name
            / "checkpoints/best_nme.pt"
        )
        model, training = load_landmark_model(
            query.raw,
            variant=model_name,
            checkpoint=checkpoint,
            output_root=comparison1_output_root,
            device=device,
        )
        return model, int(query.raw["data"]["input_size"]), checkpoint, training

    checkpoint = best_checkpoint_path(comparison2, model_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_trained_model(
        comparison2,
        model_name,
        device=device,
        checkpoint=checkpoint,
    )
    completion = read_json(
        comparison2.output_root / model_name / "training_complete.json"
    )
    return (
        model,
        int(comparison2.raw["models"][model_name]["input_size"]),
        checkpoint,
        {
            "checkpoint_epoch": int(state["epoch"]) + 1,
            "completed_epoch": int(completion["completed_epoch"]) + 1,
            "training_dataset": str(
                comparison2.raw["data"]["dataset_version"]
            ),
        },
    )


def topdown_dataset(
    comparison2: Any,
    *,
    annotation_path: Path,
    image_root: Path,
    input_size: int,
) -> FaceCropDataset:
    return FaceCropDataset(
        annotation=annotation_path,
        image_root=image_root,
        input_size=input_size,
        training=False,
        category_ids=(2,),
        augmentation=common_augmentation(
            comparison2.raw["data"]["augmentation"]
        ),
    )


@torch.inference_mode()
def evaluate_topdown(
    model_name: str,
    *,
    comparison2: Any,
    query: Any,
    comparison1_output_root: Path,
    annotation_path: Path,
    image_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    model, input_size, checkpoint, training = load_topdown_model(
        model_name,
        comparison2=comparison2,
        query=query,
        comparison1_output_root=comparison1_output_root,
        device=device,
    )
    dataset = topdown_dataset(
        comparison2,
        annotation_path=annotation_path,
        image_root=image_root,
        input_size=input_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_crops,
        drop_last=False,
    )
    metrics = CommonKeypointMetrics()
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{model_name}.jsonl"
    query_mean = torch.tensor(
        query.raw["data"].get("mean", (0.0, 0.0, 0.0)),
        device=device,
    ).view(1, 3, 1, 1)
    query_std = torch.tensor(
        query.raw["data"].get("std", (1.0, 1.0, 1.0)),
        device=device,
    ).view(1, 3, 1, 1)

    with prediction_path.open("w", encoding="utf-8") as prediction_stream:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if model_name in LANDMARK_MODELS:
                images = (
                    images - query_mean.to(dtype=images.dtype)
                ) / query_std.to(dtype=images.dtype)
                output = model(images)
                predicted = output["decoded_coordinates"]
            else:
                output = model(images)
                predicted = output["coordinates"]
            visibility_probability = output["visibility_logits"].sigmoid()
            inverse = torch.stack(
                [row["inverse_affine"] for row in batch["metadata"]]
            )
            prediction_original = normalized_to_original(
                predicted.detach().cpu(),
                inverse,
                input_size=input_size,
            )
            target_original = normalized_to_original(
                batch["coordinates"],
                inverse,
                input_size=input_size,
            )
            boxes = torch.stack(
                [row["gt_bbox_xyxy"] for row in batch["metadata"]]
            )
            metrics.register_ground_truth(batch["labeled"])
            metric_rows = metrics.update(
                prediction_original,
                target_original,
                batch["labeled"],
                batch["visible"],
                visibility_probability.detach().cpu(),
                boxes,
            )
            for index, row in enumerate(batch["metadata"]):
                prediction_stream.write(
                    json.dumps(
                        {
                            "annotation_id": int(row["annotation_id"]),
                            "image_id": int(row["image_id"]),
                            "file_name": str(row["file_name"]),
                            "bbox_xyxy": boxes[index].tolist(),
                            "predicted_keypoints": (
                                prediction_original[index].tolist()
                            ),
                            "target_keypoints": target_original[index].tolist(),
                            "labeled": batch["labeled"][index].tolist(),
                            "visible": batch["visible"][index].tolist(),
                            "visibility_probability": (
                                visibility_probability[index]
                                .detach()
                                .cpu()
                                .float()
                                .tolist()
                            ),
                            **metric_rows[index],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    result = {
        "model_key": model_name,
        "model": DISPLAY_NAMES[model_name],
        "family": "top_down",
        "input_mode": "GT ellipse long-axis square crop",
        "input_size": input_size,
        "parameters": model_parameters(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_file": str(prediction_path),
        "detector_precision": None,
        "detector_recall": None,
        "detector_f1": None,
        **training,
        **metrics.compute(),
    }
    del loader, dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def coco_ground_truth(
    annotation_path: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
]:
    coco = read_json(annotation_path)
    images = {int(row["id"]): row for row in coco.get("images", [])}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in coco.get("annotations", []):
        if int(row.get("category_id", -1)) == 2:
            annotations[int(row["image_id"])].append(row)
    return images, annotations


def yolo_training_epochs(weights: Path) -> dict[str, int | None]:
    run_dir = weights.parent.parent
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {"checkpoint_epoch": None, "completed_epoch": None}
    with results_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    completed = int(float(rows[-1]["epoch"])) if rows else None
    return {
        # Ultralytics strips optimizer/epoch metadata from best.pt.
        "checkpoint_epoch": None,
        "completed_epoch": completed,
    }


def evaluation_image_hashes(
    annotation_path: Path,
    image_root: Path,
) -> set[str]:
    coco = read_json(annotation_path)
    hashes: set[str] = set()
    for row in coco.get("images", []):
        metadata_hash = str(
            row.get("metadata", {}).get("sha256", "")
        ).strip()
        if len(metadata_hash) == 64:
            hashes.add(metadata_hash)
        else:
            hashes.add(
                sha256_file(
                    resolve_image(image_root, str(row["file_name"]))
                )
            )
    return hashes


def directory_image_hashes(path: Path) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {path}")
    return {
        sha256_file(image_path)
        for image_path in path.iterdir()
        if image_path.is_file()
    }


def leakage_audit(
    annotation_path: Path,
    image_root: Path,
) -> dict[str, Any]:
    evaluation_hashes = evaluation_image_hashes(annotation_path, image_root)
    training_directories = {
        "balanced_811_train": (
            REPO_ROOT
            / "outputs/dataset_cache/face_pose_current/images/train"
        ),
        "legacy_yolo_v2_train": (
            REPO_ROOT / "datasets/face_pose_drive_v2/images/train"
        ),
    }
    overlaps: dict[str, int] = {}
    for name, directory in training_directories.items():
        overlaps[name] = len(
            evaluation_hashes & directory_image_hashes(directory)
        )
    return {
        "evaluation_images": len(evaluation_hashes),
        "hash_algorithm": "sha256",
        "training_image_overlap": overlaps,
    }


def evaluate_yolo(
    *,
    annotation_path: Path,
    image_root: Path,
    device: torch.device,
    weights: Path,
    input_size: int,
    batch_size: int,
    score_threshold: float,
    matching_iou_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO26m evaluation requires ultralytics") from error

    images_by_id, annotations_by_image = coco_ground_truth(annotation_path)
    image_rows = sorted(
        images_by_id.values(),
        key=lambda row: str(
            resolve_image(image_root, str(row["file_name"]))
        ),
    )
    image_paths = [
        resolve_image(image_root, str(row["file_name"]))
        for row in image_rows
    ]
    image_id_by_path = {
        resolve_image(image_root, str(row["file_name"])): int(row["id"])
        for row in image_rows
    }
    model = YOLO(str(weights), task="pose")
    parameter_count = model_parameters(model.model)
    metrics = CommonKeypointMetrics()
    true_positive = false_positive = false_negative = 0
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "yolo26m_pose.jsonl"

    predictions = iter_batched_image_predictions(
        model,
        image_paths,
        imgsz=input_size,
        conf=0.001,
        device=str(device),
        batch_size=batch_size,
        max_det=100,
    )
    with prediction_path.open("w", encoding="utf-8") as prediction_stream:
        for image_path, raw_result in predictions:
            image_id = image_id_by_path[image_path.resolve()]
            all_annotations = annotations_by_image.get(image_id, [])
            eligible: list[dict[str, Any]] = []
            ignored_boxes: list[list[float]] = []
            for annotation in all_annotations:
                values = np.asarray(
                    annotation.get("keypoints", [0.0] * 15),
                    dtype=np.float64,
                ).reshape(5, 3)
                if bool((values[:, 2] > 0).any()):
                    eligible.append(annotation)
                else:
                    ignored_boxes.append(xywh_to_xyxy(annotation["bbox"]))
            for annotation in eligible:
                values = np.asarray(
                    annotation["keypoints"],
                    dtype=np.float64,
                ).reshape(5, 3)
                metrics.register_ground_truth(values[:, 2] > 0)

            selected_predictions = [
                row
                for row in yolo_result_predictions(raw_result)
                if float(row["score"]) >= score_threshold
            ]
            gt_boxes = [
                xywh_to_xyxy(annotation["bbox"])
                for annotation in eligible
            ]
            prediction_boxes = [
                row["bbox"] for row in selected_predictions
            ]
            matched, missed, extra = greedy_bbox_matches(
                gt_boxes,
                prediction_boxes,
                iou_threshold=matching_iou_threshold,
            )
            extra = [
                prediction_index
                for prediction_index in extra
                if not any(
                    bbox_iou(
                        prediction_boxes[prediction_index],
                        ignored_box,
                    )
                    >= matching_iou_threshold
                    for ignored_box in ignored_boxes
                )
            ]
            true_positive += len(matched)
            false_negative += len(missed)
            false_positive += len(extra)
            match_rows: list[dict[str, Any]] = []
            for gt_index, prediction_index, overlap in matched:
                annotation = eligible[gt_index]
                prediction = selected_predictions[prediction_index]
                values = np.asarray(
                    annotation["keypoints"],
                    dtype=np.float64,
                ).reshape(5, 3)
                predicted_keypoints = np.asarray(
                    [
                        [point["x"], point["y"]]
                        for point in prediction["keypoints"]
                    ],
                    dtype=np.float64,
                )
                visibility_probability = np.asarray(
                    [
                        (
                            float(point["score"])
                            if point["score"] is not None
                            else 0.0
                        )
                        for point in prediction["keypoints"]
                    ],
                    dtype=np.float64,
                )
                metric_row = metrics.update(
                    predicted_keypoints,
                    values[:, :2],
                    values[:, 2] > 0,
                    values[:, 2] > 1,
                    visibility_probability,
                    gt_boxes[gt_index],
                )[0]
                match_rows.append(
                    {
                        "annotation_id": int(annotation.get("id", -1)),
                        "prediction_index": int(prediction_index),
                        "iou": float(overlap),
                        "bbox_xyxy": gt_boxes[gt_index],
                        "predicted_keypoints": predicted_keypoints.tolist(),
                        "target_keypoints": values[:, :2].tolist(),
                        "labeled": (values[:, 2] > 0).tolist(),
                        "visible": (values[:, 2] > 1).tolist(),
                        "visibility_probability": (
                            visibility_probability.tolist()
                        ),
                        **metric_row,
                    }
                )
            prediction_stream.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "file_name": str(images_by_id[image_id]["file_name"]),
                        "score_threshold": score_threshold,
                        "predictions": selected_predictions,
                        "matches": match_rows,
                        "missed_annotation_ids": [
                            int(eligible[index].get("id", -1))
                            for index in missed
                        ],
                        "false_positive_prediction_indices": extra,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    result = {
        "model_key": "yolo26m_pose",
        "model": DISPLAY_NAMES["yolo26m_pose"],
        "family": "bottom_up_end_to_end",
        "input_mode": "full frame",
        "input_size": input_size,
        "parameters": parameter_count,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256_file(weights),
        "prediction_file": str(prediction_path),
        "training_dataset": "face_pose_drive_v2 (legacy, not balanced_811)",
        "score_threshold": score_threshold,
        "matching_iou_threshold": matching_iou_threshold,
        "detector_true_positive": true_positive,
        "detector_false_positive": false_positive,
        "detector_false_negative": false_negative,
        "detector_precision": precision,
        "detector_recall": recall,
        "detector_f1": (
            2.0 * precision * recall
            / max(1.0e-8, precision + recall)
        ),
        **yolo_training_epochs(weights),
        **metrics.compute(),
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def format_float(
    value: float | int | None,
    *,
    digits: int = 5,
) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.2f}%"


def epoch_label(row: Mapping[str, Any]) -> str:
    selected = row.get("checkpoint_epoch")
    completed = row.get("completed_epoch")
    selected_text = "?" if selected is None else str(selected)
    completed_text = "?" if completed is None else str(completed)
    return f"{selected_text}/{completed_text}"


def visibility_markdown_report(
    results: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> str:
    lines = [
        "# Visibility and occlusion comparison",
        "",
        "## Protocol",
        "",
        f"- split: `{protocol['split']}`",
        f"- annotation: `{protocol['annotation']}`",
        "- evaluated labels: COCO visibility `v > 0`",
        "- ground-truth visible: `v > 1`",
        "- ground-truth occluded: `v == 1`",
        "- decision threshold: visibility probability `>= 0.5`",
        "- confusion matrix: rows are ground truth; columns are prediction",
        "",
        "## Metrics",
        "",
        (
            "| Model | Evaluated points | Occluded Precision ↑ | "
            "Occluded Recall ↑ | Occluded F1 ↑ | Visible F1 ↑ | "
            "Macro F1 ↑ |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    valid_results = [
        row for row in results if row.get("comparison_valid", True)
    ]
    for row in valid_results:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["model"]),
                    str(row.get("evaluated_labeled_points", "—")),
                    format_percent(row.get("occluded_precision")),
                    format_percent(row.get("occluded_recall")),
                    format_percent(row.get("occluded_f1")),
                    format_percent(row.get("visibility_f1")),
                    format_percent(row.get("visibility_occluded_macro_f1")),
                )
            )
            + " |"
        )
    lines.extend(["", "## Confusion matrices", ""])
    for row in valid_results:
        matrix = row.get("visibility_confusion_matrix")
        if not matrix:
            continue
        values = matrix["values"]
        lines.extend(
            [
                f"### {row['model']}",
                "",
                "| GT \\\\ Prediction | Visible | Occluded |",
                "|---|---:|---:|",
                f"| Visible | {values[0][0]} | {values[0][1]} |",
                f"| Occluded | {values[1][0]} | {values[1][1]} |",
                "",
            ]
        )
    lines.extend(
        [
            "For YOLO26m, visibility/occlusion metrics use matched keypoints "
            "only. Missed faces are reflected in Coverage and E2E PCK, not "
            "in this classification matrix.",
            "",
            landmark_visibility_note(results),
            "",
        ]
    )
    return "\n".join(lines)


def landmark_visibility_note(
    results: Sequence[Mapping[str, Any]],
) -> str:
    landmark_rows = [
        row
        for row in results
        if row.get("model_key") in LANDMARK_MODELS
    ]
    if any(
        row.get("keypoint_visibility_training_contract")
        != "coco_v_gt_1_mask_v_gt_0"
        for row in landmark_rows
    ):
        return (
            "The ViTPose-376/Query-family rows shown here were trained with "
            "`v > 0` collapsed into one visibility target. Their coordinate "
            "metrics are comparable, but their visibility/occlusion metrics "
            "are not training-objective-matched to the corrected contract."
        )
    return (
        "The ViTPose-376/Query-family rows use coordinates on `v > 0`, "
        "visibility target `v > 1`, and visibility loss mask `v > 0`, "
        "matching this evaluation."
    )


def markdown_report(
    results: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> str:
    lines = [
        "# Face five-keypoint common comparison",
        "",
        "## Protocol",
        "",
        f"- split: `{protocol['split']}`",
        f"- annotation: `{protocol['annotation']}`",
        f"- annotation SHA-256: `{protocol['annotation_sha256']}`",
        (
            "- train-image SHA-256 overlap: "
            f"balanced-811="
            f"{protocol['leakage_audit']['training_image_overlap']['balanced_811_train']}, "
            f"legacy YOLO v2="
            f"{protocol['leakage_audit']['training_image_overlap']['legacy_yolo_v2_train']}"
        ),
        "- coordinate mask: COCO visibility `v > 0`",
        "- visibility target: `v > 1`, evaluated only where `v > 0`",
        "- NME/PCK normalization: `sqrt(GT bbox width² + GT bbox height²)`",
        "- top-down input: GT ellipse aligned long-axis square, scale 1.25",
        "- test-time flip: disabled",
        "- inference precision: float32",
        "",
        "## Comparison",
        "",
        (
            "| Model | Validity | Input | Params | selected/completed epoch | "
            "Coverage | NME ↓ | PCK@0.05 ↑ | PCK@0.10 ↑ | "
            "E2E PCK@0.05 ↑ | Visibility F1 ↑ |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["model"]),
                    (
                        "valid"
                        if row.get("comparison_valid", True)
                        else (
                            "INVALID: "
                            f"{row.get('evaluation_train_image_overlap', 0)} "
                            "train overlaps"
                        )
                    ),
                    f"{row['input_mode']} ({row['input_size']})",
                    f"{int(row['parameters']) / 1e6:.2f}M",
                    epoch_label(row),
                    format_percent(row.get("keypoint_coverage")),
                    format_float(row.get("nme")),
                    format_percent(row.get("pck_0.05")),
                    format_percent(row.get("pck_0.10")),
                    format_percent(row.get("end_to_end_pck_0.05")),
                    format_percent(row.get("visibility_f1")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "YOLO26m is an end-to-end full-frame reference. Its conditional "
            "NME/PCK use matched faces only; Coverage and E2E PCK include "
            "missed faces. The available weight was trained on the legacy "
            "`face_pose_drive_v2`, because the balanced-811 YOLO26m run has "
            "no trained checkpoint.",
            "",
            "Rows with a nonzero train-image overlap are reported for audit "
            "only and must not be used for ranking.",
            "",
            landmark_visibility_note(results),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    *,
    protocol: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        (metrics_dir / f"{row['model_key']}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    payload = {"protocol": protocol, "results": results}
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_fields = [
        "model_key",
        "model",
        "family",
        "comparison_valid",
        "evaluation_train_image_overlap",
        "training_dataset",
        "keypoint_visibility_training_contract",
        "input_mode",
        "input_size",
        "parameters",
        "checkpoint_epoch",
        "completed_epoch",
        "total_instances",
        "evaluated_instances",
        "total_labeled_points",
        "evaluated_labeled_points",
        "keypoint_coverage",
        "nme",
        "pck_0.05",
        "pck_0.10",
        "end_to_end_pck_0.05",
        "end_to_end_pck_0.10",
        "visibility_precision",
        "visibility_recall",
        "visibility_f1",
        "visibility_true_positive",
        "visibility_false_positive",
        "visibility_false_negative",
        "visibility_true_negative",
        "occluded_precision",
        "occluded_recall",
        "occluded_f1",
        "visibility_occluded_macro_f1",
        "detector_precision",
        "detector_recall",
        "detector_f1",
        "checkpoint",
        "checkpoint_sha256",
    ]
    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in csv_fields})
    (output_dir / "comparison.md").write_text(
        markdown_report(results, protocol=protocol),
        encoding="utf-8",
    )
    (output_dir / "visibility_comparison.md").write_text(
        visibility_markdown_report(results, protocol=protocol),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    selected = (
        list(ALL_MODELS)
        if "all" in args.models
        else list(dict.fromkeys(args.models))
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    comparison2 = load_comparison2_config(
        args.comparison2_config,
        project_root=REPO_ROOT,
    )
    query = load_landmark_config(
        args.query_config,
        project_root=REPO_ROOT,
    )
    comparison1_output_root = (
        args.comparison1_output_root
        if args.comparison1_output_root is not None
        else query.output_root
    ).expanduser()
    if not comparison1_output_root.is_absolute():
        comparison1_output_root = REPO_ROOT / comparison1_output_root
    comparison1_output_root = comparison1_output_root.resolve()
    if args.split == "external_v1_v2":
        external = query.raw["data"]["evaluation_sets"]["external_v1_v2"]
        annotation_path = Path(external["annotation"])
        image_root = Path(external["root"])
    else:
        data_cfg = data_config_for_model(
            comparison2.raw,
            "dinov3_fpn_rtmcc",
        )
        annotation_path = Path(data_cfg["annotations"][args.split])
        image_root = Path(data_cfg["image_root"])
    split_leakage_audit = leakage_audit(annotation_path, image_root)
    protocol = {
        "split": args.split,
        "annotation": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "image_root": str(image_root),
        "leakage_audit": split_leakage_audit,
        "coordinate_mask": "coco_v_gt_0",
        "visibility_target": "coco_v_gt_1",
        "visibility_evaluation_mask": "coco_v_gt_0",
        "normalization": "gt_bbox_diagonal_sqrt_width2_plus_height2",
        "pck_thresholds": [0.05, 0.10],
        "topdown_crop": {
            "source": "gt_ellipse",
            "geometry": "long_axis_square",
            "scale_factor": 1.25,
        },
        "test_time_flip": False,
        "inference_precision": "float32",
        "comparison2_config": str(args.comparison2_config.resolve()),
        "comparison2_config_sha256": sha256_file(
            args.comparison2_config.resolve()
        ),
        "query_config": str(args.query_config.resolve()),
        "query_config_sha256": sha256_file(args.query_config.resolve()),
        "comparison1_output_root": str(comparison1_output_root),
    }
    reused = reusable_results(args.reuse_results_from, protocol=protocol)
    if args.reuse_results_from is not None:
        protocol["reused_results_from"] = str(
            args.reuse_results_from.resolve()
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for model_name in selected:
        print(f"[common-eval] START {model_name}", flush=True)
        if model_name in reused:
            result = reused[model_name]
            print(
                f"[common-eval] REUSE {model_name}: "
                f"{result['reused_from']}",
                flush=True,
            )
        elif model_name == "yolo26m_pose":
            if not args.yolo_weights.is_file():
                raise FileNotFoundError(
                    f"YOLO26m weight not found: {args.yolo_weights}"
                )
            result = evaluate_yolo(
                annotation_path=annotation_path,
                image_root=image_root,
                device=device,
                weights=args.yolo_weights.resolve(),
                input_size=args.yolo_input_size,
                batch_size=min(args.batch_size, 8),
                score_threshold=args.yolo_score_threshold,
                matching_iou_threshold=args.matching_iou_threshold,
                output_dir=args.output_dir,
            )
        else:
            result = evaluate_topdown(
                model_name,
                comparison2=comparison2,
                query=query,
                comparison1_output_root=comparison1_output_root,
                annotation_path=annotation_path,
                image_root=image_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                output_dir=args.output_dir,
            )
        training_overlap_key = (
            "legacy_yolo_v2_train"
            if model_name == "yolo26m_pose"
            else "balanced_811_train"
        )
        result["evaluation_train_image_overlap"] = int(
            split_leakage_audit["training_image_overlap"][
                training_overlap_key
            ]
        )
        result["comparison_valid"] = (
            result["evaluation_train_image_overlap"] == 0
        )
        results.append(result)
        write_outputs(args.output_dir, protocol=protocol, results=results)
        print(
            "[common-eval] END "
            f"{model_name}: nme={result.get('nme')} "
            f"pck@.05={result.get('pck_0.05')} "
            f"coverage={result.get('keypoint_coverage')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
