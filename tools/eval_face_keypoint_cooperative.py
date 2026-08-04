from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_dinov3_deimv2 import average_precision, threshold_metrics  # noqa: E402
from tools.face_keypoint_cooperative_common import (  # noqa: E402
    autocast_context,
    build_loader,
    build_model,
    configure_device_math,
    load_cooperative_config,
    move_batch,
    resolve_path,
    write_json,
)
from tools.train_face_keypoint_cooperative import EpochLossAccumulator  # noqa: E402
from src.evaluation.ellipse_detection import ellipse_detection_metrics  # noqa: E402
from src.head_pose_geometry.heads import (  # noqa: E402
    CachedHeadDetector,
    HeadDetection,
    require_predicted_heads,
)
from src.head_pose_geometry.head_crop_dataset import (  # noqa: E402
    crop_tensor_to_square,
    square_head_crop_box,
)
from src.models.dinov3_deimv2_ellipse import (  # noqa: E402
    ellipse_to_aabb,
    scale_ellipses_anisotropic,
)
from src.wandb_utils import sync_evaluation_to_wandb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate independent and cooperative face/keypoint predictions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def iou_matrix(
    left: list[list[float]],
    right: torch.Tensor,
) -> np.ndarray:
    if not left or right.numel() == 0:
        return np.zeros((len(left), int(right.shape[0])), dtype=np.float32)
    first = np.asarray(left, dtype=np.float32)
    second = right.detach().cpu().numpy().astype(np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(-1)
    first_area = np.clip(first[:, 2:] - first[:, :2], 0.0, None).prod(-1)
    second_area = np.clip(second[:, 2:] - second[:, :2], 0.0, None).prod(-1)
    return intersection / np.clip(
        first_area[:, None] + second_area[None] - intersection,
        1.0e-9,
        None,
    )


def match_boxes(
    ground_truth: list[list[float]],
    predicted: torch.Tensor,
    *,
    iou_threshold: float,
) -> tuple[dict[int, int], list[int], list[int]]:
    overlaps = iou_matrix(ground_truth, predicted)
    if not ground_truth or predicted.numel() == 0:
        return {}, list(range(len(ground_truth))), list(range(int(predicted.shape[0])))
    gt_indices, prediction_indices = linear_sum_assignment(1.0 - overlaps)
    matched = {
        int(gt): int(prediction)
        for gt, prediction in zip(gt_indices, prediction_indices)
        if overlaps[gt, prediction] >= iou_threshold
    }
    return (
        matched,
        [index for index in range(len(ground_truth)) if index not in matched],
        [
            index
            for index in range(int(predicted.shape[0]))
            if index not in set(matched.values())
        ],
    )


def image_tensor(
    image: Image.Image,
    image_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    original_size = image.size
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous(),
        (int(original_size[1]), int(original_size[0])),
    )


def scale_decoded(
    decoded: dict[str, Any],
    *,
    input_size: int,
    original_hw: tuple[int, int],
) -> dict[str, Any]:
    original_h, original_w = original_hw
    scale_x = original_w / float(input_size)
    scale_y = original_h / float(input_size)
    output: dict[str, Any] = {
        "associations": decoded["associations"],
    }
    for key in ("independent_face", "refined_face"):
        value = {
            name: tensor.clone()
            for name, tensor in decoded[key].items()
        }
        value["boxes"][:, 0::2] *= scale_x
        value["boxes"][:, 1::2] *= scale_y
        value["ellipses"] = scale_ellipses_anisotropic(
            value["ellipses"],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        output[key] = value
    for key in ("independent_pose", "refined_pose"):
        value = {
            name: tensor.clone()
            for name, tensor in decoded[key].items()
        }
        value["boxes"][:, 0::2] *= scale_x
        value["boxes"][:, 1::2] *= scale_y
        value["keypoints"][..., 0] *= scale_x
        value["keypoints"][..., 1] *= scale_y
        output[key] = value
    if "head_pose_geometry" in decoded:
        value = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoded["head_pose_geometry"].items()
        }
        if "head_boxes" in value:
            value["head_boxes"][:, 0::2] *= scale_x
            value["head_boxes"][:, 1::2] *= scale_y
        for ellipse_key in ("amodal_ellipse", "visible_ellipse"):
            if ellipse_key in value:
                value[ellipse_key] = scale_ellipses_anisotropic(
                    value[ellipse_key],
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
        if "projected_keypoints" in value:
            value["projected_keypoints"][..., 0] *= scale_x
            value["projected_keypoints"][..., 1] *= scale_y
        output["head_pose_geometry"] = value
    if "external_head_cascade" in decoded:
        value = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoded["external_head_cascade"].items()
        }
        if "head_boxes" in value:
            value["head_boxes"][:, 0::2] *= scale_x
            value["head_boxes"][:, 1::2] *= scale_y
        output["external_head_cascade"] = value
    if "shared_head" in decoded:
        value = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoded["shared_head"].items()
        }
        value["boxes"][:, 0::2] *= scale_x
        value["boxes"][:, 1::2] *= scale_y
        output["shared_head"] = value
    return output


def scale_crop_decoded(
    decoded: dict[str, Any],
    *,
    crop_xyxy: tuple[float, float, float, float],
    input_size: int,
) -> dict[str, Any]:
    """Map one HEAD-crop cooperative prediction to full-image pixels."""

    x0, y0, x1, y1 = crop_xyxy
    scale_x = (x1 - x0) / float(input_size)
    scale_y = (y1 - y0) / float(input_size)
    output: dict[str, Any] = {"associations": decoded.get("associations", [])}
    for key in ("independent_face", "refined_face"):
        value = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoded[key].items()
        }
        value["boxes"][:, 0::2] = value["boxes"][:, 0::2] * scale_x + x0
        value["boxes"][:, 1::2] = value["boxes"][:, 1::2] * scale_y + y0
        value["ellipses"] = scale_ellipses_anisotropic(
            value["ellipses"],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        value["ellipses"][:, 0] += x0
        value["ellipses"][:, 1] += y0
        output[key] = value
    for key in ("independent_pose", "refined_pose"):
        value = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoded[key].items()
        }
        value["boxes"][:, 0::2] = value["boxes"][:, 0::2] * scale_x + x0
        value["boxes"][:, 1::2] = value["boxes"][:, 1::2] * scale_y + y0
        value["keypoints"][..., 0] = value["keypoints"][..., 0] * scale_x + x0
        value["keypoints"][..., 1] = value["keypoints"][..., 1] * scale_y + y0
        output[key] = value
    return output


def empty_cascade_decoded() -> dict[str, Any]:
    face = {
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
        "ellipses": torch.empty((0, 5)),
        "query_indices": torch.empty((0,), dtype=torch.long),
    }
    pose = {
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
        "keypoints": torch.empty((0, 5, 2)),
        "keypoint_scores": torch.empty((0, 5)),
        "visibility": torch.empty((0, 5)),
        "state_probabilities": torch.empty((0, 5, 3)),
        "query_indices": torch.empty((0,), dtype=torch.long),
    }
    return {
        "associations": [],
        "independent_face": {**face},
        "refined_face": {key: value.clone() for key, value in face.items()},
        "independent_pose": {**pose},
        "refined_pose": {key: value.clone() for key, value in pose.items()},
    }


def merge_crop_decoded(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_cascade_decoded()
    output: dict[str, Any] = {
        "associations": [
            association
            for row in rows
            for association in row.get("associations", [])
        ]
    }
    for branch in (
        "independent_face",
        "refined_face",
        "independent_pose",
        "refined_pose",
    ):
        merged: dict[str, Any] = {}
        keys = set().union(*(row[branch].keys() for row in rows))
        for key in keys:
            values = [row[branch].get(key) for row in rows]
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            if len(tensors) == len(values) and tensors:
                merged[key] = (
                    torch.cat(tensors, dim=0)
                    if tensors[0].ndim > 0
                    else tensors[0]
                )
            else:
                merged[key] = next(value for value in values if value is not None)
        output[branch] = merged
    return output


def tensor_lists(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_lists(item) for item in value]
    return value


def filter_branch(
    branch: dict[str, torch.Tensor],
    threshold: float,
) -> dict[str, torch.Tensor]:
    keep = torch.nonzero(branch["scores"] >= threshold, as_tuple=False).squeeze(-1)
    return {
        key: value[keep]
        if isinstance(value, torch.Tensor) and value.ndim > 0
        else value
        for key, value in branch.items()
    }


def face_records(
    image_id: int,
    decoded: dict[str, torch.Tensor],
    *,
    category_id: int,
) -> list[dict[str, Any]]:
    rows = []
    ellipse_boxes = ellipse_to_aabb(decoded["ellipses"])
    for index in range(int(decoded["boxes"].shape[0])):
        box = [float(value) for value in decoded["boxes"][index].tolist()]
        ellipse = [float(value) for value in decoded["ellipses"][index].tolist()]
        baseline_box = ellipse_boxes[index]
        center_x = float((baseline_box[0] + baseline_box[2]) * 0.5)
        center_y = float((baseline_box[1] + baseline_box[3]) * 0.5)
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "score": float(decoded["scores"][index]),
                "bbox_xyxy": box,
                "bbox": [
                    box[0],
                    box[1],
                    box[2] - box[0],
                    box[3] - box[1],
                ],
                "ellipse": ellipse,
                "bbox_ellipse": [
                    center_x,
                    center_y,
                    float((baseline_box[2] - baseline_box[0]) * 0.5),
                    float((baseline_box[3] - baseline_box[1]) * 0.5),
                    0.0,
                ],
            }
        )
    return rows


class KeypointAccumulator:
    def __init__(
        self,
        pck_thresholds: list[float],
        visibility_threshold: float,
        presence_threshold: float = 0.50,
    ) -> None:
        self.thresholds = pck_thresholds
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold
        self.errors: list[float] = []
        self.hits = {threshold: 0 for threshold in pck_thresholds}
        self.total_labeled = 0
        self.evaluated_labeled = 0
        self.visibility_correct = 0
        self.visibility_total = 0
        self.pose_confidence: list[float] = []
        # Rows are ground truth, columns are predictions. Presence uses
        # [missing, present]; state uses [missing, occluded, visible].
        self.presence_confusion = np.zeros((2, 2), dtype=np.int64)
        self.visibility_confusion = np.zeros((2, 2), dtype=np.int64)
        self.state_confusion = np.zeros((3, 3), dtype=np.int64)

    @staticmethod
    def _class_f1(confusion: np.ndarray, class_index: int) -> float:
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        return (
            2.0 * true_positive / denominator
            if denominator > 0
            else 0.0
        )

    def update(
        self,
        annotations: list[dict[str, Any]],
        matched: dict[int, int],
        prediction: dict[str, torch.Tensor],
    ) -> None:
        self.total_labeled += sum(
            int(
                (
                    np.asarray(row.get("keypoints", [0.0] * 15)).reshape(5, 3)[
                        :,
                        2,
                    ]
                    > 0
                ).sum()
            )
            for row in annotations
        )
        for gt_index, prediction_index in matched.items():
            values = np.asarray(
                annotations[gt_index].get("keypoints", [0.0] * 15),
                dtype=np.float32,
            ).reshape(5, 3)
            box = xywh_to_xyxy(annotations[gt_index]["bbox"])
            normalization = max(
                1.0,
                math.hypot(box[2] - box[0], box[3] - box[1]),
            )
            predicted_points = (
                prediction["keypoints"][prediction_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            predicted_visibility = (
                prediction["visibility"][prediction_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            predicted_presence = (
                prediction["keypoint_scores"][prediction_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            ground_truth_state = values[:, 2].astype(np.int64)
            ground_truth_presence = (ground_truth_state > 0).astype(np.int64)
            binary_presence = (
                predicted_presence >= self.presence_threshold
            ).astype(np.int64)
            for ground_truth, predicted in zip(
                ground_truth_presence,
                binary_presence,
                strict=True,
            ):
                self.presence_confusion[ground_truth, predicted] += 1
            if "state_probabilities" in prediction:
                state_probabilities = (
                    prediction["state_probabilities"][prediction_index]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                state_probabilities = np.stack(
                    (
                        1.0 - predicted_presence,
                        predicted_presence * (1.0 - predicted_visibility),
                        predicted_presence * predicted_visibility,
                    ),
                    axis=-1,
                )
            predicted_state = state_probabilities.argmax(axis=-1)
            for ground_truth, predicted in zip(
                ground_truth_state,
                predicted_state,
                strict=True,
            ):
                self.state_confusion[ground_truth, predicted] += 1
            labeled = values[:, 2] > 0
            visible = values[:, 2] > 1
            for point_index in np.nonzero(labeled)[0]:
                error = float(
                    np.linalg.norm(
                        predicted_points[point_index] - values[point_index, :2]
                    )
                    / normalization
                )
                self.errors.append(error)
                for threshold in self.thresholds:
                    self.hits[threshold] += int(error <= threshold)
                self.visibility_correct += int(
                    bool(
                        predicted_visibility[point_index]
                        >= self.visibility_threshold
                    )
                    == bool(visible[point_index])
                )
                self.visibility_confusion[
                    int(visible[point_index]),
                    int(
                        predicted_visibility[point_index]
                        >= self.visibility_threshold
                    ),
                ] += 1
                self.visibility_total += 1
                self.evaluated_labeled += 1
            self.pose_confidence.append(float(prediction["scores"][prediction_index]))

    def compute(self) -> dict[str, Any]:
        missing_f1 = self._class_f1(self.presence_confusion, 0)
        presence_f1 = self._class_f1(self.presence_confusion, 1)
        occluded_f1 = self._class_f1(self.visibility_confusion, 0)
        visibility_f1 = self._class_f1(self.visibility_confusion, 1)
        state_f1 = [
            self._class_f1(self.state_confusion, index)
            for index in range(3)
        ]
        return {
            "NME": float(np.mean(self.errors)) if self.errors else 0.0,
            **{
                f"PCK@{threshold:.2f}": self.hits[threshold]
                / max(1, self.evaluated_labeled)
                for threshold in self.thresholds
            },
            "keypoint_visibility_accuracy": self.visibility_correct
            / max(1, self.visibility_total),
            "visibility_f1": visibility_f1,
            "occluded_f1": occluded_f1,
            "visibility_occluded_macro_f1": 0.5
            * (visibility_f1 + occluded_f1),
            "keypoint_presence_accuracy": float(
                np.trace(self.presence_confusion)
                / max(1, self.presence_confusion.sum())
            ),
            "presence_f1": presence_f1,
            "missing_f1": missing_f1,
            "presence_missing_macro_f1": 0.5
            * (presence_f1 + missing_f1),
            "missing_state_f1": state_f1[0],
            "occluded_state_f1": state_f1[1],
            "visible_state_f1": state_f1[2],
            "state_macro_f1": float(np.mean(state_f1)),
            "presence_confusion_matrix": self.presence_confusion.tolist(),
            "visibility_confusion_matrix": self.visibility_confusion.tolist(),
            "state_confusion_matrix": self.state_confusion.tolist(),
            "keypoint_recall": self.evaluated_labeled / max(1, self.total_labeled),
            "pose_confidence_mean": float(np.mean(self.pose_confidence))
            if self.pose_confidence
            else 0.0,
            "evaluated_keypoints": float(self.evaluated_labeled),
            "total_labeled_keypoints": float(self.total_labeled),
        }


@torch.inference_mode()
def evaluate_split_loss(
    model: torch.nn.Module,
    *,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    maximum_samples: int | None,
) -> dict[str, float]:
    if (
        (
            getattr(model, "head_pose_geometry", None) is not None
            or getattr(model, "external_head_cascade", None) is not None
        )
        and getattr(model, "shared_head_branch", None) is None
    ):
        raise RuntimeError(
            "Split loss with target-derived HEAD routing is disabled; "
            "deployment-style HEAD predictions are required."
        )
    loader, _ = build_loader(
        config,
        split,
        training=False,
        distributed=False,
        rank=0,
        world_size=1,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        maximum_samples=maximum_samples,
    )
    accumulator = EpochLossAccumulator()
    model.eval()
    for batch in loader:
        images, targets = move_batch(batch, device)
        with autocast_context(device, config["train"]):
            output = model(images, targets)
        accumulator.update(output)
    return accumulator.mean({"distributed": False})


def evaluation_wandb_summary(
    metrics: dict[str, Any],
    *,
    split: str,
) -> dict[str, int | float | str]:
    summary: dict[str, int | float | str] = {
        f"{split}/evaluation_status": "complete",
        f"{split}/images": int(metrics["images"]),
    }
    split_loss = metrics.get("split_loss", {})
    if isinstance(split_loss, dict) and "loss_total" in split_loss:
        summary[f"{split}/loss_total"] = float(split_loss["loss_total"])
    for stage in ("independent", "refined"):
        stage_metrics = metrics.get(stage, {})
        face = stage_metrics.get("face", {})
        keypoint = stage_metrics.get("keypoint", {})
        for source, display in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("AP50", "AP50"),
            ("mAP50_95", "mAP50_95"),
        ):
            if source in face:
                summary[f"{split}/face_{stage}_{display}"] = float(
                    face[source]
                )
        for source, display in (
            ("NME", "NME"),
            ("PCK@0.03", "PCK_at_0.03"),
            ("PCK@0.05", "PCK_at_0.05"),
            ("PCK@0.10", "PCK_at_0.10"),
            ("keypoint_visibility_accuracy", "visibility_accuracy"),
            ("visibility_f1", "visibility_f1"),
            ("occluded_f1", "occluded_f1"),
            (
                "visibility_occluded_macro_f1",
                "visibility_occluded_macro_f1",
            ),
            ("keypoint_presence_accuracy", "presence_accuracy"),
            ("presence_f1", "presence_f1"),
            ("missing_f1", "missing_f1"),
            ("presence_missing_macro_f1", "presence_missing_macro_f1"),
            ("state_macro_f1", "state_macro_f1"),
            ("keypoint_recall", "keypoint_recall"),
        ):
            if source in keypoint:
                summary[f"{split}/keypoint_{stage}_{display}"] = float(
                    keypoint[source]
                )
    joint_changes = metrics.get("joint_changes", {})
    if "association_accuracy" in joint_changes:
        summary[f"{split}/association_accuracy"] = float(
            joint_changes["association_accuracy"]
        )
    return summary


def update_wandb_evaluation_summary(
    config: dict[str, Any],
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    split: str,
) -> dict[str, Any] | None:
    settings = config.get("wandb", {})
    if (
        not isinstance(settings, dict)
        or not bool(settings.get("enabled", False))
        or str(settings.get("mode", "online")) != "online"
    ):
        return None
    state_path = Path(config["output"]["root"]) / "wandb_run.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"W&B training state is missing: {state_path}"
        )
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required to sync evaluation") from error
    baseline_dir = output_dir / "baseline_aligned"
    metric_sources = {"diagnostic": output_dir / "metrics.json"}
    if (baseline_dir / "metrics.json").is_file():
        metric_sources["baseline_aligned"] = baseline_dir / "metrics.json"
    raw_headline = evaluation_wandb_summary(metrics, split=split)
    headline = {
        key.removeprefix(f"{split}/"): value
        for key, value in raw_headline.items()
    }
    return sync_evaluation_to_wandb(
        wandb,
        state_path=state_path,
        split=split,
        metric_sources=metric_sources,
        artifact_files=(
            output_dir / "predictions.jsonl",
            baseline_dir / "comparison_table.md",
            baseline_dir / "success.json",
        ),
        headline=headline,
        report_path=output_dir / "wandb_evaluation_sync.json",
    )


def main() -> int:
    args = parse_args()
    config = load_cooperative_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    configure_device_math(device, config["train"])
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else Path(config["output"]["root"]) / "evaluation" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Some already-running queues were launched before validation was added
    # to the common comparison contract.  A test invocation therefore fills
    # the missing validation artifact first, using the same checkpoint.
    if args.split == "test" and args.max_images is None:
        val_success = (
            Path(config["output"]["root"])
            / "evaluation"
            / "val"
            / "baseline_aligned"
            / "success.json"
        )
        if not val_success.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(resolve_path(args.config)),
                    "--checkpoint",
                    str(resolve_path(args.checkpoint)),
                    "--split",
                    "val",
                    "--device",
                    str(args.device),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
    model = build_model(
        config,
        device=device,
        checkpoint=resolve_path(args.checkpoint),
    ).eval()
    checkpoint_state = torch.load(
        resolve_path(args.checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_epoch = (
        int(checkpoint_state.get("epoch", 0))
        if isinstance(checkpoint_state, dict)
        else 0
    )
    del checkpoint_state
    model.set_epoch(checkpoint_epoch)
    head_crop_cfg = config.get("data", {}).get("head_crop", {})
    head_crop_enabled = bool(
        isinstance(head_crop_cfg, dict) and head_crop_cfg.get("enabled", False)
    )
    head_detector = None
    if (
        head_crop_enabled
        or (
            model.head_pose_geometry is not None
            or model.external_head_cascade is not None
        )
        and model.shared_head_branch is None
    ):
        caches = config.get("data", {}).get("predicted_head_caches", {})
        cache_path = caches.get(args.split) if isinstance(caches, dict) else None
        if not cache_path:
            raise ValueError(
                "Three-stage evaluation requires "
                f"data.predicted_head_caches.{args.split}"
            )
        head_detector = CachedHeadDetector(cache_path)
    evaluation = config.get("evaluation", {})
    face_threshold = float(
        evaluation.get(
            "face_score_threshold",
            config.get("inference", {}).get("face_score_threshold", 0.30),
        )
    )
    pose_threshold = float(
        evaluation.get(
            "pose_score_threshold",
            config.get("inference", {}).get("pose_score_threshold", 0.30),
        )
    )
    score_min = float(evaluation.get("score_min", 0.001))
    model.face_score_threshold = score_min
    model.pose_score_threshold = score_min
    iou_threshold = float(evaluation.get("matching_iou_threshold", 0.50))
    pck_thresholds = [
        float(value)
        for value in evaluation.get("pck_thresholds", (0.03, 0.05, 0.10))
    ]
    visibility_threshold = float(evaluation.get("visibility_threshold", 0.50))
    presence_threshold = float(
        evaluation.get("keypoint_presence_threshold", 0.50)
    )
    coco = json.loads(
        Path(config["data"]["annotations"][args.split]).read_text(encoding="utf-8")
    )
    allowed = {int(value) for value in config["data"].get("category_ids", (2,))}
    category_id = int(config["data"].get("target_category_id", min(allowed)))
    image_by_id = {
        int(image["id"]): image for image in coco.get("images", [])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if int(annotation.get("category_id", -1)) not in allowed:
            continue
        row = dict(annotation)
        if bool(config["data"].get("head_centric", False)):
            if (
                not bool(row.get("attribute_valid", False))
                or len(row.get("ellipse", ())) != 5
            ):
                continue
            image_metadata = image_by_id[int(row["image_id"])]
            width = float(image_metadata["width"])
            height = float(image_metadata["height"])
            face_box = ellipse_to_aabb(
                torch.tensor(row["ellipse"], dtype=torch.float64).reshape(1, 5)
            )[0]
            face_box[0::2].clamp_(0.0, width)
            face_box[1::2].clamp_(0.0, height)
            x1, y1, x2, y2 = (
                float(value) for value in face_box.tolist()
            )
            row["head_bbox"] = list(row["bbox"])
            row["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            row["area"] = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            row["category_id"] = category_id
            row["keypoints"] = [
                value
                for point, state in zip(
                    row.get("keypoints", ()),
                    row.get("keypoint_states", ()),
                    strict=True,
                )
                for value in (float(point[0]), float(point[1]), int(state))
            ]
        annotations_by_image[int(row["image_id"])].append(row)
    images = list(coco.get("images", []))
    if args.max_images is not None:
        images = images[: int(args.max_images)]
    image_root = Path(config["data"]["image_root"])
    input_size = int(config["data"].get("image_size", 960))
    gt_detection = {
        int(image["id"]): [
            {
                "image_id": int(image["id"]),
                "category_id": int(row["category_id"]),
                "bbox_xyxy": xywh_to_xyxy(row["bbox"]),
                "area": float(row.get("area", row["bbox"][2] * row["bbox"][3])),
            }
            for row in annotations_by_image.get(int(image["id"]), [])
        ]
        for image in images
    }
    gt_ellipse = {
        int(image["id"]): annotations_by_image.get(int(image["id"]), [])
        for image in images
    }
    face_predictions = {"independent": [], "refined": []}
    keypoint_metrics = {
        "independent": KeypointAccumulator(
            pck_thresholds,
            visibility_threshold,
            presence_threshold,
        ),
        "refined": KeypointAccumulator(
            pck_thresholds,
            visibility_threshold,
            presence_threshold,
        ),
    }
    joint = defaultdict(int)
    association_correct = association_total = 0
    prediction_path = output_dir / "predictions.jsonl"
    started = time.perf_counter()
    inference_seconds = 0.0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for image_index, image_info in enumerate(images):
            relative = Path(str(image_info["file_name"]))
            candidates = (image_root / relative, image_root / "images" / relative)
            path = next((value for value in candidates if value.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Image not found: {relative}")
            image = Image.open(path).convert("RGB")
            tensor, original_hw = image_tensor(image, input_size)
            head_detections = None
            if head_detector is not None:
                original_h, original_w = original_hw
                detected = head_detector.detect(
                    torch.empty((3, original_h, original_w), device="meta"),
                    image_id=int(image_info["id"]),
                )
                require_predicted_heads(detected, split=args.split)
                scale_x = input_size / float(original_w)
                scale_y = input_size / float(original_h)
                head_detections = [[
                    HeadDetection(
                        box_xyxy=(
                            item.box_xyxy[0] * scale_x,
                            item.box_xyxy[1] * scale_y,
                            item.box_xyxy[2] * scale_x,
                            item.box_xyxy[3] * scale_y,
                        ),
                        confidence=item.confidence,
                        instance_id=item.instance_id,
                        valid_image_region_xyxy=(
                            item.valid_image_region_xyxy[0] * scale_x,
                            item.valid_image_region_xyxy[1] * scale_y,
                            item.valid_image_region_xyxy[2] * scale_x,
                            item.valid_image_region_xyxy[3] * scale_y,
                        ),
                        source=item.source,
                    )
                    for item in detected
                ]]
            raw: dict[str, Any] | None = None
            if head_crop_enabled:
                if head_detections is None:
                    raise RuntimeError(
                        "HEAD-crop evaluation requires cached predicted HEADs"
                    )
                original_array = np.asarray(image, dtype=np.float32) / 255.0
                original_tensor = (
                    torch.from_numpy(original_array)
                    .permute(2, 0, 1)
                    .contiguous()
                )
                detected = head_detections[0]
                # head_detections above were scaled for the legacy full-frame
                # router. Convert them back to original-image coordinates for
                # the locked square crop contract.
                original_h, original_w = original_hw
                scale_back_x = original_w / float(input_size)
                scale_back_y = original_h / float(input_size)
                original_heads = [
                    HeadDetection(
                        box_xyxy=(
                            item.box_xyxy[0] * scale_back_x,
                            item.box_xyxy[1] * scale_back_y,
                            item.box_xyxy[2] * scale_back_x,
                            item.box_xyxy[3] * scale_back_y,
                        ),
                        confidence=item.confidence,
                        instance_id=item.instance_id,
                        source=item.source,
                    )
                    for item in detected
                ]
                crop_boxes = [
                    square_head_crop_box(
                        item.box_xyxy,
                        context_scale=float(head_crop_cfg.get("context_scale", 1.20)),
                    )
                    for item in original_heads
                ]
                crop_tensors = [
                    crop_tensor_to_square(
                        original_tensor,
                        crop_box,
                        output_size=input_size,
                        pad_value=float(head_crop_cfg.get("pad_value", 0.0)),
                    )
                    for crop_box in crop_boxes
                ]
                decoded_crops: list[dict[str, Any]] = []
                cascade_batch_size = max(
                    1,
                    int(head_crop_cfg.get("evaluation_batch_size", 4)),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.inference_mode(), autocast_context(device, config["train"]):
                    for start in range(0, len(crop_tensors), cascade_batch_size):
                        batch_crops = torch.stack(
                            crop_tensors[start : start + cascade_batch_size]
                        ).to(device)
                        batch_raw = model(batch_crops, return_raw=True)
                        batch_decoded = model.decode(
                            batch_raw,
                            image_size=(input_size, input_size),
                        )
                        decoded_crops.extend(
                            scale_crop_decoded(
                                item,
                                crop_xyxy=crop_boxes[start + offset],
                                input_size=input_size,
                            )
                            for offset, item in enumerate(batch_decoded)
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                decoded_all = merge_crop_decoded(decoded_crops)
                decoded_all["shared_head"] = {
                    "boxes": torch.tensor(
                        [item.box_xyxy for item in original_heads],
                        dtype=torch.float32,
                    ).reshape(-1, 4),
                    "scores": torch.tensor(
                        [item.confidence for item in original_heads],
                        dtype=torch.float32,
                    ),
                }
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.inference_mode(), autocast_context(device, config["train"]):
                    raw = model(
                        tensor.to(device),
                        return_raw=True,
                        head_detections=head_detections,
                    )
                    decoded = model.decode(raw, image_size=(input_size, input_size))[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                decoded_all = scale_decoded(
                    decoded,
                    input_size=input_size,
                    original_hw=original_hw,
                )
            decoded = {
                **decoded_all,
                "independent_face": filter_branch(
                    decoded_all["independent_face"],
                    face_threshold,
                ),
                "refined_face": filter_branch(
                    decoded_all["refined_face"],
                    face_threshold,
                ),
                "independent_pose": filter_branch(
                    decoded_all["independent_pose"],
                    pose_threshold,
                ),
                "refined_pose": filter_branch(
                    decoded_all["refined_pose"],
                    pose_threshold,
                ),
            }
            image_id = int(image_info["id"])
            annotations = annotations_by_image.get(image_id, [])
            gt_boxes = [xywh_to_xyxy(row["bbox"]) for row in annotations]
            matches: dict[str, tuple[dict[int, int], list[int], list[int]]] = {}
            for stage in ("independent", "refined"):
                face_key = stage + "_face"
                pose_key = stage + "_pose"
                face_match = match_boxes(
                    gt_boxes,
                    decoded[face_key]["boxes"],
                    iou_threshold=iou_threshold,
                )
                pose_match = match_boxes(
                    gt_boxes,
                    decoded[pose_key]["boxes"],
                    iou_threshold=iou_threshold,
                )
                matches[face_key] = face_match
                matches[pose_key] = pose_match
                face_predictions[stage].extend(
                    face_records(
                        image_id,
                        decoded_all[face_key],
                        category_id=category_id,
                    )
                )
                keypoint_metrics[stage].update(
                    annotations,
                    pose_match[0],
                    decoded[pose_key],
                )
                face_gt = set(face_match[0])
                pose_gt = set(pose_match[0])
                joint[f"{stage}_face_only"] += len(face_gt - pose_gt)
                joint[f"{stage}_keypoint_only"] += len(pose_gt - face_gt)
                joint[f"{stage}_both"] += len(face_gt & pose_gt)
                joint[f"{stage}_face_tp"] += len(face_gt)
                joint[f"{stage}_pose_tp"] += len(pose_gt)
                joint[f"{stage}_fp"] += len(face_match[2]) + len(pose_match[2])
            independent_face_gt = set(matches["independent_face"][0])
            independent_pose_gt = set(matches["independent_pose"][0])
            refined_face_gt = set(matches["refined_face"][0])
            refined_pose_gt = set(matches["refined_pose"][0])
            joint["face_miss_rescued_by_keypoint"] += len(
                (refined_face_gt - independent_face_gt) & independent_pose_gt
            )
            joint["keypoint_miss_rescued_by_face"] += len(
                (refined_pose_gt - independent_pose_gt) & independent_face_gt
            )
            refined_face_match = matches["refined_face"][0]
            refined_pose_match = matches["refined_pose"][0]
            if raw is not None:
                association = raw["refined"]["association_logits"][0]
                face_query_positions = decoded["refined_face"]["query_indices"]
                pose_query_positions = decoded["refined_pose"]["query_indices"]
                for gt_index in set(refined_face_match) & set(refined_pose_match):
                    face_detection_index = refined_face_match[gt_index]
                    pose_detection_index = refined_pose_match[gt_index]
                    face_query = int(face_query_positions[face_detection_index])
                    pose_query = int(pose_query_positions[pose_detection_index])
                    visible_pose_queries = pose_query_positions.to(association.device)
                    predicted_local = int(
                        association[face_query, visible_pose_queries].argmax()
                    )
                    predicted_pose_query = int(visible_pose_queries[predicted_local])
                    association_correct += int(predicted_pose_query == pose_query)
                    association_total += 1
            row = {
                "image_id": image_id,
                "file_name": str(image_info["file_name"]),
                "predictions": tensor_lists(decoded),
                "matches": {
                    key: {
                        "matched": value[0],
                        "missed_gt": value[1],
                        "false_positive": value[2],
                    }
                    for key, value in matches.items()
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": args.split,
        "images": len(images),
        "face_score_threshold": face_threshold,
        "pose_score_threshold": pose_threshold,
        "matching_iou_threshold": iou_threshold,
        "keypoint_presence_threshold": presence_threshold,
        "visibility_threshold": visibility_threshold,
        "independent": {},
        "refined": {},
    }
    thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
    for stage in ("independent", "refined"):
        ap_values = [
            average_precision(
                gt_detection,
                face_predictions[stage],
                category_id=category_id,
                iou_threshold=threshold,
            )
            for threshold in thresholds
        ]
        face_metrics = threshold_metrics(
            gt_detection,
            face_predictions[stage],
            category_id=category_id,
            score_threshold=face_threshold,
            iou_threshold=iou_threshold,
        )
        ellipse_metrics = ellipse_detection_metrics(
            gt_ellipse,
            face_predictions[stage],
            category_ids=[category_id],
            fixed_threshold=face_threshold,
        )
        metrics[stage] = {
            "face": {
                "AP50": average_precision(
                    gt_detection,
                    face_predictions[stage],
                    category_id=category_id,
                    iou_threshold=0.50,
                ),
                "mAP50_95": float(np.mean(ap_values)),
                **face_metrics,
                **ellipse_metrics,
            },
            "keypoint": keypoint_metrics[stage].compute(),
            "joint": {
                "face_only": joint[f"{stage}_face_only"],
                "keypoint_only": joint[f"{stage}_keypoint_only"],
                "both": joint[f"{stage}_both"],
            },
        }
    metrics["joint_changes"] = {
        "face_branch_misses_rescued_by_keypoint_branch": joint[
            "face_miss_rescued_by_keypoint"
        ],
        "keypoint_branch_misses_rescued_by_face_branch": joint[
            "keypoint_miss_rescued_by_face"
        ],
        "interaction_added_tp": (
            joint["refined_face_tp"]
            + joint["refined_pose_tp"]
            - joint["independent_face_tp"]
            - joint["independent_pose_tp"]
        ),
        "interaction_added_fp": joint["refined_fp"] - joint["independent_fp"],
        "association_accuracy": association_correct / max(1, association_total),
        "association_samples": association_total,
    }
    metrics["runtime"] = {
        "total_seconds": elapsed,
        "images_per_second": len(images) / max(elapsed, 1.0e-12),
        "milliseconds_per_image": elapsed * 1000.0 / max(1, len(images)),
        "model_inference_seconds": inference_seconds,
        "model_inference_milliseconds_per_image": (
            inference_seconds * 1000.0 / max(1, len(images))
        ),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "batch_size": 1,
        "input_size": input_size,
    }
    metrics["split_loss"] = evaluate_split_loss(
        model,
        config=config,
        split=args.split,
        device=device,
        maximum_samples=args.max_images,
    )
    metrics["split_loss_contract"] = {
        "batch_size": int(evaluation.get("batch_size", 1)),
        "checkpoint_epoch": checkpoint_epoch,
        "selection_use": "report_only_never_model_selection",
    }
    write_json(output_dir / "metrics.json", metrics)
    if args.max_images is None and args.split in {"val", "test"}:
        baseline_annotation = (
            REPO_ROOT
            / "outputs"
            / "dataset_cache"
            / "face_pose_drive_v2_balanced_811"
            / "annotations"
            / "face_dino_v1"
            / f"{args.split}.json"
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/evaluate_baseline_aligned_predictions.py"),
                "--format",
                "cooperative",
                "--predictions",
                str(prediction_path),
                "--annotations",
                str(baseline_annotation),
                "--split",
                str(args.split),
                "--architecture-id",
                str(config.get("experiment_name", Path(config["output"]["root"]).name)),
                "--comparison-stage",
                "refined",
                "--output-dir",
                str(output_dir / "baseline_aligned"),
                "--source-metrics",
                str(output_dir / "metrics.json"),
                "--config",
                str(resolve_path(args.config)),
                "--checkpoint",
                str(resolve_path(args.checkpoint)),
                "--seed",
                str(int(config.get("train", {}).get("seed", 0))),
                "--validation-face-threshold",
                str(face_threshold),
                "--validation-point-threshold",
                "0.70",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    update_wandb_evaluation_summary(
        config,
        output_dir=output_dir,
        metrics=metrics,
        split=args.split,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
