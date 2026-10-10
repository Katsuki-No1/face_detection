from __future__ import annotations

import argparse
import csv
import copy
from collections import defaultdict
import hashlib
import json
import math
import random
import re
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config
from src.checkpoint_retention import prune_periodic_checkpoints
from src.amp_calibration import calibration_policy, calibrate_gradients, gradients_finite, preserve_rng_and_buffers
from src.codino_runtime import OFFICIAL_TYPES, RuntimeProgress, configure_memory_budget
from src.wandb_utils import write_json_atomic
from src.data import AugmentedFaceKeypointDataset, FaceKeypointDataset, SyntheticFaceKeypointDataset, collate_face_keypoint_batch
from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import (
    parse_image_size,
    prepare_image_tensor,
    scale_detections_to_original,
)


LOSS_LOG_KEYS = (
    "loss_total",
    "loss_cls",
    "loss_center",
    "loss_ellipse",
    "loss_ellipse_center",
    "loss_ellipse_axes",
    "loss_ellipse_angle",
    "loss_ellipse_chamfer",
    "loss_depth_consistency",
    "loss_center_offset",
    "loss_bbox",
    "loss_bbox_l1",
    "loss_giou",
    "loss_mal",
    "loss_fgl",
    "loss_ddf",
    "loss_aux",
    "loss_dn",
    "loss_enc",
    "loss_pre",
    "loss_kpt",
    "match_quality",
    "num_pos",
)

VAL_LOG_KEYS = (
    "epoch",
    "split",
    "num_images",
    "num_gt",
    "loss",
    "mAP",
    "AP50",
    "AP75",
    "precision",
    "recall",
    "f1",
    "fixed_threshold",
    "best_threshold",
    "best_precision",
    "best_recall",
    "best_f1",
    "ellipse_mAP",
    "ellipse_AP50",
    "ellipse_AP75",
    "bbox_ellipse_mAP",
    "mean_ellipse_iou",
    "ellipse_center_nme",
    "ellipse_major_relative_error",
    "ellipse_minor_relative_error",
    "ellipse_angle_mae_deg",
    "ellipse_matched",
    "ellipse_angle_samples",
    "elapsed_seconds",
)


def add_training_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument("--wandb", action="store_true", help="Log full-training metrics to W&B.")
    group.add_argument(
        "--wandb-entity",
        default="sato-waseda-university",
        help="W&B user or team name. Default: sato-waseda-university.",
    )
    group.add_argument(
        "--wandb-project",
        default="face_detection",
        help="W&B project name. Default: face_detection.",
    )
    group.add_argument("--wandb-run-name", help="Optional W&B run display name.")
    group.add_argument(
        "--wandb-group",
        help="Run group. When omitted, a seed-independent training-condition group is generated.",
    )
    group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode. Default: online.",
    )
    group.add_argument(
        "--dataset-version",
        default=None,
        help="Dataset version stored in the training run config. Default: data.dataset_version or face_dataset_v7.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run config-driven DINO detectors.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Run one lightweight batch through loss computation only.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic smoke data for dry-run.")
    parser.add_argument(
        "--image-size",
        default=None,
        help="Override data.image_size for smoke runs (INT or HEIGHTxWIDTH).",
    )
    parser.add_argument("--overfit-smoke", action="store_true", help="Train on a tiny subset to confirm the model can overfit.")
    parser.add_argument("--max-samples", type=int, default=16, help="Subset size for --overfit-smoke.")
    parser.add_argument("--steps", type=int, default=40, help="Optimization steps for --overfit-smoke.")
    parser.add_argument("--lr", type=float, default=3.0e-3, help="Learning rate for --overfit-smoke.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for smoke modes.")
    parser.add_argument("--tiny-model", action="store_true", help="Use narrower channels for CPU overfit smoke.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs for full training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a checkpoint.")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progressive-resizing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for debug runs.")
    parser.add_argument(
        "--allow-dummy-backbone",
        action="store_true",
        help="Allow full training with an explicit dummy DINO backbone. Intended only for local debugging.",
    )
    add_training_wandb_arguments(parser)
    return parser.parse_args()


def training_wandb_config(config: dict[str, Any], *, dataset_version: str | None) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}) or {})
    backbone_cfg = dict(model_cfg.get("backbone", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    optimizer_cfg = dict(config.get("optimizer", train_cfg.get("optimizer", {})) or {})
    verification_cfg = dict(config.get("verification", {}) or {})
    resolved_dataset_version = str(
        dataset_version or config.get("data", {}).get("dataset_version") or "face_dataset_v7"
    )
    learning_rate = train_cfg.get("lr", optimizer_cfg.get("lr", train_cfg.get("lr_head")))
    return {
        "model": str(model_cfg.get("type") or model_cfg.get("name") or "unknown"),
        "model_size": str(
            model_cfg.get("size")
            or verification_cfg.get("official_variant")
            or backbone_cfg.get("name")
            or backbone_cfg.get("type")
            or "unknown"
        ),
        "dataset_version": resolved_dataset_version,
        "epochs": int(train_cfg.get("epochs", 0)),
        "batch_size": int(train_cfg.get("batch_size", 0)),
        "imgsz": train_cfg.get("imgsz", config.get("data", {}).get("image_size")),
        "optimizer": str(optimizer_cfg.get("type") or optimizer_cfg.get("name") or "AdamW"),
        "learning_rate": float(learning_rate) if learning_rate is not None else None,
        "seed": int(train_cfg.get("seed", 42)),
        "augmentation": copy.deepcopy(dict(config.get("augmentation", {}) or {})),
        "freeze_unfreeze": {
            "backbone_freeze": bool(backbone_cfg.get("freeze", False)),
            "backbone_policy": train_cfg.get("backbone_policy"),
            "trainable_last_blocks": backbone_cfg.get("trainable_last_blocks"),
            "unfreeze_last_n_blocks": backbone_cfg.get("unfreeze_last_n_blocks"),
            "lr_backbone": train_cfg.get("lr_backbone"),
        },
    }


def _automatic_wandb_group(run_config: dict[str, Any]) -> str:
    condition_config = copy.deepcopy(run_config)
    condition_config.pop("seed", None)
    digest = hashlib.sha256(
        json.dumps(condition_config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{run_config['model']}-{run_config['dataset_version']}-{digest}"


def training_wandb_run_id(
    *,
    entity: str | None,
    project: str,
    group: str,
    run_name: str,
    seed: int,
) -> str:
    identity = {
        "entity": entity,
        "project": project,
        "group": group,
        "run_name": run_name,
        "seed": int(seed),
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def verify_training_wandb_access(
    wandb_module: Any,
    *,
    entity: str | None,
    project: str,
    mode: str,
) -> dict[str, Any]:
    """Verify online W&B access without creating a run or exposing its API key."""

    report: dict[str, Any] = {
        "status": "pass",
        "version": str(wandb_module.__version__),
        "entity": entity,
        "project": project,
        "mode": mode,
        "api_key_configured": None,
        "project_accessible": None,
    }
    if mode != "online":
        return report
    try:
        api = wandb_module.Api(timeout=30)
        report["api_key_configured"] = bool(api.api_key)
        if not report["api_key_configured"]:
            raise RuntimeError("W&B API key is not configured; run `wandb login` before training")
        target_entity = entity or api.default_entity
        if not target_entity:
            raise RuntimeError("W&B entity could not be resolved")
        project_object = api.project(project, entity=target_entity)
        report["entity"] = target_entity
        report["project_accessible"] = bool(project_object is not None)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"W&B authentication/project preflight failed ({type(error).__name__}): {error}"
        ) from error
    if not report["project_accessible"]:
        raise RuntimeError(f"W&B project is not accessible: {report['entity']}/{project}")
    return report


def init_training_wandb(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path | None = None,
):
    if not bool(args.wandb):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging was requested, but wandb is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    run_config = training_wandb_config(config, dataset_version=args.dataset_version)
    train_name = str(config.get("train", {}).get("name") or config.get("experiment_name") or run_config["model"])
    run_name = args.wandb_run_name or f"{train_name}-seed{run_config['seed']}"
    run_group = args.wandb_group or _automatic_wandb_group(run_config)
    preflight = verify_training_wandb_access(
        wandb,
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
    )
    run_id = training_wandb_run_id(
        entity=str(preflight["entity"]) if preflight["entity"] else None,
        project=args.wandb_project,
        group=run_group,
        run_name=run_name,
        seed=int(run_config["seed"]),
    )
    preflight.update(
        {
            "group": run_group,
            "run_name": run_name,
            "run_id": run_id,
            "resume": False,
        }
    )
    resume_wandb_run = bool(args.resume_from)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        preflight_path = output_dir / "wandb_preflight.json"
        resume_wandb_run = resume_wandb_run or preflight_path.exists()
        preflight["resume"] = resume_wandb_run
        preflight_path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    init_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "group": run_group,
        "id": run_id,
        "resume": "allow" if resume_wandb_run else "never",
        "job_type": "training",
        "mode": args.wandb_mode,
        "config": run_config,
    }
    if output_dir is not None:
        local_dir = output_dir / "wandb"
        local_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(local_dir)
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    run = wandb.init(**init_kwargs)
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    if run.url:
        print(f"wandb_run: {run.url}")
    else:
        print(f"wandb_run: started ({args.wandb_mode} mode)")
    print(f"wandb_group: {run_group}")
    print(
        "wandb_preflight: "
        f"entity={preflight['entity']} project={preflight['project']} "
        f"authenticated={preflight['api_key_configured']} accessible={preflight['project_accessible']}"
    )
    return run


def load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_speed(device: str, config: dict[str, Any], *, tf32_override: bool | None) -> dict[str, Any]:
    train_cfg = config.get("train", {})
    cuda_enabled = str(device).startswith("cuda")
    tf32_enabled = bool(train_cfg.get("tf32", True) if tf32_override is None else tf32_override)
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    matmul_precision = str(train_cfg.get("matmul_precision", "high"))
    if cuda_enabled:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)
    metrics = {
        "cuda": cuda_enabled,
        "tf32": tf32_enabled if cuda_enabled else False,
        "cudnn_benchmark": cudnn_benchmark if cuda_enabled else False,
        "matmul_precision": matmul_precision if cuda_enabled else "",
    }
    return metrics


def _configured_image_size(config: dict[str, Any]) -> int | tuple[int, int]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    return data_cfg.get("image_size") or train_cfg.get("imgsz", 896)


def _annotation_candidates(data_root: Path, split: str) -> list[Path]:
    return [
        data_root / "annotations" / f"instances_{split}.json",
        data_root / "annotations" / f"{split}.json",
        data_root / f"instances_{split}.json",
        data_root / f"{split}.json",
    ]


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def _box_area_xyxy(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area_xyxy(a) + _box_area_xyxy(b) - inter
    return inter / max(union, 1.0e-9)


def _config_class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("model", {}).get("class_names") or config.get("class_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return ["head", "face"]


def _config_category_ids(config: dict[str, Any]) -> list[int]:
    ids = config.get("data", {}).get("category_ids")
    if isinstance(ids, list) and ids:
        return [int(value) for value in ids]
    return list(range(1, len(_config_class_names(config)) + 1))


def _choose_eval_image_root(root: Path, annotation_path: Path, split: str, configured: str | Path | None) -> Path | None:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    file_name = str(images[0].get("file_name", "")) if images else ""
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else root / configured_path)
    candidates.extend([root, root / "images" / split, root / "images", root / split])
    if not file_name:
        return candidates[0] if candidates else None
    file_path = Path(file_name)
    if file_path.is_absolute():
        return None
    for candidate in candidates:
        if (candidate / file_path).exists():
            return candidate
    return candidates[0] if candidates else None


def _resolve_eval_data_paths(config: dict[str, Any], split: str) -> tuple[Path, Path]:
    data_cfg = config.get("data", {})
    root_value = data_cfg.get("data_root")
    root = Path(root_value) if root_value else None
    annotation_path: Path | None = None
    annotation_paths = data_cfg.get("annotation_paths", {})
    configured_split_path = annotation_paths.get(split) if isinstance(annotation_paths, dict) else None
    if configured_split_path:
        candidate = Path(str(configured_split_path))
        if candidate.exists():
            annotation_path = candidate
    if root is not None:
        if annotation_path is None:
            for candidate in _annotation_candidates(root, split):
                if candidate.exists():
                    annotation_path = candidate
                    break
    if annotation_path is None and data_cfg.get("annotation_path"):
        candidate = Path(str(data_cfg["annotation_path"]))
        if candidate.exists() and split in candidate.name:
            annotation_path = candidate
    if annotation_path is None or not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found for validation split={split!r}.")
    if root is None:
        root = annotation_path.parent
    split_roots = data_cfg.get("image_roots", {})
    configured_root = (
        split_roots.get(split)
        if isinstance(split_roots, dict)
        else None
    ) or data_cfg.get("image_root")
    image_root = _choose_eval_image_root(root, annotation_path, split, configured_root)
    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"image_root not found for validation split={split!r}: {image_root}")
    return annotation_path, image_root


def _load_coco_eval(annotation_path: Path, selected_category_ids: list[int]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = list(data.get("images", []))
    allowed = set(int(value) for value in selected_category_ids)
    gt_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in images}
    for annotation in data.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))
        if category_id not in allowed or int(annotation.get("iscrowd", 0)):
            continue
        xyxy = _xywh_to_xyxy(annotation.get("bbox", [0, 0, 0, 0]))
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            {
                "image_id": int(annotation["image_id"]),
                "category_id": category_id,
                "bbox_xyxy": xyxy,
                "area": float(annotation.get("area", _box_area_xyxy(xyxy))),
                "ellipse": [float(value) for value in annotation["ellipse"]]
                if isinstance(annotation.get("ellipse"), list) and len(annotation["ellipse"]) == 5
                else None,
            }
        )
    return images, gt_by_image


def _image_to_eval_tensor(
    image: Image.Image,
    image_size: int | tuple[int, int],
    *,
    resize_mode: str,
    pad_color: Any,
    size_divisor: int,
):
    return prepare_image_tensor(
        image,
        image_size,
        resize_mode=resize_mode,
        pad_color=pad_color,
        size_divisor=size_divisor,
    )


def _scale_eval_detections(
    detections: dict[str, torch.Tensor],
    *,
    input_hw: tuple[int, int],
    original_hw: tuple[int, int],
    geometry: Any | None = None,
) -> dict[str, torch.Tensor]:
    if geometry is not None:
        return scale_detections_to_original(
            detections,
            input_hw=input_hw,
            original_hw=original_hw,
            geometry=geometry,
        )
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    original_h, original_w = int(original_hw[0]), int(original_hw[1])
    scale_x = original_w / float(max(1, input_w))
    scale_y = original_h / float(max(1, input_h))
    scaled = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in detections.items()}
    boxes = scaled.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y
        boxes[:, 0::2].clamp_(0, float(original_w))
        boxes[:, 1::2].clamp_(0, float(original_h))
    ellipses = scaled.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        from src.models.dinov3_deimv2_ellipse import ellipse_to_aabb, scale_ellipses_anisotropic

        ellipses = scale_ellipses_anisotropic(ellipses, scale_x=scale_x, scale_y=scale_y)
        scaled["ellipses"] = ellipses
        ellipse_boxes = ellipse_to_aabb(ellipses)
        ellipse_boxes[:, 0::2].clamp_(0, float(original_w))
        ellipse_boxes[:, 1::2].clamp_(0, float(original_h))
        scaled["ellipse_boxes"] = ellipse_boxes
    return scaled


def _detections_to_eval_records(
    image_info: dict[str, Any],
    detections: dict[str, torch.Tensor],
    category_ids: list[int],
) -> list[dict[str, Any]]:
    image_id = int(image_info["id"])
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    labels = detections.get("labels")
    labels = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.zeros((boxes.shape[0],), dtype=torch.long)
    ellipses = detections.get("ellipses")
    ellipses = ellipses.detach().cpu() if isinstance(ellipses, torch.Tensor) else None
    rows = []
    for index in range(boxes.shape[0]):
        label_index = int(labels[index])
        category_id = category_ids[label_index] if 0 <= label_index < len(category_ids) else label_index + 1
        x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        row = {
                "image_id": image_id,
                "file_name": str(image_info.get("file_name", "")),
                "category_id": int(category_id),
                "score": float(scores[index]),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        if ellipses is not None and index < ellipses.shape[0]:
            ellipse = [float(value) for value in ellipses[index].tolist()]
            row["ellipse"] = ellipse
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width >= box_height:
                row["bbox_ellipse"] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5, box_width * 0.5, box_height * 0.5, 0.0]
            else:
                row["bbox_ellipse"] = [
                    (x1 + x2) * 0.5,
                    (y1 + y2) * 0.5,
                    box_height * 0.5,
                    box_width * 0.5,
                    -math.pi * 0.5,
                ]
        rows.append(row)
    return rows


def _average_precision_eval(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float,
) -> float:
    gt_category = {
        image_id: [item for item in rows if int(item["category_id"]) == int(category_id)]
        for image_id, rows in gt_by_image.items()
    }
    total_gt = sum(len(rows) for rows in gt_category.values())
    if total_gt == 0:
        return 0.0
    preds = [row for row in predictions if int(row["category_id"]) == int(category_id)]
    preds.sort(key=lambda row: float(row["score"]), reverse=True)
    if not preds:
        return 0.0
    matched: dict[int, set[int]] = defaultdict(set)
    tp: list[float] = []
    fp: list[float] = []
    for pred in preds:
        image_id = int(pred["image_id"])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_category.get(image_id, [])):
            if index in matched[image_id]:
                continue
            overlap = _iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(1.0e-9, tp_cum + fp_cum)
    ap = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        keep = recalls >= recall_threshold
        ap += float(precisions[keep].max()) if keep.any() else 0.0
    return ap / 101.0


def _threshold_metrics_eval(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    pred_rows = [row for row in predictions if float(row["score"]) >= score_threshold]
    pred_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    tp = fp = 0
    for pred in pred_rows:
        image_id = int(pred["image_id"])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_by_image.get(image_id, [])):
            if index in matched[image_id]:
                continue
            overlap = _iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            tp += 1
        else:
            fp += 1
    total_gt = sum(len(rows) for rows in gt_by_image.values())
    fn = max(0, total_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "threshold": float(score_threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def _validation_loss(
    model: torch.nn.Module,
    *,
    annotation_path: Path,
    image_root: Path,
    category_ids: list[int],
    device: str,
    split: str,
    image_size: int | tuple[int, int],
    resize_mode: str,
    pad_color: Any,
    size_divisor: int,
    batch_size: int,
    max_images: int | None,
) -> float:
    dataset = FaceKeypointDataset(
        annotation_path=annotation_path,
        annotation_format="coco",
        image_root=image_root,
        split=split,
        image_size=image_size,
        resize_mode=resize_mode,
        pad_color=pad_color,
        size_divisor=size_divisor,
        include_negative=True,
        category_ids=category_ids,
    )
    if max_images is not None and len(dataset) > int(max_images):
        dataset = torch.utils.data.Subset(dataset, range(int(max_images)))
    if len(dataset) == 0:
        raise RuntimeError(f"Validation dataset is empty for split={split!r}")

    loader = DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_face_keypoint_batch,
    )
    total_loss = 0.0
    sample_count = 0
    with _validation_loss_mode(model):
        for images, targets in loader:
            images = images.to(device, non_blocking=str(device).startswith("cuda"))
            targets = move_targets_to_device(targets, device, non_blocking=str(device).startswith("cuda"))
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                loss_dict = model(images, targets)
            if not isinstance(loss_dict, dict) or "loss_total" not in loss_dict:
                raise RuntimeError("Model did not return loss_total while computing validation loss")
            current_batch_size = len(targets)
            total_loss += float(loss_dict["loss_total"].detach().cpu()) * current_batch_size
            sample_count += current_batch_size
    return total_loss / max(1, sample_count)


@contextmanager
def _validation_loss_mode(model: torch.nn.Module):
    """Enable detector/decoder loss branches without changing frozen eval children.

    DEIM decoders only emit auxiliary criterion inputs when their own training
    flag is true. Calling ``model.train()`` would also enable stochastic layers
    and update normalization statistics throughout the EMA model. Toggling only
    the wrapper and decoder flags produces the required loss outputs while the
    backbone, encoder, and decoder children remain in deterministic eval mode.
    """

    previous_model_training = bool(model.training)
    query_head = getattr(model, "query_head", None)
    previous_query_head_training = (
        bool(query_head.training)
        if isinstance(query_head, torch.nn.Module)
        else None
    )
    decoder = getattr(model, "decoder", None)
    previous_decoder_training = bool(decoder.training) if isinstance(decoder, torch.nn.Module) else None
    depth_decoder = (
        getattr(decoder, "decoder", None)
        if isinstance(decoder, torch.nn.Module)
        and bool(getattr(decoder, "depth_supervision_layers", ()))
        else None
    )
    previous_depth_decoder_training = (
        bool(depth_decoder.training)
        if isinstance(depth_decoder, torch.nn.Module)
        else None
    )
    model.training = True
    if isinstance(query_head, torch.nn.Module):
        # Co-DINO creates CDN queries and dn_meta in the query-head forward
        # only while this wrapper is in training mode. Toggle the wrapper
        # itself, not its dropout/normalization children, so validation-loss
        # decoding remains deterministic.
        query_head.training = True
    if isinstance(decoder, torch.nn.Module):
        decoder.training = True
    if isinstance(depth_decoder, torch.nn.Module):
        # Depth-specific auxiliary/consistency losses require all decoder
        # outputs. Toggle only the nested wrapper flag; its attention/dropout
        # children remain in deterministic eval mode.
        depth_decoder.training = True
    try:
        yield
    finally:
        model.training = previous_model_training
        if (
            isinstance(query_head, torch.nn.Module)
            and previous_query_head_training is not None
        ):
            query_head.training = previous_query_head_training
        if isinstance(decoder, torch.nn.Module) and previous_decoder_training is not None:
            decoder.training = previous_decoder_training
        if (
            isinstance(depth_decoder, torch.nn.Module)
            and previous_depth_decoder_training is not None
        ):
            depth_decoder.training = previous_depth_decoder_training


def validate_detection_split(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    device: str,
    split: str,
    image_size: int | tuple[int, int],
    fixed_threshold: float,
    score_min: float,
    max_images: int | None,
    batch_size: int = 1,
    predictions_output: Path | None = None,
    compute_validation_loss: bool = True,
) -> dict[str, Any]:
    annotation_path, image_root = _resolve_eval_data_paths(config, split)
    category_ids = _config_category_ids(config)
    data_cfg = dict(config.get("data", {}) or {})
    resize_mode = str(data_cfg.get("resize_mode", "stretch"))
    pad_color = data_cfg.get("pad_color")
    size_divisor = int(data_cfg.get("size_divisor", 32))
    images, gt_by_image = _load_coco_eval(annotation_path, category_ids)
    if max_images is not None:
        images = images[: int(max_images)]
        keep_ids = {int(image["id"]) for image in images}
        gt_by_image = {image_id: rows for image_id, rows in gt_by_image.items() if image_id in keep_ids}

    was_training = model.training
    old_model_score = getattr(model, "score_threshold", None)
    old_post_score = getattr(getattr(model, "postprocessor", None), "score_threshold", None)
    if hasattr(model, "score_threshold"):
        model.score_threshold = float(score_min)
    if hasattr(model, "postprocessor") and hasattr(model.postprocessor, "score_threshold"):
        model.postprocessor.score_threshold = float(score_min)
    model.eval()

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        val_loss = (
            _validation_loss(
                model,
                annotation_path=annotation_path,
                image_root=image_root,
                category_ids=category_ids,
                device=device,
                split=split,
                image_size=image_size,
                resize_mode=resize_mode,
                pad_color=pad_color,
                size_divisor=size_divisor,
                batch_size=batch_size,
                max_images=max_images,
            )
            if compute_validation_loss
            else None
        )
        for image_info in images:
            image_path = image_root / str(image_info.get("file_name", ""))
            image = Image.open(image_path).convert("RGB")
            (
                tensor,
                input_hw,
                original_hw,
                geometry,
                padding_mask,
            ) = _image_to_eval_tensor(
                image,
                image_size,
                resize_mode=resize_mode,
                pad_color=pad_color,
                size_divisor=size_divisor,
            )
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                detections = model(
                    tensor.to(device, non_blocking=True),
                    padding_mask=(
                        padding_mask.unsqueeze(0).to(device, non_blocking=True)
                        if padding_mask is not None
                        else None
                    ),
                )[0]
            detections = _scale_eval_detections(
                detections,
                input_hw=input_hw,
                original_hw=original_hw,
                geometry=geometry,
            )
            predictions.extend(_detections_to_eval_records(image_info, detections, category_ids))
    finally:
        if old_model_score is not None and hasattr(model, "score_threshold"):
            model.score_threshold = old_model_score
        if old_post_score is not None and hasattr(model, "postprocessor") and hasattr(model.postprocessor, "score_threshold"):
            model.postprocessor.score_threshold = old_post_score
        model.train(was_training)
    elapsed = time.perf_counter() - started
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_output.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    iou_thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
    ap_by_threshold = [
        _average_precision_eval(gt_by_image, predictions, category_id=category_id, iou_threshold=threshold)
        for category_id in category_ids
        for threshold in iou_thresholds
    ]
    ap50_values = [
        _average_precision_eval(gt_by_image, predictions, category_id=category_id, iou_threshold=0.50)
        for category_id in category_ids
    ]
    ap75_values = [
        _average_precision_eval(gt_by_image, predictions, category_id=category_id, iou_threshold=0.75)
        for category_id in category_ids
    ]
    fixed_metrics = _threshold_metrics_eval(gt_by_image, predictions, score_threshold=fixed_threshold)
    threshold_rows = [
        _threshold_metrics_eval(gt_by_image, predictions, score_threshold=round(float(value), 2))
        for value in np.arange(0.01, 0.96, 0.01)
    ]
    best_metrics = max(threshold_rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])), default=fixed_metrics)
    metrics = {
        "split": split,
        "num_images": len(images),
        "num_gt": sum(len(rows) for rows in gt_by_image.values()),
        "loss": val_loss,
        "mAP": float(np.mean(ap_by_threshold)) if ap_by_threshold else 0.0,
        "AP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "AP75": float(np.mean(ap75_values)) if ap75_values else 0.0,
        "precision": fixed_metrics["precision"],
        "recall": fixed_metrics["recall"],
        "f1": fixed_metrics["f1"],
        "fixed_threshold": float(fixed_threshold),
        "best_threshold": best_metrics["threshold"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f1": best_metrics["f1"],
        "elapsed_seconds": elapsed,
    }
    model_type = str(config.get("model", {}).get("type", "")).lower()
    if model_type in {
        "dinov3_deimv2_ellipse",
        "dino_deimv2_ellipse",
        "dinov3_codino_official_ellipse",
        "dinov3_codino_ellipse",
        "dino_codino_ellipse",
        "co_dino_ellipse",
    }:
        from src.evaluation import ellipse_detection_metrics

        metrics.update(
            ellipse_detection_metrics(
                gt_by_image,
                predictions,
                category_ids=category_ids,
                fixed_threshold=fixed_threshold,
                circle_ratio_threshold=float(config.get("loss", {}).get("ellipse", {}).get("circle_ratio_threshold", 1.05)),
            )
        )
    return metrics


def build_dataset(config: dict[str, Any], *, force_synthetic: bool = False):
    data_cfg = config.get("data", {})
    image_size = _configured_image_size(config)
    data_root = data_cfg.get("data_root")
    annotation_path = data_cfg.get("annotation_path")
    synthetic_if_missing = bool(data_cfg.get("synthetic_if_missing", True))
    missing_real_data = not data_root and not annotation_path
    if force_synthetic or (missing_real_data and synthetic_if_missing):
        dataset = SyntheticFaceKeypointDataset(
            length=4,
            image_size=(
                int(image_size)
                if isinstance(image_size, int)
                else (int(image_size[0]), int(image_size[1]))
            ),
        )
    else:
        dataset = FaceKeypointDataset(
            data_root=data_root,
            split=str(data_cfg.get("split", "train")),
            annotation_path=annotation_path,
            annotation_format=str(data_cfg.get("format", "auto")),
            image_root=data_cfg.get("image_root"),
            metadata_path=data_cfg.get("metadata_path"),
            image_size=image_size,
            resize_mode=str(data_cfg.get("resize_mode", "stretch")),
            pad_color=data_cfg.get("pad_color"),
            size_divisor=int(data_cfg.get("size_divisor", 32)),
            include_negative=bool(data_cfg.get("include_negative", True)),
            category_ids=data_cfg.get("category_ids"),
        )
    aug_cfg = config.get("augmentation", {})
    if str(data_cfg.get("split", "train")) == "train" and bool(aug_cfg.get("enabled", False)):
        dataset = AugmentedFaceKeypointDataset(dataset, aug_cfg, seed=int(config.get("train", {}).get("seed", 42)))
    return dataset


def subset_training_dataset(dataset, indices):
    """Bound all augmentation sources, retaining the outer epoch-aware wrapper."""
    if isinstance(dataset, AugmentedFaceKeypointDataset):
        bounded = AugmentedFaceKeypointDataset(
            Subset(dataset.base_dataset, indices), dataset.augmentation, seed=dataset.seed)
        bounded.set_epoch(dataset.epoch, dataset.total_epochs)
        return bounded
    return Subset(dataset, indices)


def move_targets_to_device(targets: list[dict[str, Any]], device: str, *, non_blocking: bool = False) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        updated = dict(target)
        for key in (
            "boxes",
            "ellipses",
            "labels",
            "keypoints",
            "keypoint_mask",
            "keypoint_visibility",
            "face_visibility",
            "padding_mask",
            "valid_image_region_xyxy",
        ):
            value = updated.get(key)
            if isinstance(value, torch.Tensor):
                updated[key] = value.to(device, non_blocking=non_blocking)
        moved.append(updated)
    return moved


def apply_tiny_model_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    model = config.setdefault("model", {})
    model_type = str(model.get("type", "")).lower()
    backbone = model.setdefault("backbone", {})
    backbone["embed_dim"] = 64
    backbone["dummy"] = True
    backbone["pretrained"] = False
    backbone["hub_repo"] = None
    backbone["hub_model"] = None
    backbone["hub_module"] = None
    backbone["weights"] = None
    backbone["strict_weights"] = False
    backbone["token_key"] = None
    backbone["call_method"] = "auto"
    if model_type.startswith("dinov3_codino_official"):
        model["init_checkpoint"] = None
        model.setdefault("neck", {})["in_channels"] = [64]
        model.setdefault("query_head", {}).update(embed_dims=64, num_queries=12,
            num_encoder_layers=1, num_decoder_layers=2, feedforward_channels=128,
            num_denoising_queries=10)
        return config
    if model_type in {
        "dinov3_codino_official",
        "dinov3_codino_official_ellipse",
        "dinov3_codino",
        "dino_codino",
        "co_dino",
        "dinov3_codino_ellipse",
        "dino_codino_ellipse",
        "co_dino_ellipse",
    }:
        # MMDetection 3.3's DinoTransformerDecoder creates its reference-point
        # sine embedding with the canonical 256-wide contract. Keep that
        # width in smoke runs and shrink depth/query counts instead.
        model.setdefault("adapter", {})["out_channels"] = 256
        model.setdefault("adapter", {})["hidden_dim"] = 256
        neck = model.setdefault("neck", {})
        neck["in_channels"] = [256, 256, 256]
        neck["out_channels"] = 256
        query = model.setdefault("query_head", {})
        query["embed_dims"] = 256
        query["num_queries"] = min(16, int(query.get("num_queries", 16)))
        query["num_encoder_layers"] = 1
        query["num_decoder_layers"] = 1
        query["feedforward_channels"] = 256
        query["num_denoising_queries"] = min(
            10, int(query.get("num_denoising_queries", 10))
        )
        o2m = model.setdefault("o2m_head", {})
        o2m["feat_channels"] = 256
        o2m["stacked_convs"] = 1
        o2m["max_positive_queries"] = min(
            16, int(o2m.get("max_positive_queries", 16))
        )
        return config
    model.setdefault("adapter", {})["out_channels"] = 64
    model.setdefault("adapter", {})["hidden_dim"] = 64
    model.setdefault("neck", {})["out_channels"] = 64
    encoder = model.setdefault("encoder", {})
    encoder["in_channels"] = [64, 64, 64]
    encoder["hidden_dim"] = 64
    encoder["nhead"] = 4
    encoder["dim_feedforward"] = 256
    decoder = model.setdefault("decoder", {})
    decoder["feat_channels"] = [64, 64, 64]
    decoder["hidden_dim"] = 64
    decoder["nhead"] = 4
    decoder["dim_feedforward"] = 256
    decoder["num_layers"] = min(1, int(decoder.get("num_layers", 1)))
    decoder["num_queries"] = min(16, int(decoder.get("num_queries", 16)))
    head = model.setdefault("head", {})
    head["head_channels"] = 64
    head["num_convs"] = 1
    return config


def trainable_parameter_groups(model: torch.nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    train_cfg = config.get("train", {})
    optimizer_cfg = dict(config.get("optimizer", train_cfg.get("optimizer", {})) or {})
    regex_specs = train_cfg.get("param_groups") or optimizer_cfg.get("params")
    if isinstance(regex_specs, list) and regex_specs:
        base_lr = float(train_cfg.get("lr", optimizer_cfg.get("lr", train_cfg.get("lr_head", 1.0e-4))))
        base_weight_decay = float(train_cfg.get("weight_decay", optimizer_cfg.get("weight_decay", 0.05)))
        named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        used: set[int] = set()
        groups: list[dict[str, Any]] = []
        for index, spec in enumerate(regex_specs):
            if not isinstance(spec, dict) or not spec.get("params"):
                continue
            pattern = re.compile(str(spec["params"]))
            selected = [
                parameter
                for name, parameter in named_parameters
                if id(parameter) not in used and pattern.search(name)
            ]
            if not selected:
                continue
            for parameter in selected:
                used.add(id(parameter))
            groups.append(
                {
                    "params": selected,
                    "lr": float(spec.get("lr", base_lr)),
                    "weight_decay": float(spec.get("weight_decay", base_weight_decay)),
                    "name": str(spec.get("name", f"regex_group_{index}")),
                }
            )
        unmatched = [parameter for _name, parameter in named_parameters if id(parameter) not in used]
        if unmatched:
            groups.append(
                {
                    "params": unmatched,
                    "lr": base_lr,
                    "weight_decay": base_weight_decay,
                    "name": "regex_unmatched",
                }
            )
        if groups:
            return groups

    lr_head = float(train_cfg.get("lr_head", 1.0e-4))
    lr_backbone = float(train_cfg.get("lr_backbone", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    groups = []
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "adapter_fpn_head"})
    if backbone_params and lr_backbone > 0.0:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"})
    return groups


def _raw_dino_module(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return the DINO wrapper controlling freeze mode and its raw ViT."""

    backbone = getattr(model, "backbone", None)
    dino_wrapper = getattr(backbone, "dino", None)
    raw_dino = getattr(dino_wrapper, "backbone", None)
    if isinstance(dino_wrapper, torch.nn.Module) and isinstance(
        raw_dino, torch.nn.Module
    ):
        return dino_wrapper, raw_dino
    raw_dino = getattr(backbone, "dinov3", None)
    if isinstance(backbone, torch.nn.Module) and isinstance(
        raw_dino, torch.nn.Module
    ):
        return backbone, raw_dino
    raise RuntimeError(
        "staged_unfreeze requires a DINOv3 STA backbone exposing dino.backbone or dinov3"
    )


def _set_raw_dino_trainable(
    model: torch.nn.Module,
    *,
    trainable: bool,
) -> int:
    wrapper, raw_dino = _raw_dino_module(model)
    for parameter in raw_dino.parameters():
        parameter.requires_grad_(trainable)
    if hasattr(wrapper, "freeze"):
        wrapper.freeze = not trainable
    raw_dino.train(trainable and model.training)
    return sum(parameter.numel() for parameter in raw_dino.parameters())


def build_optimizer(
    parameter_groups: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    train_cfg = config.get("train", {})
    fused_requested = bool(train_cfg.get("fused_optimizer", str(device).startswith("cuda")))
    optimizer_meta = {"name": "AdamW", "fused_requested": fused_requested, "fused": False}
    if fused_requested and str(device).startswith("cuda"):
        try:
            optimizer = torch.optim.AdamW(parameter_groups, fused=True)
            optimizer_meta["fused"] = True
            return optimizer, optimizer_meta
        except (RuntimeError, TypeError) as error:
            optimizer_meta["fallback_reason"] = str(error)
    return torch.optim.AdamW(parameter_groups), optimizer_meta


def compile_trainable_modules(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    compile_cfg = dict(config.get("train", {}).get("torch_compile", {}))
    if not bool(compile_cfg.get("enabled", False)):
        return {"enabled": False}
    if not hasattr(torch, "compile"):
        return {"enabled": False, "fallback_reason": "torch.compile is unavailable"}

    mode = compile_cfg.get("mode", "reduce-overhead")
    fullgraph = bool(compile_cfg.get("fullgraph", False))
    dynamic = bool(compile_cfg.get("dynamic", False))
    compiled = []
    for name in ("adapter", "fpn", "encoder", "decoder", "head"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            setattr(model, name, torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic))
            compiled.append(name)
    return {
        "enabled": bool(compiled),
        "compiled_modules": compiled,
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }


def checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        cleaned[key.replace("._orig_mod.", ".").replace("_orig_mod.", "")] = value
    return cleaned


class ModelEMA:
    """Full-model EMA matching the official DEIMv2 optimization recipe."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.9999,
        warmups: int = 1000,
        start: int = 0,
    ) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.warmups = int(warmups)
        self.start = int(start)
        self.before_start = 0
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def current_decay(self) -> float:
        if self.warmups <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-float(self.updates) / float(self.warmups)))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if self.before_start < self.start:
            self.before_start += 1
            return
        self.updates += 1
        decay = self.current_decay()
        source_state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            source_value = source_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], *, updates: int = 0) -> None:
        self.module.load_state_dict(state_dict, strict=True)
        self.updates = int(updates)


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = config.get("train", {})
    scheduler_name = str(train_cfg.get("scheduler", "none")).lower()
    if scheduler_name in {"", "none", "null"}:
        return None
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    flat_epochs = int(train_cfg.get("flat_epochs", train_cfg.get("flat_epoch", max(warmup_epochs, epochs // 2))))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.01))

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return max(1e-3, float(epoch_number) / float(warmup_epochs))
        if scheduler_name in {"cosine", "cos", "cosine_annealing"}:
            denom = max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, float(epoch_number - warmup_epochs) / float(denom)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_name in {"flat_cosine", "flatcosine", "flatcos"}:
            if epoch_number <= flat_epochs:
                return 1.0
            denom = max(1, epochs - flat_epochs)
            progress = min(1.0, max(0.0, float(epoch_number - flat_epochs) / float(denom)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if scheduler_name in {"constant", "fixed"}:
            return 1.0
        raise ValueError(f"Unsupported train.scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def validate_full_training_backbone(config: dict[str, Any], *, allow_dummy_backbone: bool) -> None:
    backbone_cfg = config.get("model", {}).get("backbone", {})
    if bool(backbone_cfg.get("dummy", False)) and not allow_dummy_backbone:
        raise RuntimeError(
            "Refusing full training with model.backbone.dummy=true. "
            "Use a local DINO hub_repo/hub_model/weights config, or pass --allow-dummy-backbone only for debugging."
        )


def validate_production_training_contract(
    config: dict[str, Any],
    *,
    force_synthetic: bool,
    max_train_batches: int | None,
    max_train_samples: int | None = None,
) -> None:
    train_cfg = dict(config.get("train", {}))
    if not bool(train_cfg.get("production_mode", False)):
        return
    data_cfg = dict(config.get("data", {}))
    backbone_cfg = dict(config.get("model", {}).get("backbone", {}))
    validation_cfg = dict(train_cfg.get("validation", {}) or {})
    ema_cfg = dict(train_cfg.get("ema", {}) or {})
    early_stopping_cfg = dict(train_cfg.get("early_stopping", {}) or {})
    backbone_policy = str(train_cfg.get("backbone_policy", "full_unfreeze")).lower()
    if force_synthetic or bool(data_cfg.get("synthetic_if_missing", False)):
        raise RuntimeError("Production training forbids synthetic data and synthetic_if_missing")
    if max_train_batches is not None or max_train_samples is not None:
        raise RuntimeError(
            "Production training forbids bounded batches and dataset truncation"
        )
    if not bool(data_cfg.get("leakage_audit", {}).get("enabled", False)):
        raise RuntimeError("Production training requires data.leakage_audit.enabled=true")
    if int(train_cfg.get("epochs", 0)) < int(train_cfg.get("minimum_epochs", 68)):
        raise RuntimeError("Production training epochs are below train.minimum_epochs")
    image_size = _configured_image_size(config)
    minimum_image_size = int(train_cfg.get("minimum_image_size", 960))
    actual_image_size = int(image_size if isinstance(image_size, int) else min(image_size))
    if actual_image_size < minimum_image_size:
        raise RuntimeError(
            f"Production training requires image_size >= {minimum_image_size}, got {actual_image_size}"
        )
    normalized_image_size = normalize_image_size(image_size)
    if normalized_image_size is None:
        raise RuntimeError("Production training requires an explicit image_size")
    required_divisor = int(data_cfg.get("size_divisor", 1))
    if any(value % required_divisor != 0 for value in normalized_image_size):
        raise RuntimeError(
            f"Production image_size {normalized_image_size} must be divisible by {required_divisor}"
        )
    minimum_image_pixels = int(train_cfg.get("minimum_image_pixels", 0) or 0)
    if math.prod(normalized_image_size) < minimum_image_pixels:
        raise RuntimeError(
            f"Production image_size {normalized_image_size} has fewer than "
            f"{minimum_image_pixels} pixels"
        )
    if backbone_policy == "full_unfreeze":
        if bool(backbone_cfg.get("freeze", True)):
            raise RuntimeError("backbone_policy=full_unfreeze requires model.backbone.freeze=false")
    elif backbone_policy == "freeze":
        if not bool(backbone_cfg.get("freeze", False)):
            raise RuntimeError("backbone_policy=freeze requires model.backbone.freeze=true")
        if int(backbone_cfg.get("trainable_last_blocks", 0)) != 0:
            raise RuntimeError("backbone_policy=freeze requires trainable_last_blocks=0")
    elif backbone_policy == "staged_unfreeze":
        freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
        if not bool(backbone_cfg.get("freeze", False)):
            raise RuntimeError(
                "backbone_policy=staged_unfreeze requires model.backbone.freeze=true"
            )
        if freeze_epochs <= 0 or freeze_epochs >= int(train_cfg.get("epochs", 0)):
            raise RuntimeError(
                "staged_unfreeze requires 0 < train.freeze_backbone_epochs < train.epochs"
            )
        if float(train_cfg.get("lr_backbone", 0.0)) <= 0.0:
            raise RuntimeError(
                "staged_unfreeze requires a positive train.lr_backbone"
            )
    else:
        raise RuntimeError(
            "Production train.backbone_policy must be 'full_unfreeze', 'freeze', "
            "or 'staged_unfreeze', "
            f"got {backbone_policy!r}"
        )
    if not bool(validation_cfg.get("enabled", False)) or int(validation_cfg.get("every", 0)) <= 0:
        raise RuntimeError("Production training requires periodic validation")
    if not bool(ema_cfg.get("enabled", False)):
        raise RuntimeError("Production training requires EMA for validation and deployable checkpoints")
    if bool(early_stopping_cfg.get("enabled", False)):
        patience = int(early_stopping_cfg.get("patience_validations", 0))
        minimum_epoch = int(early_stopping_cfg.get("minimum_epoch", 0))
        if patience <= 0:
            raise RuntimeError("Production early stopping requires patience_validations > 0")
        if minimum_epoch < int(train_cfg.get("minimum_epochs", 68)):
            raise RuntimeError("Production early stopping cannot precede train.minimum_epochs")
        if float(early_stopping_cfg.get("min_delta", 0.0)) < 0.0:
            raise RuntimeError("Production early stopping requires min_delta >= 0")


def positive_subset_indices(dataset, max_samples: int) -> list[int]:
    indices: list[int] = []
    for index in range(len(dataset)):
        _image, target = dataset[index]
        if int(target["boxes"].shape[0]) > 0:
            indices.append(index)
        if len(indices) >= max_samples:
            break
    if not indices:
        raise RuntimeError("Could not find positive samples for overfit smoke.")
    return indices


def make_output_dir(config: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    train_cfg = config.get("train", {})
    base = Path(train_cfg.get("output_root", "runs/dino"))
    name = train_cfg.get("name") or datetime.now().strftime("dino_%Y%m%d_%H%M%S")
    return base / str(name)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    ema: ModelEMA | None = None,
    amp_runtime: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": checkpoint_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
        "metrics": metrics,
    }
    if ema is not None:
        payload["ema"] = checkpoint_state_dict(ema.module)
        payload["ema_updates"] = int(ema.updates)
    if amp_runtime is not None:
        payload["amp_runtime"] = amp_runtime
    torch.save(payload, path)


def dataloader_kwargs(
    *,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str,
    generator: torch.Generator | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_cfg = train_cfg or {}
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "collate_fn": collate_face_keypoint_batch,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return kwargs


def normalize_image_size(value: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"image size must be int or [height, width], got {value}")
    return int(value[0]), int(value[1])


def set_dataset_image_size(dataset, image_size: tuple[int, int] | None) -> None:
    if image_size is None:
        return
    if isinstance(dataset, torch.utils.data.Subset):
        set_dataset_image_size(dataset.dataset, image_size)
        return
    if hasattr(dataset, "base_dataset"):
        set_dataset_image_size(dataset.base_dataset, image_size)
        return
    if hasattr(dataset, "image_size"):
        dataset.image_size = image_size[0] if image_size[0] == image_size[1] else image_size


def progressive_image_size_for_epoch(config: dict[str, Any], epoch: int, total_epochs: int) -> tuple[int, int] | None:
    train_cfg = config.get("train", {})
    progressive_cfg = dict(train_cfg.get("progressive_resizing", {}))
    if not bool(progressive_cfg.get("enabled", False)):
        return normalize_image_size(_configured_image_size(config))
    stages = progressive_cfg.get("stages")
    if isinstance(stages, list) and stages:
        remaining = int(epoch)
        for stage in stages:
            stage_epochs = int(stage.get("epochs", 1))
            if remaining <= stage_epochs:
                return normalize_image_size(stage.get("image_size", stage.get("size")))
            remaining -= stage_epochs
        return normalize_image_size(stages[-1].get("image_size", stages[-1].get("size")))

    start_size = normalize_image_size(progressive_cfg.get("start_size"))
    end_size = normalize_image_size(progressive_cfg.get("end_size", _configured_image_size(config)))
    warmup_epochs = max(1, int(progressive_cfg.get("warmup_epochs", max(1, total_epochs // 3))))
    if start_size is None or end_size is None:
        return end_size
    progress = min(1.0, max(0.0, float(epoch - 1) / float(max(1, warmup_epochs - 1))))
    height = int(round(start_size[0] + (end_size[0] - start_size[0]) * progress))
    width = int(round(start_size[1] + (end_size[1] - start_size[1]) * progress))
    return height, width


def safe_cache_name(value: Any) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{text[:80].strip('._') or 'sample'}_{digest}.pt"


def feature_cache_supported(dataset, config: dict[str, Any]) -> tuple[bool, str]:
    model_type = str(config.get("model", {}).get("type", "")).lower()
    if model_type in {
        "dinov3_deimv2",
        "dino_deimv2",
        "dinov3_codino_official",
        "dinov3_codino_official_ellipse",
        "dinov3_codino",
        "dino_codino",
        "co_dino",
        "dinov3_codino_ellipse",
        "dino_codino_ellipse",
        "co_dino_ellipse",
    }:
        return False, "the configured STA detector requires raw images"
    if isinstance(dataset, AugmentedFaceKeypointDataset):
        return False, "train-time augmentation is enabled"
    if bool(config.get("train", {}).get("progressive_resizing", {}).get("enabled", False)):
        return False, "progressive resizing is enabled"
    backbone_cfg = config.get("model", {}).get("backbone", {})
    if not bool(backbone_cfg.get("freeze", True)):
        return False, "backbone is not frozen"
    return True, ""


class DinoFeatureCache:
    def __init__(
        self,
        *,
        cache_dir: Path,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, image_id: Any) -> Path:
        return self.cache_dir / safe_cache_name(image_id)

    def is_complete(self, dataset) -> bool:
        for index in range(len(dataset)):
            _image, target = dataset[index]
            if not self.path_for(target.get("image_id", index)).exists():
                return False
        return True

    def build(
        self,
        *,
        model: torch.nn.Module,
        dataset,
        batch_size: int,
        num_workers: int,
        device: str,
        train_cfg: dict[str, Any],
        use_amp: bool,
    ) -> int:
        loader = DataLoader(**dataloader_kwargs(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device=device,
            train_cfg=train_cfg,
        ))
        created = 0
        model.backbone.eval()
        with torch.no_grad():
            for images, targets in loader:
                missing_indices = [index for index, target in enumerate(targets) if not self.path_for(target.get("image_id", index)).exists()]
                if not missing_indices:
                    continue
                images = images.to(device, non_blocking=str(device).startswith("cuda"))
                with torch.amp.autocast("cuda", enabled=use_amp and str(device).startswith("cuda")):
                    features = model.backbone(images)
                features = features.detach().to(self.dtype).cpu()
                for batch_index, target in enumerate(targets):
                    path = self.path_for(target.get("image_id", batch_index))
                    if path.exists():
                        continue
                    # A batch slice can be contiguous while still sharing the full batch
                    # storage. Clone it so each per-image cache file serializes only that
                    # image's feature tensor instead of duplicating the complete batch.
                    torch.save(features[batch_index].clone(), path)
                    created += 1
        return created

    def load_batch(self, targets: list[dict[str, Any]], *, device: str) -> torch.Tensor:
        features = []
        for index, target in enumerate(targets):
            path = self.path_for(target.get("image_id", index))
            if not path.exists():
                raise FileNotFoundError(f"Missing cached DINO feature: {path}")
            features.append(torch.load(path, map_location="cpu"))
        return torch.stack(features, dim=0).to(device, non_blocking=str(device).startswith("cuda"))


def maybe_build_feature_cache(
    *,
    model: torch.nn.Module,
    dataset,
    output_dir: Path,
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: str,
    use_amp: bool,
) -> tuple[DinoFeatureCache | None, dict[str, Any]]:
    cache_cfg = dict(config.get("train", {}).get("feature_cache", {}))
    if not bool(cache_cfg.get("enabled", False)):
        return None, {"enabled": False}
    supported, reason = feature_cache_supported(dataset, config)
    if not supported:
        return None, {"enabled": False, "requested": True, "fallback_reason": reason}
    dtype_name = str(cache_cfg.get("dtype", "float16")).lower()
    dtype = torch.float32 if dtype_name in {"float32", "fp32"} else torch.float16
    cache_dir = Path(cache_cfg.get("dir") or (output_dir / "feature_cache"))
    cache = DinoFeatureCache(cache_dir=cache_dir, dtype=dtype)
    started = time.monotonic()
    created = 0
    if not bool(cache_cfg.get("reuse_existing", True)) or not cache.is_complete(dataset):
        created = cache.build(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            train_cfg=config.get("train", {}),
            use_amp=use_amp,
        )
    elapsed = round(time.monotonic() - started, 3)
    return cache, {
        "enabled": True,
        "dir": str(cache_dir),
        "dtype": dtype_name,
        "created": created,
        "build_seconds": elapsed,
        "samples": len(dataset),
    }


def make_grad_scaler(*, enabled: bool, initial_scale: float | None = None):
    if initial_scale is not None and (not math.isfinite(float(initial_scale)) or float(initial_scale) <= 0):
        raise ValueError("AMP initial scale must be finite and positive")
    kwargs = {} if initial_scale is None else {"init_scale": initial_scale}
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled, **kwargs)
    return torch.cuda.amp.GradScaler(enabled=enabled, **kwargs)


def dry_run(config: dict[str, Any], *, device: str, force_synthetic: bool) -> int:
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; set data.synthetic_if_missing=true or provide local data.")
    batch_size = min(int(config.get("train", {}).get("batch_size", 2)), len(dataset))
    loader = DataLoader(**dataloader_kwargs(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        device=device,
    ))
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = move_targets_to_device(targets, device, non_blocking=str(device).startswith("cuda"))

    model = build_detector_from_config(config).to(device)
    model.train()
    losses = model(images, targets)
    assert isinstance(losses, dict)
    finite = all(torch.isfinite(value).all().item() for value in losses.values())
    print("dry_run: ok")
    print(f"batch_images: {tuple(images.shape)}")
    print(f"targets: {[int(target['boxes'].shape[0]) for target in targets]}")
    print({key: float(value.detach().cpu()) for key, value in losses.items()})
    return 0 if finite else 1


def trim_uncommitted_epoch_metrics(output_dir: Path, completed_epoch: int) -> None:
    """Rollback failed official-SFP epochs to the last completed checkpoint."""
    for name in ("metrics.csv", "val_metrics.csv"):
        path = output_dir / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        kept = [row for row in rows if int(row["epoch"]) <= completed_epoch]
        if len(kept) == len(rows):
            continue
        pending = path.with_suffix(".recovery_tmp")
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        pending.replace(path)


def train(
    config: dict[str, Any],
    *,
    device: str,
    force_synthetic: bool,
    output_dir: Path,
    resume_from: Path | None,
    save_every: int,
    use_amp: bool,
    max_train_batches: int | None,
    max_train_samples: int | None,
    allow_dummy_backbone: bool,
    wandb_run: Any | None = None,
) -> int:
    validate_full_training_backbone(config, allow_dummy_backbone=allow_dummy_backbone)
    validate_production_training_contract(
        config,
        force_synthetic=force_synthetic,
        max_train_batches=max_train_batches,
        max_train_samples=max_train_samples,
    )
    seed = int(config.get("train", {}).get("seed", 42))
    set_reproducible_seed(seed)
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    if max_train_samples is not None:
        if int(max_train_samples) < 1:
            raise ValueError("max_train_samples must be positive")
        source_dataset_size = len(dataset)
        selected_count = min(int(max_train_samples), source_dataset_size)
        subset_strategy = str(
            config.get("train", {}).get("bounded_subset_strategy", "head")
        ).lower()
        if subset_strategy == "head":
            selected_indices = list(range(selected_count))
        elif subset_strategy == "seeded_random":
            subset_generator = torch.Generator()
            subset_generator.manual_seed(seed)
            selected_indices = torch.randperm(
                source_dataset_size,
                generator=subset_generator,
            )[:selected_count].sort().values.tolist()
        elif subset_strategy == "crowded":
            # Isolated preflight only: include the densest real train images.
            base = getattr(dataset, "base_dataset", dataset)
            records = getattr(base, "samples", None)
            if records is None or not all("annotations" in item for item in records):
                raise ValueError("crowded subset requires COCO train records")
            selected_indices = sorted(range(source_dataset_size),
                key=lambda index: (-len(records[index]["annotations"]), index))[:selected_count]
        else:
            raise ValueError(
                "train.bounded_subset_strategy must be head, seeded_random or crowded; "
                f"got {subset_strategy!r}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bounded_train_subset.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy": subset_strategy,
                    "seed": seed,
                    "source_dataset_size": source_dataset_size,
                    "selected_count": selected_count,
                    "indices": selected_indices,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = subset_training_dataset(dataset, selected_indices)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; set data.synthetic_if_missing=true or provide local data.")
    train_cfg = config.get("train", {})
    early_stopping_cfg = dict(train_cfg.get("early_stopping", {}) or {})
    early_stopping_monitor = str(early_stopping_cfg.get("monitor", "val/mAP50_95"))
    validation_metric_key = {
        "val/mAP50_95": "mAP",
        "val/ellipse_mAP50_95": "ellipse_mAP",
    }.get(early_stopping_monitor, early_stopping_monitor.removeprefix("val/"))
    best_validation_checkpoint_name = str(
        early_stopping_cfg.get(
            "checkpoint_name",
            "best_val_ellipse_map.pt" if validation_metric_key == "ellipse_mAP" else "best_val_map.pt",
        )
    )
    batch_size = min(int(train_cfg.get("batch_size", 4)), len(dataset))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 100))
    generator = torch.Generator()
    generator.manual_seed(seed)

    amp_policy = calibration_policy(config)
    memory_budget = configure_memory_budget(config, device)
    if memory_budget is not None:
        write_json_atomic(output_dir / "cuda_memory_budget.json", memory_budget)
        if wandb_run is not None:
            wandb_run.config.update({"cuda_memory_budget": memory_budget}, allow_val_change=True)
    model = build_detector_from_config(config)
    if train_cfg.get("continuation_source"):
        from src.codino_continuation import load_continuation_source
        initial = load_continuation_source(config)
        model.load_state_dict(initial["model"], strict=True)
        del initial
    model = model.to(device)
    model.train()
    compile_meta = compile_trainable_modules(model, config)
    transfer_report = getattr(model, "checkpoint_transfer_report", None)
    if transfer_report is not None:
        (output_dir / "checkpoint_transfer.json").write_text(
            json.dumps(transfer_report, indent=2) + "\n", encoding="utf-8")
        if wandb_run is not None:
            wandb_run.config.update({"checkpoint_transfer": transfer_report}, allow_val_change=True)
    backbone_policy = str(train_cfg.get("backbone_policy", "full_unfreeze")).lower()
    staged_unfreeze = backbone_policy == "staged_unfreeze"
    freeze_backbone_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    ema_cfg = dict(train_cfg.get("ema", {}) or {})
    ema = (
        ModelEMA(
            model,
            decay=float(ema_cfg.get("decay", 0.9999)),
            warmups=int(ema_cfg.get("warmups", 1000)),
            start=int(ema_cfg.get("start", 0)),
        )
        if bool(ema_cfg.get("enabled", False))
        else None
    )
    # Staged parameters must be present in the optimizer from the beginning so
    # resume/scheduler state stays stable. They receive no gradients while the
    # DINO wrapper remains frozen and are activated at the declared epoch.
    if staged_unfreeze:
        _set_raw_dino_trainable(model, trainable=True)
    parameter_groups = trainable_parameter_groups(model, config)
    if staged_unfreeze:
        _set_raw_dino_trainable(model, trainable=False)
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found.")
    optimizer, optimizer_meta = build_optimizer(parameter_groups, config, device=device)
    scheduler = build_epoch_scheduler(optimizer, config, epochs=epochs)
    scaler = make_grad_scaler(enabled=use_amp and str(device).startswith("cuda"),
                             initial_scale=(amp_policy["initial_scale"] if amp_policy
                                            else train_cfg.get("amp_initial_scale")))
    amp_runtime = (dict(policy=amp_policy, calibrated_states={}, scaler=scaler.state_dict())
                   if amp_policy else None)

    start_epoch = 1
    total_step = 0
    best_loss = float("inf")
    best_val_map = float("-inf")
    previous_elapsed = 0.0
    metrics_path = output_dir / "metrics.csv"
    official_recovery = bool(train_cfg.get("runtime_batch_attempts")) and str(
        config.get("model", {}).get("type", "")).startswith("dinov3_codino_official")
    if official_recovery and resume_from is None:
        trim_uncommitted_epoch_metrics(output_dir, 0)
    if resume_from is not None:
        # Continuation restores tensors into existing model/EMA/optimizer slots;
        # avoid materializing a second complete checkpoint on the GPU.
        checkpoint = torch.load(resume_from, map_location="cpu" if train_cfg.get("continuation_source") else device)
        if amp_runtime is not None:
            saved_amp = checkpoint.get("amp_runtime")
            if not saved_amp or saved_amp.get("policy") != amp_policy:
                raise RuntimeError("Cannot change AMP calibration policy or resume an unrecorded trial")
            amp_runtime = saved_amp
            scaler.load_state_dict(saved_amp["scaler"])
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        if ema is not None:
            ema_state = checkpoint.get("ema")
            if isinstance(ema_state, dict) and isinstance(ema_state.get("module"), dict):
                ema_state = ema_state["module"]
            if isinstance(ema_state, dict):
                ema.load_state_dict(ema_state, updates=int(checkpoint.get("ema_updates", 0)))
            else:
                ema.load_state_dict(checkpoint_state_dict(model), updates=int(checkpoint.get("step", 0)))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if official_recovery:
            trim_uncommitted_epoch_metrics(output_dir, completed_epoch)
        total_step = int(checkpoint.get("step", 0))
        if train_cfg.get("continuation_source"):
            del checkpoint
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if rows:
                previous_elapsed = float(rows[-1].get("elapsed_seconds") or 0.0)
                best_loss = min(float(row["loss_total"]) for row in rows if row.get("loss_total"))
        existing_val_metrics = output_dir / "val_metrics.csv"
        if existing_val_metrics.exists():
            with existing_val_metrics.open(encoding="utf-8") as file:
                val_rows = list(csv.DictReader(file))
            values = [float(row[validation_metric_key]) for row in val_rows if row.get(validation_metric_key)]
            if values:
                best_val_map = max(values)
        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step(completed_epoch)
        print(f"resume_from: {resume_from}")
        print(f"resume_epoch: {start_epoch}/{epochs}")
        if start_epoch > epochs:
            print(f"training_already_complete: {output_dir}")
            return 0

    if staged_unfreeze and start_epoch > freeze_backbone_epochs:
        unfreezed_parameters = _set_raw_dino_trainable(model, trainable=True)
        print(
            "staged_unfreeze_resume: "
            f"start_epoch={start_epoch} parameters={unfreezed_parameters}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    first_image_size = progressive_image_size_for_epoch(config, start_epoch, epochs)
    set_dataset_image_size(dataset, first_image_size)
    feature_cache, feature_cache_meta = maybe_build_feature_cache(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_amp=bool(scaler.is_enabled()),
    )
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainable_params = [parameter for group in parameter_groups for parameter in group["params"]]
    backbone_wrapper_fully_frozen = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    dino_module = getattr(model.backbone, "dinov3", None) or getattr(model.backbone, "dino", None)
    dino_backbone_frozen = bool(dino_module is not None) and all(
        not parameter.requires_grad for parameter in dino_module.parameters()
    )
    run_meta = {
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "requested_batch_size": int(train_cfg.get("batch_size", 4)),
        "epochs": epochs,
        "device": device,
        "amp": bool(scaler.is_enabled()),
        "amp_calibration": amp_policy,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if str(device).startswith("cuda") else False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if str(device).startswith("cuda") else False,
        "matmul_precision": str(train_cfg.get("matmul_precision", "")) if str(device).startswith("cuda") else "",
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "persistent_workers": bool(train_cfg.get("persistent_workers", True)) if num_workers > 0 else False,
        "prefetch_factor": int(train_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
        "optimizer": optimizer_meta,
        "ema": {
            "enabled": ema is not None,
            "decay": float(ema_cfg.get("decay", 0.9999)),
            "warmups": int(ema_cfg.get("warmups", 1000)),
            "start": int(ema_cfg.get("start", 0)),
        },
        "torch_compile": compile_meta,
        "feature_cache": feature_cache_meta,
        "progressive_resizing": dict(train_cfg.get("progressive_resizing", {"enabled": False})),
        "early_stopping": dict(train_cfg.get("early_stopping", {"enabled": False})),
        "seed": seed,
        "scheduler": str(train_cfg.get("scheduler", "none")),
        "warmup_epochs": int(train_cfg.get("warmup_epochs", 0)),
        "backbone_policy": backbone_policy,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "backbone_unfreeze_epoch": (
            freeze_backbone_epochs + 1 if staged_unfreeze else None
        ),
        "frozen_backbone": backbone_wrapper_fully_frozen,
        "backbone_wrapper_fully_frozen": backbone_wrapper_fully_frozen,
        "dino_backbone_frozen": dino_backbone_frozen,
        "trainable_parameters": count_parameters(trainable_params),
        "parameter_groups": [
            {"name": group.get("name", ""), "lr": group["lr"], "weight_decay": group["weight_decay"], "parameters": count_parameters(group["params"])}
            for group in parameter_groups
        ],
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    isolated_probe = bool(train_cfg.get("official_sfp_preflight_probe", False))
    runtime_progress = (RuntimeProgress(output_dir / "runtime_progress.json", device)
                        if config.get("model", {}).get("type") in OFFICIAL_TYPES and not isolated_probe
                        else None)
    if isolated_probe:
        if train_cfg.get("production_mode", False) or not str(device).startswith("cuda"):
            raise ValueError("Official-SFP timing probe must be isolated CUDA non-production")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    probe_started, probe_images, probe_train_seconds = None, 0, None
    started = time.monotonic()
    fieldnames = ["epoch", *LOSS_LOG_KEYS, "lr", "elapsed_seconds"]
    validation_cfg = dict(train_cfg.get("validation", {}) or {})
    validation_enabled = bool(validation_cfg.get("enabled", False))
    validation_every = int(validation_cfg.get("every", validation_cfg.get("val_every", 0)) or 0)
    validation_split = str(validation_cfg.get("split", "val"))
    validation_image_size_raw = validation_cfg.get("image_size", train_cfg.get("imgsz", config.get("data", {}).get("image_size", 896)))
    parsed_validation_size = parse_image_size(validation_image_size_raw)
    if parsed_validation_size is None:
        raise ValueError("validation.image_size must be configured")
    validation_image_size = parsed_validation_size
    validation_fixed_threshold = float(validation_cfg.get("fixed_threshold", validation_cfg.get("threshold", 0.30)))
    validation_score_min = float(validation_cfg.get("score_min", config.get("inference", {}).get("score_thr", 0.001)))
    validation_max_images = validation_cfg.get("max_images")
    validation_max_images = int(validation_max_images) if validation_max_images is not None else None
    val_metrics_path = output_dir / "val_metrics.csv"
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("patience_validations", 0))
    early_stopping_minimum_epoch = int(early_stopping_cfg.get("minimum_epoch", epochs + 1))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_best = float("-inf")
    early_stopping_best_epoch = 0
    early_stopping_bad_validations = 0
    if resume_from is not None and val_metrics_path.exists():
        with val_metrics_path.open(encoding="utf-8") as val_file:
            prior_val_rows = list(csv.DictReader(val_file))
        for prior_row in prior_val_rows:
            value = prior_row.get(validation_metric_key)
            if value in {None, ""}:
                continue
            prior_score = float(value)
            if prior_score > early_stopping_best + early_stopping_min_delta:
                early_stopping_best = prior_score
                early_stopping_best_epoch = int(prior_row["epoch"])
                early_stopping_bad_validations = 0
            else:
                early_stopping_bad_validations += 1
    early_stopping_triggered = False
    write_header = not (resume_from is not None and metrics_path.exists())
    mode = "a" if resume_from is not None and metrics_path.exists() else "w"
    loader = None
    loader_image_size: tuple[int, int] | None = None
    with metrics_path.open(mode, encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            if staged_unfreeze and epoch == freeze_backbone_epochs + 1:
                unfreezed_parameters = _set_raw_dino_trainable(
                    model, trainable=True
                )
                print(
                    "staged_unfreeze: "
                    f"epoch={epoch} parameters={unfreezed_parameters} "
                    f"lr_backbone={float(train_cfg.get('lr_backbone', 0.0))}"
                )
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch, int(train_cfg.get("augmentation_schedule_epochs", epochs)))
            epoch_image_size = progressive_image_size_for_epoch(config, epoch, epochs)
            if loader is None or epoch_image_size != loader_image_size:
                set_dataset_image_size(dataset, epoch_image_size)
                loader_image_size = epoch_image_size
                loader = DataLoader(**dataloader_kwargs(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    device=device,
                    generator=generator,
                    train_cfg=train_cfg,
                ))
            if amp_runtime is not None and scaler.is_enabled():
                state = "unfrozen" if any(p.requires_grad for p in model.backbone.parameters()) else "frozen"
                if state not in amp_runtime["calibrated_states"]:
                    if feature_cache is not None:
                        raise ValueError("Experimental calibration does not support cached features")
                    report_path = output_dir / f"amp_calibration_{state}.json"
                    with preserve_rng_and_buffers(model):
                        calibration_generator = torch.Generator()
                        calibration_generator.set_state(generator.get_state())
                        calibration_loader = iter(DataLoader(**dataloader_kwargs(
                            dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, device=device, generator=calibration_generator,
                            train_cfg=train_cfg)))
                        windows = []
                        for _ in range(amp_policy["required_finite_windows"]):
                            window = []
                            for _ in range(gradient_accumulation_steps):
                                try:
                                    window.append(next(calibration_loader))
                                except StopIteration:
                                    break
                            if not window:
                                window = windows[0]
                            windows.append(window)

                        def backward_calibration_window(index):
                            window = windows[index % len(windows)]
                            for cpu_images, cpu_targets in window:
                                calibration_images = cpu_images.to(device)
                                calibration_targets = move_targets_to_device(cpu_targets, device)
                                with torch.amp.autocast("cuda", enabled=True):
                                    calibration_loss = model(calibration_images, calibration_targets)["loss_total"]
                                if not torch.isfinite(calibration_loss).all():
                                    raise RuntimeError("Non-finite forward loss during AMP calibration")
                                scaler.scale(calibration_loss / len(window)).backward()

                        calibrated = calibrate_gradients(model=model, optimizer=optimizer,
                            scaler=scaler, policy=amp_policy, backward_window=backward_calibration_window,
                            report=lambda item: write_json_atomic(report_path, item))
                    amp_runtime["calibrated_states"][state] = calibrated
                    del windows, calibration_loader
                    print(f"amp_calibration_complete: state={state} scale={scaler.get_scale()} windows={len(calibrated['windows'])} optimizer_updates=0", flush=True)
                    if wandb_run is not None:
                        wandb_run.config.update({f"amp_calibration_{state}": calibrated}, allow_val_change=True)
            sums = {key: 0.0 for key in LOSS_LOG_KEYS}
            observed_loss_keys: set[str] = set()
            batches = 0
            optimizer_steps_this_epoch = 0
            total_batches = len(loader)
            if max_train_batches is not None:
                total_batches = min(total_batches, int(max_train_batches))
            optimizer.zero_grad(set_to_none=True)
            if runtime_progress is not None:
                runtime_progress.update(phase="training", epoch=epoch, completed_batches=0,
                    total_batches=total_batches, optimizer_step=total_step, force=True)
            for batch_index, (images, targets) in enumerate(loader, start=1):
                if max_train_batches is not None and batch_index > max_train_batches:
                    break
                image_size = (int(images.shape[-2]), int(images.shape[-1]))
                if feature_cache is None:
                    images = images.to(device, non_blocking=True)
                    dino_feature = None
                else:
                    dino_feature = feature_cache.load_batch(targets, device=device)
                targets = move_targets_to_device(targets, device, non_blocking=str(device).startswith("cuda"))
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    if dino_feature is None:
                        loss_dict = model(images, targets)
                    else:
                        loss_dict = model.forward_from_dino_feature(dino_feature, image_size=image_size, targets=targets)
                    assert isinstance(loss_dict, dict)
                    loss = loss_dict["loss_total"]
                if not bool(torch.isfinite(loss).all().item()):
                    printable = {
                        key: float(value.detach().cpu()) if torch.isfinite(value.detach()).all().item() else "nonfinite"
                        for key, value in loss_dict.items()
                    }
                    raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_index}: {printable}")
                window_start = ((batch_index - 1) // gradient_accumulation_steps) * gradient_accumulation_steps + 1
                window_size = min(gradient_accumulation_steps, total_batches - window_start + 1)
                scaler.scale(loss / max(1, window_size)).backward()
                should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == total_batches
                if should_step:
                    scaler.unscale_(optimizer)
                    if (isolated_probe or amp_policy) and not gradients_finite(trainable_params):
                        raise RuntimeError("Non-finite gradient in official-SFP verification (after calibration if enabled)")
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=float(train_cfg.get("max_grad_norm", 10.0)),
                        **({"error_if_nonfinite": True} if amp_policy else {}),
                    )
                    scale_before_step = scaler.get_scale() if scaler.is_enabled() else 1.0
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after_step = scaler.get_scale() if scaler.is_enabled() else 1.0
                    optimizer_stepped = (not scaler.is_enabled()) or scale_after_step >= scale_before_step
                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_stepped:
                        total_step += 1
                        optimizer_steps_this_epoch += 1
                        if ema is not None:
                            ema.update(model)
                if isolated_probe:
                    if probe_started is not None:
                        probe_images += len(targets)
                    elif total_step >= 2:
                        torch.cuda.synchronize(device)
                        probe_started = time.monotonic()
                batches += 1
                observed_loss_keys.update(str(key) for key in loss_dict)
                for key in sums:
                    value = loss_dict.get(key)
                    if value is not None:
                        sums[key] += float(value.detach().cpu())
                if runtime_progress is not None:
                    runtime_progress.update(phase="training", epoch=epoch,
                        completed_batches=batches, total_batches=total_batches,
                        optimizer_step=total_step, loss=sums["loss_total"] / batches,
                        force=batches == total_batches)

            if batches == 0:
                raise RuntimeError("No batches were processed.")
            row = {key: sums[key] / batches for key in sums}
            row["epoch"] = epoch
            row["lr"] = float(optimizer.param_groups[0]["lr"])
            if isolated_probe:
                torch.cuda.synchronize(device)
                probe_train_seconds = time.monotonic() - probe_started if probe_started else None
            row["elapsed_seconds"] = round(previous_elapsed + time.monotonic() - started, 3)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            metrics_file.flush()

            loss_parts = " ".join(
                f"{key}={row[key]:.6f}"
                for key in LOSS_LOG_KEYS
                if key in row and key != "loss_total" and (row[key] != 0.0 or key in loss_dict)
            )
            print(
                f"epoch {epoch}/{epochs}: "
                f"loss_total={row['loss_total']:.6f} "
                f"{loss_parts}"
            )

            metrics = {key: float(row[key]) for key in LOSS_LOG_KEYS if key in row}
            wandb_metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train/loss": float(row["loss_total"]),
                "train/lr": float(row["lr"]),
            }
            if "loss_cls" in observed_loss_keys:
                wandb_metrics["train/cls_loss"] = float(row["loss_cls"])
            if "loss_bbox" in observed_loss_keys:
                wandb_metrics["train/bbox_loss"] = float(row["loss_bbox"])
            elif "loss_bbox_l1" in observed_loss_keys:
                wandb_metrics["train/bbox_loss"] = float(row["loss_bbox_l1"])
            if "loss_kpt" in observed_loss_keys:
                wandb_metrics["train/keypoint_loss"] = float(row["loss_kpt"])
            if amp_runtime is not None:
                amp_runtime["scaler"] = scaler.state_dict()
                wandb_metrics["train/amp_scale"] = scaler.get_scale()
            if not official_recovery:
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=total_step,
                    config=config,
                    metrics=metrics,
                    ema=ema,
                    amp_runtime=amp_runtime,
                )
            if row["loss_total"] < best_loss:
                best_loss = row["loss_total"]
                save_checkpoint(
                    output_dir / "checkpoints" / "best_loss.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=total_step,
                    config=config,
                    metrics=metrics,
                    ema=ema,
                    amp_runtime=amp_runtime,
                )
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=total_step,
                    config=config,
                    metrics=metrics,
                    ema=ema,
                    amp_runtime=amp_runtime,
                )
                prune_periodic_checkpoints(output_dir / "checkpoints", keep=1)
            if validation_enabled and validation_every > 0 and (epoch % validation_every == 0 or epoch == epochs):
                if runtime_progress is not None:
                    runtime_progress.update(phase="validation", epoch=epoch,
                        completed_batches=batches, total_batches=total_batches,
                        optimizer_step=total_step, force=True)
                val_metrics = validate_detection_split(
                    ema.module if ema is not None else model,
                    config,
                    device=device,
                    split=validation_split,
                    image_size=validation_image_size,
                    fixed_threshold=validation_fixed_threshold,
                    score_min=validation_score_min,
                    max_images=validation_max_images,
                    batch_size=int(validation_cfg.get("batch_size", batch_size)),
                )
                val_row = {"epoch": epoch, **val_metrics}
                val_write_header = not val_metrics_path.exists() or val_metrics_path.stat().st_size == 0
                with val_metrics_path.open("a", encoding="utf-8", newline="") as val_file:
                    val_writer = csv.DictWriter(val_file, fieldnames=list(VAL_LOG_KEYS))
                    if val_write_header:
                        val_writer.writeheader()
                    val_writer.writerow({key: val_row.get(key, "") for key in VAL_LOG_KEYS})
                current_validation_score = float(val_metrics[validation_metric_key])
                if current_validation_score > best_val_map:
                    best_val_map = current_validation_score
                    save_checkpoint(
                        output_dir / "checkpoints" / best_validation_checkpoint_name,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=total_step,
                        config=config,
                        metrics={**metrics, **{f"val_{key}": float(value) for key, value in val_metrics.items() if isinstance(value, (int, float))}},
                        ema=ema,
                        amp_runtime=amp_runtime,
                    )
                if current_validation_score > early_stopping_best + early_stopping_min_delta:
                    early_stopping_best = current_validation_score
                    early_stopping_best_epoch = epoch
                    early_stopping_bad_validations = 0
                else:
                    early_stopping_bad_validations += 1
                print(
                    f"val {validation_split} epoch {epoch}/{epochs}: "
                    f"loss={val_metrics['loss']:.6f} "
                    f"mAP={val_metrics['mAP']:.4f} "
                    f"AP50={val_metrics['AP50']:.4f} "
                    f"F1@{validation_fixed_threshold:.2f}={val_metrics['f1']:.4f} "
                    f"P={val_metrics['precision']:.4f} "
                    f"R={val_metrics['recall']:.4f} "
                    f"bestF1={val_metrics['best_f1']:.4f}@{val_metrics['best_threshold']:.2f}"
                )
                if "ellipse_mAP" in val_metrics:
                    print(
                        f"val ellipse epoch {epoch}/{epochs}: "
                        f"mAP={val_metrics['ellipse_mAP']:.4f} "
                        f"AP50={val_metrics['ellipse_AP50']:.4f} "
                        f"AP75={val_metrics['ellipse_AP75']:.4f} "
                        f"meanIoU={val_metrics['mean_ellipse_iou']:.4f} "
                        f"angleMAE={val_metrics['ellipse_angle_mae_deg']:.2f}deg"
                    )
                wandb_metrics.update(
                    {
                        "val/loss": float(val_metrics["loss"]),
                        "val/map50": float(val_metrics["AP50"]),
                        "val/map50_95": float(val_metrics["mAP"]),
                        "val/precision": float(val_metrics["precision"]),
                        "val/recall": float(val_metrics["recall"]),
                        "val/f1": float(val_metrics["f1"]),
                    }
                )
                if "ellipse_mAP" in val_metrics:
                    wandb_metrics.update(
                        {
                            "val/ellipse_map50_95": float(val_metrics["ellipse_mAP"]),
                            "val/ellipse_map50": float(val_metrics["ellipse_AP50"]),
                            "val/ellipse_map75": float(val_metrics["ellipse_AP75"]),
                            "val/ellipse_mean_iou": float(val_metrics["mean_ellipse_iou"]),
                            "val/ellipse_angle_mae_deg": float(val_metrics["ellipse_angle_mae_deg"]),
                        }
                    )
                if "nme" in val_metrics:
                    wandb_metrics["val/nme"] = float(val_metrics["nme"])
                if "pck_005" in val_metrics:
                    wandb_metrics["val/pck_005"] = float(val_metrics["pck_005"])
                if early_stopping_enabled:
                    wandb_metrics["early_stopping/bad_validations"] = early_stopping_bad_validations
                    wandb_metrics["early_stopping/best_score"] = early_stopping_best
                    if (
                        epoch >= early_stopping_minimum_epoch
                        and early_stopping_bad_validations >= early_stopping_patience
                    ):
                        early_stopping_triggered = True
                        early_stop_payload = {
                            "status": "early_stopped",
                            "monitor": early_stopping_monitor,
                            "stopped_at_epoch": epoch,
                            "configured_epochs": epochs,
                            "best_epoch": early_stopping_best_epoch,
                            "best_score": early_stopping_best,
                            "min_delta": early_stopping_min_delta,
                            "patience_validations": early_stopping_patience,
                            "bad_validations": early_stopping_bad_validations,
                            "minimum_epoch": early_stopping_minimum_epoch,
                            "best_checkpoint": str(output_dir / "checkpoints" / best_validation_checkpoint_name),
                        }
                        (output_dir / "early_stop.json").write_text(
                            json.dumps(early_stop_payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        wandb_metrics["early_stopping/triggered"] = 1
                        print(
                            "early_stopping: "
                            f"epoch={epoch} best_epoch={early_stopping_best_epoch} "
                            f"best_mAP={early_stopping_best:.6f} "
                            f"bad_validations={early_stopping_bad_validations}/{early_stopping_patience}"
                        )
            if official_recovery:
                # A validation OOM, including on the final epoch, must retry
                # that epoch rather than resume past the missed validation.
                save_checkpoint(output_dir / "checkpoints" / "last.pt",
                    model=model, optimizer=optimizer, epoch=epoch, step=total_step,
                    config=config, metrics=metrics, ema=ema, amp_runtime=amp_runtime)
            if scheduler is not None and optimizer_steps_this_epoch > 0:
                scheduler.step()
            if wandb_run is not None:
                wandb_run.log(wandb_metrics, step=epoch)
            if runtime_progress is not None:
                runtime_progress.update(phase="epoch_completed", epoch=epoch,
                    completed_batches=batches, total_batches=total_batches,
                    optimizer_step=total_step, force=True)
            if early_stopping_triggered:
                break
    prune_periodic_checkpoints(output_dir / "checkpoints", keep=0)
    retention_cfg = dict(train_cfg.get("checkpoint_retention", {}) or {})
    if bool(retention_cfg.get("only_last_and_validation_best", False)):
        (output_dir / "checkpoints" / "best_loss.pt").unlink(missing_ok=True)
    print(f"training_complete: {output_dir} early_stopped={early_stopping_triggered}")
    if isolated_probe:
        torch.cuda.synchronize(device)
        (output_dir / "isolated_probe.json").write_text(json.dumps(dict(
            train_seconds_after_warmup=probe_train_seconds, measured_train_images=probe_images,
            warmup_optimizer_updates=2, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)),indent=2)+"\n",encoding="utf-8")
    return 0


def overfit_smoke(
    config: dict[str, Any],
    *,
    device: str,
    force_synthetic: bool,
    max_samples: int,
    steps: int,
    lr: float,
) -> int:
    smoke_config = copy.deepcopy(config)
    model_type = str(smoke_config.get("model", {}).get("type", "")).lower()
    ellipse_smoke = model_type in {
        "dinov3_codino_official_ellipse",
        "dinov3_deimv2_ellipse",
        "dino_deimv2_ellipse",
        "dinov3_codino_ellipse",
        "dino_codino_ellipse",
        "co_dino_ellipse",
    }
    if ellipse_smoke:
        smoke_config.setdefault("augmentation", {})["enabled"] = False
        smoke_config.setdefault("model", {}).setdefault("decoder", {})["num_denoising"] = 0
    dataset = build_dataset(smoke_config, force_synthetic=force_synthetic)
    indices = positive_subset_indices(dataset, max_samples=max_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    batch_size = min(int(smoke_config.get("train", {}).get("batch_size", 4)), len(subset))
    train_cfg = smoke_config.get("train", {})
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(**dataloader_kwargs(
        dataset=subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        train_cfg=train_cfg,
    ))
    model = build_detector_from_config(smoke_config).to(device)
    model.train()
    if hasattr(model, "set_epoch"):
        model.set_epoch(1)
    if ellipse_smoke:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("decoder.ellipse_head.")
                or name.startswith("decoder.ellipse_refinement.")
            )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    losses: list[float] = []
    total_losses: list[float] = []
    iterator = iter(loader)
    for step in range(steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device)
        targets = move_targets_to_device(targets, device, non_blocking=str(device).startswith("cuda"))
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        assert isinstance(loss_dict, dict)
        optimized_loss_name = "loss_ellipse" if ellipse_smoke else "loss_total"
        loss = loss_dict[optimized_loss_name]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        total_losses.append(float(loss_dict["loss_total"].detach().cpu()))
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            print(
                f"step {step + 1}/{steps}: "
                f"{optimized_loss_name}={losses[-1]:.6f} "
                f"loss_total={total_losses[-1]:.6f}"
            )

    window = min(max(5, len(subset)), max(1, len(losses) // 3))
    initial_avg = sum(losses[:window]) / window
    final_avg = sum(losses[-window:]) / window
    loss_drop_ratio = final_avg / max(initial_avg, 1e-12)
    overfit_ok = final_avg < initial_avg
    print("overfit_smoke: done")
    print(f"samples: {len(subset)}")
    print(f"batch_size: {batch_size}")
    print(f"steps: {steps}")
    print(f"optimized_loss: {optimized_loss_name}")
    print(f"initial_avg_loss: {initial_avg:.6f}")
    print(f"final_avg_loss: {final_avg:.6f}")
    print(f"loss_drop_ratio: {loss_drop_ratio:.6f}")
    print(f"overfit_ok: {overfit_ok}")
    return 0 if overfit_ok else 2


def resolve_device(
    config: dict[str, Any],
    requested: str | None,
    *,
    training: bool,
    allow_cpu_training: bool = False,
) -> str:
    device = requested or config.get("train", {}).get("device")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif training:
            raise RuntimeError("CUDA device is required for training. Pass --device cuda:0 on a GPU machine.")
        else:
            device = "cpu"
    device = str(device)
    if training and not device.startswith("cuda") and not allow_cpu_training:
        raise RuntimeError(f"Training must use a CUDA device, got {device!r}. Pass --device cuda:0.")
    return device


def resolve_amp(config: dict[str, Any], requested: bool | None) -> bool:
    return bool(config.get("train", {}).get("amp", True) if requested is None else requested)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.image_size is not None:
        image_size_override = parse_image_size(args.image_size)
        if image_size_override is None:
            raise ValueError("--image-size must be INT or HEIGHTxWIDTH")
        config.setdefault("data", {})["image_size"] = image_size_override
        config.setdefault("train", {})["imgsz"] = image_size_override
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        config.setdefault("train", {})["prefetch_factor"] = args.prefetch_factor
    if args.persistent_workers is not None:
        config.setdefault("train", {})["persistent_workers"] = args.persistent_workers
    if args.torch_compile is not None:
        config.setdefault("train", {}).setdefault("torch_compile", {})["enabled"] = args.torch_compile
    if args.feature_cache is not None:
        config.setdefault("train", {}).setdefault("feature_cache", {})["enabled"] = args.feature_cache
    if args.fused_optimizer is not None:
        config.setdefault("train", {})["fused_optimizer"] = args.fused_optimizer
    if args.progressive_resizing is not None:
        config.setdefault("train", {}).setdefault("progressive_resizing", {})["enabled"] = args.progressive_resizing
    if args.tiny_model:
        config = apply_tiny_model_overrides(config)
    device = resolve_device(
        config,
        args.device,
        training=bool(args.overfit_smoke or not args.dry_run),
        allow_cpu_training=bool(args.overfit_smoke and args.tiny_model),
    )
    speed_settings = configure_torch_speed(device, config, tf32_override=args.tf32)
    print(f"speed_settings: {speed_settings}")
    if args.overfit_smoke:
        return overfit_smoke(
            config,
            device=str(device),
            force_synthetic=args.synthetic,
            max_samples=args.max_samples,
            steps=args.steps,
            lr=args.lr,
        )
    if args.dry_run:
        return dry_run(config, device=str(device), force_synthetic=args.synthetic)
    output_dir = make_output_dir(config, args.output_dir)
    wandb_run = init_training_wandb(config, args, output_dir=output_dir)
    try:
        result = train(
            config,
            device=str(device),
            force_synthetic=args.synthetic,
            output_dir=output_dir,
            resume_from=args.resume_from,
            save_every=int(
                args.save_every
                if args.save_every is not None
                else config.get("train", {}).get("checkpoint_every", 10)
            ),
            use_amp=resolve_amp(config, args.amp),
            max_train_batches=args.max_train_batches,
            allow_dummy_backbone=bool(args.allow_dummy_backbone),
            wandb_run=wandb_run,
        )
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise
    if wandb_run is not None:
        wandb_run.finish(exit_code=result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
