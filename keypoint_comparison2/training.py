from __future__ import annotations

import copy
import contextlib
import csv
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .config import ComparisonConfig, data_config_for_model
from .data import (
    CropAugmentation,
    FaceCropDataset,
    JointFaceDataset,
    collate_crops,
    collate_joint,
)
from .metrics import KeypointMetrics
from .models import build_model
from src.wandb_utils import (
    stable_wandb_run_id,
    verify_wandb_access,
    write_json_atomic,
)
from src.checkpoint_retention import prune_periodic_checkpoints


JOINT_WANDB_METRIC_SCHEMA_VERSION = 2


def comparison_wandb_run_id(
    config: Mapping[str, Any],
    model_name: str,
    *,
    entity: str | None,
    project: str,
) -> str:
    return stable_wandb_run_id(
        {
            "entity": entity,
            "project": project,
            "experiment_name": config["experiment_name"],
            "model_name": model_name,
            "dataset_version": config["data"]["dataset_version"],
            "seed": int(config["train"].get("seed", 42)),
            "metric_schema_version": int(
                config["train"].get("wandb", {}).get(
                    "metric_schema_version",
                    JOINT_WANDB_METRIC_SCHEMA_VERSION,
                )
            ),
        }
    )


def _read_comparison_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in {None, ""}:
                    continue
                if key == "backbone_frozen":
                    row[key] = str(value).lower() == "true"
                elif key in {
                    "epoch",
                    "updates",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                }:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def comparison_wandb_metric_row(
    row: Mapping[str, Any],
    *,
    previous_elapsed_seconds: float = 0.0,
) -> dict[str, int | float | bool]:
    elapsed = float(row["elapsed_seconds"])
    result: dict[str, int | float | bool] = {
        "epoch": int(row["epoch"]),
        "overview/train_total_loss": float(row["loss_total"]),
        "overview/val_total_loss": float(row["validation_loss"]),
        "optimization/learning_rate_min": float(row["lr"]),
        "optimization/learning_rate_max": float(row["maximum_lr"]),
        "progress/optimizer_updates_epoch": int(row["updates"]),
        "progress/batch_size": int(row["batch_size"]),
        "progress/effective_batch_size": int(row["effective_batch_size"]),
        "progress/backbone_frozen": bool(row["backbone_frozen"]),
        "timing/epoch_minutes": max(0.0, elapsed - previous_elapsed_seconds)
        / 60.0,
        "timing/elapsed_hours": elapsed / 3600.0,
    }
    if "train_seconds" in row:
        result["timing/train_seconds"] = float(row["train_seconds"])
    if "validation_seconds" in row:
        result["timing/validation_seconds"] = float(row["validation_seconds"])
    conditional_metric = "conditional_semantic_macro_f1_nme_0.10"
    if conditional_metric in row:
        result[f"overview/{conditional_metric}"] = float(
            row[conditional_metric]
        )
    for key, value in row.items():
        if key.startswith(("loss_", "face_loss_")) and key != "loss_total":
            result[f"loss_components/train_{key.removeprefix('loss_')}"] = float(
                value
            )
    return result


def initialize_comparison_wandb(
    config: Mapping[str, Any],
    model_name: str,
    *,
    output_dir: Path,
) -> Any:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "train.wandb.enabled=true requires the wandb package"
        ) from error
    settings = dict(config["train"].get("wandb", {}))
    entity_value = settings.get("entity")
    entity = str(entity_value) if entity_value else None
    project = str(settings.get("project", "face_detection"))
    mode = str(settings.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("train.wandb.mode must be online, offline, or disabled")
    schema_version = int(
        settings.get("metric_schema_version", JOINT_WANDB_METRIC_SCHEMA_VERSION)
    )
    run_id = comparison_wandb_run_id(
        config,
        model_name,
        entity=entity,
        project=project,
    )
    state_path = output_dir / "wandb_run.json"
    previous = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    if previous.get("run_id") not in {None, run_id}:
        raise RuntimeError(
            "Stored W&B run ID does not match this comparison condition: "
            f"{previous.get('run_id')} != {run_id}"
        )
    preflight = verify_wandb_access(
        wandb,
        entity=entity,
        project=project,
        mode=mode,
    )
    write_json_atomic(output_dir / "wandb_preflight.json", preflight)
    local_dir = output_dir / "wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    live_sync_seconds = float(settings.get("live_sync_seconds", 30.0))
    if live_sync_seconds <= 0.0:
        raise ValueError("train.wandb.live_sync_seconds must be positive")
    configured_name = settings.get("name") or (
        f"{config['experiment_name']}-{model_name}"
    )
    run = wandb.init(
        entity=entity,
        project=project,
        group=settings.get("group"),
        name=f"{configured_name}-metrics-v{schema_version}",
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(local_dir),
        tags=list(settings.get("tags", [])),
        config=dict(config),
        job_type="training",
        notes=(
            "DEIMv2 Joint training: overview contains epoch train/validation "
            "loss; loss_components contains joint detector/landmark losses; "
            "test task metrics are attached by evaluation."
        ),
        settings=wandb.Settings(
            x_file_stream_transmit_interval=live_sync_seconds,
        ),
    )
    run.define_metric("epoch")
    for namespace in (
        "overview/*",
        "loss_components/*",
        "optimization/*",
        "progress/*",
        "timing/*",
    ):
        run.define_metric(namespace, step_metric="epoch", summary="last")
    run.config.update(
        {
            "wandb_metric_schema_version": schema_version,
            "wandb_metric_guide": {
                "overview": "epoch train and model-selection validation loss",
                "loss_components": "joint face, box, landmark, and state losses",
                "evaluation": "test detector, PCK/NME, visibility, and artifacts",
            },
        },
        allow_val_change=True,
    )
    state = {
        **previous,
        "run_id": run_id,
        "url": run.url,
        "entity": preflight.get("entity"),
        "project": project,
        "group": settings.get("group"),
        "name": run.name,
        "mode": mode,
        "metric_schema_version": schema_version,
        "last_logged_epoch": int(previous.get("last_logged_epoch", -1)),
        "last_elapsed_seconds": float(previous.get("last_elapsed_seconds", 0.0)),
    }
    write_json_atomic(state_path, state)
    for row in _read_comparison_metric_rows(output_dir / "metrics.csv"):
        epoch = int(row["epoch"])
        if epoch <= int(state["last_logged_epoch"]):
            continue
        run.log(
            comparison_wandb_metric_row(
                row,
                previous_elapsed_seconds=float(state["last_elapsed_seconds"]),
            ),
            step=epoch,
        )
        state["last_logged_epoch"] = epoch
        state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
        write_json_atomic(state_path, state)
    return run


def log_comparison_wandb_epoch(
    run: Any,
    *,
    output_dir: Path,
    row: Mapping[str, Any],
) -> None:
    state_path = output_dir / "wandb_run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run.log(
        comparison_wandb_metric_row(
            row,
            previous_elapsed_seconds=float(state.get("last_elapsed_seconds", 0.0)),
        ),
        step=int(row["epoch"]),
    )
    state["last_logged_epoch"] = int(row["epoch"])
    state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
    write_json_atomic(state_path, state)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BatchPlan:
    batch_size: int
    gradient_accumulation_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def as_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
        }


def configured_batch_plans(model_config: Mapping[str, Any]) -> tuple[BatchPlan, ...]:
    configured = BatchPlan(
        batch_size=int(model_config["batch_size"]),
        gradient_accumulation_steps=int(
            model_config["gradient_accumulation_steps"]
        ),
    )
    fallback = model_config.get("batch_size_fallback", {})
    if not isinstance(fallback, Mapping) or not bool(fallback.get("enabled", False)):
        return (configured,)
    candidates = fallback.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "deimv2_joint.batch_size_fallback.candidates must be a non-empty list"
        )
    plans = tuple(
        BatchPlan(
            batch_size=int(candidate["batch_size"]),
            gradient_accumulation_steps=int(
                candidate["gradient_accumulation_steps"]
            ),
        )
        for candidate in candidates
    )
    if plans[0] != configured:
        raise ValueError(
            "deimv2_joint batch settings must match the first fallback candidate"
        )
    if any(
        plan.batch_size < 1 or plan.gradient_accumulation_steps < 1
        for plan in plans
    ):
        raise ValueError("DEIMv2 Joint batch fallback values must be positive")
    if len(set(plans)) != len(plans):
        raise ValueError("DEIMv2 Joint batch fallback candidates must be unique")
    if len({plan.effective_batch_size for plan in plans}) != 1:
        raise ValueError(
            "DEIMv2 Joint batch fallback candidates must preserve effective batch"
        )
    return plans


def _is_cuda_oom(error: BaseException) -> bool:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if cuda_oom_type and isinstance(error, cuda_oom_type):
        return True
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "out of memory" in message
        and ("cuda" in message or "cublas" in message or "cudnn" in message)
    )


def _release_cuda_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if torch.cuda.is_available() and (
        device is None or device.type == "cuda"
    ):
        torch.cuda.empty_cache()


def select_batch_plan(
    plans: tuple[BatchPlan, ...],
    probe: Callable[[BatchPlan], Mapping[str, Any]],
    *,
    device: torch.device | None = None,
) -> tuple[BatchPlan, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for plan in plans:
        try:
            measurements = dict(probe(plan))
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            attempts.append(
                {
                    **plan.as_dict(),
                    "status": "cuda_oom",
                    "error": str(error),
                }
            )
            _release_cuda_memory(device)
            continue
        attempts.append(
            {
                **plan.as_dict(),
                "status": "selected",
                **measurements,
            }
        )
        _release_cuda_memory(device)
        return plan, attempts
    raise RuntimeError(
        "CUDA OOM for every configured DEIMv2 Joint batch candidate: "
        + ", ".join(
            f"batch={plan.batch_size}/accum={plan.gradient_accumulation_steps}"
            for plan in plans
        )
    )


def _apply_batch_plan(
    config: ComparisonConfig,
    model_name: str,
    plan: BatchPlan,
) -> None:
    model_config = config.raw["models"][model_name]
    model_config["batch_size"] = plan.batch_size
    model_config[
        "gradient_accumulation_steps"
    ] = plan.gradient_accumulation_steps


def _load_reusable_batch_plan(
    report_path: Path,
    plans: tuple[BatchPlan, ...],
) -> BatchPlan | None:
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report["selected"]
        plan = BatchPlan(
            batch_size=int(selected["batch_size"]),
            gradient_accumulation_steps=int(
                selected["gradient_accumulation_steps"]
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return plan if plan in plans else None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ModelEMA:
    """On-device EMA state with safe temporary application for validation."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            shadow = self.shadow[key]
            if value.is_floating_point():
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value)

    @contextlib.contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        state = model.state_dict()
        backup = {key: value.detach().clone() for key, value in state.items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value for key, value in state["shadow"].items()}


def _augmentation(config: Mapping[str, Any]) -> CropAugmentation:
    values = dict(config)
    for key in ("occlusion_area", "occlusion_count"):
        if key in values:
            values[key] = tuple(values[key])
    return CropAugmentation(**values)


def build_loaders(
    config: ComparisonConfig,
    model_name: str,
    *,
    maximum_train_samples: int | None = None,
    maximum_val_samples: int | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    raw = config.raw
    model_cfg = raw["models"][model_name]
    workers = int(raw["train"].get("num_workers", 8))
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers:
        common["prefetch_factor"] = int(raw["train"].get("prefetch_factor", 4))
    if model_name == "deimv2_joint":
        data_cfg = data_config_for_model(raw, model_name)
        train_dataset = JointFaceDataset(
            annotation=data_cfg["annotations"]["train"],
            image_root=data_cfg["image_root"],
            image_size=int(model_cfg["input_size"]),
            category_ids=data_cfg["category_ids"],
            target_category_id=int(data_cfg.get("target_category_id", data_cfg["category_ids"][0])),
            training=True,
            head_annotation=data_cfg.get("head_annotations", {}).get("train"),
        )
        val_dataset = JointFaceDataset(
            annotation=data_cfg["annotations"]["val"],
            image_root=data_cfg["image_root"],
            image_size=int(model_cfg["input_size"]),
            category_ids=data_cfg["category_ids"],
            target_category_id=int(data_cfg.get("target_category_id", data_cfg["category_ids"][0])),
            training=False,
            head_annotation=data_cfg.get("head_annotations", {}).get("val"),
        )
        collate_fn = collate_joint
    else:
        data_cfg = data_config_for_model(raw, model_name)
        augmentation = _augmentation(raw["data"]["augmentation"])
        train_dataset = FaceCropDataset(
            annotation=data_cfg["annotations"]["train"],
            image_root=data_cfg["image_root"],
            input_size=int(model_cfg["input_size"]),
            training=True,
            category_ids=data_cfg["category_ids"],
            augmentation=augmentation,
            ellipse_error_model=raw["data"]["ellipse_error_model"],
        )
        val_dataset = FaceCropDataset(
            annotation=data_cfg["annotations"]["val"],
            image_root=data_cfg["image_root"],
            input_size=int(model_cfg["input_size"]),
            training=False,
            category_ids=data_cfg["category_ids"],
            augmentation=augmentation,
        )
        collate_fn = collate_crops
    if maximum_train_samples is not None:
        if maximum_train_samples < 1:
            raise ValueError("maximum_train_samples must be positive")
        train_dataset = Subset(
            train_dataset,
            range(min(int(maximum_train_samples), len(train_dataset))),
        )
    if maximum_val_samples is not None:
        if maximum_val_samples < 1:
            raise ValueError("maximum_val_samples must be positive")
        val_dataset = Subset(
            val_dataset,
            range(min(int(maximum_val_samples), len(val_dataset))),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        **common,
    )
    validation_batch_size = (
        1 if model_name == "deimv2_joint" else min(int(raw["evaluation"]["batch_size"]), int(model_cfg["batch_size"]) * 2)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **common,
    )
    return train_loader, val_loader


def _no_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or any(token in name.lower() for token in ("norm", "embedding", "queries"))


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> AdamW:
    base_lr = float(config.get("lr", 5e-4))
    backbone_lr = float(config.get("lr_backbone", 1e-5))
    weight_decay = float(config.get("weight_decay", 0.05))
    layer_decay = float(config.get("layer_decay", 1.0))
    block_pattern = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
    block_indices = [
        int(match.group(1))
        for name, _ in model.named_parameters()
        if ".dino." in name and (match := block_pattern.search(name))
    ]
    last_block = max(block_indices, default=-1)

    def parameter_lr(name: str, is_backbone: bool) -> float:
        if not is_backbone:
            return base_lr
        if layer_decay >= 1.0 or ".dino." not in name or last_block < 0:
            return backbone_lr
        match = block_pattern.search(name)
        if match:
            layer_id = int(match.group(1)) + 1
        elif ".patch_embed." in name or any(
            token in name for token in ("cls_token", "storage_tokens", "mask_token")
        ):
            layer_id = 0
        else:
            layer_id = last_block + 2
        maximum_layer = last_block + 2
        return backbone_lr * (layer_decay ** (maximum_layer - layer_id))

    grouped: dict[tuple[float, float], list[nn.Parameter]] = defaultdict(list)
    covered: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = (
            name.startswith("backbone.")
            or name.startswith("face_detector.backbone.")
            or ".dino." in name
        )
        lr = parameter_lr(name, is_backbone)
        decay = 0.0 if _no_decay(name, parameter) else weight_decay
        grouped[(lr, decay)].append(parameter)
        if id(parameter) in covered:
            raise RuntimeError(f"Optimizer parameter is duplicated: {name}")
        covered.add(id(parameter))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if covered != trainable:
        raise RuntimeError("Optimizer does not cover every trainable parameter exactly once")
    groups = [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in grouped.items()
    ]
    fused = bool(config.get("fused_optimizer", True)) and torch.cuda.is_available()
    return AdamW(groups, betas=(0.9, 0.999), fused=fused)


def _set_dinov3_trainable(model: nn.Module, trainable: bool) -> bool:
    encoder = getattr(model, "backbone", None)
    if encoder is None:
        face_detector = getattr(model, "face_detector", None)
        encoder = getattr(face_detector, "backbone", None)
    dino = getattr(encoder, "dino", None)
    if not isinstance(dino, nn.Module):
        return False
    for parameter in dino.parameters():
        parameter.requires_grad_(trainable)
    if not trainable:
        dino.eval()
    return True


def _selection_metric(config: ComparisonConfig, model_name: str) -> str:
    selection = config.raw["train"].get("checkpoint_selection", {})
    key = "joint_metric" if model_name == "deimv2_joint" else "topdown_metric"
    return str(selection.get(key, "validation_loss" if model_name == "deimv2_joint" else "nme"))


def _selection_direction(config: ComparisonConfig, model_name: str) -> str:
    selection = config.raw["train"].get("checkpoint_selection", {})
    key = "joint_direction" if model_name == "deimv2_joint" else "topdown_direction"
    default = "minimize"
    direction = str(selection.get(key, default))
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Invalid checkpoint selection direction: {direction}")
    return direction


def best_checkpoint_path(config: ComparisonConfig, model_name: str) -> Path:
    return config.output_root / model_name / f"checkpoints/best_{_selection_metric(config, model_name)}.pt"


def _scheduler(optimizer: AdamW, *, epochs: int, warmup_epochs: int, minimum_ratio: float) -> LambdaLR:
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(1e-5, (epoch + 1) / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, schedule)


def _autocast(device: torch.device, train_cfg: Mapping[str, Any]):
    dtype = torch.bfloat16 if str(train_cfg.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")


def _forward_batch(
    model: nn.Module,
    model_name: str,
    batch: Any,
    device: torch.device,
) -> dict[str, Any]:
    if model_name == "deimv2_joint":
        images, targets = batch
        images = images.to(device, non_blocking=True)
        moved = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        return model(images, moved)
    images = batch["image"].to(device, non_blocking=True)
    targets = {
        "coordinates": batch["coordinates"].to(device, non_blocking=True),
        "labeled": batch["labeled"].to(device, non_blocking=True),
        "visible": batch["visible"].to(device, non_blocking=True),
    }
    return model(
        images,
        targets,
        clean_images=batch["clean_image"].to(device, non_blocking=True),
        occlusion_masks=batch["occlusion_mask"].to(device, non_blocking=True),
    )


@contextlib.contextmanager
def _joint_validation_loss_mode(model: nn.Module):
    """Enable DEIMv2 loss outputs while keeping validation deterministic.

    The official DEIMv2 decoder emits ``aux_outputs`` only when its own
    ``training`` flag is true, and its criterion requires those outputs even
    when validation loss is computed under ``torch.inference_mode``.  Calling
    ``model.train()`` here would also enable dropout and normalization updates
    throughout the backbone.  Toggle only the wrappers whose flags select the
    loss-output branch; all child modules remain in evaluation mode.
    """

    face_detector = getattr(model, "face_detector", None)
    decoder = getattr(face_detector, "decoder", None)
    modules = [
        module
        for module in (model, face_detector, decoder)
        if isinstance(module, nn.Module)
    ]
    previous = [bool(module.training) for module in modules]
    for module in modules:
        module.training = True
    try:
        yield
    finally:
        for module, was_training in zip(modules, previous):
            module.training = was_training


def _probe_deimv2_joint_batch_plan(
    config: ComparisonConfig,
    plan: BatchPlan,
    *,
    device: torch.device,
) -> dict[str, Any]:
    probe_config = ComparisonConfig(
        path=config.path,
        project_root=config.project_root,
        raw=copy.deepcopy(config.raw),
    )
    _apply_batch_plan(probe_config, "deimv2_joint", plan)
    probe_config.raw["train"]["num_workers"] = 0
    seed_everything(int(probe_config.raw["train"].get("seed", 42)))

    model: nn.Module | None = None
    optimizer: AdamW | None = None
    scaler: torch.amp.GradScaler | None = None
    ema: ModelEMA | None = None
    train_loader: DataLoader[Any] | None = None
    iterator: Iterator[Any] | None = None
    try:
        model = build_model("deimv2_joint", probe_config.raw).to(device)
        if hasattr(model, "face_detector") and hasattr(model.face_detector, "set_epoch"):
            model.face_detector.set_epoch(
                int(probe_config.raw["train"].get("freeze_backbone_epochs", 0))
            )
        # Probe the higher-memory, unfrozen phase so a plan selected during the
        # frozen warmup cannot fail when DINO fine-tuning starts.
        _set_dinov3_trainable(model, True)
        optimizer = build_optimizer(model, probe_config.raw["train"])
        scaler = torch.amp.GradScaler(
            device.type,
            init_scale=float(
                probe_config.raw["train"].get(
                    "grad_scaler_init_scale",
                    65536.0,
                )
            ),
            enabled=(
                bool(probe_config.raw["train"].get("amp", True))
                and device.type == "cuda"
                and str(probe_config.raw["train"].get("amp_dtype")) == "float16"
            ),
        )
        ema = ModelEMA(
            model,
            float(probe_config.raw["train"]["ema"].get("decay", 0.9998)),
        )
        train_loader, _ = build_loaders(probe_config, "deimv2_joint")
        iterator = iter(train_loader)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(plan.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            with _autocast(device, probe_config.raw["train"]):
                output = _forward_batch(
                    model,
                    "deimv2_joint",
                    batch,
                    device,
                )
                loss = output["loss_total"] / plan.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss during DEIMv2 Joint batch probe"
                )
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(probe_config.raw["train"].get("max_grad_norm", 1.0)),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "Non-finite gradient norm during DEIMv2 Joint batch probe"
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        if device.type != "cuda":
            return {"device": str(device)}
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        mib = 1024**2
        return {
            "device": str(device),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / mib,
                2,
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / mib,
                2,
            ),
            "device_total_mib": round(total_bytes / mib, 2),
            "device_free_after_update_mib": round(free_bytes / mib, 2),
        }
    finally:
        del iterator, train_loader, ema, scaler, optimizer, model
        _release_cuda_memory(device)


@torch.inference_mode()
def validate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    train_cfg: Mapping[str, Any],
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    metrics = KeypointMetrics()
    selection_cfg = dict(train_cfg.get("checkpoint_selection", {}) or {})
    conditional_metric = "conditional_semantic_macro_f1_nme_0.10"
    measure_conditional = (
        model_name == "deimv2_joint"
        and str(selection_cfg.get("joint_metric")) == conditional_metric
    )
    semantic_tp = torch.zeros(5, dtype=torch.int64)
    semantic_fp = torch.zeros(5, dtype=torch.int64)
    semantic_fn = torch.zeros(5, dtype=torch.int64)
    eligible_faces = 0
    for batch in loader:
        loss_mode = (
            _joint_validation_loss_mode(model)
            if model_name == "deimv2_joint"
            else contextlib.nullcontext()
        )
        with loss_mode, _autocast(device, train_cfg):
            output = _forward_batch(model, model_name, batch, device)
        total_loss += float(output["loss_total"])
        batches += 1
        if measure_conditional:
            presence_logits = output.get("presence_logits")
            matches = output.get("match_indices")
            if presence_logits is None or matches is None:
                raise RuntimeError(
                    "Conditional Joint validation requires factorized presence "
                    "logits and Face-to-GT assignments"
                )
            candidates = output["candidate_indices"]
            face_scores = output["pred_logits"].sigmoid().amax(-1)
            predicted_keypoints = output["pred_keypoints"]
            predicted_presence = presence_logits.sigmoid()
            images, targets = batch
            input_side = float(max(images.shape[-2:]))
            face_threshold = float(
                selection_cfg.get("face_score_threshold", 0.33)
            )
            point_threshold = float(
                selection_cfg.get("point_presence_threshold", 0.70)
            )
            nme_threshold = float(selection_cfg.get("nme_threshold", 0.10))
            normalization_scale = float(
                selection_cfg.get("head_normalization_scale", 1.20)
            )
            for batch_index, ((prediction_indices, target_indices), target) in enumerate(
                zip(matches, targets)
            ):
                for prediction_index, target_index in zip(
                    prediction_indices.tolist(), target_indices.tolist()
                ):
                    positions = torch.nonzero(
                        candidates[batch_index] == int(prediction_index),
                        as_tuple=False,
                    )
                    if positions.numel() == 0:
                        continue
                    position = int(positions[0, 0])
                    if float(face_scores[batch_index, position]) < face_threshold:
                        continue
                    labeled = target["labeled"][int(target_index)].bool()
                    if not bool(labeled.any()):
                        continue
                    head_box = target["head_boxes"][int(target_index)].float()
                    if not bool(torch.isfinite(head_box).all()):
                        continue
                    head_side = normalization_scale * float(
                        torch.maximum(
                            head_box[2] - head_box[0],
                            head_box[3] - head_box[1],
                        )
                    )
                    if head_side <= 0.0:
                        continue
                    eligible_faces += 1
                    target_keypoints = target["keypoints"][int(target_index)].to(
                        predicted_keypoints.device
                    )
                    error = torch.linalg.vector_norm(
                        predicted_keypoints[batch_index, position]
                        - target_keypoints,
                        dim=-1,
                    ) / (head_side / input_side)
                    predicted = (
                        predicted_presence[batch_index, position]
                        >= point_threshold
                    ).cpu()
                    correct = predicted & labeled & (error.cpu() <= nme_threshold)
                    semantic_tp += correct.to(torch.int64)
                    semantic_fp += (predicted & ~correct).to(torch.int64)
                    semantic_fn += (labeled & ~correct).to(torch.int64)
        if model_name != "deimv2_joint":
            metrics.update(
                output["coordinates"],
                batch["coordinates"],
                batch["labeled"],
                visibility_probability=output["visibility_logits"].sigmoid(),
                visibility_target=batch["visible"],
            )
    result = {"validation_loss": total_loss / max(1, batches)}
    if measure_conditional:
        denominator = 2 * semantic_tp + semantic_fp + semantic_fn
        semantic_f1 = torch.where(
            denominator > 0,
            2.0 * semantic_tp.float() / denominator.float(),
            torch.zeros_like(denominator, dtype=torch.float32),
        )
        result.update(
            {
                conditional_metric: float(semantic_f1.mean()),
                "conditional_eligible_faces": float(eligible_faces),
                "conditional_semantic_tp": float(semantic_tp.sum()),
                "conditional_semantic_fp": float(semantic_fp.sum()),
                "conditional_semantic_fn": float(semantic_fn.sum()),
            }
        )
    if model_name != "deimv2_joint":
        result.update(metrics.compute())
    return result


def _write_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: float,
    selection_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "selection_metric": selection_metric,
        },
        temporary,
    )
    temporary.replace(path)


def train_model(
    config: ComparisonConfig,
    model_name: str,
    *,
    device: torch.device,
    resume: bool = True,
    maximum_train_samples: int | None = None,
    maximum_val_samples: int | None = None,
) -> Path:
    train_cfg = config.raw["train"]
    seed_everything(int(train_cfg.get("seed", 42)))
    output_dir = config.output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "checkpoints/last.pt"

    if model_name == "deimv2_joint":
        model_cfg = config.raw["models"][model_name]
        plans = configured_batch_plans(model_cfg)
        fallback_cfg = model_cfg.get("batch_size_fallback", {})
        fallback_enabled = (
            isinstance(fallback_cfg, Mapping)
            and bool(fallback_cfg.get("enabled", False))
        )
        if fallback_enabled:
            report_path = output_dir / "batch_size_selection.json"
            selected = (
                _load_reusable_batch_plan(report_path, plans)
                if resume and last_checkpoint.is_file()
                else None
            )
            if selected is not None:
                attempts: list[dict[str, Any]] = []
                source = "reused_for_resume"
            elif device.type == "cuda":
                selected, attempts = select_batch_plan(
                    plans,
                    lambda plan: _probe_deimv2_joint_batch_plan(
                        config,
                        plan,
                        device=device,
                    ),
                    device=device,
                )
                source = "measured"
            else:
                selected = plans[0]
                attempts = [
                    {
                        **selected.as_dict(),
                        "status": "selected_without_cuda_probe",
                        "device": str(device),
                    }
                ]
                source = "non_cuda"
            _apply_batch_plan(config, model_name, selected)
            if source != "reused_for_resume":
                _atomic_json(
                    report_path,
                    {
                        "status": "selected",
                        "selection_source": source,
                        "criterion": (
                            "one complete unfrozen optimizer update including "
                            "forward, backward, AdamW state creation, gradient "
                            "clipping, optimizer step, and EMA update"
                        ),
                        "candidates": [plan.as_dict() for plan in plans],
                        "attempts": attempts,
                        "selected": selected.as_dict(),
                    },
                )
            print(
                "DEIMv2 Joint batch plan: "
                f"batch={selected.batch_size}, "
                "gradient_accumulation_steps="
                f"{selected.gradient_accumulation_steps}, "
                f"source={source}"
            )

    # The sizing probe constructs a throwaway model and consumes RNG. Reset the
    # seed so batch selection cannot alter initialization or data order.
    seed_everything(int(train_cfg.get("seed", 42)))
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    wandb_run = None
    wandb_cfg = dict(train_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        wandb_run = initialize_comparison_wandb(
            config.raw,
            model_name,
            output_dir=output_dir,
        )
    model = build_model(model_name, config.raw).to(device)
    if hasattr(model, "face_detector") and hasattr(model.face_detector, "set_epoch"):
        model.face_detector.set_epoch(0)
    optimizer = build_optimizer(model, train_cfg)
    epochs = int(train_cfg["epochs"])
    scheduler = _scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
        minimum_ratio=float(train_cfg.get("minimum_lr_ratio", 0.01)),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        init_scale=float(train_cfg.get("grad_scaler_init_scale", 65536.0)),
        enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda" and str(train_cfg.get("amp_dtype")) == "float16",
    )
    ema = ModelEMA(model, float(train_cfg["ema"].get("decay", 0.9998)))
    train_loader, val_loader = build_loaders(
        config,
        model_name,
        maximum_train_samples=maximum_train_samples,
        maximum_val_samples=maximum_val_samples,
    )
    model_cfg = config.raw["models"][model_name]
    accumulation = int(model_cfg["gradient_accumulation_steps"])
    selection_metric = _selection_metric(config, model_name)
    selection_direction = _selection_direction(config, model_name)
    best_checkpoint = best_checkpoint_path(config, model_name)
    start_epoch = 0
    best = float("-inf") if selection_direction == "maximize" else float("inf")
    if resume and last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best = float(
            state.get(
                "best_metric",
                state.get(
                    "best_validation_loss",
                    float("-inf")
                    if selection_direction == "maximize"
                    else float("inf"),
                ),
            )
        )
    patience = 0
    completed_epoch = start_epoch - 1
    stopped_early = False
    early_cfg = train_cfg.get("early_stopping", {})
    previous_rows = _read_comparison_metric_rows(output_dir / "metrics.csv")
    elapsed_before_start = (
        float(previous_rows[-1].get("elapsed_seconds", 0.0))
        if previous_rows
        else 0.0
    )
    training_started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        completed_epoch = epoch
        epoch_train_started = time.monotonic()
        if hasattr(model, "face_detector") and hasattr(model.face_detector, "set_epoch"):
            model.face_detector.set_epoch(epoch)
        model.train()
        backbone_frozen = (
            epoch < int(train_cfg.get("freeze_backbone_epochs", 0))
            and _set_dinov3_trainable(model, False)
        )
        if not backbone_frozen:
            _set_dinov3_trainable(model, True)
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = defaultdict(float)
        updates = 0
        for step, batch in enumerate(train_loader):
            with _autocast(device, train_cfg):
                output = _forward_batch(model, model_name, batch, device)
                loss = output["loss_total"] / accumulation
            scaler.scale(loss).backward()
            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                updates += 1
            for key, value in output.items():
                if key.startswith("loss") or key.startswith("face_loss"):
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        running[key] += float(value.detach())
        scheduler.step()
        train_seconds = time.monotonic() - epoch_train_started
        validation_started = time.monotonic()
        with ema.apply(model):
            validation = validate(model, model_name, val_loader, device=device, train_cfg=train_cfg)
        validation_seconds = time.monotonic() - validation_started
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "updates": updates,
            "batch_size": int(model_cfg["batch_size"]),
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": int(model_cfg["batch_size"]) * accumulation,
            "lr": optimizer.param_groups[0]["lr"],
            "maximum_lr": max(group["lr"] for group in optimizer.param_groups),
            "backbone_frozen": backbone_frozen,
            "elapsed_seconds": elapsed_before_start
            + (time.monotonic() - training_started),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            **{key: value / max(1, len(train_loader)) for key, value in running.items()},
            **validation,
        }
        _write_metrics(output_dir / "metrics.csv", row)
        if wandb_run is not None:
            log_comparison_wandb_epoch(
                wandb_run,
                output_dir=output_dir,
                row=row,
            )
        if selection_metric not in validation:
            raise RuntimeError(
                f"Checkpoint selection metric {selection_metric!r} is absent from validation: "
                f"{sorted(validation)}"
            )
        selected_value = float(validation[selection_metric])
        if not math.isfinite(selected_value):
            raise RuntimeError(f"Checkpoint selection metric is not finite: {selection_metric}={selected_value}")
        min_delta = float(early_cfg.get("min_delta", 0.0001))
        improved = (
            selected_value > best + min_delta
            if selection_direction == "maximize"
            else selected_value < best - min_delta
        )
        if improved:
            best = selected_value
            patience = 0
        elif epoch + 1 >= int(early_cfg.get("minimum_epoch", 100)):
            patience += 1
        else:
            patience = 0
        _save_checkpoint(
            last_checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best=best,
            selection_metric=selection_metric,
        )
        if improved:
            _save_checkpoint(
                best_checkpoint,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best=best,
                selection_metric=selection_metric,
            )
        if (epoch + 1) % int(train_cfg.get("checkpoint_every", 5)) == 0:
            _save_checkpoint(
                output_dir / f"checkpoints/epoch_{epoch + 1:04d}.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best=best,
                selection_metric=selection_metric,
            )
            prune_periodic_checkpoints(output_dir / "checkpoints", keep=1)
        if (
            bool(early_cfg.get("enabled", True))
            and epoch + 1 >= int(early_cfg.get("minimum_epoch", 100))
            and patience >= int(early_cfg.get("patience", 25))
        ):
            stopped_early = True
            break
    if wandb_run is not None:
        wandb_run.finish()
    if not best_checkpoint.is_file() or not last_checkpoint.is_file():
        raise RuntimeError(
            f"{model_name} finished without required checkpoints: "
            f"best={best_checkpoint}, last={last_checkpoint}"
        )
    prune_periodic_checkpoints(output_dir / "checkpoints", keep=0)
    completion_path = output_dir / "training_complete.json"
    temporary = completion_path.with_suffix(completion_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": model_name,
                "completed_epoch": completed_epoch,
                "configured_epochs": epochs,
                "batch_size": int(model_cfg["batch_size"]),
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": int(model_cfg["batch_size"]) * accumulation,
                "stopped_early": stopped_early,
                "selection_metric": selection_metric,
                "best_metric": best,
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(completion_path)
    return best_checkpoint


def load_trained_model(
    config: ComparisonConfig,
    model_name: str,
    *,
    device: torch.device,
    checkpoint: Path | None = None,
) -> nn.Module:
    model = build_model(model_name, config.raw).to(device)
    path = checkpoint or best_checkpoint_path(config, model_name)
    state = torch.load(path, map_location=device, weights_only=False)
    ema_state = state.get("ema", {}).get("shadow")
    model.load_state_dict(ema_state or state["model"], strict=True)
    return model.eval()
