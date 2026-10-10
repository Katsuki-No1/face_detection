from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checkpoint_retention import prune_periodic_checkpoints

from .config import (
    KEYPOINT_TARGET_POLICY,
    LandmarkExperimentConfig,
    configured_keypoint_target_policy,
    configured_state_head_mode,
)
from .dataset import CropAugmentation, TopDownFaceLandmarkDataset, collate_landmark_batch
from .face_overlap import face_context_kwargs
from .evaluation import evaluate_gt_boxes
from .losses import LandmarkLoss, MessengerRecoveryLoss
from .model import DinoFaceLandmarkModel, build_landmark_model
from .continuous_recovery import ContinuousRecoveryBlock, ContinuousRecoveryLoss
from .continuous_teacher import prepare_continuous_teacher, continuous_training_losses


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def reset_fixed_augmentation_rng(
    *,
    seed: int,
    augmentation_epoch: int,
    loader_generator: torch.Generator,
) -> None:
    """Replay one deterministic sample order and augmentation realization."""
    replay_seed = int(seed) + int(augmentation_epoch)
    random.seed(replay_seed)
    np.random.seed(replay_seed)
    loader_generator.manual_seed(replay_seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelEMA:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        effective_decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            source_value = source[key].detach()
            if value.is_floating_point():
                value.mul_(effective_decay).add_(source_value, alpha=1.0 - effective_decay)
            else:
                value.copy_(source_value)


@dataclass(frozen=True)
class OptimizerStepResult:
    optimizer_step_succeeded: bool
    gradient_norm: float
    scale_before_step: float
    scale_after_step: float


def step_optimizer_with_amp_recovery(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    max_grad_norm: float,
) -> OptimizerStepResult:
    """Step once while allowing GradScaler to recover from an AMP overflow."""

    scaler.unscale_(optimizer)
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(max_grad_norm),
    )
    gradient_norm = float(gradient_norm_tensor.detach())
    gradient_is_finite = math.isfinite(gradient_norm)
    scale_before_step = float(scaler.get_scale())
    if not gradient_is_finite and not scaler.is_enabled():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "Non-finite gradient norm without an enabled GradScaler"
        )

    # unscale_ records found_inf. GradScaler.step skips a contaminated update,
    # and update backs the scale off so the next batch can recover.
    scaler.step(optimizer)
    scaler.update()
    scale_after_step = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    return OptimizerStepResult(
        optimizer_step_succeeded=gradient_is_finite
        and (not scaler.is_enabled() or scale_after_step >= scale_before_step),
        gradient_norm=gradient_norm,
        scale_before_step=scale_before_step,
        scale_after_step=scale_after_step,
    )


def restore_grad_scaler_state(
    scaler: torch.amp.GradScaler,
    checkpoint: Mapping[str, Any],
) -> bool:
    state = checkpoint.get("grad_scaler")
    if not isinstance(state, Mapping):
        return False
    scaler.load_state_dict(dict(state))
    return True


def _augmentation(config: Mapping[str, Any]) -> CropAugmentation:
    value = dict(config.get("augmentation", {}))
    return CropAugmentation(
        roi_geometry=str(value.get("roi_geometry", "long_axis_square")),
        scale_factor=float(value.get("scale_factor", 1.25)),
        scale_jitter=float(value.get("scale_jitter", 0.20)),
        center_jitter=float(value.get("center_jitter", 0.08)),
        rotation_deg=float(value.get("rotation_deg", 25.0)),
        horizontal_flip_probability=float(value.get("horizontal_flip_probability", 0.50)),
        brightness=float(value.get("brightness", 0.20)),
        contrast=float(value.get("contrast", 0.20)),
        saturation=float(value.get("saturation", 0.15)),
        blur_probability=float(value.get("blur_probability", 0.10)),
        blur_kernel_max=int(value.get("blur_kernel_max", 5)),
        occlusion_probability=float(value.get("occlusion_probability", 0.10)),
        occlusion_area=tuple(float(item) for item in value.get("occlusion_area", (0.02, 0.12))),
        landmark_guided_occlusion=(
            dict(value["landmark_guided_occlusion"])
            if isinstance(value.get("landmark_guided_occlusion"), Mapping)
            else None
        ),
        paired_occlusion=(
            dict(value["paired_occlusion"])
            if isinstance(value.get("paired_occlusion"), Mapping)
            else None
        ),
    )


def make_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    training: bool,
    evaluation_occlusion_kind: str | None = None,
) -> TopDownFaceLandmarkDataset:
    data = config["data"]
    split_roots = data.get("image_roots", {})
    for_cfg = config.get("model", {}).get("face_overlap_robustness", {}) or {}
    for_enabled = bool(for_cfg.get("enabled", False))
    if for_enabled and training and split != "train":
        raise ValueError("FOR training supervision must come from train only")
    detector_identity = None
    if for_enabled:
        detector_identity = {
            "detector_config_sha256": sha256_file(Path(config["detector"]["config"])),
            "detector_checkpoint_sha256": sha256_file(Path(config["detector"]["checkpoint"])),
            "score_threshold": float(config["detector"]["score_threshold"]),
            "maximum_detections": int(config["detector"]["maximum_detections"]),
        }
    return TopDownFaceLandmarkDataset(
        annotation_path=data["annotations"][split],
        image_root=split_roots.get(split, data["root"]),
        input_size=int(data["input_size"]),
        training=training,
        augmentation=_augmentation(data),
        mean=data.get("mean", (0.0, 0.0, 0.0)),
        std=data.get("std", (1.0, 1.0, 1.0)),
        category_ids=data.get("category_ids", (1,)),
        require_visible_keypoint=not bool(
            data.get("include_missing_keypoint_instances", False)
        ),
        ellipse_error_model=(
            data.get("ellipse_error_model") if training else None
        ),
        detector_crop_cache=(
            data.get("face_context_caches", {}).get(split)
            if for_enabled
            else (data.get("validation_detector_crop_cache") if split == "val" and not training else None)
        ),
        face_overlap_query_dim=int(for_cfg.get("query_dim", 256)) if for_enabled else None,
        face_overlap_detector_identity=detector_identity,
        detector_matching_iou=float(
            data.get("validation_detector_matching_iou", 0.50)
        ),
        seed=int(config.get("train", {}).get("seed", 42)),
        evaluation_occlusion_kind=evaluation_occlusion_kind,
        structure_head_annotations=(
            data.get("structure_head_annotations") if training and
            config.get("model", {}).get("continuous_recovery", {}).get("structure_enabled", False)
            else None
        ),
    )


def _optimizer(model: DinoFaceLandmarkModel, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    backbone_lr = float(train_cfg.get("lr_backbone", 1.0e-5))
    cnn_lr = float(train_cfg.get("lr_cnn", 1.0e-4))
    head_lr = float(train_cfg.get("lr_head", 5.0e-4))
    messenger_lr = float(train_cfg.get("lr_messenger", 1.0e-4))

    def split(module: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            is_token_or_position = any(
                token in name
                for token in ("pos_embed", "position", "cls_token", "register_tokens", "mask_token", "landmark_queries")
            )
            if parameter.ndim <= 1 or name.endswith(".bias") or is_token_or_position:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return decay, no_decay

    backbone_decay, backbone_no_decay = split(model.backbone)
    head_decay, head_no_decay = split(model.head)
    cnn_decay: list[torch.nn.Parameter] = []
    cnn_no_decay: list[torch.nn.Parameter] = []
    if model.rgb_refinement is not None:
        cnn_decay, cnn_no_decay = split(model.rgb_refinement)
    messenger_decay: list[torch.nn.Parameter] = []
    messenger_no_decay: list[torch.nn.Parameter] = []
    if model.messenger_recovery is not None:
        messenger_decay, messenger_no_decay = split(model.messenger_recovery)
    groups = [
        {
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone_decay",
        },
        {
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "name": "backbone_no_decay",
        },
        {
            "params": cnn_decay,
            "lr": cnn_lr,
            "weight_decay": weight_decay,
            "name": "cnn_decay",
        },
        {
            "params": cnn_no_decay,
            "lr": cnn_lr,
            "weight_decay": 0.0,
            "name": "cnn_no_decay",
        },
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head_no_decay"},
        {"params": messenger_decay, "lr": messenger_lr, "weight_decay": weight_decay, "name": "messenger_decay"},
        {"params": messenger_no_decay, "lr": messenger_lr, "weight_decay": 0.0, "name": "messenger_no_decay"},
    ]
    groups = [group for group in groups if group["params"]]
    kwargs: dict[str, Any] = {
        "lr": head_lr,
        "betas": tuple(float(value) for value in train_cfg.get("betas", (0.9, 0.999))),
        "weight_decay": weight_decay,
    }
    if bool(train_cfg.get("fused_optimizer", True)) and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def checkpoint_state(
    model: DinoFaceLandmarkModel,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    epoch: int,
    step: int,
    variant: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    backbone_initialization: Mapping[str, Any] | None = None,
    finetune_initialization: Mapping[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "variant": variant,
        "config": dict(config),
        "seed": int(config.get("train", {}).get("seed", 0)),
        "keypoint_target_policy": configured_keypoint_target_policy(
            dict(config)
        ),
        "state_head_mode": configured_state_head_mode(dict(config)),
        "backbone_initialization": dict(
            backbone_initialization or {}
        ),
        "finetune_initialization": dict(finetune_initialization or {}),
        "metrics": dict(metrics),
    }


def initialize_from_finetune_checkpoint(
    model: DinoFaceLandmarkModel,
    checkpoint_config: Mapping[str, Any],
    *,
    variant: str,
    expected_target_policy: Mapping[str, str],
    expected_state_head_mode: str,
) -> dict[str, Any]:
    path = Path(str(checkpoint_config["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint was not found: {path}")
    expected_sha256 = str(checkpoint_config.get("sha256", "")).strip()
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Fine-tune checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Fine-tune checkpoint is not a mapping: {path}")
    source_variant = str(checkpoint.get("variant", ""))
    required_variant = str(checkpoint_config.get("variant", variant))
    if source_variant != required_variant or variant != required_variant:
        raise RuntimeError(
            "Fine-tune checkpoint variant mismatch: "
            f"source={source_variant!r}, required={required_variant!r}, target={variant!r}"
        )
    if dict(checkpoint.get("keypoint_target_policy", {})) != dict(expected_target_policy):
        raise RuntimeError("Fine-tune checkpoint uses a different keypoint-target policy")
    if str(checkpoint.get("state_head_mode", "legacy_binary")) != expected_state_head_mode:
        raise RuntimeError("Fine-tune checkpoint uses a different state-head mode")
    prefer_ema = bool(checkpoint_config.get("prefer_ema", True))
    state = checkpoint.get("ema") if prefer_ema else None
    state = state or checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise TypeError(f"Fine-tune checkpoint has no usable model state: {path}")
    strict = bool(checkpoint_config.get("strict", True))

    def prefixes(name: str) -> tuple[str, ...]:
        raw = checkpoint_config.get(name, ())
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise TypeError(f"finetune_checkpoint.{name} must be a string list")
        values = tuple(str(value) for value in raw)
        if any(not value for value in values):
            raise ValueError(
                f"finetune_checkpoint.{name} cannot contain an empty prefix"
            )
        return values

    ignore_prefixes = prefixes("ignore_prefixes")
    allowed_missing_prefixes = prefixes("allowed_missing_prefixes")
    allowed_unexpected_prefixes = prefixes("allowed_unexpected_prefixes")
    if strict and (
        ignore_prefixes
        or allowed_missing_prefixes
        or allowed_unexpected_prefixes
    ):
        raise ValueError(
            "finetune_checkpoint strict=true cannot be combined with "
            "ignore/allowed key prefixes"
        )
    filtered_state = {
        str(key): value
        for key, value in state.items()
        if not any(str(key).startswith(prefix) for prefix in ignore_prefixes)
    }
    ignored_source_keys = sorted(set(state).difference(filtered_state))
    missing, unexpected = model.load_state_dict(filtered_state, strict=strict)
    if allowed_missing_prefixes:
        disallowed_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if disallowed_missing:
            raise RuntimeError(
                "Fine-tune checkpoint has disallowed missing tensors: "
                f"{disallowed_missing[:20]}"
            )
    if allowed_unexpected_prefixes:
        disallowed_unexpected = [
            key
            for key in unexpected
            if not any(
                key.startswith(prefix)
                for prefix in allowed_unexpected_prefixes
            )
        ]
        if disallowed_unexpected:
            raise RuntimeError(
                "Fine-tune checkpoint has disallowed unexpected tensors: "
                f"{disallowed_unexpected[:20]}"
            )
    elif allowed_missing_prefixes and unexpected:
        raise RuntimeError(
            "Fine-tune checkpoint has unexpected tensors: "
            f"{list(unexpected)[:20]}"
        )
    return {
        "path": path.as_posix(),
        "sha256": actual_sha256,
        "source_epoch": checkpoint.get("epoch"),
        "source_step": checkpoint.get("step"),
        "source_variant": source_variant,
        "prefer_ema": prefer_ema,
        "fresh_optimizer_scheduler_scaler": True,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "ignore_prefixes": list(ignore_prefixes),
        "ignored_source_keys": ignored_source_keys,
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "allowed_unexpected_prefixes": list(allowed_unexpected_prefixes),
    }


def load_landmark_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    prefer_ema: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Landmark checkpoint is not a mapping: {path}")
    state = checkpoint.get("ema") if prefer_ema else None
    state = state or checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError(f"Landmark checkpoint has no model state: {path}")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "keypoint_target_policy": checkpoint.get("keypoint_target_policy"),
        "state_head_mode": checkpoint.get(
            "state_head_mode",
            "legacy_binary",
        ),
        "metrics": checkpoint.get("metrics", {}),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def initialize_refinement_from_query(
    model: DinoFaceLandmarkModel,
    query_checkpoint: Path,
    *,
    expected_target_policy: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(query_checkpoint, map_location="cpu", weights_only=False)
    expected_policy = dict(expected_target_policy or KEYPOINT_TARGET_POLICY)
    if checkpoint.get("keypoint_target_policy") != expected_policy:
        raise RuntimeError(
            "Query refinement cannot initialize from a checkpoint trained "
            "with a different or legacy collapsed visibility target"
        )
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError(f"Query checkpoint has no state dict: {query_checkpoint}")
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("head."):
            remapped["head.query_head." + key[len("head.") :]] = value
        else:
            remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    allowed_missing = [key for key in missing if key.startswith("head.refinement_stages.")]
    disallowed = sorted(set(missing) - set(allowed_missing))
    if disallowed or unexpected:
        raise RuntimeError(f"Query-to-refinement initialization mismatch: missing={disallowed}, unexpected={unexpected}")
    return {
        "query_checkpoint": str(query_checkpoint),
        "loaded_tensors": len(remapped),
        "new_refinement_tensors": len(allowed_missing),
    }


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        value: dict[str, Any] = dict(row)
        for key in ("epoch", "step"):
            if value.get(key) not in {None, ""}:
                value[key] = int(float(value[key]))
        for key, item in list(value.items()):
            if key in {"epoch", "step", "backbone_frozen"} or item in {None, ""}:
                continue
            try:
                value[key] = float(item)
            except (TypeError, ValueError):
                pass
        if isinstance(value.get("backbone_frozen"), str):
            value["backbone_frozen"] = value["backbone_frozen"].lower() == "true"
        parsed.append(value)
    return parsed


def update_early_stopping_patience(
    *,
    epoch: int,
    improved: bool,
    minimum_epoch: int,
    no_improvement_validations: int,
) -> int:
    """Update patience without counting validations before ``minimum_epoch``."""
    if improved or epoch < minimum_epoch:
        return 0
    return no_improvement_validations + 1


def _wandb_tag(value: object, *, limit: int = 64) -> str:
    """Return a deterministic W&B tag that respects its length limit."""
    text = str(value)
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: limit - len(digest) - 1]}-{digest}"


def _wandb_run(config: LandmarkExperimentConfig, variant: str, *, enabled: bool):
    if not enabled:
        return None
    import wandb

    value = config.raw.get("wandb", {})
    config_fingerprint = hashlib.sha256(
        json.dumps(config.raw, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    identity = (
        f"{config.raw['experiment_name']}:{variant}:seed={config.raw['train']['seed']}:"
        f"config={config_fingerprint}"
    )
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return wandb.init(
        entity=value["entity"],
        project=value["project"],
        group=value["group"],
        name=f"{config.raw['experiment_name']}-{variant}",
        id=run_id,
        resume="allow",
        mode=value.get("mode", "online"),
        dir=str(config.output_root / "_wandb"),
        config={"variant": variant, **config.raw},
        tags=[
            _wandb_tag(tag)
            for tag in (
                "DINOv3",
                "face-landmark",
                variant,
                config.raw["data"].get("dataset_version", ""),
            )
            if str(tag)
        ],
    )


def train_variant(
    config: LandmarkExperimentConfig,
    *,
    variant: str,
    device: torch.device,
    resume_from: Path | None = None,
    wandb_enabled: bool = False,
) -> dict[str, Any]:
    raw = config.raw
    train_cfg = raw["train"]
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)
    output_dir = config.output_root / variant
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(raw, split="train", training=True)
    val_dataset = make_dataset(raw, split="val", training=False)
    max_train_samples = train_cfg.get("max_train_samples")
    max_val_samples = train_cfg.get("max_val_samples")
    if max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset,
            range(min(int(max_train_samples), len(train_dataset))),
        )
    if max_val_samples is not None:
        val_dataset = torch.utils.data.Subset(
            val_dataset,
            range(min(int(max_val_samples), len(val_dataset))),
        )
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(train_cfg.get("num_workers", 4))
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        collate_fn=collate_landmark_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    model, initialization = build_landmark_model(raw, variant=variant)
    model.to(device)
    refinement_initialization = None
    if variant == "query_refinement" and bool(train_cfg.get("refinement_initialize_from_query", True)):
        query_checkpoint = config.output_root / "query" / "checkpoints" / "best_nme.pt"
        if not query_checkpoint.is_file():
            raise FileNotFoundError(
                f"Query refinement requires the trained query checkpoint first: {query_checkpoint}"
            )
        refinement_initialization = initialize_refinement_from_query(
            model,
            query_checkpoint,
            expected_target_policy=configured_keypoint_target_policy(raw),
        )
    finetune_initialization = None
    finetune_checkpoint = train_cfg.get("finetune_checkpoint")
    if finetune_checkpoint is not None and (resume_from is None or isinstance(model.messenger_recovery, ContinuousRecoveryBlock)):
        if not isinstance(finetune_checkpoint, Mapping):
            raise TypeError("train.finetune_checkpoint must be a mapping")
        finetune_initialization = initialize_from_finetune_checkpoint(
            model,
            finetune_checkpoint,
            variant=variant,
            expected_target_policy=configured_keypoint_target_policy(raw),
            expected_state_head_mode=configured_state_head_mode(raw),
        )
    criterion = LandmarkLoss(raw.get("loss", {})).to(device)
    continuous_teacher = None
    common_initialization = None
    if isinstance(model.messenger_recovery, ContinuousRecoveryBlock):
        if train_cfg.get("record_common_initialization", False) and resume_from is None:
            from .recovery_acceptance import common_state_hash
            common_initialization = dict(common_state_sha256=common_state_hash(model),
                seed=seed, finetune_initialization=finetune_initialization,
                excluded_prefix="messenger_recovery.reference_memory.",
                reference_configuration=raw["model"]["continuous_recovery"])
            expected_common = train_cfg.get("expected_common_initialization_sha256")
            if expected_common and common_initialization["common_state_sha256"] != expected_common:
                raise RuntimeError("Common ABC initialization differs from the accepted initial state")
            (output_dir / "common_initialization.json").write_text(json.dumps(common_initialization, indent=2) + "\n")
        continuous_teacher = prepare_continuous_teacher(model, train_dataset, raw, output_dir, device)
    recovery_criterion = (
        ContinuousRecoveryLoss(model.messenger_recovery.config, patch_size=int(raw["model"]["backbone"].get("patch_size", 16))).to(device)
        if isinstance(model.messenger_recovery, ContinuousRecoveryBlock) else MessengerRecoveryLoss(
            reconstruction_weight=float(
                raw.get("loss", {}).get("messenger_reconstruction_weight", 0.10)
            ),
            gate_weight=float(
                raw.get("loss", {}).get("messenger_gate_weight", 0.01)
            ),
            patch_size=int(
                raw.get("model", {}).get("backbone", {}).get("patch_size", 16)
            ),
        ).to(device)
        if model.messenger_recovery is not None
        else None
    )
    optimizer = _optimizer(model, train_cfg)
    accumulation = int(train_cfg["gradient_accumulation_steps"])
    epochs = int(train_cfg["epochs"])
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = epochs * steps_per_epoch
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(train_cfg.get("warmup_epochs", 5)) * steps_per_epoch,
        minimum_ratio=float(train_cfg.get("minimum_lr_ratio", 0.01)),
    )
    ema = ModelEMA(model, decay=float(train_cfg.get("ema", {}).get("decay", 0.9998)))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(train_cfg.get("amp", True)),
        init_scale=float(train_cfg.get("grad_scaler_init_scale", 65536.0)),
        growth_interval=int(train_cfg.get("grad_scaler_growth_interval", 2000)),
    )
    start_epoch = 1
    global_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if isinstance(model.messenger_recovery, ContinuousRecoveryBlock):
            if checkpoint.get("config", {}).get("model", {}).get("continuous_recovery") != raw["model"]["continuous_recovery"]:
                raise RuntimeError("Refusing to resume a different continuous recovery contract")
            if train_cfg.get("record_common_initialization", False):
                previous_config = checkpoint.get("config", {})
                if (previous_config.get("experiment_name") != raw.get("experiment_name")
                        or previous_config.get("data", {}).get("augmentation", {}).get("paired_occlusion")
                        != raw["data"]["augmentation"]["paired_occlusion"]):
                    raise RuntimeError("Refusing to resume a different ABC arm or material contract")
        if (
            checkpoint.get("keypoint_target_policy")
            != configured_keypoint_target_policy(raw)
            or checkpoint.get("state_head_mode", "legacy_binary")
            != configured_state_head_mode(raw)
        ):
            raise RuntimeError(
                "Refusing to resume a checkpoint with a different state-head "
                "or keypoint-target contract; start a new run"
            )
        if finetune_checkpoint is not None:
            if not isinstance(finetune_checkpoint, Mapping):
                raise TypeError("train.finetune_checkpoint must be a mapping")
            finetune_initialization = dict(
                checkpoint.get("finetune_initialization", {})
            )
            expected_path = Path(str(finetune_checkpoint["path"])).resolve()
            resumed_path = Path(
                str(finetune_initialization.get("path", ""))
            ).resolve()
            if resumed_path != expected_path or str(
                finetune_initialization.get("sha256", "")
            ) != str(finetune_checkpoint.get("sha256", "")):
                raise RuntimeError(
                    "Refusing to resume from a checkpoint with a different "
                    "fine-tune initialization contract"
                )
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.module.load_state_dict(checkpoint.get("ema", checkpoint["model"]), strict=True)
        ema.updates = int(checkpoint.get("ema_updates", 0))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_grad_scaler_state(scaler, checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("step", 0))
    run = _wandb_run(config, variant, enabled=wandb_enabled)
    if run is not None and common_initialization is not None:
        run.config.update({"common_initialization": common_initialization})
    history = _read_history(output_dir / "metrics.csv") if resume_from is not None else []
    early_cfg = dict(train_cfg.get("early_stopping", {}))
    validation_cfg = dict(train_cfg.get("validation", {}) or {})
    validation_enabled = bool(validation_cfg.get("enabled", True))
    validation_every = int(validation_cfg.get("every", 1))
    validation_batch_size = int(
        validation_cfg.get(
            "batch_size",
            raw.get("evaluation", {}).get("batch_size", train_cfg["batch_size"]),
        )
    )
    if validation_every <= 0:
        raise RuntimeError("train.validation.every must be positive")
    early_min_delta = float(early_cfg.get("min_delta", 1e-4))
    early_minimum_epoch = int(early_cfg.get("minimum_epoch", 60))
    early_patience_validations = int(
        early_cfg.get("patience_validations", early_cfg.get("patience", 20))
    )
    selection_cfg = dict(train_cfg.get("checkpoint_selection", {}) or {})
    selection_metric = str(selection_cfg.get("metric", "nme"))
    selection_direction = str(selection_cfg.get("direction", "lower"))
    if selection_direction not in {"lower", "higher"}:
        raise RuntimeError("train.checkpoint_selection.direction must be lower or higher")
    selection_filename = str(
        selection_cfg.get("filename", "best_nme.pt")
    )
    if Path(selection_filename).name != selection_filename or not selection_filename.endswith(".pt"):
        raise RuntimeError("train.checkpoint_selection.filename must be a plain .pt filename")
    selection_history_key = f"val/{selection_metric}"
    best_epoch = 0
    best_nme = float("inf")
    best_selection_value = (
        float("inf") if selection_direction == "lower" else float("-inf")
    )
    no_improvement_validations = 0
    last_val_metrics: dict[str, Any] = {}
    total_amp_overflow_skips = sum(
        int(float(previous.get("train/amp_overflow_skips", 0) or 0))
        for previous in history
    )
    consecutive_amp_overflow_skips = 0
    maximum_consecutive_amp_overflows = int(
        train_cfg.get("maximum_consecutive_amp_overflows", 8)
    )
    if maximum_consecutive_amp_overflows < 1:
        raise RuntimeError("train.maximum_consecutive_amp_overflows must be positive")
    for previous in history:
        if previous.get("val/nme") not in {None, ""}:
            best_nme = min(best_nme, float(previous["val/nme"]))
        if previous.get(selection_history_key) in {None, ""}:
            continue
        previous_value = float(previous[selection_history_key])
        improvement = (
            best_selection_value - previous_value
            if selection_direction == "lower"
            else previous_value - best_selection_value
        )
        improved = improvement > early_min_delta
        if improved:
            best_epoch = int(previous["epoch"])
            best_selection_value = previous_value
        no_improvement_validations = update_early_stopping_patience(
            epoch=int(previous["epoch"]),
            improved=improved,
            minimum_epoch=early_minimum_epoch,
            no_improvement_validations=no_improvement_validations,
        )
    backbone_policy = str(train_cfg.get("backbone_policy", "staged_unfreeze"))
    if backbone_policy not in {"staged_unfreeze", "freeze"}:
        raise RuntimeError(f"Unsupported train.backbone_policy={backbone_policy!r}")
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 5))
    started = time.perf_counter()
    epoch0_scalars = {}
    if train_cfg.get("epoch0_validation", False) and resume_from is not None and (output_dir / "epoch0_validation.json").exists():
        epoch0_scalars = json.loads((output_dir / "epoch0_validation.json").read_text())["metrics"]
    try:
        if train_cfg.get("epoch0_validation", False) and resume_from is None:
            # Diagnostic only: never add epoch zero to selection history/patience.
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                epoch0_metrics, _ = evaluate_gt_boxes(ema.module, val_dataset, device=device,
                    batch_size=validation_batch_size, num_workers=num_workers,
                    input_size=int(raw["data"]["input_size"]), attention_batches=0)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            epoch0_scalars = {k: v for k, v in epoch0_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
            (output_dir / "epoch0_validation.json").write_text(json.dumps(dict(
                diagnostic_only=True, selection_eligible=False, metrics=epoch0_scalars), indent=2) + "\n")
            if run is not None:
                run.log({f"epoch0/{k}": v for k, v in epoch0_scalars.items()})
        for epoch in range(start_epoch, epochs + 1):
            synchronized_timing = bool(train_cfg.get("synchronized_timing", False))
            if synchronized_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            epoch_dataset = (
                train_dataset.dataset
                if isinstance(train_dataset, torch.utils.data.Subset)
                else train_dataset
            )
            if hasattr(epoch_dataset, "set_epoch"):
                epoch_dataset.set_epoch(
                    int(train_cfg.get("fixed_augmentation_epoch", epoch))
                )
            fixed_augmentation_epoch = train_cfg.get("fixed_augmentation_epoch")
            if fixed_augmentation_epoch is not None:
                reset_fixed_augmentation_rng(
                    seed=seed,
                    augmentation_epoch=int(fixed_augmentation_epoch),
                    loader_generator=generator,
                )
            backbone_frozen = backbone_policy == "freeze" or epoch <= freeze_epochs
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(not backbone_frozen)
            model.train()
            if isinstance(model.messenger_recovery, ContinuousRecoveryBlock):
                model.messenger_recovery.mixing_enabled = epoch > model.messenger_recovery.config.mixing_warmup_epochs
            if backbone_frozen:
                model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            running: dict[str, float] = {}
            batches = 0
            epoch_amp_overflow_skips = 0
            for batch_index, (images, targets) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                augmentation_valid_mask = None
                if variant == "query_refinement":
                    augmentation_valid_mask = torch.stack(
                        [target["labeled"].to(dtype=torch.bool) for target in targets]
                    ).to(device=device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    output = model(
                        images,
                        augmentation_valid_mask=augmentation_valid_mask,
                        return_recovery_diagnostics=recovery_criterion is not None,
                        **face_context_kwargs(model, targets, device),
                    )
                    losses = criterion(output, targets)
                    if isinstance(recovery_criterion, ContinuousRecoveryLoss):
                        messenger_losses = continuous_training_losses(model, continuous_teacher, recovery_criterion, output, targets, device)
                        losses["loss_total"] = losses["loss_total"] + messenger_losses["loss_messenger_total"]
                        losses.update(messenger_losses)
                    elif recovery_criterion is not None:
                        diagnostics = output.get("recovery_diagnostics")
                        if not isinstance(diagnostics, dict):
                            raise RuntimeError(
                                "Messenger training requires recovery diagnostics"
                            )
                        selected = [
                            index
                            for index, target in enumerate(targets)
                            if bool(target["occlusion_mask"].any())
                        ]
                        if selected:
                            clean_images = torch.stack(
                                [targets[index]["clean_image"] for index in selected]
                            ).to(device=device, non_blocking=True)
                            pixel_masks = torch.stack(
                                [targets[index]["occlusion_mask"] for index in selected]
                            ).to(device=device, non_blocking=True)
                            with torch.no_grad():
                                clean_feature_map = model.extract_backbone_feature_map(
                                    clean_images
                                )
                            selected_diagnostics = {
                                key: value[selected]
                                for key, value in diagnostics.items()
                                if key != "attention"
                            }
                            messenger_losses = recovery_criterion(
                                selected_diagnostics,
                                clean_feature_map,
                                pixel_masks,
                            )
                        else:
                            zero = diagnostics["messenger_tokens"].sum() * 0.0
                            messenger_losses = {
                                "loss_messenger_reconstruction": zero,
                                "loss_messenger_gate": zero,
                                "loss_messenger_total": zero,
                                "messenger_masked_patches": zero,
                            }
                        losses["loss_total"] = (
                            losses["loss_total"]
                            + messenger_losses["loss_messenger_total"]
                        )
                        losses.update(messenger_losses)
                    loss = losses["loss_total"] / accumulation
                if not bool(torch.isfinite(loss).all().item()):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch} batch {batch_index}"
                    )
                scaler.scale(loss).backward()
                should_step = batch_index % accumulation == 0 or batch_index == len(loader)
                if should_step:
                    step_result = step_optimizer_with_amp_recovery(
                        model,
                        optimizer,
                        scaler,
                        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                    )
                    if not step_result.optimizer_step_succeeded:
                        epoch_amp_overflow_skips += 1
                        total_amp_overflow_skips += 1
                        consecutive_amp_overflow_skips += 1
                        LOGGER.warning(
                            "[train:%s] skipped AMP-overflow update at epoch=%d "
                            "batch=%d grad_norm=%s scale=%.1f->%.1f consecutive=%d",
                            variant,
                            epoch,
                            batch_index,
                            step_result.gradient_norm,
                            step_result.scale_before_step,
                            step_result.scale_after_step,
                            consecutive_amp_overflow_skips,
                        )
                        if (
                            consecutive_amp_overflow_skips
                            > maximum_consecutive_amp_overflows
                        ):
                            raise RuntimeError(
                                "Exceeded maximum consecutive AMP overflows at "
                                f"epoch {epoch} batch {batch_index}"
                            )
                    else:
                        consecutive_amp_overflow_skips = 0
                        scheduler.step()
                        ema.update(model)
                        global_step += 1
                for key, value in losses.items():
                    running[key] = running.get(key, 0.0) + float(value.detach().cpu())
                batches += 1
            if synchronized_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - epoch_started
            should_validate = validation_enabled and (
                epoch % validation_every == 0 or epoch == epochs
            )
            val_metrics: dict[str, Any] | None = None
            validation_seconds: float | None = None
            if should_validate:
                if synchronized_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_started = time.perf_counter()
                val_metrics, _rows = evaluate_gt_boxes(
                    ema.module,
                    val_dataset,
                    device=device,
                    batch_size=validation_batch_size,
                    num_workers=num_workers,
                    input_size=int(raw["data"]["input_size"]),
                    attention_batches=0,
                )
                if synchronized_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_seconds = time.perf_counter() - validation_started
                last_val_metrics = dict(val_metrics)
            row: dict[str, Any] = {
                "epoch": epoch,
                "step": global_step,
                "backbone_frozen": backbone_frozen,
                "lr_backbone": next(group["lr"] for group in optimizer.param_groups if str(group["name"]).startswith("backbone")),
                "lr_head": next(group["lr"] for group in optimizer.param_groups if str(group["name"]).startswith("head")),
                **{f"train/{key}": value / max(1, batches) for key, value in running.items()},
                "train/amp_overflow_skips": epoch_amp_overflow_skips,
                "train/amp_overflow_skips_total": total_amp_overflow_skips,
                "train/grad_scaler_scale": float(scaler.get_scale()),
                "validation/completed": bool(should_validate),
                "timing/synchronized": synchronized_timing,
                "timing/train_seconds": training_seconds,
                "timing/validation_seconds": validation_seconds,
                "validation/every": validation_every,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            cnn_group = next(
                (
                    group
                    for group in optimizer.param_groups
                    if str(group["name"]).startswith("cnn")
                ),
                None,
            )
            if cnn_group is not None:
                row["lr_cnn"] = cnn_group["lr"]
            messenger_group = next(
                (
                    group
                    for group in optimizer.param_groups
                    if str(group["name"]).startswith("messenger")
                ),
                None,
            )
            if messenger_group is not None:
                row["lr_messenger"] = messenger_group["lr"]
            if val_metrics is not None:
                if selection_metric in epoch0_scalars and val_metrics.get(selection_metric) is not None:
                    row["validation/selection_delta_from_epoch0"] = float(val_metrics[selection_metric]) - float(epoch0_scalars[selection_metric])
                row.update(
                    {
                        "val/nme": float(val_metrics["nme"]),
                        "val/pck_0.05": val_metrics["pck_0.05"],
                        "val/pck_0.10": val_metrics["pck_0.10"],
                        "val/visibility_f1": val_metrics.get("visibility_f1"),
                        "val/occluded_f1": val_metrics.get("occluded_f1"),
                        "val/visibility_occluded_macro_f1": val_metrics.get(
                            "visibility_occluded_macro_f1"
                        ),
                        "val/presence_f1": val_metrics.get("presence_f1"),
                        "val/missing_f1": val_metrics.get("missing_f1"),
                        "val/presence_missing_macro_f1": val_metrics.get(
                            "presence_missing_macro_f1"
                        ),
                        "val/state_macro_f1": val_metrics.get("state_macro_f1"),
                        "val/conditional_semantic_macro_f1_nme_0.10": val_metrics.get(
                            "conditional_semantic_macro_f1_nme_0.10"
                        ),
                    }
                )
            history.append(row)
            _write_history(output_dir / "metrics.csv", history)
            state = checkpoint_state(
                model,
                ema,
                optimizer,
                scheduler,
                epoch=epoch,
                step=global_step,
                variant=variant,
                config=raw,
                metrics=val_metrics or last_val_metrics,
                backbone_initialization=initialization,
                finetune_initialization=finetune_initialization,
                scaler=scaler,
            )
            torch.save(state, checkpoint_dir / "last.pt")
            checkpoint_period = int(train_cfg.get("checkpoint_every", 5))
            if checkpoint_period > 0 and epoch % checkpoint_period == 0:
                torch.save(state, checkpoint_dir / f"epoch_{epoch:04d}.pt")
                prune_periodic_checkpoints(checkpoint_dir, keep=1)
            if val_metrics is not None:
                nme = float(val_metrics["nme"])
                best_nme = min(best_nme, nme)
                selection_value = float(val_metrics[selection_metric])
                improvement = (
                    best_selection_value - selection_value
                    if selection_direction == "lower"
                    else selection_value - best_selection_value
                )
                improved = improvement > early_min_delta
                if improved:
                    best_selection_value = selection_value
                    best_epoch = epoch
                    torch.save(state, checkpoint_dir / selection_filename)
                no_improvement_validations = update_early_stopping_patience(
                    epoch=epoch,
                    improved=improved,
                    minimum_epoch=early_minimum_epoch,
                    no_improvement_validations=no_improvement_validations,
                )
            if run is not None:
                run.log(row, step=epoch)
            if val_metrics is not None:
                LOGGER.info(
                    "[train:%s] epoch=%d/%d nme=%.6f pck@.05=%.4f best_%s=%.6f frozen=%s train=%.1fs val=%.1fs",
                    variant,
                    epoch,
                    epochs,
                    float(val_metrics["nme"]),
                    float(val_metrics["pck_0.05"]),
                    selection_metric,
                    best_selection_value,
                    backbone_frozen,
                    training_seconds,
                    validation_seconds or 0.0,
                )
                if (
                    bool(early_cfg.get("enabled", True))
                    and epoch >= early_minimum_epoch
                    and no_improvement_validations >= early_patience_validations
                ):
                    break
            else:
                LOGGER.info(
                    "[train:%s] epoch=%d/%d validation=deferred frozen=%s train=%.1fs",
                    variant,
                    epoch,
                    epochs,
                    backbone_frozen,
                    training_seconds,
                )
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    elapsed = time.perf_counter() - started
    prune_periodic_checkpoints(checkpoint_dir, keep=0)
    summary = {
        "status": "complete",
        "variant": variant,
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "best_epoch": best_epoch,
        "best_nme": best_nme,
        "checkpoint_selection": {
            "metric": selection_metric,
            "direction": selection_direction,
            "best_value": best_selection_value,
            "minimum_delta": early_min_delta,
        },
        "last_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "dino_initialization": initialization,
        "refinement_initialization": refinement_initialization,
        "finetune_initialization": finetune_initialization,
        "backbone_policy": backbone_policy,
        "keypoint_target_policy": configured_keypoint_target_policy(raw),
        "state_head_mode": configured_state_head_mode(raw),
        "best_checkpoint": str(checkpoint_dir / selection_filename),
        "validation_every": validation_every,
        "completed_validations": sum(
            row.get("val/nme") not in {None, ""} for row in history
        ),
        "early_stopping_patience_validations": early_patience_validations,
        "messenger_recovery": {
            "enabled": model.messenger_recovery is not None,
            "reconstruction_weight": float(
                raw.get("loss", {}).get("messenger_reconstruction_weight", 0.0)
            ),
            "gate_weight": float(
                raw.get("loss", {}).get("messenger_gate_weight", 0.0)
            ),
        },
    }
    if isinstance(model.messenger_recovery, ContinuousRecoveryBlock):
        summary["continuous_recovery"] = {
            "configuration": raw["model"]["continuous_recovery"],
            "teacher_provenance": str(output_dir / "continuous_teacher_provenance.json"),
            "teacher_policy": "fixed_baseline",
            "student_unfreeze_epoch": freeze_epochs + 1,
        }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if run is not None:
        run.summary.update(summary)
        run.finish(exit_code=0)
    return summary
