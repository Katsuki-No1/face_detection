from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import LandmarkExperimentConfig
from .dataset import CropAugmentation, TopDownFaceLandmarkDataset, collate_landmark_batch
from .evaluation import evaluate_gt_boxes
from .losses import LandmarkLoss
from .model import DinoFaceLandmarkModel, build_landmark_model


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


def _augmentation(config: Mapping[str, Any]) -> CropAugmentation:
    value = dict(config.get("augmentation", {}))
    return CropAugmentation(
        scale_factor=float(value.get("scale_factor", 1.30)),
        scale_jitter=float(value.get("scale_jitter", 0.20)),
        center_jitter=float(value.get("center_jitter", 0.08)),
        rotation_deg=float(value.get("rotation_deg", 20.0)),
        horizontal_flip_probability=float(value.get("horizontal_flip_probability", 0.50)),
        brightness=float(value.get("brightness", 0.20)),
        contrast=float(value.get("contrast", 0.20)),
        saturation=float(value.get("saturation", 0.15)),
        blur_probability=float(value.get("blur_probability", 0.10)),
        blur_kernel_max=int(value.get("blur_kernel_max", 5)),
        occlusion_probability=float(value.get("occlusion_probability", 0.10)),
        occlusion_area=tuple(float(item) for item in value.get("occlusion_area", (0.02, 0.12))),
    )


def make_dataset(config: Mapping[str, Any], *, split: str, training: bool) -> TopDownFaceLandmarkDataset:
    data = config["data"]
    return TopDownFaceLandmarkDataset(
        annotation_path=data["annotations"][split],
        image_root=data["root"],
        input_size=int(data["input_size"]),
        training=training,
        augmentation=_augmentation(data),
        mean=data.get("mean", (0.0, 0.0, 0.0)),
        std=data.get("std", (1.0, 1.0, 1.0)),
        category_ids=data.get("category_ids", (1,)),
    )


def _optimizer(model: DinoFaceLandmarkModel, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    backbone_lr = float(train_cfg.get("lr_backbone", 1.0e-5))
    head_lr = float(train_cfg.get("lr_head", 5.0e-4))

    def split(module: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, parameter in module.named_parameters():
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
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head_no_decay"},
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
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "variant": variant,
        "config": dict(config),
        "metrics": dict(metrics),
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
        "metrics": checkpoint.get("metrics", {}),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def initialize_refinement_from_query(model: DinoFaceLandmarkModel, query_checkpoint: Path) -> dict[str, Any]:
    checkpoint = torch.load(query_checkpoint, map_location="cpu", weights_only=False)
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
        tags=["DINOv3", "face-landmark", variant, str(config.raw["data"].get("dataset_version", ""))],
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
        refinement_initialization = initialize_refinement_from_query(model, query_checkpoint)
    criterion = LandmarkLoss(raw.get("loss", {})).to(device)
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True)))
    start_epoch = 1
    global_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.module.load_state_dict(checkpoint.get("ema", checkpoint["model"]), strict=True)
        ema.updates = int(checkpoint.get("ema_updates", 0))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("step", 0))
    run = _wandb_run(config, variant, enabled=wandb_enabled)
    history = _read_history(output_dir / "metrics.csv") if resume_from is not None else []
    previous_nmes = [
        (int(row["epoch"]), float(row["val/nme"]))
        for row in history
        if row.get("val/nme") not in {None, ""}
    ]
    best_epoch, best_nme = min(previous_nmes, key=lambda item: item[1]) if previous_nmes else (0, float("inf"))
    no_improvement = max(0, (int(history[-1]["epoch"]) - best_epoch)) if history else 0
    early_cfg = dict(train_cfg.get("early_stopping", {}))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 5))
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.perf_counter()
            backbone_frozen = epoch <= freeze_epochs
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(not backbone_frozen)
            model.train()
            if backbone_frozen:
                model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            running: dict[str, float] = {}
            batches = 0
            for batch_index, (images, targets) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    output = model(images)
                    losses = criterion(output, targets)
                    loss = losses["loss_total"] / accumulation
                scaler.scale(loss).backward()
                should_step = batch_index % accumulation == 0 or batch_index == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    ema.update(model)
                    global_step += 1
                for key, value in losses.items():
                    running[key] = running.get(key, 0.0) + float(value.detach().cpu())
                batches += 1
            val_metrics, _rows = evaluate_gt_boxes(
                ema.module,
                val_dataset,
                device=device,
                batch_size=int(raw.get("evaluation", {}).get("batch_size", train_cfg["batch_size"])),
                num_workers=num_workers,
                input_size=int(raw["data"]["input_size"]),
                attention_batches=0,
            )
            nme = float(val_metrics["nme"])
            row: dict[str, Any] = {
                "epoch": epoch,
                "step": global_step,
                "backbone_frozen": backbone_frozen,
                "lr_backbone": next(group["lr"] for group in optimizer.param_groups if str(group["name"]).startswith("backbone")),
                "lr_head": next(group["lr"] for group in optimizer.param_groups if str(group["name"]).startswith("head")),
                **{f"train/{key}": value / max(1, batches) for key, value in running.items()},
                "val/nme": nme,
                "val/pck_0.05": val_metrics["pck_0.05"],
                "val/pck_0.10": val_metrics["pck_0.10"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
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
                metrics=val_metrics,
            )
            torch.save(state, checkpoint_dir / "last.pt")
            if epoch % int(train_cfg.get("checkpoint_every", 5)) == 0:
                torch.save(state, checkpoint_dir / f"epoch_{epoch:04d}.pt")
            improvement = best_nme - nme
            if improvement > float(early_cfg.get("min_delta", 1e-4)):
                best_nme = nme
                best_epoch = epoch
                no_improvement = 0
                torch.save(state, checkpoint_dir / "best_nme.pt")
            else:
                no_improvement += 1
            if run is not None:
                run.log(row, step=epoch)
            LOGGER.info(
                "[train:%s] epoch=%d/%d nme=%.6f pck@.05=%.4f best_nme=%.6f frozen=%s time=%.1fs",
                variant,
                epoch,
                epochs,
                nme,
                float(val_metrics["pck_0.05"]),
                best_nme,
                backbone_frozen,
                row["epoch_seconds"],
            )
            if (
                bool(early_cfg.get("enabled", True))
                and epoch >= int(early_cfg.get("minimum_epoch", 60))
                and no_improvement >= int(early_cfg.get("patience", 20))
            ):
                break
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "variant": variant,
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "best_epoch": best_epoch,
        "best_nme": best_nme,
        "last_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "dino_initialization": initialization,
        "refinement_initialization": refinement_initialization,
        "best_checkpoint": str(checkpoint_dir / "best_nme.pt"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if run is not None:
        run.summary.update(summary)
        run.finish(exit_code=0)
    return summary
