#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.head_codino.checkpoint import (
    initialize_detection_head,
    load_checkpoint_exact,
)
from src.checkpoint_retention import prune_periodic_checkpoints
from src.models.head_codino.runtime import (
    build_dataloader,
    build_model_from_runtime_config,
    cuda_operator_preflight,
    load_runtime_config,
    move_targets_to_device,
    representative_subset_indices,
    resolve_path,
    seed_everything,
)
from src.models.head_codino.structures import collate_head_detection_batch
from src.wandb_utils import (
    stable_wandb_run_id,
    verify_wandb_access,
    write_json_atomic,
)


HEAD_WANDB_METRIC_SCHEMA_VERSION = 2
HEAD_LOSS_NAMES = ("loss_cls", "loss_bbox", "loss_giou", "loss_total")
OVERFIT_REQUIRED_DECREASES = ("loss_bbox", "loss_giou", "loss_total")


def assess_overfit_run(
    first_losses: dict[str, float] | None,
    last_losses: dict[str, float] | None,
    *,
    optimizer_steps: int,
    box_movement: float | None,
) -> dict[str, bool]:
    """Assess learning while treating Quality Focal loss as a diagnostic.

    Its positive target is the current detached IoU.  Consequently the
    classification loss can rise from a near-zero initial value as box quality
    improves, so monotonic classification-loss decrease is not required.
    """
    losses_are_finite = bool(
        first_losses is not None
        and last_losses is not None
        and all(
            math.isfinite(first_losses[key]) and math.isfinite(last_losses[key])
            for key in HEAD_LOSS_NAMES
        )
    )
    return {
        "optimizer_step_completed": optimizer_steps > 0,
        "losses_are_finite": losses_are_finite,
        "required_losses_decreased": bool(
            losses_are_finite
            and all(last_losses[key] < first_losses[key] for key in OVERFIT_REQUIRED_DECREASES)
        ),
        "prediction_boxes_moved": bool(
            box_movement is not None
            and math.isfinite(box_movement)
            and box_movement > 0.0
        ),
    }


def head_codino_wandb_run_id(
    config: dict[str, Any],
    *,
    entity: str | None,
    project: str,
) -> str:
    return stable_wandb_run_id(
        {
            "architecture": "shared_head_codino_vits16plus_960",
            "entity": entity,
            "project": project,
            "train_annotation": config["data"]["train"]["annotation"],
            "image_size": int(config["data"]["image_size"]),
            "seed": int(config["train"].get("seed", 42)),
            "output_dir": config["train"]["output_dir"],
            "metric_schema_version": int(
                config.get("wandb", {}).get(
                    "metric_schema_version",
                    HEAD_WANDB_METRIC_SCHEMA_VERSION,
                )
            ),
        }
    )


def _read_head_metric_rows(path: Path) -> list[dict[str, int | float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, int | float]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, int | float] = {}
            for key, value in raw.items():
                if value in {None, ""}:
                    continue
                row[key] = (
                    int(float(value))
                    if key in {"epoch", "global_step"}
                    else float(value)
                )
            rows.append(row)
    return rows


def head_codino_wandb_metric_row(
    row: dict[str, int | float],
    *,
    previous_elapsed_seconds: float = 0.0,
) -> dict[str, int | float]:
    elapsed = float(row["elapsed_seconds"])
    result: dict[str, int | float] = {
        "epoch": int(row["epoch"]),
        "progress/global_step": int(row["global_step"]),
        "timing/epoch_minutes": max(0.0, elapsed - previous_elapsed_seconds)
        / 60.0,
        "timing/elapsed_hours": elapsed / 3600.0,
        "optimization/learning_rate_min": float(row["learning_rate_min"]),
        "optimization/learning_rate_max": float(row["learning_rate_max"]),
    }
    for split in ("train", "val"):
        result[f"overview/{split}_total_loss"] = float(
            row[f"{split}/loss_total"]
        )
        for loss_name in HEAD_LOSS_NAMES[:-1]:
            result[f"loss_components/{split}_{loss_name.removeprefix('loss_')}"] = float(
                row[f"{split}/{loss_name}"]
            )
    return result


def initialize_head_wandb(
    config: dict[str, Any],
    *,
    output_dir: Path,
) -> Any:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is required when HEAD training logging is enabled"
        ) from error
    settings = dict(config.get("wandb", {}))
    entity_value = settings.get("entity")
    entity = str(entity_value) if entity_value else None
    project = str(settings.get("project", "face_detection"))
    mode = str(settings.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    schema_version = int(
        settings.get("metric_schema_version", HEAD_WANDB_METRIC_SCHEMA_VERSION)
    )
    run_id = head_codino_wandb_run_id(
        config,
        entity=entity,
        project=project,
    )
    state_path = output_dir / "wandb_run.json"
    previous = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    previous_id = previous.get("run_id")
    if previous_id not in {None, run_id}:
        raise RuntimeError(
            "Stored W&B run ID does not match this HEAD training condition: "
            f"{previous_id} != {run_id}"
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
        raise ValueError("wandb.live_sync_seconds must be positive")
    run = wandb.init(
        entity=entity,
        project=project,
        group=settings.get("group"),
        name=settings.get("name"),
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(local_dir),
        tags=list(settings.get("tags", [])),
        config=config,
        job_type="training",
        notes=(
            "Shared HEAD training: overview contains epoch train/validation loss; "
            "loss_components contains classification/box/GIoU diagnostics; final "
            "COCO metrics are attached by evaluation."
        ),
        settings=wandb.Settings(
            x_file_stream_transmit_interval=live_sync_seconds,
        ),
    )
    run.define_metric("epoch")
    run.define_metric("progress/global_step")
    run.define_metric(
        "train_batch/*",
        step_metric="progress/global_step",
        summary="last",
    )
    for namespace in ("overview/*", "loss_components/*", "optimization/*", "timing/*", "progress/*"):
        run.define_metric(namespace, step_metric="epoch", summary="last")
    run.config.update(
        {
            "wandb_metric_schema_version": schema_version,
            "wandb_metric_guide": {
                "overview": "epoch train/validation total loss",
                "loss_components": "classification, L1 box, and GIoU losses",
                "optimization": "optimizer learning-rate range",
                "evaluation": "val/test COCO metrics and source artifacts",
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
    for row in _read_head_metric_rows(output_dir / "metrics.csv"):
        epoch = int(row["epoch"])
        if epoch <= int(state["last_logged_epoch"]):
            continue
        run.log(
            head_codino_wandb_metric_row(
                row,
                previous_elapsed_seconds=float(state["last_elapsed_seconds"]),
            ),
            step=int(row["global_step"]),
        )
        state["last_logged_epoch"] = epoch
        state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
        write_json_atomic(state_path, state)
    return run


def log_head_wandb_epoch(
    run: Any,
    *,
    output_dir: Path,
    row: dict[str, int | float],
) -> None:
    state_path = output_dir / "wandb_run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run.log(
        head_codino_wandb_metric_row(
            row,
            previous_elapsed_seconds=float(state.get("last_elapsed_seconds", 0.0)),
        ),
        step=int(row["global_step"]),
    )
    state["last_logged_epoch"] = int(row["epoch"])
    state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
    write_json_atomic(state_path, state)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            current = source[name].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                value.copy_(current)


def _distributed_context() -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        rank = 0
    return distributed, rank, world_size, torch.device("cuda", local_rank)


def _optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.AdamW:
    groups = []
    group_specs = (
        ("backbone", float(train_cfg["backbone_lr"])),
        ("sfp", float(train_cfg["sfp_lr"])),
        ("query_head", float(train_cfg["head_lr"])),
    )
    assigned: set[int] = set()
    for prefix, learning_rate in group_specs:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(prefix + ".") and parameter.requires_grad
        ]
        assigned.update(id(parameter) for parameter in parameters)
        if parameters:
            groups.append({"params": parameters, "lr": learning_rate, "name": prefix})
    unassigned = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    if unassigned:
        raise RuntimeError(f"{len(unassigned)} trainable parameters were not assigned an optimizer group")
    return torch.optim.AdamW(
        groups,
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
        fused=bool(train_cfg.get("fused_adamw", True)),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv3 compact Co-DINO Head detector")
    parser.add_argument("--config", default="configs/head_codino/production.yaml")
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-head-from",
        help=(
            "Warm-start SFP and Co-DINO from a prior detector checkpoint "
            "without restoring its frozen DINO or optimizer state."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overfit-images",
        type=int,
        choices=range(4, 9),
        metavar="N",
        help="Restrict training to the first 4–8 real images",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    train_cfg = dict(config["train"])
    distributed, rank, world_size, device = _distributed_context()
    seed_everything(int(train_cfg.get("seed", 42)), rank=rank)
    preflight = cuda_operator_preflight(device)
    if rank == 0:
        print(json.dumps({"preflight": preflight, "world_size": world_size}, ensure_ascii=False))

    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_path is not None:
        # A full strict checkpoint supplies the backbone; initial pretrained
        # weights are unnecessary and may no longer be present.
        config = copy.deepcopy(config)
        config["model"]["backbone_weights"] = None
    model = build_model_from_runtime_config(config).to(device)
    initialization_report = None
    if args.initialize_head_from and resume_path is None:
        initialization_report = initialize_detection_head(
            model,
            Path(args.initialize_head_from).expanduser().resolve(),
            prefer_ema=True,
        )
    start_epoch = 0
    global_step = 0
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        load_checkpoint_exact(model, resume_path, prefer_ema=False)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        global_step = int(resume_payload.get("global_step", 0))

    optimizer = _optimizer(model, train_cfg)
    optimizer_report = [
        {
            "name": str(group.get("name", index)),
            "lr": float(group["lr"]),
            "parameters": sum(
                int(parameter.numel()) for parameter in group["params"]
            ),
            "tensors": len(group["params"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in train_cfg.get("milestones", [])],
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(train_cfg.get("grad_scaler_init_scale", 65536.0)),
        enabled=bool(train_cfg.get("amp", True)),
    )
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.9998)))
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload.get("scaler", {}))
        ema.module.load_state_dict(resume_payload["ema"], strict=True)

    training_model: torch.nn.Module
    if distributed:
        training_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        training_model = model
    bounded_subset_audit: dict[str, Any] = {}
    loader, sampler = build_dataloader(config, "train", distributed=distributed)
    if (
        args.max_train_samples is not None
        and args.max_train_samples < len(loader.dataset)
    ):
        subset_indices, subset_report = representative_subset_indices(
            loader.dataset,
            int(args.max_train_samples),
        )
        bounded_subset_audit["train"] = subset_report
        subset = Subset(loader.dataset, subset_indices)
        sampler = (
            DistributedSampler(subset, shuffle=True, drop_last=False)
            if distributed
            else None
        )
        loader = DataLoader(
            subset,
            batch_size=int(train_cfg.get("batch_size", 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
            collate_fn=collate_head_detection_batch,
            drop_last=False,
        )
    if args.overfit_images is not None:
        subset = Subset(loader.dataset, range(args.overfit_images))
        sampler = (
            DistributedSampler(subset, shuffle=True, drop_last=False) if distributed else None
        )
        loader = DataLoader(
            subset,
            batch_size=int(train_cfg.get("batch_size", 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
            collate_fn=collate_head_detection_batch,
            drop_last=False,
        )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_path(train_cfg["output_dir"], config=config)
    )
    if rank == 0 and initialization_report is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "initialization_audit.json",
            initialization_report,
        )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "optimizer_groups.json",
            {"groups": optimizer_report},
        )
    validation_loader = None
    validate_each_epoch = bool(
        train_cfg.get("validate_each_epoch", False)
    ) and args.overfit_images is None
    metrics_path = (
        args.metrics_csv.resolve()
        if args.metrics_csv
        else (output_dir / "metrics.csv" if validate_each_epoch else None)
    )
    metric_rows: list[dict[str, int | float]] = (
        _read_head_metric_rows(metrics_path)
        if metrics_path is not None
        else []
    )
    best_val_loss = min(
        (float(row["val/loss_total"]) for row in metric_rows),
        default=float("inf"),
    )
    elapsed_before_start = (
        float(metric_rows[-1].get("elapsed_seconds", 0.0))
        if metric_rows
        else 0.0
    )
    training_started = time.monotonic()
    if metrics_path is not None:
        validation_loader, _ = build_dataloader(config, "val", distributed=False)
        if (
            args.max_val_samples is not None
            and args.max_val_samples < len(validation_loader.dataset)
        ):
            validation_indices, validation_report = representative_subset_indices(
                validation_loader.dataset,
                int(args.max_val_samples),
            )
            bounded_subset_audit["val"] = validation_report
            validation_subset = Subset(
                validation_loader.dataset,
                validation_indices,
            )
            validation_loader = DataLoader(
                validation_subset,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
                shuffle=False,
                num_workers=int(config.get("evaluation", {}).get("num_workers", 2)),
                pin_memory=True,
                persistent_workers=int(config.get("evaluation", {}).get("num_workers", 2)) > 0,
                collate_fn=collate_head_detection_batch,
                drop_last=False,
            )
    if rank == 0 and bounded_subset_audit:
        write_json_atomic(
            output_dir / "bounded_subset_audit.json",
            bounded_subset_audit,
        )
    wandb_run = None
    wandb_cfg = config.get("wandb", {})
    wandb_enabled = (
        bool(args.wandb)
        if args.wandb is not None
        else bool(isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled", False))
    )
    if rank == 0 and wandb_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = initialize_head_wandb(config, output_dir=output_dir)
    overfit_success_path = output_dir / "overfit_success.json"
    if rank == 0 and args.overfit_images is not None:
        overfit_success_path.unlink(missing_ok=True)
    accumulation = int(train_cfg.get("accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive")
    clip_norm = float(train_cfg.get("gradient_clip_norm", 0.1))
    log_interval = int(train_cfg.get("log_interval", 20))
    optimizer.zero_grad(set_to_none=True)
    should_stop = False
    last_completed_epoch = start_epoch - 1
    first_losses: dict[str, float] | None = None
    last_losses: dict[str, float] | None = None
    track_overfit = rank == 0 and args.overfit_images is not None
    probe_batch = None
    initial_probe_boxes = None
    if rank == 0 and not distributed and args.overfit_images is not None:
        probe_batch, _probe_targets = collate_head_detection_batch([loader.dataset[0]])
        probe_batch = probe_batch.to(device)
        model.eval()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=bool(train_cfg.get("amp", True))
        ):
            initial_probe_boxes = model(probe_batch, return_raw=True)["pred_boxes"].float().cpu()

    try:
        epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
        for epoch in range(start_epoch, epochs):
            improved = False
            if sampler is not None:
                sampler.set_epoch(epoch)
            training_model.train()
            epoch_loss_sums = {name: 0.0 for name in HEAD_LOSS_NAMES}
            epoch_loss_batches = 0
            for batch_index, (batch, targets) in enumerate(loader):
                batch = batch.to(device, non_blocking=True)
                targets = move_targets_to_device(targets, device)
                final_microbatch = batch_index + 1 == len(loader)
                synchronize = (batch_index + 1) % accumulation == 0 or final_microbatch
                ddp_context = (
                    training_model.no_sync()
                    if distributed and not synchronize
                    else torch.enable_grad()
                )
                with ddp_context:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=bool(train_cfg.get("amp", True)),
                    ):
                        losses = training_model(batch, targets)
                        final_window = len(loader) - (batch_index // accumulation) * accumulation
                        window_size = min(accumulation, final_window)
                        scaled_loss = losses["loss_total"] / window_size
                    for loss_name in HEAD_LOSS_NAMES:
                        epoch_loss_sums[loss_name] += float(
                            losses[loss_name].detach()
                        )
                    epoch_loss_batches += 1
                    scaler.scale(scaled_loss).backward()
                scalar_losses: dict[str, float] = {}
                if track_overfit:
                    scalar_losses = {
                        key: float(value.detach())
                        for key, value in losses.items()
                        if key in {"loss_cls", "loss_bbox", "loss_giou", "loss_total"}
                    }
                    if first_losses is None:
                        first_losses = scalar_losses
                    last_losses = scalar_losses
                if not synchronize:
                    continue
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_norm,
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_succeeded = scaler.get_scale() >= scale_before_step
                if not optimizer_step_succeeded:
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "amp_step": "skipped_non_finite_gradient",
                                    "new_scale": scaler.get_scale(),
                                },
                                ensure_ascii=False,
                            )
                        )
                    continue
                ema.update(model)
                global_step += 1
                if rank == 0 and global_step % log_interval == 0:
                    if not scalar_losses:
                        scalar_losses = {
                            key: float(value.detach())
                            for key, value in losses.items()
                            if key in {"loss_cls", "loss_bbox", "loss_giou", "loss_total"}
                        }
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss_total": float(losses["loss_total"].detach()),
                        "gradient_norm": float(gradient_norm),
                        "lr": [group["lr"] for group in optimizer.param_groups],
                        **scalar_losses,
                    }
                    print(json.dumps(record, ensure_ascii=False))
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train_batch/loss_total": record["loss_total"],
                                "train_batch/loss_cls": record.get("loss_cls"),
                                "train_batch/loss_bbox": record.get("loss_bbox"),
                                "train_batch/loss_giou": record.get("loss_giou"),
                                "train_batch/gradient_norm": record["gradient_norm"],
                                "optimization/learning_rate_min": min(record["lr"]),
                                "optimization/learning_rate_max": max(record["lr"]),
                                "epoch": epoch + 1,
                                "progress/global_step": global_step,
                            },
                            step=global_step,
                        )
                if args.max_steps is not None and global_step >= args.max_steps:
                    should_stop = True
                    break
            scheduler.step()
            if rank == 0 and validation_loader is not None:
                validation_sums = {name: 0.0 for name in HEAD_LOSS_NAMES}
                validation_batches = 0
                ema.module.train()
                with torch.inference_mode():
                    for validation_batch, validation_targets in validation_loader:
                        validation_batch = validation_batch.to(device, non_blocking=True)
                        validation_targets = move_targets_to_device(
                            validation_targets,
                            device,
                        )
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=bool(train_cfg.get("amp", True)),
                        ):
                            validation_losses = ema.module(
                                validation_batch,
                                validation_targets,
                            )
                        for loss_name in HEAD_LOSS_NAMES:
                            validation_sums[loss_name] += float(
                                validation_losses[loss_name].detach()
                            )
                        validation_batches += 1
                ema.module.eval()
                row: dict[str, int | float] = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "elapsed_seconds": elapsed_before_start
                    + (time.monotonic() - training_started),
                    "learning_rate_min": min(
                        float(group["lr"]) for group in optimizer.param_groups
                    ),
                    "learning_rate_max": max(
                        float(group["lr"]) for group in optimizer.param_groups
                    ),
                }
                row.update(
                    {
                        f"train/{name}": total / max(1, epoch_loss_batches)
                        for name, total in epoch_loss_sums.items()
                    }
                )
                row.update(
                    {
                        f"val/{name}": total / max(1, validation_batches)
                        for name, total in validation_sums.items()
                    }
                )
                current_val_loss = float(row["val/loss_total"])
                improved = current_val_loss < best_val_loss
                if improved:
                    best_val_loss = current_val_loss
                metric_rows.append(row)
                assert metrics_path is not None
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_metrics = metrics_path.with_suffix(
                    metrics_path.suffix + ".tmp"
                )
                with temporary_metrics.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=list(row),
                    )
                    writer.writeheader()
                    writer.writerows(metric_rows)
                temporary_metrics.replace(metrics_path)
                if wandb_run is not None:
                    log_head_wandb_epoch(
                        wandb_run,
                        output_dir=output_dir,
                        row=row,
                    )
            if distributed:
                dist.barrier()
            if rank == 0:
                if (
                    (epoch + 1) % int(train_cfg.get("save_every", 1)) == 0
                    or should_stop
                ):
                    _save_checkpoint(
                        output_dir / f"epoch_{epoch + 1:03d}.pth",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        config=config,
                    )
                    prune_periodic_checkpoints(output_dir, keep=1)
                # Resume and selection checkpoints must remain current even
                # when periodic epoch snapshots are intentionally sparse.
                _save_checkpoint(
                    output_dir / "last.pth",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                )
                if improved:
                    _save_checkpoint(
                        output_dir / "best_val_loss.pth",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        config=config,
                    )
            if distributed:
                dist.barrier()
            if should_stop:
                break
            last_completed_epoch = epoch
        if rank == 0 and args.overfit_images is not None:
            box_movement = None
            if probe_batch is not None and initial_probe_boxes is not None:
                model.eval()
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=bool(train_cfg.get("amp", True))
                ):
                    final_probe_boxes = (
                        model(probe_batch, return_raw=True)["pred_boxes"].float().cpu()
                    )
                box_movement = float((final_probe_boxes - initial_probe_boxes).abs().mean())
            report = {
                "images": args.overfit_images,
                "optimizer_steps": global_step,
                "first": first_losses,
                "last": last_losses,
                "probe_pred_box_mean_abs_movement": box_movement,
                "decreased": {
                    key: bool(last_losses[key] < first_losses[key])
                    for key in ("loss_cls", "loss_bbox", "loss_giou", "loss_total")
                    if first_losses is not None and last_losses is not None
                },
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "overfit_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({"overfit": report}, ensure_ascii=False))
            required_decreases = OVERFIT_REQUIRED_DECREASES
            assertions = assess_overfit_run(
                first_losses,
                last_losses,
                optimizer_steps=global_step,
                box_movement=box_movement,
            )
            if not all(assertions.values()):
                raise RuntimeError(
                    "Co-DINO real-data overfit gate failed: "
                    f"{assertions}. See {output_dir / 'overfit_report.json'}"
                )
            success = {
                "status": "passed",
                "report": str(output_dir / "overfit_report.json"),
                "assertions": assertions,
                "required_loss_decreases": list(required_decreases),
                "optimizer_steps": global_step,
            }
            temporary = output_dir / ".overfit_success.json.tmp"
            temporary.write_text(
                json.dumps(success, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(overfit_success_path)
        elif (
            rank == 0
            and args.max_steps is None
            and last_completed_epoch + 1 >= epochs
        ):
            # A successful production run needs only its continuously updated
            # resume checkpoint and validation-selected best checkpoint.
            # Periodic snapshots are recovery aids while training is active.
            prune_periodic_checkpoints(output_dir, keep=0)
            completion = {
                "status": "complete",
                "epochs": int(epochs),
                "last_epoch": int(last_completed_epoch + 1),
                "global_step": int(global_step),
                "checkpoint": str(output_dir / "last.pth"),
                "best_checkpoint": str(output_dir / "best_val_loss.pth"),
                "best_val_loss": best_val_loss,
                "config": str(Path(args.config).expanduser().resolve()),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = output_dir / ".training_complete.json.tmp"
            temporary.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output_dir / "training_complete.json")
            print(json.dumps({"training_complete": completion}, ensure_ascii=False))
    finally:
        if rank == 0 and wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
