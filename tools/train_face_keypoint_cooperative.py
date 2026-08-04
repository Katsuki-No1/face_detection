from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.checkpoint_retention import prune_periodic_checkpoints


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.face_keypoint_cooperative_common import (  # noqa: E402
    ModelEMA,
    apply_training_stage_loss,
    autocast_context,
    build_loader,
    build_model,
    build_optimizer,
    close_distributed,
    configure_device_math,
    distributed_context,
    enforce_training_stage_modes,
    finite_loss_dict,
    initialize_shared_head_checkpoint,
    load_cooperative_config,
    initialize_stage_checkpoint,
    move_batch,
    resolve_path,
    resume_checkpoint,
    save_checkpoint,
    scheduler_lambda,
    seed_everything,
    set_dataset_epoch,
    training_stage_policy,
    unwrap_model,
    write_json,
)

WANDB_METRIC_SCHEMA_VERSION = 2
HISTORY_BASE_FIELDS = (
    "epoch",
    "global_step",
    "updates",
    "elapsed_seconds",
    "learning_rate_min",
    "learning_rate_max",
)
WANDB_COMPONENTS = {
    "loss_face_independent": "face_independent",
    "loss_keypoint_independent": "keypoint_independent",
    "loss_association": "association",
    "loss_face_refined": "face_refined",
    "loss_keypoint_refined": "keypoint_refined",
    "loss_visibility": "visibility_classification",
}
WANDB_METRIC_GUIDE = {
    "overview/train_total_loss": (
        "重み付き学習loss。低いほど良い。NMEそのものではない。"
    ),
    "overview/val_total_loss": (
        "重み付きvalidation loss。同じvalidation batch契約のepoch間だけで比較する。"
    ),
    "overview/train_association_accuracy": (
        "対応するpose queryを選べた学習時の割合。"
    ),
    "overview/val_association_accuracy": (
        "対応するpose queryを選べたvalidation時の割合。"
    ),
    "loss_components/*": (
        "各branchの重み付け前loss。independent=協調前、refined=協調後。"
    ),
    "optimization/*": "optimizer group間の学習率の最小値と最大値。",
    "timing/*": "1 epochの所要時間と累積学習時間。",
    "evaluation/val_face_*": (
        "validationの顔検出指標。設定済みscore thresholdとIoU=0.50で算出する。"
    ),
    "task_metrics": (
        "PCK・NME・検出AP・Occluded F1は学習lossではなく、別途evaluationで算出する。"
    ),
}


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
            "effective_batch_size_per_process": self.effective_batch_size,
        }


@dataclass(frozen=True)
class OptimizerStepResult:
    optimizer_step_succeeded: bool
    gradient_norm: float
    scale_before_step: float
    scale_after_step: float


class EpochLossAccumulator:
    """Accumulate scalar losses on-device and synchronize once per epoch."""

    def __init__(self) -> None:
        self.totals: dict[str, torch.Tensor] = {}
        self.batch_count = 0

    @staticmethod
    def _keys(output: Mapping[str, Any]) -> tuple[str, ...]:
        keys = tuple(
            sorted(
                key
                for key, value in output.items()
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 0
                    and (
                        key.startswith("loss")
                        or key == "association_accuracy"
                    )
                )
            )
        )
        if "loss_total" not in keys:
            raise RuntimeError(
                "Model output did not contain a scalar loss_total"
            )
        return keys

    def update(self, output: Mapping[str, Any]) -> None:
        keys = self._keys(output)
        for key in keys:
            value = output[key].detach().to(dtype=torch.float64)
            if key in self.totals:
                self.totals[key].add_(value)
            else:
                self.totals[key] = value.clone()
        self.batch_count += 1

    def mean(self, context: Mapping[str, Any]) -> dict[str, float]:
        if not self.totals or self.batch_count < 1:
            raise RuntimeError("No scalar losses were accumulated")

        keys = tuple(sorted(self.totals))
        if bool(context["distributed"]):
            rank_keys: list[tuple[str, ...] | None] = [
                None
            ] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rank_keys, keys)
            keys = tuple(
                sorted(
                    {
                        key
                        for rank_key_set in rank_keys
                        if rank_key_set is not None
                        for key in rank_key_set
                    }
                )
            )

        reference = self.totals["loss_total"]
        zero = reference.new_zeros(())
        totals = torch.stack(
            [self.totals.get(key, zero) for key in keys]
        )

        packed = torch.cat(
            (
                totals,
                totals.new_tensor((float(self.batch_count),)),
            )
        )
        if bool(context["distributed"]):
            torch.distributed.all_reduce(
                packed,
                op=torch.distributed.ReduceOp.SUM,
            )
        means = packed[:-1] / packed[-1].clamp_min(1.0)
        # This is the epoch's only device-to-host conversion for all losses.
        host_values = means.cpu().tolist()
        invalid = [
            key
            for key, value in zip(
                keys,
                host_values,
                strict=True,
            )
            if not math.isfinite(value)
        ]
        if invalid:
            raise FloatingPointError(
                f"Non-finite epoch scalar losses: {invalid}"
            )
        return dict(
            zip(
                keys,
                (float(value) for value in host_values),
                strict=True,
            )
        )


class FaceDetectionMetricAccumulator:
    """Accumulate fixed-threshold validation detection counts."""

    stages = ("independent", "refined")

    def __init__(
        self,
        *,
        score_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.counts: torch.Tensor | None = None

    @staticmethod
    def _iou(
        ground_truth: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        top_left = torch.maximum(
            ground_truth[:, None, :2],
            predictions[None, :, :2],
        )
        bottom_right = torch.minimum(
            ground_truth[:, None, 2:],
            predictions[None, :, 2:],
        )
        intersection = (
            (bottom_right - top_left)
            .clamp_min(0.0)
            .prod(dim=-1)
        )
        ground_truth_area = (
            (ground_truth[:, 2:] - ground_truth[:, :2])
            .clamp_min(0.0)
            .prod(dim=-1)
        )
        prediction_area = (
            (predictions[:, 2:] - predictions[:, :2])
            .clamp_min(0.0)
            .prod(dim=-1)
        )
        return intersection / (
            ground_truth_area[:, None]
            + prediction_area[None, :]
            - intersection
        ).clamp_min(1.0e-9)

    def _image_counts(
        self,
        ground_truth: torch.Tensor,
        prediction: Mapping[str, torch.Tensor],
    ) -> tuple[int, int, int]:
        keep = prediction["scores"] >= self.score_threshold
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        if boxes.shape[0] > 0:
            boxes = boxes[torch.argsort(scores, descending=True)]
        matched = [False] * int(ground_truth.shape[0])
        true_positive = 0
        if ground_truth.shape[0] > 0 and boxes.shape[0] > 0:
            overlaps = self._iou(
                ground_truth,
                boxes,
            ).detach().cpu().tolist()
            for prediction_index in range(int(boxes.shape[0])):
                candidates = (
                    (row[prediction_index], ground_truth_index)
                    for ground_truth_index, row in enumerate(overlaps)
                    if not matched[ground_truth_index]
                )
                best_iou, ground_truth_index = max(
                    candidates,
                    default=(-1.0, -1),
                )
                if best_iou >= self.iou_threshold:
                    matched[ground_truth_index] = True
                    true_positive += 1
        false_positive = int(boxes.shape[0]) - true_positive
        false_negative = int(ground_truth.shape[0]) - true_positive
        return true_positive, false_positive, false_negative

    def update(
        self,
        decoded: list[Mapping[str, Any]],
        targets: list[Mapping[str, Any]],
    ) -> None:
        if len(decoded) != len(targets):
            raise RuntimeError(
                "Decoded validation batch does not match target batch"
            )
        if self.counts is None:
            self.counts = targets[0]["boxes"].new_zeros(
                len(self.stages) * 3,
                dtype=torch.float64,
            )
        for prediction, target in zip(decoded, targets, strict=True):
            for stage_index, stage in enumerate(self.stages):
                values = self._image_counts(
                    target["boxes"],
                    prediction[f"{stage}_face"],
                )
                offset = stage_index * 3
                self.counts[offset : offset + 3] += self.counts.new_tensor(
                    values
                )

    def compute(
        self,
        context: Mapping[str, Any],
    ) -> dict[str, int | float]:
        if self.counts is None:
            raise RuntimeError("No validation detections were accumulated")
        counts = self.counts.clone()
        if bool(context["distributed"]):
            torch.distributed.all_reduce(
                counts,
                op=torch.distributed.ReduceOp.SUM,
            )
        result: dict[str, int | float] = {}
        for stage_index, stage in enumerate(self.stages):
            offset = stage_index * 3
            true_positive, false_positive, false_negative = (
                int(value)
                for value in counts[offset : offset + 3].cpu().tolist()
            )
            precision = true_positive / max(
                1,
                true_positive + false_positive,
            )
            recall = true_positive / max(
                1,
                true_positive + false_negative,
            )
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            prefix = f"face_{stage}"
            result.update(
                {
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_f1": f1,
                    f"{prefix}_tp": true_positive,
                    f"{prefix}_fp": false_positive,
                    f"{prefix}_fn": false_negative,
                }
            )
        return result


def configured_batch_plans(
    train_config: Mapping[str, Any],
) -> tuple[BatchPlan, ...]:
    configured = BatchPlan(
        batch_size=int(train_config.get("batch_size", 1)),
        gradient_accumulation_steps=int(
            train_config.get("gradient_accumulation_steps", 1)
        ),
    )
    fallback = train_config.get("batch_size_fallback", {})
    if not isinstance(fallback, Mapping) or not bool(
        fallback.get("enabled", False)
    ):
        return (configured,)

    raw_candidates = fallback.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(
            "train.batch_size_fallback.candidates must be a non-empty list"
        )
    plans = []
    for index, raw_plan in enumerate(raw_candidates):
        if not isinstance(raw_plan, Mapping):
            raise TypeError(
                "Each train.batch_size_fallback candidate must be a mapping; "
                f"candidate {index} is {type(raw_plan)}"
            )
        plan = BatchPlan(
            batch_size=int(raw_plan["batch_size"]),
            gradient_accumulation_steps=int(
                raw_plan["gradient_accumulation_steps"]
            ),
        )
        if plan.batch_size < 1 or plan.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Batch fallback candidate {index} must contain positive values"
            )
        plans.append(plan)

    if plans[0] != configured:
        raise ValueError(
            "train.batch_size and train.gradient_accumulation_steps must match "
            "the first batch_size_fallback candidate"
        )
    if len({plan.effective_batch_size for plan in plans}) != 1:
        raise ValueError(
            "All batch_size_fallback candidates must preserve the same "
            "effective batch size"
        )
    if len(set(plans)) != len(plans):
        raise ValueError("batch_size_fallback candidates must be unique")
    return tuple(plans)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if cuda_oom_type and isinstance(error, cuda_oom_type):
        return True
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "out of memory" in message
        and ("cuda" in message or "cublas" in message or "cudnn" in message)
    )


def release_cuda_memory(device: torch.device | None = None) -> None:
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
        oom_attempt: dict[str, Any] | None = None
        try:
            measurements = dict(probe(plan))
        except RuntimeError as error:
            if not is_cuda_out_of_memory(error):
                raise
            oom_attempt = {
                **plan.as_dict(),
                "status": "cuda_oom",
                "error": str(error),
            }
        if oom_attempt is not None:
            # Leave the exception handler before clearing the CUDA cache so
            # Python no longer retains the failed probe frame via its traceback.
            attempts.append(oom_attempt)
            release_cuda_memory(device)
            continue
        attempts.append(
            {
                **plan.as_dict(),
                "status": "selected",
                **measurements,
            }
        )
        release_cuda_memory(device)
        return plan, attempts
    raise RuntimeError(
        "CUDA OOM for every configured cooperative-DETR batch candidate: "
        + ", ".join(
            f"batch={plan.batch_size}/accum="
            f"{plan.gradient_accumulation_steps}"
            for plan in plans
        )
    )


def apply_batch_plan(config: dict[str, Any], plan: BatchPlan) -> None:
    config["train"]["batch_size"] = plan.batch_size
    config["train"][
        "gradient_accumulation_steps"
    ] = plan.gradient_accumulation_steps


def make_grad_scaler(
    device: torch.device,
    train_config: Mapping[str, Any],
) -> torch.amp.GradScaler:
    amp_dtype = str(train_config.get("amp_dtype", "bfloat16")).lower()
    return torch.amp.GradScaler(
        device.type,
        init_scale=float(
            train_config.get("grad_scaler_init_scale", 65536.0)
        ),
        enabled=(
            bool(train_config.get("amp", True))
            and device.type == "cuda"
            and amp_dtype == "float16"
        ),
    )


def step_optimizer_with_grad_scaler(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    max_grad_norm: float,
) -> OptimizerStepResult:
    """Apply one optimizer update while preserving AMP overflow recovery.

    ``GradScaler.unscale_`` records whether any gradient is non-finite. The
    following ``step`` skips that optimizer update and ``update`` backs off
    the scale. Raising immediately after clipping would bypass both actions
    and turn an expected transient FP16 overflow into a failed training run.
    """

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

    scaler.step(optimizer)
    scaler.update()
    scale_after_step = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)

    optimizer_step_succeeded = gradient_is_finite and (
        not scaler.is_enabled()
        or scale_after_step >= scale_before_step
    )
    return OptimizerStepResult(
        optimizer_step_succeeded=optimizer_step_succeeded,
        gradient_norm=gradient_norm,
        scale_before_step=scale_before_step,
        scale_after_step=scale_after_step,
    )


def probe_batch_plan(
    config: dict[str, Any],
    plan: BatchPlan,
    *,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    probe_config = copy.deepcopy(config)
    apply_batch_plan(probe_config, plan)
    # The probe measures GPU feasibility, not data-loader parallelism.
    probe_config["train"]["num_workers"] = 0
    seed_everything(
        int(probe_config["train"].get("seed", 42)) + rank,
        deterministic=bool(
            probe_config["train"].get("deterministic", True)
        ),
    )

    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    ema: ModelEMA | None = None
    scaler: torch.amp.GradScaler | None = None
    loader: torch.utils.data.DataLoader[Any] | None = None
    iterator: Any = None
    try:
        model = build_model(probe_config, device=device)
        optimizer, _ = build_optimizer(unwrap_model(model), probe_config)
        scaler = make_grad_scaler(device, probe_config["train"])
        ema = ModelEMA(
            model,
            decay=float(
                probe_config["train"].get("ema", {}).get("decay", 0.9998)
            ),
        )
        loader, _ = build_loader(
            probe_config,
            "train",
            training=True,
            distributed=False,
            rank=rank,
            world_size=1,
            batch_size=plan.batch_size,
        )
        iterator = iter(loader)
        model.train()
        enforce_training_stage_modes(model, probe_config)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(plan.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            images, targets = move_batch(batch, device)
            with autocast_context(device, probe_config["train"]):
                output = model(images, targets)
                apply_training_stage_loss(output, probe_config)
                loss = output["loss_total"] / plan.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss during cooperative-DETR batch probe"
                )
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(probe_config["train"].get("max_grad_norm", 1.0)),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "Non-finite gradient norm during cooperative-DETR batch probe"
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
        del iterator, loader, scaler, ema, optimizer, model
        release_cuda_memory(device)


def load_reusable_batch_plan(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FaceKeypointCooperativeDETR on full-frame face5 data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="Previous stage checkpoint; loads compatible tensors only.",
    )
    parser.add_argument(
        "--initialize-shared-head-from",
        type=Path,
        default=None,
        help=(
            "Initialize the integrated P3/P4/P5 + Co-DINO HEAD branch from "
            "the common standalone Head Co-DINO checkpoint. On resume, the "
            "cooperative checkpoint remains authoritative."
        ),
    )
    parser.add_argument(
        "--initialize-shared-dino-from-head-checkpoint",
        action="store_true",
        help=(
            "Also load the standalone HEAD checkpoint's fine-tuned DINO into "
            "the integrated shared backbone."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overfit-smoke", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=64)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable W&B for full training. When omitted, use "
            "the top-level wandb.enabled config value."
        ),
    )
    return parser.parse_args()


def metric_history_fields(
    rows: list[Mapping[str, Any]],
) -> list[str]:
    keys = {key for row in rows for key in row}
    ordered = [key for key in HISTORY_BASE_FIELDS if key in keys]
    ordered.extend(sorted(key for key in keys if key.startswith("train/")))
    ordered.extend(sorted(key for key in keys if key.startswith("val/")))
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def write_metric_history(
    path: Path,
    rows: list[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError("Metric history must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = metric_history_fields(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    fields = list(row)
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            fields = list(csv.DictReader(handle).fieldnames or ())
        if not fields:
            raise RuntimeError(f"Metric history has no header: {path}")
        missing = sorted(set(fields).difference(row))
        unexpected = sorted(set(row).difference(fields))
        detection_schema_extension = (
            not missing
            and bool(unexpected)
            and all(
                key.startswith("val/face_")
                for key in unexpected
            )
        )
        if detection_schema_extension:
            write_metric_history(
                path,
                [*read_metric_history(path), row],
            )
            return
        if missing or unexpected:
            raise RuntimeError(
                "Metric history schema changed: "
                f"missing={missing}, unexpected={unexpected}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def cooperative_wandb_run_id(
    config: Mapping[str, Any],
    *,
    entity: str | None,
    project: str,
) -> str:
    identity = {
        "entity": entity,
        "project": project,
        "experiment_name": str(config["experiment_name"]),
        "dataset_version": str(config["data"]["dataset_version"]),
        "seed": int(config["train"].get("seed", 42)),
        "metric_schema_version": int(
            config.get("wandb", {}).get(
                "metric_schema_version",
                WANDB_METRIC_SCHEMA_VERSION,
            )
        ),
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def read_metric_history(path: Path) -> list[dict[str, int | float]]:
    if not path.is_file():
        return []
    integer_fields = {"epoch", "global_step", "updates"}
    rows: list[dict[str, int | float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, int | float] = {}
            for key, value in raw.items():
                if value in {None, ""}:
                    continue
                row[key] = (
                    int(float(value))
                    if key in integer_fields
                    else float(value)
                )
            if "epoch" in row:
                rows.append(row)
    return rows


def wandb_metric_row(
    row: Mapping[str, Any],
    *,
    previous_elapsed_seconds: float = 0.0,
    detailed_losses: bool = False,
) -> dict[str, int | float]:
    elapsed_seconds = float(row["elapsed_seconds"])
    result: dict[str, int | float] = {
        "epoch": int(row["epoch"]),
        "progress/global_step": int(row["global_step"]),
        "progress/optimizer_updates": int(row["updates"]),
        "timing/epoch_minutes": max(
            0.0,
            elapsed_seconds - float(previous_elapsed_seconds),
        )
        / 60.0,
        "timing/elapsed_hours": elapsed_seconds / 3600.0,
        "optimization/learning_rate_min": float(
            row["learning_rate_min"]
        ),
        "optimization/learning_rate_max": float(
            row["learning_rate_max"]
        ),
    }
    for split in ("train", "val"):
        total_key = f"{split}/loss_total"
        association_key = f"{split}/association_accuracy"
        result[f"overview/{split}_total_loss"] = float(row[total_key])
        result[f"overview/{split}_association_accuracy"] = float(
            row[association_key]
        )
        for source, display in WANDB_COMPONENTS.items():
            result[f"loss_components/{split}_{display}"] = float(
                row[f"{split}/{source}"]
            )
    for stage in FaceDetectionMetricAccumulator.stages:
        for metric in ("precision", "recall", "f1", "tp", "fp", "fn"):
            source = f"val/face_{stage}_{metric}"
            if source in row:
                result[
                    f"evaluation/val_face_{stage}_{metric}"
                ] = float(row[source])
    if detailed_losses:
        for key, value in row.items():
            if key.startswith(("train/", "val/")):
                result[f"diagnostics/{key.replace('/', '_')}"] = float(value)
    return result


def verify_wandb_access(
    wandb_module: Any,
    *,
    entity: str | None,
    project: str,
    mode: str,
) -> dict[str, Any]:
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
            raise RuntimeError(
                "W&B API key is not configured; run `wandb login` first"
            )
        resolved_entity = entity or api.default_entity
        if not resolved_entity:
            raise RuntimeError("W&B entity could not be resolved")
        report["entity"] = resolved_entity
        report["project_accessible"] = bool(
            api.project(project, entity=resolved_entity)
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            "W&B authentication/project preflight failed "
            f"({type(error).__name__}): {error}"
        ) from error
    if not report["project_accessible"]:
        raise RuntimeError(
            f"W&B project is not accessible: {report['entity']}/{project}"
        )
    return report


def initialize_wandb(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> Any:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed"
        ) from error

    settings = config.get("wandb", {})
    if not isinstance(settings, Mapping):
        raise TypeError("wandb must be a mapping")
    entity_value = settings.get("entity")
    entity = str(entity_value) if entity_value else None
    project = str(settings.get("project", "face_detection"))
    mode = str(settings.get("mode", "online"))
    metric_schema_version = int(
        settings.get(
            "metric_schema_version",
            WANDB_METRIC_SCHEMA_VERSION,
        )
    )
    detailed_losses = bool(settings.get("log_detailed_losses", False))
    live_sync_seconds = float(settings.get("live_sync_seconds", 30.0))
    if live_sync_seconds <= 0.0:
        raise ValueError("wandb.live_sync_seconds must be positive")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            "wandb.mode must be online, offline, or disabled"
        )
    run_id = cooperative_wandb_run_id(
        config,
        entity=entity,
        project=project,
    )
    state_path = output_dir / "wandb_run.json"
    previous: dict[str, Any] = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    previous_id = previous.get("run_id")
    if previous_id not in {None, run_id}:
        previous_schema_version = int(
            previous.get("metric_schema_version", 1)
        )
        if previous_schema_version == metric_schema_version:
            raise RuntimeError(
                "Stored W&B run ID does not match the configured experiment: "
                f"{previous_id} != {run_id}"
            )
        write_json(
            output_dir
            / f"wandb_run.metrics_v{previous_schema_version}.json",
            previous,
        )
        previous = {
            "replaces_run_id": previous_id,
            "replaces_run_url": previous.get("url"),
        }
    preflight = verify_wandb_access(
        wandb,
        entity=entity,
        project=project,
        mode=mode,
    )
    write_json(output_dir / "wandb_preflight.json", preflight)
    local_dir = output_dir / "wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    configured_name = (
        settings.get("name")
        or f"{config['experiment_name']}-seed"
        f"{int(config['train'].get('seed', 42))}"
    )
    run = wandb.init(
        entity=entity,
        project=project,
        group=settings.get("group"),
        name=f"{configured_name}-metrics-v{metric_schema_version}",
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(local_dir),
        config=dict(config),
        tags=list(settings.get("tags", ())),
        job_type="training",
        notes=(
            "整理済みtraining metrics。overview=主要推移、"
            "loss_components=branch別診断、optimization=学習率、"
            "timing=所要時間。PCK・NME・AP・Occluded F1はevaluationで別途算出。"
        ),
        settings=wandb.Settings(
            x_file_stream_transmit_interval=live_sync_seconds,
        ),
    )
    run.define_metric("epoch")
    run.define_metric("overview/*", step_metric="epoch", summary="last")
    run.define_metric(
        "loss_components/*",
        step_metric="epoch",
        summary="last",
    )
    run.define_metric(
        "optimization/*",
        step_metric="epoch",
        summary="last",
    )
    run.define_metric("timing/*", step_metric="epoch", summary="last")
    run.define_metric("progress/*", step_metric="epoch", summary="last")
    run.define_metric(
        "evaluation/*",
        step_metric="epoch",
        summary="last",
    )
    if detailed_losses:
        run.define_metric(
            "diagnostics/*",
            step_metric="epoch",
            summary="last",
        )
    run.config.update(
        {
            "wandb_metric_schema_version": metric_schema_version,
            "wandb_metric_guide": WANDB_METRIC_GUIDE,
            "wandb_detailed_losses_logged": detailed_losses,
        },
        allow_val_change=True,
    )
    state = {
        **previous,
        "run_id": run_id,
        "entity": preflight.get("entity"),
        "project": project,
        "group": settings.get("group"),
        "name": run.name,
        "mode": mode,
        "url": run.url,
        "metric_schema_version": metric_schema_version,
        "detailed_losses": detailed_losses,
        "live_sync_seconds": live_sync_seconds,
        "last_logged_epoch": int(
            previous.get("last_logged_epoch", -1)
        ),
        "last_elapsed_seconds": float(
            previous.get("last_elapsed_seconds", 0.0)
        ),
    }
    write_json(state_path, state)
    for row in read_metric_history(output_dir / "metrics.csv"):
        epoch = int(row["epoch"])
        if epoch <= int(state["last_logged_epoch"]):
            continue
        run.log(
            wandb_metric_row(
                row,
                previous_elapsed_seconds=float(
                    state["last_elapsed_seconds"]
                ),
                detailed_losses=detailed_losses,
            ),
            step=epoch,
        )
        state["last_logged_epoch"] = epoch
        state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
        write_json(state_path, state)
    run_location = (
        run.url
        or f"{preflight.get('entity')}/{project}/{run_id}"
    )
    print(f"wandb_run: {run_location}")
    return run


def log_wandb_row(
    run: Any,
    *,
    output_dir: Path,
    row: Mapping[str, Any],
) -> None:
    epoch = int(row["epoch"])
    state_path = output_dir / "wandb_run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run.log(
        wandb_metric_row(
            row,
            previous_elapsed_seconds=float(
                state.get("last_elapsed_seconds", 0.0)
            ),
            detailed_losses=bool(
                state.get("detailed_losses", False)
            ),
        ),
        step=epoch,
    )
    state["last_logged_epoch"] = epoch
    state["last_elapsed_seconds"] = float(row["elapsed_seconds"])
    write_json(state_path, state)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    *,
    device: torch.device,
    config: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, int | float]:
    model.eval()
    accumulator = EpochLossAccumulator()
    evaluation = config.get("evaluation", {})
    detection_accumulator = FaceDetectionMetricAccumulator(
        score_threshold=float(
            evaluation.get(
                "face_score_threshold",
                config.get("inference", {}).get(
                    "face_score_threshold",
                    0.30,
                ),
            )
        ),
        iou_threshold=float(
            evaluation.get("matching_iou_threshold", 0.50)
        ),
    )
    for batch in loader:
        images, targets = move_batch(batch, device)
        with autocast_context(device, config["train"]):
            output = model(images, targets)
            apply_training_stage_loss(output, config)
        accumulator.update(output)
        decoded = unwrap_model(model).decode(
            output,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
        )
        detection_accumulator.update(decoded, targets)
    result = {
        f"val/{key}": value
        for key, value in accumulator.mean(context).items()
    }
    result.update(
        {
            f"val/{key}": value
            for key, value in detection_accumulator.compute(
                context
            ).items()
        }
    )
    return result


def overfit_branch_override(model: nn.Module) -> tuple[bool, bool]:
    """Use every refinement direction allowed by the model contract."""
    unwrapped = unwrap_model(model)
    return (
        (True, False)
        if unwrapped.strict_sequential_cascade is not None
        else (True, True)
    )


def run_overfit(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    *,
    device: torch.device,
    config: dict[str, Any],
    steps: int,
    output_dir: Path,
) -> int:
    success_path = output_dir / "overfit_success.json"
    success_path.unlink(missing_ok=True)
    unwrapped = unwrap_model(model)
    branch_override = overfit_branch_override(model)
    optimizer, optimizer_report = build_optimizer(unwrapped, config)
    scaler = make_grad_scaler(device, config["train"])
    accumulation = int(
        config["train"].get("gradient_accumulation_steps", 1)
    )
    model.train()
    enforce_training_stage_modes(model, config)
    iterator = iter(loader)
    history = []
    updates = 0
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, targets = move_batch(batch, device)
        with autocast_context(device, config["train"]):
            output = model(
                images,
                targets,
                branch_override=branch_override,
            )
            apply_training_stage_loss(output, config)
        loss = output["loss_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite overfit loss at step={step}: {loss}"
            )
        # Exercise the same FP16 backward/optimizer path as production.  The
        # previous smoke path skipped GradScaler and could silently apply
        # non-finite gradients even though the production path was valid.
        scaler.scale(loss / accumulation).backward()
        should_update = (
            (step + 1) % accumulation == 0 or step + 1 == steps
        )
        if should_update:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["train"].get("max_grad_norm", 1.0)),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite overfit gradient norm at step={step}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        values = finite_loss_dict(output)
        history.append({"step": step, **values})
    window = max(1, min(8, steps // 4))
    metric_keys = tuple(key for key in history[0] if key != "step")
    initial = {
        key: sum(float(row[key]) for row in history[:window]) / window
        for key in metric_keys
    }
    final = {
        key: sum(float(row[key]) for row in history[-window:]) / window
        for key in metric_keys
    }
    report = {
        "steps": steps,
        "optimizer_updates": updates,
        "gradient_accumulation_steps": accumulation,
        "branch_override": list(branch_override),
        "comparison_window": window,
        "initial": initial,
        "final": final,
        "ratios": {
            key: final[key] / max(1.0e-12, initial[key])
            for key in (
                "loss_face_independent",
                "loss_keypoint_independent",
                "loss_association",
                "loss_face_refined",
                "loss_keypoint_refined",
                "loss_total",
            )
        },
        "optimizer_groups": optimizer_report,
        "history": history,
    }
    write_json(output_dir / "overfit_report.json", report)
    stage_policy = training_stage_policy(config)
    required = (
        tuple(stage_policy["loss_weights"])
        if stage_policy is not None
        else (
            "loss_face_independent",
            "loss_keypoint_independent",
            "loss_association",
            "loss_face_refined",
            "loss_keypoint_refined",
        )
    )
    failed = [
        key
        for key in required
        if final[key] >= initial[key]
    ]
    if failed:
        raise RuntimeError(
            f"Overfit smoke did not reduce all required losses: {failed}. "
            f"See {output_dir / 'overfit_report.json'}"
        )
    write_json(
        success_path,
        {
            "status": "passed",
            "report": str(output_dir / "overfit_report.json"),
            "required_loss_decreases": list(required),
            "steps": steps,
            "optimizer_updates": updates,
            "gradient_accumulation_steps": accumulation,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    config = load_cooperative_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.overfit_smoke:
        config["train"]["num_workers"] = 0
        config["data"].setdefault("augmentation", {})["enabled"] = False
        args.max_samples = args.max_samples or 4
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else Path(config["output"]["root"])
    )
    context = distributed_context(args.device)
    device: torch.device = context["device"]
    configure_device_math(device, config["train"])
    if context["is_main"]:
        output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = output_dir / "checkpoints"
    last_checkpoint = checkpoints / "last.pt"
    resume_path = (
        resolve_path(args.resume_from)
        if args.resume_from is not None
        else last_checkpoint
    )
    plans = configured_batch_plans(config["train"])
    fallback_config = config["train"].get("batch_size_fallback", {})
    fallback_enabled = (
        isinstance(fallback_config, Mapping)
        and bool(fallback_config.get("enabled", False))
    )
    batch_report_path = output_dir / "batch_size_selection.json"
    if fallback_enabled and not args.overfit_smoke:
        selected = (
            load_reusable_batch_plan(batch_report_path, plans)
            if args.resume and resume_path.is_file()
            else None
        )
        if selected is not None:
            attempts: list[dict[str, Any]] = []
            selection_source = "reused_for_resume"
        elif device.type == "cuda":
            selected, attempts = select_batch_plan(
                plans,
                lambda plan: probe_batch_plan(
                    config,
                    plan,
                    device=device,
                    rank=int(context["rank"]),
                ),
                device=device,
            )
            selection_source = "measured"
        else:
            selected = plans[0]
            attempts = [
                {
                    **selected.as_dict(),
                    "status": "selected_without_cuda_probe",
                    "device": str(device),
                }
            ]
            selection_source = "non_cuda"

        if context["distributed"]:
            selected_index = torch.tensor(
                plans.index(selected),
                device=device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(
                selected_index,
                op=torch.distributed.ReduceOp.MAX,
            )
            distributed_selection = plans[int(selected_index.item())]
            if distributed_selection != selected:
                attempts.append(
                    {
                        **distributed_selection.as_dict(),
                        "status": "selected_by_distributed_fallback",
                    }
                )
            selected = distributed_selection
        apply_batch_plan(config, selected)
        if context["is_main"] and selection_source != "reused_for_resume":
            write_json(
                batch_report_path,
                {
                    "status": "selected",
                    "selection_source": selection_source,
                    "criterion": (
                        "one complete optimizer update including forward, "
                        "backward, AdamW state creation, gradient clipping, "
                        "optimizer step, and EMA update"
                    ),
                    "world_size": int(context["world_size"]),
                    "candidates": [plan.as_dict() for plan in plans],
                    "attempts": attempts,
                    "selected": selected.as_dict(),
                    "effective_global_batch_size": (
                        selected.effective_batch_size
                        * int(context["world_size"])
                    ),
                },
            )
        if context["is_main"]:
            print(
                "Cooperative DETR batch plan: "
                f"batch={selected.batch_size}, "
                "gradient_accumulation_steps="
                f"{selected.gradient_accumulation_steps}, "
                f"source={selection_source}"
            )

    # The sizing probe consumes RNG and constructs a throwaway model. Reset the
    # seed and build every training object from scratch so selection cannot
    # change the actual initialization or sample order.
    seed_everything(
        int(config["train"].get("seed", 42)) + int(context["rank"]),
        deterministic=bool(config["train"].get("deterministic", True)),
    )
    if context["is_main"]:
        write_json(output_dir / "config_resolved.json", config)
    model = build_model(config, device=device)
    if (
        args.initialize_shared_head_from is not None
        and not (args.resume and resume_path.is_file())
    ):
        initialize_shared_head_checkpoint(
            model,
            resolve_path(args.initialize_shared_head_from),
            prefer_ema=True,
            load_shared_dino=bool(
                args.initialize_shared_dino_from_head_checkpoint
            ),
        )
    if args.initialize_from is not None:
        if not (args.resume and resume_path.is_file()):
            stage_report = initialize_stage_checkpoint(
                model,
                resolve_path(args.initialize_from),
            )
            model.pretrained_load_info["stage_initialization"] = stage_report
    if context["is_main"]:
        write_json(
            output_dir / "pretrained_load_audit.json",
            model.pretrained_load_info,
        )
    if context["distributed"]:
        model = DistributedDataParallel(
            model,
            device_ids=[context["local_rank"]] if device.type == "cuda" else None,
            output_device=context["local_rank"] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    train_loader, train_sampler = build_loader(
        config,
        "train",
        training=True,
        distributed=bool(context["distributed"]),
        rank=int(context["rank"]),
        world_size=int(context["world_size"]),
        maximum_samples=args.max_samples,
    )
    val_loader, _ = build_loader(
        config,
        "val",
        training=False,
        distributed=bool(context["distributed"]),
        rank=int(context["rank"]),
        world_size=int(context["world_size"]),
        maximum_samples=args.max_samples,
    )
    if context["is_main"] and args.max_samples is not None:
        write_json(
            output_dir / "bounded_subset_audit.json",
            {
                split: getattr(
                    loader.dataset,
                    "representative_subset_report",
                    None,
                )
                for split, loader in (("train", train_loader), ("val", val_loader))
            },
        )
    if args.overfit_smoke:
        if context["distributed"]:
            raise ValueError("--overfit-smoke must run as a single process")
        return run_overfit(
            model,
            train_loader,
            device=device,
            config=config,
            steps=int(args.overfit_steps),
            output_dir=output_dir,
        )

    optimizer, optimizer_report = build_optimizer(unwrap_model(model), config)
    if context["is_main"]:
        write_json(output_dir / "optimizer_groups.json", optimizer_report)
    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 150))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        scheduler_lambda(
            epochs=epochs,
            warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
            flat_epochs=int(train_cfg.get("flat_epochs", 75)),
            minimum_ratio=float(train_cfg.get("minimum_lr_ratio", 0.01)),
        ),
    )
    scaler = make_grad_scaler(device, train_cfg)
    ema = ModelEMA(
        model,
        decay=float(train_cfg.get("ema", {}).get("decay", 0.9998)),
    )
    best_checkpoint = checkpoints / "best_val_loss.pt"
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if args.resume and resume_path.is_file():
        state = resume_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            current_config=config,
        )
        start_epoch = state["start_epoch"]
        global_step = state["step"]
        best_metric = state["best_metric"]
        if context["is_main"]:
            metric_contract = dict(
                state["validation_metric_contract"]
            )
            metric_contract.update(
                {
                    "resume_checkpoint": str(resume_path),
                    "selection_metric": "val/loss_total",
                    "reason": (
                        "validation_batch_size_changed"
                        if metric_contract["reset_best_metric"]
                        else "validation_batch_size_unchanged"
                    ),
                }
            )
            write_json(
                output_dir / "validation_metric_contract.json",
                metric_contract,
            )
            if metric_contract["reset_best_metric"]:
                print(
                    "Validation batch size changed across resume; "
                    "resetting best val/loss_total before checkpoint "
                    "selection."
                )

    wandb_settings = config.get("wandb", {})
    if not isinstance(wandb_settings, Mapping):
        raise TypeError("wandb must be a mapping")
    wandb_enabled = (
        bool(wandb_settings.get("enabled", False))
        if args.wandb is None
        else bool(args.wandb)
    )
    if context["is_main"] and wandb_enabled:
        # Keep W&B epoch logging transactional. Long-lived resumed runs can
        # buffer history until process exit, which makes a saved checkpoint
        # appear in local state before the corresponding server row exists.
        bootstrap_run = initialize_wandb(
            config,
            output_dir=output_dir,
        )
        bootstrap_run.finish(exit_code=0)
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 8))
    early = dict(train_cfg.get("early_stopping", {}))
    patience = 0
    previous_history = read_metric_history(output_dir / "metrics.csv")
    elapsed_offset = (
        float(previous_history[-1]["elapsed_seconds"])
        if previous_history
        else 0.0
    )
    started = time.perf_counter()
    max_consecutive_amp_skips = int(
        train_cfg.get("max_consecutive_amp_skips", 8)
    )
    if max_consecutive_amp_skips < 1:
        raise ValueError("train.max_consecutive_amp_skips must be positive")
    consecutive_amp_skips = 0
    for epoch in range(start_epoch, epochs):
        unwrap_model(model).set_epoch(epoch)
        set_dataset_epoch(train_loader.dataset, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        enforce_training_stage_modes(model, config)
        optimizer.zero_grad(set_to_none=True)
        train_accumulator = EpochLossAccumulator()
        updates = 0
        for batch_index, batch in enumerate(train_loader):
            images, targets = move_batch(batch, device)
            with autocast_context(device, train_cfg):
                output = model(images, targets)
                apply_training_stage_loss(output, config)
                loss = output["loss_total"] / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}: {loss}"
                )
            scaler.scale(loss).backward()
            should_update = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                step_result = step_optimizer_with_grad_scaler(
                    model,
                    optimizer,
                    scaler,
                    max_grad_norm=float(
                        train_cfg.get("max_grad_norm", 1.0)
                    ),
                )
                if not step_result.optimizer_step_succeeded:
                    consecutive_amp_skips += 1
                    if context["is_main"]:
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "batch": batch_index,
                                    "amp_step": (
                                        "skipped_non_finite_gradient"
                                    ),
                                    "gradient_norm": str(
                                        step_result.gradient_norm
                                    ),
                                    "scale_before_step": (
                                        step_result.scale_before_step
                                    ),
                                    "scale_after_step": (
                                        step_result.scale_after_step
                                    ),
                                    "consecutive_amp_skips": (
                                        consecutive_amp_skips
                                    ),
                                },
                                ensure_ascii=False,
                            )
                        )
                else:
                    consecutive_amp_skips = 0
                    ema.update(model)
                    global_step += 1
                    updates += 1
                if consecutive_amp_skips > max_consecutive_amp_skips:
                    raise FloatingPointError(
                        "Repeated non-finite gradient norms exceeded "
                        "train.max_consecutive_amp_skips at "
                        f"epoch={epoch}, batch={batch_index}: "
                        f"{consecutive_amp_skips} consecutive skips"
                    )
            train_accumulator.update(output)
        scheduler.step()
        training = train_accumulator.mean(context)
        with ema.apply(model):
            validation = validate(
                model,
                val_loader,
                device=device,
                config=config,
                context=context,
            )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "updates": updates,
            "elapsed_seconds": (
                elapsed_offset + time.perf_counter() - started
            ),
            "learning_rate_min": min(group["lr"] for group in optimizer.param_groups),
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            **{
                f"train/{key}": value
                for key, value in training.items()
            },
            **validation,
        }
        metric = float(validation["val/loss_total"])
        improved = metric < best_metric - float(early.get("min_delta", 1.0e-4))
        if improved:
            best_metric = metric
            patience = 0
        else:
            patience += 1
        if context["is_main"]:
            append_history(output_dir / "metrics.csv", row)
            save_checkpoint(
                last_checkpoint,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                config=config,
                metrics=row,
            )
            if improved:
                save_checkpoint(
                    best_checkpoint,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_metric,
                    config=config,
                    metrics=row,
                )
            if (epoch + 1) % int(train_cfg.get("checkpoint_every", 5)) == 0:
                save_checkpoint(
                    checkpoints / f"epoch_{epoch + 1:04d}.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_metric,
                    config=config,
                    metrics=row,
                )
                prune_periodic_checkpoints(checkpoints, keep=1)
            if wandb_enabled:
                # metrics.csv is the durable source. initialize_wandb replays
                # exactly the rows newer than wandb_run.json, and finish waits
                # for the server upload before training advances.
                epoch_run = initialize_wandb(
                    config,
                    output_dir=output_dir,
                )
                epoch_run.finish(exit_code=0)
        stop = (
            bool(early.get("enabled", True))
            and epoch + 1 >= int(early.get("minimum_epoch", 100))
            and patience >= int(early.get("patience", 25))
        )
        if context["distributed"]:
            stop_tensor = torch.tensor(int(stop), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
        if stop:
            break
    if context["is_main"]:
        prune_periodic_checkpoints(checkpoints, keep=0)
        write_json(
            output_dir / "training_complete.json",
            {
                "status": "complete",
                "best_metric": best_metric,
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "global_step": global_step,
                "elapsed_seconds": (
                    elapsed_offset + time.perf_counter() - started
                ),
            },
        )
    close_distributed(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
