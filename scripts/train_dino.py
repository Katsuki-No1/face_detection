from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config
from src.data import AugmentedFaceKeypointDataset, FaceKeypointDataset, SyntheticFaceKeypointDataset, collate_face_keypoint_batch
from src.models import build_detector_from_config


LOSS_LOG_KEYS = (
    "loss_total",
    "loss_cls",
    "loss_center",
    "loss_ellipse",
    "loss_center_offset",
    "loss_bbox",
    "loss_bbox_l1",
    "loss_giou",
    "loss_kpt",
    "match_quality",
    "num_pos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run config-driven DINO detectors.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Run one lightweight batch through loss computation only.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic smoke data for dry-run.")
    parser.add_argument("--image-size", type=int, default=None, help="Override data.image_size for smoke runs.")
    parser.add_argument("--overfit-smoke", action="store_true", help="Train on a tiny subset to confirm the model can overfit.")
    parser.add_argument("--max-samples", type=int, default=16, help="Subset size for --overfit-smoke.")
    parser.add_argument("--steps", type=int, default=40, help="Optimization steps for --overfit-smoke.")
    parser.add_argument("--lr", type=float, default=3.0e-3, help="Learning rate for --overfit-smoke.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for smoke modes.")
    parser.add_argument("--tiny-model", action="store_true", help="Use narrower channels for CPU overfit smoke.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs for full training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a checkpoint.")
    parser.add_argument("--save-every", type=int, default=10)
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
    return parser.parse_args()


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
    return {
        "cuda": cuda_enabled,
        "tf32": tf32_enabled if cuda_enabled else False,
        "cudnn_benchmark": cudnn_benchmark if cuda_enabled else False,
        "matmul_precision": matmul_precision if cuda_enabled else "",
    }


def _configured_image_size(config: dict[str, Any]) -> int | tuple[int, int]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    return data_cfg.get("image_size") or train_cfg.get("imgsz", 896)


def build_dataset(config: dict[str, Any], *, force_synthetic: bool = False):
    data_cfg = config.get("data", {})
    image_size = _configured_image_size(config)
    data_root = data_cfg.get("data_root")
    annotation_path = data_cfg.get("annotation_path")
    synthetic_if_missing = bool(data_cfg.get("synthetic_if_missing", True))
    missing_real_data = not data_root and not annotation_path
    if force_synthetic or (missing_real_data and synthetic_if_missing):
        dataset = SyntheticFaceKeypointDataset(length=4, image_size=int(image_size if isinstance(image_size, int) else image_size[0]))
    else:
        dataset = FaceKeypointDataset(
            data_root=data_root,
            split=str(data_cfg.get("split", "train")),
            annotation_path=annotation_path,
            annotation_format=str(data_cfg.get("format", "auto")),
            image_root=data_cfg.get("image_root"),
            metadata_path=data_cfg.get("metadata_path"),
            image_size=image_size,
            include_negative=bool(data_cfg.get("include_negative", True)),
            category_ids=data_cfg.get("category_ids"),
        )
    aug_cfg = config.get("augmentation", {})
    if str(data_cfg.get("split", "train")) == "train" and bool(aug_cfg.get("enabled", False)):
        dataset = AugmentedFaceKeypointDataset(dataset, aug_cfg, seed=int(config.get("train", {}).get("seed", 42)))
    return dataset


def move_targets_to_device(targets: list[dict[str, Any]], device: str, *, non_blocking: bool = False) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        updated = dict(target)
        for key in ("boxes", "ellipses", "labels", "keypoints", "keypoint_mask"):
            value = updated.get(key)
            if isinstance(value, torch.Tensor):
                updated[key] = value.to(device, non_blocking=non_blocking)
        moved.append(updated)
    return moved


def apply_tiny_model_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    backbone = config.setdefault("model", {}).setdefault("backbone", {})
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
    config["model"].setdefault("adapter", {})["out_channels"] = 64
    config["model"].setdefault("adapter", {})["hidden_dim"] = 64
    config["model"].setdefault("neck", {})["out_channels"] = 64
    config["model"].setdefault("encoder", {})["hidden_dim"] = 64
    config["model"].setdefault("encoder", {})["num_heads"] = 4
    config["model"].setdefault("decoder", {})["hidden_dim"] = 64
    config["model"].setdefault("decoder", {})["num_heads"] = 4
    config["model"].setdefault("decoder", {})["num_layers"] = min(1, int(config["model"].setdefault("decoder", {}).get("num_layers", 1)))
    config["model"].setdefault("decoder", {})["num_queries"] = min(16, int(config["model"].setdefault("decoder", {}).get("num_queries", 16)))
    head = config["model"].setdefault("head", {})
    head["head_channels"] = 64
    head["num_convs"] = 1
    return config


def trainable_parameter_groups(model: torch.nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    train_cfg = config.get("train", {})
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

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return max(1e-3, float(epoch_number) / float(warmup_epochs))
        if scheduler_name in {"cosine", "cos", "cosine_annealing"}:
            denom = max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, float(epoch_number - warmup_epochs) / float(denom)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": checkpoint_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


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
                    torch.save(features[batch_index].contiguous(), path)
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


def make_grad_scaler(*, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
    allow_dummy_backbone: bool,
) -> int:
    validate_full_training_backbone(config, allow_dummy_backbone=allow_dummy_backbone)
    seed = int(config.get("train", {}).get("seed", 42))
    set_reproducible_seed(seed)
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; set data.synthetic_if_missing=true or provide local data.")
    train_cfg = config.get("train", {})
    batch_size = min(int(train_cfg.get("batch_size", 4)), len(dataset))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 100))
    generator = torch.Generator()
    generator.manual_seed(seed)

    model = build_detector_from_config(config).to(device)
    model.train()
    compile_meta = compile_trainable_modules(model, config)
    parameter_groups = trainable_parameter_groups(model, config)
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found.")
    optimizer, optimizer_meta = build_optimizer(parameter_groups, config, device=device)
    scheduler = build_epoch_scheduler(optimizer, config, epochs=epochs)
    scaler = make_grad_scaler(enabled=use_amp and str(device).startswith("cuda"))

    start_epoch = 1
    total_step = 0
    best_loss = float("inf")
    previous_elapsed = 0.0
    metrics_path = output_dir / "metrics.csv"
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        total_step = int(checkpoint.get("step", 0))
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if rows:
                previous_elapsed = float(rows[-1].get("elapsed_seconds") or 0.0)
                best_loss = min(float(row["loss_total"]) for row in rows if row.get("loss_total"))
        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step(completed_epoch)
        print(f"resume_from: {resume_from}")
        print(f"resume_epoch: {start_epoch}/{epochs}")
        if start_epoch > epochs:
            print(f"training_already_complete: {output_dir}")
            return 0

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
    frozen_backbone = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    run_meta = {
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "requested_batch_size": int(train_cfg.get("batch_size", 4)),
        "epochs": epochs,
        "device": device,
        "amp": bool(scaler.is_enabled()),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if str(device).startswith("cuda") else False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if str(device).startswith("cuda") else False,
        "matmul_precision": str(train_cfg.get("matmul_precision", "")) if str(device).startswith("cuda") else "",
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "persistent_workers": bool(train_cfg.get("persistent_workers", True)) if num_workers > 0 else False,
        "prefetch_factor": int(train_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
        "optimizer": optimizer_meta,
        "torch_compile": compile_meta,
        "feature_cache": feature_cache_meta,
        "progressive_resizing": dict(train_cfg.get("progressive_resizing", {"enabled": False})),
        "seed": seed,
        "scheduler": str(train_cfg.get("scheduler", "none")),
        "warmup_epochs": int(train_cfg.get("warmup_epochs", 0)),
        "frozen_backbone": frozen_backbone,
        "trainable_parameters": count_parameters(trainable_params),
        "parameter_groups": [
            {"name": group.get("name", ""), "lr": group["lr"], "weight_decay": group["weight_decay"], "parameters": count_parameters(group["params"])}
            for group in parameter_groups
        ],
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    fieldnames = ["epoch", *LOSS_LOG_KEYS, "lr", "elapsed_seconds"]
    write_header = not (resume_from is not None and metrics_path.exists())
    mode = "a" if resume_from is not None and metrics_path.exists() else "w"
    loader = None
    loader_image_size: tuple[int, int] | None = None
    with metrics_path.open(mode, encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch, epochs)
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
            sums = {key: 0.0 for key in LOSS_LOG_KEYS}
            batches = 0
            optimizer_steps_this_epoch = 0
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
                optimizer.zero_grad(set_to_none=True)
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
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(train_cfg.get("max_grad_norm", 10.0)))
                scale_before_step = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(optimizer)
                scaler.update()
                scale_after_step = scaler.get_scale() if scaler.is_enabled() else 1.0
                optimizer_stepped = (not scaler.is_enabled()) or scale_after_step >= scale_before_step

                if optimizer_stepped:
                    total_step += 1
                    optimizer_steps_this_epoch += 1
                batches += 1
                for key in sums:
                    value = loss_dict.get(key)
                    if value is not None:
                        sums[key] += float(value.detach().cpu())

            if batches == 0:
                raise RuntimeError("No batches were processed.")
            row = {key: sums[key] / batches for key in sums}
            row["epoch"] = epoch
            row["lr"] = float(optimizer.param_groups[0]["lr"])
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
            save_checkpoint(output_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer, epoch=epoch, step=total_step, config=config, metrics=metrics)
            if row["loss_total"] < best_loss:
                best_loss = row["loss_total"]
                save_checkpoint(output_dir / "checkpoints" / "best_loss.pt", model=model, optimizer=optimizer, epoch=epoch, step=total_step, config=config, metrics=metrics)
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=total_step,
                    config=config,
                    metrics=metrics,
                )
            if scheduler is not None and optimizer_steps_this_epoch > 0:
                scheduler.step()
    print(f"training_complete: {output_dir}")
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
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    indices = positive_subset_indices(dataset, max_samples=max_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    batch_size = min(int(config.get("train", {}).get("batch_size", 4)), len(subset))
    train_cfg = config.get("train", {})
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(**dataloader_kwargs(
        dataset=subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        train_cfg=train_cfg,
    ))
    model = build_detector_from_config(config).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    losses: list[float] = []
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
        loss = loss_dict["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            print(f"step {step + 1}/{steps}: loss_total={losses[-1]:.6f}")

    window = min(5, len(losses))
    initial_avg = sum(losses[:window]) / window
    final_avg = sum(losses[-window:]) / window
    loss_drop_ratio = final_avg / max(initial_avg, 1e-12)
    overfit_ok = final_avg < initial_avg
    print("overfit_smoke: done")
    print(f"samples: {len(subset)}")
    print(f"batch_size: {batch_size}")
    print(f"steps: {steps}")
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
        config.setdefault("data", {})["image_size"] = args.image_size
        config.setdefault("train", {})["imgsz"] = args.image_size
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
    return train(
        config,
        device=str(device),
        force_synthetic=args.synthetic,
        output_dir=make_output_dir(config, args.output_dir),
        resume_from=args.resume_from,
        save_every=args.save_every,
        use_amp=resolve_amp(config, args.amp),
        max_train_batches=args.max_train_batches,
        allow_dummy_backbone=bool(args.allow_dummy_backbone),
    )


if __name__ == "__main__":
    raise SystemExit(main())
