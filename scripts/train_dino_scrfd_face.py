from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import FaceKeypointDataset, SyntheticFaceKeypointDataset, collate_face_keypoint_batch
from src.models import build_dino_scrfd_face_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run the DINO-SCRFD face/keypoint detector.")
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
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for debug runs.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
        return SyntheticFaceKeypointDataset(length=2, image_size=int(image_size if isinstance(image_size, int) else image_size[0]))
    return FaceKeypointDataset(
        data_root=data_root,
        split=str(data_cfg.get("split", "train")),
        annotation_path=annotation_path,
        annotation_format=str(data_cfg.get("format", "auto")),
        image_root=data_cfg.get("image_root"),
        metadata_path=data_cfg.get("metadata_path"),
        image_size=image_size,
        include_negative=bool(data_cfg.get("include_negative", True)),
    )


def move_targets_to_device(targets: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    return [
        {
            **target,
            "boxes": target["boxes"].to(device),
            "labels": target["labels"].to(device),
            "keypoints": target["keypoints"].to(device),
            "keypoint_mask": target["keypoint_mask"].to(device),
        }
        for target in targets
    ]


def apply_tiny_model_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    backbone = config.setdefault("model", {}).setdefault("backbone", {})
    backbone["embed_dim"] = 64
    backbone["dummy"] = True
    backbone["pretrained"] = False
    config["model"].setdefault("adapter", {})["out_channels"] = 64
    config["model"].setdefault("neck", {})["out_channels"] = 64
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


def count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


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
    base = Path(train_cfg.get("output_root", "runs/dino_scrfd_face"))
    name = train_cfg.get("name") or datetime.now().strftime("dino_scrfd_%Y%m%d_%H%M%S")
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
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def dry_run(config: dict[str, Any], *, device: str, force_synthetic: bool) -> int:
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; set data.synthetic_if_missing=true or provide local data.")
    batch_size = min(int(config.get("train", {}).get("batch_size", 2)), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_face_keypoint_batch,
    )
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = move_targets_to_device(targets, device)

    model = build_dino_scrfd_face_detector(config).to(device)
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
    save_every: int,
    use_amp: bool,
    max_train_batches: int | None,
) -> int:
    dataset = build_dataset(config, force_synthetic=force_synthetic)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; set data.synthetic_if_missing=true or provide local data.")
    train_cfg = config.get("train", {})
    batch_size = min(int(train_cfg.get("batch_size", 4)), len(dataset))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_face_keypoint_batch,
    )

    model = build_dino_scrfd_face_detector(config).to(device)
    model.train()
    parameter_groups = trainable_parameter_groups(model, config)
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found.")
    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and str(device).startswith("cuda"))
    epochs = int(train_cfg.get("epochs", 100))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainable_params = [parameter for group in parameter_groups for parameter in group["params"]]
    frozen_backbone = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    run_meta = {
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "device": device,
        "amp": bool(scaler.is_enabled()),
        "frozen_backbone": frozen_backbone,
        "trainable_parameters": count_parameters(trainable_params),
        "parameter_groups": [
            {"name": group.get("name", ""), "lr": group["lr"], "weight_decay": group["weight_decay"], "parameters": count_parameters(group["params"])}
            for group in parameter_groups
        ],
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_path = output_dir / "metrics.csv"
    total_step = 0
    best_loss = float("inf")
    started = time.monotonic()
    fieldnames = ["epoch", "loss_total", "loss_cls", "loss_bbox", "loss_kpt", "num_pos", "lr", "elapsed_seconds"]
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            sums = {key: 0.0 for key in ["loss_total", "loss_cls", "loss_bbox", "loss_kpt", "num_pos"]}
            batches = 0
            for batch_index, (images, targets) in enumerate(loader, start=1):
                if max_train_batches is not None and batch_index > max_train_batches:
                    break
                images = images.to(device, non_blocking=True)
                targets = move_targets_to_device(targets, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    loss_dict = model(images, targets)
                    assert isinstance(loss_dict, dict)
                    loss = loss_dict["loss_total"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(train_cfg.get("max_grad_norm", 10.0)))
                scaler.step(optimizer)
                scaler.update()

                total_step += 1
                batches += 1
                for key in sums:
                    sums[key] += float(loss_dict[key].detach().cpu())

            if batches == 0:
                raise RuntimeError("No batches were processed.")
            row = {key: sums[key] / batches for key in sums}
            row["epoch"] = epoch
            row["lr"] = float(parameter_groups[0]["lr"])
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            writer.writerow({key: row[key] for key in fieldnames})
            metrics_file.flush()

            print(
                f"epoch {epoch}/{epochs}: "
                f"loss_total={row['loss_total']:.6f} "
                f"loss_cls={row['loss_cls']:.6f} "
                f"loss_bbox={row['loss_bbox']:.6f} "
                f"loss_kpt={row['loss_kpt']:.6f} "
                f"num_pos={row['num_pos']:.2f}"
            )

            metrics = {key: float(row[key]) for key in ["loss_total", "loss_cls", "loss_bbox", "loss_kpt", "num_pos"]}
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
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_face_keypoint_batch,
    )
    model = build_dino_scrfd_face_detector(config).to(device)
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
        targets = move_targets_to_device(targets, device)
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
    if args.tiny_model:
        config = apply_tiny_model_overrides(config)
    device = args.device or config.get("train", {}).get("device", "cpu")
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
        save_every=args.save_every,
        use_amp=bool(args.amp),
        max_train_batches=args.max_train_batches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
