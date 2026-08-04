#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison1.landmark_detection.config import load_landmark_config  # noqa: E402
from keypoint_comparison1.landmark_detection.model import VARIANTS  # noqa: E402
from keypoint_comparison1.landmark_detection.pipeline import (  # noqa: E402
    compare_existing,
    evaluate_conditions,
    preflight,
    train_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the DINOv3 face-landmark A-G experiment matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "evaluate", "compare", "all"),
        default="preflight",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(*VARIANTS, "all"),
        default=["all"],
        help=(
            "Training heads. A/C share vitpose; B/D share query; E uses query_refinement; "
            "F/G share the parameter-matched vitpose_376."
        ),
    )
    parser.add_argument(
        "--evaluation-set",
        action="append",
        dest="evaluation_sets",
        help="Configured evaluation-set name. Repeat for multiple sets; defaults to evaluation.default_sets.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume each head from checkpoints/last.pt when available.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override wandb.enabled for training.",
    )
    parser.add_argument("--force-detector-cache", action="store_true")
    parser.add_argument("--skip-detector-preflight", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Gate-only output override; production uses outputs.root from the config.",
    )
    parser.add_argument(
        "--initialize-from-deim-checkpoint",
        type=Path,
        help="Gate-only DINO initialization override.",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        help="Gate-only detector checkpoint override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Bounded gate-training override applied after production config validation.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return torch.device(value)


def _configure_logging(output_root: Path) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> int:
    args = parse_args()
    config = load_landmark_config(args.config, project_root=REPO_ROOT)
    if args.output_root is not None:
        config.raw["outputs"]["root"] = str(args.output_root.resolve())
    if args.initialize_from_deim_checkpoint is not None:
        config.raw["model"]["backbone"]["initialize_from_deim_checkpoint"] = str(
            args.initialize_from_deim_checkpoint.resolve()
        )
    if args.detector_checkpoint is not None:
        config.raw["detector"]["checkpoint"] = str(
            args.detector_checkpoint.resolve()
        )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config.raw["train"]["epochs"] = int(args.epochs)
    for key, value in (
        ("max_train_samples", args.max_train_samples),
        ("max_val_samples", args.max_val_samples),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
            config.raw["train"][key] = int(value)
    _configure_logging(config.output_root)
    device = _device(args.device)
    train_cfg = config.raw["train"]
    if bool(train_cfg.get("tf32", True)) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    variants = list(VARIANTS) if "all" in args.variants else args.variants
    evaluation_sets = args.evaluation_sets or list(config.raw["evaluation"]["default_sets"])
    wandb_enabled = bool(config.raw.get("wandb", {}).get("enabled", False)) if args.wandb is None else args.wandb

    if args.stage in {"preflight", "all"}:
        preflight(
            config,
            device=device,
            check_detector=not args.skip_detector_preflight,
            require_wandb=args.stage == "all" and wandb_enabled,
        )
    if args.stage in {"train", "all"}:
        # A training-only call still verifies W&B before starting the first epoch.
        if args.stage == "train":
            preflight(
                config,
                device=device,
                check_detector=False,
                require_wandb=wandb_enabled,
            )
        train_variants(
            config,
            variants=variants,
            device=device,
            resume=args.resume,
            wandb_enabled=wandb_enabled,
        )
    if args.stage in {"evaluate", "all"}:
        evaluate_conditions(
            config,
            evaluation_sets=evaluation_sets,
            device=device,
            force_detector_cache=args.force_detector_cache,
            variants=variants,
        )
    if args.stage == "compare":
        compare_existing(config, evaluation_sets=evaluation_sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
