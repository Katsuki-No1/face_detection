#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison2.config import (  # noqa: E402
    MODEL_NAMES,
    load_config,
    resolved_config_sha256,
)
from keypoint_comparison2.evaluation import evaluate_model  # noqa: E402
from keypoint_comparison2.preflight import run_preflight  # noqa: E402
from keypoint_comparison2.training import train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production runner for face keypoint comparison 2")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "keypoint_comparison2/configs/production.yaml",
    )
    parser.add_argument("--stage", choices=("preflight", "train", "evaluate", "all"), default="preflight")
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Evaluation split; training and preflight ignore this option",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        help="Override the configured DINO backbone freeze duration for bounded training gates.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--forward-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one real forward/backward per selected model after asset/data checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, project_root=REPO_ROOT)
    original_config_sha256 = resolved_config_sha256(config.raw)
    if args.output_root is not None:
        config.raw["output"]["root"] = str(args.output_root.resolve())
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config.raw["train"]["epochs"] = int(args.epochs)
        config.raw["train"]["early_stopping"]["enabled"] = False
    if (args.batch_size is None) != (args.gradient_accumulation_steps is None):
        raise ValueError(
            "--batch-size and --gradient-accumulation-steps must be provided together"
        )
    if args.batch_plan is not None:
        plan_path = args.batch_plan.expanduser().resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("status") != "passed":
            raise RuntimeError("Batch plan has not passed")
        if plan.get("resolved_config_sha256") != original_config_sha256:
            raise RuntimeError(
                "Batch plan was produced for a different resolved config"
            )
        selected = dict(plan["selected"])
        batch_size = int(selected["microbatch"])
        accumulation = int(selected["accumulation_steps"])
        fallback_plans = [
            {
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
            },
            *[
                {
                    "batch_size": int(value["microbatch"]),
                    "gradient_accumulation_steps": int(
                        value["accumulation_steps"]
                    ),
                }
                for value in plan.get("fallback_plans", [])
            ],
        ]
        joint = config.raw["models"]["deimv2_joint"]
        joint["batch_size"] = batch_size
        joint["gradient_accumulation_steps"] = accumulation
        joint["batch_size_fallback"] = {
            "enabled": True,
            "candidates": fallback_plans,
        }
        config.raw["train"]["resolved_batch_plan"] = plan
        config.raw["train"]["resolved_batch_plan_path"] = str(plan_path)
    if args.batch_size is not None:
        if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
            raise ValueError("Batch size and accumulation must be positive")
        joint = config.raw["models"]["deimv2_joint"]
        joint["batch_size"] = int(args.batch_size)
        joint["gradient_accumulation_steps"] = int(
            args.gradient_accumulation_steps
        )
        joint["batch_size_fallback"] = {"enabled": False}
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config.raw["train"]["num_workers"] = int(args.num_workers)
    if args.freeze_backbone_epochs is not None:
        if args.freeze_backbone_epochs < 0:
            raise ValueError("--freeze-backbone-epochs must be non-negative")
        config.raw["train"]["freeze_backbone_epochs"] = int(args.freeze_backbone_epochs)
    if args.wandb is not None:
        config.raw["train"]["wandb"]["enabled"] = bool(args.wandb)
    selected = list(MODEL_NAMES) if "all" in args.models else args.models
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.raw["train"].get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(config.raw["train"].get("tf32", True))
        torch.set_float32_matmul_precision("high")
    if args.stage in {"preflight", "all"}:
        report = run_preflight(config, models=selected, device=device, forward=args.forward_preflight)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage in {"train", "all"}:
        for model_name in selected:
            checkpoint = train_model(
                config,
                model_name,
                device=device,
                resume=args.resume,
                maximum_train_samples=args.max_train_samples,
                maximum_val_samples=args.max_val_samples,
            )
            print(f"{model_name}: {checkpoint}")
    if args.stage in {"evaluate", "all"}:
        results = {}
        for model_name in selected:
            results[model_name] = evaluate_model(
                config,
                model_name,
                device=device,
                split=args.split,
            )
        comparison_path = config.output_root / f"comparison_{args.split}.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
