#!/usr/bin/env python3
"""Launch multi-scale Query Refinement with its validation-selected batch plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_multiscale_query_refinement.yaml"
)
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "outputs/training_gates/deimv2_l_multiscale_query_refinement_seed42/gpu_step/batch_plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.batch_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "passed":
        raise RuntimeError(f"Batch plan has not passed: {plan_path}")
    if plan.get("candidates") != [4, 8, 16]:
        raise RuntimeError("Batch plan did not test the required candidates 4, 8, and 16")
    selected_batch = int(plan["selected_batch_size"])
    selected_accumulation = int(plan["selected_gradient_accumulation_steps"])
    safe_batches = {
        int(attempt["batch_size"])
        for attempt in plan.get("attempts", ())
        if attempt.get("status") == "safe"
    }
    if selected_batch not in safe_batches:
        raise RuntimeError("Selected microbatch is not recorded as safe")

    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        str(PROJECT_ROOT / "keypoint_comparison1/scripts/run_landmark_experiments.py"),
        "--config",
        str(args.config.resolve()),
        "--stage",
        "train",
        "--variants",
        "query_refinement",
        "--device",
        str(args.device),
        "--batch-size",
        str(selected_batch),
        "--gradient-accumulation-steps",
        str(selected_accumulation),
        "--batch-plan",
        str(plan_path),
        "--resume" if args.resume else "--no-resume",
        "--wandb" if args.wandb else "--no-wandb",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
