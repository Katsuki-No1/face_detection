#!/usr/bin/env python3
"""Run DINOv3+DEIMv2 production with a validated runtime fallback plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wandb_utils import write_json_atomic  # noqa: E402
from tools.dinov3_deimv2_common import (  # noqa: E402
    load_config,
    resolved_config_sha256,
)
from scripts.wider_pretrained_artifact import promote_wider_checkpoint  # noqa: E402


def _stream(command: list[str]) -> tuple[int, str]:
    tail: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 400:
            tail.pop(0)
    return int(process.wait()), "".join(tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--promote-checkpoint-to", type=Path)
    parser.add_argument("--promotion-dataset-manifest", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    plan_path = args.batch_plan.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    if plan.get("status") != "passed":
        raise RuntimeError("Batch plan has not passed")
    if plan.get("resolved_config_sha256") != resolved_config_sha256(config):
        raise RuntimeError("Batch plan belongs to a different resolved config")
    selected = dict(plan["selected"])
    plans = [
        {
            "microbatch": int(selected["microbatch"]),
            "accumulation_steps": int(selected["accumulation_steps"]),
        },
        *[
            {
                "microbatch": int(value["microbatch"]),
                "accumulation_steps": int(value["accumulation_steps"]),
            }
            for value in plan.get("fallback_plans", [])
        ],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for index, batch in enumerate(plans):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools/train_dinov3_deimv2.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(args.data_root.expanduser().resolve()),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--batch-plan",
            str(plan_path),
            "--batch-size",
            str(batch["microbatch"]),
            "--gradient-accumulation-steps",
            str(batch["accumulation_steps"]),
            "--wandb",
            "--wandb-entity",
            args.wandb_entity,
            "--wandb-project",
            args.wandb_project,
            "--wandb-run-name",
            args.wandb_run_name,
            "--wandb-group",
            args.wandb_group,
            "--wandb-mode",
            "online",
            "--dataset-version",
            args.dataset_version,
        ]
        last = output_dir / "checkpoints/last.pt"
        resumed = last.is_file()
        if resumed:
            command.extend(("--resume-from", str(last)))
        exit_code, output_tail = _stream(command)
        oom = "out of memory" in output_tail.lower()
        attempts.append(
            {
                **batch,
                "exit_code": exit_code,
                "cuda_oom": oom,
                "resumed": resumed,
            }
        )
        write_json_atomic(
            output_dir / "runtime_batch_attempts.json",
            {"attempts": attempts, "batch_plan": str(plan_path)},
        )
        if exit_code == 0:
            if args.promote_checkpoint_to is not None:
                if args.promotion_dataset_manifest is None:
                    raise RuntimeError(
                        "--promotion-dataset-manifest is required with checkpoint promotion"
                    )
                payload = promote_wider_checkpoint(
                    component="detector",
                    source=output_dir / "checkpoints/best_val_map.pt",
                    destination=args.promote_checkpoint_to,
                    config=config_path,
                    dataset_manifest=args.promotion_dataset_manifest,
                    initialization=(
                        "official COCO DEIMv2 DINOv3-L, then WIDER FACE bbox pretraining"
                    ),
                )
                print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return 0
        if (
            not oom
            or not bool(plan.get("runtime_oom_fallback", False))
            or index + 1 >= len(plans)
        ):
            return exit_code
        print(
            json.dumps(
                {
                    "runtime_oom_fallback": True,
                    "from_microbatch": batch["microbatch"],
                    "to_microbatch": plans[index + 1]["microbatch"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
