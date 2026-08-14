#!/usr/bin/env python3
"""Train ellipse-local DEIMv2 Joint with its validated fallback plan."""

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

from keypoint_comparison2.config import (  # noqa: E402
    load_config,
    resolved_config_sha256,
)
from src.wandb_utils import write_json_atomic  # noqa: E402


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
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    plan_path = args.batch_plan.expanduser().resolve()
    config = load_config(config_path, project_root=PROJECT_ROOT)
    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "passed":
        raise RuntimeError("Batch plan has not passed")
    if plan.get("resolved_config_sha256") != resolved_config_sha256(config.raw):
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
    output_dir = config.output_root / "deimv2_joint"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for index, batch in enumerate(plans):
        resume = output_dir / "checkpoints/last.pt"
        resumed = resume.is_file()
        command = [
            sys.executable,
            str(PROJECT_ROOT / "keypoint_comparison2/run.py"),
            "--config",
            str(config_path),
            "--stage",
            "train",
            "--models",
            "deimv2_joint",
            "--device",
            args.device,
            "--batch-plan",
            str(plan_path),
            "--batch-size",
            str(batch["microbatch"]),
            "--gradient-accumulation-steps",
            str(batch["accumulation_steps"]),
            "--resume" if resumed else "--no-resume",
            "--wandb" if args.wandb else "--no-wandb",
        ]
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
