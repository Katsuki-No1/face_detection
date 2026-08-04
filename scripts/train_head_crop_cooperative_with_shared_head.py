#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare common HEAD caches, then train the HEAD-crop cooperative model."
    )
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/prepare_shared_head_caches.py"),
            "--config",
            str(args.head_config.resolve()),
            "--checkpoint",
            str(args.head_checkpoint.resolve()),
            "--output-root",
            str(args.cache_output_root.resolve()),
            "--minimum-score",
            "0.30",
            "--maximum-per-image",
            "100",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/train_face_keypoint_cooperative.py"),
        "--config",
        str(args.config.resolve()),
        "--device",
        args.device,
        "--resume",
        "--wandb" if args.wandb else "--no-wandb",
    ]
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
