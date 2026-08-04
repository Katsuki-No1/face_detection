#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.head_pose_geometry.pose_encoder import (
    FusedSixDRepNet360,
    load_fused_onnx_checkpoint,
)

EXPECTED = {
    "sixdrepnet360_1x3x224x224_full.onnx": (
        "422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82"
    ),
    "sixdrepnet360_Nx3x224x224_full.onnx": (
        "852ea8e28014cee888a779fa235082fe3424e0ef2dd786c5c5496745741a46d8"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and dry-load a user-supplied 6DRepNet360 ONNX file."
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    report = load_fused_onnx_checkpoint(
        FusedSixDRepNet360(),
        args.checkpoint,
    )
    expected = EXPECTED.get(args.checkpoint.name)
    if expected is not None and report["sha256"] != expected:
        raise ValueError(f"Checksum mismatch: {report['sha256']} != {expected}")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
