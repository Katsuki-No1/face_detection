#!/usr/bin/env python3
"""Run one gate command and atomically record success only on exit code zero."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after --")
    marker = args.success_marker.resolve()
    marker.unlink(missing_ok=True)
    started = time.time()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    payload = {
        "status": "passed",
        "gate_stage": str(args.stage),
        "command": command,
        "elapsed_seconds": time.time() - started,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
