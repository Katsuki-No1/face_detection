#!/usr/bin/env python3
"""Validate a production training configuration without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=(
            "cooperative",
            "head_codino",
            "keypoint_comparison1",
            "keypoint_comparison2",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_summarize(kind: str, path: Path) -> dict[str, Any]:
    if kind == "cooperative":
        from tools.face_keypoint_cooperative_common import load_cooperative_config

        raw = load_cooperative_config(path)
        return {
            "experiment_name": raw.get("experiment_name"),
            "experiment_objective": raw.get("experiment_objective"),
            "output_root": raw.get("output", {}).get("root"),
            "epochs": raw.get("train", {}).get("epochs"),
            "amp_dtype": raw.get("train", {}).get("amp_dtype"),
            "strict_sequential_cascade": bool(
                raw.get("model", {})
                .get("strict_sequential_cascade", {})
                .get("enabled", False)
            ),
            "execution_order": raw.get("verification", {}).get(
                "expected_execution_order"
            ),
        }
    if kind == "head_codino":
        from src.models.head_codino.runtime import (
            load_runtime_config,
            validate_architecture_contract,
        )

        config = load_runtime_config(path)
        validate_architecture_contract(config)
        return {
            "experiment_name": config.get("experiment", {}).get("name"),
            "experiment_objective": config.get("experiment_objective"),
            "output_root": config.get("train", {}).get("output_dir"),
            "epochs": config.get("train", {}).get("epochs"),
            "image_size": config.get("data", {}).get("image_size"),
        }
    if kind == "keypoint_comparison1":
        from keypoint_comparison1.landmark_detection.config import load_landmark_config

        config = load_landmark_config(path, project_root=REPO_ROOT)
        return {
            "experiment_name": config.raw.get("experiment_name"),
            "output_root": str(config.output_root),
            "epochs": config.raw.get("train", {}).get("epochs"),
            "state_head_mode": config.raw.get("model", {}).get("state_head_mode"),
        }
    from keypoint_comparison2.config import load_config

    config = load_config(path, project_root=REPO_ROOT)
    return {
        "experiment_name": config.raw.get("experiment_name"),
        "output_root": str(config.output_root),
        "epochs": config.raw.get("train", {}).get("epochs"),
        "models": sorted(config.raw.get("models", {})),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    marker = args.success_marker.resolve()
    marker.unlink(missing_ok=True)
    summary = load_and_summarize(args.kind, config_path)
    payload = {
        "status": "passed",
        "gate_stage": "schema",
        "kind": args.kind,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "summary": summary,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
