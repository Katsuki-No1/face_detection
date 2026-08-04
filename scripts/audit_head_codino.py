#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.head_codino.checkpoint import audit_checkpoint, parameter_audit, sha256_file
from src.models.head_codino.runtime import (
    build_model_from_runtime_config,
    cuda_operator_preflight,
    load_runtime_config,
    resolve_path,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Head Co-DINO parameters and checkpoints")
    parser.add_argument("--config", default="configs/head_codino/production.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--use-model-state", action="store_true")
    parser.add_argument("--skip-cuda-preflight", action="store_true")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    requested_weight = config.get("model", {}).get("backbone_weights")
    weight_path = resolve_path(requested_weight, config=config) if requested_weight else None
    weight_status: dict[str, object]
    if args.checkpoint:
        config = copy.deepcopy(config)
        config["model"]["backbone_weights"] = None
        weight_status = {
            "status": "not_required",
            "reason": "full detector checkpoint supplied",
            "configured_path": str(weight_path) if weight_path else None,
        }
    elif weight_path is not None and not weight_path.exists():
        config = copy.deepcopy(config)
        config["model"]["backbone_weights"] = None
        weight_status = {
            "status": "skipped",
            "reason": "configured official ViT-S+ weight is missing",
            "configured_path": str(weight_path),
        }
    elif weight_path is not None:
        weight_status = {
            "status": "strict_loaded",
            "path": str(weight_path),
            "sha256": sha256_file(weight_path),
        }
    else:
        weight_status = {"status": "random_initialization"}
    report = {
        "cuda_preflight": (
            {"status": "skipped"} if args.skip_cuda_preflight else cuda_operator_preflight()
        ),
        "backbone_weight": weight_status,
    }
    model = build_model_from_runtime_config(config)
    report["component_parameters"] = parameter_audit(model)
    if args.checkpoint:
        checkpoint_report, _state = audit_checkpoint(
            model,
            args.checkpoint,
            prefer_ema=not args.use_model_state,
        )
        report["checkpoint"] = checkpoint_report
    else:
        report["checkpoint"] = {"status": "skipped", "reason": "no checkpoint supplied"}
    if args.output:
        save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.checkpoint and not report["checkpoint"]["strict_compatible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
