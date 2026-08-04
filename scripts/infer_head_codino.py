#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.head_codino.checkpoint import load_checkpoint_exact
from src.models.head_codino.runtime import (
    build_model_from_runtime_config,
    cuda_operator_preflight,
    load_runtime_config,
    read_rgb_tensor,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native-resolution Head Co-DINO inference")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--config", default="configs/head_codino/production.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/head_codino/inference.json")
    parser.add_argument("--use-model-state", action="store_true")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    config = copy.deepcopy(config)
    config["model"]["backbone_weights"] = None
    device = torch.device("cuda")
    print(json.dumps({"preflight": cuda_operator_preflight(device)}, ensure_ascii=False))
    model = build_model_from_runtime_config(config).to(device).eval()
    load_checkpoint_exact(model, args.checkpoint, prefer_ema=not args.use_model_state)
    image_paths = [Path(value).expanduser().resolve() for value in args.images]
    images = [read_rgb_tensor(path).to(device) for path in image_paths]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predictions = model.predict(images)
    records = []
    for path, prediction in zip(image_paths, predictions, strict=True):
        records.append(
            {
                "image": str(path),
                "original_size": prediction["original_size"].tolist(),
                "boxes": prediction["boxes"].cpu().tolist(),
                "scores": prediction["scores"].cpu().tolist(),
                "labels": prediction["labels"].cpu().tolist(),
            }
        )
    save_json(records, args.output)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
