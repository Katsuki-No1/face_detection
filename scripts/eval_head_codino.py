#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.head_codino.checkpoint import load_checkpoint_exact
from src.models.head_codino.runtime import (
    build_dataloader,
    build_model_from_runtime_config,
    cuda_operator_preflight,
    load_runtime_config,
    move_targets_to_device,
    resolve_path,
    save_json,
    tensor_predictions_to_coco,
)
from src.wandb_utils import sync_evaluation_to_wandb


def main() -> None:
    elapsed_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate Head Co-DINO with COCO bbox metrics")
    parser.add_argument("--config", default="configs/head_codino/production.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--output")
    parser.add_argument("--use-model-state", action="store_true", help="Use model instead of EMA")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    split = args.split or str(config.get("evaluation", {}).get("split", "val"))
    config = copy.deepcopy(config)
    config["model"]["backbone_weights"] = None
    device = torch.device("cuda")
    print(json.dumps({"preflight": cuda_operator_preflight(device)}, ensure_ascii=False))
    model = build_model_from_runtime_config(config).to(device).eval()
    audit = load_checkpoint_exact(model, args.checkpoint, prefer_ema=not args.use_model_state)
    loader, _ = build_dataloader(config, split)
    predictions = []
    inference_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch, targets in loader:
            batch = batch.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                batch_predictions = model.predict(batch)
            torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started
            predictions.extend(tensor_predictions_to_coco(batch_predictions, targets))

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else resolve_path(config["evaluation"]["output"], config=config)
    )
    save_json(predictions, output)
    annotation_path = resolve_path(config["data"][split]["annotation"], config=config)
    coco_gt = COCO(str(annotation_path))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": list(coco_gt.dataset["images"]),
            "categories": list(coco_gt.dataset["categories"]),
            "annotations": [],
        }
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    )
    metrics = {
        "split": split,
        "category_id": 1,
        "max_dets": 100,
        "checkpoint_sha256": audit["sha256"],
        "prediction_count": len(predictions),
        **{name: float(value) for name, value in zip(names, evaluator.stats, strict=True)},
    }
    elapsed = time.perf_counter() - elapsed_started
    image_count = len(loader.dataset)
    metrics["runtime"] = {
        "total_seconds": elapsed,
        "images_per_second": image_count / max(elapsed, 1.0e-12),
        "milliseconds_per_image": elapsed * 1000.0 / max(1, image_count),
        "model_inference_seconds": inference_seconds,
        "model_inference_milliseconds_per_image": (
            inference_seconds * 1000.0 / max(1, image_count)
        ),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "batch_size": int(config.get("evaluation", {}).get("batch_size", 1)),
        "input_size": int(config["data"]["image_size"]),
    }
    metrics_path = output.with_name(output.stem + "_metrics.json")
    save_json(metrics, metrics_path)
    baseline_dir = output.parent / "baseline_aligned"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/evaluate_baseline_aligned_predictions.py"),
            "--format",
            "head_coco",
            "--predictions",
            str(output),
            "--annotations",
            str(annotation_path),
            "--split",
            split,
            "--architecture-id",
            "shared_head_codino_vits16plus_960",
            "--comparison-stage",
            "head",
            "--output-dir",
            str(baseline_dir),
            "--source-metrics",
            str(metrics_path),
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--checkpoint",
            str(Path(args.checkpoint).expanduser().resolve()),
            "--seed",
            str(int(config.get("train", {}).get("seed", 42))),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    wandb_cfg = config.get("wandb", {})
    if isinstance(wandb_cfg, dict) and bool(wandb_cfg.get("enabled", False)):
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is required to sync HEAD evaluation") from error
        training_output = resolve_path(config["train"]["output_dir"], config=config)
        sync_evaluation_to_wandb(
            wandb,
            state_path=training_output / "wandb_run.json",
            split=split,
            metric_sources={
                "coco_head": metrics_path,
                "baseline_aligned": baseline_dir / "metrics.json",
            },
            artifact_files=(
                output,
                baseline_dir / "comparison_table.md",
                baseline_dir / "success.json",
            ),
            headline={
                "head_AP": metrics["AP"],
                "head_AP50": metrics["AP50"],
                "head_AP75": metrics["AP75"],
                "head_AR100": metrics["AR_100"],
            },
            report_path=metrics_path.parent / "wandb_evaluation_sync.json",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
