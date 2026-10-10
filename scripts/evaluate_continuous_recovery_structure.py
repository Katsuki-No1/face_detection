#!/usr/bin/env python3
"""Validation-only structure sensitivity diagnostic; never selects a setting."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint, make_dataset
from keypoint_comparison1.landmark_detection.evaluation import evaluate_gt_boxes
from keypoint_comparison1.landmark_detection.continuous_teacher import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    success = args.output.parent / "success.json"
    success.unlink(missing_ok=True)
    config = load_landmark_config(args.config, project_root=ROOT)
    raw = copy.deepcopy(config.raw)
    checkpoint = config.output_root / "query_refinement/checkpoints" / raw["train"]["checkpoint_selection"]["filename"]
    model, _ = build_landmark_model(raw, variant="query_refinement", initialize_from_detector=False)
    load_landmark_checkpoint(model, checkpoint, prefer_ema=True, strict=True)
    if model.messenger_recovery.structure is None:
        raise ValueError("Structure diagnostic requires the structural branch")
    device = torch.device(args.device)
    model.to(device).eval()
    val = raw["data"]["evaluation_sets"]["v2_val"]
    if file_hash(val["annotation"]) != val["sha256"]:
        raise ValueError("Validation annotation hash mismatch")
    raw["data"]["annotations"]["val"] = val["annotation"]
    raw["data"].setdefault("image_roots", {})["val"] = val["root"]
    raw["data"].pop("validation_detector_crop_cache", None)
    dataset = make_dataset(raw, split="val", training=False)
    metrics = {}
    for name, mode in (("structure", None), ("zero_correction", "zero"), ("shuffle_context", "shuffle")):
        model.messenger_recovery.structure_ablation = mode
        values, _ = evaluate_gt_boxes(model, dataset, device=device, batch_size=16, num_workers=0,
                                     input_size=int(raw["data"]["input_size"]), attention_batches=0)
        metrics[name] = {key: value for key, value in values.items() if isinstance(value, (float, int)) and math.isfinite(value)}
        if not math.isfinite(float(values["nme"])):
            raise ValueError(f"Non-finite diagnostic NME: {name}")
    result = dict(schema_version=1, status="passed", split="val", diagnostic_only=True,
                  denominator="GT Face/ellipse crops; not primary deployed detection evaluation",
                  shuffle="cyclic shift by one sample within each fixed batch of 16; singleton unchanged",
                  selection="none; common baseline_aligned validation remains authoritative",
                  seed=raw["train"]["seed"], checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                  config=str(args.config), config_sha256=file_hash(args.config), annotation=val,
                  metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    success.write_text(json.dumps(dict(status="passed", report=str(args.output)), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
