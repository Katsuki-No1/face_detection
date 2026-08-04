#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.active_learning.tracking import bbox_iou
from src.active_learning.presence import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the auxiliary one-class RTMDet head detector.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rtmdet/rtmdet_s_head_mmdet_hf_960.py")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "outputs/experiments/rtmdet_s_head_mmdet_hf_960")
    parser.add_argument("--source-checkpoint", type=Path, default=PROJECT_ROOT / "outsput/experience/work_dirs/mmdet/rtmdet_hf_full_rtm_hfdp_20260704_170447/best_coco_bbox_mAP_50_epoch_67.pth")
    parser.add_argument("--baseline-config", type=Path, default=PROJECT_ROOT / "outsput/experience/configs/mmdet/rtmdet_hf_full.py")
    parser.add_argument("--baseline-checkpoint", type=Path, default=PROJECT_ROOT / "outsput/experience/work_dirs/mmdet/rtmdet_hf_full_rtm_hfdp_20260704_170447/best_coco_bbox_mAP_50_epoch_67.pth")
    parser.add_argument("--test-annotation", type=Path, default=PROJECT_ROOT / "outsput/experience/data/mmdet_hf/test.json")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "outsput/experience/data/mmdet_hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def prepare_backbone_neck_checkpoint(source: Path, output: Path) -> Path:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    retained = {
        key: value for key, value in state.items()
        if key.startswith("backbone.") or key.startswith("neck.")
    }
    if not retained:
        raise RuntimeError(f"No backbone/neck parameters found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": retained, "meta": {"source_checkpoint": source.resolve().as_posix()}}, output)
    return output


def train(args: argparse.Namespace) -> Path:
    from mmengine.config import Config
    from mmengine.runner import Runner

    init_checkpoint = prepare_backbone_neck_checkpoint(args.source_checkpoint, args.work_dir / "head_backbone_neck_init.pth")
    cfg = Config.fromfile(args.config.resolve().as_posix())
    cfg.work_dir = args.work_dir.resolve().as_posix()
    cfg.load_from = init_checkpoint.resolve().as_posix()
    runner = Runner.from_cfg(cfg)
    runner.train()
    checkpoints = sorted(args.work_dir.glob("best_coco_bbox_mAP_50*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No validation-best checkpoint was produced under {args.work_dir}")
    return checkpoints[-1]


def _evaluate_threshold(
    ground_truth: dict[int, list[Sequence[float]]],
    predictions: dict[int, list[tuple[float, Sequence[float]]]],
    threshold: float,
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for image_id, boxes in ground_truth.items():
        matched: set[int] = set()
        for score, prediction in sorted(predictions.get(image_id, []), reverse=True):
            if score < threshold:
                continue
            candidates = [(bbox_iou(prediction, box), index) for index, box in enumerate(boxes) if index not in matched]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(boxes) - len(matched)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"threshold": threshold, "precision": precision, "recall": recall, "tp": true_positive, "fp": false_positive, "fn": false_negative}


def evaluate(config: Path, checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mmdet.apis import inference_detector, init_detector

    data = json.loads(args.test_annotation.read_text(encoding="utf-8"))
    categories = {str(row["name"]).lower(): int(row["id"]) for row in data["categories"]}
    head_id = categories["head"]
    images = {int(row["id"]): row for row in data["images"]}
    ground_truth: dict[int, list[Sequence[float]]] = {image_id: [] for image_id in images}
    for annotation in data["annotations"]:
        if int(annotation["category_id"]) != head_id:
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        ground_truth[int(annotation["image_id"])].append((x, y, x + width, y + height))
    old_force_load = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = init_detector(config.resolve().as_posix(), checkpoint.resolve().as_posix(), device=args.device)
    finally:
        if old_force_load is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_force_load
    predictions: dict[int, list[tuple[float, Sequence[float]]]] = {}
    for image_id, image in images.items():
        path = args.image_root / str(image["file_name"])
        frame = cv2.imread(path.as_posix())
        if frame is None:
            raise FileNotFoundError(path)
        result = inference_detector(model, frame)
        instances = result.pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy().astype(int)
        predictions[image_id] = [(float(score), box.tolist()) for score, box, label in zip(scores, boxes, labels, strict=True) if label == 0]
    sweep = [_evaluate_threshold(ground_truth, predictions, float(threshold)) for threshold in np.linspace(0.01, 0.99, 99)]
    feasible = [row for row in sweep if row["precision"] >= 0.50]
    selected = max(feasible, key=lambda row: (row["recall"], row["precision"], row["threshold"]), default=max(sweep, key=lambda row: row["precision"]))
    return {"config": config.resolve().as_posix(), "checkpoint": checkpoint.resolve().as_posix(), "selected": selected, "sweep": sweep}


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = None
    if args.skip_train:
        checkpoints = sorted(args.work_dir.glob("best_coco_bbox_mAP_50*.pth"), key=lambda path: path.stat().st_mtime)
        if not checkpoints:
            raise FileNotFoundError("--skip-train was used but no best checkpoint exists")
        candidate_checkpoint = checkpoints[-1]
    else:
        candidate_checkpoint = train(args)
    candidate = evaluate(args.config, candidate_checkpoint, args)
    baseline = evaluate(args.baseline_config, args.baseline_checkpoint, args)
    candidate_recall = float(candidate["selected"]["recall"])
    baseline_recall = float(baseline["selected"]["recall"])
    adopted = candidate_recall >= 0.80 and candidate_recall - baseline_recall >= 0.02
    result = {
        "adopted": adopted,
        "requirements": {"minimum_recall": 0.80, "minimum_recall_gain": 0.02, "minimum_precision": 0.50},
        "candidate": candidate,
        "baseline": baseline,
        "selected_config": (args.config if adopted else args.baseline_config).resolve().as_posix(),
        "selected_checkpoint": (candidate_checkpoint if adopted else args.baseline_checkpoint).resolve().as_posix(),
        "selected_checkpoint_sha256": sha256_file(candidate_checkpoint if adopted else args.baseline_checkpoint),
        "failure_reason": "" if adopted else "dedicated head detector did not meet the recall adoption gate",
    }
    (args.work_dir / "adoption.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("adopted", "selected_config", "selected_checkpoint", "failure_reason")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
