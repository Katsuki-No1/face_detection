#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keypoint_comparison1.landmark_detection.config import (
    TWO_STAGE_FACE_KEYPOINT_PIPELINE,
    configured_evaluation_pipeline_stages,
    load_landmark_config,
)
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.evaluation import (
    evaluate_gt_boxes,
    sha256_file,
)
from keypoint_comparison1.landmark_detection.messenger_recovery import (
    patch_occlusion_ratios,
)
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import (
    load_landmark_checkpoint,
    make_dataset,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw/recovered landmark and DINO feature behavior."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _percentiles(values: list[np.ndarray]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "p05", "p25", "p50", "p75", "p95")}
    array = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "mean": float(array.mean()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def _truncated_slice(
    rows: list[dict[str, Any]],
    truncated_by_annotation: dict[int, np.ndarray],
) -> dict[str, Any]:
    errors: list[float] = []
    per_semantic = [dict(tp=0, predicted=0, gt=0) for _ in range(5)]
    for row in rows:
        mask = truncated_by_annotation.get(int(row["annotation_id"]))
        if mask is None or row.get("predicted_keypoints") is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"])
        denominator = 1.20 * max(1.0e-6, x2 - x1, y2 - y1)
        scores = row.get("presence_scores") or row.get("landmark_scores")
        for index in np.flatnonzero(mask):
            predicted = np.asarray(row["predicted_keypoints"][index], dtype=np.float64)
            target = np.asarray(row["gt_keypoints"][index], dtype=np.float64)
            error = float(np.linalg.norm(predicted - target) / denominator)
            errors.append(error)
            entry = per_semantic[int(index)]
            entry["gt"] += 1
            present = scores is not None and float(scores[index]) >= 0.70 - 1.0e-7
            entry["predicted"] += int(present)
            entry["tp"] += int(present and error <= 0.10)
    f1_values: list[float] = []
    for entry in per_semantic:
        if not entry["gt"]:
            continue
        precision = entry["tp"] / entry["predicted"] if entry["predicted"] else 0.0
        recall = entry["tp"] / entry["gt"]
        f1_values.append(2.0 * precision * recall / max(1.0e-8, precision + recall))
    return {
        "points": len(errors),
        "nme": float(np.mean(errors)) if errors else None,
        "pck_0.05": float(np.mean(np.asarray(errors) <= 0.05)) if errors else None,
        "pck_0.10": float(np.mean(np.asarray(errors) <= 0.10)) if errors else None,
        "semantic_macro_f1_nme_0.10": float(np.mean(f1_values)) if f1_values else None,
        "denominator": "1.20 * max(GT Head width, GT Head height)",
        "provenance": "deterministic synthetic boundary occlusion",
    }


@torch.inference_mode()
def _feature_diagnostics(
    model: torch.nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    patch_size: int,
) -> tuple[dict[str, Any] | None, dict[int, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_landmark_batch,
    )
    truncated_by_annotation: dict[int, np.ndarray] = {}
    alpha_all: list[np.ndarray] = []
    alpha_masked: list[np.ndarray] = []
    alpha_unmasked: list[np.ndarray] = []
    alpha_visible_keypoint: list[np.ndarray] = []
    alpha_occluded_keypoint: list[np.ndarray] = []
    cosine_original: list[np.ndarray] = []
    cosine_messenger: list[np.ndarray] = []
    cosine_recovered: list[np.ndarray] = []
    improvement: list[np.ndarray] = []
    messenger_improvement: list[np.ndarray] = []
    attention_diagonal_max: float | None = None
    model.eval()
    for batch_index, (images, targets) in enumerate(loader):
        for target in targets:
            truncated_by_annotation[int(target["annotation_id"])] = (
                target["synthetic_truncated"].numpy().astype(bool)
            )
        if getattr(model, "messenger_recovery", None) is None:
            continue
        images = images.to(device, non_blocking=True)
        clean_images = torch.stack([target["clean_image"] for target in targets]).to(
            device, non_blocking=True
        )
        masks = torch.stack([target["occlusion_mask"] for target in targets]).to(
            device, non_blocking=True
        )
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(
                images,
                recovery_mode="recovered",
                return_recovery_diagnostics=True,
                return_recovery_attention=batch_index == 0,
            )
            clean_map = model.extract_backbone_feature_map(clean_images)
        diagnostics = output["recovery_diagnostics"]
        clean = clean_map.flatten(2).transpose(1, 2)
        original = diagnostics["original_tokens"]
        messenger = diagnostics["messenger_tokens"]
        recovered = diagnostics["recovered_tokens"]
        alpha = diagnostics["gate_alpha"]
        ratios = patch_occlusion_ratios(masks, patch_size=patch_size).to(alpha.device)
        masked = ratios > 0.0
        original_cosine = F.cosine_similarity(original, clean, dim=-1)
        messenger_cosine = F.cosine_similarity(messenger, clean, dim=-1)
        recovered_cosine = F.cosine_similarity(recovered, clean, dim=-1)
        alpha_all.append(alpha.float().cpu().numpy().reshape(-1))
        alpha_masked.append(alpha[masked].float().cpu().numpy())
        alpha_unmasked.append(alpha[~masked].float().cpu().numpy())
        cosine_original.append(original_cosine[masked].float().cpu().numpy())
        cosine_messenger.append(messenger_cosine[masked].float().cpu().numpy())
        cosine_recovered.append(recovered_cosine[masked].float().cpu().numpy())
        improvement.append(
            (recovered_cosine[masked] > original_cosine[masked]).float().cpu().numpy()
        )
        messenger_improvement.append(
            (messenger_cosine[masked] > original_cosine[masked]).float().cpu().numpy()
        )
        height = images.shape[-2] // patch_size
        width = images.shape[-1] // patch_size
        for sample_index, target in enumerate(targets):
            points = target["keypoints"].numpy()
            visible = target["original_visible"].numpy().astype(bool)
            labeled = target["original_labeled"].numpy().astype(bool)
            patch_x = np.clip((points[:, 0] * width).astype(int), 0, width - 1)
            patch_y = np.clip((points[:, 1] * height).astype(int), 0, height - 1)
            indices = patch_y * width + patch_x
            values = alpha[sample_index, torch.as_tensor(indices, device=device)]
            if bool((labeled & visible).any()):
                alpha_visible_keypoint.append(
                    values[torch.as_tensor(labeled & visible, device=device)].float().cpu().numpy()
                )
            if bool((labeled & ~visible).any()):
                alpha_occluded_keypoint.append(
                    values[torch.as_tensor(labeled & ~visible, device=device)].float().cpu().numpy()
                )
        if batch_index == 0:
            attention = diagnostics.get("attention")
            if attention is not None:
                diagonal = attention.diagonal(dim1=-2, dim2=-1)
                attention_diagonal_max = float(diagonal.abs().max().cpu())
    if getattr(model, "messenger_recovery", None) is None:
        return None, truncated_by_annotation
    return {
        "alpha": _percentiles(alpha_all),
        "alpha_masked": _percentiles(alpha_masked),
        "alpha_unmasked": _percentiles(alpha_unmasked),
        "alpha_visible_keypoint_patch": _percentiles(alpha_visible_keypoint),
        "alpha_natural_occluded_keypoint_patch": _percentiles(alpha_occluded_keypoint),
        "masked_patch_cosine_original_clean": _percentiles(cosine_original),
        "masked_patch_cosine_messenger_clean": _percentiles(cosine_messenger),
        "masked_patch_cosine_recovered_clean": _percentiles(cosine_recovered),
        "recovered_improved_patch_fraction": _percentiles(improvement)["mean"],
        "messenger_improved_patch_fraction": _percentiles(messenger_improvement)["mean"],
        "self_attention_diagonal_absolute_max": attention_diagonal_max,
    }, truncated_by_annotation


def main() -> int:
    args = _args()
    config = load_landmark_config(args.config)
    if configured_evaluation_pipeline_stages(config.raw) != TWO_STAGE_FACE_KEYPOINT_PIPELINE:
        raise RuntimeError("Messenger evaluation requires the declared HEAD-free two-stage pipeline")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = build_landmark_model(
        config.raw,
        variant="query_refinement",
        initialize_from_detector=False,
    )
    checkpoint_report = load_landmark_checkpoint(
        model, args.checkpoint, prefer_ema=True, strict=True
    )
    model.to(device)
    evaluation_config = copy.deepcopy(config.raw)
    evaluation_set_name = "v2_val" if args.split == "val" else "v2_test"
    evaluation_set = evaluation_config["data"]["evaluation_sets"][evaluation_set_name]
    evaluation_config["data"]["annotations"][args.split] = evaluation_set["annotation"]
    evaluation_config["data"]["root"] = evaluation_set["root"]
    evaluation_config["data"].setdefault("image_roots", {})[args.split] = evaluation_set[
        "root"
    ]
    ordinary = make_dataset(evaluation_config, split=args.split, training=False)
    boundary = make_dataset(
        evaluation_config,
        split=args.split,
        training=False,
        evaluation_occlusion_kind="boundary",
    )
    eval_kwargs = {
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_size": int(config.raw["data"]["input_size"]),
        "attention_batches": 0,
    }
    recovered_metrics, _ = evaluate_gt_boxes(
        model, ordinary, recovery_mode="recovered", **eval_kwargs
    )
    raw_metrics, _ = evaluate_gt_boxes(
        model, ordinary, recovery_mode="raw", **eval_kwargs
    )
    boundary_recovered, boundary_recovered_rows = evaluate_gt_boxes(
        model, boundary, recovery_mode="recovered", **eval_kwargs
    )
    boundary_raw, boundary_raw_rows = evaluate_gt_boxes(
        model, boundary, recovery_mode="raw", **eval_kwargs
    )
    feature_diagnostics, truncated = _feature_diagnostics(
        model,
        boundary,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=int(config.raw["model"]["backbone"].get("patch_size", 16)),
    )
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "architecture_id": config.raw["evaluation"]["architecture_id"],
        "split": args.split,
        "pipeline_contract": {
            "type": "cascade",
            "stages": list(TWO_STAGE_FACE_KEYPOINT_PIPELINE),
            "head_metrics": None,
            "unsupported_reason": "declared_pipeline_has_no_head_stage",
        },
        "checkpoint": {
            **checkpoint_report,
            "sha256": sha256_file(args.checkpoint),
        },
        "config": {
            "path": str(config.path),
            "sha256": sha256_file(config.path),
        },
        "annotation": {
            "path": evaluation_set["annotation"],
            "sha256": sha256_file(Path(evaluation_set["annotation"])),
            "declared_sha256": evaluation_set.get("sha256"),
        },
        "seed": int(config.raw["train"]["seed"]),
        "coordinate_contract": {
            "crop": "256x256 long-axis square, scale 1.25",
            "nme": "1.20 * max(GT Head width, GT Head height) for semantic/truncated metrics",
            "keypoint_states": {"missing": 0, "occluded": 1, "visible": 2},
        },
        "augmentation_provenance": config.raw["data"]["augmentation"].get(
            "paired_occlusion"
        ),
        "git": {"revision": git_revision, "dirty": git_dirty},
        "ordinary": {"recovered": recovered_metrics, "raw": raw_metrics},
        "boundary": {
            "recovered": boundary_recovered,
            "raw": boundary_raw,
            "truncated_recovered": _truncated_slice(
                boundary_recovered_rows, truncated
            ),
            "truncated_raw": _truncated_slice(boundary_raw_rows, truncated),
        },
        "feature_recovery": feature_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.success_marker is not None:
        args.success_marker.parent.mkdir(parents=True, exist_ok=True)
        args.success_marker.write_text(
            json.dumps(
                {
                    "status": "success",
                    "metrics": str(args.output),
                    "checkpoint_sha256": payload["checkpoint"]["sha256"],
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
