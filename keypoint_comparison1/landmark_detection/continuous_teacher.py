"""Training-only fixed teacher and train-only feature scale provenance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .heads import sine_position_encoding_2d


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_continuous_teacher(model, dataset, raw, output_dir, device):
    """Called after baseline initialization, before loading a resumed student.

    Teacher is deliberately outside student/optimizer/checkpoints: its identity
    is the immutable warm-start checkpoint hash. No offline student features.
    """
    teacher = copy.deepcopy(model.backbone).eval().requires_grad_(False)
    block = model.messenger_recovery
    cfg = block.config
    source = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    source = copy.copy(source)
    source.training = False
    source.evaluation_occlusion_kind = None
    source.structure_heads = None
    indices = np.random.default_rng(int(raw["train"]["seed"])).choice(len(source), min(cfg.scale_samples, len(source)), replace=False).tolist()
    total = torch.zeros(block.embed_dim, dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for index in indices:
            image, _ = source[index]
            feature = teacher(image[None].to(device)).float().flatten(2).transpose(1, 2).reshape(-1, block.embed_dim).double()
            total += feature.sum(0)
            squares += feature.square().sum(0)
            count += len(feature)
    scale = (squares / count - (total / count).square()).clamp_min(0).sqrt().clamp_min(cfg.scale_floor).float()
    if not torch.isfinite(scale).all():
        raise ValueError("Non-finite train-only teacher statistics")
    block.channel_scale.copy_(scale)
    block.scale_ready.fill_(True)
    provenance = {
        "schema_version": 1, "teacher": "fixed_baseline_no_ema", "student": "live_crop_dino_staged_unfreeze",
        "checkpoint": raw["train"]["finetune_checkpoint"],
        "annotation_path": str(source.annotation_path), "annotation_sha256": file_hash(source.annotation_path),
        "split": "train", "seed": raw["train"]["seed"], "sample_indices": indices,
        "sample_annotation_ids": [source.samples[i]["annotation"].get("id") for i in indices],
        "method": "population_std_over_train_clean_gt_ellipse_crop_patches_float64", "augmentation": "none",
        "patch_count": count, "floor": cfg.scale_floor, "channel_scale": scale.cpu().tolist(),
        "structure_head_annotations": raw["data"].get("structure_head_annotations"),
        "structure_head_sha256": file_hash(raw["data"]["structure_head_annotations"]) if raw["data"].get("structure_head_annotations") else None,
        "structure_supervision_inventory": source.structure_supervision_report,
        "coordinate_convention": "rotated_crop_axes_x_right_y_down; isotropic_inverse_scale; semantic_flip_1_0_2_4_3; GT_Head_side=1.20*max(w,h)",
    }
    path = Path(output_dir) / "continuous_teacher_provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise RuntimeError("Fixed teacher/statistics provenance changed; use a new output directory")
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    # W&B receives config.raw; preserve scale values and provenance there too.
    raw["continuous_teacher_provenance"] = provenance
    return teacher


def continuous_training_losses(model, teacher, criterion, output, targets, device):
    block = model.messenger_recovery
    clean = torch.stack([target["clean_image"] for target in targets]).to(device)
    masks = torch.stack([target["occlusion_mask"] for target in targets]).to(device)
    with torch.no_grad():
        clean_map = teacher(clean)
    clean_pi = None
    clean_descriptor = None
    if block.config.consistency_weight:
        b, c, h, w = clean_map.shape
        position = sine_position_encoding_2d(h, w, c, device=device, dtype=clean_map.dtype)
        # Shared trainable structure head: clean descriptor anchors it directly;
        # KL target alone is stop-gradient. No EMA teacher is used.
        _, clean_pi, clean_descriptor = block.structure(clean_map.flatten(2).transpose(1, 2), position)
    return criterion(output["recovery_diagnostics"], clean_map, masks, channel_scale=block.channel_scale,
                     targets=targets, clean_pi=clean_pi, clean_descriptor=clean_descriptor)
