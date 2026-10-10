#!/usr/bin/env python3
"""Bounded CPU autograd audit on the real warm-started crop DINO/Query head.

Synthetic inputs only; does not launch a trainer, save checkpoints or use test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keypoint_comparison1.landmark_detection.config import load_landmark_config, configured_keypoint_target_policy, configured_state_head_mode
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import initialize_from_finetune_checkpoint, _optimizer
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.continuous_recovery import ContinuousRecoveryLoss


def norms(model):
    block = model.messenger_recovery
    modules = dict(crop_dino=model.backbone, query_refinement=model.head, decoder=block.decoder)
    for i, layer in enumerate(block.layers):
        modules.update({f"messenger_{i+1}_{key}": getattr(layer, key) for key in ("q", "k", "v", "out", "ffn", "alpha")})
    if block.local_gate is not None:
        modules["adoption_gate"] = block.local_gate
    if block.structure is not None:
        modules.update(structure_attention=block.structure.attention, structure_projection=block.structure.project,
                       structure_correction=block.structure_gate, descriptor_decoder=block.structure.descriptor)
    result = {}
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if any(not torch.isfinite(grad).all() for grad in grads):
            raise AssertionError(f"Non-finite gradient in {name}")
        result[name] = sum(float(grad.abs().sum()) for grad in grads)
    for name, parameter in {"messenger_token": block.messenger_token, **({"structure_prototypes": block.structure.prototypes} if block.structure else {})}.items():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            raise AssertionError(name)
        result[name] = float(grad.abs().sum()) if grad is not None else 0.
    return result


def audit(arm):
    torch.manual_seed(42)
    path = ROOT / f"keypoint_comparison1/configs/continuous_recovery_{arm}.yaml"
    raw = load_landmark_config(path).raw
    model, _ = build_landmark_model(raw, variant="query_refinement", initialize_from_detector=False)
    initialization = initialize_from_finetune_checkpoint(model, raw["train"]["finetune_checkpoint"], variant="query_refinement",
        expected_target_policy=configured_keypoint_target_policy(raw), expected_state_head_mode=configured_state_head_mode(raw))
    teacher = copy.deepcopy(model.backbone).eval().requires_grad_(False)
    block = model.messenger_recovery
    optimizer = _optimizer(model, raw["train"])
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        clean = teacher(images)
    occluded = images.clone(); occluded[:, :, :32, :32] = 0
    masks = torch.zeros(2, 64, 64); masks[:, :32, :32] = 1
    targets = [dict(keypoints=torch.rand(5, 2), labeled=torch.ones(5, dtype=torch.bool), visible=torch.ones(5, dtype=torch.bool),
                    structure_pairs=torch.randn(10, 2) * .1, structure_pairs_valid=torch.ones(10, dtype=torch.bool)) for _ in range(2)]
    task_loss = LandmarkLoss(raw["loss"])
    # Consistency is covered by the production-helper unit test; the audit's
    # warm-up uses a detached uniform clean assignment to open zero-init paths.
    def forward():
        out = model(occluded, return_recovery_diagnostics=True)
        return out, task_loss(out, targets)["loss_coordinates"]
    _, loss = forward(); loss.backward()
    initial = norms(model)
    assert initial["crop_dino"] > 0 and initial["decoder"] > 0
    if block.structure:
        assert initial["structure_correction"] > 0 and initial["structure_prototypes"] == 0
    optimizer.zero_grad(set_to_none=True)
    model.backbone.requires_grad_(False)
    auxiliary = ContinuousRecoveryLoss(block.config)
    for _ in range(4):
        out, loss = forward()
        aux = auxiliary(out["recovery_diagnostics"], clean, masks, channel_scale=block.channel_scale,
                        targets=targets, clean_pi=torch.full((2, block.config.prototypes), 1 / block.config.prototypes))
        (loss + aux["loss_messenger_total"]).backward()
        assert all(p.grad is None for p in model.backbone.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model.backbone.requires_grad_(True)
    _, loss = forward(); loss.backward()
    after = norms(model)
    for name, norm in after.items():
        assert norm > 0 if name != "descriptor_decoder" else norm == 0, (arm, name, norm)
    selected = next(p for p in model.backbone.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    before = selected.detach().clone(); optimizer.step()
    assert not torch.equal(before, selected)
    return dict(arm=arm, status="passed", seed=42, device="cpu", synthetic_input_shape=list(images.shape),
                production_training="not run", checkpoint=initialization, config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                initial_keypoint_only_gradient_l1=initial, after_four_auxiliary_steps_and_unfreeze_keypoint_only_gradient_l1=after,
                optimizer_unique_complete=True, backbone_updated_after_unfreeze=True, teacher_gradients_absent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(schema_version=1, audits=[],
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()))
    for arm in ("option1", "option3", "combined_option1"):
        result["audits"].append(audit(arm))
        print(f"{arm}: passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
