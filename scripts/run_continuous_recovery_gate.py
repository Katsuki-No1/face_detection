#!/usr/bin/env python3
"""Isolated schema, production-path batch preflight, overfit and small-scale gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_multiscale_query_refinement_gate import (
    run_training, learning_gate, remove_checkpoints, write_json,
)
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.continuous_teacher import file_hash


def validate_real_mixed_contract(raw):
    """Explicit opt-in ablation; never relax existing recovery-only contracts."""
    paired = raw["data"]["augmentation"]["paired_occlusion"]
    if paired.get("appearance") != "real_mixed":
        return False
    arm = raw.get("real_mixed_ablation_arm")
    if arm not in {"recovery_on", "recovery_off"}:
        raise ValueError("Real mixed experiment needs an explicit recovery ablation arm")
    if bool(raw["model"]["continuous_recovery"]["enabled"]) != (arm == "recovery_on"):
        raise ValueError("Real mixed ablation arm and enabled flag disagree")
    contract = raw.get("augmentation_contract", {})
    if (contract.get("version") != "real_mixed_v1"
            or contract.get("family_probabilities") != {"face": .5, "non_face": .5}
            or contract.get("recipients") != "original_face_occluded_false_and_all_five_visible"
            or contract.get("crop_outside_keypoints") != "missing"
            or contract.get("gradient") != dict(width=48, strength=2, alpha=False, edit_recipient_outside_mask=False)
            or paired.get("probability") != .30 or paired.get("object_overlap_policy") is not None):
        raise ValueError("Real mixed augmentation contract differs from its implementation")
    return True


def schema_gate(config, work_dir):
    raw = load_landmark_config(config).raw
    cfg = raw["model"]["continuous_recovery"]
    real_mixed = validate_real_mixed_contract(raw)
    if not cfg["enabled"] and not real_mixed:
        raise ValueError("Continuous recovery must be enabled")
    conditional_policy = raw["data"]["augmentation"]["paired_occlusion"].get("object_overlap_policy")
    if raw["evaluation"].get("pipeline_contract") != {"type": "cascade", "stages": ["full_image_face_ellipse", "face_crop_keypoint"]}:
        raise ValueError("Continuous recovery requires explicit HEAD-free deployed topology")
    baseline = raw["train"]["finetune_checkpoint"]
    if file_hash(baseline["path"]) != baseline["sha256"]:
        raise ValueError("Warm-start/fixed-teacher checkpoint hash mismatch")
    # The canonical source-video inventories are an independent guard against
    # replacing a split with another equally sized set.
    canonical = PROJECT_ROOT / "outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1"
    annotation_hashes = {}
    for split, inventory in {"train": (44, 4237, 7129), "val": (6, 530, 882), "test": (6, 528, 806)}.items():
        path = canonical / f"{split}.json"
        data = json.loads(path.read_text())
        actual = (len({image["source_video_id"] for image in data["images"]}), len(data["images"]), len(data["annotations"]))
        if actual != inventory:
            raise ValueError(f"Canonical {split} inventory mismatch: {actual}")
        annotation_hashes[split] = file_hash(path)
        if annotation_hashes[split] != raw["data"]["continuous_source_inventory"][split]["sha256"]:
            raise ValueError(f"Canonical {split} annotation hash mismatch")
    if Path(raw["data"]["structure_head_annotations"]).resolve() != (canonical / "train.json").resolve():
        raise ValueError("Structure supervision must use canonical train Heads only")
    command = [str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "keypoint_comparison1/tests/test_continuous_recovery.py", "tests/test_continuous_recovery_schedule.py"]
    abc = "continuous_abc_" in raw["evaluation"]["architecture_id"]
    if abc:
        command.extend(["keypoint_comparison1/tests/test_recovery_abc.py", "tests/test_recovery_abc_schedule.py"])
        expected = dict(alpha_weight=0.0, candidate_weight=.10, output_weight=.10, adoption_weight=.01,
                        gram_weight=.05, descriptor_weight=.10, consistency_weight=.01, layers=2,
                        mixing_warmup_epochs=2, beta=.10, structure_enabled=True, adoption="optimal")
        for key, value in expected.items():
            if cfg[key] != value:
                raise ValueError(f"ABC locked condition changed: {key}")
        if raw["train"]["seed"] != 42 or raw["train"]["intended_effective_batch_size"] != 64:
            raise ValueError("ABC requires seed42/effective batch64")
        if raw["train"]["freeze_backbone_epochs"] != 5 or raw["train"]["epochs"] != 40:
            raise ValueError("ABC backbone schedule/epoch budget changed")
        if raw["train"]["warmup_epochs"] != 5 or raw["train"]["flat_epochs"] != 5:
            raise ValueError("ABC decay must start immediately after warmup5")
        if raw["data"]["augmentation"]["paired_occlusion"]["probability"] != .30:
            raise ValueError("ABC paired augmentation must remain 30%")
    if conditional_policy is not None:
        command.extend(["keypoint_comparison1/tests/test_occlusion_policy.py", "tests/test_conditional_object_training_schedule.py"])
    if real_mixed:
        command.extend(["tests/test_real_mixed_occlusion.py", "tests/test_cross_video_face_occlusion_prototype.py"])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    from keypoint_comparison1.landmark_detection.training import make_dataset
    dataset = make_dataset(raw, split="train", training=True)
    material_audit = []
    distribution_audit = None
    if conditional_policy is not None:
        # Fresh full-train audit bound to this production config, not a historical marker.
        directory = work_dir / "distribution_audit"
        subprocess.run([str(PROJECT_ROOT / ".venv/bin/python"), "scripts/audit_occlusion_distribution.py",
            "--config", str(config), "--output", str(directory), "--epochs", "1", "12", "40", "--seeds", "42", "43"],
            cwd=PROJECT_ROOT, check=True)
        report = directory / "distribution.json"
        value = json.loads(report.read_text())
        if value.get("status") != "passed" or not all(value["checks"].values()):
            raise ValueError("Conditional object distribution audit failed")
        distribution_audit = dict(path=str(report), sha256=file_hash(report), summary=value["summary"], checks=value["checks"])
    if real_mixed:
        audits = []
        for epoch in (1, 12, 40):
            directory = work_dir / f"real_mixed_distribution_epoch_{epoch}"
            subprocess.run([str(PROJECT_ROOT / ".venv/bin/python"), "scripts/audit_real_mixed_occlusion.py",
                "--config", str(config), "--output", str(directory), "--epoch", str(epoch), "--seeds", "42", "43"],
                cwd=PROJECT_ROOT, check=True)
            report = directory / "report.json"
            value = json.loads(report.read_text())
            if (value.get("status") != "passed" or value.get("scope") != "full_rgb_loader_audit"
                    or not all(value["checks"].values())):
                raise ValueError("Real mixed distribution audit failed")
            audits.append(dict(path=str(report), sha256=file_hash(report), checks=value["checks"]))
        distribution_audit = dict(real_mixed=audits)
    if abc and conditional_policy is None and not real_mixed:
        import torch
        from keypoint_comparison1.landmark_detection.training import seed_everything
        dataset.set_epoch(1)
        for index in range(min(96, len(dataset))):
            seed_everything(42 + index)
            image, target = dataset[index]
            if not torch.isfinite(image).all() or not torch.isfinite(target["clean_image"]).all():
                raise ValueError("Non-finite paired input")
            meta = target["paired_occlusion"]
            if meta.get("applied"):
                material_audit.append(dict(sample_index=index,annotation_id=int(target["annotation_id"]),**meta))
                if not .1 <= meta["face_overlap"] <= .4:
                    raise ValueError("Real-data paired mask violated overlap bounds")
        if not material_audit:
            raise ValueError("No real-data augmentation was exercised")
        write_json(work_dir / "paired_material_samples.json", dict(samples=material_audit,split="train",seed=42))
    return dict(config=raw, canonical_annotation_sha256=annotation_hashes, tests=command,
                structure_supervision_inventory=dataset.structure_supervision_report,
                real_train_paired_samples=material_audit, distribution_audit=distribution_audit)


def select_plan(attempts):
    safe = [attempt for attempt in attempts if attempt["status"] == "safe"]
    if not safe:
        raise RuntimeError("No safe continuous-recovery microbatch")
    fastest = min(attempt["projected_production_seconds"] for attempt in safe)
    return min((attempt for attempt in safe if attempt["projected_production_seconds"] <= fastest * 1.03), key=lambda attempt: attempt["batch_size"])


def gpu_gate(config, work_dir, architecture):
    raw = load_landmark_config(config).raw
    # Exercise the actual joint-training branch, not only frozen-DINO warmup.
    # Overrides exist solely in this isolated probe config and are recorded.
    probe = work_dir / "joint_path_probe.yaml"
    write_json(probe, {"extends": str(config), "train": {"freeze_backbone_epochs": 0}, "model": {"continuous_recovery": {"mixing_warmup_epochs": 0}}})
    from keypoint_comparison1.landmark_detection.training import make_dataset
    train_count = len(make_dataset(raw, split="train", training=True))
    val_count = len(make_dataset(raw, split="val", training=False))
    attempts = []
    effective = int(raw["train"]["intended_effective_batch_size"])
    for batch in (4, 8, 16):
        attempt_started = time.perf_counter()
        try:
            result = run_training(config=probe, output_root=work_dir / f"batch_{batch}", epochs=2,
                                  max_train_samples=128, max_val_samples=64, batch_size=batch,
                                  accumulation=effective // batch, fixed_augmentation_epoch=0)
            if result["optimizer_steps_minimum"] < 4:
                raise ValueError("Probe must repeat optimizer steps after warmup")
            for row in result["rows"]:
                if not all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                    raise ValueError("Non-finite probe metric")
            if raw["train"].get("synchronized_timing", False):
                steps = [float(row["step"]) for row in result["rows"]]
                if len(steps) != 2 or steps[0] < 2 or steps[1] - steps[0] < 2:
                    raise ValueError("Probe did not complete repeated actual optimizer updates")
                if any(row.get("timing/synchronized") not in (True, "True") for row in result["rows"]):
                    raise ValueError("Probe timing was not CUDA synchronized")
                result["actual_optimizer_updates"] = int(steps[-1])
            measured = result["rows"][-1]
            train_seconds = float(measured["timing/train_seconds"])
            val_seconds = float(measured["timing/validation_seconds"])
            if train_seconds <= 0 or val_seconds <= 0:
                raise ValueError("Missing measured train/validation timing")
            result.update(status="safe", timing_warmup_epochs=1, measured_training_samples_per_second=128 / train_seconds,
                          measured_validation_seconds=val_seconds,
                          projected_production_seconds=(train_seconds / 128 * train_count + val_seconds / 64 * val_count) * raw["train"]["epochs"])
        except Exception as error:
            result = dict(status="unsafe", batch_size=batch, error=str(error),
                          elapsed_seconds=time.perf_counter() - attempt_started,
                          output_root=str(work_dir / f"batch_{batch}"))
        attempts.append(result)
        write_json(work_dir / "attempts.json", {"attempts": attempts})
    selected = select_plan(attempts)
    plan = dict(status="passed", architecture_id=architecture, candidates=[4, 8, 16], effective_batch_size=effective,
                selected_batch_size=selected["batch_size"], selected_gradient_accumulation_steps=effective // selected["batch_size"],
                production_config_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest(),
                selection_rule="shortest_projected_elapsed_within_3_percent_choose_smaller", attempts=attempts,
                isolated_probe_overrides={"freeze_backbone_epochs": 0, "mixing_warmup_epochs": 0},
                runtime_cuda_oom_fallback=True)
    write_json(work_dir / "batch_plan.json", plan)
    remove_checkpoints(work_dir)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["schema", "gpu_step", "overfit", "small_scale"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--architecture-id", required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.success_marker.unlink(missing_ok=True)
    started = time.time()
    try:
        if args.stage == "schema":
            details = schema_gate(args.config.resolve(), args.work_dir)
        elif args.stage == "gpu_step":
            details = gpu_gate(args.config.resolve(), args.work_dir, args.architecture_id)
        else:
            details = learning_gate(args.config.resolve(), args.work_dir, small_scale=args.stage == "small_scale")
        result = dict(status="passed", architecture=args.architecture_id, gate_stage=args.stage,
                      elapsed_seconds=time.time() - started, details=details)
        raw = load_landmark_config(args.config).raw
        result["production_config_sha256"] = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
        if "continuous_abc_" in args.architecture_id:
            from keypoint_comparison1.landmark_detection.recovery_acceptance import implementation_hashes
            result["implementation_sha256"] = implementation_hashes()
        write_json(args.work_dir / "report.json", result)
        write_json(args.success_marker, result)
    except Exception as error:
        write_json(args.work_dir / "report.json", dict(status="failed", error=str(error), gate_stage=args.stage))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
