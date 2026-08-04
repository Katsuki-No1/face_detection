from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from face_keypoint_cooperative_common import (  # noqa: E402
    apply_training_stage_loss,
    autocast_context,
    build_dataset,
    build_loader,
    build_model,
    configure_device_math,
    enforce_training_stage_modes,
    finite_loss_dict,
    initialize_shared_head_checkpoint,
    load_cooperative_config,
    move_batch,
    resolve_path,
    set_dataset_epoch,
    write_json,
)
from src.models.face_keypoint_cooperative.cascade import (  # noqa: E402
    detections_from_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run architecture, checkpoint, gradient, numerical, and runtime audits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--initialize-shared-head-from", type=Path, default=None)
    parser.add_argument(
        "--initialize-shared-dino-from-head-checkpoint",
        action="store_true",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    parser.add_argument("--augmentation-audit-samples", type=int, default=100)
    return parser.parse_args()


def audit_ellipse_border_crop(
    config: dict[str, Any],
    *,
    maximum_samples: int,
) -> dict[str, Any]:
    augmentation = config.get("data", {}).get("augmentation", {})
    probability = (
        float(augmentation.get("ellipse_border_crop", 0.0) or 0.0)
        if isinstance(augmentation, dict)
        else 0.0
    )
    if probability <= 0.0:
        return {"enabled": False, "passed": True}
    dataset = build_dataset(
        config,
        "train",
        training=True,
        maximum_samples=max(1, int(maximum_samples)),
    )
    set_dataset_epoch(dataset, 0)
    counts = {"center": 0, "contour": 0, "not_applied": 0}
    violations: list[dict[str, Any]] = []
    examples = []
    for index in range(len(dataset)):
        _image, target = dataset[index]
        debug = target.get("augmentation_debug", {}).get("ellipse_border_crop")
        if not isinstance(debug, dict):
            counts["not_applied"] += 1
            continue
        mode = str(debug.get("mode"))
        if mode not in {"center", "contour"}:
            violations.append({"index": index, "reason": f"unknown mode {mode!r}"})
            continue
        counts[mode] += 1
        cx, cy, major, minor, theta = (
            float(value) for value in debug["selected_ellipse"]
        )
        height = width = int(config["data"].get("image_size", 960))
        center_outside = cx < 0.0 or cx >= width or cy < 0.0 or cy >= height
        extent_x = math.sqrt(
            (major * math.cos(theta)) ** 2 + (minor * math.sin(theta)) ** 2
        )
        extent_y = math.sqrt(
            (major * math.sin(theta)) ** 2 + (minor * math.cos(theta)) ** 2
        )
        contour_outside = (
            cx - extent_x < 0.0
            or cy - extent_y < 0.0
            or cx + extent_x > width
            or cy + extent_y > height
        )
        face_count = int(target["ellipses"].shape[0])
        head_count = int(target["head_boxes"].shape[0])
        mapping = target["geometry_target_indices"]
        alignment_valid = (
            int(target["boxes"].shape[0]) == face_count
            and int(target["keypoints"].shape[0]) == face_count
            and int(target["head_confidence"].shape[0]) == head_count
            and int(mapping.shape[0]) == head_count
            and len(target["head_sources"]) == head_count
            and bool(((mapping >= -1) & (mapping < max(1, face_count))).all())
        )
        if (
            center_outside != (mode == "center")
            or not contour_outside
            or not alignment_valid
        ):
            violations.append(
                {
                    "index": index,
                    "mode": mode,
                    "center_outside": center_outside,
                    "contour_outside": contour_outside,
                    "alignment_valid": alignment_valid,
                }
            )
        if len(examples) < 4:
            examples.append(
                {
                    "index": index,
                    "mode": mode,
                    "side": debug.get("side"),
                    "center": [cx, cy],
                    "crop_xyxy": debug.get("crop_xyxy"),
                }
            )
    applied = counts["center"] + counts["contour"]
    passed = applied > 0 and counts["center"] > 0 and counts["contour"] > 0 and not violations
    return {
        "enabled": True,
        "passed": passed,
        "configured_probability": probability,
        "samples": len(dataset),
        "counts": counts,
        "violations": violations,
        "examples": examples,
    }


def recursively_finite(value: Any, path: str = "output") -> list[str]:
    failures = []
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        if not torch.isfinite(value).all():
            failures.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            failures.extend(recursively_finite(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(recursively_finite(item, f"{path}[{index}]"))
    return failures


def gradient_norm(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    total_tensors = sum(parameter.requires_grad for parameter in parameters)
    return {
        "trainable_tensors": total_tensors,
        "gradient_tensors": len(gradients),
        "none_tensors": total_tensors - len(gradients),
        "norm": (
            float(torch.stack(gradients).norm())
            if gradients
            else 0.0
        ),
    }


def target_with_instances(
    target: dict[str, torch.Tensor],
    count: int,
) -> dict[str, torch.Tensor]:
    """Return exactly ``count`` aligned instances for edge-case auditing."""
    available = int(target["boxes"].shape[0])
    if count > 0 and available == 0:
        raise RuntimeError(
            f"Cannot construct the requested {count}-face audit case from "
            "an empty source target."
        )
    indices = (
        torch.arange(count, device=target["boxes"].device) % available
        if count > 0
        else torch.empty(
            0,
            device=target["boxes"].device,
            dtype=torch.long,
        )
    )
    head_fields = {
        "head_boxes",
        "head_confidence",
        "head_sources",
        "geometry_target_indices",
        "head_detection_boxes",
        "head_detection_labels",
        "head_detection_to_face_indices",
    }
    output = {}
    for key, value in target.items():
        if key == "image_id":
            output[key] = value
        elif key in head_fields:
            continue
        elif isinstance(value, torch.Tensor) and value.ndim > 0:
            output[key] = value[indices].clone()
        else:
            output[key] = value
    if "head_boxes" in target:
        output["head_boxes"] = output["boxes"].clone()
        output["head_confidence"] = torch.ones(
            count,
            device=output["boxes"].device,
            dtype=output["boxes"].dtype,
        )
        output["head_sources"] = ["audit_predicted"] * count
        output["geometry_target_indices"] = torch.arange(
            count,
            device=output["boxes"].device,
            dtype=torch.long,
        )
        output["head_detection_boxes"] = output["head_boxes"].clone()
        output["head_detection_labels"] = torch.ones(
            count,
            device=output["boxes"].device,
            dtype=torch.long,
        )
        output["head_detection_to_face_indices"] = torch.arange(
            count,
            device=output["boxes"].device,
            dtype=torch.long,
        )
    return output


def selected_named_parameters(
    model: torch.nn.Module,
    fragments: tuple[str, ...],
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if any(fragment in name for fragment in fragments)
    ]


def main() -> int:
    args = parse_args()
    config = load_cooperative_config(args.config)
    config["train"]["num_workers"] = 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    configure_device_math(device, config["train"])
    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else Path(config["output"]["root"]) / "audit.json"
    )
    success_path = output_path.parent / "audit_success.json"
    success_path.unlink(missing_ok=True)
    augmentation_audit = audit_ellipse_border_crop(
        config,
        maximum_samples=int(args.augmentation_audit_samples),
    )
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    model = build_model(config, device=device, checkpoint=checkpoint)
    if args.initialize_shared_head_from is not None and checkpoint is None:
        initialize_shared_head_checkpoint(
            model,
            resolve_path(args.initialize_shared_head_from),
            prefer_ema=True,
            load_shared_dino=bool(
                args.initialize_shared_dino_from_head_checkpoint
            ),
        )
    loader, _ = build_loader(
        config,
        "train",
        training=False,
        distributed=False,
        rank=0,
        world_size=1,
        batch_size=int(args.batch_size),
        maximum_samples=int(args.max_samples),
    )
    images, targets = move_batch(next(iter(loader)), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    enforce_training_stage_modes(model, config)
    model.zero_grad(set_to_none=True)
    strict_sequential = model.strict_sequential_cascade is not None
    audit_branch_override = (True, False) if strict_sequential else (True, True)
    with autocast_context(device, config["train"]):
        output = model(
            images,
            targets,
            return_attention=True,
            branch_override=audit_branch_override,
        )
        apply_training_stage_loss(output, config)
    losses = finite_loss_dict(output)
    finite_failures = recursively_finite(output)
    output["loss_total"].backward()
    groups = model.parameter_groups()
    gradient_report = {
        name: gradient_norm(parameters)
        for name, parameters in groups.items()
    }
    gradient_report["face_output_head"] = gradient_norm(
        selected_named_parameters(
            model.face_detector,
            (
                "dec_score_head",
                "dec_bbox_head",
                "ellipse_head",
            ),
        )
    )
    gradient_report["keypoint_output_head"] = gradient_norm(
        selected_named_parameters(
            model.pose_decoder,
            (
                "coordinate_heads",
                "pose_confidence_heads",
                "keypoint_confidence_heads",
                "visibility_head",
            ),
        )
    )
    if model.pose_decoder.state_head_mode == "factorized_presence_visibility":
        gradient_report["keypoint_presence_refinement"] = gradient_norm(
            selected_named_parameters(
                model.pose_decoder,
                ("presence_delta",),
            )
        )
    candidate_gradient_groups = [
        "backbone",
        "sta_neck",
        "shared_encoder",
        "face_decoder",
        "keypoint_decoder",
        "association",
        "cooperative_decoder",
        "face_output_head",
        "keypoint_output_head",
    ]
    if model.pose_decoder.state_head_mode == "factorized_presence_visibility":
        candidate_gradient_groups.append("keypoint_presence_refinement")
    optional_gradient_modules = {
        "shared_head_branch": model.shared_head_branch,
        "external_head_cascade": model.external_head_cascade,
        "head_pose_geometry": model.head_pose_geometry,
    }
    candidate_gradient_groups.extend(
        name
        for name, module in optional_gradient_modules.items()
        if module is not None
    )
    required_gradient_groups = [
        name
        for name in candidate_gradient_groups
        if gradient_report[name]["trainable_tensors"] > 0
    ]
    missing_gradients = [
        name
        for name in required_gradient_groups
        if gradient_report[name]["gradient_tensors"] == 0
        or gradient_report[name]["none_tensors"] != 0
        or gradient_report[name]["norm"] == 0.0
    ]

    refined = output["refined"]
    gate_face = refined["gate_keypoint_to_face"]
    gate_pose = refined["gate_face_to_pose"]
    numerical = {
        "finite_failures": finite_failures,
        "losses_finite": not any(
            not math_value_is_finite(value)
            for key, value in losses.items()
            if key.startswith("loss")
        ),
        "keypoint_to_face_gate_min": float(gate_face.detach().amin()),
        "keypoint_to_face_gate_max": float(gate_face.detach().amax()),
        "face_to_pose_gate_min": float(gate_pose.detach().amin()),
        "face_to_pose_gate_max": float(gate_pose.detach().amax()),
        "face_boxes_min": float(refined["pred_face_boxes"].detach().amin()),
        "face_boxes_max": float(refined["pred_face_boxes"].detach().amax()),
        "keypoints_min": float(refined["pred_keypoints"].detach().amin()),
        "keypoints_max": float(refined["pred_keypoints"].detach().amax()),
    }

    model.train()
    enforce_training_stage_modes(model, config)
    branch_cases = {}
    configured_branch_cases = (
        {"strict_face_to_pose_only": (True, False)}
        if strict_sequential
        else {
            "both": (True, True),
            "face_to_pose_only": (True, False),
            "pose_to_face_only": (False, True),
            "neither": (False, False),
        }
    )
    with torch.no_grad():
        for name, override in configured_branch_cases.items():
            with autocast_context(device, config["train"]):
                case = model(
                    images,
                    targets,
                    branch_override=override,
                )
                apply_training_stage_loss(case, config)
            branch_cases[name] = {
                "face_to_pose_enabled": tensor_values(
                    case["refined"]["branch_face_to_pose_enabled"]
                ),
                "pose_to_face_enabled": tensor_values(
                    case["refined"]["branch_pose_to_face_enabled"]
                ),
                "loss_total": float(case["loss_total"]),
            }
    reverse_refinement_rejected = not strict_sequential
    if strict_sequential:
        try:
            with torch.no_grad(), autocast_context(device, config["train"]):
                model(images, targets, branch_override=(True, True))
        except ValueError as error:
            reverse_refinement_rejected = "forbids keypoint-to-FACE" in str(error)

    edge_cases: dict[str, Any] = {}
    with torch.no_grad():
        for name, count in {
            "zero_faces": 0,
            "one_face": 1,
            "two_faces": 2,
        }.items():
            case_targets = [
                target_with_instances(target, count)
                for target in targets
            ]
            with autocast_context(device, config["train"]):
                case = model(
                    images,
                    case_targets,
                    branch_override=audit_branch_override,
                )
                apply_training_stage_loss(case, config)
            edge_cases[name] = {
                "loss_total": float(case["loss_total"]),
                "finite": not recursively_finite(case),
                "face_queries": int(
                    case["independent_face"]["pred_boxes"].shape[1]
                ),
                "pose_queries": int(
                    case["independent_pose"]["pred_keypoints"].shape[1]
                ),
            }

    model.eval()
    audit_head_detections = (
        detections_from_targets(
            targets,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
        )
        if (
            (
                model.external_head_cascade is not None
                or model.head_pose_geometry is not None
            )
            and model.shared_head_branch is None
        )
        else None
    )
    with torch.inference_mode(), autocast_context(device, config["train"]):
        validation_output = model(images, targets)
        apply_training_stage_loss(validation_output, config)
    validation_finite_failures = recursively_finite(
        validation_output,
        "validation_output",
    )
    validation_loss_path = {
        "finite_failures": validation_finite_failures,
        "loss_total": float(validation_output["loss_total"]),
        "model_eval_mode_restored": not model.training,
        "face_decoder_eval_mode_restored": (
            not model.face_detector.decoder.training
        ),
    }
    with torch.inference_mode():
        for _ in range(int(args.warmup_iterations)):
            with autocast_context(device, config["train"]):
                model(images, head_detections=audit_head_detections)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(int(args.timed_iterations)):
            started = time.perf_counter()
            with autocast_context(device, config["train"]):
                model(images, head_detections=audit_head_detections)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - started) * 1000.0)

    pose_config = config["model"]["pose_branch"]
    cooperative_config = config["model"]["cooperative_decoder"]
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "by_group": {
            name: sum(parameter.numel() for parameter in values)
            for name, values in groups.items()
        },
    }
    verification = config.get("verification", {})
    dino_modules = [
        name
        for name, _module in model.named_modules()
        if name.endswith("backbone.dino.backbone")
    ]
    architecture_contract_failures: list[str] = []
    expected_dino_modules = verification.get(
        "expected_full_image_dino_modules"
    )
    if (
        expected_dino_modules is not None
        and len(dino_modules) != int(expected_dino_modules)
    ):
        architecture_contract_failures.append(
            "full_image_dino_modules:"
            f"{len(dino_modules)}!={int(expected_dino_modules)}"
        )
    for key, group_name in (
        ("expected_shared_dino_parameters", "backbone"),
        ("expected_shared_head_branch_parameters", "shared_head_branch"),
    ):
        expected = verification.get(key)
        actual = parameters["by_group"].get(group_name, 0)
        if expected is not None and actual != int(expected):
            architecture_contract_failures.append(
                f"{group_name}_parameters:{actual}!={int(expected)}"
            )
    sequential_output = output.get("strict_sequential_cascade")
    expected_order = verification.get("expected_execution_order")
    if expected_order is not None:
        actual_order = (
            list(sequential_output.get("execution_order", ()))
            if isinstance(sequential_output, dict)
            else []
        )
        if actual_order != list(expected_order):
            architecture_contract_failures.append(
                f"execution_order:{actual_order}!={list(expected_order)}"
            )
    expected_reverse = verification.get("expected_pose_to_face_refinement")
    actual_reverse = bool(
        torch.as_tensor(
            output["refined"]["branch_pose_to_face_enabled"]
        ).bool().any()
    )
    if expected_reverse is not None and actual_reverse != bool(expected_reverse):
        architecture_contract_failures.append(
            f"pose_to_face_refinement:{actual_reverse}!={bool(expected_reverse)}"
        )
    if strict_sequential:
        if not reverse_refinement_rejected:
            architecture_contract_failures.append(
                "reverse_refinement_override_was_not_rejected"
            )
        for stage_name in ("head_to_face", "face_to_keypoint"):
            stage = (
                sequential_output.get(stage_name, {})
                if isinstance(sequential_output, dict)
                else {}
            )
            coverage = stage.get("feature_coverage")
            expected_source = "ground_truth_augmented"
            if not isinstance(coverage, torch.Tensor) or not torch.isfinite(
                coverage
            ).all():
                architecture_contract_failures.append(
                    f"{stage_name}_feature_coverage_invalid"
                )
            if stage.get("source") != expected_source:
                architecture_contract_failures.append(
                    f"{stage_name}_source:{stage.get('source')}!={expected_source}"
                )
    sequential_report = None
    if isinstance(sequential_output, dict):
        sequential_report = {
            "execution_order": list(sequential_output.get("execution_order", ())),
            "unidirectional_refinement": bool(
                sequential_output.get("unidirectional_refinement", False)
            ),
        }
        for stage_name in ("head_to_face", "face_to_keypoint"):
            stage = sequential_output.get(stage_name, {})
            if not isinstance(stage, dict):
                continue
            sequential_report[stage_name] = {
                "source": stage.get("source"),
                "roi_count": tensor_values(stage["roi_count"]),
                "feature_coverage": (
                    stage["feature_coverage"].detach().float().cpu().tolist()
                ),
            }
    report = {
        "status": "passed"
        if not (
            finite_failures
            or missing_gradients
            or validation_finite_failures
            or not validation_loss_path["model_eval_mode_restored"]
            or not validation_loss_path["face_decoder_eval_mode_restored"]
            or numerical["keypoint_to_face_gate_min"] < 0.0
            or numerical["keypoint_to_face_gate_max"] > 1.0
            or numerical["face_to_pose_gate_min"] < 0.0
            or numerical["face_to_pose_gate_max"] > 1.0
            or not augmentation_audit["passed"]
            or architecture_contract_failures
        )
        else "failed",
        "architecture": {
            "shared_backbone": True,
            "full_image_dino_modules": dino_modules,
            "full_image_dino_execution_count": len(dino_modules),
            "shared_head_branch_enabled": (
                model.shared_head_branch is not None
            ),
            "architecture_contract_failures": architecture_contract_failures,
            "shared_hybrid_encoder": True,
            "face_branch_class": type(model.face_detector).__name__,
            "face_branch_raw_auxiliary_outputs": len(
                output["independent_face"]["raw_output"].get("aux_outputs", [])
            ),
            "pose_is_face_proposal_independent": not strict_sequential,
            "strict_sequential_cascade_enabled": strict_sequential,
            "strict_sequential_cascade": sequential_report,
            "reverse_refinement_rejected": reverse_refinement_rejected,
            "pose_state_head_mode": model.pose_decoder.state_head_mode,
            "pose_presence_refinement_stages": sum(
                hasattr(stage, "presence_delta")
                for stage in model.pose_decoder.refinement_stages
            ),
            "pose_decoder_layers": len(model.pose_decoder.layers),
            "pose_auxiliary_outputs": len(
                output["independent_pose"]["auxiliary_outputs"]
            ),
            "pose_uses_high_resolution_feature": tuple(
                output["independent_pose"]["projected_features"][0].shape
            ),
            "cooperative_decoder_layers": len(
                model.cooperative_decoder.layers
            ),
            "association_shape": tuple(
                output["refined"]["association_logits"].shape
            ),
            "branch_cases": branch_cases,
            "independent_and_refined_losses": all(
                key in losses
                for key in (
                    "loss_face_independent",
                    "loss_keypoint_independent",
                    "loss_face_refined",
                    "loss_keypoint_refined",
                    "loss_association",
                    "loss_visibility",
                )
            ),
            "padding_mask_supported": True,
            "no_object_supervision": True,
            "validation_loss_path": validation_loss_path,
            "keypoint_order": [
                "left_eye",
                "right_eye",
                "nose_tip",
                "left_mouth_corner",
                "right_mouth_corner",
            ],
        },
        "checkpoint": model.pretrained_load_info,
        "gradients": gradient_report,
        "missing_gradient_groups": missing_gradients,
        "numerical": numerical,
        "edge_cases": edge_cases,
        "augmentation": {
            "ellipse_border_crop": augmentation_audit,
        },
        "parameters": parameters,
        "performance": {
            "batch_size": int(images.shape[0]),
            "input_size": list(images.shape[-2:]),
            "amp": bool(config["train"].get("amp", True)),
            "amp_dtype": str(config["train"].get("amp_dtype", "bfloat16")),
            "inference_time_ms_mean": statistics.mean(times),
            "inference_time_ms_median": statistics.median(times),
            "inference_time_ms_samples": times,
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                if device.type == "cuda"
                else 0.0
            ),
            "pose_queries": int(pose_config.get("num_queries", 100)),
            "face_queries": int(config["model"].get("maximum_face_queries", 100)),
            "pose_decoder_layers": int(pose_config.get("layers", 4)),
            "cooperative_decoder_layers": int(
                cooperative_config.get("layers", 3)
            ),
            "hidden_dimension": int(pose_config.get("dimension", 256)),
            "association_topk": int(
                cooperative_config.get("topk_associations", 4)
            ),
        },
        "losses": losses,
    }
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError(f"Audit failed; see {output_path}")
    write_json(
        success_path,
        {
            "status": "passed",
            "audit": str(output_path),
            "ellipse_border_crop": augmentation_audit,
        },
    )
    return 0


def math_value_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def tensor_values(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten()]


if __name__ == "__main__":
    raise SystemExit(main())
