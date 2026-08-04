from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from src.config_utils import load_yaml_config
from src.head_pose_geometry import (
    CanonicalHeadGeometryAdapter,
    HeadCropAugmentation,
    HeadCropBuilder,
    HeadPoseGeometryExtension,
    HeadPoseGeometryLoss,
    UncertaintyGatedGeometryFusion,
    build_pose_encoder,
)
from src.head_pose_geometry.losses import geometry_loss_weights_from_mapping
from src.models.dinov3_deimv2_ellipse import build_dinov3_deimv2_ellipse_detector

from .checkpoint import (
    initialize_pose_from_query_refinement,
    load_cooperative_checkpoint_strict,
    load_face_checkpoint_strict,
)
from .cascade import ExternalHeadCascadeRouter
from .model import (
    BidirectionalCooperativeDecoder,
    CooperativeCriterion,
    FaceKeypointCooperativeDETR,
    FacePoseAssociation,
    IndependentPoseQueryDecoder,
    KEYPOINT_NAMES,
    PoseHungarianMatcher,
)
from .sequential import (
    StrictSequentialCascade,
    validate_strict_sequential_config,
)
from .shared_head import SharedDINOHeadBranch


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return dict(config.get(name, {}) or {})


def _build_head_pose_geometry(
    model_config: Mapping[str, Any],
    *,
    pose_query_dimension: int,
) -> HeadPoseGeometryExtension | None:
    extension_cfg = _section(model_config, "head_pose_geometry")
    if not bool(extension_cfg.get("enabled", False)):
        return None
    pose_cfg = _section(extension_cfg, "pose_encoder")
    checkpoint = pose_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "model.head_pose_geometry.pose_encoder.checkpoint is required "
            "when the extension is enabled"
        )
    embedding_dimension = int(pose_cfg.get("embedding_dimension", 128))
    pose_encoder = build_pose_encoder(
        checkpoint=checkpoint,
        training_mode=str(pose_cfg.get("training_mode", "frozen")),
        embedding_dimension=embedding_dimension,
        allow_full_finetune=bool(
            pose_cfg.get("allow_full_finetune", False)
        ),
    )
    crop_cfg = _section(extension_cfg, "crop")
    augmentation_cfg = _section(crop_cfg, "augmentation")
    augmentation = HeadCropAugmentation(
        **{
            name: augmentation_cfg.get(
                name,
                getattr(HeadCropAugmentation(), name),
            )
            for name in HeadCropAugmentation.__dataclass_fields__
        }
    )
    crop_builder = HeadCropBuilder(
        output_size=crop_cfg.get("output_size", 224),
        context_scale=float(crop_cfg.get("context_scale", 1.35)),
        pad_value=float(crop_cfg.get("pad_value", 0.0)),
        augmentation=augmentation,
    )
    geometry_cfg = _section(extension_cfg, "geometry")
    geometry = (
        CanonicalHeadGeometryAdapter(
            embedding_dimension,
            hidden_dimension=int(geometry_cfg.get("hidden_dimension", 128)),
            contour_points=int(geometry_cfg.get("contour_points", 48)),
            shape_dimension=int(geometry_cfg.get("shape_dimension", 0)),
            shape_limit=float(geometry_cfg.get("shape_limit", 0.15)),
            minimum_scale=float(geometry_cfg.get("minimum_scale", 1.0e-4)),
            visibility_sharpness=float(
                geometry_cfg.get("visibility_sharpness", 10.0)
            ),
        )
        if bool(geometry_cfg.get("enabled", False))
        else None
    )
    fusion_cfg = _section(extension_cfg, "fusion")
    fusion = (
        UncertaintyGatedGeometryFusion(
            embedding_dimension,
            hidden_dimension=int(fusion_cfg.get("hidden_dimension", 128)),
            keypoints=5,
        )
        if geometry is not None and bool(fusion_cfg.get("enabled", False))
        else None
    )
    criterion = (
        HeadPoseGeometryLoss(
            weights=geometry_loss_weights_from_mapping(
                _section(extension_cfg, "loss_weights")
            ),
            minimum_scale=float(geometry_cfg.get("minimum_scale", 1.0e-4)),
            maximum_scale=float(geometry_cfg.get("maximum_scale", 1.5)),
        )
        if geometry is not None
        else None
    )
    return HeadPoseGeometryExtension(
        crop_builder=crop_builder,
        pose_encoder=pose_encoder,
        geometry_adapter=geometry,
        fusion=fusion,
        criterion=criterion,
        pose_query_dimension=(
            pose_query_dimension
            if bool(extension_cfg.get("inject_pose_queries", True))
            else None
        ),
        query_injection_temperature=float(
            extension_cfg.get("query_injection_temperature", 0.20)
        ),
        use_valid_pixel_and_border_inputs=bool(
            extension_cfg.get("use_valid_pixel_and_border_inputs", True)
        ),
        augmentation_consistency_enabled=bool(
            extension_cfg.get("augmentation_consistency_enabled", True)
        ),
    )


def build_face_keypoint_cooperative_detr(
    config: Mapping[str, Any],
    *,
    face_detector: nn.Module | None = None,
    initialize_pretrained: bool = True,
) -> FaceKeypointCooperativeDETR:
    cfg = dict(config)
    validate_strict_sequential_config(cfg)
    model_cfg = dict(cfg.get("model", cfg))
    face_cfg = _section(model_cfg, "face_detector")
    load_info: dict[str, Any] = {}
    if face_detector is None:
        config_path = Path(str(face_cfg["config"]))
        if not config_path.is_file():
            raise FileNotFoundError(f"Face detector config was not found: {config_path}")
        detector_config = load_yaml_config(config_path)
        checkpoint = face_cfg.get("checkpoint")
        face_initialization = str(
            face_cfg.get(
                "initialization",
                "fine_tuned" if checkpoint else "pretrained",
            )
        ).strip().lower().replace("-", "_")
        if face_initialization not in {"pretrained", "fine_tuned", "random"}:
            raise ValueError(
                "model.face_detector.initialization must be one of "
                "'pretrained', 'fine_tuned', or 'random'"
            )
        if face_initialization == "pretrained" and checkpoint:
            raise ValueError(
                "A face-fine-tuned checkpoint cannot be supplied when "
                "model.face_detector.initialization='pretrained'"
            )
        if face_initialization == "fine_tuned" and not checkpoint:
            raise ValueError(
                "model.face_detector.initialization='fine_tuned' requires "
                "model.face_detector.checkpoint"
            )
        # When a complete face checkpoint is supplied below, avoid first
        # performing the ellipse builder's partial base-DEIM initialization.
        # This keeps the initialization contract unambiguous: the final face
        # state comes from one strict, fully architecture-compatible load.
        if (
            face_initialization in {"fine_tuned", "random"}
            or not initialize_pretrained
        ):
            detector_config.pop("init_checkpoint", None)
            detector_config.pop("pretrained_checkpoint", None)
            detector_model = detector_config.get("model", {})
            if isinstance(detector_model, dict):
                detector_model.pop("init_checkpoint", None)
                detector_model.pop("pretrained_checkpoint", None)
        face_detector = build_dinov3_deimv2_ellipse_detector(detector_config)
        if initialize_pretrained:
            if face_initialization == "fine_tuned":
                report = load_face_checkpoint_strict(
                    face_detector,
                    checkpoint,
                    prefer_ema=bool(face_cfg.get("prefer_ema", True)),
                )
                load_info["face"] = {
                    **report,
                    "initialization": "fine_tuned",
                }
            elif face_initialization == "pretrained":
                report = dict(
                    getattr(face_detector, "pretrained_load_info", {}) or {}
                )
                if not report:
                    raise RuntimeError(
                        "The face detector was configured for pretrained "
                        "initialization, but its detector config did not load "
                        "a pretrained checkpoint"
                    )
                load_info["face"] = {
                    **report,
                    "initialization": "pretrained",
                }
            else:
                load_info["face"] = {
                    "initialization": "random",
                    "checkpoint": None,
                }
    encoded_channels = list(
        model_cfg.get(
            "encoded_channels",
            [
                int(getattr(face_detector.decoder, "hidden_dim", 224)),
            ]
            * 3,
        )
    )
    face_dimension = int(
        getattr(face_detector.decoder, "hidden_dim", encoded_channels[0])
    )
    pose_cfg = _section(model_cfg, "pose_branch")
    pose_dimension = int(pose_cfg.get("dimension", 256))
    pose_decoder = IndependentPoseQueryDecoder(
        encoded_channels,
        dimension=pose_dimension,
        num_queries=int(pose_cfg.get("num_queries", 100)),
        num_keypoints=int(pose_cfg.get("num_keypoints", 5)),
        layers=int(pose_cfg.get("layers", 4)),
        heads=int(pose_cfg.get("heads", 8)),
        feedforward=int(pose_cfg.get("feedforward", 1024)),
        deformable_points=pose_cfg.get("deformable_points", [4, 4, 4]),
        dropout=float(pose_cfg.get("dropout", 0.0)),
        auxiliary_loss=bool(pose_cfg.get("auxiliary_loss", True)),
        local_refinement=list(pose_cfg.get("local_refinement", [])),
        state_head_mode=str(
            pose_cfg.get("state_head_mode", "legacy_binary")
        ),
    )
    query_checkpoint = pose_cfg.get("checkpoint")
    pose_initialization = str(
        pose_cfg.get(
            "initialization",
            "fine_tuned" if query_checkpoint else "random",
        )
    ).strip().lower().replace("-", "_")
    if pose_initialization not in {"fine_tuned", "random"}:
        raise ValueError(
            "model.pose_branch.initialization must be either "
            "'fine_tuned' or 'random'"
        )
    if pose_initialization == "random" and query_checkpoint:
        raise ValueError(
            "A fine-tuned pose checkpoint cannot be supplied when "
            "model.pose_branch.initialization='random'"
        )
    if pose_initialization == "fine_tuned" and not query_checkpoint:
        raise ValueError(
            "model.pose_branch.initialization='fine_tuned' requires "
            "model.pose_branch.checkpoint"
        )
    if initialize_pretrained:
        if pose_initialization == "fine_tuned":
            report = initialize_pose_from_query_refinement(
                pose_decoder,
                query_checkpoint,
                prefer_ema=bool(pose_cfg.get("prefer_ema", True)),
                minimum_eligible_fraction=float(
                    pose_cfg.get("minimum_eligible_load_fraction", 0.65)
                ),
            )
            load_info["pose"] = {
                **report,
                "initialization": "fine_tuned",
            }
        else:
            load_info["pose"] = {
                "initialization": "random",
                "checkpoint": None,
                "reason": (
                    "no architecture-compatible public pretrained pose head"
                ),
            }
    association_cfg = _section(model_cfg, "association")
    association = FacePoseAssociation(
        face_dimension,
        pose_dimension,
        appearance_dimension=int(
            association_cfg.get("appearance_dimension", 128)
        ),
        hidden_dimension=int(association_cfg.get("hidden_dimension", 256)),
        inside_temperature=float(association_cfg.get("inside_temperature", 12.0)),
    )
    cooperative_cfg = _section(model_cfg, "cooperative_decoder")
    cooperative = BidirectionalCooperativeDecoder(
        face_dimension,
        pose_dimension,
        association,
        layers=int(cooperative_cfg.get("layers", 3)),
        heads=int(cooperative_cfg.get("heads", 8)),
        levels=len(encoded_channels),
        deformable_points=cooperative_cfg.get(
            "deformable_points",
            [4, 4, 4],
        ),
        feedforward=int(cooperative_cfg.get("feedforward", 1024)),
        topk_associations=int(cooperative_cfg.get("topk_associations", 4)),
        dropout=float(cooperative_cfg.get("dropout", 0.0)),
        drop_face_to_pose=float(
            cooperative_cfg.get("drop_face_to_pose", 0.10)
        ),
        drop_pose_to_face=float(
            cooperative_cfg.get("drop_pose_to_face", 0.10)
        ),
        auxiliary_loss=bool(cooperative_cfg.get("auxiliary_loss", True)),
    )
    matcher_cfg = _section(model_cfg, "pose_matcher")
    matcher = PoseHungarianMatcher(
        cost_class=float(matcher_cfg.get("cost_class", 2.0)),
        cost_box=float(matcher_cfg.get("cost_box", 2.0)),
        cost_keypoint=float(matcher_cfg.get("cost_keypoint", 5.0)),
    )
    loss_cfg = _section(cfg, "loss")
    criterion = CooperativeCriterion(
        matcher,
        weights=dict(loss_cfg.get("weights", {})),
        auxiliary_weight=float(loss_cfg.get("auxiliary_weight", 0.5)),
    )
    inference_cfg = _section(cfg, "inference")
    head_pose_geometry = _build_head_pose_geometry(
        model_cfg,
        pose_query_dimension=pose_dimension,
    )
    cascade_cfg = _section(model_cfg, "external_head_cascade")
    external_head_cascade = (
        ExternalHeadCascadeRouter(
            hidden_dimension=int(cascade_cfg.get("hidden_dimension", 64)),
            face_to_head_threshold=float(
                cascade_cfg.get("face_to_head_threshold", 0.50)
            ),
            loss_weight=float(cascade_cfg.get("loss_weight", 2.0)),
            hard_negative_ratio=int(
                cascade_cfg.get("hard_negative_ratio", 10)
            ),
        )
        if bool(cascade_cfg.get("enabled", False))
        else None
    )
    shared_head_cfg = _section(model_cfg, "shared_head_branch")
    shared_head_branch = (
        SharedDINOHeadBranch(
            in_channels=int(shared_head_cfg.get("in_channels", 384)),
            category_id=int(shared_head_cfg.get("category_id", 1)),
            score_threshold=float(
                shared_head_cfg.get("score_threshold", 0.001)
            ),
            max_detections=int(
                shared_head_cfg.get("max_detections", 100)
            ),
            loss_weight=float(shared_head_cfg.get("loss_weight", 1.0)),
            teacher_force_targets=bool(
                shared_head_cfg.get("teacher_force_targets", False)
            ),
        )
        if bool(shared_head_cfg.get("enabled", False))
        else None
    )
    sequential_cfg = _section(model_cfg, "strict_sequential_cascade")
    strict_sequential_cascade = (
        StrictSequentialCascade(sequential_cfg)
        if bool(sequential_cfg.get("enabled", False))
        else None
    )
    model = FaceKeypointCooperativeDETR(
        face_detector,
        pose_decoder=pose_decoder,
        association=association,
        cooperative_decoder=cooperative,
        criterion=criterion,
        head_pose_geometry=head_pose_geometry,
        external_head_cascade=external_head_cascade,
        shared_head_branch=shared_head_branch,
        strict_sequential_cascade=strict_sequential_cascade,
        maximum_face_queries=int(model_cfg.get("maximum_face_queries", 100)),
        face_score_threshold=float(
            inference_cfg.get("face_score_threshold", 0.05)
        ),
        pose_score_threshold=float(
            inference_cfg.get("pose_score_threshold", 0.05)
        ),
        association_threshold=float(
            inference_cfg.get("association_threshold", 0.50)
        ),
        detections_per_image=int(
            inference_cfg.get("detections_per_image", 100)
        ),
    )
    if head_pose_geometry is not None:
        load_info["head_pose_geometry"] = {
            "pose": dict(head_pose_geometry.pose_encoder.pretrained_load_info),
            "geometry_enabled": head_pose_geometry.geometry_enabled,
            "fusion_enabled": head_pose_geometry.fusion_enabled,
        }
    if external_head_cascade is not None:
        load_info["external_head_cascade"] = {
            "initialization": "containment_prior_plus_random_residual",
            "checkpoint": None,
        }
    if shared_head_branch is not None:
        load_info["shared_head_branch"] = {
            "initialization": "random_sfp_plus_compact_codino",
            "checkpoint": None,
            "backbone": "shared_face_dinov3_vits16plus",
        }
    if strict_sequential_cascade is not None:
        load_info["strict_sequential_cascade"] = {
            "initialization": "parameter_free_roi_feature_routing",
            "checkpoint": None,
            "execution_order": ["head", "face", "keypoint"],
            "training_roi_source": "ground_truth_augmented",
            "evaluation_roi_source": "predicted",
        }
    if initialize_pretrained:
        load_info["association"] = {
            "initialization": "random",
            "checkpoint": None,
        }
        load_info["cooperative_decoder"] = {
            "initialization": "random",
            "checkpoint": None,
        }
    model.pretrained_load_info = load_info
    return model


__all__ = [
    "BidirectionalCooperativeDecoder",
    "CooperativeCriterion",
    "ExternalHeadCascadeRouter",
    "FaceKeypointCooperativeDETR",
    "FacePoseAssociation",
    "IndependentPoseQueryDecoder",
    "KEYPOINT_NAMES",
    "PoseHungarianMatcher",
    "SharedDINOHeadBranch",
    "StrictSequentialCascade",
    "build_face_keypoint_cooperative_detr",
    "initialize_pose_from_query_refinement",
    "load_cooperative_checkpoint_strict",
    "load_face_checkpoint_strict",
    "validate_strict_sequential_config",
]
