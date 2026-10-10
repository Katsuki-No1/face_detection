from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_config
from src.data.landmark_occlusion import LandmarkOcclusionConfig

from .heads import FACTORIZED_STATE_HEAD, LEGACY_BINARY_STATE_HEAD
from .dataset import PairedOcclusionConfig
from .messenger_recovery import MessengerRecoveryConfig
from .continuous_recovery import ContinuousRecoveryConfig
from .model import VARIANTS
from .query_error_augmentation import QueryErrorAugmentationConfig


class LandmarkConfigError(ValueError):
    pass


LEGACY_KEYPOINT_TARGET_POLICY = {
    "coordinate_mask": "coco_v_gt_0",
    "visibility_target": "coco_v_gt_1",
    "visibility_loss_mask": "coco_v_gt_0",
}

KEYPOINT_TARGET_POLICY = {
    "coordinate_mask": "coco_v_gt_0",
    "presence_target": "coco_v_gt_0",
    "presence_loss_mask": "all_keypoint_slots",
    "visibility_target": "coco_v_gt_1",
    "visibility_loss_mask": "coco_v_gt_0",
}

TWO_STAGE_FACE_KEYPOINT_PIPELINE = (
    "full_image_face_ellipse",
    "face_crop_keypoint",
)
THREE_STAGE_HEAD_FACE_KEYPOINT_PIPELINE = (
    "full_image_head",
    "head_crop_face_ellipse",
    "face_crop_keypoint",
)


def configured_evaluation_pipeline_stages(
    config: dict[str, Any],
) -> tuple[str, ...]:
    """Return and validate the explicitly declared evaluation topology."""

    evaluation = config.get("evaluation", {})
    contract = evaluation.get("pipeline_contract")
    if contract is None:
        raise LandmarkConfigError(
            "evaluation.pipeline_contract must explicitly declare whether "
            "validation is a two-stage FACE/keypoint cascade or a three-stage "
            "HEAD/FACE/keypoint cascade"
        )
    if not isinstance(contract, dict):
        raise LandmarkConfigError(
            "evaluation.pipeline_contract must be a mapping"
        )
    if str(contract.get("type", "")) != "cascade":
        raise LandmarkConfigError(
            "Landmark evaluation pipeline_contract.type must be cascade"
        )
    stages = tuple(str(value) for value in contract.get("stages", ()))
    supported = {
        TWO_STAGE_FACE_KEYPOINT_PIPELINE,
        THREE_STAGE_HEAD_FACE_KEYPOINT_PIPELINE,
    }
    if stages not in supported:
        raise LandmarkConfigError(
            "evaluation.pipeline_contract.stages must explicitly declare "
            f"one of {sorted(supported)}, got {stages}"
        )
    predicted_head = evaluation.get("predicted_head")
    if stages == TWO_STAGE_FACE_KEYPOINT_PIPELINE and predicted_head is not None:
        raise LandmarkConfigError(
            "A two-stage FACE/keypoint pipeline must not configure an external "
            "evaluation.predicted_head"
        )
    if stages == THREE_STAGE_HEAD_FACE_KEYPOINT_PIPELINE and not isinstance(
        predicted_head, dict
    ):
        raise LandmarkConfigError(
            "A three-stage HEAD/FACE/keypoint pipeline requires "
            "evaluation.predicted_head"
        )
    return stages


def configured_state_head_mode(config: dict[str, Any]) -> str:
    return str(
        config.get("model", {}).get(
            "state_head_mode",
            LEGACY_BINARY_STATE_HEAD,
        )
    )


def configured_keypoint_target_policy(
    config: dict[str, Any],
) -> dict[str, str]:
    policy = config.get("data", {}).get("keypoint_target_policy")
    return dict(policy) if isinstance(policy, dict) else {}


@dataclass(frozen=True)
class LandmarkExperimentConfig:
    path: Path
    project_root: Path
    raw: dict[str, Any]

    @property
    def output_root(self) -> Path:
        return Path(self.raw["outputs"]["root"])

    @property
    def input_size(self) -> int:
        return int(self.raw["data"]["input_size"])


def _resolved(root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def load_landmark_config(path: str | Path, *, project_root: str | Path | None = None) -> LandmarkExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_config(config_path)
    root = Path(project_root).expanduser().resolve() if project_root else config_path.parents[2].resolve()
    if int(raw.get("schema_version", 0)) != 1:
        raise LandmarkConfigError("schema_version must be 1")
    raw["project_root"] = str(root)
    data = raw.get("data")
    model = raw.get("model")
    detector = raw.get("detector")
    outputs = raw.get("outputs")
    train = raw.get("train")
    if not all(isinstance(value, dict) for value in (data, model, detector, outputs, train)):
        raise LandmarkConfigError("data, model, detector, outputs, and train mappings are required")
    data["root"] = _resolved(root, data["root"])
    if data.get("face_context_caches"):
        data["face_context_caches"] = {key: _resolved(root, value)
                                      for key, value in data["face_context_caches"].items()}
    paired_materials = data.get("augmentation", {}).get("paired_occlusion", {}) or {}
    if paired_materials.get("material_manifest"):
        paired_materials["material_manifest"] = _resolved(root, paired_materials["material_manifest"])
    split_roots = data.get("image_roots")
    if isinstance(split_roots, dict):
        for split, value in tuple(split_roots.items()):
            split_roots[split] = _resolved(root, value)
    if data.get("ellipse_error_model"):
        data["ellipse_error_model"] = _resolved(root, data["ellipse_error_model"])
    if data.get("structure_head_annotations"):
        data["structure_head_annotations"] = _resolved(root, data["structure_head_annotations"])
    if data.get("validation_detector_crop_cache"):
        data["validation_detector_crop_cache"] = _resolved(
            root,
            data["validation_detector_crop_cache"],
        )
    for key, value in dict(data.get("annotations", {})).items():
        data["annotations"][key] = _resolved(root, value)
    for name, value in dict(data.get("evaluation_sets", {})).items():
        value["root"] = _resolved(root, value["root"])
        value["annotation"] = _resolved(root, value["annotation"])
    evaluation = raw.get("evaluation", {})
    if evaluation.get("baseline_aligned_output_root"):
        evaluation["baseline_aligned_output_root"] = _resolved(
            root,
            evaluation["baseline_aligned_output_root"],
        )
    backbone = model.get("backbone", {})
    for key in ("hub_repo", "weights", "initialize_from_deim_checkpoint"):
        if backbone.get(key):
            backbone[key] = _resolved(root, backbone[key])
    rgb_refinement = model.get("rgb_refinement", {}) or {}
    if rgb_refinement.get("initialize_from_deim_checkpoint"):
        rgb_refinement["initialize_from_deim_checkpoint"] = _resolved(
            root,
            rgb_refinement["initialize_from_deim_checkpoint"],
        )
    finetune_checkpoint = train.get("finetune_checkpoint")
    if isinstance(finetune_checkpoint, dict) and finetune_checkpoint.get("path"):
        finetune_checkpoint["path"] = _resolved(root, finetune_checkpoint["path"])
    detector["config"] = _resolved(root, detector["config"])
    detector["checkpoint"] = _resolved(root, detector["checkpoint"])
    outputs["root"] = _resolved(root, outputs["root"])
    validate_landmark_config(raw)
    return LandmarkExperimentConfig(path=config_path, project_root=root, raw=raw)


def validate_landmark_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    train = config["train"]
    for_cfg = model.get("face_overlap_robustness", {}) or {}
    if not isinstance(for_cfg, dict) or not isinstance(for_cfg.get("enabled", False), bool):
        raise LandmarkConfigError("face_overlap_robustness must be a mapping with boolean enabled")
    if for_cfg.get("enabled", False):
        if int(for_cfg.get("query_dim", 256)) <= 0:
            raise LandmarkConfigError("FOR query_dim must be positive")
        if config.get("evaluation", {}).get("pipeline_contract", {}).get("stages") != ["full_image_face_ellipse", "face_crop_keypoint"]:
            raise LandmarkConfigError("FOR requires HEAD-free full_image_face_ellipse -> face_crop_keypoint topology")
        if not all(data.get("face_context_caches", {}).get(split) for split in ("train", "val")):
            raise LandmarkConfigError("FOR requires explicit train/val face_context_caches")
        augmentation = data.get("augmentation", {})
        if (float(augmentation.get("occlusion_probability", 0)) > 0
                or (augmentation.get("landmark_guided_occlusion") or {}).get("probability", 0) > 0
                or (augmentation.get("paired_occlusion") or {}).get("probability", 0) > 0):
            raise LandmarkConfigError("FOR cached queries require crop-only synthetic occlusion disabled")
    configured_evaluation_pipeline_stages(config)
    input_size = int(data.get("input_size", 0))
    patch_size = int(model.get("backbone", {}).get("patch_size", 16))
    if input_size < 256 or input_size % patch_size != 0:
        raise LandmarkConfigError("Production landmark input_size must be >=256 and divisible by DINO patch_size")
    if int(model.get("num_landmarks", 0)) != 5:
        raise LandmarkConfigError("This comparison requires exactly five face landmarks")
    variants = set(model.get("variants", {}))
    if variants != set(VARIANTS):
        raise LandmarkConfigError(f"model.variants must define exactly {VARIANTS}")
    feature_pyramid = model.get("feature_pyramid", {}) or {}
    if not isinstance(feature_pyramid, dict):
        raise LandmarkConfigError("model.feature_pyramid must be a mapping")
    semantic_feature_levels = model.get("semantic_feature_levels", {}) or {}
    if not isinstance(semantic_feature_levels, dict):
        raise LandmarkConfigError(
            "model.semantic_feature_levels must be a mapping"
        )
    if bool(feature_pyramid.get("enabled", False)) and bool(
        semantic_feature_levels.get("enabled", False)
    ):
        raise LandmarkConfigError(
            "feature_pyramid and semantic_feature_levels are mutually exclusive"
        )
    kv_feature_pyramid = model.get("kv_feature_pyramid", {}) or {}
    if not isinstance(kv_feature_pyramid, dict):
        raise LandmarkConfigError("model.kv_feature_pyramid must be a mapping")
    if bool(kv_feature_pyramid.get("enabled", False)):
        if not bool(semantic_feature_levels.get("enabled", False)):
            raise LandmarkConfigError(
                "kv_feature_pyramid requires semantic_feature_levels"
            )
        if list(kv_feature_pyramid.get("output_strides", ())) != [16, 32, 64]:
            raise LandmarkConfigError(
                "kv_feature_pyramid.output_strides must be [16, 32, 64]"
            )
        if kv_feature_pyramid.get("semantic_fusion") != "learned_per_scale":
            raise LandmarkConfigError(
                "kv_feature_pyramid.semantic_fusion must be learned_per_scale"
            )
    rgb_refinement = model.get("rgb_refinement", {}) or {}
    if not isinstance(rgb_refinement, dict):
        raise LandmarkConfigError("model.rgb_refinement must be a mapping")
    if bool(rgb_refinement.get("enabled", False)):
        if not isinstance(rgb_refinement.get("freeze", False), bool):
            raise LandmarkConfigError("rgb_refinement.freeze must be boolean")
        if not bool(semantic_feature_levels.get("enabled", False)):
            raise LandmarkConfigError(
                "rgb_refinement requires semantic_feature_levels"
            )
        if bool(feature_pyramid.get("enabled", False)) or bool(
            kv_feature_pyramid.get("enabled", False)
        ):
            raise LandmarkConfigError(
                "rgb_refinement cannot be combined with a spatial or K/V pyramid"
            )
        if list(rgb_refinement.get("feature_sources", ())) != [
            "stride8",
            "stride4",
        ]:
            raise LandmarkConfigError(
                "rgb_refinement.feature_sources must be [stride8, stride4]"
            )
        if int(rgb_refinement.get("base_channels", 0)) != 32:
            raise LandmarkConfigError(
                "rgb_refinement.base_channels must be 32"
            )
        if int(rgb_refinement.get("projection_channels", 0)) != 128:
            raise LandmarkConfigError(
                "rgb_refinement.projection_channels must be 128"
            )
        if rgb_refinement.get("normalization") != "group_norm":
            raise LandmarkConfigError(
                "rgb_refinement.normalization must be group_norm"
            )
        if rgb_refinement.get("fusion") != "refinement_only":
            raise LandmarkConfigError(
                "rgb_refinement.fusion must be refinement_only"
            )
        if not rgb_refinement.get("initialize_from_deim_checkpoint"):
            raise LandmarkConfigError(
                "rgb_refinement requires initialize_from_deim_checkpoint"
            )
        checkpoint_sha256 = str(
            rgb_refinement.get("initialize_from_deim_checkpoint_sha256", "")
        )
        if len(checkpoint_sha256) != 64:
            raise LandmarkConfigError(
                "rgb_refinement requires the DEIM checkpoint SHA-256"
            )
        stages = model.get("variants", {}).get("query_refinement", {}).get(
            "stages", ()
        )
        expected_stages = [
            {
                "feature_source": "stride8",
                "radius_in_feature_cells": 2.0,
                "maximum_delta": 0.08,
                "samples_per_axis": 3,
            },
            {
                "feature_source": "stride4",
                "radius_in_feature_cells": 2.0,
                "maximum_delta": 0.04,
                "samples_per_axis": 3,
            },
        ]
        if list(stages) != expected_stages:
            raise LandmarkConfigError(
                "RGB refinement stages must be the locked stride8-to-stride4 "
                "coarse-to-fine contract"
            )
        if float(train.get("lr_cnn", 0.0)) <= 0.0:
            raise LandmarkConfigError(
                "Enabled rgb_refinement requires a positive train.lr_cnn"
            )
        query_refinement = model.get("variants", {}).get(
            "query_refinement", {}
        )
        if not isinstance(query_refinement.get("freeze", False), bool):
            raise LandmarkConfigError(
                "query_refinement.freeze must be boolean"
            )
    messenger_raw = model.get("messenger_recovery", {}) or {}
    if not isinstance(messenger_raw, dict):
        raise LandmarkConfigError("model.messenger_recovery must be a mapping")
    try:
        messenger = MessengerRecoveryConfig.from_mapping(messenger_raw)
        messenger.validate(
            embed_dim=int(model.get("backbone", {}).get("embed_dim", 384))
        )
    except ValueError as error:
        raise LandmarkConfigError(str(error)) from error
    try:
        continuous = ContinuousRecoveryConfig.from_mapping(
            model.get("continuous_recovery"), embed_dim=int(model.get("backbone", {}).get("embed_dim", 384))
        )
    except ValueError as error:
        raise LandmarkConfigError(str(error)) from error
    if continuous.enabled:
        if continuous.multi_scale or continuous.multi_level:
            backbone_config = model.get("backbone", {})
            if (backbone_config.get("name") != "dinov3_vits16"
                    or int(backbone_config.get("embed_dim", 384)) != 384
                    or int(backbone_config.get("patch_size", 16)) != 16
                    or int(config.get("data", {}).get("input_size", 256)) != 256):
                raise LandmarkConfigError("Recovery references require DINOv3 ViT-S/16, 384 channels, crop 256")
        if messenger.enabled:
            raise LandmarkConfigError("Legacy and continuous recovery are mutually exclusive")
        if configured_evaluation_pipeline_stages(config) != TWO_STAGE_FACE_KEYPOINT_PIPELINE:
            raise LandmarkConfigError("Continuous recovery requires HEAD-free FACE -> crop keypoint topology")
        if any(bool(value.get("enabled", False)) for value in (feature_pyramid, semantic_feature_levels, rgb_refinement)):
            raise LandmarkConfigError("Continuous recovery preserves the single-map Query Refinement")
        if float(train.get("lr_messenger", 0)) <= 0 or not train.get("finetune_checkpoint"):
            raise LandmarkConfigError("Continuous recovery requires lr_messenger and fixed teacher initialization")
        if not config.get("data", {}).get("augmentation", {}).get("paired_occlusion"):
            raise LandmarkConfigError("Continuous recovery requires paired clean/occluded training")
        if continuous.structure_enabled and not config.get("data", {}).get("structure_head_annotations"):
            raise LandmarkConfigError("Structure supervision requires canonical train Head annotations")
    if messenger.enabled:
        if bool(feature_pyramid.get("enabled", False)) or bool(
            semantic_feature_levels.get("enabled", False)
        ) or bool(rgb_refinement.get("enabled", False)):
            raise LandmarkConfigError(
                "Messenger v1 cannot be combined with feature pyramids or RGB refinement"
            )
        if float(train.get("lr_messenger", 0.0)) <= 0.0:
            raise LandmarkConfigError(
                "Enabled messenger_recovery requires positive train.lr_messenger"
            )
        loss = config.get("loss", {}) or {}
        if abs(float(loss.get("messenger_reconstruction_weight", -1.0)) - 0.10) > 1.0e-12:
            raise LandmarkConfigError(
                "Messenger v1 requires loss.messenger_reconstruction_weight=0.10"
            )
        if abs(float(loss.get("messenger_gate_weight", -1.0)) - 0.01) > 1.0e-12:
            raise LandmarkConfigError(
                "Messenger v1 requires loss.messenger_gate_weight=0.01"
            )
    if bool(feature_pyramid.get("enabled", False)):
        intermediate_layers = feature_pyramid.get("intermediate_layers")
        if (
            not isinstance(intermediate_layers, list)
            or len(intermediate_layers) != 3
            or len({int(index) for index in intermediate_layers}) != 3
        ):
            raise LandmarkConfigError(
                "Enabled feature_pyramid requires three unique intermediate_layers"
            )
        if [int(index) for index in intermediate_layers] != sorted(
            int(index) for index in intermediate_layers
        ):
            raise LandmarkConfigError(
                "feature_pyramid.intermediate_layers must be ordered shallow-to-deep"
            )
        if int(feature_pyramid.get("out_channels", 0)) <= 0:
            raise LandmarkConfigError(
                "Enabled feature_pyramid requires positive out_channels"
            )
        if list(feature_pyramid.get("output_strides", ())) != [8, 16, 32]:
            raise LandmarkConfigError(
                "feature_pyramid.output_strides must explicitly be [8, 16, 32]"
            )
    elif feature_pyramid.get("intermediate_layers"):
        raise LandmarkConfigError(
            "feature_pyramid.intermediate_layers requires enabled=true"
        )
    if bool(semantic_feature_levels.get("enabled", False)):
        intermediate_layers = semantic_feature_levels.get("intermediate_layers")
        if (
            not isinstance(intermediate_layers, list)
            or len(intermediate_layers) != 3
            or len({int(index) for index in intermediate_layers}) != 3
        ):
            raise LandmarkConfigError(
                "Enabled semantic_feature_levels requires three unique "
                "intermediate_layers"
            )
        if [int(index) for index in intermediate_layers] != sorted(
            int(index) for index in intermediate_layers
        ):
            raise LandmarkConfigError(
                "semantic_feature_levels.intermediate_layers must be ordered "
                "shallow-to-deep"
            )
        if int(semantic_feature_levels.get("output_stride", 0)) != patch_size:
            raise LandmarkConfigError(
                "semantic_feature_levels.output_stride must equal the DINO "
                "patch_size"
            )
        if semantic_feature_levels.get("fusion") != "decoder_memory_concat":
            raise LandmarkConfigError(
                "semantic_feature_levels.fusion must be decoder_memory_concat"
            )
    elif semantic_feature_levels.get("intermediate_layers"):
        raise LandmarkConfigError(
            "semantic_feature_levels.intermediate_layers requires enabled=true"
        )
    if train.get("production_mode") is not True:
        raise LandmarkConfigError("train.production_mode must be true")
    if int(train.get("epochs", 0)) < int(train.get("minimum_epochs", 100)):
        raise LandmarkConfigError("Production epochs are below train.minimum_epochs")
    early_stopping = train.get("early_stopping", {})
    if bool(early_stopping.get("enabled", True)) and int(early_stopping.get("minimum_epoch", 0)) < int(
        train.get("minimum_epochs", 100)
    ):
        raise LandmarkConfigError("early_stopping.minimum_epoch must not be below train.minimum_epochs")
    if float(train.get("fraction", 0.0)) != 1.0:
        raise LandmarkConfigError("Production training forbids dataset subsampling")
    if int(train.get("batch_size", 0)) <= 0 or int(train.get("gradient_accumulation_steps", 0)) <= 0:
        raise LandmarkConfigError("batch_size and gradient_accumulation_steps must be positive")
    checkpoint_selection = train.get("checkpoint_selection", {}) or {}
    if not isinstance(checkpoint_selection, dict):
        raise LandmarkConfigError("train.checkpoint_selection must be a mapping")
    if checkpoint_selection:
        if checkpoint_selection.get("direction") not in {"lower", "higher"}:
            raise LandmarkConfigError(
                "train.checkpoint_selection.direction must be lower or higher"
            )
        filename = str(checkpoint_selection.get("filename", ""))
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise LandmarkConfigError(
                "train.checkpoint_selection.filename must be a plain .pt filename"
            )
    if not bool(train.get("amp", False)):
        raise LandmarkConfigError("Production training requires AMP")
    if float(train.get("grad_scaler_init_scale", 65536.0)) <= 0.0:
        raise LandmarkConfigError("train.grad_scaler_init_scale must be positive")
    if int(train.get("grad_scaler_growth_interval", 2000)) <= 0:
        raise LandmarkConfigError("train.grad_scaler_growth_interval must be positive")
    if not bool(train.get("ema", {}).get("enabled", False)):
        raise LandmarkConfigError("Production training requires EMA")
    backbone_policy = str(train.get("backbone_policy", "staged_unfreeze"))
    if backbone_policy not in {"staged_unfreeze", "freeze"}:
        raise LandmarkConfigError(
            "train.backbone_policy must be 'staged_unfreeze' or 'freeze'"
        )
    if backbone_policy == "freeze" and model.get("backbone", {}).get("freeze") is not True:
        raise LandmarkConfigError(
            "train.backbone_policy=freeze requires model.backbone.freeze=true"
        )
    if not bool(data.get("leakage_audit", {}).get("enabled", False)):
        raise LandmarkConfigError("Production training requires data.leakage_audit.enabled=true")
    state_head_mode = configured_state_head_mode(config)
    if state_head_mode not in {
        LEGACY_BINARY_STATE_HEAD,
        FACTORIZED_STATE_HEAD,
    }:
        raise LandmarkConfigError(
            "model.state_head_mode must be either "
            f"{LEGACY_BINARY_STATE_HEAD!r} or {FACTORIZED_STATE_HEAD!r}"
        )
    expected_policy = (
        KEYPOINT_TARGET_POLICY
        if state_head_mode == FACTORIZED_STATE_HEAD
        else LEGACY_KEYPOINT_TARGET_POLICY
    )
    if data.get("keypoint_target_policy") != expected_policy:
        raise LandmarkConfigError(
            "data.keypoint_target_policy does not match "
            f"model.state_head_mode={state_head_mode!r}; expected "
            f"{expected_policy}"
        )
    if (
        state_head_mode == FACTORIZED_STATE_HEAD
        and data.get("include_missing_keypoint_instances") is not True
    ):
        raise LandmarkConfigError(
            "The factorized state head requires "
            "data.include_missing_keypoint_instances=true"
        )
    query_refinement = model.get("variants", {}).get("query_refinement", {})
    query_error_augmentation = query_refinement.get(
        "input_keypoint_error_augmentation"
    )
    if query_error_augmentation is not None:
        try:
            parsed_query_error_augmentation = QueryErrorAugmentationConfig.from_mapping(
                query_error_augmentation,
                num_landmarks=int(model["num_landmarks"]),
            )
        except ValueError as error:
            raise LandmarkConfigError(str(error)) from error
        if parsed_query_error_augmentation.enabled:
            if query_refinement.get("iterative_reference_points") is not True:
                raise LandmarkConfigError(
                    "input keypoint error augmentation requires "
                    "model.variants.query_refinement.iterative_reference_points=true"
                )
            if str(query_refinement.get("coordinate_update_mode")) != (
                "inverse_sigmoid_delta"
            ):
                raise LandmarkConfigError(
                    "input keypoint error augmentation requires "
                    "coordinate_update_mode=inverse_sigmoid_delta"
                )
    pretraining = data.get("pretraining", {})
    wider_crop_only_pretraining = (
        isinstance(pretraining, dict)
        and bool(pretraining.get("enabled", False))
        and str(pretraining.get("source", "")).startswith(
            "wider_face_retinaface_v1"
        )
        and str(pretraining.get("crop_geometry_provenance", ""))
        == "bbox_derived_circle_crop_only"
        and str(pretraining.get("state_supervision", ""))
        == "five_flag0_present_visible_only"
    )
    detector_crop_geometry = str(config["detector"].get("crop_geometry"))
    if detector_crop_geometry != "ellipse" and not (
        wider_crop_only_pretraining and detector_crop_geometry == "bbox"
    ):
        raise LandmarkConfigError(
            "detector.crop_geometry must be ellipse, except for declared "
            "WIDER bbox-derived pretraining/evaluation"
        )
    augmentation = data.get("augmentation", {})
    if str(augmentation.get("roi_geometry", "")) != "long_axis_square":
        raise LandmarkConfigError(
            "Production crops must use roi_geometry=long_axis_square"
        )
    if abs(float(augmentation.get("scale_factor", 0.0)) - 1.25) > 1.0e-8:
        raise LandmarkConfigError(
            "GT ellipse crop margin must remain scale_factor=1.25"
        )
    if not 0.0 < float(augmentation.get("scale_jitter", 0.0)) < 1.0:
        raise LandmarkConfigError(
            "Independent major/minor axis jitter must be between zero and one"
        )
    if float(augmentation.get("center_jitter", 0.0)) <= 0.0:
        raise LandmarkConfigError("GT ellipse center jitter must be positive")
    if float(augmentation.get("rotation_deg", 0.0)) <= 0.0:
        raise LandmarkConfigError("GT ellipse angle jitter must be positive")
    landmark_occlusion = augmentation.get("landmark_guided_occlusion")
    if landmark_occlusion is not None:
        if not isinstance(landmark_occlusion, dict):
            raise LandmarkConfigError(
                "data.augmentation.landmark_guided_occlusion must be a mapping"
            )
        try:
            LandmarkOcclusionConfig.from_mapping(landmark_occlusion)
        except ValueError as error:
            raise LandmarkConfigError(str(error)) from error
    paired_occlusion = augmentation.get("paired_occlusion")
    if paired_occlusion is not None:
        if not isinstance(paired_occlusion, dict):
            raise LandmarkConfigError(
                "data.augmentation.paired_occlusion must be a mapping"
            )
        if landmark_occlusion is not None:
            raise LandmarkConfigError(
                "paired_occlusion and landmark_guided_occlusion are mutually exclusive"
            )
        if float(augmentation.get("occlusion_probability", 0.0)) != 0.0:
            raise LandmarkConfigError(
                "paired_occlusion requires legacy occlusion_probability=0"
            )
        try:
            paired = PairedOcclusionConfig.from_mapping(paired_occlusion)
        except ValueError as error:
            raise LandmarkConfigError(str(error)) from error
        if abs(paired.probability - 0.30) > 1.0e-12:
            raise LandmarkConfigError(
                "Messenger experiment paired_occlusion.probability must be 0.30"
            )
    if messenger.enabled and paired_occlusion is None:
        raise LandmarkConfigError(
            "Enabled messenger_recovery requires paired_occlusion"
        )
    if not str(data.get("ellipse_error_model", "")).strip() and not wider_crop_only_pretraining:
        raise LandmarkConfigError(
            "Production training requires data.ellipse_error_model calibrated "
            "from the completed ellipse-DEIM"
        )
