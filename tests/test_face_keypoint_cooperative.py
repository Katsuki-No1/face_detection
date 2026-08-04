from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from src.config_utils import load_yaml_config
from src.head_pose_geometry.crops import HeadCropBuilder
from src.head_pose_geometry.extension import HeadPoseGeometryExtension
from src.head_pose_geometry.fusion import UncertaintyGatedGeometryFusion
from src.head_pose_geometry.geometry import CanonicalHeadGeometryAdapter
from src.head_pose_geometry.heads import HeadDetection
from src.head_pose_geometry.head_crop_dataset import (
    crop_tensor_to_square,
    square_head_crop_box,
)
from src.head_pose_geometry.losses import HeadPoseGeometryLoss
from src.head_pose_geometry.pose_encoder import PoseEncoderAdapter
from src.models.face_keypoint_cooperative import (
    ExternalHeadCascadeRouter,
    FaceKeypointCooperativeDETR,
    FacePoseAssociation,
    PoseHungarianMatcher,
    StrictSequentialCascade,
    build_face_keypoint_cooperative_detr,
    initialize_pose_from_query_refinement,
    load_cooperative_checkpoint_strict,
    load_face_checkpoint_strict,
    validate_strict_sequential_config,
)
from src.models.face_keypoint_cooperative.shared_head import (
    SharedDINOHeadBranch,
)
from src.models.face_keypoint_cooperative.checkpoint import (
    initialize_shared_dino_from_head_codino,
)
from tools.eval_face_keypoint_cooperative import KeypointAccumulator
from tools.face_keypoint_cooperative_common import (
    CooperativeJointAugmentation,
    HeadCentricEllipseBorderCropAugmentation,
    ModelEMA,
    apply_training_stage_loss,
    build_dataset,
    build_optimizer,
    initialize_stage_checkpoint,
    load_cooperative_config,
    resume_checkpoint,
    save_checkpoint,
    set_dataset_epoch,
    training_stage_policy,
)
from tools.train_face_keypoint_cooperative import (
    BatchPlan,
    EpochLossAccumulator,
    FaceDetectionMetricAccumulator,
    append_history,
    cooperative_wandb_run_id,
    configured_batch_plans,
    overfit_branch_override,
    read_metric_history,
    select_batch_plan,
    step_optimizer_with_grad_scaler,
    wandb_metric_row,
    write_metric_history,
)

COOPERATIVE_CONFIG_PATH = Path(
    "configs/face_keypoint_cooperative/"
    "face_keypoint_cooperative_detr_l_960.yaml"
)
PUBLIC_COCO_CHECKPOINT = (
    "checkpoint/DEIMv2/deimv2_dinov3_l_coco.safetensors"
)


def test_shared_head_branch_has_no_backbone_or_unused_sfp_p6() -> None:
    branch = SharedDINOHeadBranch(max_detections=20)

    assert not hasattr(branch, "backbone")
    assert not hasattr(branch.sfp, "p6")
    assert sum(parameter.numel() for parameter in branch.parameters()) == 7_814_676


def test_causal_face_and_keypoint_stage_contracts_are_disjoint() -> None:
    face = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_face_stage.yaml"
    )
    keypoint = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_keypoint_stage.yaml"
    )
    face_policy = training_stage_policy(face)
    keypoint_policy = training_stage_policy(keypoint)
    assert face["experiment_objective"]["type"] == "cascade_comparison"
    assert keypoint["experiment_objective"]["type"] == "cascade_comparison"
    assert face["experiment_objective"]["final_performance_claim"] is False
    assert keypoint["experiment_objective"]["joint_fine_tuning"] is False
    assert face_policy == {
        "name": "face",
        "trainable_groups": [
            "sta_neck",
            "shared_encoder",
            "face_decoder",
            "external_head_cascade",
        ],
        "loss_weights": {
            "loss_face_independent": 1.0,
            "loss_external_head_routing": 1.0,
        },
    }
    assert keypoint_policy == {
        "name": "keypoint",
        "trainable_groups": [
            "keypoint_decoder",
            "association",
            "cooperative_decoder",
        ],
        "loss_weights": {
            "loss_keypoint_independent": 10.0,
            "loss_association": 2.0,
            "loss_keypoint_refined": 10.0,
            "loss_visibility": 1.0,
        },
    }
    assert set(face_policy["trainable_groups"]).isdisjoint(
        keypoint_policy["trainable_groups"]
    )
    assert "backbone" not in face_policy["trainable_groups"]
    assert "shared_head_branch" not in face_policy["trainable_groups"]
    assert "face_decoder" not in keypoint_policy["trainable_groups"]


def test_small_scale_cooperative_subset_covers_every_source_video() -> None:
    config = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_face_stage.yaml"
    )
    dataset = build_dataset(
        config,
        "train",
        training=True,
        maximum_samples=128,
    )

    report = dataset.representative_subset_report
    assert report["strategy"] == "source_video_balanced_even_spacing"
    assert report["available_source_groups"] == 44
    assert report["selected_source_groups"] == 44
    assert report["selected_samples"] == 128


def test_stage_loss_excludes_frozen_branch_objectives() -> None:
    config = {
        "experiment_objective": {
            "type": "cascade_comparison",
            "joint_fine_tuning": False,
            "final_performance_claim": False,
        },
        "train": {
            "stage": {
                "name": "face",
                "trainable_groups": ["face_decoder"],
                "loss_weights": {
                    "loss_face_independent": 1.0,
                    "loss_external_head_routing": 2.0,
                },
            }
        }
    }
    output = {
        "loss_total": torch.tensor(100.0),
        "loss_face_independent": torch.tensor(3.0),
        "loss_external_head_routing": torch.tensor(4.0),
        "loss_keypoint_independent": torch.tensor(90.0),
    }
    total = apply_training_stage_loss(output, config)
    assert total.item() == pytest.approx(11.0)
    assert output["loss_total"] is total


def test_stagewise_training_rejects_final_performance_claim() -> None:
    config = {
        "experiment_objective": {
            "type": "final_performance",
            "joint_fine_tuning": True,
            "final_performance_claim": True,
        },
        "train": {
            "stage": {
                "name": "face",
                "trainable_groups": ["face_decoder"],
                "loss_weights": {"loss_face_independent": 1.0},
            }
        },
    }
    with pytest.raises(ValueError, match="cascade_comparison"):
        training_stage_policy(config)


def test_finetuned_head_dino_transfers_into_shared_backbone(tmp_path: Path) -> None:
    target = nn.Linear(3, 2)
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_bias = torch.tensor([7.0, 8.0])
    checkpoint = tmp_path / "head.pth"
    torch.save(
        {
            "ema": {
                "backbone.model.weight": expected_weight,
                "backbone.model.bias": expected_bias,
            }
        },
        checkpoint,
    )

    report = initialize_shared_dino_from_head_codino(target, checkpoint)

    assert report["loaded_tensors"] == 2
    assert torch.equal(target.weight, expected_weight)
    assert torch.equal(target.bias, expected_bias)


def test_head_crop_contract_keeps_square_context_and_outside_padding() -> None:
    crop = square_head_crop_box((2.0, 3.0, 6.0, 11.0), context_scale=1.20)
    assert crop == pytest.approx((-0.8, 2.2, 8.8, 11.8))
    image = torch.ones((3, 12, 8), dtype=torch.float32)
    sampled = crop_tensor_to_square(image, crop, output_size=24)
    assert sampled.shape == (3, 24, 24)
    assert sampled[:, :, 0].mean() < 1.0
    assert sampled[:, 8:16, 8:16].mean() == pytest.approx(1.0)


def test_head_crop_cooperative_config_uses_shared_predicted_head_cache() -> None:
    config = load_cooperative_config(
        "configs/face_keypoint_cooperative/head_crop_cooperative_detr_l_960.yaml"
    )
    assert config["model"]["shared_head_branch"]["enabled"] is False
    assert "strict_sequential_cascade" not in config["model"]
    assert config["data"]["head_crop"] == {
        "enabled": True,
        "require_predicted_heads": True,
        "context_scale": 1.20,
        "output_size": 960,
        "pad_value": 0.0,
        "evaluation_batch_size": 4,
    }
    assert all(
        "shared_head_codino_vits16plus_960_balanced_811/head_cache" in path
        for path in config["data"]["predicted_head_caches"].values()
    )


def _strict_sequential_config() -> dict[str, Any]:
    return {
        "model": {
            "shared_head_branch": {
                "enabled": True,
                "teacher_force_targets": True,
            },
            "external_head_cascade": {"enabled": True},
            "strict_sequential_cascade": {
                "enabled": True,
                "training_head_source": "ground_truth",
                "training_face_source": "ground_truth",
                "evaluation_head_source": "predicted",
                "evaluation_face_source": "predicted",
                "enforce_unidirectional_refinement": True,
                "head_to_face": {
                    "context_scale": 1.25,
                    "square": True,
                    "center_jitter": 0.08,
                    "log_scale_jitter": 0.12,
                },
                "face_to_keypoint": {
                    "context_scale": 1.25,
                    "square": True,
                    "center_jitter": 0.08,
                    "log_scale_jitter": 0.12,
                },
            },
        },
        "data": {"head_crop": {"enabled": False}},
        "verification": {
            "expected_execution_order": ["head", "face", "keypoint"],
            "expected_pose_to_face_refinement": False,
        },
    }


def test_strict_sequential_config_rejects_parallel_or_predicted_training() -> None:
    config = _strict_sequential_config()
    validate_strict_sequential_config(config)

    config["model"]["strict_sequential_cascade"][
        "enforce_unidirectional_refinement"
    ] = False
    with pytest.raises(ValueError, match="unidirectional"):
        validate_strict_sequential_config(config)

    config = _strict_sequential_config()
    config["model"]["strict_sequential_cascade"][
        "training_face_source"
    ] = "predicted"
    with pytest.raises(ValueError, match="training_face_source"):
        validate_strict_sequential_config(config)

    config = _strict_sequential_config()
    config["data"]["head_crop"]["enabled"] = True
    with pytest.raises(ValueError, match="pixel HEAD-crop cooperative"):
        validate_strict_sequential_config(config)

    config = _strict_sequential_config()
    config["verification"]["expected_execution_order"] = [
        "head",
        "keypoint",
        "face",
    ]
    with pytest.raises(ValueError, match="expected_execution_order"):
        validate_strict_sequential_config(config)


def test_strict_sequential_roi_features_are_causally_conditioned() -> None:
    strict_cfg = _strict_sequential_config()["model"][
        "strict_sequential_cascade"
    ]
    cascade = StrictSequentialCascade(strict_cfg)
    features = [torch.ones((1, 4, 8, 8)), torch.ones((1, 4, 4, 4))]
    heads = [
        [
            HeadDetection(
                box_xyxy=(16.0, 16.0, 48.0, 48.0),
                confidence=0.9,
                instance_id=0,
                source="predicted",
            )
        ]
    ]

    face_features, head_stage = cascade.head_conditioned_features(
        features,
        heads,
        image_size=(64, 64),
        training=False,
    )
    assert head_stage["source"] == "predicted"
    assert head_stage["roi_count"].tolist() == [1]
    assert 0.0 < float(head_stage["feature_coverage"].mean()) < 1.0
    assert torch.count_nonzero(face_features[0]) < face_features[0].numel()

    no_face_features, face_stage = cascade.face_conditioned_features(
        features,
        [torch.empty((0, 4))],
        training=False,
    )
    assert face_stage["roi_count"].tolist() == [0]
    assert all(torch.count_nonzero(feature) == 0 for feature in no_face_features)

    keypoint_features, face_stage = cascade.face_conditioned_features(
        features,
        [torch.tensor([[0.40, 0.40, 0.60, 0.60]])],
        training=False,
    )
    assert face_stage["source"] == "predicted"
    assert 0.0 < float(face_stage["feature_coverage"].mean()) < 1.0
    assert not torch.equal(keypoint_features[0], face_features[0])


def test_strict_sequential_training_uses_expanded_jittered_gt_rois() -> None:
    strict_cfg = _strict_sequential_config()["model"][
        "strict_sequential_cascade"
    ]
    cascade = StrictSequentialCascade(strict_cfg)
    gt_box = torch.tensor([[0.25, 0.30, 0.65, 0.50]])

    expected_unjittered = torch.tensor([[0.20, 0.15, 0.70, 0.65]])
    for stage in (cascade.head_to_face, cascade.face_to_keypoint):
        assert stage["context_scale"] == pytest.approx(1.25)
        unjittered = cascade._augment_boxes(
            gt_box,
            stage,
            training=False,
        )
        assert torch.allclose(unjittered, expected_unjittered)

        torch.manual_seed(42)
        jittered = cascade._augment_boxes(
            gt_box,
            stage,
            training=True,
        )
        assert not torch.allclose(jittered, unjittered)
        jittered_size = jittered[:, 2:] - jittered[:, :2]
        assert bool((jittered_size > 0.40).all())


def test_overfit_branch_contract_matches_strict_sequential_direction(
    model: FaceKeypointCooperativeDETR,
) -> None:
    assert overfit_branch_override(model) == (True, True)

    strict_config = _strict_sequential_config()["model"][
        "strict_sequential_cascade"
    ]
    model.strict_sequential_cascade = StrictSequentialCascade(strict_config)
    assert overfit_branch_override(model) == (True, False)


class _Matcher(nn.Module):
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        del epoch
        matches = []
        for target in targets:
            count = min(outputs["pred_boxes"].shape[1], target["boxes"].shape[0])
            indices = torch.arange(
                count,
                device=outputs["pred_boxes"].device,
                dtype=torch.long,
            )
            matches.append((indices, indices))
        return {"indices": matches}


class _FaceCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = _Matcher()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "aux_outputs" in outputs
        matches = self.matcher(outputs, targets, epoch=epoch)["indices"]
        object_terms = []
        box_terms = []
        zero = outputs["pred_logits"].sum() * 0.0
        for batch_index, ((source, destination), target) in enumerate(
            zip(matches, targets, strict=True)
        ):
            labels = torch.zeros_like(outputs["pred_logits"][batch_index])
            labels[source] = 1.0
            object_terms.append(
                F.binary_cross_entropy_with_logits(
                    outputs["pred_logits"][batch_index],
                    labels,
                )
            )
            if source.numel():
                box_terms.append(
                    F.l1_loss(
                        outputs["pred_boxes"][batch_index, source],
                        target["boxes"][destination],
                    )
                )
        return {
            "loss_mal": torch.stack(object_terms).mean(),
            "loss_bbox": torch.stack(box_terms).mean() if box_terms else zero,
        }


class _FaceLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.projection(value).tanh()


class _FaceInnerDecoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FaceLayer(dimension), _FaceLayer(dimension)]
        )


class _FaceDecoder(nn.Module):
    def __init__(self, dimension: int = 32, queries: int = 8) -> None:
        super().__init__()
        self.hidden_dim = dimension
        self.queries = nn.Parameter(torch.randn(queries, dimension) * 0.02)
        self.decoder = _FaceInnerDecoder(dimension)
        self.score = nn.Linear(dimension, 1)
        self.box = nn.Linear(dimension, 4)
        self.ellipse = nn.Linear(dimension, 6)

    def forward(
        self,
        features: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        del targets
        batch = features[0].shape[0]
        hidden = (
            self.queries.unsqueeze(0).expand(batch, -1, -1)
            + features[0].mean(dim=(2, 3)).unsqueeze(1)
        )
        auxiliary = []
        for layer in self.decoder.layers:
            hidden = layer(hidden)
            auxiliary.append(
                {
                    "pred_logits": self.score(hidden),
                    "pred_boxes": self.box(hidden).sigmoid(),
                }
            )
        boxes = self.box(hidden).sigmoid()
        raw_ellipse = self.ellipse(hidden)
        axes = 0.01 + 0.2 * raw_ellipse[..., 2:4].sigmoid()
        angle = 0.5 * torch.atan2(
            raw_ellipse[..., 5],
            raw_ellipse[..., 4],
        )
        output = {
            "pred_logits": self.score(hidden),
            "pred_boxes": boxes,
            "pred_ellipses": torch.cat(
                (boxes[..., :2], axes, angle.unsqueeze(-1)),
                dim=-1,
            ),
        }
        if self.training:
            output["aux_outputs"] = auxiliary[:-1]
        return output


class _Backbone(nn.Module):
    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.dino = nn.Conv2d(3, dimension, kernel_size=3, padding=1)
        self.sta = nn.Conv2d(3, dimension, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = F.gelu(self.dino(images) + self.sta(images))
        return (
            F.avg_pool2d(feature, 4),
            F.avg_pool2d(feature, 8),
            F.avg_pool2d(feature, 16),
        )

    def forward_with_semantic(
        self,
        images: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        features = self(images)
        return features, features


class _TinySharedHead(nn.Module):
    loss_weight = 1.0

    def forward(
        self,
        semantic_feature: torch.Tensor,
        *,
        image_size: tuple[int, int],
        targets: list[dict[str, Any]] | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del padding_mask
        height, width = image_size
        rows = []
        for batch_index in range(semantic_feature.shape[0]):
            if targets is None:
                boxes = semantic_feature.new_tensor(
                    [[0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height]]
                )
            else:
                boxes = torch.as_tensor(
                    targets[batch_index]["head_boxes"],
                    device=semantic_feature.device,
                    dtype=semantic_feature.dtype,
                ).reshape(-1, 4)
            rows.append(
                {
                    "boxes": boxes,
                    "scores": torch.ones(
                        boxes.shape[0],
                        device=boxes.device,
                        dtype=boxes.dtype,
                    ),
                    "labels": torch.zeros(
                        boxes.shape[0],
                        device=boxes.device,
                        dtype=torch.long,
                    ),
                }
            )
        detections = [
            [
                HeadDetection(
                    box_xyxy=tuple(float(value) for value in box),
                    confidence=1.0,
                    instance_id=index,
                    source="tiny_shared_head",
                )
                for index, box in enumerate(row["boxes"].detach().cpu())
            ]
            for row in rows
        ]
        zero = semantic_feature.sum() * 0.0
        return {
            "predictions": rows,
            "detector_predictions": rows,
            "detections": detections,
            "losses": {"loss_total": zero},
        }


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(32, 32, kernel_size=1) for _ in range(3)]
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            F.gelu(projection(feature))
            for projection, feature in zip(
                self.projections,
                features,
                strict=True,
            )
        ]


class _FaceDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.encoder = _Encoder()
        self.decoder = _FaceDecoder()
        self.criterion = _FaceCriterion()
        self.category_to_label = {2: 0}
        self.current_epoch = 0

    def _targets_to_official(
        self,
        targets: list[dict[str, torch.Tensor]],
        image_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, torch.Tensor]]:
        height, width = image_size
        converted = []
        for target in targets:
            boxes = target["boxes"].to(device=device, dtype=dtype)
            x1, y1, x2, y2 = boxes.unbind(-1)
            normalized_boxes = torch.stack(
                (
                    (x1 + x2) * 0.5 / width,
                    (y1 + y2) * 0.5 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ),
                dim=-1,
            )
            ellipses = target["ellipses"].to(device=device, dtype=dtype).clone()
            if ellipses.numel():
                ellipses[..., :2] /= ellipses.new_tensor((width, height))
                ellipses[..., 2:4] /= max(width, height)
            converted.append(
                {
                    "boxes": normalized_boxes,
                    "ellipses": ellipses,
                    "labels": torch.zeros(
                        boxes.shape[0],
                        dtype=torch.long,
                        device=device,
                    ),
                }
            )
        return converted

    @staticmethod
    def _format_losses(
        losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {"loss_total": sum(losses.values()), **losses}


class _TinyPoseBackbone(nn.Module):
    feature_dimension = 8

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.layer4 = nn.Identity()
        self.linear_reg = nn.Linear(4, 6)
        self.projection = nn.Linear(4, self.feature_dimension)

    def forward_raw(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.features(image).flatten(1)
        return self.linear_reg(value), self.projection(value)


def _head_pose_extension(*, geometry: bool) -> HeadPoseGeometryExtension:
    encoder = PoseEncoderAdapter(
        _TinyPoseBackbone(),  # type: ignore[arg-type]
        training_mode="domain_adapter",
        embedding_dimension=16,
    )
    adapter = CanonicalHeadGeometryAdapter(16) if geometry else None
    return HeadPoseGeometryExtension(
        crop_builder=HeadCropBuilder(output_size=32),
        pose_encoder=encoder,
        geometry_adapter=adapter,
        fusion=(
            UncertaintyGatedGeometryFusion(16, hidden_dimension=32)
            if geometry
            else None
        ),
        criterion=HeadPoseGeometryLoss() if geometry else None,
        pose_query_dimension=32,
    )


def _config() -> dict[str, Any]:
    return {
        "model": {
            "encoded_channels": [32, 32, 32],
            "maximum_face_queries": 6,
            "pose_branch": {
                "dimension": 32,
                "num_queries": 6,
                "layers": 2,
                "heads": 4,
                "feedforward": 64,
                "deformable_points": [2, 2, 2],
                "auxiliary_loss": True,
                "local_refinement": [
                    {
                        "radius_in_feature_cells": 1.0,
                        "maximum_delta": 0.03,
                        "samples_per_axis": 3,
                    }
                ],
            },
            "association": {
                "appearance_dimension": 16,
                "hidden_dimension": 32,
            },
            "cooperative_decoder": {
                "layers": 2,
                "heads": 4,
                "feedforward": 64,
                "deformable_points": [2, 2, 2],
                "topk_associations": 2,
                "drop_face_to_pose": 0.5,
                "drop_pose_to_face": 0.5,
            },
            "pose_matcher": {},
        },
        "loss": {
            "weights": {
                "keypoint_independent": 1.0,
                "keypoint_refined": 1.0,
            }
        },
        "inference": {
            "face_score_threshold": 0.001,
            "pose_score_threshold": 0.001,
        },
    }


def _target(faces: int = 1) -> dict[str, torch.Tensor]:
    boxes = torch.tensor(
        [
            [8.0 + 20 * index, 10.0, 30.0 + 20 * index, 42.0]
            for index in range(faces)
        ],
        dtype=torch.float32,
    ).reshape(-1, 4)
    ellipses = torch.tensor(
        [
            [19.0 + 20 * index, 26.0, 14.0, 10.0, 0.1]
            for index in range(faces)
        ],
        dtype=torch.float32,
    ).reshape(-1, 5)
    keypoints = torch.tensor(
        [
            [
                [0.22 + 0.3 * index, 0.30],
                [0.36 + 0.3 * index, 0.30],
                [0.29 + 0.3 * index, 0.42],
                [0.24 + 0.3 * index, 0.56],
                [0.35 + 0.3 * index, 0.56],
            ]
            for index in range(faces)
        ],
        dtype=torch.float32,
    ).reshape(-1, 5, 2)
    return {
        "boxes": boxes,
        "ellipses": ellipses,
        "labels": torch.full((faces,), 2, dtype=torch.long),
        "keypoints": keypoints,
        "labeled": torch.ones((faces, 5), dtype=torch.bool),
        "visible": torch.tensor(
            [[True, True, True, False, True]] * faces,
            dtype=torch.bool,
        ),
    }


def _geometry_target(faces: int = 1) -> dict[str, Any]:
    target: dict[str, Any] = _target(faces)
    target.update(
        {
            "head_boxes": target["boxes"].clone(),
            "head_detection_boxes": target["boxes"].clone(),
            "head_detection_labels": torch.zeros(faces, dtype=torch.long),
            "head_detection_to_face_indices": torch.arange(
                faces, dtype=torch.long
            ),
            "head_confidence": torch.ones(faces),
            "head_sources": ["ground_truth"] * faces,
            "rotation_matrix": torch.eye(3).repeat(faces, 1, 1),
            "pose_supervision_mask": torch.tensor(
                [index == 0 for index in range(faces)]
            ),
            "pose_weight": torch.ones(faces),
        }
    )
    return target


@pytest.fixture()
def model() -> FaceKeypointCooperativeDETR:
    return build_face_keypoint_cooperative_detr(
        _config(),
        face_detector=_FaceDetector(),
        initialize_pretrained=False,
    )


def test_production_config_uses_public_pretrained_weights_only() -> None:
    config = load_yaml_config(COOPERATIVE_CONFIG_PATH)
    face_config = config["model"]["face_detector"]
    pose_config = config["model"]["pose_branch"]

    assert face_config["initialization"] == "pretrained"
    assert "checkpoint" not in face_config
    detector_config = load_yaml_config(Path(face_config["config"]))
    initialization = detector_config["model"]["init_checkpoint"]
    assert initialization["path"] == PUBLIC_COCO_CHECKPOINT
    assert initialization["prefer_ema"] is False
    assert initialization["skip_class_heads"] is True

    assert pose_config["initialization"] == "random"
    assert "checkpoint" not in pose_config
    assert "prefer_ema" not in pose_config
    assert "minimum_eligible_load_fraction" not in pose_config


def test_default_pose_query_refinement_preserves_legacy_state_contract(
    model: FaceKeypointCooperativeDETR,
) -> None:
    assert model.pose_decoder.state_head_mode == "legacy_binary"
    assert not any(
        "presence_delta" in key
        for key in model.pose_decoder.state_dict()
    )


def test_factorized_pose_query_refinement_updates_both_state_logits() -> None:
    config = _config()
    config["model"]["pose_branch"]["state_head_mode"] = (
        "factorized_presence_visibility"
    )
    factorized = build_face_keypoint_cooperative_detr(
        config,
        face_detector=_FaceDetector(),
        initialize_pretrained=False,
    )
    output = factorized(
        torch.rand(1, 3, 64, 64),
        [_target()],
        branch_override=(True, True),
    )
    independent = output["independent_pose"]
    expected = torch.stack(
        (
            1.0 - independent["pred_keypoint_logits"].sigmoid(),
            independent["pred_keypoint_logits"].sigmoid()
            * (1.0 - independent["pred_visibility_logits"].sigmoid()),
            independent["pred_keypoint_logits"].sigmoid()
            * independent["pred_visibility_logits"].sigmoid(),
        ),
        dim=-1,
    )
    assert torch.allclose(
        independent["pred_state_probabilities"],
        expected,
    )
    assert torch.allclose(
        output["refined"]["pred_state_probabilities"].sum(dim=-1),
        torch.ones_like(output["refined"]["pred_keypoint_logits"]),
    )
    output["loss_total"].backward()
    presence_delta_parameters = [
        parameter
        for name, parameter in factorized.pose_decoder.named_parameters()
        if "refinement_stages" in name and "presence_delta" in name
    ]
    assert presence_delta_parameters
    assert all(parameter.grad is not None for parameter in presence_delta_parameters)
    assert any(
        float(parameter.grad.norm()) > 0.0
        for parameter in presence_delta_parameters
        if parameter.grad is not None
    )


def test_factorized_query_refinement_checkpoint_maps_presence_to_all_layers(
    tmp_path: Path,
) -> None:
    config = _config()
    config["model"]["pose_branch"]["state_head_mode"] = (
        "factorized_presence_visibility"
    )
    factorized = build_face_keypoint_cooperative_detr(
        config,
        face_detector=_FaceDetector(),
        initialize_pretrained=False,
    )
    dimension = factorized.pose_decoder.dimension
    presence_weight = torch.full((1, dimension), 0.125)
    presence_bias = torch.tensor([0.75])
    delta_weight = torch.full((1, dimension), -0.25)
    delta_bias = torch.tensor([-0.50])
    checkpoint = tmp_path / "factorized_query_refinement.pt"
    torch.save(
        {
            "ema": {
                "head.query_head.presence_head.weight": presence_weight,
                "head.query_head.presence_head.bias": presence_bias,
                "head.refinement_stages.0.presence_delta.weight": delta_weight,
                "head.refinement_stages.0.presence_delta.bias": delta_bias,
            }
        },
        checkpoint,
    )

    report = initialize_pose_from_query_refinement(
        factorized.pose_decoder,
        checkpoint,
        minimum_eligible_fraction=1.0,
    )

    assert report["eligible_fraction"] == pytest.approx(1.0)
    assert report["loaded_source_tensors"] == 4
    assert len(report["checkpoint_sha256"]) == 64
    assert report["target_state_head_mode"] == (
        "factorized_presence_visibility"
    )
    assert report["source_factorized_state_head"] is True
    assert len(
        report["source_to_target"][
            "head.query_head.presence_head.weight"
        ]
    ) == len(factorized.pose_decoder.layers)
    for head in factorized.pose_decoder.keypoint_confidence_heads:
        assert torch.equal(head.weight, presence_weight)
        assert torch.equal(head.bias, presence_bias)
    stage = factorized.pose_decoder.refinement_stages[0]
    assert torch.equal(stage.presence_delta.weight, delta_weight)
    assert torch.equal(stage.presence_delta.bias, delta_bias)


def test_model_forward_loss_and_backward(
    model: FaceKeypointCooperativeDETR,
) -> None:
    images = torch.rand(2, 3, 64, 64)
    padding_mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    padding_mask[:, :, 56:] = True
    output = model(
        images,
        [_target(1), _target(2)],
        padding_mask=padding_mask,
        branch_override=(True, True),
        return_attention=True,
    )
    assert torch.isfinite(output["loss_total"])
    assert output["independent_face"]["pred_boxes"].shape == (2, 6, 4)
    assert output["independent_pose"]["pred_keypoints"].shape == (2, 6, 5, 2)
    assert output["independent_pose"]["memory_mask"].dtype == torch.bool
    assert output["independent_pose"]["memory_mask"].any()
    assert output["refined"]["association_logits"].shape == (2, 6, 6)
    assert len(output["independent_pose"]["auxiliary_outputs"]) == 1
    assert len(output["refined"]["auxiliary_outputs"]) == 1
    for key in (
        "loss_face_independent",
        "loss_keypoint_independent",
        "loss_association",
        "loss_face_refined",
        "loss_keypoint_refined",
        "loss_visibility",
    ):
        assert key in output
        assert torch.isfinite(output[key])
    output["loss_total"].backward()
    for name, parameters in model.parameter_groups().items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if trainable:
            assert all(parameter.grad is not None for parameter in trainable), name
            assert any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.norm()) > 0.0
                for parameter in trainable
            ), name


@pytest.mark.parametrize("faces", [0, 1, 2])
def test_empty_single_and_multiple_face_targets(
    model: FaceKeypointCooperativeDETR,
    faces: int,
) -> None:
    output = model(torch.rand(1, 3, 64, 64), [_target(faces)])
    assert torch.isfinite(output["loss_total"])
    assert output["independent_pose"]["pred_pose_logits"].shape == (1, 6)


def test_branch_dropout_directions_and_inference_decode(
    model: FaceKeypointCooperativeDETR,
) -> None:
    images = torch.rand(1, 3, 64, 64)
    model.train()
    for override in ((True, True), (True, False), (False, True), (False, False)):
        output = model(images, [_target()], branch_override=override)
        assert output["refined"]["branch_face_to_pose_enabled"].item() == int(
            override[0]
        )
        assert output["refined"]["branch_pose_to_face_enabled"].item() == int(
            override[1]
        )
    model.eval()
    decoded = model(images)[0]
    assert set(decoded) >= {
        "independent_face",
        "independent_pose",
        "refined_face",
        "refined_pose",
        "associations",
    }
    assert decoded["refined_pose"]["keypoints"].shape[-2:] == (5, 2)

    raw = model(images, return_raw=True)
    model.face_score_threshold = 0.5
    model.pose_score_threshold = 0.5
    raw["independent_face"]["pred_logits"].fill_(8.0)
    raw["refined"]["pred_face_logits"].fill_(8.0)
    raw["independent_pose"]["pred_pose_logits"].fill_(-8.0)
    raw["refined"]["pred_pose_logits"].fill_(-8.0)
    face_only = model.decode(raw, image_size=(64, 64))[0]
    assert face_only["refined_face"]["scores"].numel() > 0
    assert face_only["refined_pose"]["scores"].numel() == 0

    raw["independent_face"]["pred_logits"].fill_(-8.0)
    raw["refined"]["pred_face_logits"].fill_(-8.0)
    raw["independent_pose"]["pred_pose_logits"].fill_(8.0)
    raw["refined"]["pred_pose_logits"].fill_(8.0)
    pose_only = model.decode(raw, image_size=(64, 64))[0]
    assert pose_only["refined_face"]["scores"].numel() == 0
    assert pose_only["refined_pose"]["scores"].numel() > 0


def test_eval_with_targets_produces_loss_and_restores_decoder_mode(
    model: FaceKeypointCooperativeDETR,
) -> None:
    model.eval()
    assert not model.face_detector.decoder.training
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64), [_target()])
    assert torch.isfinite(output["loss_total"])
    assert not model.training
    assert not model.face_detector.decoder.training


def test_association_and_pose_matcher() -> None:
    association = FacePoseAssociation(
        16,
        16,
        appearance_dimension=8,
        hidden_dimension=16,
    )
    face_embeddings = torch.randn(1, 2, 16, requires_grad=True)
    pose_embeddings = torch.randn(1, 3, 16, requires_grad=True)
    output = association(
        face_embeddings,
        pose_embeddings,
        face_boxes=torch.rand(1, 2, 4).clamp(0.1, 0.9),
        face_ellipses=torch.cat(
            (
                torch.rand(1, 2, 2),
                torch.rand(1, 2, 2) * 0.2 + 0.05,
                torch.zeros(1, 2, 1),
            ),
            dim=-1,
        ),
        face_logits=torch.randn(1, 2, 1),
        pose_boxes=torch.rand(1, 3, 4).clamp(0.1, 0.9),
        keypoints=torch.rand(1, 3, 5, 2),
        pose_logits=torch.randn(1, 3),
        keypoint_logits=torch.randn(1, 3, 5),
        visibility_logits=torch.randn(1, 3, 5),
    )
    assert output["logits"].shape == (1, 2, 3)
    assert output["geometry"].shape == (1, 2, 3, 15)
    output["logits"].sum().backward()
    assert face_embeddings.grad is not None
    assert pose_embeddings.grad is not None

    matcher = PoseHungarianMatcher()
    matches = matcher(
        {
            "pred_pose_logits": torch.tensor([[8.0, -8.0]]),
            "pred_pose_boxes": torch.tensor(
                [[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]]
            ),
            "pred_keypoints": torch.tensor(
                [[[[0.4, 0.4]] * 5, [[0.1, 0.1]] * 5]]
            ),
        },
        [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                "keypoints": torch.tensor([[[0.4, 0.4]] * 5]),
                "labeled": torch.ones(1, 5, dtype=torch.bool),
            }
        ],
    )
    assert matches[0][0].tolist() == [0]
    assert matches[0][1].tolist() == [0]


def test_strict_checkpoint_loaders(
    model: FaceKeypointCooperativeDETR,
    tmp_path: Path,
) -> None:
    face_checkpoint = tmp_path / "face.pt"
    torch.save({"ema": model.face_detector.state_dict()}, face_checkpoint)
    report = load_face_checkpoint_strict(
        model.face_detector,
        face_checkpoint,
    )
    assert report["fraction"] == 1.0

    cooperative_checkpoint = tmp_path / "cooperative.pt"
    torch.save({"model": model.state_dict()}, cooperative_checkpoint)
    report = load_cooperative_checkpoint_strict(
        model,
        cooperative_checkpoint,
        prefer_ema=False,
    )
    assert report["loaded_tensors"] == len(model.state_dict())

    ema = ModelEMA(model)
    ema_checkpoint = tmp_path / "cooperative_ema.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
        },
        ema_checkpoint,
    )
    report = load_cooperative_checkpoint_strict(
        model,
        ema_checkpoint,
        prefer_ema=True,
    )
    assert report["loaded_tensors"] == len(model.state_dict())

    incomplete_ema = ema.state_dict()
    missing_float_key = next(iter(incomplete_ema["shadow"]))
    incomplete_ema["shadow"].pop(missing_float_key)
    incomplete_ema_checkpoint = tmp_path / "cooperative_incomplete_ema.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": incomplete_ema,
        },
        incomplete_ema_checkpoint,
    )
    with pytest.raises(RuntimeError, match="Cooperative checkpoint mismatch"):
        load_cooperative_checkpoint_strict(
            model,
            incomplete_ema_checkpoint,
            prefer_ema=True,
        )

    broken = dict(model.face_detector.state_dict())
    broken.pop(next(iter(broken)))
    broken_checkpoint = tmp_path / "broken.pt"
    torch.save({"model": broken}, broken_checkpoint)
    with pytest.raises(RuntimeError, match="not architecture-compatible"):
        load_face_checkpoint_strict(
            model.face_detector,
            broken_checkpoint,
            prefer_ema=False,
        )


class _AugmentationDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return (
            torch.full((3, 64, 64), index / 4.0),
            {
                "boxes": torch.tensor([[8.0, 12.0, 40.0, 48.0]]),
                "ellipses": torch.tensor([[24.0, 30.0, 16.0, 10.0, 0.2]]),
                "labels": torch.tensor([2], dtype=torch.long),
                "keypoints": torch.tensor(
                    [
                        [
                            [0.25, 0.30],
                            [0.50, 0.30],
                            [0.375, 0.45],
                            [0.28, 0.60],
                            [0.47, 0.60],
                        ]
                    ]
                ),
                "labeled": torch.ones(1, 5, dtype=torch.bool),
                "visible": torch.ones(1, 5, dtype=torch.bool),
                "image_id": torch.tensor(index, dtype=torch.long),
            },
        )


class _HeadCentricAugmentationDataset(_AugmentationDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        image, target = super().__getitem__(index)
        target.update(
            {
                "ellipse_mask": torch.ones(1, dtype=torch.bool),
                "head_boxes": torch.tensor([[4.0, 6.0, 48.0, 54.0]]),
                "head_confidence": torch.tensor([0.9]),
                "geometry_target_indices": torch.tensor([0], dtype=torch.long),
                "head_sources": ["cached_predicted"],
                "rotation_matrix": torch.eye(3).reshape(1, 3, 3),
                "pose_supervision_mask": torch.tensor([False]),
                "pose_weight": torch.tensor([0.0]),
                "instance_ids": torch.tensor([index], dtype=torch.long),
                "sample_is_synthetic": torch.tensor([False]),
            }
        )
        return image, target


def test_head_centric_border_crop_transforms_head_face_and_query_targets() -> None:
    dataset = HeadCentricEllipseBorderCropAugmentation(
        _HeadCentricAugmentationDataset(),
        {
            "ellipse_border_crop": 1.0,
            "ellipse_border_crop_center_outside_probability": 1.0,
            "ellipse_border_crop_center_margin": [0.10, 0.10],
            "ellipse_border_crop_min_scale": 0.50,
            "minimum_axis": 2.0,
        },
        seed=42,
    )

    image, target = dataset[0]

    assert image.shape == (3, 64, 64)
    assert target["ellipses"].shape == (1, 5)
    center = target["ellipses"][0, :2]
    assert bool(((center < 0.0) | (center >= 64.0)).any())
    assert bool((target["boxes"] >= 0.0).all())
    assert bool((target["boxes"] <= 64.0).all())
    assert bool((target["head_boxes"] >= 0.0).all())
    assert bool((target["head_boxes"] <= 64.0).all())
    assert target["geometry_target_indices"].tolist() == [0]
    assert target["head_sources"] == ["cached_predicted"]
    assert bool((target["keypoints"] >= 0.0).all())
    assert bool((target["keypoints"] <= 1.0).all())
    debug = target["augmentation_debug"]["ellipse_border_crop"]
    assert debug["mode"] == "center"
    assert debug["center_outside"] is True
    assert debug["contour_outside"] is True


def test_joint_augmentation_preserves_target_semantics() -> None:
    dataset = CooperativeJointAugmentation(
        _AugmentationDataset(),
        {
            "mosaic_probability": 1.0,
            "close_mosaic_epochs": 1,
            "mixup_probability": 0.0,
            "degrees": 10.0,
            "translate": 0.05,
            "scale": 0.10,
            "horizontal_flip_probability": 1.0,
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.05,
            "blur_probability": 1.0,
            "blur_kernel_max": 6,
            "noise_probability": 1.0,
            "noise_std": 0.01,
            "occlusion_probability": 0.0,
        },
        seed=17,
        total_epochs=5,
    )
    set_dataset_epoch(dataset, 0)
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert torch.isfinite(image).all()
    assert target["boxes"].shape[0] == 4
    assert target["ellipses"].shape == (4, 5)
    assert target["keypoints"].shape == (4, 5, 2)
    assert target["labeled"].shape == target["visible"].shape == (4, 5)
    assert (target["boxes"] >= 0.0).all()
    assert (target["boxes"] <= 64.0).all()
    assert (target["keypoints"] >= 0.0).all()
    assert (target["keypoints"] <= 1.0).all()
    assert (target["visible"] <= target["labeled"]).all()

    # Occlusion changes visibility, never whether a keypoint is annotated.
    occlusion = CooperativeJointAugmentation(
        _AugmentationDataset(),
        {
            "occlusion_probability": 1.0,
            "occlusion_area": (100.0, 100.0),
        },
        seed=3,
        total_epochs=1,
    )
    _, occluded_target = occlusion[0]
    assert occluded_target["labeled"].all()
    assert not occluded_target["visible"].any()


def test_training_checkpoint_save_and_resume(
    model: FaceKeypointCooperativeDETR,
    tmp_path: Path,
) -> None:
    config = {
        "train": {
            "lr": 1.0e-4,
            "learning_rates": {},
            "weight_decay": 0.01,
            "fused_optimizer": False,
        }
    }
    optimizer, _ = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _epoch: 1.0,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ema = ModelEMA(model)
    reference = next(model.parameters()).detach().clone()
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        step=17,
        best_metric=0.25,
        config=config,
        metrics={"val/loss_total": 0.25},
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    resumed = resume_checkpoint(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
    )
    assert resumed == {
        "start_epoch": 4,
        "step": 17,
        "best_metric": 0.25,
    }
    assert torch.equal(next(model.parameters()).detach(), reference)

    changed_contract = resume_checkpoint(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
        current_config={
            **config,
            "evaluation": {"batch_size": 8},
        },
    )
    assert changed_contract["best_metric"] == float("inf")
    assert changed_contract["validation_metric_contract"] == {
        "reset_best_metric": True,
        "checkpoint_batch_size": 1,
        "current_batch_size": 8,
    }

    with pytest.raises(ValueError, match="architecture contract"):
        resume_checkpoint(
            path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=torch.device("cpu"),
            current_config={
                **config,
                "model": {
                    "strict_sequential_cascade": {"enabled": True},
                },
            },
        )


def test_cooperative_wandb_identity_and_history_are_resume_safe(
    tmp_path: Path,
) -> None:
    config = {
        "experiment_name": "cooperative-test",
        "data": {"dataset_version": "balanced-811"},
        "train": {"seed": 42},
    }
    first = cooperative_wandb_run_id(
        config,
        entity="team",
        project="face_detection",
    )
    assert first == cooperative_wandb_run_id(
        config,
        entity="team",
        project="face_detection",
    )
    changed = {
        **config,
        "train": {"seed": 43},
    }
    assert first != cooperative_wandb_run_id(
        changed,
        entity="team",
        project="face_detection",
    )

    history_path = tmp_path / "metrics.csv"
    history_path.write_text(
        "epoch,global_step,updates,train/loss_total,val/loss_total\n"
        "0,10,5,2.5,2.0\n"
        "1,20,5,1.5,1.0\n",
        encoding="utf-8",
    )
    assert read_metric_history(history_path) == [
        {
            "epoch": 0,
            "global_step": 10,
            "updates": 5,
            "train/loss_total": 2.5,
            "val/loss_total": 2.0,
        },
        {
            "epoch": 1,
            "global_step": 20,
            "updates": 5,
            "train/loss_total": 1.5,
            "val/loss_total": 1.0,
        },
    ]

    append_history(
        history_path,
        {
            "val/loss_total": 0.5,
            "updates": 5,
            "global_step": 30,
            "train/loss_total": 0.75,
            "epoch": 2,
        },
    )
    assert read_metric_history(history_path)[-1] == {
        "epoch": 2,
        "global_step": 30,
        "updates": 5,
        "train/loss_total": 0.75,
        "val/loss_total": 0.5,
    }
    append_history(
        history_path,
        {
            "epoch": 3,
            "global_step": 40,
            "updates": 5,
            "train/loss_total": 0.6,
            "val/loss_total": 0.4,
            "val/face_refined_precision": 0.8,
        },
    )
    assert read_metric_history(history_path)[-1][
        "val/face_refined_precision"
    ] == 0.8


def test_cooperative_metric_history_rewrite_and_compact_wandb_row(
    tmp_path: Path,
) -> None:
    components = {
        "loss_face_independent": 1.0,
        "loss_keypoint_independent": 2.0,
        "loss_association": 3.0,
        "loss_face_refined": 4.0,
        "loss_keypoint_refined": 5.0,
        "loss_visibility": 6.0,
    }
    row: dict[str, int | float] = {
        "epoch": 4,
        "global_step": 100,
        "updates": 20,
        "elapsed_seconds": 180.0,
        "learning_rate_min": 1.0e-6,
        "learning_rate_max": 1.0e-4,
        "train/loss_total": 10.0,
        "train/association_accuracy": 0.4,
        "val/loss_total": 9.0,
        "val/association_accuracy": 0.5,
    }
    for split in ("train", "val"):
        row.update(
            {
                f"{split}/{key}": value
                for key, value in components.items()
            }
        )
    row["train/loss_debug_aux_0"] = 99.0
    row["val/loss_debug_aux_0"] = 98.0
    row["val/face_refined_precision"] = 0.8
    row["val/face_refined_recall"] = 0.6
    row["val/face_refined_f1"] = 0.6857142857

    history_path = tmp_path / "metrics.csv"
    write_metric_history(history_path, [row])
    assert read_metric_history(history_path) == [row]

    compact = wandb_metric_row(
        row,
        previous_elapsed_seconds=60.0,
    )
    assert compact["overview/train_total_loss"] == 10.0
    assert compact["overview/val_association_accuracy"] == 0.5
    assert compact["loss_components/train_visibility_classification"] == 6.0
    assert compact["timing/epoch_minutes"] == 2.0
    assert compact["evaluation/val_face_refined_precision"] == 0.8
    assert not any("debug_aux" in key for key in compact)

    detailed = wandb_metric_row(
        row,
        previous_elapsed_seconds=60.0,
        detailed_losses=True,
    )
    assert detailed["diagnostics/train_loss_debug_aux_0"] == 99.0


def test_cooperative_validation_detection_metrics_match_fixed_threshold() -> None:
    accumulator = FaceDetectionMetricAccumulator(
        score_threshold=0.5,
        iou_threshold=0.5,
    )
    prediction = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.4]),
    }
    accumulator.update(
        [
            {
                "independent_face": prediction,
                "refined_face": prediction,
            }
        ],
        [
            {
                "boxes": torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 10.0],
                        [60.0, 60.0, 70.0, 70.0],
                    ]
                )
            }
        ],
    )
    assert accumulator.compute({"distributed": False}) == {
        "face_independent_precision": 0.5,
        "face_independent_recall": 0.5,
        "face_independent_f1": 0.5,
        "face_independent_tp": 1,
        "face_independent_fp": 1,
        "face_independent_fn": 1,
        "face_refined_precision": 0.5,
        "face_refined_recall": 0.5,
        "face_refined_f1": 0.5,
        "face_refined_tp": 1,
        "face_refined_fp": 1,
        "face_refined_fn": 1,
    }


def test_cooperative_batch_fallback_preserves_effective_batch() -> None:
    plans = configured_batch_plans(
        {
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "batch_size_fallback": {
                "enabled": True,
                "candidates": [
                    {
                        "batch_size": 8,
                        "gradient_accumulation_steps": 2,
                    },
                    {
                        "batch_size": 4,
                        "gradient_accumulation_steps": 4,
                    },
                    {
                        "batch_size": 2,
                        "gradient_accumulation_steps": 8,
                    },
                    {
                        "batch_size": 1,
                        "gradient_accumulation_steps": 16,
                    },
                ],
            },
        }
    )
    assert plans == (
        BatchPlan(8, 2),
        BatchPlan(4, 4),
        BatchPlan(2, 8),
        BatchPlan(1, 16),
    )
    assert {plan.effective_batch_size for plan in plans} == {16}


def test_cooperative_batch_fallback_only_advances_on_cuda_oom() -> None:
    plans = (
        BatchPlan(8, 2),
        BatchPlan(4, 4),
        BatchPlan(2, 8),
        BatchPlan(1, 16),
    )
    probed = []

    def probe(plan: BatchPlan) -> dict[str, float]:
        probed.append(plan)
        if plan.batch_size > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {"peak_reserved_mib": 17_000.0}

    selected, attempts = select_batch_plan(plans, probe)
    assert selected == BatchPlan(2, 8)
    assert probed == list(plans[:3])
    assert [attempt["status"] for attempt in attempts] == [
        "cuda_oom",
        "cuda_oom",
        "selected",
    ]

    def unrelated_failure(_plan: BatchPlan) -> dict[str, float]:
        raise RuntimeError("unrelated failure")

    with pytest.raises(RuntimeError, match="unrelated failure"):
        select_batch_plan(plans, unrelated_failure)


def test_cooperative_epoch_losses_are_aggregated_before_host_conversion() -> None:
    accumulator = EpochLossAccumulator()
    accumulator.update(
        {
            "loss_total": torch.tensor(4.0),
            "loss_face": torch.tensor(2.0),
            "association_accuracy": torch.tensor(0.25),
            "ignored": torch.tensor([100.0]),
        }
    )
    accumulator.update(
        {
            "loss_total": torch.tensor(2.0),
            "loss_face": torch.tensor(1.0),
            "association_accuracy": torch.tensor(0.75),
            "ignored": torch.tensor([200.0]),
        }
    )

    assert accumulator.mean({"distributed": False}) == {
        "association_accuracy": 0.5,
        "loss_face": 1.5,
        "loss_total": 3.0,
    }


def test_cooperative_epoch_loss_accumulator_validates_epoch_contract() -> None:
    accumulator = EpochLossAccumulator()
    with pytest.raises(RuntimeError, match="scalar loss_total"):
        accumulator.update({"loss_face": torch.tensor(1.0)})

    accumulator.update(
        {
            "loss_total": torch.tensor(1.0),
            "loss_face": torch.tensor(0.5),
        }
    )
    accumulator.update({"loss_total": torch.tensor(1.0)})
    assert accumulator.mean({"distributed": False}) == {
        "loss_face": 0.25,
        "loss_total": 1.0,
    }

    nonfinite = EpochLossAccumulator()
    nonfinite.update({"loss_total": torch.tensor(float("nan"))})
    with pytest.raises(FloatingPointError, match="loss_total"):
        nonfinite.mean({"distributed": False})


def test_amp_optimizer_step_skips_overflow_and_recovers() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=2.0,
        growth_interval=2000,
        enabled=True,
    )

    # The scalar loss is finite at zero, but sqrt has an infinite derivative
    # there. This reproduces the production failure contract without CUDA.
    overflow_loss = torch.sqrt(model.weight.sum() - 1.0)
    assert torch.isfinite(overflow_loss)
    scaler.scale(overflow_loss).backward()
    overflow = step_optimizer_with_grad_scaler(
        model,
        optimizer,
        scaler,
        max_grad_norm=0.1,
    )

    assert not overflow.optimizer_step_succeeded
    assert overflow.gradient_norm == float("inf")
    assert overflow.scale_before_step == 2.0
    assert overflow.scale_after_step == 1.0
    assert model.weight.item() == pytest.approx(1.0)
    assert model.weight.grad is None

    finite_loss = model.weight.square().sum()
    scaler.scale(finite_loss).backward()
    recovered = step_optimizer_with_grad_scaler(
        model,
        optimizer,
        scaler,
        max_grad_norm=0.1,
    )

    assert recovered.optimizer_step_succeeded
    assert recovered.scale_before_step == 1.0
    assert recovered.scale_after_step == 1.0
    assert model.weight.item() < 1.0
    assert model.weight.grad is None


def test_nonfinite_gradient_without_amp_remains_fatal() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    torch.sqrt(model.weight.sum() - 1.0).backward()

    with pytest.raises(
        FloatingPointError,
        match="without an enabled GradScaler",
    ):
        step_optimizer_with_grad_scaler(
            model,
            optimizer,
            scaler,
            max_grad_norm=0.1,
        )

    assert model.weight.item() == pytest.approx(1.0)
    assert model.weight.grad is None


def test_cooperative_production_validation_uses_batch_eight() -> None:
    config = load_yaml_config(COOPERATIVE_CONFIG_PATH)
    assert config["evaluation"]["batch_size"] == 8
    assert config["train"]["amp_dtype"] == "float16"
    assert config["train"]["grad_scaler_init_scale"] == 1.0
    assert config["train"]["max_consecutive_amp_skips"] == 8


def test_head_pose_geometry_disabled_adds_no_state_or_output(
    model: FaceKeypointCooperativeDETR,
) -> None:
    assert model.head_pose_geometry is None
    assert not any(
        key.startswith("head_pose_geometry.")
        for key in model.state_dict()
    )
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.rand(1, 3, 64, 64),
            return_raw=True,
            branch_override=(True, True),
        )
    assert "head_pose_geometry" not in output


def test_stage_checkpoint_allows_only_new_geometry_tensors(
    model: FaceKeypointCooperativeDETR,
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage1.pt"
    torch.save({"model": model.state_dict()}, source)
    reference = next(model.parameters()).detach().clone()
    model.head_pose_geometry = _head_pose_extension(geometry=False)
    report = initialize_stage_checkpoint(model, source)
    assert report["new_stage_tensors"]
    assert all(
        key.startswith("head_pose_geometry.")
        for key in report["new_stage_tensors"]
    )
    assert torch.equal(next(model.parameters()).detach(), reference)


def test_stage_checkpoint_combines_full_state_with_trainable_ema(
    model: FaceKeypointCooperativeDETR,
    tmp_path: Path,
) -> None:
    full_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    ema_key = next(
        key
        for key, value in full_state.items()
        if torch.is_floating_point(value) and value.numel() > 0
    )
    ema_value = torch.full_like(full_state[ema_key], 0.125)
    source = tmp_path / "face_stage.pt"
    torch.save(
        {
            "model": full_state,
            "ema": {"shadow": {ema_key: ema_value}},
        },
        source,
    )
    for value in model.state_dict().values():
        value.zero_()

    report = initialize_stage_checkpoint(model, source)

    assert report["new_stage_tensors"] == []
    assert report["loaded_tensors"] == len(full_state)
    assert torch.equal(model.state_dict()[ema_key], ema_value)


@pytest.mark.parametrize("geometry", (False, True))
def test_integrated_head_pose_geometry_forward_backward(
    model: FaceKeypointCooperativeDETR,
    geometry: bool,
) -> None:
    model.head_pose_geometry = _head_pose_extension(geometry=geometry)
    images = torch.rand(2, 3, 64, 64)
    output = model(
        images,
        [_geometry_target(1), _geometry_target(2)],
        branch_override=(True, True),
    )
    assert "head_pose_geometry" in output
    assert torch.isfinite(output["loss_total"])
    if geometry:
        assert "loss_head_pose_geometry" in output
        assert torch.isfinite(output["loss_head_pose_geometry"])
        assert "head_pose_geometry_fusion" in output["refined"]
    output["loss_total"].backward()
    extension_gradients = [
        parameter.grad
        for parameter in model.head_pose_geometry.parameters()
        if parameter.requires_grad
    ]
    assert extension_gradients
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and bool(torch.count_nonzero(gradient))
        for gradient in extension_gradients
    )


def test_geometry_training_allows_missed_predicted_heads(
    model: FaceKeypointCooperativeDETR,
) -> None:
    model.head_pose_geometry = _head_pose_extension(geometry=True)
    target = _geometry_target(2)
    target["head_boxes"] = target["head_boxes"][1:2]
    target["head_confidence"] = torch.tensor((0.7,))
    target["head_sources"] = ["cached_predicted"]
    target["geometry_target_indices"] = torch.tensor((1,))
    output = model(
        torch.rand(1, 3, 64, 64),
        [target],
        branch_override=(True, True),
    )
    assert torch.isfinite(output["loss_total"])
    fusion = output["refined"]["head_pose_geometry_fusion"][0]
    assert fusion["target_indices"].tolist() == [1]
    assert fusion["geometry_indices"].tolist() == [0]


def test_external_head_router_prefers_contained_face_and_backpropagates() -> None:
    router = ExternalHeadCascadeRouter(
        hidden_dimension=16,
        face_to_head_threshold=0.50,
        loss_weight=2.0,
        hard_negative_ratio=2,
    )
    face_boxes = torch.tensor(
        [[[0.50, 0.50, 0.20, 0.20], [0.90, 0.90, 0.10, 0.10]]],
        requires_grad=True,
    )
    face_logits = torch.zeros((1, 2, 1), requires_grad=True)
    heads = [
        [
            HeadDetection(
                box_xyxy=(20.0, 20.0, 80.0, 80.0),
                confidence=0.90,
                instance_id=0,
                source="head_codino",
            )
        ]
    ]
    routing = router(
        face_boxes=face_boxes,
        face_logits=face_logits,
        head_detections=heads,
        image_size=(100, 100),
    )
    probabilities = routing["best_head_probability"][0]
    assert probabilities[0] > 0.99
    assert probabilities[0] > probabilities[1]
    assert routing["best_head_index"].tolist() == [[0, 0]]

    losses = router.loss(
        routing,
        face_matches=[(torch.tensor([0]), torch.tensor([0]))],
        geometry_target_indices=[torch.tensor([0])],
    )
    assert torch.isfinite(losses["loss_external_head_routing"])
    losses["loss_external_head_routing"].backward()
    final = router.residual[-1]
    assert isinstance(final, nn.Linear)
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()


def test_three_stage_model_has_no_inference_gt_fallback_and_keeps_diagnostics(
    model: FaceKeypointCooperativeDETR,
) -> None:
    model.external_head_cascade = ExternalHeadCascadeRouter(
        hidden_dimension=16,
        face_to_head_threshold=0.50,
    )
    images = torch.rand(1, 3, 64, 64)
    model.eval()
    with pytest.raises(ValueError, match="requires predicted head_detections"):
        model(images)

    raw = model(images, return_raw=True, head_detections=[[]])
    raw["independent_face"]["pred_logits"].fill_(8.0)
    raw["refined"]["pred_face_logits"].fill_(8.0)
    raw["independent_pose"]["pred_pose_logits"].fill_(8.0)
    raw["refined"]["pred_pose_logits"].fill_(8.0)
    raw["refined"]["association_logits"].fill_(8.0)
    decoded = model.decode(raw, image_size=(64, 64))[0]
    assert decoded["independent_face"]["scores"].numel() > 0
    assert decoded["cooperative_refined_face"]["scores"].numel() > 0
    assert decoded["cooperative_refined_pose"]["scores"].numel() > 0
    assert decoded["refined_face"]["scores"].numel() == 0
    assert decoded["refined_pose"]["scores"].numel() == 0
    assert decoded["external_head_cascade"]["head_boxes"].shape == (0, 4)

    model.train()
    training = model(images, [_geometry_target()])
    assert torch.isfinite(training["loss_external_head_routing"])
    assert torch.isfinite(training["loss_total"])


def test_strict_three_stage_forward_is_head_then_face_then_keypoint(
    model: FaceKeypointCooperativeDETR,
) -> None:
    strict_config = _strict_sequential_config()["model"][
        "strict_sequential_cascade"
    ]
    strict_config["face_score_threshold"] = 0.0
    strict_config["maximum_face_rois"] = 100
    model.shared_head_branch = _TinySharedHead()
    model.external_head_cascade = ExternalHeadCascadeRouter(
        hidden_dimension=16,
        face_to_head_threshold=0.0,
    )
    model.strict_sequential_cascade = StrictSequentialCascade(strict_config)
    model.cooperative_decoder.enforce_face_to_pose_only()

    images = torch.rand(1, 3, 64, 64)
    model.train()
    output = model(
        images,
        [_geometry_target()],
        branch_override=(True, False),
    )
    sequential = output["strict_sequential_cascade"]
    assert sequential["execution_order"] == ("head", "face", "keypoint")
    assert sequential["head_to_face"]["source"] == "ground_truth_augmented"
    assert sequential["face_to_keypoint"]["source"] == (
        "ground_truth_augmented"
    )
    assert sequential["head_to_face"]["roi_count"].tolist() == [1]
    assert sequential["face_to_keypoint"]["roi_count"].tolist() == [1]
    assert output["refined"]["branch_face_to_pose_enabled"].bool().all()
    assert not output["refined"]["branch_pose_to_face_enabled"].bool().any()
    assert all(
        not parameter.requires_grad
        for layer in model.cooperative_decoder.layers
        for module in (
            layer.pose_to_face_attention,
            layer.keypoint_to_face_gate,
        )
        for parameter in module.parameters()
    )

    with pytest.raises(ValueError, match="forbids keypoint-to-FACE"):
        model(images, [_geometry_target()], branch_override=(True, True))

    model.eval()
    raw = model(images, return_raw=True)
    evaluation = raw["strict_sequential_cascade"]
    assert evaluation["execution_order"] == ("head", "face", "keypoint")
    assert evaluation["head_to_face"]["source"] == "predicted"
    assert evaluation["face_to_keypoint"]["source"] == "predicted"
    assert evaluation["head_to_face"]["roi_count"].tolist() == [1]
    assert evaluation["face_to_keypoint"]["roi_count"].tolist() == [6]


def test_three_stage_ablation_configs_change_only_hpg_arm() -> None:
    baseline = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_landmark_query_baseline.yaml"
    )
    introduced = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_landmark_query_hpg.yaml"
    )

    assert baseline["model"]["external_head_cascade"]["enabled"] is True
    strict = baseline["model"]["strict_sequential_cascade"]
    assert strict["enabled"] is True
    assert strict["training_head_source"] == "ground_truth"
    assert strict["training_face_source"] == "ground_truth"
    assert strict["evaluation_head_source"] == "predicted"
    assert strict["evaluation_face_source"] == "predicted"
    assert strict["enforce_unidirectional_refinement"] is True
    assert strict["head_to_face"]["context_scale"] == pytest.approx(1.25)
    assert strict["face_to_keypoint"]["context_scale"] == pytest.approx(1.25)
    assert strict["head_to_face"]["center_jitter"] > 0.0
    assert strict["face_to_keypoint"]["log_scale_jitter"] > 0.0
    assert introduced["model"]["strict_sequential_cascade"] == strict
    assert introduced["model"]["external_head_cascade"] == (
        baseline["model"]["external_head_cascade"]
    )
    assert baseline["model"]["head_pose_geometry"]["enabled"] is False
    assert introduced["model"]["head_pose_geometry"]["enabled"] is True
    assert baseline["evaluation"]["batch_size"] == 8
    assert introduced["evaluation"]["batch_size"] == 1
    assert baseline["data"] == introduced["data"]
    assert baseline["train"]["seed"] == introduced["train"]["seed"] == 42
    for key in ("epochs", "weight_decay"):
        assert baseline["train"][key] == introduced["train"][key]
    assert baseline["train"]["batch_size"] == 4
    assert baseline["train"]["gradient_accumulation_steps"] == 4
    assert introduced["train"]["batch_size"] == 4
    assert introduced["train"]["gradient_accumulation_steps"] == 4
    assert baseline["train"]["grad_scaler_init_scale"] == 1.0
    assert introduced["train"]["grad_scaler_init_scale"] == 1.0
    assert (
        baseline["train"]["batch_size"]
        * baseline["train"]["gradient_accumulation_steps"]
        == introduced["train"]["batch_size"]
        * introduced["train"]["gradient_accumulation_steps"]
        == 16
    )
    assert introduced["train"]["batch_size_fallback"]["candidates"][0] == {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
    }
    assert baseline["train"]["batch_size_fallback"]["candidates"][0] == {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
    }


def test_three_stage_border_crop_config_enables_factorized_refinement() -> None:
    baseline = load_cooperative_config(
        "configs/three_stage/codino_deimv2_l_landmark_query_baseline.yaml"
    )
    border_crop = load_cooperative_config(
        "configs/three_stage/"
        "codino_deimv2_l_landmark_query_ellipse_border_crop.yaml"
    )

    assert (
        baseline["model"]["pose_branch"]["initialization"]
        == "random"
    )
    pose_branch = border_crop["model"]["pose_branch"]
    assert pose_branch["initialization"] == "fine_tuned"
    assert (
        pose_branch["state_head_mode"]
        == "factorized_presence_visibility"
    )
    assert pose_branch["checkpoint"].endswith(
        "coco_visibility/query_refinement/checkpoints/best_nme.pt"
    )
    assert pose_branch["prefer_ema"] is True
    assert (
        border_crop["model"]["face_detector"]
        == baseline["model"]["face_detector"]
    )
    assert (
        border_crop["model"]["cooperative_decoder"]
        == baseline["model"]["cooperative_decoder"]
    )
    assert border_crop["data"]["head_centric"] is True
    assert border_crop["data"]["augmentation"]["ellipse_border_crop"] == pytest.approx(0.20)
    assert (
        border_crop["data"]["augmentation"][
            "ellipse_border_crop_center_outside_probability"
        ]
        == pytest.approx(0.50)
    )
    assert border_crop["train"]["seed"] == baseline["train"]["seed"] == 42
    assert border_crop["evaluation"]["keypoint_presence_threshold"] == pytest.approx(
        0.50
    )
    assert border_crop["output"]["root"].endswith(
        "codino_deimv2_l_landmark_query_ellipse_border_crop_seed42"
    )


def test_keypoint_accumulator_reports_factorized_state_metrics() -> None:
    accumulator = KeypointAccumulator(
        [0.05],
        visibility_threshold=0.50,
        presence_threshold=0.50,
    )
    states = [0, 1, 2, 0, 2]
    accumulator.update(
        [
            {
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "keypoints": [
                    value
                    for index, state in enumerate(states)
                    for value in (10.0 * index, 5.0 * index, state)
                ],
            }
        ],
        {0: 0},
        {
            "keypoints": torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [10.0, 5.0],
                        [20.0, 10.0],
                        [30.0, 15.0],
                        [40.0, 20.0],
                    ]
                ]
            ),
            "keypoint_scores": torch.tensor(
                [[0.01, 0.99, 0.99, 0.01, 0.99]]
            ).to(dtype=torch.bfloat16),
            "visibility": torch.tensor(
                [[0.50, 0.01, 0.99, 0.50, 0.99]]
            ).to(dtype=torch.bfloat16),
            "state_probabilities": torch.tensor(
                [
                    [
                        [0.99, 0.005, 0.005],
                        [0.01, 0.98, 0.01],
                        [0.01, 0.01, 0.98],
                        [0.99, 0.005, 0.005],
                        [0.01, 0.01, 0.98],
                    ]
                ]
            ).to(dtype=torch.bfloat16),
            "scores": torch.tensor([0.90]),
        },
    )

    metrics = accumulator.compute()
    assert metrics["presence_missing_macro_f1"] == pytest.approx(1.0)
    assert metrics["visibility_occluded_macro_f1"] == pytest.approx(1.0)
    assert metrics["state_macro_f1"] == pytest.approx(1.0)
    assert metrics["state_confusion_matrix"] == [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]
def test_inverse_sigmoid_is_finite_at_fp16_boundaries() -> None:
    from src.models.face_keypoint_cooperative.model import _inverse_sigmoid

    values = torch.tensor([0.0, 1.0], dtype=torch.float16)
    result = _inverse_sigmoid(values)

    assert torch.isfinite(result).all()
