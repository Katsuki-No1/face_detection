from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.head_pose_geometry.cache import (
    PseudoPoseFilterConfig,
    PseudoPoseRecord,
    deterministic_json_bytes,
    filter_pseudo_pose,
    pose_cache_payload,
)
from src.head_pose_geometry.crops import (
    HeadCropBuilder,
    transform_points_homogeneous,
)
from src.head_pose_geometry.dataset import (
    HeadCentricFaceDataset,
    InterleavedRealSyntheticDataset,
    audit_source_video_splits,
)
from src.head_pose_geometry.extension import HeadPoseGeometryExtension
from src.head_pose_geometry.geometry import (
    CanonicalHeadGeometryAdapter,
    differentiable_covariance_ellipse,
)
from src.head_pose_geometry.heads import HeadDetection, require_predicted_heads
from src.head_pose_geometry.losses import HeadPoseGeometryLoss
from src.head_pose_geometry.pose_encoder import (
    FusedSixDRepNet360,
    PoseEncoderAdapter,
    load_fused_onnx_checkpoint,
)
from src.head_pose_geometry.rotations import (
    doubled_angle_distance,
    doubled_angle_vector,
    euler_xyz_to_matrix,
    flip_rotation_horizontal,
    geodesic_rotation_distance,
    matrix_to_euler_xyz,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


class TinyPoseBackbone(nn.Module):
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
        self.feature_projection = nn.Linear(4, self.feature_dimension)

    def forward_raw(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.features(image).flatten(1)
        return self.linear_reg(value), self.feature_projection(value)


def tiny_pose_encoder(*, embedding: int = 16) -> PoseEncoderAdapter:
    torch.manual_seed(4)
    return PoseEncoderAdapter(
        TinyPoseBackbone(),  # type: ignore[arg-type]
        training_mode="domain_adapter",
        embedding_dimension=embedding,
    )


@pytest.mark.parametrize(
    "euler",
    (
        (0.0, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (-math.pi / 2, 0.0, 0.0),
        (0.0, math.pi / 2 - 1.0e-4, 0.0),
        (0.0, 0.0, math.pi / 2),
        (math.pi - 1.0e-4, 0.0, 0.0),
    ),
)
def test_rotation_round_trip_and_special_angles(
    euler: tuple[float, float, float],
) -> None:
    angles = torch.tensor([euler], dtype=torch.float64)
    matrix = euler_xyz_to_matrix(angles)
    reconstructed = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))
    assert torch.allclose(reconstructed, matrix, atol=1.0e-8)
    assert torch.allclose(torch.det(reconstructed), torch.ones(1, dtype=torch.float64))
    diagnostic = matrix_to_euler_xyz(matrix)
    assert torch.allclose(
        euler_xyz_to_matrix(diagnostic),
        matrix,
        atol=2.0e-6,
    )


def test_rotation_6d_degenerate_input_is_proper_so3() -> None:
    raw = torch.zeros(5, 6, requires_grad=True)
    matrix = rotation_6d_to_matrix(raw)
    identity = torch.eye(3).expand_as(matrix)
    assert torch.allclose(matrix.transpose(-1, -2) @ matrix, identity, atol=1.0e-6)
    assert torch.all(torch.det(matrix) > 0.999)
    matrix.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_geodesic_distance_identity_quarter_and_near_pi() -> None:
    identity = torch.eye(3, dtype=torch.float64)
    rotations = euler_xyz_to_matrix(
        torch.tensor(
            ((0.0, 0.0, 0.0), (0.0, math.pi / 2, 0.0), (math.pi - 1.0e-7, 0.0, 0.0)),
            dtype=torch.float64,
        )
    )
    distance = geodesic_rotation_distance(identity, rotations)
    assert torch.allclose(
        distance,
        torch.tensor((0.0, math.pi / 2, math.pi - 1.0e-7), dtype=torch.float64),
        atol=1.0e-7,
    )


def test_horizontal_flip_is_involution_and_changes_yaw_roll_sign() -> None:
    angles = torch.tensor(((0.31, -0.72, 0.44),), dtype=torch.float64)
    matrix = euler_xyz_to_matrix(angles)
    flipped = flip_rotation_horizontal(matrix)
    assert torch.allclose(flip_rotation_horizontal(flipped), matrix)
    flipped_angles = matrix_to_euler_xyz(flipped)
    assert torch.allclose(
        flipped_angles,
        torch.tensor(((0.31, 0.72, -0.44),), dtype=torch.float64),
        atol=1.0e-7,
    )


def test_crop_transform_round_trip_and_padding_valid_mask() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 40 * 60).reshape(3, 40, 60)
    crop = HeadCropBuilder(output_size=(24, 32), context_scale=1.5).build(
        image,
        [
            HeadDetection(
                box_xyxy=(-8.0, 5.0, 28.0, 35.0),
                confidence=0.8,
                source="predicted",
            )
        ],
    )
    points = torch.tensor([[[0.0, 0.0], [31.0, 23.0], [15.0, 9.0]]])
    image_points = transform_points_homogeneous(points, crop.crop_to_image)
    restored = transform_points_homogeneous(image_points, crop.image_to_crop)
    assert torch.allclose(restored, points, atol=1.0e-5)
    assert 0.0 < float(crop.valid_mask.mean()) < 1.0
    invalid = crop.valid_mask.expand_as(crop.rgb) == 0
    assert torch.count_nonzero(crop.rgb[invalid]) == 0


def test_geometry_projects_points_and_keeps_amodal_coordinates() -> None:
    geometry = CanonicalHeadGeometryAdapter(12, contour_points=32)
    rotation = torch.eye(3).unsqueeze(0)
    output = geometry(
        rotation_matrix=rotation,
        pose_embedding=torch.zeros(1, 12),
        head_boxes_xyxy=torch.tensor(((-0.6, 0.1, -0.1, 0.8),)),
        border_distances=torch.tensor(((-0.6, 0.1, 1.1, 0.2),)),
        head_confidence=torch.ones(1),
        valid_fraction=torch.tensor((0.4,)),
    )
    assert output["projected_contour"].shape == (1, 32, 2)
    assert output["projected_keypoints"].shape == (1, 5, 2)
    assert float(output["amodal_ellipse"][0, 0]) < 0.0
    assert torch.all(output["amodal_ellipse"][:, 2] >= output["amodal_ellipse"][:, 3])
    assert torch.isfinite(output["keypoint_visibility"]).all()
    assert torch.allclose(
        output["keypoint_visibility"],
        torch.sigmoid(output["keypoint_visibility_logits"]),
    )
    assert torch.allclose(
        output["face_observability"],
        torch.sigmoid(output["face_observability_logits"]),
    )


def test_differentiable_ellipse_fit_and_periodic_angle() -> None:
    theta = torch.tensor(0.37)
    samples = torch.arange(256) * (2.0 * math.pi / 256)
    local = torch.stack((3.0 * torch.cos(samples), 1.2 * torch.sin(samples)), dim=-1)
    rotation = torch.tensor(
        ((torch.cos(theta), -torch.sin(theta)), (torch.sin(theta), torch.cos(theta)))
    )
    points = (local @ rotation.T + torch.tensor((4.0, -2.0))).requires_grad_()
    fit = differentiable_covariance_ellipse(points)
    assert torch.allclose(
        fit["ellipse"][:4], torch.tensor((4.0, -2.0, 3.0, 1.2)), atol=2.0e-4
    )
    assert float(doubled_angle_distance(fit["angle_vector"], theta)) < 1.0e-6
    assert torch.allclose(
        doubled_angle_vector(theta), doubled_angle_vector(theta + math.pi)
    )
    fit["ellipse"].sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_visibility_mask_respects_annotation_policy() -> None:
    geometry = CanonicalHeadGeometryAdapter(8)
    rotation = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    output = geometry(
        rotation_matrix=rotation,
        pose_embedding=torch.zeros(2, 8),
        head_boxes_xyxy=torch.tensor(((0.1, 0.1, 0.8, 0.9),) * 2),
        border_distances=torch.ones(2, 4),
        head_confidence=torch.ones(2),
        valid_fraction=torch.ones(2),
    )
    output["rotation_matrix"] = rotation
    target = {
        "ellipses": output["amodal_ellipse"].detach(),
        "ellipse_mask": torch.ones(2, dtype=torch.bool),
        "keypoints": output["projected_keypoints"].detach(),
        "keypoint_labeled": torch.tensor(
            ((True, False, False, False, False), (False,) * 5)
        ),
        "keypoint_visible": torch.ones(2, 5, dtype=torch.bool),
        "head_boxes_xyxy": torch.tensor(((0.1, 0.1, 0.8, 0.9),) * 2),
    }
    losses = HeadPoseGeometryLoss()(output, target)
    assert torch.isfinite(losses["loss_geometry_visibility"])
    assert float(losses["loss_geometry_keypoints"]) == pytest.approx(0.0, abs=1.0e-7)


def test_visibility_losses_use_logits_instead_of_probability_outputs() -> None:
    geometry = CanonicalHeadGeometryAdapter(8)
    rotation = torch.eye(3).unsqueeze(0)
    output = geometry(
        rotation_matrix=rotation,
        pose_embedding=torch.zeros(1, 8),
        head_boxes_xyxy=torch.tensor(((0.1, 0.1, 0.8, 0.9),)),
        border_distances=torch.ones(1, 4),
        head_confidence=torch.ones(1),
        valid_fraction=torch.ones(1),
    )
    output["rotation_matrix"] = rotation
    # Inference consumers retain probability outputs, but the loss must use
    # the corresponding logits so that AMP does not execute probability BCE.
    output["face_observability"] = torch.full_like(
        output["face_observability"], float("nan")
    )
    output["keypoint_visibility"] = torch.full_like(
        output["keypoint_visibility"], float("nan")
    )
    target = {
        "ellipses": output["amodal_ellipse"].detach(),
        "ellipse_mask": torch.ones(1, dtype=torch.bool),
        "keypoints": output["projected_keypoints"].detach(),
        "keypoint_labeled": torch.ones(1, 5, dtype=torch.bool),
        "keypoint_visible": torch.ones(1, 5, dtype=torch.bool),
        "head_boxes_xyxy": torch.tensor(((0.1, 0.1, 0.8, 0.9),)),
    }
    losses = HeadPoseGeometryLoss()(output, target)
    assert torch.isfinite(losses["loss_geometry_observability"])
    assert torch.isfinite(losses["loss_geometry_visibility"])


def test_pseudo_filter_uses_multiple_consistency_signals() -> None:
    accepted, reasons = filter_pseudo_pose(
        confidence=0.9,
        uncertainty=0.2,
        crop_jitter_consistency_radians=math.radians(3),
        horizontal_flip_consistency_radians=math.radians(4),
    )
    assert accepted and not reasons
    accepted, reasons = filter_pseudo_pose(
        confidence=0.9,
        uncertainty=0.2,
        crop_jitter_consistency_radians=math.radians(30),
        horizontal_flip_consistency_radians=math.radians(4),
    )
    assert not accepted
    assert "crop_jitter_consistency" in reasons
    assert "insufficient_consistency_checks" in reasons


def test_pose_cache_is_deterministic_and_train_only() -> None:
    record = PseudoPoseRecord(
        image_id="2",
        instance_id="1",
        split="train",
        source_video_id="video-a",
        rotation_matrix=torch.eye(3).tolist(),
        teacher_model="teacher",
        teacher_checkpoint_id="weights",
        teacher_checkpoint_sha256="abc",
        confidence=0.9,
        uncertainty=0.1,
        crop_jitter_consistency_radians=0.01,
        horizontal_flip_consistency_radians=0.02,
        temporal_consistency_radians=None,
        second_teacher_consistency_radians=None,
        accepted=True,
        rejection_reasons=[],
    )
    first = pose_cache_payload(
        [record],
        teacher={"name": "teacher"},
        filter_config=PseudoPoseFilterConfig(),
        seed=7,
    )
    second = pose_cache_payload(
        [record],
        teacher={"name": "teacher"},
        filter_config=PseudoPoseFilterConfig(),
        seed=7,
    )
    assert deterministic_json_bytes(first) == deterministic_json_bytes(second)
    record.split = "val"
    with pytest.raises(ValueError, match="only be cached for train"):
        pose_cache_payload(
            [record],
            teacher={},
            filter_config=PseudoPoseFilterConfig(),
            seed=7,
        )


def test_source_video_leakage_check(tmp_path: Path) -> None:
    paths = {}
    for split, video in (("train", "a"), ("val", "b"), ("test", "c")):
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps({"images": [{"id": 1, "source_video_id": video}]}),
            encoding="utf-8",
        )
        paths[split] = path
    report = audit_source_video_splits(paths)
    assert all(not values for values in report["overlap"].values())
    paths["test"].write_text(
        json.dumps({"images": [{"id": 1, "source_video_id": "a"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leakage"):
        audit_source_video_splits(paths)


def test_head_centric_dataset_predicted_guard_and_synthetic_mix(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (20, 10), "gray").save(tmp_path / "frame.png")
    annotation = {
        "images": [
            {
                "id": 1,
                "file_name": "frame.png",
                "width": 20,
                "height": 10,
                "source_video_id": "video-a",
            }
        ],
        "annotations": [
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 1, 15, 9],
                "attribute_valid": True,
                "ellipse": [-1, 5, 5, 2, 0.2],
                "keypoints": [[3, 3], [7, 3], [5, 5], [4, 7], [6, 7]],
                "keypoint_states": [2, 2, 1, 0, 0],
                "rotation_matrix": torch.eye(3).tolist(),
            },
            {
                "id": 4,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 2, 8, 7],
                "attribute_valid": False,
            },
        ],
        "categories": [{"id": 1, "name": "head"}],
    }
    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(ValueError, match="predicted_head_caches"):
        HeadCentricFaceDataset(
            annotation=annotation_path,
            image_root=tmp_path,
            split="val",
            image_size=32,
        )
    cache_path = tmp_path / "heads.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    "1": [
                        {
                            "box_xyxy": [1, 0, 18, 10],
                            "confidence": 0.7,
                            "source": "predicted_test",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    validation = HeadCentricFaceDataset(
        annotation=annotation_path,
        image_root=tmp_path,
        split="val",
        image_size=32,
        predicted_head_cache=cache_path,
    )
    _, validation_target = validation[0]
    assert validation_target["head_sources"] == ["cached_predicted"]
    assert validation_target["geometry_target_indices"].tolist() == [0]
    assert float(validation_target["ellipses"][0, 0]) < 0.0
    shared_validation = HeadCentricFaceDataset(
        annotation=annotation_path,
        image_root=tmp_path,
        split="val",
        image_size=32,
        require_predicted_heads=False,
    )
    _, shared_target = shared_validation[0]
    assert shared_target["boxes"].shape == (1, 4)
    assert shared_target["head_detection_boxes"].shape == (2, 4)
    assert shared_target["head_detection_to_face_indices"].tolist() == [0, -1]
    synthetic = HeadCentricFaceDataset(
        annotation=annotation_path,
        image_root=tmp_path,
        split="train",
        image_size=32,
        synthetic_rotation_supervision=True,
    )
    _, synthetic_target = synthetic[0]
    assert synthetic_target["pose_supervision_mask"].all()
    assert synthetic_target["sample_is_synthetic"].all()
    mixed = InterleavedRealSyntheticDataset(
        validation,
        synthetic,
        synthetic_fraction=0.5,
        seed=2,
    )
    assert len(mixed) == 2
    assert mixed[1][1]["sample_is_synthetic"].all()

    cache_path.write_text(
        json.dumps({"schema_version": 1, "entries": {"1": []}}),
        encoding="utf-8",
    )
    missed = HeadCentricFaceDataset(
        annotation=annotation_path,
        image_root=tmp_path,
        split="test",
        image_size=32,
        predicted_head_cache=cache_path,
    )
    _, missed_target = missed[0]
    assert missed_target["boxes"].shape == (1, 4)
    assert missed_target["head_boxes"].shape == (0, 4)
    assert missed_target["geometry_target_indices"].numel() == 0


def test_deployment_heads_reject_ground_truth() -> None:
    with pytest.raises(ValueError, match="requires predicted"):
        require_predicted_heads(
            [HeadDetection((0, 0, 10, 10), 1.0, source="ground_truth")],
            split="test",
        )


def _extension(*, geometry: bool) -> HeadPoseGeometryExtension:
    encoder = tiny_pose_encoder()
    adapter = CanonicalHeadGeometryAdapter(16) if geometry else None
    return HeadPoseGeometryExtension(
        crop_builder=HeadCropBuilder(output_size=32),
        pose_encoder=encoder,
        geometry_adapter=adapter,
        fusion=None,
        criterion=HeadPoseGeometryLoss() if geometry else None,
        pose_query_dimension=24,
    )


def test_pose_only_forward_backward_and_checkpoint_resume(tmp_path: Path) -> None:
    extension = _extension(geometry=False)
    images = torch.rand(1, 3, 64, 64)
    detections = [[HeadDetection((8, 7, 50, 58), 0.8, source="predicted")]]
    output = extension(images, detections)
    query = {
        "pose_embeddings": torch.zeros(1, 3, 24),
        "pred_pose_boxes": torch.tensor(
            (((0.45, 0.45, 0.4, 0.5), (0.1, 0.1, 0.2, 0.2), (0.8, 0.8, 0.1, 0.1)),)
        ),
    }
    injected = extension.inject_pose_queries(query, output)
    loss = injected["pose_embeddings"].square().mean()
    loss.backward()
    assert extension.pose_query_projection.weight.grad is not None  # type: ignore[union-attr]
    path = tmp_path / "resume.pth"
    torch.save(extension.state_dict(), path)
    restored = _extension(geometry=False)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for first, second in zip(
        extension.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_geometry_forward_backward_mixed_real_synthetic() -> None:
    extension = _extension(geometry=True)
    images = torch.rand(1, 3, 64, 64)
    detections = [
        [
            HeadDetection((4, 4, 34, 58), 0.9, source="ground_truth"),
            HeadDetection((30, 2, 68, 62), 0.7, source="ground_truth"),
        ]
    ]
    output = extension(images, detections)[0]
    target = {
        "ellipses": output["amodal_ellipse"].detach()
        + torch.tensor(((0.02, 0.0, 0.0, 0.0, 0.0),) * 2),
        "ellipse_mask": torch.ones(2, dtype=torch.bool),
        "keypoints": output["projected_keypoints"].detach(),
        "keypoint_labeled": torch.ones(2, 5, dtype=torch.bool),
        "keypoint_visible": torch.ones(2, 5, dtype=torch.bool),
        "head_boxes_xyxy": output["normalized_head_boxes_xyxy"],
        "rotation_matrix": torch.stack(
            (torch.eye(3), euler_xyz_to_matrix(torch.tensor((0.2, 0.1, 0.0))))
        ),
        "pose_supervision_mask": torch.tensor((True, False)),
        "pose_weight": torch.tensor((0.8, 0.0)),
    }
    losses = HeadPoseGeometryLoss()(output, target)
    losses["loss_head_pose_geometry"].backward()
    geometry = extension.geometry_adapter
    assert geometry is not None
    assert geometry.predictor[-1].weight.grad is not None
    assert torch.isfinite(losses["loss_head_pose_geometry"])


def test_geometry_extension_returns_complete_zero_loss_contract_without_heads() -> None:
    extension = _extension(geometry=True)
    outputs = extension(torch.rand(1, 3, 64, 64), [[]])

    losses = extension.losses(outputs, [{}], image_size=(64, 64))

    assert set(losses) == {
        *HeadPoseGeometryLoss.component_names,
        "loss_head_pose_geometry",
    }
    assert all(float(value.detach()) == 0.0 for value in losses.values())


def test_tiny_geometry_overfit_reduces_2d_loss() -> None:
    torch.manual_seed(11)
    student = CanonicalHeadGeometryAdapter(6, hidden_dimension=24)
    teacher = CanonicalHeadGeometryAdapter(6, hidden_dimension=24)
    with torch.no_grad():
        teacher.predictor[-1].bias.copy_(
            torch.tensor((0.3, -0.2, 0.15, -0.1, 0.5, 0.4))
        )
    rotation = euler_xyz_to_matrix(torch.tensor(((0.2, -0.4, 0.1),) * 3))
    embedding = torch.randn(3, 6)
    head_boxes = torch.tensor(
        ((0.1, 0.1, 0.5, 0.8), (0.4, 0.0, 0.9, 0.7), (-0.2, 0.2, 0.3, 0.9))
    )
    arguments = {
        "rotation_matrix": rotation,
        "pose_embedding": embedding,
        "head_boxes_xyxy": head_boxes,
        "border_distances": torch.randn(3, 4),
        "head_confidence": torch.ones(3),
        "valid_fraction": torch.tensor((1.0, 0.8, 0.5)),
    }
    with torch.no_grad():
        truth = teacher(**arguments)
    target = {
        "ellipses": truth["amodal_ellipse"],
        "ellipse_mask": torch.ones(3, dtype=torch.bool),
        "keypoints": truth["projected_keypoints"],
        "keypoint_labeled": torch.ones(3, 5, dtype=torch.bool),
        "keypoint_visible": (truth["keypoint_visibility"] > 0.5),
        "head_boxes_xyxy": head_boxes,
    }
    criterion = HeadPoseGeometryLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.03)
    initial = None
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        output = student(**arguments)
        output["rotation_matrix"] = rotation
        loss = criterion(output, target)["loss_head_pose_geometry"]
        initial = float(loss.detach()) if initial is None else initial
        loss.backward()
        optimizer.step()
    final_output = student(**arguments)
    final_output["rotation_matrix"] = rotation
    final = float(criterion(final_output, target)["loss_head_pose_geometry"])
    assert initial is not None and final < 0.45 * initial
    axes = student(**arguments)["amodal_ellipse"][:, 2:4]
    assert torch.isfinite(axes).all()
    assert torch.all(axes[:, 0] >= axes[:, 1])
    assert torch.all(axes > 0)


def test_converted_pretrained_checkpoint_has_explicit_mapping() -> None:
    path = Path("checkpoint/head_pose_geometry/sixdrepnet360_1x3x224x224_full.onnx")
    if not path.is_file():
        pytest.skip("User-supplied converted 6DRepNet360 checkpoint is absent")
    report = load_fused_onnx_checkpoint(FusedSixDRepNet360(), path)
    assert report["loaded_convolutions"] == 53
    assert report["loaded_tensors"] == 108
    assert report["key_mapping"]
    assert report["sha256"] == (
        "422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82"
    )
