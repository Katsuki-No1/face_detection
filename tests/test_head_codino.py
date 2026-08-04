from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from mmengine.structures import InstanceData
from PIL import Image

from src.data import CocoDetectionDataset
from src.models import build_detector_from_config
from src.models.head_codino import (
    CompactCoDINO,
    DINOv3ViTS16PlusBackbone,
    HeadCoDINODetector,
    HeadImageBatch,
    SimpleFeaturePyramid,
    cxcywh_to_xyxy,
    denormalize_boxes_cxcywh,
    expand_square_roi,
    normalize_boxes_xyxy,
    original_to_padded_normalized_cxcywh,
    padded_to_original_normalized_cxcywh,
    parameter_audit,
    postprocess_head_detections,
    xyxy_to_cxcywh,
)
from src.models.head_codino.runtime import (
    build_dataset,
    load_runtime_config,
    representative_subset_indices,
)
from scripts.select_head_codino_threshold import metrics_at_threshold


DINOV3_SOURCE = Path(__file__).resolve().parents[1] / "models" / "source" / "dinov3"


def test_bounded_subset_covers_source_videos_and_their_timelines() -> None:
    class Dataset:
        samples = [
            {"image": {"source_video_id": source}}
            for source in ("a", "b", "c")
            for _ in range(10)
        ]

        def __len__(self) -> int:
            return len(self.samples)

    indices, report = representative_subset_indices(Dataset(), 9)

    assert indices == [0, 4, 9, 10, 14, 19, 20, 24, 29]
    assert report["strategy"] == "source_video_balanced_even_spacing"
    assert report["selected_source_groups"] == 3
    assert report["selected_per_source_group"] == {"a": 3, "b": 3, "c": 3}


def test_image_batch_normalizes_and_pads_only_right_bottom() -> None:
    first = torch.zeros((3, 33, 65), dtype=torch.uint8)
    second = torch.ones((3, 64, 32), dtype=torch.float32)
    batch = HeadImageBatch.from_images([first, second])
    assert batch.tensors.shape == (2, 3, 64, 96)
    assert batch.original_sizes.tolist() == [[33, 65], [64, 32]]
    assert batch.individual_padded_sizes.tolist() == [[64, 96], [64, 32]]
    assert batch.padding_mask[0, :33, :65].sum() == 0
    assert batch.padding_mask[0, 33:, :].all()
    assert batch.padding_mask[0, :, 65:].all()
    p3_mask = batch.level_mask((8, 12), stride=8)
    assert (~p3_mask[0]).sum() == 5 * 9
    assert (~p3_mask[1]).sum() == 8 * 4


def test_backbone_uses_only_patch_tokens_with_correct_hw_order() -> None:
    backbone = DINOv3ViTS16PlusBackbone(source_root=DINOV3_SOURCE, weights=None)
    batch = HeadImageBatch.from_images([torch.rand(3, 63, 95)])
    feature = backbone(batch.image_list())[0]
    assert feature.shape == (384, 4, 6)


def test_sfp_shapes_and_exact_parameters_for_odd_patch_grid() -> None:
    sfp = SimpleFeaturePyramid()
    outputs = sfp(torch.rand(1, 384, 5, 7))
    assert {key: tuple(value.shape) for key, value in outputs.items()} == {
        "p3": (1, 256, 10, 14),
        "p4": (1, 256, 5, 7),
        "p5": (1, 256, 3, 4),
        "p6": (1, 256, 2, 2),
    }
    assert sum(parameter.numel() for parameter in sfp.parameters()) == 4_699_264
    assert all(
        module.bias is None
        for module in sfp.modules()
        if isinstance(module, torch.nn.Conv2d)
    )


def test_exact_component_and_total_parameter_contract() -> None:
    model = HeadCoDINODetector(backbone_weights=None)
    assert parameter_audit(model) == {
        "dinov3_vits16plus": {
            "expected": 28_697_472,
            "actual": 28_697_472,
            "matches": True,
        },
        "sfp": {"expected": 4_699_264, "actual": 4_699_264, "matches": True},
        "compact_codino": {
            "expected": 5_132_436,
            "actual": 5_132_436,
            "matches": True,
        },
        "total": {"expected": 38_529_172, "actual": 38_529_172, "matches": True},
    }


def test_model_builder_registration() -> None:
    model = build_detector_from_config(
        {
            "model": {
                "type": "head_codino",
                "dinov3_source_root": str(DINOV3_SOURCE),
                "backbone_weights": None,
            }
        }
    )
    assert isinstance(model, HeadCoDINODetector)


def test_rtmdet_hf_adapter_keeps_head_only_and_negative_images() -> None:
    config = load_runtime_config(
        Path(__file__).resolve().parents[1] / "configs" / "head_codino" / "smoke.yaml"
    )
    dataset = build_dataset(config, "train")
    assert len(dataset) > 0
    assert any(not sample["annotations"] for sample in dataset.samples)
    assert all(
        annotation["category_id"] == 1
        for sample in dataset.samples
        for annotation in sample["annotations"]
    )


def test_head_dataset_uses_valid_bbox_and_ignores_invalid_ellipse(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (100, 80)).save(tmp_path / "frame.jpg")
    annotation_path = tmp_path / "head.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "frame.jpg", "width": 100, "height": 80}
                ],
                "annotations": [
                    {
                        "id": 7344,
                        "image_id": 7,
                        "category_id": 1,
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "ellipse": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "keypoints": [[0.0, 0.0]] * 5,
                        "keypoint_states": [0] * 5,
                    }
                ],
                "categories": [{"id": 1, "name": "Head"}],
            }
        ),
        encoding="utf-8",
    )
    dataset = CocoDetectionDataset(
        annotation_path=annotation_path,
        image_root=tmp_path,
        category_ids=[1],
    )

    _, target = dataset[0]

    assert torch.equal(
        target["boxes"],
        torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
    )
    assert target["labels"].tolist() == [1]
    assert {"ellipses", "keypoints", "keypoint_mask"}.isdisjoint(target)
    assert {"ellipse", "keypoints", "keypoint_states"}.isdisjoint(
        dataset.samples[0]["annotations"][0]
    )


def test_balanced_three_stage_config_uses_exact_shared_split() -> None:
    config = load_runtime_config(
        Path("configs/head_codino/face_pose_drive_v2_balanced_811.yaml")
    )
    assert config["model"]["score_threshold"] == 0.001
    assert config["train"]["seed"] == 42
    for split in ("train", "val", "test"):
        assert config["data"][split]["annotation"].endswith(
            f"face_pose_drive_v2_balanced_811/annotations/face_dino_v1/{split}.json"
        )


def test_shared_head_config_matches_three_stage_square_input() -> None:
    config = load_runtime_config(
        Path("configs/head_codino/shared_three_stage_960.yaml")
    )
    assert config["data"]["image_size"] == 960
    assert config["model"]["score_threshold"] == 0.001
    assert config["train"]["amp_dtype"] == "float16"
    assert config["train"]["grad_scaler_init_scale"] == 1024.0
    assert config["train"]["output_dir"].endswith(
        "shared_head_codino_vits16plus_960_balanced_811"
    )


def test_unfrozen_shared_head_stage_updates_dino_and_uses_new_output() -> None:
    config = load_runtime_config(
        Path("configs/head_codino/shared_three_stage_960_unfrozen.yaml")
    )
    assert config["model"]["freeze_backbone"] is False
    assert config["experiment_objective"] == {
        "type": "cascade_comparison",
        "claim_scope": "upstream_component_for_cascade_selection",
        "joint_fine_tuning": False,
        "final_performance_claim": False,
    }
    assert config["model"]["unfreeze_last_n_blocks"] == 0
    assert config["train"]["backbone_lr"] == pytest.approx(1.0e-5)
    assert config["train"]["validate_each_epoch"] is True
    assert config["train"]["output_dir"].endswith(
        "shared_head_codino_vits16plus_960_balanced_811_unfrozen"
    )


def test_balanced_three_stage_head_dataset_reads_nested_landmark_export() -> None:
    config = load_runtime_config(
        Path("configs/head_codino/face_pose_drive_v2_balanced_811.yaml")
    )
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split)
        image, target = dataset[0]
        assert image.shape[0] == 3
        assert target["boxes"].shape[-1] == 4
        assert target["labels"].eq(1).all()
        assert {"ellipses", "keypoints", "keypoint_mask"}.isdisjoint(target)


def test_head_threshold_metrics_use_one_to_one_iou_matching() -> None:
    import numpy as np

    ground_truth = {
        1: [
            np.asarray((0.0, 0.0, 10.0, 10.0)),
            np.asarray((20.0, 20.0, 30.0, 30.0)),
        ]
    }
    predictions = {
        1: [
            (0.90, np.asarray((0.0, 0.0, 10.0, 10.0))),
            (0.80, np.asarray((0.0, 0.0, 10.0, 10.0))),
            (0.70, np.asarray((20.0, 20.0, 30.0, 30.0))),
        ]
    }
    metrics = metrics_at_threshold(
        ground_truth,
        predictions,
        threshold=0.75,
        iou_threshold=0.50,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_standard_vits_checkpoint_is_explicitly_rejected(tmp_path: Path) -> None:
    path = tmp_path / "standard_vits.pth"
    torch.save({"model": {"blocks.0.mlp.fc1.weight": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match=r"standard DINOv3 ViT-S/16.*not ViT-S\+/16"):
        DINOv3ViTS16PlusBackbone(source_root=DINOV3_SOURCE, weights=path)


def test_geometry_roundtrip_clipping_threshold_sorting_and_no_nms() -> None:
    boxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes)
    assert torch.allclose(
        normalize_boxes_xyxy(boxes, (100, 200)),
        torch.tensor([[0.10, 0.35, 0.10, 0.30]]),
    )
    assert torch.allclose(
        denormalize_boxes_cxcywh(normalize_boxes_xyxy(boxes, (100, 200)), (100, 200)),
        boxes,
    )
    normalized_original = normalize_boxes_xyxy(boxes, (100, 200))
    normalized_padded = original_to_padded_normalized_cxcywh(
        normalized_original, (100, 200), (128, 224)
    )
    assert torch.allclose(
        padded_to_original_normalized_cxcywh(normalized_padded, (100, 200), (128, 224)),
        normalized_original,
    )
    # Identical boxes are deliberately both retained: production uses no NMS.
    logits = torch.tensor([[[2.0], [1.0], [-1.0], [3.0]]])
    normalized = torch.tensor(
        [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.0, 0.1], [2.0, 2.0, 0.1, 0.1]]]
    )
    result = postprocess_head_detections(
        logits,
        normalized,
        torch.tensor([[100, 200]]),
        score_threshold=0.30,
        max_detections=100,
    )[0]
    assert result["scores"].tolist() == sorted(result["scores"].tolist(), reverse=True)
    assert result["boxes"].shape[0] == 2
    assert torch.allclose(result["boxes"][0], torch.tensor([50.0, 25.0, 150.0, 75.0]))
    assert torch.equal(result["labels"], torch.zeros(2, dtype=torch.long))
    capped = postprocess_head_detections(
        torch.full((1, 101, 1), 10.0),
        torch.full((1, 101, 4), 0.5),
        torch.tensor([[100, 200]]),
    )[0]
    assert capped["boxes"].shape[0] == 100
    square = expand_square_roi(boxes, (100, 200), scale=1.20)
    assert torch.allclose(square, torch.tensor([[2.0, 17.0, 38.0, 53.0]]))


def test_compact_codino_configuration_and_parameter_contract() -> None:
    module = CompactCoDINO()
    assert module.num_queries == 100
    assert module.num_feature_levels == 3
    assert module.decoder.num_layers == 3
    assert module.encoder.num_layers == 1
    assert module.dn_query_generator.num_dn_queries == 50
    assert module.dn_query_generator.label_noise_scale == 0.5
    assert module.dn_query_generator.box_noise_scale == 0.4
    assert sum(parameter.numel() for parameter in module.parameters()) == 5_132_436


def test_hungarian_assignment_prefers_better_predictions_and_is_one_to_one() -> None:
    module = CompactCoDINO()
    predictions = InstanceData(
        bboxes=torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 90.0, 90.0]]),
        scores=torch.tensor([[4.0], [3.0]]),
    )
    ground_truth = InstanceData(
        bboxes=torch.tensor([[11.0, 11.0, 31.0, 31.0], [59.0, 59.0, 89.0, 89.0]]),
        labels=torch.zeros(2, dtype=torch.long),
    )
    costs = [
        cost(
            pred_instances=predictions,
            gt_instances=ground_truth,
            img_meta={"img_shape": (100, 100)},
        )
        for cost in module.bbox_head.assigner.match_costs
    ]
    total_cost = torch.stack(costs).sum(0)
    assert total_cost.diagonal().sum() < total_cost.flip(1).diagonal().sum()
    assignment = module.bbox_head.assigner.assign(
        predictions,
        ground_truth,
        img_meta={"img_shape": (100, 100)},
    )
    assert assignment.gt_inds.tolist() == [1, 2]


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator is production-only")
def test_mixed_native_resolution_forward_backward_dn_and_empty_gt() -> None:
    model = HeadCoDINODetector(backbone_weights=None).cuda().train()
    batch = HeadImageBatch.from_images(
        [
            torch.rand(3, 128, 160, device="cuda"),
            torch.rand(3, 95, 127, device="cuda"),
        ]
    )
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 60.0, 70.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)},
    ]
    with torch.autocast("cuda", dtype=torch.float16):
        raw = model(batch, targets=targets, return_raw=True, return_features=True)
        losses = model(batch, targets)
    assert raw["pred_logits"].shape == (2, 100, 1)
    assert raw["pred_boxes"].shape == (2, 100, 4)
    assert ((raw["pred_boxes"] >= 0) & (raw["pred_boxes"] <= 1)).all()
    assert len(raw["aux_outputs"]) == 2
    assert raw["dn_outputs"]["pred_logits"].shape[0] == 3
    num_dn = raw["dn_meta"]["num_denoising_queries"]
    assert raw["_dn_mask"][num_dn:, :num_dn].all()
    assert list(raw["pyramid"]) == ["p3", "p4", "p5", "p6"]
    assert set(losses) == {
        "loss_cls",
        "loss_bbox",
        "loss_giou",
        "d0.loss_cls",
        "d0.loss_bbox",
        "d0.loss_giou",
        "d1.loss_cls",
        "d1.loss_bbox",
        "d1.loss_giou",
        "enc_loss_cls",
        "enc_loss_bbox",
        "enc_loss_giou",
        "dn_loss_cls",
        "dn_loss_bbox",
        "dn_loss_giou",
        "d0.dn_loss_cls",
        "d0.dn_loss_bbox",
        "d0.dn_loss_giou",
        "d1.dn_loss_cls",
        "d1.dn_loss_bbox",
        "d1.dn_loss_giou",
        "loss_total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_total"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(parameter.grad is not None for parameter in model.query_head.downsample.parameters())
    model.eval()
    with torch.inference_mode():
        inference_raw = model(batch, return_raw=True)
    assert "dn_outputs" not in inference_raw
    assert "dn_meta" not in inference_raw


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator is production-only")
def test_same_image_is_consistent_alone_and_in_mixed_batch() -> None:
    torch.manual_seed(7)
    model = HeadCoDINODetector(backbone_weights=None).cuda().eval()
    image = torch.rand(3, 95, 127, device="cuda")
    other = torch.rand(3, 128, 160, device="cuda")
    with torch.inference_mode():
        alone = model([image], return_raw=True)
        mixed = model([image, other], return_raw=True)
    assert torch.allclose(alone["pred_logits"][0], mixed["pred_logits"][0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(alone["pred_boxes"][0], mixed["pred_boxes"][0], atol=2e-5, rtol=2e-5)
