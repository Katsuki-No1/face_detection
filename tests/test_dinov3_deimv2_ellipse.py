from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from script.script_data.export_face_pose_ellipse_coco import export_ellipse_coco
from src.config_utils import load_yaml_config
from src.data import FaceKeypointDataset
from src.models.dinov3_deimv2.official import _load_pretrained_checkpoint
from src.models.dinov3_deimv2_ellipse import (
    EllipseDEIMCriterion,
    canonicalize_ellipses,
    ellipse_to_aabb,
    scale_ellipses_anisotropic,
)


CONFIG_PATH = Path(
    "configs/dinov3_deimv2/"
    "dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml"
)
CLEAN_BALANCED_CONFIG_PATH = Path(
    "configs/dinov3_deimv2/"
    "dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_960_ellipse_from_coco.yaml"
)


class _NoopMatcher(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        device = outputs["pred_boxes"].device
        return {
            "indices": [
                (
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                )
                for _ in targets
            ]
        }


def test_axis_aligned_box_does_not_determine_unique_ellipse() -> None:
    first = torch.tensor([[0.0, 0.0, 10.0, 5.0, 0.0]])
    theta = math.radians(15.0)
    cos_squared = math.cos(theta) ** 2
    sin_squared = math.sin(theta) ** 2
    determinant = cos_squared**2 - sin_squared**2
    major_squared = (100.0 * cos_squared - 25.0 * sin_squared) / determinant
    minor_squared = (25.0 * cos_squared - 100.0 * sin_squared) / determinant
    second = torch.tensor([[0.0, 0.0, math.sqrt(major_squared), math.sqrt(minor_squared), theta]])

    assert not torch.allclose(first, second)
    assert torch.allclose(ellipse_to_aabb(first), ellipse_to_aabb(second), atol=1.0e-5)


def test_ellipse_canonicalization_swaps_axes_and_wraps_modulo_pi() -> None:
    raw = torch.tensor([[4.0, 5.0, 2.0, 6.0, 0.0], [4.0, 5.0, 6.0, 2.0, math.pi]])
    canonical = canonicalize_ellipses(raw)

    assert torch.allclose(canonical[:, 2], torch.tensor([6.0, 6.0]))
    assert torch.allclose(canonical[:, 3], torch.tensor([2.0, 2.0]))
    assert canonical[0, 4].item() == pytest.approx(-math.pi * 0.5)
    assert canonical[1, 4].item() == pytest.approx(0.0, abs=1.0e-6)


def test_anisotropic_scale_preserves_center_and_refits_axes() -> None:
    ellipse = torch.tensor([[10.0, 20.0, 6.0, 2.0, 0.0]])
    scaled = scale_ellipses_anisotropic(ellipse, scale_x=2.0, scale_y=0.5)

    assert scaled[0, 0].item() == pytest.approx(20.0)
    assert scaled[0, 1].item() == pytest.approx(10.0)
    assert scaled[0, 2].item() == pytest.approx(12.0)
    assert scaled[0, 3].item() == pytest.approx(1.0)


def test_empty_ellipse_loss_is_finite() -> None:
    criterion = EllipseDEIMCriterion(
        matcher=_NoopMatcher(),
        weight_dict={},
        losses=[],
        num_classes=1,
        use_uni_set=False,
    )
    predictions = torch.zeros((2, 4, 5), requires_grad=True)
    outputs = {"pred_ellipses": predictions}
    targets = [
        {"ellipses": torch.zeros((0, 5))},
        {"ellipses": torch.zeros((0, 5))},
    ]
    empty = torch.zeros(0, dtype=torch.long)
    losses = criterion._ellipse_losses(outputs, targets, [(empty, empty), (empty, empty)])

    assert all(torch.isfinite(value) for value in losses.values())
    assert all(value.item() == pytest.approx(0.0) for value in losses.values())


def test_ellipse_config_is_additive_and_uses_ellipse_monitor() -> None:
    config = load_yaml_config(CONFIG_PATH)

    assert config["model"]["type"] == "dinov3_deimv2_ellipse"
    assert config["model"]["decoder"]["num_layers"] == 4
    assert config["data"]["annotation_path"].endswith(
        "dataset_adapters/face_pose_drive_v2_balanced_811/face_keypoints_train.json"
    )
    assert config["data"]["dataset_version"] == "face_pose_drive_v2_balanced_811"
    assert config["train"]["early_stopping"]["monitor"] == "val/ellipse_mAP50_95"
    assert config["train"]["epochs"] == 70


def test_balanced_ellipse_config_starts_from_public_coco_weights() -> None:
    config = load_yaml_config(CLEAN_BALANCED_CONFIG_PATH)
    initialization = config["model"]["init_checkpoint"]

    assert config["data"]["dataset_version"] == "face_pose_drive_v2_balanced_811"
    assert initialization["path"].endswith("checkpoint/DEIMv2/deimv2_dinov3_l_coco.safetensors")
    assert initialization["prefer_ema"] is False
    assert initialization["skip_class_heads"] is True
    assert config["experiment_name"].endswith("_balanced_811_960_ellipse_from_coco")
    assert config["augmentation"]["ellipse_border_crop"] == pytest.approx(0.20)
    assert config["augmentation"]["ellipse_border_crop_center_outside_probability"] == pytest.approx(0.50)
    assert config["augmentation"]["preserve_outside_ellipse_center"] is True


def test_coco_loader_prefers_explicit_ellipse(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.json"
    image_path = tmp_path / "sample.png"
    from PIL import Image

    Image.new("RGB", (32, 24)).save(image_path)
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image_path.name, "width": 32, "height": 24}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [2.0, 3.0, 10.0, 12.0],
                        "area": 120.0,
                        "iscrowd": 0,
                        "ellipse": [7.0, 9.0, 7.0, 3.0, 0.4],
                    }
                ],
                "categories": [{"id": 2, "name": "face"}],
            }
        ),
        encoding="utf-8",
    )
    dataset = FaceKeypointDataset(
        annotation_path=annotation_path,
        annotation_format="coco",
        image_root=tmp_path,
        category_ids=[2],
    )

    _image, target = dataset[0]
    assert torch.allclose(target["ellipses"], torch.tensor([[7.0, 9.0, 7.0, 3.0, 0.4]]))


def test_ellipse_exporter_keeps_source_coco_unchanged(tmp_path: Path) -> None:
    records = []
    source_text = {}
    for split in ("train", "val", "test"):
        file_name = f"images/{split}/{split}_sample.jpg"
        records.append(
            {
                "output_image": file_name,
                "image_name": f"{split}_sample.jpg",
                "split": split,
                "objects": [
                    {
                        "cvat_ellipse": {
                            "cx": 20.0,
                            "cy": 15.0,
                            "rx": 8.0,
                            "ry": 4.0,
                            "rotation": 0.0,
                        }
                    }
                ],
            }
        )
        coco = {
            "images": [{"id": 1, "file_name": file_name, "width": 40, "height": 30}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [12.0, 11.0, 16.0, 8.0],
                    "area": 128.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 2, "name": "face"}],
        }
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(coco), encoding="utf-8")
        source_text[split] = path.read_text(encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"records": records}), encoding="utf-8")

    summary = export_ellipse_coco(
        tmp_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "ellipse_annotations",
        expected_annotations=None,
    )

    assert summary["totals"]["ellipses"] == 3
    for split in ("train", "val", "test"):
        assert (tmp_path / f"{split}.json").read_text(encoding="utf-8") == source_text[split]
        converted = json.loads((tmp_path / "ellipse_annotations" / f"{split}.json").read_text(encoding="utf-8"))
        assert converted["annotations"][0]["ellipse"] == pytest.approx([20.0, 15.0, 8.0, 4.0, 0.0])


def test_pretrained_loader_can_prefer_nested_ema(tmp_path: Path) -> None:
    model = nn.Linear(2, 1, bias=False)
    checkpoint = {
        "model": {"weight": torch.ones((1, 2))},
        "ema": {"module": {"weight": torch.full((1, 2), 3.0)}},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    info = _load_pretrained_checkpoint(
        model,  # type: ignore[arg-type]
        {"path": path, "prefer_ema": True, "skip_class_heads": False, "verbose": False},
    )

    assert info["loaded"] == 1
    assert torch.equal(model.weight, torch.full_like(model.weight, 3.0))
