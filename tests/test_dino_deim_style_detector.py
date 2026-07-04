import unittest
from pathlib import Path

import torch

from src.config_utils import load_yaml_config
from src.models import build_detector_from_config, build_dino_deim_style_detector


def small_config(encoder_type: str = "lightweight_transformer") -> dict:
    return {
        "model": {
            "type": "dino_deim_style",
            "num_classes": 1,
            "class_names": ["face"],
            "backbone": {
                "name": "dinov2_vits14",
                "patch_size": 14,
                "embed_dim": 16,
                "pretrained": False,
                "dummy": True,
                "freeze": True,
                "num_prefix_tokens": 1,
            },
            "adapter": {"hidden_dim": 16, "out_scales": ["p3", "p4", "p5"], "norm": "gn", "activation": "relu"},
            "encoder": {"type": encoder_type, "hidden_dim": 16, "num_layers": 1, "num_heads": 4, "dropout": 0.0},
            "decoder": {"hidden_dim": 16, "num_queries": 7, "num_layers": 1, "num_heads": 4, "dropout": 0.0},
        },
        "loss": {"cls_weight": 2.0, "bbox_l1_weight": 5.0, "giou_weight": 2.0, "no_object_weight": 0.1},
        "inference": {"score_thr": 0.01, "use_nms": False, "topk": 7, "detections_per_image": 5},
    }


def positive_target() -> dict:
    return {
        "boxes": torch.tensor([[16.0, 14.0, 42.0, 44.0]]),
        "labels": torch.tensor([1]),
        "keypoints": torch.zeros((1, 5, 2)),
        "keypoint_mask": torch.zeros((1, 5), dtype=torch.bool),
        "image_id": "dummy",
        "orig_size": (56, 56),
        "size": (56, 56),
    }


def negative_target() -> dict:
    return {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "keypoints": torch.zeros((0, 5, 2)),
        "keypoint_mask": torch.zeros((0, 5), dtype=torch.bool),
        "image_id": "negative",
        "orig_size": (56, 56),
        "size": (56, 56),
    }


class DINODeimStyleDetectorTest(unittest.TestCase):
    def test_build_from_generic_config_and_forward_shapes(self) -> None:
        model = build_detector_from_config(small_config())
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        outputs = model.forward_raw(torch.zeros(2, 3, 56, 56))
        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 7, 1))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 7, 4))
        self.assertTrue(bool((outputs["pred_boxes"] >= 0).all().item()))
        self.assertTrue(bool((outputs["pred_boxes"] <= 1).all().item()))

    def test_loss_and_backward_with_positive_target(self) -> None:
        model = build_dino_deim_style_detector(small_config())
        model.train()
        losses = model(torch.zeros(1, 3, 56, 56), [positive_target()])
        self.assertIn("loss_total", losses)
        self.assertIn("loss_giou", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertGreaterEqual(float(losses["num_pos"]), 1.0)
        losses["loss_total"].backward()
        grads = [parameter.grad for parameter in model.decoder.parameters() if parameter.requires_grad]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all().item() for grad in grads))

    def test_negative_sample_loss_is_finite(self) -> None:
        model = build_dino_deim_style_detector(small_config(encoder_type="identity"))
        losses = model(torch.zeros(1, 3, 56, 56), [negative_target()])
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertEqual(float(losses["loss_bbox"].detach()), 0.0)
        self.assertEqual(float(losses["loss_giou"].detach()), 0.0)
        self.assertEqual(float(losses["num_pos"]), 0.0)

    def test_forward_from_cached_dino_feature(self) -> None:
        model = build_dino_deim_style_detector(small_config())
        images = torch.zeros(1, 3, 56, 56)
        with torch.no_grad():
            dino_feature = model.backbone(images)
        losses = model.forward_from_dino_feature(dino_feature, image_size=(56, 56), targets=[positive_target()])
        self.assertTrue(torch.isfinite(losses["loss_total"]))

    def test_dense_o2o_and_mal_add_extra_positive_matches(self) -> None:
        config = small_config(encoder_type="identity")
        config["loss"]["dense_o2o"] = {"enabled": True, "extra_matches_per_gt": 2, "min_iou": 0.0}
        config["loss"]["mal"] = {"enabled": True, "quality_power": 1.0, "min_quality": 0.25, "cls_quality_target": True}
        model = build_dino_deim_style_detector(config)
        losses = model(torch.zeros(1, 3, 56, 56), [positive_target()])
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertIn("match_quality", losses)
        self.assertGreaterEqual(float(losses["num_pos"]), 3.0)

    def test_sta_and_decoder_refinement_keep_bbox_shape(self) -> None:
        config = small_config(encoder_type="identity")
        config["model"]["adapter"]["refine_layers"] = 1
        config["model"]["adapter"]["cross_scale_fusion"] = True
        config["model"]["decoder"]["num_queries"] = 9
        config["model"]["decoder"]["refine_boxes"] = True
        config["model"]["decoder"]["box_refine_steps"] = 1
        model = build_dino_deim_style_detector(config)
        outputs = model.forward_raw(torch.zeros(1, 3, 56, 56))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 9, 4))
        self.assertTrue(bool((outputs["pred_boxes"] >= 0).all().item()))
        self.assertTrue(bool((outputs["pred_boxes"] <= 1).all().item()))

    def test_inference_returns_bbox_records(self) -> None:
        model = build_dino_deim_style_detector(small_config(encoder_type="identity"))
        model.eval()
        with torch.no_grad():
            detections = model(torch.zeros(1, 3, 56, 56))
        self.assertEqual(len(detections), 1)
        self.assertIn("boxes", detections[0])
        self.assertIn("scores", detections[0])
        self.assertIn("labels", detections[0])
        self.assertEqual(detections[0]["boxes"].shape[-1], 4)

    def test_e1_to_e5_configs_progressively_enable_features(self) -> None:
        root = Path("configs/dino_deim_style_ablation")
        for target in ("face", "head"):
            e1 = load_yaml_config(root / target / "E1_baseline.yaml")
            e2 = load_yaml_config(root / target / "E2_stronger_aug.yaml")
            e3 = load_yaml_config(root / target / "E3_dense_o2o.yaml")
            e4 = load_yaml_config(root / target / "E4_mal.yaml")
            e5 = load_yaml_config(root / target / "E5_sta_decoder_refine.yaml")
            self.assertFalse(e1["augmentation"]["enabled"])
            self.assertTrue(e2["augmentation"]["enabled"])
            self.assertGreater(e2["augmentation"]["mosaic"], 0.0)
            self.assertGreater(e2["augmentation"]["mixup"], 0.0)
            self.assertTrue(e3["loss"]["dense_o2o"]["enabled"])
            self.assertTrue(e4["loss"]["mal"]["enabled"])
            self.assertTrue(e5["model"]["adapter"]["cross_scale_fusion"])
            self.assertTrue(e5["model"]["decoder"]["refine_boxes"])


if __name__ == "__main__":
    unittest.main()
