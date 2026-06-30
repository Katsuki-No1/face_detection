import unittest

import torch

from src.models import build_dino_centerface_ellipse_detector
from src.models.ellipse_postprocess import DinoCenterFaceEllipsePostProcessor, ellipse_to_bbox_tensor
from src.models.ellipse_target_assigner import CenterFaceEllipseTargetAssigner


def small_config(num_keypoints: int = 5, predict_bbox: bool = True, freeze: bool = True) -> dict:
    return {
        "model": {
            "type": "dino_centerface_ellipse",
            "backbone": {
                "name": "dinov2_vits14",
                "patch_size": 14,
                "embed_dim": 8,
                "pretrained": False,
                "dummy": True,
                "freeze": freeze,
                "num_prefix_tokens": 1,
            },
            "adapter": {"out_channels": 8, "norm": "gn", "activation": "relu"},
            "neck": {"out_channels": 8, "num_levels": 3, "norm": "gn", "activation": "relu"},
            "head": {
                "num_keypoints": num_keypoints,
                "predict_bbox": predict_bbox,
                "predict_theta": True,
                "head_channels": 8,
                "num_convs": 1,
            },
        },
        "loss": {"center": 1.0, "ellipse": 5.0, "keypoint": 1.0, "bbox": 0.5},
        "inference": {"score_threshold": 0.01, "nms_threshold": 0.4, "topk": 50, "detections_per_image": 10},
    }


def positive_target() -> dict:
    return {
        "boxes": torch.tensor([[16.0, 14.0, 42.0, 44.0]]),
        "ellipses": torch.tensor([[29.0, 29.0, 13.0, 15.0, 0.1]]),
        "labels": torch.tensor([1]),
        "keypoints": torch.tensor([[[22.0, 24.0], [36.0, 24.0], [29.0, 30.0], [24.0, 38.0], [34.0, 38.0]]]),
        "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        "image_id": "dummy",
        "orig_size": (56, 56),
    }


def raw_predictions() -> dict[str, list[torch.Tensor]]:
    return {
        "center_logits": [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)],
        "ellipse_reg": [torch.zeros(1, 5, 8, 8), torch.zeros(1, 5, 4, 4), torch.zeros(1, 5, 2, 2)],
        "kpt_reg": [torch.zeros(1, 10, 8, 8), torch.zeros(1, 10, 4, 4), torch.zeros(1, 10, 2, 2)],
        "bbox_reg": [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 2, 2)],
    }


class DinoCenterFaceEllipseTest(unittest.TestCase):
    def test_forward_shapes_and_freeze_policy(self) -> None:
        model = build_dino_centerface_ellipse_detector(small_config())
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        for module in (model.adapter, model.fpn, model.head):
            self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))

        outputs = model.forward_raw(torch.zeros(2, 3, 56, 56))
        self.assertEqual(len(outputs["center_logits"]), 3)
        self.assertEqual(tuple(outputs["center_logits"][0].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(outputs["ellipse_reg"][1].shape), (2, 5, 4, 4))
        self.assertEqual(tuple(outputs["kpt_reg"][2].shape), (2, 10, 2, 2))
        self.assertEqual(tuple(outputs["bbox_reg"][0].shape), (2, 4, 8, 8))
        self.assertTrue(bool((outputs["ellipse_reg"][0][:, 2:4] > 0).all().item()))

    def test_target_assignment_draws_heatmap_and_positive_ellipse(self) -> None:
        assigner = CenterFaceEllipseTargetAssigner()
        assignments = assigner.assign(raw_predictions(), [positive_target()], (56, 56))
        self.assertEqual(int(assignments["positive_count"].item()), 1)
        self.assertEqual(int(assignments["visible_kpt_count"].item()), 10)
        center_targets = assignments["center_targets"]
        self.assertTrue(any(bool(level.max().eq(1.0).item()) for level in center_targets))
        ellipse_targets = assignments["ellipse_targets"]
        positive_masks = assignments["positive_masks"]
        for target, mask in zip(ellipse_targets, positive_masks):
            if bool(mask.any().item()):
                values = target[mask.expand_as(target)].reshape(5, -1)[:, 0]
                self.assertAlmostEqual(float(values[2]), 13.0, places=4)
                self.assertAlmostEqual(float(values[3]), 15.0, places=4)

    def test_training_forward_returns_finite_ellipse_loss(self) -> None:
        model = build_dino_centerface_ellipse_detector(small_config())
        model.train()
        losses = model(torch.zeros(1, 3, 56, 56), [positive_target()])
        self.assertIn("loss_ellipse", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertGreaterEqual(float(losses["num_pos"]), 1.0)

    def test_forward_from_dino_feature_returns_loss_dict(self) -> None:
        model = build_dino_centerface_ellipse_detector(small_config())
        model.train()
        images = torch.zeros(1, 3, 56, 56)
        with torch.no_grad():
            dino_feature = model.backbone(images)
        losses = model.forward_from_dino_feature(dino_feature, image_size=(56, 56), targets=[positive_target()])
        self.assertIn("loss_total", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))

    def test_decode_returns_image_coordinate_ellipses(self) -> None:
        center_logits = torch.full((1, 1, 4, 4), -10.0)
        center_logits[0, 0, 1, 2] = 10.0
        ellipse_reg = torch.zeros((1, 5, 4, 4))
        ellipse_reg[0, :, 1, 2] = torch.tensor([0.25, 0.75, 8.0, 6.0, 0.2])
        kpt_reg = torch.zeros((1, 10, 4, 4))
        bbox_reg = torch.full((1, 4, 4, 4), 5.0)
        processor = DinoCenterFaceEllipsePostProcessor(score_threshold=0.5, nms_threshold=0.5, topk=10, detections_per_image=5)
        results = processor.decode(
            {"center_logits": [center_logits], "ellipse_reg": [ellipse_reg], "kpt_reg": [kpt_reg], "bbox_reg": [bbox_reg]},
            (56, 56),
        )
        self.assertEqual(tuple(results[0]["ellipses"].shape), (1, 5))
        self.assertAlmostEqual(float(results[0]["ellipses"][0, 0]), 31.5, places=4)
        self.assertAlmostEqual(float(results[0]["ellipses"][0, 1]), 24.5, places=4)
        self.assertEqual(tuple(results[0]["keypoints"].shape), (1, 5, 2))
        self.assertEqual(tuple(results[0]["bboxes"].shape), (1, 4))

    def test_decode_can_use_ellipse_derived_bbox(self) -> None:
        center_logits = torch.full((1, 1, 4, 4), -10.0)
        center_logits[0, 0, 1, 2] = 10.0
        ellipse_reg = torch.zeros((1, 5, 4, 4))
        ellipse_reg[0, :, 1, 2] = torch.tensor([0.25, 0.75, 8.0, 6.0, 0.2])
        kpt_reg = torch.zeros((1, 10, 4, 4))
        bbox_reg = torch.full((1, 4, 4, 4), 40.0)
        processor = DinoCenterFaceEllipsePostProcessor(
            score_threshold=0.5,
            nms_threshold=0.5,
            topk=10,
            detections_per_image=5,
            bbox_source="ellipse",
        )
        results = processor.decode(
            {"center_logits": [center_logits], "ellipse_reg": [ellipse_reg], "kpt_reg": [kpt_reg], "bbox_reg": [bbox_reg]},
            (56, 56),
        )
        expected_boxes = ellipse_to_bbox_tensor(results[0]["ellipses"], image_size=(56, 56))
        self.assertTrue(torch.allclose(results[0]["bboxes"], expected_boxes))
        self.assertLess(float(results[0]["bboxes"][0, 0]), float(results[0]["ellipses"][0, 0]))
        self.assertGreater(float(results[0]["bboxes"][0, 2]), float(results[0]["ellipses"][0, 0]))

    def test_keypoint_count_is_configurable(self) -> None:
        model = build_dino_centerface_ellipse_detector(small_config(num_keypoints=3, predict_bbox=False))
        outputs = model.forward_raw(torch.zeros(1, 3, 56, 56))
        self.assertEqual(tuple(outputs["kpt_reg"][0].shape), (1, 6, 8, 8))
        self.assertNotIn("bbox_reg", outputs)
        losses = model(torch.zeros(1, 3, 56, 56), [positive_target()])
        self.assertTrue(torch.isfinite(losses["loss_total"]))


if __name__ == "__main__":
    unittest.main()
