import unittest

import torch

from src.models import build_dino_scrfd_face_detector


def small_config() -> dict:
    return {
        "model": {
            "backbone": {
                "name": "dinov2_vits14",
                "patch_size": 14,
                "embed_dim": 8,
                "pretrained": False,
                "dummy": True,
                "freeze": True,
                "num_prefix_tokens": 1,
            },
            "adapter": {"out_channels": 8, "norm": "gn", "activation": "relu"},
            "neck": {"out_channels": 8, "num_levels": 3, "norm": "gn", "activation": "relu"},
            "head": {"num_classes": 1, "num_keypoints": 5, "head_channels": 8, "num_convs": 1},
        },
        "loss": {"cls_weight": 1.0, "bbox_weight": 2.0, "kpt_weight": 1.0},
        "postprocess": {"score_threshold": 0.01, "nms_threshold": 0.4, "topk": 50, "detections_per_image": 10},
    }


def positive_target() -> dict:
    return {
        "boxes": torch.tensor([[16.0, 14.0, 42.0, 44.0]]),
        "labels": torch.tensor([1]),
        "keypoints": torch.tensor([[[22.0, 24.0], [36.0, 24.0], [29.0, 30.0], [24.0, 38.0], [34.0, 38.0]]]),
        "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        "image_id": "dummy",
        "orig_size": (56, 56),
    }


class DinoSCRFDModelTest(unittest.TestCase):
    def test_forward_shapes_and_freeze_policy(self) -> None:
        model = build_dino_scrfd_face_detector(small_config())
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        for module in (model.adapter, model.fpn, model.head):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))

        outputs = model.forward_raw(torch.zeros(2, 3, 56, 56))
        self.assertEqual(len(outputs["cls_logits"]), 3)
        self.assertEqual(tuple(outputs["cls_logits"][0].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(outputs["bbox_reg"][1].shape), (2, 4, 4, 4))
        self.assertEqual(tuple(outputs["kpt_reg"][2].shape), (2, 10, 2, 2))

    def test_training_forward_returns_loss_dict(self) -> None:
        model = build_dino_scrfd_face_detector(small_config())
        model.train()
        losses = model(torch.zeros(1, 3, 56, 56), [positive_target()])
        self.assertIn("loss_total", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertGreaterEqual(float(losses["num_pos"]), 1.0)

    def test_inference_forward_returns_decoded_detections(self) -> None:
        model = build_dino_scrfd_face_detector(small_config())
        model.eval()
        with torch.no_grad():
            detections = model(torch.zeros(1, 3, 56, 56))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["boxes"].shape[-1], 4)
        self.assertEqual(detections[0]["keypoints"].shape[-2:], (5, 2))

    def test_negative_sample_loss_is_finite(self) -> None:
        model = build_dino_scrfd_face_detector(small_config())
        empty_target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "keypoints": torch.zeros((0, 5, 2)),
            "keypoint_mask": torch.zeros((0, 5), dtype=torch.bool),
            "image_id": "negative",
            "orig_size": (56, 56),
        }
        losses = model(torch.zeros(1, 3, 56, 56), [empty_target])
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertEqual(float(losses["num_pos"]), 0.0)

    def test_missing_keypoints_skip_landmark_loss(self) -> None:
        model = build_dino_scrfd_face_detector(small_config())
        target = positive_target()
        target["keypoint_mask"] = torch.zeros((1, 5), dtype=torch.bool)
        losses = model(torch.zeros(1, 3, 56, 56), [target])
        self.assertEqual(float(losses["loss_kpt"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
