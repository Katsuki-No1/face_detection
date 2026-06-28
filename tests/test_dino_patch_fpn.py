import unittest

import torch

from src.dino_patch_fpn import (
    CNNAdapter,
    DenseFaceKeypointHead,
    DenseFaceKeypointLoss,
    DinoPatchFeatureMapBackbone,
    DinoPatchFPNFaceKeypointModel,
    FakeTokenBackbone,
    PatchTokenToFeatureMap,
    SimpleFPN,
    dummy_face_target,
    parse_yolo_pose_label_line,
)


class DinoPatchFPNTest(unittest.TestCase):
    def test_patch_token_reshape(self) -> None:
        tokens = torch.zeros(2, 32 * 32, 8)
        converter = PatchTokenToFeatureMap(14)
        feature = converter(tokens, image_size=(448, 448))
        self.assertEqual(tuple(feature.shape), (2, 8, 32, 32))

    def test_prefix_token_exclusion(self) -> None:
        tokens = torch.zeros(2, 1 + 32 * 32, 8)
        converter = PatchTokenToFeatureMap(14, num_prefix_tokens=1)
        feature = converter(tokens, image_size=(448, 448))
        self.assertEqual(tuple(feature.shape), (2, 8, 32, 32))

    def test_patch_only_token_key_does_not_drop_prefix_again(self) -> None:
        fake = FakeTokenBackbone(embed_dim=8, patch_size=14, output_key="x_norm_patchtokens")
        wrapper = DinoPatchFeatureMapBackbone(
            fake,
            patch_size=14,
            num_prefix_tokens=1,
            token_key="x_norm_patchtokens",
        )
        feature = wrapper(torch.zeros(2, 3, 448, 448))
        self.assertEqual(tuple(feature.shape), (2, 8, 32, 32))

    def test_token_count_mismatch_has_clear_error(self) -> None:
        tokens = torch.zeros(2, 32 * 32 - 1, 8)
        converter = PatchTokenToFeatureMap(14)
        with self.assertRaisesRegex(ValueError, "Patch token count mismatch"):
            converter(tokens, image_size=(448, 448))

    def test_cnn_adapter_shape(self) -> None:
        adapter = CNNAdapter(8, 16, num_layers=2)
        feature = adapter(torch.zeros(2, 8, 32, 32))
        self.assertEqual(tuple(feature.shape), (2, 16, 32, 32))

    def test_simple_fpn_shape(self) -> None:
        fpn = SimpleFPN(16, 16)
        outputs = fpn(torch.zeros(2, 16, 32, 32))
        self.assertEqual(set(outputs), {"p2", "p3", "p4", "p5"})
        self.assertEqual(tuple(outputs["p2"].shape), (2, 16, 64, 64))
        self.assertEqual(tuple(outputs["p3"].shape), (2, 16, 32, 32))
        self.assertEqual(tuple(outputs["p4"].shape), (2, 16, 16, 16))
        self.assertEqual(tuple(outputs["p5"].shape), (2, 16, 8, 8))

    def test_dense_head_output_shape(self) -> None:
        head = DenseFaceKeypointHead(16, num_classes=2, num_keypoints=5)
        outputs = head({"p3": torch.zeros(2, 16, 32, 32)})
        self.assertEqual(tuple(outputs["p3"]["cls_logits"].shape), (2, 2, 32, 32))
        self.assertEqual(tuple(outputs["p3"]["bbox_reg"].shape), (2, 4, 32, 32))
        self.assertEqual(tuple(outputs["p3"]["centerness"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(outputs["p3"]["keypoints"].shape), (2, 15, 32, 32))

    def test_integrated_model_forward_smoke(self) -> None:
        fake = FakeTokenBackbone(embed_dim=16, patch_size=14, num_prefix_tokens=1, output_key="tokens")
        model = DinoPatchFPNFaceKeypointModel(
            fake,
            patch_size=14,
            dino_embed_dim=16,
            num_prefix_tokens=1,
            adapter_out_channels=16,
            fpn_out_channels=16,
        )
        model.eval()
        with torch.no_grad():
            features = model.forward_features(torch.zeros(2, 3, 448, 448))
            outputs = model(torch.zeros(2, 3, 448, 448))
        self.assertEqual(tuple(features["token_map"].shape), (2, 16, 32, 32))
        self.assertEqual(tuple(outputs["p2"]["cls_logits"].shape), (2, 2, 64, 64))
        self.assertEqual(tuple(outputs["p5"]["keypoints"].shape), (2, 15, 8, 8))

    def test_optional_ellipse_head_on_off(self) -> None:
        head_without_ellipse = DenseFaceKeypointHead(16, use_ellipse=False)
        outputs = head_without_ellipse({"p3": torch.zeros(1, 16, 8, 8)})
        self.assertNotIn("ellipse", outputs["p3"])

        head_with_ellipse = DenseFaceKeypointHead(16, use_ellipse=True)
        outputs = head_with_ellipse({"p3": torch.zeros(1, 16, 8, 8)})
        self.assertEqual(tuple(outputs["p3"]["ellipse"].shape), (1, 5, 8, 8))

    def test_loss_forward_with_dummy_target(self) -> None:
        head = DenseFaceKeypointHead(8, use_ellipse=True)
        predictions = head({"p3": torch.zeros(1, 8, 8, 8)})
        losses = DenseFaceKeypointLoss()(predictions, [dummy_face_target()])
        self.assertIn("loss_total", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))

    def test_parse_one_yolo_pose_label(self) -> None:
        line = "0 0.5 0.5 0.25 0.5 0.4 0.45 2 0.6 0.45 2 0.5 0.5 2 0.45 0.6 2 0.55 0.6 2"
        target = parse_yolo_pose_label_line(line, width=200, height=100)
        self.assertEqual(target["labels"], [1])
        self.assertEqual(target["boxes"], [[75.0, 25.0, 125.0, 75.0]])
        self.assertEqual(target["keypoints"][0][2], [100.0, 50.0, 2.0])


if __name__ == "__main__":
    unittest.main()
