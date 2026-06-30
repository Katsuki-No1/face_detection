import unittest

import torch

from src.models.target_assigner import CenterFaceTargetAssigner


def predictions() -> dict[str, list[torch.Tensor]]:
    return {
        "cls_logits": [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)],
        "center_offset": [torch.full((1, 2, 8, 8), 0.5), torch.full((1, 2, 4, 4), 0.5), torch.full((1, 2, 2, 2), 0.5)],
        "bbox_reg": [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 2, 2)],
        "kpt_reg": [torch.zeros(1, 10, 8, 8), torch.zeros(1, 10, 4, 4), torch.zeros(1, 10, 2, 2)],
    }


class DinoSCRFDTargetAssignerTest(unittest.TestCase):
    def test_positive_sample_is_assigned(self) -> None:
        assigner = CenterFaceTargetAssigner()
        target = {
            "boxes": torch.tensor([[16.0, 14.0, 42.0, 44.0]]),
            "keypoints": torch.tensor([[[22.0, 24.0], [36.0, 24.0], [29.0, 30.0], [24.0, 38.0], [34.0, 38.0]]]),
            "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        }
        assignments = assigner.assign(predictions(), [target], (56, 56))
        self.assertEqual(int(assignments["positive_count"].item()), 1)
        self.assertEqual(int(assignments["visible_kpt_count"].item()), 10)
        cls_targets = assignments["cls_targets"]
        self.assertTrue(any(bool(level.any().item()) for level in cls_targets))

    def test_non_cell_centered_target_uses_fractional_offset(self) -> None:
        assigner = CenterFaceTargetAssigner()
        preds = {
            "cls_logits": [torch.zeros(1, 1, 7, 5)],
            "center_offset": [torch.full((1, 2, 7, 5), 0.5)],
            "bbox_reg": [torch.zeros(1, 4, 7, 5)],
            "kpt_reg": [torch.zeros(1, 10, 7, 5)],
        }
        target = {
            "boxes": torch.tensor([[13.0, 12.0, 33.0, 22.0]]),
            "keypoints": torch.tensor([[[18.0, 15.0], [28.0, 15.0], [23.0, 17.0], [19.0, 20.0], [27.0, 20.0]]]),
            "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        }
        assignments = assigner.assign(preds, [target], (35, 50))
        self.assertEqual(int(assignments["positive_count"].item()), 1)
        row, col = torch.nonzero(assignments["positive_masks"][0][0, 0], as_tuple=False)[0].tolist()
        self.assertEqual((row, col), (3, 2))
        self.assertTrue(torch.allclose(assignments["center_offset_targets"][0][0, :, row, col], torch.tensor([0.3, 0.4]), atol=1e-6))
        self.assertTrue(torch.allclose(assignments["bbox_targets"][0][0, :, row, col], torch.tensor([10.0, 5.0, 10.0, 5.0])))
        self.assertTrue(torch.allclose(assignments["kpt_targets"][0][0, 0:2, row, col], torch.tensor([-5.0, -2.0])))

    def test_negative_sample_has_no_positive_cells(self) -> None:
        assigner = CenterFaceTargetAssigner()
        target = {"boxes": torch.zeros((0, 4)), "keypoints": torch.zeros((0, 5, 2)), "keypoint_mask": torch.zeros((0, 5), dtype=torch.bool)}
        assignments = assigner.assign(predictions(), [target], (56, 56))
        self.assertEqual(int(assignments["positive_count"].item()), 0)
        self.assertEqual(int(assignments["visible_kpt_count"].item()), 0)

    def test_missing_keypoints_keep_bbox_positive(self) -> None:
        assigner = CenterFaceTargetAssigner()
        target = {
            "boxes": torch.tensor([[16.0, 14.0, 42.0, 44.0]]),
            "keypoints": torch.zeros((1, 5, 2)),
            "keypoint_mask": torch.zeros((1, 5), dtype=torch.bool),
        }
        assignments = assigner.assign(predictions(), [target], (56, 56))
        self.assertEqual(int(assignments["positive_count"].item()), 1)
        self.assertEqual(int(assignments["visible_kpt_count"].item()), 0)


if __name__ == "__main__":
    unittest.main()
