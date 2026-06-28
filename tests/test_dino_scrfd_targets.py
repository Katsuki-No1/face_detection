import unittest

import torch

from src.models.target_assigner import CenterFaceTargetAssigner


def predictions() -> dict[str, list[torch.Tensor]]:
    return {
        "cls_logits": [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)],
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
