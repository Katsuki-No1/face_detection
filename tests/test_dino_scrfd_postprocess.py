import unittest

import torch

from src.models.postprocess import DinoSCRFDPostProcessor, _fallback_nms


class DinoSCRFDPostprocessTest(unittest.TestCase):
    def test_decode_result_shapes(self) -> None:
        cls = torch.full((1, 1, 4, 4), -10.0)
        cls[0, 0, 1, 1] = 10.0
        bbox = torch.full((1, 4, 4, 4), 5.0)
        kpt = torch.zeros((1, 10, 4, 4))
        processor = DinoSCRFDPostProcessor(score_threshold=0.5, nms_threshold=0.5, topk=10, detections_per_image=5)
        results = processor.decode({"cls_logits": [cls], "bbox_reg": [bbox], "kpt_reg": [kpt]}, (56, 56))
        self.assertEqual(len(results), 1)
        self.assertEqual(tuple(results[0]["boxes"].shape), (1, 4))
        self.assertEqual(tuple(results[0]["scores"].shape), (1,))
        self.assertEqual(tuple(results[0]["keypoints"].shape), (1, 5, 2))
        self.assertTrue(torch.allclose(results[0]["boxes"][0], torch.tensor([16.0, 16.0, 26.0, 26.0])))

    def test_decode_uses_center_offset_and_rectangular_stride(self) -> None:
        cls = torch.full((1, 1, 3, 5), -10.0)
        cls[0, 0, 1, 2] = 10.0
        center_offset = torch.full((1, 2, 3, 5), 0.5)
        center_offset[0, :, 1, 2] = torch.tensor([0.2, 0.7])
        bbox = torch.zeros((1, 4, 3, 5))
        bbox[0, :, 1, 2] = torch.tensor([3.0, 4.0, 5.0, 6.0])
        kpt = torch.zeros((1, 10, 3, 5))
        processor = DinoSCRFDPostProcessor(score_threshold=0.5, nms_threshold=0.5, topk=10, detections_per_image=5)
        results = processor.decode({"cls_logits": [cls], "center_offset": [center_offset], "bbox_reg": [bbox], "kpt_reg": [kpt]}, (30, 50))
        self.assertEqual(len(results), 1)
        self.assertTrue(torch.allclose(results[0]["boxes"][0], torch.tensor([19.0, 13.0, 27.0, 23.0]), atol=1e-6))
        self.assertTrue(torch.allclose(results[0]["keypoints"][0, 0], torch.tensor([22.0, 17.0]), atol=1e-6))

    def test_assign_decode_roundtrip_for_non_cell_centered_target(self) -> None:
        from src.models.target_assigner import CenterFaceTargetAssigner

        predictions = {
            "cls_logits": [torch.full((1, 1, 7, 5), -20.0)],
            "center_offset": [torch.full((1, 2, 7, 5), 0.5)],
            "bbox_reg": [torch.zeros(1, 4, 7, 5)],
            "kpt_reg": [torch.zeros(1, 10, 7, 5)],
        }
        target = {
            "boxes": torch.tensor([[13.0, 12.0, 33.0, 22.0]]),
            "keypoints": torch.tensor([[[18.0, 15.0], [28.0, 15.0], [23.0, 17.0], [19.0, 20.0], [27.0, 20.0]]]),
            "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        }
        assignments = CenterFaceTargetAssigner().assign(predictions, [target], (35, 50))
        row, col = torch.nonzero(assignments["positive_masks"][0][0, 0], as_tuple=False)[0].tolist()
        predictions["cls_logits"][0][0, 0, row, col] = 20.0
        predictions["center_offset"][0][0, :, row, col] = assignments["center_offset_targets"][0][0, :, row, col]
        predictions["bbox_reg"][0][0, :, row, col] = assignments["bbox_targets"][0][0, :, row, col]
        predictions["kpt_reg"][0][0, :, row, col] = assignments["kpt_targets"][0][0, :, row, col]

        processor = DinoSCRFDPostProcessor(score_threshold=0.5, nms_threshold=0.5, topk=10, detections_per_image=5)
        results = processor.decode(predictions, (35, 50))
        self.assertTrue(torch.allclose(results[0]["boxes"][0], target["boxes"][0], atol=1e-6))
        self.assertTrue(torch.allclose(results[0]["keypoints"][0], target["keypoints"][0], atol=1e-6))

    def test_fallback_nms_suppresses_overlaps(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [30.0, 30.0, 40.0, 40.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = _fallback_nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
