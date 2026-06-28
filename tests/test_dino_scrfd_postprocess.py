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

    def test_fallback_nms_suppresses_overlaps(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [30.0, 30.0, 40.0, 40.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = _fallback_nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
