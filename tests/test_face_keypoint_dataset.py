import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import FaceKeypointDataset, ellipse_to_xyxy


class FaceKeypointDatasetTest(unittest.TestCase):
    def test_yolo_pose_dataset_reads_positive_and_negative_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            Image.new("RGB", (100, 80)).save(root / "images" / "train" / "positive.jpg")
            Image.new("RGB", (100, 80)).save(root / "images" / "train" / "negative.jpg")
            line = "0 0.5 0.5 0.4 0.5 0.4 0.4 2 0.6 0.4 2 0.5 0.5 2 0.45 0.6 2 0.55 0.6 2"
            (root / "labels" / "train" / "positive.txt").write_text(line + "\n", encoding="utf-8")
            (root / "labels" / "train" / "negative.txt").write_text("", encoding="utf-8")
            dataset = FaceKeypointDataset(data_root=root, split="train", image_size=56)
            self.assertEqual(len(dataset), 2)
            targets = [dataset[index][1] for index in range(len(dataset))]
            positive = next(target for target in targets if target["boxes"].shape[0] == 1)
            negative = next(target for target in targets if target["boxes"].shape[0] == 0)
            self.assertEqual(tuple(positive["boxes"].shape), (1, 4))
            self.assertEqual(tuple(positive["keypoints"].shape), (1, 5, 2))
            self.assertTrue(bool(positive["keypoint_mask"].all().item()))
            self.assertEqual(tuple(negative["boxes"].shape), (0, 4))

    def test_cvat_ellipse_becomes_bbox_and_keeps_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 100)).save(root / "ellipse.jpg")
            annotation = {
                "records": [
                    {
                        "image_name": "ellipse.jpg",
                        "image_path": str(root / "ellipse.jpg"),
                        "width": 100,
                        "height": 100,
                        "faces": [
                            {
                                "type": "ellipse",
                                "cvat_ellipse": {"cx": 50.0, "cy": 45.0, "rx": 20.0, "ry": 10.0, "rotation": 0.0},
                            }
                        ],
                        "heads": [{"bbox": {"xtl": 30, "ytl": 20, "xbr": 75, "ybr": 90}}],
                    }
                ]
            }
            path = root / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            dataset = FaceKeypointDataset(annotation_path=path, annotation_format="cvat")
            _image, target = dataset[0]
            self.assertEqual(target["boxes"].tolist(), [[30.0, 35.0, 70.0, 55.0]])
            self.assertEqual(target["metadata"]["ellipse_metadata"][0]["cx"], 50.0)
            self.assertEqual(len(target["metadata"]["head_boxes"]), 1)

    def test_rotated_ellipse_outer_bbox(self) -> None:
        box = ellipse_to_xyxy({"cx": 0.0, "cy": 0.0, "rx": 10.0, "ry": 20.0, "rotation": 90.0})
        self.assertAlmostEqual(box[0], -20.0, places=5)
        self.assertAlmostEqual(box[2], 20.0, places=5)


if __name__ == "__main__":
    unittest.main()
