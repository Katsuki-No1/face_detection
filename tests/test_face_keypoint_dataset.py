import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import FaceKeypointDataset, ellipse_params_to_xyxy, ellipse_to_xyxy


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
            self.assertEqual(tuple(positive["ellipses"].shape), (1, 5))
            self.assertEqual(tuple(positive["keypoints"].shape), (1, 5, 2))
            self.assertTrue(bool(positive["keypoint_mask"].all().item()))
            self.assertEqual(tuple(negative["boxes"].shape), (0, 4))
            self.assertEqual(tuple(negative["ellipses"].shape), (0, 5))

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
            self.assertEqual(target["ellipses"].tolist(), [[50.0, 45.0, 20.0, 10.0, 0.0]])
            self.assertEqual(target["metadata"]["ellipse_metadata"][0]["cx"], 50.0)
            self.assertEqual(len(target["metadata"]["head_boxes"]), 1)

    def test_coco_detection_preserves_head_face_category_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (100, 80)).save(image_dir / "sample.jpg")
            annotation = {
                "images": [{"id": 10, "file_name": "images/train/sample.jpg", "width": 100, "height": 80}],
                "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "face"}],
                "annotations": [
                    {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 15, 40, 30], "area": 1200},
                    {"id": 2, "image_id": 10, "category_id": 2, "bbox": [20, 25, 20, 15], "area": 300},
                ],
            }
            path = root / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            dataset = FaceKeypointDataset(annotation_path=path, annotation_format="coco", image_root=root, category_ids=[1, 2])
            _image, target = dataset[0]
            self.assertEqual(target["labels"].tolist(), [1, 2])
            self.assertEqual(tuple(target["boxes"].shape), (2, 4))

    def test_rotated_ellipse_outer_bbox(self) -> None:
        box = ellipse_to_xyxy({"cx": 0.0, "cy": 0.0, "rx": 10.0, "ry": 20.0, "rotation": 90.0})
        self.assertAlmostEqual(box[0], -20.0, places=5)
        self.assertAlmostEqual(box[2], 20.0, places=5)

    def test_non_uniform_resize_recomputes_rotated_ellipse_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (1280, 720)).save(root / "ellipse.jpg")
            annotation = {
                "records": [
                    {
                        "image_name": "ellipse.jpg",
                        "image_path": str(root / "ellipse.jpg"),
                        "objects": [
                            {
                                "cvat_ellipse": {"cx": 720.0, "cy": 300.0, "rx": 120.0, "ry": 220.0, "rotation": 70.0},
                                "keypoints": {},
                            }
                        ],
                    }
                ]
            }
            path = root / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            dataset = FaceKeypointDataset(annotation_path=path, annotation_format="cvat", image_size=896)
            _image, target = dataset[0]
            expected = ellipse_params_to_xyxy(target["ellipses"][0].tolist())
            actual = target["boxes"][0].tolist()
            for observed, wanted in zip(actual, expected):
                self.assertAlmostEqual(observed, wanted, places=3)


if __name__ == "__main__":
    unittest.main()
