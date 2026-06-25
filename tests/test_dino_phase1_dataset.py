import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.phase1_dino.dataset import box_iou, clip_box, generate_phase1_crops


class DinoPhase1DatasetTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        data_root = root / "dataset"
        (data_root / "images" / "test").mkdir(parents=True)
        (data_root / "labels" / "test").mkdir(parents=True)
        Image.new("RGB", (100, 100), (30, 30, 30)).save(data_root / "images" / "test" / "positive.jpg")
        Image.new("RGB", (100, 100), (120, 120, 120)).save(data_root / "images" / "test" / "negative.jpg")
        keypoints = [
            0.42,
            0.45,
            2,
            0.58,
            0.45,
            2,
            0.50,
            0.52,
            2,
            0.45,
            0.60,
            2,
            0.55,
            0.60,
            2,
        ]
        line = " ".join(str(value) for value in [0, 0.5, 0.5, 0.4, 0.4, *keypoints])
        (data_root / "labels" / "test" / "positive.txt").write_text(line + "\n", encoding="utf-8")
        (data_root / "labels" / "test" / "negative.txt").write_text("", encoding="utf-8")
        (data_root / "face_pose.yaml").write_text(
            f"path: {data_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: face\nkpt_shape: [5, 3]\n",
            encoding="utf-8",
        )
        return data_root

    def test_clip_box(self) -> None:
        self.assertEqual(clip_box([-10, -5, 20, 30], 100, 100), [0.0, 0.0, 20.0, 30.0])
        self.assertIsNone(clip_box([-10, -5, -1, 30], 100, 100))

    def test_generate_crop_metadata_from_dummy_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = self.make_dataset(Path(directory))
            config = {
                "data": {"data_root": str(data_root), "split": "test", "seed": 7, "max_images": 2},
                "crops": {
                    "use_face_crop": True,
                    "use_head_crop": False,
                    "use_eye_crop": True,
                    "use_background_crop": True,
                    "use_hard_negative_crop": True,
                    "background_crops_per_image": 1,
                    "hard_negative_crops_per_image": 1,
                    "crop_padding_ratio": 0.1,
                    "min_crop_size": 16,
                    "max_background_iou": 0.01,
                },
            }
            crops, images = generate_phase1_crops(config)
            self.assertEqual(len(images), 2)
            crop_types = {crop["crop_type"] for crop in crops}
            self.assertIn("face", crop_types)
            self.assertIn("eye_region", crop_types)
            self.assertIn("background", crop_types)
            self.assertIn("hard_negative", crop_types)
            face_box = [30.0, 30.0, 70.0, 70.0]
            for crop in crops:
                if crop["crop_type"] in {"background", "hard_negative"} and crop["image_id"] == "positive":
                    self.assertLessEqual(box_iou(face_box, crop["bbox_xyxy"]), 0.01)
            required = {
                "image_id",
                "image_path",
                "crop_id",
                "crop_type",
                "label",
                "bbox_xyxy",
                "original_width",
                "original_height",
                "face_size_category",
                "tags",
                "split",
            }
            self.assertTrue(required.issubset(crops[0]))


if __name__ == "__main__":
    unittest.main()
