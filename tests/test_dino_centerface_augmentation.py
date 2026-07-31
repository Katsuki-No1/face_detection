import unittest
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config_utils import load_yaml_config
from src.data import SyntheticFaceKeypointDataset, collate_face_keypoint_batch, ellipse_params_to_xyxy
from src.data.ellipse_augmentation import (
    AugmentedFaceKeypointDataset,
    apply_ellipse_border_crop,
    apply_horizontal_flip,
    apply_mixup,
    apply_mosaic,
    apply_photometric,
    transform_target_homography,
)
from src.models.ellipse_target_assigner import CenterFaceEllipseTargetAssigner


CONFIG_DIR = Path("configs/dino_centerface_ellipse_aug_ablation")


def sample_target() -> dict:
    return {
        "boxes": torch.tensor([[10.0, 12.0, 30.0, 40.0]]),
        "ellipses": torch.tensor([[20.0, 26.0, 10.0, 14.0, 0.2]]),
        "labels": torch.tensor([1]),
        "keypoints": torch.tensor([[[14.0, 20.0], [26.0, 20.0], [20.0, 26.0], [16.0, 34.0], [24.0, 34.0]]]),
        "keypoint_mask": torch.tensor([[True, True, True, True, True]]),
        "image_id": "sample",
        "orig_size": (56, 56),
        "size": (56, 56),
        "metadata": {},
    }


class DinoCenterFaceAugmentationTest(unittest.TestCase):
    @staticmethod
    def _first_batch_after_full_epoch(loader: DataLoader) -> tuple[torch.Tensor, list[dict]]:
        batches = list(loader)
        return batches[0]

    def test_flip_swaps_left_right_keypoints(self) -> None:
        image = torch.zeros((3, 56, 56), dtype=torch.float32)
        _flipped_image, target = apply_horizontal_flip(image, sample_target())
        self.assertAlmostEqual(float(target["ellipses"][0, 0]), 36.0)
        expected_box = ellipse_params_to_xyxy(target["ellipses"][0].tolist())
        for observed, expected in zip(target["boxes"][0].tolist(), expected_box):
            self.assertAlmostEqual(observed, expected, places=4)
        self.assertAlmostEqual(float(target["keypoints"][0, 0, 0]), 30.0)
        self.assertAlmostEqual(float(target["keypoints"][0, 1, 0]), 42.0)
        self.assertTrue(bool(target["keypoint_mask"][0, 0]))
        self.assertTrue(bool(target["keypoint_mask"][0, 1]))

    def test_mosaic_keeps_instances_and_moves_centers(self) -> None:
        dataset = SyntheticFaceKeypointDataset(length=1, image_size=56, positive=True)
        samples = [dataset[0], dataset[0], dataset[0], dataset[0]]
        _image, target = apply_mosaic(samples)
        self.assertEqual(tuple(target["ellipses"].shape), (4, 5))
        centers = target["ellipses"][:, :2]
        self.assertTrue(bool((centers[:, 0] >= 0).all()))
        self.assertTrue(bool((centers[:, 0] <= 56).all()))
        self.assertTrue(bool((centers[:, 1] >= 0).all()))
        self.assertTrue(bool((centers[:, 1] <= 56).all()))

    def test_mixup_concatenates_instances(self) -> None:
        dataset = SyntheticFaceKeypointDataset(length=1, image_size=56, positive=True)
        image, target = apply_mixup(dataset[0], dataset[0], rng=random.Random(0))
        self.assertEqual(tuple(image.shape), (3, 56, 56))
        self.assertEqual(tuple(target["boxes"].shape), (2, 4))
        self.assertEqual(tuple(target["ellipses"].shape), (2, 5))
        self.assertIn("mixup_lambda", target["metadata"])

    def test_geometry_augmented_sample_assigns_positive_heatmap(self) -> None:
        base = SyntheticFaceKeypointDataset(length=4, image_size=56, positive=True)
        dataset = AugmentedFaceKeypointDataset(
            base,
            {
                "enabled": True,
                "degrees": 10.0,
                "translate": 0.05,
                "scale": 0.10,
                "shear": 1.0,
                "perspective": 0.0001,
                "fliplr": 1.0,
            },
            seed=7,
        )
        image, target = dataset[0]
        predictions = {
            "center_logits": [torch.zeros((1, 1, 8, 8))],
            "ellipse_reg": [torch.zeros((1, 5, 8, 8))],
            "kpt_reg": [torch.zeros((1, 10, 8, 8))],
            "bbox_reg": [torch.zeros((1, 4, 8, 8))],
        }
        assignments = CenterFaceEllipseTargetAssigner().assign(predictions, [target], tuple(image.shape[-2:]))
        self.assertGreaterEqual(int(assignments["positive_count"].item()), 1)
        self.assertGreater(float(assignments["center_targets"][0].max()), 0.0)

    def test_homography_can_preserve_intersecting_outside_center(self) -> None:
        target = sample_target()
        matrix = torch.tensor(
            [[1.0, 0.0, -25.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ).numpy()

        dropped = transform_target_homography(target, matrix, width=56, height=56)
        preserved = transform_target_homography(
            target,
            matrix,
            width=56,
            height=56,
            allow_outside_center=True,
        )

        self.assertEqual(int(dropped["ellipses"].shape[0]), 0)
        self.assertEqual(int(preserved["ellipses"].shape[0]), 1)
        self.assertLess(float(preserved["ellipses"][0, 0]), 0.0)
        self.assertGreater(float(preserved["boxes"][0, 2]), 0.0)

    def test_border_crop_places_contour_outside_with_center_inside(self) -> None:
        image = torch.zeros((3, 56, 56), dtype=torch.float32)
        _cropped, target = apply_ellipse_border_crop(
            image,
            sample_target(),
            {
                "ellipse_border_crop_center_outside_probability": 0.0,
                "ellipse_border_crop_contour_margin": [0.25, 0.25],
                "ellipse_border_crop_min_scale": 0.50,
            },
            rng=random.Random(0),
        )

        ellipse = target["ellipses"][0]
        box = ellipse_params_to_xyxy(ellipse.tolist())
        self.assertGreaterEqual(float(ellipse[0]), 0.0)
        self.assertLess(float(ellipse[0]), 56.0)
        self.assertGreaterEqual(float(ellipse[1]), 0.0)
        self.assertLess(float(ellipse[1]), 56.0)
        self.assertTrue(box[0] < 0.0 or box[1] < 0.0 or box[2] > 56.0 or box[3] > 56.0)
        debug = target["metadata"]["augmentation_debug"]["ellipse_border_crop"]
        self.assertEqual(debug["mode"], "contour")
        self.assertFalse(debug["center_outside"])
        self.assertTrue(debug["contour_outside"])

    def test_border_crop_places_center_outside_and_keeps_clipped_box(self) -> None:
        image = torch.zeros((3, 56, 56), dtype=torch.float32)
        _cropped, target = apply_ellipse_border_crop(
            image,
            sample_target(),
            {
                "ellipse_border_crop_center_outside_probability": 1.0,
                "ellipse_border_crop_center_margin": [0.10, 0.10],
                "ellipse_border_crop_min_scale": 0.50,
            },
            rng=random.Random(0),
        )

        ellipse = target["ellipses"][0]
        center_outside = (
            float(ellipse[0]) < 0.0
            or float(ellipse[0]) >= 56.0
            or float(ellipse[1]) < 0.0
            or float(ellipse[1]) >= 56.0
        )
        self.assertTrue(center_outside)
        self.assertGreater(float(target["boxes"][0, 2] - target["boxes"][0, 0]), 0.0)
        self.assertGreater(float(target["boxes"][0, 3] - target["boxes"][0, 1]), 0.0)
        self.assertTrue(bool((target["boxes"] >= 0.0).all()))
        self.assertTrue(bool((target["boxes"] <= 56.0).all()))
        debug = target["metadata"]["augmentation_debug"]["ellipse_border_crop"]
        self.assertEqual(debug["mode"], "center")
        self.assertTrue(debug["center_outside"])
        self.assertTrue(debug["contour_outside"])

    def test_blur_photometric_augmentation_changes_image_only(self) -> None:
        image = torch.zeros((3, 56, 56), dtype=torch.float32)
        image[:, :, ::2] = 1.0
        blurred = apply_photometric(image, {"blur": 1.0}, rng=random.Random(0))
        self.assertEqual(tuple(blurred.shape), tuple(image.shape))
        self.assertGreater(float((blurred - image).abs().mean()), 0.01)

    def test_blur_strength_controls_radius(self) -> None:
        image = torch.zeros((3, 56, 56), dtype=torch.float32)
        image[:, :, ::2] = 1.0
        weak = apply_photometric(image, {"blur": 1.0, "blur_radius_min": 0.1, "blur_radius_max": 0.1}, rng=random.Random(0))
        strong = apply_photometric(image, {"blur": 1.0, "blur_radius_min": 2.0, "blur_radius_max": 2.0}, rng=random.Random(0))
        weak_delta = float((weak - image).abs().mean())
        strong_delta = float((strong - image).abs().mean())
        self.assertGreater(strong_delta, weak_delta)

    def test_persistent_workers_observe_epoch_updates(self) -> None:
        dataset = AugmentedFaceKeypointDataset(
            SyntheticFaceKeypointDataset(length=4, image_size=32, positive=True),
            {"enabled": True, "noise": 0.03},
            seed=0,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=1,
            collate_fn=collate_face_keypoint_batch,
        )
        dataset.set_epoch(1, 120)
        epoch_one, _targets_one = self._first_batch_after_full_epoch(loader)
        dataset.set_epoch(2, 120)
        epoch_two, _targets_two = self._first_batch_after_full_epoch(loader)
        self.assertEqual(dataset.epoch, 2)
        self.assertEqual(dataset.total_epochs, 120)
        self.assertFalse(torch.equal(epoch_one, epoch_two))

    def test_persistent_workers_honor_close_mosaic(self) -> None:
        dataset = AugmentedFaceKeypointDataset(
            SyntheticFaceKeypointDataset(length=4, image_size=32, positive=True),
            {"enabled": True, "mosaic": 1.0, "close_mosaic": 20},
            seed=0,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=1,
            collate_fn=collate_face_keypoint_batch,
        )
        dataset.set_epoch(100, 120)
        _images_active, targets_active = self._first_batch_after_full_epoch(loader)
        dataset.set_epoch(101, 120)
        _images_closed, targets_closed = self._first_batch_after_full_epoch(loader)
        self.assertTrue(str(targets_active[0]["image_id"]).startswith("mosaic__"))
        self.assertFalse(str(targets_closed[0]["image_id"]).startswith("mosaic__"))

    def test_ablation_configs_keep_fixed_training_conditions(self) -> None:
        base = load_yaml_config(CONFIG_DIR / "E0_baseline.yaml")
        fixed_keys = ("model", "loss", "inference", "data", "train")
        for path in sorted(CONFIG_DIR.glob("E*.yaml")):
            config = load_yaml_config(path)
            for key in fixed_keys:
                left = dict(config[key]) if key == "train" else config[key]
                right = dict(base[key]) if key == "train" else base[key]
                if key == "train":
                    left.pop("name", None)
                    right.pop("name", None)
                self.assertEqual(left, right, f"{path.name} changed fixed key {key}")
        self.assertEqual(load_yaml_config(CONFIG_DIR / "E7_all.yaml")["augmentation"]["mosaic"], 0.3)
        self.assertEqual(load_yaml_config(CONFIG_DIR / "E7_all.yaml")["augmentation"]["fliplr"], 0.5)
        for name in ("E3_photometric_only", "E5_mosaic_photometric", "E6_geometry_photometric", "E7_all"):
            self.assertEqual(load_yaml_config(CONFIG_DIR / f"{name}.yaml")["augmentation"]["blur"], 0.0)
        self.assertFalse(load_yaml_config(CONFIG_DIR / "E8_blur_weak.yaml")["experiment"]["active"])
        self.assertEqual(load_yaml_config(CONFIG_DIR / "E9_blur_medium.yaml")["augmentation"]["blur_radius_max"], 1.5)
        self.assertEqual(load_yaml_config(CONFIG_DIR / "E9_blur_medium.yaml")["augmentation"]["blur"], 0.20)
        self.assertFalse(load_yaml_config(CONFIG_DIR / "E10_blur_strong.yaml")["experiment"]["active"])


if __name__ == "__main__":
    unittest.main()
