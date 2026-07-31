from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ellipse_error_model import EllipseErrorSampler
from src.data.face_crop_cache import (
    canonicalize_ellipse,
    ellipse_aligned_rectangle,
    load_detector_prediction_cache,
    match_annotations_to_detector_crops,
    perturb_ellipse,
)

from .geometry import CropTransform, build_crop_transform, original_to_normalized_crop, warp_crop


LANDMARK_NAMES = ("left_eye", "right_eye", "nose_tip", "left_mouth_corner", "right_mouth_corner")
FLIP_INDEX = (1, 0, 2, 4, 3)


@dataclass(frozen=True)
class CropAugmentation:
    roi_geometry: str = "long_axis_square"
    scale_factor: float = 1.25
    scale_jitter: float = 0.20
    center_jitter: float = 0.08
    rotation_deg: float = 25.0
    horizontal_flip_probability: float = 0.50
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.15
    blur_probability: float = 0.10
    blur_kernel_max: int = 5
    occlusion_probability: float = 0.10
    occlusion_area: tuple[float, float] = (0.02, 0.12)


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def _resolve_image(root: Path, file_name: str) -> Path:
    path = Path(file_name)
    candidates = [path] if path.is_absolute() else [root / path, root / "images" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to resolve image {file_name!r} below {root}")


def _parse_keypoints(
    values: Sequence[float],
    count: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((count, 2), dtype=np.float32)
    labeled = np.zeros((count,), dtype=np.bool_)
    visible = np.zeros((count,), dtype=np.bool_)
    for index in range(count):
        offset = index * 3
        if offset + 2 >= len(values):
            continue
        points[index] = [float(values[offset]), float(values[offset + 1])]
        visibility = float(values[offset + 2])
        labeled[index] = visibility > 0
        visible[index] = visibility > 1
    return points, labeled, visible


def _photometric(image: np.ndarray, augmentation: CropAugmentation) -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    if augmentation.brightness > 0:
        value += random.uniform(-augmentation.brightness, augmentation.brightness)
    if augmentation.contrast > 0:
        mean = value.mean(axis=(0, 1), keepdims=True)
        value = mean + (value - mean) * random.uniform(1.0 - augmentation.contrast, 1.0 + augmentation.contrast)
    if augmentation.saturation > 0:
        gray = value.mean(axis=2, keepdims=True)
        value = gray + (value - gray) * random.uniform(1.0 - augmentation.saturation, 1.0 + augmentation.saturation)
    value = np.clip(value, 0.0, 1.0)
    result = np.round(value * 255.0).astype(np.uint8)
    if augmentation.blur_probability > 0 and random.random() < augmentation.blur_probability:
        maximum = max(3, int(augmentation.blur_kernel_max) | 1)
        choices = list(range(3, maximum + 1, 2))
        kernel = random.choice(choices)
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)
    return result


def _random_occlusion(
    image: np.ndarray,
    augmentation: CropAugmentation,
) -> tuple[np.ndarray, np.ndarray]:
    occlusion_mask = np.zeros(image.shape[:2], dtype=np.bool_)
    if augmentation.occlusion_probability <= 0 or random.random() >= augmentation.occlusion_probability:
        return image, occlusion_mask
    height, width = image.shape[:2]
    low, high = augmentation.occlusion_area
    area = random.uniform(low, high) * height * width
    aspect = random.uniform(0.5, 2.0)
    box_width = min(width, max(1, int(round((area * aspect) ** 0.5))))
    box_height = min(height, max(1, int(round((area / aspect) ** 0.5))))
    x1 = random.randint(0, max(0, width - box_width))
    y1 = random.randint(0, max(0, height - box_height))
    fill = np.asarray([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    result = image.copy()
    result[y1 : y1 + box_height, x1 : x1 + box_width] = fill
    occlusion_mask[y1 : y1 + box_height, x1 : x1 + box_width] = True
    return result, occlusion_mask


def image_to_tensor(image: np.ndarray, *, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).contiguous() / 255.0
    mean_tensor = torch.tensor(mean, dtype=value.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=value.dtype).view(3, 1, 1)
    return (value - mean_tensor) / std_tensor.clamp_min(1e-6)


class TopDownFaceLandmarkDataset(Dataset):
    """One face per sample using an augmented GT-ellipse crop by default."""

    def __init__(
        self,
        *,
        annotation_path: str | Path,
        image_root: str | Path,
        input_size: int = 256,
        training: bool = False,
        augmentation: CropAugmentation | None = None,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std: Sequence[float] = (1.0, 1.0, 1.0),
        category_ids: Sequence[int] | None = None,
        require_visible_keypoint: bool = True,
        ellipse_error_model: str | Path | None = None,
        detector_crop_cache: str | Path | None = None,
        detector_matching_iou: float = 0.50,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.augmentation = augmentation or CropAugmentation()
        if self.augmentation.roi_geometry != "long_axis_square":
            raise ValueError(
                "TopDownFaceLandmarkDataset requires "
                "roi_geometry=long_axis_square; anisotropic warping is disabled"
            )
        self.ellipse_error_sampler = (
            EllipseErrorSampler.from_path(ellipse_error_model)
            if ellipse_error_model is not None
            else None
        )
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        allowed = {int(value) for value in category_ids} if category_ids else None
        coco = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in coco.get("images", [])}
        crop_assignments: dict[int, dict[str, Any]] | None = None
        self.detector_crop_report: dict[str, Any] | None = None
        if detector_crop_cache is not None:
            if allowed is None:
                raise ValueError(
                    "category_ids are required with detector crop training"
                )
            _metadata, predictions = load_detector_prediction_cache(
                detector_crop_cache
            )
            crop_assignments, self.detector_crop_report = (
                match_annotations_to_detector_crops(
                    coco,
                    predictions,
                    category_ids=tuple(allowed),
                    minimum_iou=float(detector_matching_iou),
                    require_ellipse=True,
                )
            )
        samples: list[dict[str, Any]] = []
        for annotation in coco.get("annotations", []):
            if allowed is not None and int(annotation.get("category_id", -1)) not in allowed:
                continue
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                continue
            keypoints, labeled, visible = _parse_keypoints(
                annotation.get("keypoints", [])
            )
            if require_visible_keypoint and not bool(labeled.any()):
                continue
            image = images.get(int(annotation["image_id"]))
            if image is None:
                continue
            crop_assignment = (
                crop_assignments.get(int(annotation.get("id", -1)))
                if crop_assignments is not None
                else None
            )
            if crop_assignments is not None and crop_assignment is None:
                continue
            gt_bbox = xywh_to_xyxy(bbox)
            raw_ellipse = annotation.get("ellipse")
            gt_ellipse = (
                canonicalize_ellipse(raw_ellipse)
                if isinstance(raw_ellipse, (list, tuple))
                and len(raw_ellipse) == 5
                else None
            )
            if crop_assignment is None and gt_ellipse is None:
                raise ValueError(
                    "GT-ellipse crop training requires every usable annotation "
                    f"to contain [cx, cy, major, minor, theta]: id="
                    f"{annotation.get('id')}"
                )
            gt_crop_bbox, gt_crop_rotation = (
                ellipse_aligned_rectangle(gt_ellipse)
                if gt_ellipse is not None
                else (gt_bbox, 0.0)
            )
            samples.append(
                {
                    "annotation": annotation,
                    "image": image,
                    "image_width": float(image["width"]),
                    "image_height": float(image["height"]),
                    "image_path": _resolve_image(self.image_root, str(image["file_name"])),
                    "bbox_xyxy": (
                        list(crop_assignment["crop_bbox_xyxy"])
                        if crop_assignment is not None
                        else gt_crop_bbox
                    ),
                    "gt_bbox_xyxy": gt_bbox,
                    "gt_ellipse": gt_ellipse,
                    "crop_rotation_degrees": (
                        float(crop_assignment["crop_rotation_degrees"])
                        if crop_assignment is not None
                        else float(gt_crop_rotation)
                    ),
                    "detector_crop": crop_assignment,
                    "keypoints": keypoints,
                    "labeled": labeled,
                    "visible": visible,
                }
            )
        if not samples:
            raise ValueError(f"No usable face landmark instances in {self.annotation_path}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _transform(
        self,
        bbox: Sequence[float],
        *,
        base_rotation_degrees: float,
        warp_bbox_aspect: bool,
    ) -> CropTransform:
        if not self.training:
            return build_crop_transform(
                bbox,
                output_size=self.input_size,
                scale_factor=self.augmentation.scale_factor,
                rotation_deg=base_rotation_degrees,
                warp_bbox_aspect=warp_bbox_aspect,
            )
        scale = self.augmentation.scale_factor * random.uniform(
            1.0 - self.augmentation.scale_jitter,
            1.0 + self.augmentation.scale_jitter,
        )
        offset = (
            random.uniform(-self.augmentation.center_jitter, self.augmentation.center_jitter),
            random.uniform(-self.augmentation.center_jitter, self.augmentation.center_jitter),
        )
        rotation = base_rotation_degrees + random.uniform(
            -self.augmentation.rotation_deg,
            self.augmentation.rotation_deg,
        )
        return build_crop_transform(
            bbox,
            output_size=self.input_size,
            scale_factor=scale,
            center_offset_xy=offset,
            rotation_deg=rotation,
            warp_bbox_aspect=warp_bbox_aspect,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        crop_bbox = list(sample["bbox_xyxy"])
        crop_ellipse = sample["gt_ellipse"]
        crop_error_residual = np.zeros((5,), dtype=np.float32)
        crop_error_area_bin = -1
        crop_error_clean = True
        crop_error_severity = 0.0
        if self.training and sample["detector_crop"] is None:
            if self.ellipse_error_sampler is not None:
                error_sample = self.ellipse_error_sampler.sample(
                    sample["gt_ellipse"],
                    image_width=sample["image_width"],
                    image_height=sample["image_height"],
                )
                crop_ellipse = error_sample.ellipse
                crop_error_residual = np.asarray(
                    error_sample.residual,
                    dtype=np.float32,
                )
                crop_error_area_bin = error_sample.area_bin
                crop_error_clean = error_sample.clean
                crop_error_severity = error_sample.severity
            else:
                # Legacy fallback for isolated tests. Production configuration
                # requires the calibrated ellipse-DEIM error model.
                crop_ellipse = perturb_ellipse(
                    sample["gt_ellipse"],
                    center_offset=(
                        random.uniform(
                            -self.augmentation.center_jitter,
                            self.augmentation.center_jitter,
                        ),
                        random.uniform(
                            -self.augmentation.center_jitter,
                            self.augmentation.center_jitter,
                        ),
                    ),
                    axis_scale=(
                        random.uniform(
                            1.0 - self.augmentation.scale_jitter,
                            1.0 + self.augmentation.scale_jitter,
                        ),
                        random.uniform(
                            1.0 - self.augmentation.scale_jitter,
                            1.0 + self.augmentation.scale_jitter,
                        ),
                    ),
                    angle_offset_radians=np.deg2rad(
                        random.uniform(
                            -self.augmentation.rotation_deg,
                            self.augmentation.rotation_deg,
                        )
                    ),
                )
            crop_bbox, crop_rotation = ellipse_aligned_rectangle(crop_ellipse)
            transform = build_crop_transform(
                crop_bbox,
                output_size=self.input_size,
                scale_factor=self.augmentation.scale_factor,
                rotation_deg=crop_rotation,
                warp_bbox_aspect=False,
            )
        else:
            transform = self._transform(
                crop_bbox,
                base_rotation_degrees=float(sample["crop_rotation_degrees"]),
                warp_bbox_aspect=False,
            )
        crop = warp_crop(image, transform)
        keypoints = original_to_normalized_crop(sample["keypoints"], transform)
        labeled = sample["labeled"].copy()
        visible = sample["visible"].copy()
        inside = (
            (keypoints[:, 0] >= 0.0)
            & (keypoints[:, 0] <= 1.0)
            & (keypoints[:, 1] >= 0.0)
            & (keypoints[:, 1] <= 1.0)
        )
        labeled &= inside
        visible &= labeled
        occlusion_mask = np.zeros(crop.shape[:2], dtype=np.bool_)
        flipped = False
        if self.training:
            crop = _photometric(crop, self.augmentation)
            crop, occlusion_mask = _random_occlusion(crop, self.augmentation)
            pixel_x = np.clip(
                np.rint(keypoints[:, 0] * (crop.shape[1] - 1)).astype(np.int64),
                0,
                crop.shape[1] - 1,
            )
            pixel_y = np.clip(
                np.rint(keypoints[:, 1] * (crop.shape[0] - 1)).astype(np.int64),
                0,
                crop.shape[0] - 1,
            )
            synthetically_occluded = occlusion_mask[pixel_y, pixel_x]
            visible &= ~synthetically_occluded
            if random.random() < self.augmentation.horizontal_flip_probability:
                crop = np.ascontiguousarray(crop[:, ::-1])
                occlusion_mask = np.ascontiguousarray(occlusion_mask[:, ::-1])
                keypoints[:, 0] = 1.0 - keypoints[:, 0]
                keypoints = keypoints[np.asarray(FLIP_INDEX)]
                labeled = labeled[np.asarray(FLIP_INDEX)]
                visible = visible[np.asarray(FLIP_INDEX)]
                flipped = True
        annotation = sample["annotation"]
        image_info = sample["image"]
        target = {
            "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
            "labeled": torch.from_numpy(labeled.astype(np.bool_)),
            "visible": torch.from_numpy(visible.astype(np.bool_)),
            "original_keypoints": torch.from_numpy(sample["keypoints"].copy()),
            "original_labeled": torch.from_numpy(sample["labeled"].copy()),
            "original_visible": torch.from_numpy(sample["visible"].copy()),
            "occlusion_mask": torch.from_numpy(occlusion_mask),
            "bbox_xyxy": torch.tensor(
                sample["gt_bbox_xyxy"],
                dtype=torch.float32,
            ),
            "detector_crop_bbox_xyxy": torch.tensor(
                crop_bbox,
                dtype=torch.float32,
            ),
            "gt_ellipse": torch.tensor(
                sample["gt_ellipse"],
                dtype=torch.float32,
            ),
            "crop_ellipse": torch.tensor(
                crop_ellipse,
                dtype=torch.float32,
            ),
            "crop_error_residual": torch.from_numpy(crop_error_residual),
            "crop_error_area_bin": int(crop_error_area_bin),
            "crop_error_clean": bool(crop_error_clean),
            "crop_error_severity": float(crop_error_severity),
            "detector_matching_iou": (
                float(sample["detector_crop"]["matching_iou"])
                if sample["detector_crop"] is not None
                else 1.0
            ),
            "crop_rotation_degrees": float(transform.rotation_deg),
            "affine": torch.from_numpy(transform.matrix.copy()),
            "inverse_affine": torch.from_numpy(transform.inverse_matrix.copy()),
            "annotation_id": int(annotation.get("id", index)),
            "image_id": int(annotation["image_id"]),
            "file_name": str(image_info["file_name"]),
            "source_video_id": str(image_info.get("metadata", {}).get("source_video_id", "")),
            "crop_side": float(transform.side),
            "flipped": flipped,
        }
        return image_to_tensor(crop, mean=self.mean, std=self.std), target


def collate_landmark_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
