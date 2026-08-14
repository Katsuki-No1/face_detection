from __future__ import annotations

import json
import math
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


FLIP_INDEX = np.asarray((1, 0, 2, 4, 3), dtype=np.int64)


@dataclass(frozen=True)
class CropAugmentation:
    roi_geometry: str = "long_axis_square"
    scale_factor: float = 1.25
    scale_jitter: float = 0.20
    center_jitter: float = 0.08
    rotation_deg: float = 25.0
    horizontal_flip_probability: float = 0.50
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.20
    blur_probability: float = 0.15
    blur_kernel_max: int = 7
    occlusion_probability: float = 0.35
    occlusion_area: tuple[float, float] = (0.03, 0.25)
    occlusion_count: tuple[int, int] = (1, 3)


@dataclass(frozen=True)
class AffineCrop:
    matrix: np.ndarray
    inverse: np.ndarray
    output_size: int
    source_width: float
    source_height: float
    rotation_degrees: float


def _resolve_image(root: Path, file_name: str) -> Path:
    path = Path(file_name)
    candidates = (path,) if path.is_absolute() else (root / path, root / "images" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {file_name!r} below {root}")


def _keypoints(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.size < 15:
        array = np.pad(array, (0, 15 - array.size))
    array = array[:15].reshape(5, 3)
    return array[:, :2].copy(), array[:, 2] > 0, array[:, 2] > 1


def _xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def _canonicalize_ellipse(values: Sequence[float]) -> np.ndarray:
    ellipse = np.asarray(values, dtype=np.float64).reshape(5).copy()
    if not np.isfinite(ellipse).all() or ellipse[2] <= 0.0 or ellipse[3] <= 0.0:
        raise ValueError(f"Invalid ellipse: {values}")
    if ellipse[3] > ellipse[2]:
        ellipse[2], ellipse[3] = ellipse[3], ellipse[2]
        ellipse[4] += math.pi * 0.5
    ellipse[4] = (ellipse[4] + math.pi * 0.5) % math.pi - math.pi * 0.5
    return ellipse.astype(np.float32)


def _scale_ellipse(values: Sequence[float], scale_x: float, scale_y: float) -> np.ndarray:
    cx, cy, major, minor, theta = _canonicalize_ellipse(values).astype(np.float64)
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    covariance = rotation @ np.diag((major * major, minor * minor)) @ rotation.T
    scaling = np.diag((float(scale_x), float(scale_y)))
    transformed = scaling @ covariance @ scaling
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    order = np.argsort(eigenvalues)[::-1]
    radii = np.sqrt(np.clip(eigenvalues[order], 1e-12, None))
    major_axis = eigenvectors[:, order[0]]
    scaled_theta = math.atan2(float(major_axis[1]), float(major_axis[0]))
    return _canonicalize_ellipse(
        (cx * scale_x, cy * scale_y, radii[0], radii[1], scaled_theta)
    )


def _flip_ellipse(values: Sequence[float], width: float) -> np.ndarray:
    cx, cy, major, minor, theta = _canonicalize_ellipse(values)
    return _canonicalize_ellipse(
        (float(width) - float(cx), cy, major, minor, math.pi - float(theta))
    )


def build_affine_crop(
    bbox_xyxy: Sequence[float],
    *,
    output_size: int,
    scale: float,
    center_offset: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
    warp_bbox_aspect: bool = False,
) -> AffineCrop:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid face bbox: {bbox_xyxy}")
    base = max(width, height)
    if warp_bbox_aspect:
        source_width = max(2.0, width * scale)
        source_height = max(2.0, height * scale)
        center_scale_x, center_scale_y = width, height
    else:
        source_width = source_height = max(2.0, base * scale)
        center_scale_x = center_scale_y = base
    center = np.asarray(
        (
            (x1 + x2) * 0.5 + center_offset[0] * center_scale_x,
            (y1 + y2) * 0.5 + center_offset[1] * center_scale_y,
        ),
        dtype=np.float32,
    )
    radians = np.deg2rad(rotation)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    up = np.asarray(
        (
            source_height * 0.5 * sine,
            -source_height * 0.5 * cosine,
        ),
        dtype=np.float32,
    )
    right = np.asarray(
        (
            source_width * 0.5 * cosine,
            source_width * 0.5 * sine,
        ),
        dtype=np.float32,
    )
    target_center = np.asarray(((output_size - 1) * 0.5, (output_size - 1) * 0.5), dtype=np.float32)
    half = (output_size - 1) * 0.5
    source = np.stack((center, center + up, center + right))
    target = np.stack(
        (
            target_center,
            target_center + np.asarray((0.0, -half), dtype=np.float32),
            target_center + np.asarray((half, 0.0), dtype=np.float32),
        )
    )
    matrix = cv2.getAffineTransform(source, target).astype(np.float32)
    return AffineCrop(
        matrix=matrix,
        inverse=cv2.invertAffineTransform(matrix).astype(np.float32),
        output_size=output_size,
        source_width=float(source_width),
        source_height=float(source_height),
        rotation_degrees=float(rotation),
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points.astype(np.float32), np.ones((len(points), 1), np.float32)), axis=1)
    return homogeneous @ matrix.T


def _photometric(image: np.ndarray, augmentation: CropAugmentation) -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    value += random.uniform(-augmentation.brightness, augmentation.brightness)
    mean = value.mean(axis=(0, 1), keepdims=True)
    value = mean + (value - mean) * random.uniform(1 - augmentation.contrast, 1 + augmentation.contrast)
    gray = value.mean(axis=2, keepdims=True)
    value = gray + (value - gray) * random.uniform(1 - augmentation.saturation, 1 + augmentation.saturation)
    result = np.uint8(np.round(np.clip(value, 0, 1) * 255))
    if random.random() < augmentation.blur_probability:
        kernels = list(range(3, max(3, augmentation.blur_kernel_max | 1) + 1, 2))
        result = cv2.GaussianBlur(result, (random.choice(kernels),) * 2, 0)
    return result


def _occlude(image: np.ndarray, augmentation: CropAugmentation) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    if random.random() >= augmentation.occlusion_probability:
        return image, mask
    result = image.copy()
    count = random.randint(*augmentation.occlusion_count)
    for _ in range(count):
        area = random.uniform(*augmentation.occlusion_area) * image.shape[0] * image.shape[1] / count
        aspect = random.uniform(0.4, 2.5)
        width = max(1, min(image.shape[1], int(round((area * aspect) ** 0.5))))
        height = max(1, min(image.shape[0], int(round((area / aspect) ** 0.5))))
        x1 = random.randint(0, image.shape[1] - width)
        y1 = random.randint(0, image.shape[0] - height)
        if random.random() < 0.5:
            fill = np.asarray([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
            result[y1 : y1 + height, x1 : x1 + width] = fill
        else:
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            result[y1 : y1 + height, x1 : x1 + width] = noise
        mask[y1 : y1 + height, x1 : x1 + width] = 1.0
    return result, mask


def _tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).contiguous() / 255.0


class FaceCropDataset(Dataset[dict[str, Any]]):
    """Top-down data using augmented GT ellipses in production training."""

    def __init__(
        self,
        *,
        annotation: str | Path,
        image_root: str | Path,
        input_size: int,
        training: bool,
        category_ids: Sequence[int] = (1,),
        augmentation: CropAugmentation | None = None,
        ellipse_error_model: str | Path | None = None,
        detector_crop_cache: str | Path | None = None,
        detector_matching_iou: float = 0.50,
        require_ellipse_crop: bool = True,
    ) -> None:
        self.annotation = Path(annotation)
        self.image_root = Path(image_root)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.augmentation = augmentation or CropAugmentation()
        if self.augmentation.roi_geometry != "long_axis_square":
            raise ValueError(
                "FaceCropDataset requires roi_geometry=long_axis_square; "
                "anisotropic ellipse-rectangle warping is disabled"
            )
        self.ellipse_error_sampler = (
            EllipseErrorSampler.from_path(ellipse_error_model)
            if ellipse_error_model is not None
            else None
        )
        data = json.loads(self.annotation.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in data.get("images", [])}
        allowed = {int(value) for value in category_ids}
        crop_assignments: dict[int, dict[str, Any]] | None = None
        self.detector_crop_report: dict[str, Any] | None = None
        if detector_crop_cache is not None:
            _metadata, predictions = load_detector_prediction_cache(
                detector_crop_cache
            )
            crop_assignments, self.detector_crop_report = (
                match_annotations_to_detector_crops(
                    data,
                    predictions,
                    category_ids=category_ids,
                    minimum_iou=float(detector_matching_iou),
                    require_ellipse=bool(require_ellipse_crop),
                )
            )
        self.samples: list[dict[str, Any]] = []
        for annotation_row in data.get("annotations", []):
            if int(annotation_row.get("category_id", -1)) not in allowed:
                continue
            bbox = annotation_row.get("bbox", ())
            if len(bbox) != 4 or float(bbox[2]) <= 1 or float(bbox[3]) <= 1:
                continue
            points, labeled, visible = _keypoints(annotation_row.get("keypoints", ()))
            if not labeled.any():
                continue
            image = images.get(int(annotation_row["image_id"]))
            if image is None:
                continue
            crop_assignment = (
                crop_assignments.get(int(annotation_row.get("id", -1)))
                if crop_assignments is not None
                else None
            )
            if crop_assignments is not None and crop_assignment is None:
                continue
            gt_bbox = _xywh_to_xyxy(bbox)
            raw_ellipse = annotation_row.get("ellipse")
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
                    f"{annotation_row.get('id')}"
                )
            gt_crop_bbox, gt_crop_rotation = (
                ellipse_aligned_rectangle(gt_ellipse)
                if gt_ellipse is not None
                else (gt_bbox.tolist(), 0.0)
            )
            self.samples.append(
                {
                    "annotation": annotation_row,
                    "image": image,
                    "image_width": float(image["width"]),
                    "image_height": float(image["height"]),
                    "path": _resolve_image(self.image_root, str(image["file_name"])),
                    "bbox": (
                        np.asarray(
                            crop_assignment["crop_bbox_xyxy"],
                            dtype=np.float32,
                        )
                        if crop_assignment is not None
                        else np.asarray(gt_crop_bbox, dtype=np.float32)
                    ),
                    "gt_bbox": gt_bbox,
                    "gt_ellipse": gt_ellipse,
                    "crop_rotation_degrees": (
                        float(crop_assignment["crop_rotation_degrees"])
                        if crop_assignment is not None
                        else float(gt_crop_rotation)
                    ),
                    "detector_crop": crop_assignment,
                    "points": points,
                    "labeled": labeled,
                    "visible": visible,
                }
            )
        if not self.samples:
            raise ValueError(f"No usable five-keypoint annotations: {self.annotation}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = np.asarray(Image.open(sample["path"]).convert("RGB"))
        crop_bbox = sample["bbox"].copy()
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
                # Kept only for small standalone tests and legacy configs.
                # Production configs are validated to require a calibrated
                # ellipse-DEIM residual model.
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
                    angle_offset_radians=math.radians(
                        random.uniform(
                            -self.augmentation.rotation_deg,
                            self.augmentation.rotation_deg,
                        )
                    ),
                )
            crop_box_list, rotation = ellipse_aligned_rectangle(crop_ellipse)
            crop_bbox = np.asarray(crop_box_list, dtype=np.float32)
            scale = self.augmentation.scale_factor
            offset = (0.0, 0.0)
        elif self.training:
            scale = self.augmentation.scale_factor * random.uniform(
                1 - self.augmentation.scale_jitter,
                1 + self.augmentation.scale_jitter,
            )
            offset = (
                random.uniform(-self.augmentation.center_jitter, self.augmentation.center_jitter),
                random.uniform(-self.augmentation.center_jitter, self.augmentation.center_jitter),
            )
            rotation = float(sample["crop_rotation_degrees"]) + random.uniform(
                -self.augmentation.rotation_deg,
                self.augmentation.rotation_deg,
            )
        else:
            scale = self.augmentation.scale_factor
            offset = (0.0, 0.0)
            rotation = float(sample["crop_rotation_degrees"])
        transform = build_affine_crop(
            crop_bbox,
            output_size=self.input_size,
            scale=scale,
            center_offset=offset,
            rotation=rotation,
            warp_bbox_aspect=False,
        )
        crop = cv2.warpAffine(
            image,
            transform.matrix,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        coordinates = _transform_points(sample["points"], transform.matrix) / (self.input_size - 1)
        inside = np.logical_and(coordinates >= 0, coordinates <= 1).all(axis=1)
        labeled = sample["labeled"].copy() & inside
        visible = sample["visible"].copy() & inside
        if self.training:
            crop = _photometric(crop, self.augmentation)
            if random.random() < self.augmentation.horizontal_flip_probability:
                crop = np.ascontiguousarray(crop[:, ::-1])
                coordinates[:, 0] = 1.0 - coordinates[:, 0]
                coordinates = coordinates[FLIP_INDEX]
                labeled = labeled[FLIP_INDEX]
                visible = visible[FLIP_INDEX]
        clean = crop.copy()
        crop, occlusion_mask = _occlude(crop, self.augmentation) if self.training else (
            crop,
            np.zeros((self.input_size, self.input_size), dtype=np.float32),
        )
        if self.training and bool(occlusion_mask.any()):
            pixel_x = np.rint(np.clip(coordinates[:, 0], 0.0, 1.0) * (self.input_size - 1)).astype(np.int64)
            pixel_y = np.rint(np.clip(coordinates[:, 1], 0.0, 1.0) * (self.input_size - 1)).astype(np.int64)
            synthetically_occluded = occlusion_mask[pixel_y, pixel_x] > 0.5
            # Coordinates remain supervised so the model learns recovery, while
            # visibility correctly becomes negative for a covered landmark.
            visible = visible & ~synthetically_occluded
        return {
            "image": _tensor(crop),
            "clean_image": _tensor(clean),
            "occlusion_mask": torch.from_numpy(occlusion_mask).unsqueeze(0),
            "coordinates": torch.from_numpy(coordinates.astype(np.float32)),
            "labeled": torch.from_numpy(labeled),
            "visible": torch.from_numpy(visible),
            "bbox_xyxy": torch.from_numpy(crop_bbox.copy()),
            "gt_bbox_xyxy": torch.from_numpy(sample["gt_bbox"].copy()),
            "gt_ellipse": (
                torch.tensor(sample["gt_ellipse"], dtype=torch.float32)
                if sample["gt_ellipse"] is not None
                else torch.full((5,), float("nan"), dtype=torch.float32)
            ),
            "crop_ellipse": (
                torch.tensor(crop_ellipse, dtype=torch.float32)
                if crop_ellipse is not None
                else torch.full((5,), float("nan"), dtype=torch.float32)
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
            "crop_rotation_degrees": float(rotation),
            "crop_roi_size": torch.tensor(
                [transform.source_width, transform.source_height],
                dtype=torch.float32,
            ),
            "crop_roi_is_square": bool(
                abs(transform.source_width - transform.source_height) < 1.0e-6
            ),
            "affine": torch.from_numpy(transform.matrix.copy()),
            "inverse_affine": torch.from_numpy(transform.inverse.copy()),
            "image_id": int(sample["annotation"]["image_id"]),
            "annotation_id": int(sample["annotation"].get("id", index)),
            "file_name": str(sample["image"]["file_name"]),
        }


def collate_crops(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("image", "clean_image", "occlusion_mask", "coordinates", "labeled", "visible")
    output = {key: torch.stack([row[key] for row in batch]) for key in tensor_keys}
    output["metadata"] = [
        {key: value for key, value in row.items() if key not in tensor_keys}
        for row in batch
    ]
    return output


class JointFaceDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Full-frame COCO dataset for DEIMv2 bbox + five-keypoint joint training."""

    def __init__(
        self,
        *,
        annotation: str | Path,
        image_root: str | Path,
        image_size: int = 960,
        category_ids: Sequence[int] = (1,),
        target_category_id: int | None = None,
        training: bool = False,
        head_annotation: str | Path | None = None,
    ) -> None:
        data = json.loads(Path(annotation).read_text(encoding="utf-8"))
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        allowed = {int(value) for value in category_ids}
        self.target_category_id = int(target_category_id) if target_category_id is not None else None
        self.training = bool(training)
        head_rows_by_file: dict[str, list[dict[str, Any]]] = {}
        if head_annotation is not None:
            head_data = json.loads(
                Path(head_annotation).read_text(encoding="utf-8")
            )
            head_images = {
                int(row["id"]): str(row["file_name"])
                for row in head_data.get("images", [])
            }
            for row in head_data.get("annotations", []):
                ellipse = row.get("ellipse")
                if (
                    not bool(row.get("attribute_valid", True))
                    or not isinstance(ellipse, (list, tuple))
                    or len(ellipse) != 5
                ):
                    continue
                file_name = head_images.get(int(row["image_id"]))
                if file_name is not None:
                    head_rows_by_file.setdefault(file_name, []).append(row)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in data.get("annotations", []):
            if int(row.get("category_id", -1)) in allowed:
                grouped.setdefault(int(row["image_id"]), []).append(row)
        self.images = []
        for image in data.get("images", []):
            file_name = str(image["file_name"])
            annotations = []
            for original in grouped.get(int(image["id"]), []):
                row = dict(original)
                face_ellipse = row.get("ellipse")
                candidates = head_rows_by_file.get(file_name, [])
                matched_head = None
                if (
                    candidates
                    and isinstance(face_ellipse, (list, tuple))
                    and len(face_ellipse) == 5
                ):
                    matched_head = min(
                        candidates,
                        key=lambda candidate: math.hypot(
                            float(candidate["ellipse"][0])
                            - float(face_ellipse[0]),
                            float(candidate["ellipse"][1])
                            - float(face_ellipse[1]),
                        ),
                    )
                    center_distance = math.hypot(
                        float(matched_head["ellipse"][0])
                        - float(face_ellipse[0]),
                        float(matched_head["ellipse"][1])
                        - float(face_ellipse[1]),
                    )
                    if center_distance > 1.0e-3:
                        matched_head = None
                row["_head_bbox_xyxy"] = (
                    _xywh_to_xyxy(matched_head["bbox"])
                    if matched_head is not None
                    else np.full((4,), np.nan, dtype=np.float32)
                )
                annotations.append(row)
            self.images.append(
                (
                    image,
                    _resolve_image(self.image_root, file_name),
                    annotations,
                )
            )
        if not self.images:
            raise ValueError(f"No images in {annotation}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info, path, annotations = self.images[index]
        image = Image.open(path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        boxes, head_boxes, ellipses, labels, keypoints, labeled, visible = [], [], [], [], [], [], []
        for row in annotations:
            box = _xywh_to_xyxy(row["bbox"])
            head_box = np.asarray(row["_head_bbox_xyxy"], dtype=np.float32).copy()
            raw_ellipse = row.get("ellipse")
            if isinstance(raw_ellipse, (list, tuple)) and len(raw_ellipse) == 5:
                ellipse = _scale_ellipse(raw_ellipse, scale_x, scale_y)
            else:
                center_x = (float(box[0]) + float(box[2])) * 0.5
                center_y = (float(box[1]) + float(box[3])) * 0.5
                ellipse = _scale_ellipse(
                    (
                        center_x,
                        center_y,
                        (float(box[2]) - float(box[0])) * 0.5,
                        (float(box[3]) - float(box[1])) * 0.5,
                        0.0,
                    ),
                    scale_x,
                    scale_y,
                )
            box[0::2] *= scale_x
            box[1::2] *= scale_y
            head_box[0::2] *= scale_x
            head_box[1::2] *= scale_y
            points, point_labeled, point_visible = _keypoints(row.get("keypoints", ()))
            points[:, 0] *= scale_x / self.image_size
            points[:, 1] *= scale_y / self.image_size
            boxes.append(box)
            head_boxes.append(head_box)
            ellipses.append(ellipse)
            labels.append(self.target_category_id if self.target_category_id is not None else int(row["category_id"]))
            keypoints.append(points)
            labeled.append(point_labeled)
            visible.append(point_visible)
        image_array = np.asarray(image)
        if self.training:
            image_array = _photometric(image_array, CropAugmentation(occlusion_probability=0.0))
            if random.random() < 0.5:
                image_array = np.ascontiguousarray(image_array[:, ::-1])
                for box in boxes:
                    old_x1, old_x2 = float(box[0]), float(box[2])
                    box[0], box[2] = self.image_size - old_x2, self.image_size - old_x1
                for head_box in head_boxes:
                    if np.isfinite(head_box).all():
                        old_x1, old_x2 = float(head_box[0]), float(head_box[2])
                        head_box[0], head_box[2] = (
                            self.image_size - old_x2,
                            self.image_size - old_x1,
                        )
                ellipses = [
                    _flip_ellipse(ellipse, float(self.image_size))
                    for ellipse in ellipses
                ]
                keypoints = [
                    np.stack((1.0 - points[:, 0], points[:, 1]), axis=-1)[FLIP_INDEX]
                    for points in keypoints
                ]
                labeled = [values[FLIP_INDEX] for values in labeled]
                visible = [values[FLIP_INDEX] for values in visible]
        target = {
            "boxes": torch.tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4),
            "head_boxes": torch.tensor(
                np.asarray(head_boxes), dtype=torch.float32
            ).reshape(-1, 4),
            "ellipses": torch.tensor(np.asarray(ellipses), dtype=torch.float32).reshape(-1, 5),
            "labels": torch.tensor(labels, dtype=torch.long),
            "keypoints": torch.tensor(np.asarray(keypoints), dtype=torch.float32).reshape(-1, 5, 2),
            "labeled": torch.tensor(np.asarray(labeled), dtype=torch.bool).reshape(-1, 5),
            "visible": torch.tensor(np.asarray(visible), dtype=torch.bool).reshape(-1, 5),
            "image_id": torch.tensor(int(image_info["id"]), dtype=torch.long),
        }
        return _tensor(image_array), target


def collate_joint(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images)), list(targets)
