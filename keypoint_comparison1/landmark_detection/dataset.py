from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from src.data.landmark_occlusion import (
    LandmarkOcclusionConfig,
    apply_landmark_occlusion_numpy,
    transform_ellipse_affine,
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
    landmark_guided_occlusion: dict[str, Any] | None = None
    paired_occlusion: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairedOcclusionConfig:
    probability: float = 0.30
    kinds: tuple[str, ...] = ("rectangle", "irregular", "boundary")
    kind_probabilities: tuple[float, ...] = (1.0 / 3.0,) * 3
    face_overlap: tuple[float, float] = (0.10, 0.40)
    landmark_anchor_probability: float = 0.70
    center_offset_fraction: float = 0.12
    maximum_attempts: int = 128
    one_face_per_source_image: bool = True
    appearance: str = "solid"
    material_manifest: str | None = None
    material_manifest_sha256: str | None = None
    object_overlap_policy: dict[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "PairedOcclusionConfig":
        raw = dict(value or {})
        kinds = tuple(str(item) for item in raw.get("kinds", cls.kinds))
        probabilities = tuple(
            float(item)
            for item in raw.get(
                "kind_probabilities",
                [1.0 / max(1, len(kinds))] * len(kinds),
            )
        )
        overlap = tuple(float(item) for item in raw.get("face_overlap", (0.10, 0.40)))
        config = cls(
            probability=float(raw.get("probability", 0.30)),
            kinds=kinds,
            kind_probabilities=probabilities,
            face_overlap=(overlap[0], overlap[1]),
            landmark_anchor_probability=float(
                raw.get("landmark_anchor_probability", 0.70)
            ),
            center_offset_fraction=float(raw.get("center_offset_fraction", 0.12)),
            maximum_attempts=int(raw.get("maximum_attempts", 128)),
            one_face_per_source_image=bool(
                raw.get("one_face_per_source_image", True)
            ),
            appearance=str(raw.get("appearance", "solid")),
            material_manifest=raw.get("material_manifest"),
            material_manifest_sha256=raw.get("material_manifest_sha256"),
            object_overlap_policy=raw.get("object_overlap_policy"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.object_overlap_policy is not None:
            from .occlusion_policy import ObjectOverlapPolicy
            if self.appearance != "object" or tuple(self.face_overlap) != (.10, .40):
                raise ValueError("Conditional object policy requires object appearance and standard10-40% bounds")
            ObjectOverlapPolicy.from_mapping(self.object_overlap_policy)
        if self.appearance not in {"solid", "texture", "object", "real_mixed"}:
            raise ValueError("paired_occlusion.appearance must be solid/texture/object/real_mixed")
        if self.appearance != "solid" and (not self.material_manifest or not self.material_manifest_sha256):
            raise ValueError("Pasted occlusion requires a locked material manifest")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("paired_occlusion.probability must be in [0, 1]")
        if not self.kinds or set(self.kinds) != {
            "rectangle",
            "irregular",
            "boundary",
        }:
            raise ValueError(
                "paired_occlusion.kinds must contain rectangle, irregular, and boundary"
            )
        if len(self.kind_probabilities) != len(self.kinds):
            raise ValueError(
                "paired_occlusion.kind_probabilities must match kinds"
            )
        if any(value <= 0.0 for value in self.kind_probabilities):
            raise ValueError(
                "paired_occlusion.kind_probabilities must all be positive"
            )
        if abs(sum(self.kind_probabilities) - 1.0) > 1.0e-6:
            raise ValueError(
                "paired_occlusion.kind_probabilities must sum to one"
            )
        if not 0.0 < self.face_overlap[0] <= self.face_overlap[1] < 1.0:
            raise ValueError(
                "paired_occlusion.face_overlap must satisfy 0 < min <= max < 1"
            )
        if not self.one_face_per_source_image:
            raise ValueError(
                "paired_occlusion.one_face_per_source_image must remain true"
            )

    def landmark_mapping(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "regular_probability": 0.50,
            "landmark_anchor_probability": self.landmark_anchor_probability,
            "face_overlap": list(self.face_overlap),
            "center_offset_fraction": self.center_offset_fraction,
            "maximum_attempts": self.maximum_attempts,
        }


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
        seed: int = 42,
        evaluation_occlusion_kind: str | None = None,
        structure_head_annotations: str | Path | None = None,
        face_overlap_query_dim: int | None = None,
        face_overlap_detector_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.augmentation = augmentation or CropAugmentation()
        self.face_overlap_query_dim = face_overlap_query_dim
        if face_overlap_query_dim is not None:
            if detector_crop_cache is None:
                raise ValueError("FOR requires a query-bearing predicted-face crop cache")
            # Cached full-image queries must not reveal pixels hidden later in a crop.
            if (self.training and (self.augmentation.occlusion_probability > 0
                    or (self.augmentation.landmark_guided_occlusion or {}).get("probability", 0) > 0
                    or (self.augmentation.paired_occlusion or {}).get("probability", 0) > 0)) or evaluation_occlusion_kind:
                raise ValueError("FOR cached queries cannot be used with crop-only synthetic occlusion; recompute on the augmented full image")
        self.real_occlusion_bank = None
        if self.training and self.augmentation.paired_occlusion:
            paired = PairedOcclusionConfig.from_mapping(self.augmentation.paired_occlusion)
            if paired.appearance == "real_mixed":
                from .real_occlusion import RealOcclusionBank
                self.real_occlusion_bank = RealOcclusionBank(paired.material_manifest,
                    paired.material_manifest_sha256,annotation_path=self.annotation_path,image_root=self.image_root)
            elif paired.appearance != "solid":
                from .occlusion_materials import load_materials
                load_materials(paired.material_manifest, paired.material_manifest_sha256)
        self.seed = int(seed)
        self.evaluation_occlusion_kind = evaluation_occlusion_kind
        if evaluation_occlusion_kind not in {None, "boundary"}:
            raise ValueError(
                "evaluation_occlusion_kind currently supports only 'boundary'"
            )
        if self.training and evaluation_occlusion_kind is not None:
            raise ValueError(
                "evaluation_occlusion_kind is only valid when training=false"
            )
        self._epoch_state = torch.tensor([1], dtype=torch.int64).share_memory_()
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
        self.structure_heads = None
        if structure_head_annotations is not None:
            if not training:
                raise ValueError("Structure GT Head lookup is training-only")
            head_coco = json.loads(Path(structure_head_annotations).read_text(encoding="utf-8"))
            head_images = {int(item["id"]): item["file_name"] for item in head_coco["images"]}
            self.structure_heads = {}
            for item in head_coco["annotations"]:
                if item.get("ellipse") and item.get("attribute_valid", False):
                    key = (head_images[int(item["image_id"])], tuple(round(float(x), 4) for x in canonicalize_ellipse(item["ellipse"])))
                    if key in self.structure_heads:
                        raise ValueError(f"Ambiguous canonical Head/ellipse identity: {key}")
                    self.structure_heads[key] = item
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
            if face_overlap_query_dim is not None:
                from .face_overlap import validate_context_cache
                validate_context_cache(_metadata, predictions, annotation_path=self.annotation_path,
                                       image_ids=set(images), query_dim=face_overlap_query_dim,
                                       detector_identity=face_overlap_detector_identity)
            if _metadata.get("matching_contract") == "predicted_face_center_in_gt_head_square_1.20_score_then_distance_one_to_one":
                import hashlib
                from src.data.head_anchored_keypoint_cache import assigned_head_crops
                if self.training:
                    raise ValueError("Head-anchored predicted crop cache is validation-only")
                if hashlib.sha256(self.annotation_path.read_bytes()).hexdigest() != _metadata.get("annotation_sha256"):
                    raise ValueError("Head-anchored annotation hash mismatch")
                crop_assignments, self.detector_crop_report = assigned_head_crops(coco, predictions)
            else:
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
                    "face_occluded": bool(
                        (annotation.get("attributes") or {}).get(
                            "face_occluded",
                            False,
                        )
                    ),
                }
            )
        if not samples:
            raise ValueError(f"No usable face landmark instances in {self.annotation_path}")
        self.samples = samples
        self.structure_supervision_report = None
        if self.structure_heads is not None:
            matched = sum(
                (str(sample["image"]["file_name"]), tuple(round(float(x), 4) for x in sample["gt_ellipse"])) in self.structure_heads
                for sample in samples
            )
            self.structure_supervision_report = {
                "samples": len(samples), "canonical_supervised_head_matches": matched,
                "excluded": len(samples) - matched,
                "excluded_reason": "no_unique_attribute_valid_canonical_head_with_same_image_and_ellipse",
                "matching": "exact_file_name_and_canonical_ellipse_rounded_to_4_decimals",
            }
            if matched == 0:
                raise ValueError("No canonical GT Heads matched structure training data")
        self._sample_indices_by_image: dict[int, tuple[int, ...]] = {}
        grouped: dict[int, list[int]] = {}
        for sample_index, sample in enumerate(self.samples):
            grouped.setdefault(int(sample["annotation"]["image_id"]), []).append(sample_index)
        self._sample_indices_by_image = {
            image_id: tuple(indices) for image_id, indices in grouped.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state[0].item())

    def set_epoch(self, epoch: int) -> None:
        self._epoch_state[0] = int(epoch)

    def _landmark_occlusion_rng(self, index: int) -> random.Random | None:
        raw = self.augmentation.landmark_guided_occlusion
        if not isinstance(raw, dict):
            return None
        config = LandmarkOcclusionConfig.from_mapping(raw)
        image_id = int(self.samples[index]["annotation"]["image_id"])
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{image_id}:landmark_occlusion".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
        if config.probability <= 0.0 or rng.random() >= config.probability:
            return None
        selected = rng.choice(self._sample_indices_by_image[image_id])
        return rng if selected == int(index) else None

    def _paired_occlusion_rng(self, index: int) -> random.Random | None:
        raw = self.augmentation.paired_occlusion
        if not isinstance(raw, dict):
            return None
        config = PairedOcclusionConfig.from_mapping(raw)
        image_id = int(self.samples[index]["annotation"]["image_id"])
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{image_id}:paired_occlusion".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
        if config.probability <= 0.0 or rng.random() >= config.probability:
            return None
        if config.one_face_per_source_image:
            candidates = self._sample_indices_by_image[image_id]
            if self.real_occlusion_bank is not None:
                candidates = [i for i in candidates if self.real_occlusion_bank.eligible_sample(self.samples[i])]
                if not candidates:
                    return None
            selected = rng.choice(candidates)
            if selected != int(index):
                return None
        return rng

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
        clean_crop: np.ndarray | None = None
        paired_occlusion_metadata: dict[str, Any] = {
            "applied": False,
            "reason": "disabled",
        }
        synthetic_truncated = np.zeros((len(keypoints),), dtype=np.bool_)
        keypoint_visibility = np.where(labeled, np.where(visible, 2, 1), 0).astype(
            np.int64
        )
        natural_keypoint_visibility = keypoint_visibility.copy()
        flipped = False
        if self.training:
            crop = _photometric(crop, self.augmentation)
            if isinstance(self.augmentation.paired_occlusion, dict):
                clean_crop = crop.copy()
                paired_config = PairedOcclusionConfig.from_mapping(
                    self.augmentation.paired_occlusion
                )
                paired_rng = self._paired_occlusion_rng(index)
                if paired_rng is not None:
                    pixel_keypoints = keypoints.copy()
                    pixel_keypoints[:, 0] *= crop.shape[1] - 1
                    pixel_keypoints[:, 1] *= crop.shape[0] - 1
                    crop_ellipse_pixels = transform_ellipse_affine(
                        sample["gt_ellipse"], transform.matrix
                    )
                    selected_kind = paired_rng.choices(
                        paired_config.kinds,
                        weights=paired_config.kind_probabilities,
                        k=1,
                    )[0]
                    if paired_config.appearance == "real_mixed":
                        from .real_occlusion import apply_real_mixed
                        crop, occlusion_mask, keypoint_visibility, paired_occlusion_metadata = apply_real_mixed(
                            clean_crop, sample=sample, face_ellipse=crop_ellipse_pixels,
                            keypoints=pixel_keypoints, keypoint_visibility=natural_keypoint_visibility,
                            bank=self.real_occlusion_bank, rng=paired_rng)
                    elif paired_config.object_overlap_policy is None:
                        (
                            crop,
                            occlusion_mask,
                            keypoint_visibility,
                            paired_occlusion_metadata,
                        ) = apply_landmark_occlusion_numpy(
                            crop,
                            face_ellipse=crop_ellipse_pixels,
                            keypoints=pixel_keypoints,
                            keypoint_visibility=keypoint_visibility,
                            config=paired_config.landmark_mapping(),
                            rng=paired_rng,
                            force=True,
                            force_kind=selected_kind,
                        )
                    if paired_config.appearance not in {"solid", "real_mixed"}:
                        from .occlusion_materials import apply_material_occlusion
                        # Restore the pre-mask RNG so texture and solid use identical geometry.
                        material_rng = self._paired_occlusion_rng(index)
                        material_rng.choices(paired_config.kinds, weights=paired_config.kind_probabilities, k=1)
                        crop, occlusion_mask, keypoint_visibility, paired_occlusion_metadata = apply_material_occlusion(
                            clean_crop, face_ellipse=crop_ellipse_pixels, keypoints=pixel_keypoints,
                            keypoint_visibility=natural_keypoint_visibility, config=paired_config,
                            rng=material_rng, force_kind=selected_kind)
                    paired_occlusion_metadata = {
                        **paired_occlusion_metadata,
                        "pair_kind": ("object" if paired_config.object_overlap_policy is not None or paired_config.appearance == "real_mixed" else selected_kind),
                    }
                    if bool(paired_occlusion_metadata.get("applied", False)) and paired_occlusion_metadata["pair_kind"] == "boundary":
                        for point_index, point in enumerate(pixel_keypoints):
                            x = int(round(float(point[0])))
                            y = int(round(float(point[1])))
                            synthetic_truncated[point_index] = (
                                keypoint_visibility[point_index] > 0
                                and 0 <= x < crop.shape[1]
                                and 0 <= y < crop.shape[0]
                                and bool(occlusion_mask[y, x])
                            )
                else:
                    paired_occlusion_metadata = {
                        "applied": False,
                        "reason": "probability_or_one_face_selection",
                    }
            else:
                landmark_occlusion_rng = self._landmark_occlusion_rng(index)
                if isinstance(self.augmentation.landmark_guided_occlusion, dict):
                    if landmark_occlusion_rng is not None:
                        pixel_keypoints = keypoints.copy()
                        pixel_keypoints[:, 0] *= crop.shape[1] - 1
                        pixel_keypoints[:, 1] *= crop.shape[0] - 1
                        crop_ellipse_pixels = transform_ellipse_affine(
                            sample["gt_ellipse"], transform.matrix
                        )
                        (
                            crop,
                            occlusion_mask,
                            keypoint_visibility,
                            landmark_occlusion_metadata,
                        ) = apply_landmark_occlusion_numpy(
                            crop,
                            face_ellipse=crop_ellipse_pixels,
                            keypoints=pixel_keypoints,
                            keypoint_visibility=keypoint_visibility,
                            config=self.augmentation.landmark_guided_occlusion,
                            rng=landmark_occlusion_rng,
                            force=True,
                        )
                    else:
                        landmark_occlusion_metadata = {
                            "applied": False,
                            "reason": "one_face_per_source_image_selection",
                        }
                else:
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
            keypoint_visibility[
                synthetically_occluded & (keypoint_visibility > 0)
            ] = 1
            labeled = keypoint_visibility > 0
            visible = keypoint_visibility > 1
            if random.random() < self.augmentation.horizontal_flip_probability:
                crop = np.ascontiguousarray(crop[:, ::-1])
                if clean_crop is not None:
                    clean_crop = np.ascontiguousarray(clean_crop[:, ::-1])
                occlusion_mask = np.ascontiguousarray(occlusion_mask[:, ::-1])
                keypoints[:, 0] = 1.0 - keypoints[:, 0]
                keypoints = keypoints[np.asarray(FLIP_INDEX)]
                labeled = labeled[np.asarray(FLIP_INDEX)]
                visible = visible[np.asarray(FLIP_INDEX)]
                keypoint_visibility = keypoint_visibility[np.asarray(FLIP_INDEX)]
                natural_keypoint_visibility = natural_keypoint_visibility[np.asarray(FLIP_INDEX)]
                synthetic_truncated = synthetic_truncated[np.asarray(FLIP_INDEX)]
                flipped = True
        elif self.evaluation_occlusion_kind == "boundary":
            if not isinstance(self.augmentation.paired_occlusion, dict):
                raise ValueError(
                    "boundary evaluation requires augmentation.paired_occlusion"
                )
            clean_crop = crop.copy()
            paired_config = PairedOcclusionConfig.from_mapping(
                self.augmentation.paired_occlusion
            )
            digest = hashlib.sha256(
                f"{self.seed}:{sample['annotation'].get('id', index)}:boundary_eval".encode(
                    "utf-8"
                )
            ).digest()
            evaluation_rng = random.Random(
                int.from_bytes(digest[:8], byteorder="big", signed=False)
            )
            pixel_keypoints = keypoints.copy()
            pixel_keypoints[:, 0] *= crop.shape[1] - 1
            pixel_keypoints[:, 1] *= crop.shape[0] - 1
            crop_ellipse_pixels = transform_ellipse_affine(
                sample["gt_ellipse"], transform.matrix
            )
            (
                crop,
                occlusion_mask,
                keypoint_visibility,
                paired_occlusion_metadata,
            ) = apply_landmark_occlusion_numpy(
                crop,
                face_ellipse=crop_ellipse_pixels,
                keypoints=pixel_keypoints,
                keypoint_visibility=keypoint_visibility,
                config=paired_config.landmark_mapping(),
                rng=evaluation_rng,
                force=True,
                force_kind="boundary",
            )
            paired_occlusion_metadata = {
                **paired_occlusion_metadata,
                "pair_kind": "boundary",
                "deterministic_evaluation": True,
            }
            for point_index, point in enumerate(pixel_keypoints):
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                synthetic_truncated[point_index] = (
                    keypoint_visibility[point_index] > 0
                    and 0 <= x < crop.shape[1]
                    and 0 <= y < crop.shape[0]
                    and bool(occlusion_mask[y, x])
                )
            keypoint_visibility[
                synthetic_truncated & (keypoint_visibility > 0)
            ] = 1
            labeled = keypoint_visibility > 0
            visible = keypoint_visibility > 1
        annotation = sample["annotation"]
        image_info = sample["image"]
        target = {
            "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
            "labeled": torch.from_numpy(labeled.astype(np.bool_)),
            "visible": torch.from_numpy(visible.astype(np.bool_)),
            "keypoint_visibility": torch.from_numpy(
                keypoint_visibility.astype(np.int64)
            ),
            "original_keypoints": torch.from_numpy(sample["keypoints"].copy()),
            "original_labeled": torch.from_numpy(sample["labeled"].copy()),
            "original_visible": torch.from_numpy(sample["visible"].copy()),
            "occlusion_mask": torch.from_numpy(occlusion_mask),
            "synthetic_truncated": torch.from_numpy(synthetic_truncated),
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
            "face_occluded": torch.tensor(
                bool(sample["face_occluded"] or occlusion_mask.any()),
                dtype=torch.bool,
            ),
        }
        if self.face_overlap_query_dim is not None:
            from .face_overlap import crop_face_context
            target.update(crop_face_context(sample["detector_crop"], transform, flipped=flipped))
        if self.structure_heads is not None:
            from .continuous_recovery import relative_structure_targets

            identity = (str(image_info["file_name"]), tuple(round(float(x), 4) for x in sample["gt_ellipse"]))
            head = self.structure_heads.get(identity)
            # Unsupervised/ambiguous Heads are excluded, never replaced by Face size.
            if head is None:
                target["structure_pairs"] = torch.zeros(10, 2)
                target["structure_pairs_valid"] = torch.zeros(10, dtype=torch.bool)
            else:
                states = np.asarray(head["keypoint_states"]) > 0
                if flipped:
                    states = states[np.asarray(FLIP_INDEX)]
                if self.real_occlusion_bank is not None:
                    # Crop-outside points are missing for ALL supervision paths,
                    # including the optional relative-structure recovery loss.
                    states &= target["labeled"].numpy()
                head_side = 1.20 * max(float(head["bbox"][2]), float(head["bbox"][3]))
                target["structure_pairs"], target["structure_pairs_valid"] = relative_structure_targets(
                    target["keypoints"], torch.from_numpy(states), target["inverse_affine"], self.input_size, head_side
                )
        if self.training and isinstance(
            self.augmentation.landmark_guided_occlusion, dict
        ):
            target["landmark_occlusion"] = landmark_occlusion_metadata
        if (
            self.training and isinstance(self.augmentation.paired_occlusion, dict)
        ) or self.evaluation_occlusion_kind is not None:
            if clean_crop is None:
                raise RuntimeError("paired occlusion did not preserve a clean crop")
            target["clean_image"] = image_to_tensor(
                clean_crop,
                mean=self.mean,
                std=self.std,
            )
            target["paired_occlusion"] = paired_occlusion_metadata
            target["natural_keypoint_visibility"] = torch.from_numpy(natural_keypoint_visibility.copy())
            target["paired_occlusion"]["post_material_shared_horizontal_flip"] = flipped
            target["paired_occlusion"]["dataset_seed"] = self.seed
            target["paired_occlusion"]["augmentation_epoch"] = self.epoch
        return image_to_tensor(crop, mean=self.mean, std=self.std), target


def collate_landmark_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
