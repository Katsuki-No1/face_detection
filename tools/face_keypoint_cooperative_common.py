from __future__ import annotations

import copy
import json
import math
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[1]

from keypoint_comparison2.data import JointFaceDataset, collate_joint  # noqa: E402
from src.config_utils import load_yaml_config  # noqa: E402
from src.data.ellipse_augmentation import select_ellipse_border_crop  # noqa: E402
from src.head_pose_geometry.dataset import (  # noqa: E402
    HeadCentricFaceDataset,
    InterleavedRealSyntheticDataset,
)
from src.head_pose_geometry.head_crop_dataset import (  # noqa: E402
    HeadCropCooperativeDataset,
)
from src.models.face_keypoint_cooperative import (  # noqa: E402
    FaceKeypointCooperativeDETR,
    build_face_keypoint_cooperative_detr,
    load_cooperative_checkpoint_strict,
    validate_strict_sequential_config,
)
from src.models.face_keypoint_cooperative.checkpoint import (  # noqa: E402
    initialize_shared_dino_from_head_codino,
    initialize_shared_head_from_head_codino,
)
from src.models.head_codino.runtime import (  # noqa: E402
    representative_subset_indices,
)


class HeadCentricEllipseBorderCropAugmentation(Dataset[Any]):
    """Target-aware amodal border crop for the predicted-HEAD training path."""

    FACE_ALIGNED_FIELDS = (
        "ellipse_mask",
        "labels",
        "rotation_matrix",
        "pose_supervision_mask",
        "pose_weight",
        "instance_ids",
        "sample_is_synthetic",
        "template_vertices",
        "weak_perspective_scale",
        "translation",
        "shape_code",
    )

    def __init__(
        self,
        base: Dataset[Any],
        config: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.config = dict(config)
        self.seed = int(seed)
        self._epoch_state = torch.tensor([0], dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self._epoch_state[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state[0])

    @staticmethod
    def _clone_target(target: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (
                value.clone()
                if isinstance(value, torch.Tensor)
                else list(value)
                if isinstance(value, list)
                else copy.deepcopy(value)
            )
            for key, value in target.items()
        }

    @staticmethod
    def _transform_ellipses(
        ellipses: torch.Tensor,
        *,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float,
    ) -> torch.Tensor:
        centers = ellipses[:, :2].clone()
        centers[:, 0] = centers[:, 0] * scale_x + offset_x
        centers[:, 1] = centers[:, 1] * scale_y + offset_y
        theta = ellipses[:, 4]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        covariance = (
            rotation
            @ torch.diag_embed(ellipses[:, 2:4].square())
            @ rotation.transpose(-1, -2)
        )
        scaling = ellipses.new_tensor(((scale_x, 0.0), (0.0, scale_y)))
        covariance = scaling @ covariance @ scaling.T
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        axes = eigenvalues.clamp_min(1.0e-8).sqrt().flip(-1)
        major_vector = eigenvectors[..., :, 1]
        transformed_theta = torch.atan2(major_vector[..., 1], major_vector[..., 0])
        return torch.cat((centers, axes, transformed_theta.unsqueeze(-1)), dim=-1)

    @staticmethod
    def _ellipse_boxes(ellipses: torch.Tensor) -> torch.Tensor:
        centers = ellipses[:, :2]
        axes = ellipses[:, 2:4]
        theta = ellipses[:, 4]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        half_width = torch.sqrt(
            (axes[:, 0] * cosine).square() + (axes[:, 1] * sine).square()
        )
        half_height = torch.sqrt(
            (axes[:, 0] * sine).square() + (axes[:, 1] * cosine).square()
        )
        return torch.stack(
            (
                centers[:, 0] - half_width,
                centers[:, 1] - half_height,
                centers[:, 0] + half_width,
                centers[:, 1] + half_height,
            ),
            dim=-1,
        )

    def _transform_target(
        self,
        target: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        output = self._clone_target(target)
        face_count = int(output["ellipses"].shape[0])
        x0, y0, x1, y1 = (int(value) for value in selected["crop_xyxy"])
        scale_x = width / float(x1 - x0)
        scale_y = height / float(y1 - y0)
        offset_x = -x0 * scale_x
        offset_y = -y0 * scale_y

        ellipses = self._transform_ellipses(
            output["ellipses"],
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        raw_boxes = self._ellipse_boxes(ellipses)
        boxes = raw_boxes.clone()
        boxes[:, 0::2].clamp_(0.0, float(width))
        boxes[:, 1::2].clamp_(0.0, float(height))
        minimum_axis = float(self.config.get("minimum_axis", 2.0))
        valid = (
            (ellipses[:, 2] >= minimum_axis)
            & (ellipses[:, 3] >= minimum_axis)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        old_to_new = torch.full(
            (face_count,),
            -1,
            dtype=torch.long,
            device=ellipses.device,
        )
        old_to_new[valid] = torch.arange(
            int(valid.sum()),
            dtype=torch.long,
            device=ellipses.device,
        )

        keypoints = output["keypoints"].clone()
        pixel_points = keypoints * keypoints.new_tensor((width, height))
        pixel_points[..., 0] = pixel_points[..., 0] * scale_x + offset_x
        pixel_points[..., 1] = pixel_points[..., 1] * scale_y + offset_y
        keypoints = pixel_points / keypoints.new_tensor((width, height))
        inside = ((keypoints >= 0.0) & (keypoints <= 1.0)).all(dim=-1)
        labeled = output["labeled"] & inside
        visible = output["visible"] & inside

        output["boxes"] = boxes[valid]
        output["ellipses"] = ellipses[valid]
        output["keypoints"] = keypoints[valid].clamp(0.0, 1.0)
        output["labeled"] = labeled[valid]
        output["visible"] = visible[valid]
        for field in self.FACE_ALIGNED_FIELDS:
            value = output.get(field)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == face_count:
                output[field] = value[valid]

        head_boxes = output.get("head_boxes")
        if isinstance(head_boxes, torch.Tensor):
            head_boxes = head_boxes.clone()
            head_boxes[:, 0::2] = head_boxes[:, 0::2] * scale_x + offset_x
            head_boxes[:, 1::2] = head_boxes[:, 1::2] * scale_y + offset_y
            head_boxes[:, 0::2].clamp_(0.0, float(width))
            head_boxes[:, 1::2].clamp_(0.0, float(height))
            head_valid = (
                (head_boxes[:, 2] > head_boxes[:, 0])
                & (head_boxes[:, 3] > head_boxes[:, 1])
            )
            mapping = output["geometry_target_indices"].clone()
            mapped = torch.full_like(mapping, -1)
            mapped_targets = mapping >= 0
            mapped[mapped_targets] = old_to_new[mapping[mapped_targets]]
            output["head_boxes"] = head_boxes[head_valid]
            output["head_confidence"] = output["head_confidence"][head_valid]
            output["geometry_target_indices"] = mapped[head_valid]
            if isinstance(output.get("head_sources"), list):
                kept_heads = torch.nonzero(head_valid, as_tuple=False).flatten().tolist()
                output["head_sources"] = [output["head_sources"][index] for index in kept_heads]

        detection_boxes = output.get("head_detection_boxes")
        if isinstance(detection_boxes, torch.Tensor):
            detection_boxes = detection_boxes.clone()
            detection_boxes[:, 0::2] = (
                detection_boxes[:, 0::2] * scale_x + offset_x
            )
            detection_boxes[:, 1::2] = (
                detection_boxes[:, 1::2] * scale_y + offset_y
            )
            detection_boxes[:, 0::2].clamp_(0.0, float(width))
            detection_boxes[:, 1::2].clamp_(0.0, float(height))
            detection_valid = (
                (detection_boxes[:, 2] > detection_boxes[:, 0])
                & (detection_boxes[:, 3] > detection_boxes[:, 1])
            )
            detection_mapping = output[
                "head_detection_to_face_indices"
            ].clone()
            mapped_detection = torch.full_like(detection_mapping, -1)
            mapped_targets = detection_mapping >= 0
            mapped_detection[mapped_targets] = old_to_new[
                detection_mapping[mapped_targets]
            ]
            output["head_detection_boxes"] = detection_boxes[detection_valid]
            output["head_detection_labels"] = output[
                "head_detection_labels"
            ][detection_valid]
            output["head_detection_to_face_indices"] = mapped_detection[
                detection_valid
            ]

        selected_ellipse = self._transform_ellipses(
            target["ellipses"][int(selected["instance_index"])].reshape(1, 5),
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
        )[0]
        center_outside = bool(
            selected_ellipse[0] < 0.0
            or selected_ellipse[0] >= float(width)
            or selected_ellipse[1] < 0.0
            or selected_ellipse[1] >= float(height)
        )
        if center_outside != (selected["mode"] == "center"):
            raise RuntimeError("Ellipse border crop did not preserve the selected mode")
        output["augmentation_debug"] = {
            "ellipse_border_crop": {
                **dict(selected),
                "crop_xyxy": [x0, y0, x1, y1],
                "selected_ellipse": [
                    float(value)
                    for value in selected_ellipse.detach().cpu().tolist()
                ],
                "center_outside": center_outside,
                "contour_outside": True,
            }
        }
        return output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image, target = self.base[index]
        probability = float(self.config.get("ellipse_border_crop", 0.0) or 0.0)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index))
        if probability <= 0.0 or rng.random() >= probability:
            return image, self._clone_target(target)
        height, width = int(image.shape[-2]), int(image.shape[-1])
        selected = select_ellipse_border_crop(
            target["ellipses"],
            self.config,
            width=width,
            height=height,
            rng=rng,
        )
        if selected is None:
            return image, self._clone_target(target)
        x0, y0, x1, y1 = (int(value) for value in selected["crop_xyxy"])
        cropped = F.interpolate(
            image[:, y0:y1, x0:x1].unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        transformed_target = self._transform_target(
            target,
            selected,
            width=width,
            height=height,
        )
        return cropped.contiguous(), transformed_target


class CooperativeJointAugmentation(Dataset[Any]):
    """Deterministic full-frame augmentation preserving v=0/1/2 semantics."""

    def __init__(
        self,
        base: Dataset[Any],
        config: Mapping[str, Any],
        *,
        seed: int,
        total_epochs: int,
    ) -> None:
        self.base = base
        self.config = dict(config)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self._epoch_state = torch.tensor([0], dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self._epoch_state[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state[0])

    @staticmethod
    def _clone_target(target: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }

    @staticmethod
    def _concat_targets(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        tensor_keys = ("boxes", "ellipses", "labels", "keypoints", "labeled", "visible")
        output = {
            key: torch.cat(
                [target[key] for target in targets],
                dim=0,
            )
            for key in tensor_keys
        }
        output["image_id"] = torch.tensor(-1, dtype=torch.long)
        return output

    def _mosaic(
        self,
        samples: Sequence[tuple[torch.Tensor, Mapping[str, Any]]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        image = samples[0][0]
        height, width = int(image.shape[-2]), int(image.shape[-1])
        half_height, half_width = height // 2, width // 2
        canvas = image.new_full((3, height, width), 114.0 / 255.0)
        placements = (
            (0, 0),
            (half_width, 0),
            (0, half_height),
            (half_width, half_height),
        )
        targets = []
        for (source, raw_target), (offset_x, offset_y) in zip(
            samples,
            placements,
            strict=True,
        ):
            resized = F.interpolate(
                source.unsqueeze(0),
                size=(half_height, half_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            canvas[
                :,
                offset_y : offset_y + half_height,
                offset_x : offset_x + half_width,
            ] = resized
            target = self._clone_target(raw_target)
            target["boxes"][:, 0::2] = (
                target["boxes"][:, 0::2] * (half_width / width) + offset_x
            )
            target["boxes"][:, 1::2] = (
                target["boxes"][:, 1::2] * (half_height / height) + offset_y
            )
            target["ellipses"][:, 0] = (
                target["ellipses"][:, 0] * (half_width / width) + offset_x
            )
            target["ellipses"][:, 1] = (
                target["ellipses"][:, 1] * (half_height / height) + offset_y
            )
            target["ellipses"][:, 2] *= half_width / width
            target["ellipses"][:, 3] *= half_height / height
            target["keypoints"][..., 0] = (
                target["keypoints"][..., 0] * (half_width / width)
                + offset_x / width
            )
            target["keypoints"][..., 1] = (
                target["keypoints"][..., 1] * (half_height / height)
                + offset_y / height
            )
            targets.append(target)
        return canvas, self._concat_targets(targets)

    @staticmethod
    def _affine_target(
        target: Mapping[str, Any],
        matrix: np.ndarray,
        *,
        width: int,
        height: int,
        minimum_axis: float,
    ) -> dict[str, Any]:
        cloned = CooperativeJointAugmentation._clone_target(target)
        if cloned["boxes"].shape[0] == 0:
            return cloned
        device = cloned["boxes"].device
        dtype = cloned["boxes"].dtype
        linear = torch.tensor(matrix[:, :2], dtype=dtype, device=device)
        translation = torch.tensor(matrix[:, 2], dtype=dtype, device=device)
        ellipses = cloned["ellipses"]
        centers = ellipses[:, :2] @ linear.T + translation
        theta = ellipses[:, 4]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        covariance = (
            rotation
            @ torch.diag_embed(ellipses[:, 2:4].square())
            @ rotation.transpose(-1, -2)
        )
        transformed_covariance = linear @ covariance @ linear.T
        eigenvalues, eigenvectors = torch.linalg.eigh(transformed_covariance)
        axes = eigenvalues.clamp_min(1.0e-8).sqrt().flip(-1)
        major_vector = eigenvectors[..., :, 1]
        transformed_theta = torch.atan2(
            major_vector[..., 1],
            major_vector[..., 0],
        )
        transformed_ellipses = torch.cat(
            (centers, axes, transformed_theta.unsqueeze(-1)),
            dim=-1,
        )
        cos_theta = torch.cos(transformed_theta)
        sin_theta = torch.sin(transformed_theta)
        half_width = torch.sqrt(
            (axes[:, 0] * cos_theta).square()
            + (axes[:, 1] * sin_theta).square()
        )
        half_height = torch.sqrt(
            (axes[:, 0] * sin_theta).square()
            + (axes[:, 1] * cos_theta).square()
        )
        boxes = torch.stack(
            (
                centers[:, 0] - half_width,
                centers[:, 1] - half_height,
                centers[:, 0] + half_width,
                centers[:, 1] + half_height,
            ),
            dim=-1,
        )
        boxes[:, 0::2].clamp_(0.0, float(width))
        boxes[:, 1::2].clamp_(0.0, float(height))
        keypoints = cloned["keypoints"].clone()
        pixel_points = keypoints * keypoints.new_tensor((width, height))
        pixel_points = pixel_points @ linear.T + translation
        keypoints = pixel_points / keypoints.new_tensor((width, height))
        inside = ((keypoints >= 0.0) & (keypoints <= 1.0)).all(dim=-1)
        labeled = cloned["labeled"] & inside
        visible = cloned["visible"] & inside
        valid = (
            (centers[:, 0] >= 0.0)
            & (centers[:, 0] < width)
            & (centers[:, 1] >= 0.0)
            & (centers[:, 1] < height)
            & (axes[:, 0] >= minimum_axis)
            & (axes[:, 1] >= minimum_axis)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        cloned["boxes"] = boxes[valid]
        cloned["ellipses"] = transformed_ellipses[valid]
        cloned["labels"] = cloned["labels"][valid]
        cloned["keypoints"] = keypoints[valid].clamp(0.0, 1.0)
        cloned["labeled"] = labeled[valid]
        cloned["visible"] = visible[valid]
        return cloned

    @staticmethod
    def _flip(
        image: torch.Tensor,
        target: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        width = int(image.shape[-1])
        output = CooperativeJointAugmentation._clone_target(target)
        image = torch.flip(image, dims=(-1,))
        old_x1 = output["boxes"][:, 0].clone()
        old_x2 = output["boxes"][:, 2].clone()
        output["boxes"][:, 0] = width - old_x2
        output["boxes"][:, 2] = width - old_x1
        output["ellipses"][:, 0] = width - output["ellipses"][:, 0]
        output["ellipses"][:, 4] = math.pi - output["ellipses"][:, 4]
        order = torch.tensor((1, 0, 2, 4, 3), dtype=torch.long)
        output["keypoints"][..., 0] = 1.0 - output["keypoints"][..., 0]
        output["keypoints"] = output["keypoints"][:, order]
        output["labeled"] = output["labeled"][:, order]
        output["visible"] = output["visible"][:, order]
        return image, output

    def _photometric(
        self,
        image: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        config = self.config
        value = image.permute(1, 2, 0).contiguous().numpy().astype(np.float32)
        brightness = float(config.get("brightness", 0.0))
        contrast = float(config.get("contrast", 0.0))
        saturation = float(config.get("saturation", 0.0))
        if brightness:
            value += rng.uniform(-brightness, brightness)
        if contrast:
            mean = value.mean(axis=(0, 1), keepdims=True)
            value = mean + (value - mean) * rng.uniform(
                1.0 - contrast,
                1.0 + contrast,
            )
        if saturation:
            gray = value.mean(axis=2, keepdims=True)
            value = gray + (value - gray) * rng.uniform(
                1.0 - saturation,
                1.0 + saturation,
            )
        value = np.clip(value, 0.0, 1.0)
        if rng.random() < float(config.get("blur_probability", 0.0)):
            try:
                import cv2
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("OpenCV is required for configured blur") from error
            maximum_kernel = max(3, int(config.get("blur_kernel_max", 7)))
            if maximum_kernel % 2 == 0:
                maximum_kernel -= 1
            kernels = list(range(3, maximum_kernel + 1, 2))
            kernel = rng.choice(kernels)
            value = cv2.GaussianBlur(value, (kernel, kernel), 0)
        noise = float(config.get("noise_std", 0.0))
        if noise and rng.random() < float(config.get("noise_probability", 0.0)):
            generator = np.random.default_rng(rng.randrange(2**32))
            value = np.clip(
                value + generator.normal(0.0, noise, value.shape),
                0.0,
                1.0,
            )
        return torch.from_numpy(value).permute(2, 0, 1).to(dtype=image.dtype)

    def _occlude(
        self,
        image: torch.Tensor,
        target: dict[str, Any],
        rng: random.Random,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if rng.random() >= float(self.config.get("occlusion_probability", 0.0)):
            return image, target
        height, width = int(image.shape[-2]), int(image.shape[-1])
        area_range = self.config.get("occlusion_area", (0.02, 0.15))
        area = rng.uniform(float(area_range[0]), float(area_range[1])) * height * width
        aspect = rng.uniform(0.5, 2.0)
        rectangle_width = max(1, min(width, int(math.sqrt(area * aspect))))
        rectangle_height = max(1, min(height, int(math.sqrt(area / aspect))))
        x1 = rng.randrange(max(1, width - rectangle_width + 1))
        y1 = rng.randrange(max(1, height - rectangle_height + 1))
        output = image.clone()
        if rng.random() < 0.5:
            output[
                :,
                y1 : y1 + rectangle_height,
                x1 : x1 + rectangle_width,
            ] = torch.rand((3, 1, 1), dtype=output.dtype)
        else:
            generator = torch.Generator().manual_seed(rng.randrange(2**31))
            output[
                :,
                y1 : y1 + rectangle_height,
                x1 : x1 + rectangle_width,
            ] = torch.rand(
                (3, rectangle_height, rectangle_width),
                generator=generator,
                dtype=output.dtype,
            )
        target = self._clone_target(target)
        pixel = target["keypoints"] * target["keypoints"].new_tensor(
            (width, height)
        )
        covered = (
            (pixel[..., 0] >= x1)
            & (pixel[..., 0] < x1 + rectangle_width)
            & (pixel[..., 1] >= y1)
            & (pixel[..., 1] < y1 + rectangle_height)
        )
        target["visible"] = target["visible"] & ~covered
        return output, target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index))
        image, target = self.base[index]
        target = self._clone_target(target)
        close_mosaic = int(self.config.get("close_mosaic_epochs", 0))
        mosaic_active = self.epoch < self.total_epochs - close_mosaic
        if (
            mosaic_active
            and len(self.base) >= 4
            and rng.random() < float(self.config.get("mosaic_probability", 0.0))
        ):
            indices = [index] + [rng.randrange(len(self.base)) for _ in range(3)]
            image, target = self._mosaic([self.base[value] for value in indices])
        if (
            len(self.base) >= 2
            and rng.random() < float(self.config.get("mixup_probability", 0.0))
        ):
            other_image, other_target = self.base[rng.randrange(len(self.base))]
            mix = rng.betavariate(0.4, 0.4)
            image = (mix * image + (1.0 - mix) * other_image).clamp(0.0, 1.0)
            target = self._concat_targets((target, other_target))
        degrees = float(self.config.get("degrees", 0.0))
        translate = float(self.config.get("translate", 0.0))
        scale = float(self.config.get("scale", 0.0))
        if degrees or translate or scale:
            try:
                import cv2
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("OpenCV is required for geometry augmentation") from error
            height, width = int(image.shape[-2]), int(image.shape[-1])
            matrix = cv2.getRotationMatrix2D(
                (width * 0.5, height * 0.5),
                rng.uniform(-degrees, degrees),
                rng.uniform(max(0.1, 1.0 - scale), 1.0 + scale),
            )
            matrix[0, 2] += rng.uniform(-translate, translate) * width
            matrix[1, 2] += rng.uniform(-translate, translate) * height
            array = image.permute(1, 2, 0).contiguous().numpy()
            array = cv2.warpAffine(
                array,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114.0 / 255.0,) * 3,
            )
            image = torch.from_numpy(array).permute(2, 0, 1).to(image.dtype)
            target = self._affine_target(
                target,
                matrix,
                width=width,
                height=height,
                minimum_axis=float(self.config.get("minimum_axis", 2.0)),
            )
        if rng.random() < float(self.config.get("horizontal_flip_probability", 0.0)):
            image, target = self._flip(image, target)
        image = self._photometric(image, rng)
        image, target = self._occlude(image, target, rng)
        return image.contiguous(), target


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_cooperative_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(resolve_path(path))
    model = config.get("model", {})
    if isinstance(model, dict):
        for section_name in ("face_detector", "pose_branch"):
            section = model.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for key in ("config", "checkpoint"):
                value = section.get(key)
                if value:
                    section[key] = str(resolve_path(value))
        extension = model.get("head_pose_geometry", {})
        if isinstance(extension, dict):
            pose_encoder = extension.get("pose_encoder", {})
            if isinstance(pose_encoder, dict) and pose_encoder.get("checkpoint"):
                pose_encoder["checkpoint"] = str(
                    resolve_path(pose_encoder["checkpoint"])
                )
    data = config.get("data", {})
    if isinstance(data, dict):
        value = data.get("image_root")
        if value:
            data["image_root"] = str(resolve_path(value))
        annotations = data.get("annotations", {})
        if isinstance(annotations, dict):
            for split, value in annotations.items():
                annotations[split] = str(resolve_path(value))
        for cache_name in ("predicted_head_caches",):
            caches = data.get(cache_name, {})
            if isinstance(caches, dict):
                for split, value in caches.items():
                    if value:
                        caches[split] = str(resolve_path(value))
        if data.get("pose_pseudo_cache"):
            data["pose_pseudo_cache"] = str(
                resolve_path(data["pose_pseudo_cache"])
            )
        synthetic = data.get("synthetic", {})
        if isinstance(synthetic, dict):
            for key in ("annotation", "image_root"):
                if synthetic.get(key):
                    synthetic[key] = str(resolve_path(synthetic[key]))
    output = config.get("output", {})
    if isinstance(output, dict) and output.get("root"):
        output["root"] = str(resolve_path(output["root"]))
    validate_strict_sequential_config(config)
    training_stage_policy(config)
    return config


def training_stage_policy(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and return an optional causal training-stage contract."""

    train = config.get("train", {})
    raw = train.get("stage") if isinstance(train, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("train.stage must be a mapping")
    objective = config.get("experiment_objective", {})
    if not isinstance(objective, Mapping):
        raise TypeError("experiment_objective must be a mapping")
    if str(objective.get("type")) != "cascade_comparison":
        raise ValueError(
            "Stagewise frozen training requires "
            "experiment_objective.type=cascade_comparison"
        )
    if bool(objective.get("joint_fine_tuning", False)) or bool(
        objective.get("final_performance_claim", False)
    ):
        raise ValueError(
            "Cascade-comparison stages cannot claim joint fine-tuning or "
            "final performance"
        )
    name = str(raw.get("name", "")).strip()
    trainable_groups = raw.get("trainable_groups")
    loss_weights = raw.get("loss_weights")
    if not name:
        raise ValueError("train.stage.name is required")
    if not isinstance(trainable_groups, Sequence) or isinstance(
        trainable_groups, (str, bytes)
    ) or not trainable_groups:
        raise ValueError("train.stage.trainable_groups must be a non-empty list")
    groups = [str(value) for value in trainable_groups]
    if len(groups) != len(set(groups)):
        raise ValueError("train.stage.trainable_groups contains duplicates")
    if not isinstance(loss_weights, Mapping) or not loss_weights:
        raise ValueError("train.stage.loss_weights must be a non-empty mapping")
    weights = {str(key): float(value) for key, value in loss_weights.items()}
    if any(
        not key.startswith("loss_")
        or not math.isfinite(value)
        or value <= 0.0
        for key, value in weights.items()
    ):
        raise ValueError(
            "train.stage.loss_weights requires positive finite loss_* weights"
        )
    return {
        "name": name,
        "trainable_groups": groups,
        "loss_weights": weights,
    }


def configure_training_stage(
    model: FaceKeypointCooperativeDETR,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Freeze every parameter group outside the configured causal stage."""

    policy = training_stage_policy(config)
    if policy is None:
        return None
    grouped = model.parameter_groups()
    unknown = sorted(set(policy["trainable_groups"]) - set(grouped))
    if unknown:
        raise ValueError(f"Unknown train.stage parameter groups: {unknown}")
    trainable = set(policy["trainable_groups"])
    for group_name, parameters in grouped.items():
        if group_name not in trainable:
            for parameter in parameters:
                parameter.requires_grad_(False)
    report = {
        **policy,
        "parameter_counts": {
            name: sum(
                int(parameter.numel())
                for parameter in parameters
                if parameter.requires_grad
            )
            for name, parameters in grouped.items()
        },
    }
    model.pretrained_load_info["training_stage"] = report
    return report


def enforce_training_stage_modes(
    model: nn.Module,
    config: Mapping[str, Any],
) -> None:
    """Keep frozen upstream modules deterministic after ``model.train()``."""

    policy = training_stage_policy(config)
    if policy is None:
        return
    value = unwrap_model(model)
    active = set(policy["trainable_groups"])
    backbone = value.face_detector.backbone
    if "backbone" not in active and "sta_neck" not in active:
        backbone.eval()
    elif "backbone" not in active:
        dino = getattr(backbone, "dino", None)
        if dino is None:
            raise TypeError("Cannot isolate the frozen DINO submodule")
        dino.eval()
    module_groups = {
        "shared_encoder": value.face_detector.encoder,
        "face_decoder": value.face_detector.decoder,
        "keypoint_decoder": value.pose_decoder,
        "association": value.association,
        "cooperative_decoder": value.cooperative_decoder,
        "head_pose_geometry": value.head_pose_geometry,
        "external_head_cascade": value.external_head_cascade,
        "shared_head_branch": value.shared_head_branch,
    }
    for group_name, module in module_groups.items():
        if module is not None and group_name not in active:
            module.eval()


def apply_training_stage_loss(
    output: dict[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Replace the joint total with the configured stage-specific objective."""

    policy = training_stage_policy(config)
    if policy is None:
        return output["loss_total"]
    terms = []
    for key, weight in policy["loss_weights"].items():
        value = output.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise KeyError(
                f"Training stage {policy['name']!r} requires scalar {key!r}"
            )
        terms.append(value * float(weight))
    total = sum(terms[1:], terms[0])
    output["loss_total"] = total
    return total


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def distributed_context(device_argument: str) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        requested = torch.device(device_argument)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        device = requested
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
    }


def close_distributed(context: Mapping[str, Any]) -> None:
    if bool(context["distributed"]) and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def configure_device_math(
    device: torch.device,
    train_config: Mapping[str, Any],
) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(train_config.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def build_dataset(
    config: Mapping[str, Any],
    split: str,
    *,
    training: bool,
    maximum_samples: int | None = None,
) -> torch.utils.data.Dataset[Any]:
    data = config["data"]
    head_centric = bool(data.get("head_centric", False))
    if head_centric:
        head_crop_cfg = data.get("head_crop", {})
        head_crop_enabled = bool(
            isinstance(head_crop_cfg, Mapping)
            and head_crop_cfg.get("enabled", False)
        )
        shared_head_cfg = (
            config.get("model", {}).get("shared_head_branch", {})
            if isinstance(config.get("model", {}), Mapping)
            else {}
        )
        internal_shared_head = bool(
            isinstance(shared_head_cfg, Mapping)
            and shared_head_cfg.get("enabled", False)
        )
        head_caches = data.get("predicted_head_caches", {})
        dataset = HeadCentricFaceDataset(
            annotation=data["annotations"][split],
            image_root=data["image_root"],
            split=split,
            image_size=int(data.get("image_size", 960)),
            target_category_id=int(data.get("target_category_id", 2)),
            predicted_head_cache=(
                head_caches.get(split)
                if isinstance(head_caches, Mapping)
                else None
            ),
            pose_pseudo_cache=(
                data.get("pose_pseudo_cache")
                if training and split == "train"
                else None
            ),
            require_predicted_heads=(
                bool(head_crop_cfg.get("require_predicted_heads", True))
                if head_crop_enabled
                else not internal_shared_head
            ),
        )
        if head_crop_enabled:
            dataset = HeadCropCooperativeDataset(
                dataset,
                output_size=int(
                    head_crop_cfg.get("output_size", data.get("image_size", 960))
                ),
                context_scale=float(head_crop_cfg.get("context_scale", 1.20)),
                pad_value=float(head_crop_cfg.get("pad_value", 0.0)),
            )
    else:
        dataset = JointFaceDataset(
            annotation=data["annotations"][split],
            image_root=data["image_root"],
            image_size=int(data.get("image_size", 960)),
            category_ids=tuple(int(value) for value in data.get("category_ids", (2,))),
            target_category_id=int(data.get("target_category_id", 2)),
            # Joint geometric transforms must be applied to the face, ellipse,
            # and keypoint targets together.
            training=False,
        )
    augmentation = data.get("augmentation", {})
    if (
        training
        and head_centric
        and isinstance(augmentation, Mapping)
        and bool(augmentation.get("enabled", True))
        and float(augmentation.get("ellipse_border_crop", 0.0) or 0.0) > 0.0
    ):
        dataset = HeadCentricEllipseBorderCropAugmentation(
            dataset,
            augmentation,
            seed=int(config["train"].get("seed", 42)),
        )
    if (
        training
        and not head_centric
        and isinstance(augmentation, Mapping)
        and bool(augmentation.get("enabled", True))
    ):
        dataset = CooperativeJointAugmentation(
            dataset,
            augmentation,
            seed=int(config["train"].get("seed", 42)),
            total_epochs=int(config["train"].get("epochs", 150)),
        )
    synthetic_cfg = data.get("synthetic", {})
    if (
        training
        and split == "train"
        and isinstance(synthetic_cfg, Mapping)
        and bool(synthetic_cfg.get("enabled", False))
    ):
        for required in ("annotation", "image_root"):
            if not synthetic_cfg.get(required):
                raise ValueError(
                    f"data.synthetic.{required} is required when enabled"
                )
        synthetic_dataset = HeadCentricFaceDataset(
            annotation=synthetic_cfg["annotation"],
            image_root=synthetic_cfg["image_root"],
            split="train",
            image_size=int(data.get("image_size", 960)),
            target_category_id=int(data.get("target_category_id", 2)),
            synthetic_rotation_supervision=True,
        )
        dataset = InterleavedRealSyntheticDataset(
            dataset,
            synthetic_dataset,
            synthetic_fraction=float(
                synthetic_cfg.get("fraction", 0.25)
            ),
            seed=int(config["train"].get("seed", 42)),
        )
    if maximum_samples is not None and maximum_samples < len(dataset):
        indices, subset_report = representative_subset_indices(
            dataset,
            int(maximum_samples),
        )
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset.representative_subset_report = subset_report
    return dataset


def set_dataset_epoch(dataset: Dataset[Any], epoch: int) -> None:
    """Propagate the epoch through Subset wrappers to deterministic workers."""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
        return
    wrapped = getattr(dataset, "dataset", None)
    if isinstance(wrapped, Dataset):
        set_dataset_epoch(wrapped, epoch)


def build_loader(
    config: Mapping[str, Any],
    split: str,
    *,
    training: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    batch_size: int | None = None,
    maximum_samples: int | None = None,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    dataset = build_dataset(
        config,
        split,
        training=training,
        maximum_samples=maximum_samples,
    )
    train = config["train"]
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            seed=int(train.get("seed", 42)),
            drop_last=False,
        )
        if distributed
        else None
    )
    workers = int(train.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=int(
            batch_size
            or (
                train.get("batch_size", 2)
                if training
                else config.get("evaluation", {}).get("batch_size", 1)
            )
        ),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=int(train.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=False,
        collate_fn=collate_joint,
    )
    return loader, sampler


def move_batch(
    batch: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = batch
    return (
        images.to(device, non_blocking=True),
        [
            {
                key: (
                    value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in target.items()
            }
            for target in targets
        ],
    )


def autocast_context(
    device: torch.device,
    train_config: Mapping[str, Any],
):
    dtype_name = str(train_config.get("amp_dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.updates = 0
        unwrapped = unwrap_model(model)
        source = unwrapped.state_dict()
        self.averaged_keys = {
            name
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad and torch.is_floating_point(parameter)
        }
        self.shadow = {
            key: value.detach().clone()
            for key, value in source.items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        source = unwrap_model(model).state_dict()
        effective_decay = min(
            self.decay,
            (1.0 + self.updates) / (10.0 + self.updates),
        )
        for key, shadow in self.shadow.items():
            value = source[key].detach()
            if key in self.averaged_keys:
                shadow.lerp_(value, 1.0 - effective_decay)
            else:
                # Frozen upstream weights and buffers must remain bit-exact;
                # repeatedly averaging equal values introduces float rounding.
                shadow.copy_(value)

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        unwrapped = unwrap_model(model)
        state = unwrapped.state_dict()
        backup = {
            key: state[key].detach().clone()
            for key in self.shadow
        }
        with torch.no_grad():
            for key, value in self.shadow.items():
                state[key].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for key, value in backup.items():
                    state[key].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "updates": self.updates,
            "averaged_keys": sorted(self.averaged_keys),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        shadow = state.get("shadow", state)
        if not isinstance(shadow, Mapping):
            raise TypeError("EMA checkpoint has no shadow state")
        if set(shadow) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(shadow))
            unexpected = sorted(set(shadow) - set(self.shadow))
            raise RuntimeError(
                f"EMA state mismatch: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        self.shadow = {
            str(key): value.detach().clone()
            for key, value in shadow.items()
        }
        self.decay = float(state.get("decay", self.decay))
        self.updates = int(state.get("updates", 0))
        stored_keys = state.get("averaged_keys")
        if stored_keys is not None:
            if not isinstance(stored_keys, Sequence) or isinstance(
                stored_keys, (str, bytes)
            ):
                raise TypeError("EMA averaged_keys must be a sequence")
            if set(str(key) for key in stored_keys) != self.averaged_keys:
                raise RuntimeError(
                    "EMA trainable-parameter contract changed across resume"
                )


def unwrap_model(model: nn.Module) -> FaceKeypointCooperativeDETR:
    value = model.module if hasattr(model, "module") else model
    if not isinstance(value, FaceKeypointCooperativeDETR):
        raise TypeError(f"Expected FaceKeypointCooperativeDETR, got {type(value)}")
    return value


def build_model(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> FaceKeypointCooperativeDETR:
    model = build_face_keypoint_cooperative_detr(config).to(device)
    configure_training_stage(model, config)
    if checkpoint is not None:
        load_cooperative_checkpoint_strict(model, checkpoint, prefer_ema=True)
    return model


def initialize_shared_head_checkpoint(
    model: FaceKeypointCooperativeDETR,
    checkpoint: str | Path,
    *,
    prefer_ema: bool = True,
    load_shared_dino: bool = False,
) -> dict[str, Any]:
    if model.shared_head_branch is None:
        raise ValueError(
            "A Head Co-DINO checkpoint was supplied, but "
            "model.shared_head_branch is disabled"
        )
    report = initialize_shared_head_from_head_codino(
        model.shared_head_branch,
        resolve_path(checkpoint),
        prefer_ema=prefer_ema,
    )
    model.pretrained_load_info["shared_head_branch"] = {
        **report,
        "initialization": "shared_head_codino_checkpoint",
        "backbone": "shared_face_dinov3_vits16plus",
    }
    if load_shared_dino:
        dino = getattr(model.face_detector.backbone, "dino", None)
        dino_backbone = getattr(dino, "backbone", None)
        if dino_backbone is None:
            raise TypeError("Shared FACE backbone has no loadable DINO module")
        dino_report = initialize_shared_dino_from_head_codino(
            dino_backbone,
            resolve_path(checkpoint),
            prefer_ema=prefer_ema,
        )
        model.pretrained_load_info["shared_dino_backbone"] = {
            **dino_report,
            "initialization": "head_codino_checkpoint",
        }
        report = {**report, "shared_dino_backbone": dino_report}
    return report


def initialize_stage_checkpoint(
    model: FaceKeypointCooperativeDETR,
    path: str | Path,
) -> dict[str, Any]:
    """Load a complete previous stage, overlaying its trainable EMA tensors."""

    checkpoint = Path(path)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state: Any = raw
    if isinstance(raw, Mapping):
        full_state = raw.get("model")
        if isinstance(full_state, Mapping):
            # Stage EMA intentionally tracks trainable parameters only. The
            # full model state remains authoritative for frozen parameters and
            # buffers; EMA shadows replace only the tensors they contain.
            combined = dict(full_state)
            ema = raw.get("ema")
            if isinstance(ema, Mapping):
                shadow = ema.get("shadow", ema)
                if isinstance(shadow, Mapping):
                    combined.update(shadow)
            state = combined
        else:
            for key in ("state_dict", "ema"):
                candidate = raw.get(key)
                if key == "ema" and isinstance(candidate, Mapping):
                    candidate = candidate.get("shadow", candidate)
                if isinstance(candidate, Mapping):
                    state = candidate
                    break
    if not isinstance(state, Mapping):
        raise TypeError(f"No model state mapping in {checkpoint}")
    source = {}
    for raw_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        source[key] = value
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = sorted(
        key
        for key in set(source) & set(target)
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    # Stage transitions may add or resize only optional cascade/geometry
    # tensors. Any discrepancy in the historical direct model is a hard error.
    optional_prefixes = (
        "head_pose_geometry.",
        "external_head_cascade.",
        "shared_head_branch.",
    )
    forbidden_unexpected = [
        key for key in unexpected if not key.startswith(optional_prefixes)
    ]
    forbidden_shape = [
        key for key in shape_mismatch if not key.startswith(optional_prefixes)
    ]
    if forbidden_unexpected or forbidden_shape:
        raise RuntimeError(
            "Stage checkpoint is incompatible with the direct model: "
            f"unexpected={forbidden_unexpected[:20]}, "
            f"shape_mismatch={forbidden_shape[:20]}"
        )
    missing = sorted(set(target) - set(compatible))
    forbidden_missing = [
        key for key in missing if not key.startswith(optional_prefixes)
    ]
    if forbidden_missing:
        raise RuntimeError(
            "Stage checkpoint is missing historical direct tensors: "
            f"{forbidden_missing[:20]}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint),
        "loaded_tensors": len(compatible),
        "loaded_elements": sum(int(value.numel()) for value in compatible.values()),
        "new_stage_tensors": missing,
        "ignored_previous_stage_tensors": unexpected,
        "resized_stage_tensors": shape_mismatch,
    }


def build_optimizer(
    model: FaceKeypointCooperativeDETR,
    config: Mapping[str, Any],
) -> tuple[AdamW, list[dict[str, Any]]]:
    train = config["train"]
    learning_rates = dict(train.get("learning_rates", {}))
    default_lr = float(train.get("lr", 5.0e-4))
    weight_decay = float(train.get("weight_decay", 0.05))
    grouped = model.parameter_groups()
    optimizer_groups = []
    report = []
    assigned: set[int] = set()
    for group_name, parameters in grouped.items():
        unique = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        assigned.update(id(parameter) for parameter in unique)
        if not unique:
            continue
        lr = float(learning_rates.get(group_name, default_lr))
        decay_parameters = [
            parameter
            for parameter in unique
            if parameter.ndim > 1
        ]
        no_decay_parameters = [
            parameter
            for parameter in unique
            if parameter.ndim <= 1
        ]
        if decay_parameters:
            optimizer_groups.append(
                {
                    "params": decay_parameters,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group_name": group_name,
                }
            )
        if no_decay_parameters:
            optimizer_groups.append(
                {
                    "params": no_decay_parameters,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "group_name": group_name + "_no_decay",
                }
            )
        report.append(
            {
                "name": group_name,
                "lr": lr,
                "parameters": sum(parameter.numel() for parameter in unique),
                "tensors": len(unique),
            }
        )
    all_trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing_ids = set(all_trainable) - assigned
    if missing_ids:
        missing_names = [all_trainable[value] for value in missing_ids]
        raise RuntimeError(
            f"Trainable parameters were not assigned to an optimizer group: "
            f"{missing_names[:30]}"
        )
    optimizer = AdamW(
        optimizer_groups,
        lr=default_lr,
        betas=tuple(float(value) for value in train.get("betas", (0.9, 0.999))),
        eps=float(train.get("eps", 1.0e-8)),
        fused=bool(train.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    return optimizer, report


def scheduler_lambda(
    *,
    epochs: int,
    warmup_epochs: int,
    flat_epochs: int,
    minimum_ratio: float,
):
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(1.0e-3, (epoch + 1) / max(1, warmup_epochs))
        if epoch < flat_epochs:
            return 1.0
        progress = min(
            1.0,
            (epoch - flat_epochs) / max(1, epochs - flat_epochs),
        )
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return schedule


def reduce_mean(value: torch.Tensor, context: Mapping[str, Any]) -> torch.Tensor:
    detached = value.detach().clone()
    if bool(context["distributed"]):
        torch.distributed.all_reduce(detached, op=torch.distributed.ReduceOp.SUM)
        detached /= int(context["world_size"])
    return detached


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    best_metric: float,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "best_metric": float(best_metric),
            "config": copy.deepcopy(dict(config)),
            "metrics": dict(metrics),
        },
        temporary,
    )
    temporary.replace(path)


def _strict_sequential_resume_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        return {"enabled": False}
    sequential = model.get("strict_sequential_cascade", {})
    if not isinstance(sequential, Mapping) or not bool(
        sequential.get("enabled", False)
    ):
        return {"enabled": False}
    shared_head = model.get("shared_head_branch", {})
    shared_head = shared_head if isinstance(shared_head, Mapping) else {}
    external_head = model.get("external_head_cascade", {})
    external_head = external_head if isinstance(external_head, Mapping) else {}
    head_to_face = sequential.get("head_to_face", {})
    head_to_face = head_to_face if isinstance(head_to_face, Mapping) else {}
    face_to_keypoint = sequential.get("face_to_keypoint", {})
    face_to_keypoint = (
        face_to_keypoint if isinstance(face_to_keypoint, Mapping) else {}
    )
    return {
        "enabled": True,
        "training_head_source": sequential.get("training_head_source"),
        "training_face_source": sequential.get("training_face_source"),
        "evaluation_head_source": sequential.get("evaluation_head_source"),
        "evaluation_face_source": sequential.get("evaluation_face_source"),
        "enforce_unidirectional_refinement": sequential.get(
            "enforce_unidirectional_refinement"
        ),
        "face_score_threshold": sequential.get("face_score_threshold"),
        "maximum_face_rois": sequential.get("maximum_face_rois"),
        "head_to_face": copy.deepcopy(dict(head_to_face)),
        "face_to_keypoint": copy.deepcopy(dict(face_to_keypoint)),
        "shared_head_enabled": bool(shared_head.get("enabled", False)),
        "shared_head_teacher_force_targets": bool(
            shared_head.get("teacher_force_targets", False)
        ),
        "external_head_cascade_enabled": bool(
            external_head.get("enabled", False)
        ),
    }


def resume_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    current_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if current_config is not None:
        checkpoint_config = state.get("config", {})
        checkpoint_contract = _strict_sequential_resume_contract(
            checkpoint_config if isinstance(checkpoint_config, Mapping) else {}
        )
        current_contract = _strict_sequential_resume_contract(current_config)
        if checkpoint_contract != current_contract:
            raise ValueError(
                "Checkpoint strict-sequential architecture contract does not "
                "match the current config; resume is unsafe and a fresh "
                "training run is required. "
                f"checkpoint={checkpoint_contract}, current={current_contract}"
            )
    unwrap_model(model).load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    result = {
        "start_epoch": int(state["epoch"]) + 1,
        "step": int(state.get("step", 0)),
        "best_metric": float(state.get("best_metric", float("inf"))),
    }
    if current_config is not None:
        checkpoint_config = state.get("config", {})
        checkpoint_evaluation = (
            checkpoint_config.get("evaluation", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        current_evaluation = current_config.get("evaluation", {})
        checkpoint_batch_size = int(
            checkpoint_evaluation.get("batch_size", 1)
            if isinstance(checkpoint_evaluation, Mapping)
            else 1
        )
        current_batch_size = int(
            current_evaluation.get("batch_size", 1)
            if isinstance(current_evaluation, Mapping)
            else 1
        )
        reset = checkpoint_batch_size != current_batch_size
        if reset:
            # Validation losses aggregated with different batch sizes are not
            # guaranteed to be directly comparable for detection criteria.
            result["best_metric"] = float("inf")
        result["validation_metric_contract"] = {
            "reset_best_metric": reset,
            "checkpoint_batch_size": checkpoint_batch_size,
            "current_batch_size": current_batch_size,
        }
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_loss_dict(output: Mapping[str, Any]) -> dict[str, float]:
    losses = {}
    for key, value in output.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 0
            and (key.startswith("loss") or key == "association_accuracy")
        ):
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite scalar {key}={value}")
            losses[key] = float(value.detach())
    if "loss_total" not in losses:
        raise RuntimeError("Model output did not contain a scalar loss_total")
    return losses


__all__ = [
    "CooperativeJointAugmentation",
    "HeadCentricEllipseBorderCropAugmentation",
    "ModelEMA",
    "REPO_ROOT",
    "autocast_context",
    "apply_training_stage_loss",
    "build_dataset",
    "build_loader",
    "build_model",
    "build_optimizer",
    "close_distributed",
    "configure_device_math",
    "configure_training_stage",
    "distributed_context",
    "finite_loss_dict",
    "enforce_training_stage_modes",
    "load_cooperative_config",
    "initialize_stage_checkpoint",
    "initialize_shared_head_checkpoint",
    "move_batch",
    "reduce_mean",
    "resolve_path",
    "resume_checkpoint",
    "save_checkpoint",
    "scheduler_lambda",
    "seed_everything",
    "set_dataset_epoch",
    "training_stage_policy",
    "unwrap_model",
    "write_json",
]
