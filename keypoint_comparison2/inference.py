from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import (
    image_to_tensor,
    load_checkpoint,
    load_config,
    scale_detections_to_original,
)

from .data import build_affine_crop
from src.data.face_crop_cache import crop_geometry_from_detection


class DEIMv2LFaceDetector:
    """Fixed DEIMv2-L face detector shared by all top-down keypoint models."""

    def __init__(self, config: Mapping[str, Any], *, device: torch.device) -> None:
        cfg = dict(config)
        self.device = device
        self.input_size = int(cfg.get("input_size", 960))
        self.score_threshold = float(cfg.get("score_threshold", 0.33))
        self.maximum_detections = int(cfg.get("maximum_detections", 100))
        self.crop_geometry = str(cfg.get("crop_geometry", "auto"))
        if self.crop_geometry not in {"auto", "bbox", "ellipse"}:
            raise ValueError("detector.crop_geometry must be auto, bbox, or ellipse")
        self.model = build_detector_from_config(load_config(Path(str(cfg["config"])))).to(device)
        missing, unexpected = load_checkpoint(self.model, Path(str(cfg["checkpoint"])), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DEIMv2-L checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if hasattr(self.model, "score_threshold"):
            self.model.score_threshold = min(0.001, self.score_threshold)
        if hasattr(self.model, "postprocessor") and hasattr(self.model.postprocessor, "score_threshold"):
            self.model.postprocessor.score_threshold = min(0.001, self.score_threshold)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        tensor, input_hw, original_hw = image_to_tensor(pil.convert("RGB"), self.input_size)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            prediction = self.model(tensor.to(self.device))[0]
        prediction = scale_detections_to_original(prediction, input_hw=input_hw, original_hw=original_hw)
        order = prediction["scores"].argsort(descending=True)
        has_ellipses = (
            isinstance(prediction.get("ellipses"), torch.Tensor)
            and isinstance(prediction.get("ellipse_boxes"), torch.Tensor)
            and prediction["ellipses"].shape[0] == prediction["boxes"].shape[0]
            and prediction["ellipse_boxes"].shape[0] == prediction["boxes"].shape[0]
        )
        if self.crop_geometry == "ellipse" and not has_ellipses:
            raise RuntimeError(
                "detector.crop_geometry=ellipse requires an ellipse DEIM checkpoint output"
            )
        use_ellipse = has_ellipses and self.crop_geometry in {"auto", "ellipse"}
        results = []
        for index in order.tolist():
            score = float(prediction["scores"][index])
            if score < self.score_threshold:
                continue
            detector_box = prediction["boxes"][index].detach().cpu().float().tolist()
            box = (
                prediction["ellipse_boxes"][index].detach().cpu().float().tolist()
                if use_ellipse
                else detector_box
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            result = {
                "bbox_xyxy": box,
                "detector_bbox_xyxy": detector_box,
                "score": score,
                "crop_geometry": "ellipse" if use_ellipse else "bbox",
            }
            if has_ellipses:
                result["ellipse"] = (
                    prediction["ellipses"][index].detach().cpu().float().tolist()
                )
                result["ellipse_bbox_xyxy"] = (
                    prediction["ellipse_boxes"][index].detach().cpu().float().tolist()
                )
            results.append(result)
            if len(results) == self.maximum_detections:
                break
        return results


class TopDownFaceKeypointPipeline:
    """DEIM ellipse -> long-axis square with real background -> top-down pose."""

    def __init__(
        self,
        detector: DEIMv2LFaceDetector,
        pose_model: nn.Module,
        *,
        input_size: int,
        device: torch.device,
        crop_scale: float = 1.25,
        batch_size: int = 32,
        flip_test: bool = True,
    ) -> None:
        self.detector = detector
        self.pose_model = pose_model.to(device).eval()
        self.input_size = int(input_size)
        self.device = device
        self.crop_scale = float(crop_scale)
        self.batch_size = int(batch_size)
        self.flip_test = bool(flip_test)

    @torch.inference_mode()
    def __call__(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        detections = self.detector(rgb)
        if not detections:
            return []
        crops, transforms = [], []
        for detection in detections:
            crop_box, crop_rotation = crop_geometry_from_detection(
                detection,
                require_ellipse=True,
            )
            transform = build_affine_crop(
                crop_box,
                output_size=self.input_size,
                scale=self.crop_scale,
                rotation=crop_rotation,
                warp_bbox_aspect=False,
            )
            crop = cv2.warpAffine(
                rgb,
                transform.matrix,
                (self.input_size, self.input_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            crops.append(torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1) / 255.0)
            transforms.append(transform)
        predictions: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[start : start + self.batch_size]).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                output = self.pose_model(batch)
                coordinates = output["coordinates"]
                visibility = output["visibility_logits"].sigmoid()
                if self.flip_test:
                    flipped_output = self.pose_model(torch.flip(batch, dims=(-1,)))
                    flipped_coordinates = flipped_output["coordinates"][:, [1, 0, 2, 4, 3]].clone()
                    flipped_coordinates[..., 0] = 1.0 - flipped_coordinates[..., 0]
                    flipped_visibility = flipped_output["visibility_logits"].sigmoid()[:, [1, 0, 2, 4, 3]]
                    coordinates = 0.5 * (coordinates + flipped_coordinates)
                    visibility = 0.5 * (visibility + flipped_visibility)
            predictions.extend(
                zip(
                    coordinates.detach().cpu().float(),
                    visibility.detach().cpu().float(),
                )
            )
        results = []
        for detection, transform, (coordinates, visibility) in zip(detections, transforms, predictions):
            points = coordinates.numpy() * (self.input_size - 1)
            homogeneous = np.concatenate((points, np.ones((5, 1), dtype=np.float32)), axis=1)
            original = homogeneous @ transform.inverse.T
            results.append(
                {
                    **detection,
                    "keypoints": original.tolist(),
                    "visibility": visibility.tolist(),
                }
            )
        return results


class JointFaceKeypointPipeline:
    """Single-pass DEIMv2 face bbox + five landmark query inference."""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: int,
        device: torch.device,
        score_threshold: float = 0.33,
        visibility_threshold: float = 0.50,
    ) -> None:
        self.model = model.to(device).eval()
        self.input_size = int(input_size)
        self.device = device
        self.score_threshold = float(score_threshold)
        self.visibility_threshold = float(visibility_threshold)

    @torch.inference_mode()
    def __call__(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        original_width, original_height = pil.size
        resized = pil.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(tensor.to(self.device))
        boxes = output["pred_boxes"][0].detach().cpu().float()
        keypoints = output["pred_keypoints"][0].detach().cpu().float()
        scores = output["scores"][0].detach().cpu().float()
        visibility = output["visibility"][0].detach().cpu().float()
        ellipse_values: torch.Tensor | None = None
        ellipse_boxes: torch.Tensor | None = None
        if "pred_ellipses" in output:
            from src.models.dinov3_deimv2_ellipse import (
                ellipse_to_aabb,
                scale_ellipses_anisotropic,
            )

            ellipse_values = output["pred_ellipses"][0].detach().cpu().float().clone()
            ellipse_values[:, :4] *= float(self.input_size)
            ellipse_values = scale_ellipses_anisotropic(
                ellipse_values,
                scale_x=original_width / float(self.input_size),
                scale_y=original_height / float(self.input_size),
            )
            ellipse_boxes = ellipse_to_aabb(ellipse_values)
            ellipse_boxes[:, 0::2].clamp_(0, float(original_width))
            ellipse_boxes[:, 1::2].clamp_(0, float(original_height))
        results = []
        for index, (box, points, score, point_visibility) in enumerate(
            zip(boxes, keypoints, scores, visibility)
        ):
            if float(score) < self.score_threshold:
                continue
            cx, cy, width, height = box.tolist()
            bbox = [
                (cx - width / 2) * original_width,
                (cy - height / 2) * original_height,
                (cx + width / 2) * original_width,
                (cy + height / 2) * original_height,
            ]
            scaled_points = points * torch.tensor([original_width, original_height])
            result = {
                "bbox_xyxy": bbox,
                "score": float(score),
                "keypoints": scaled_points.tolist(),
                "visibility": point_visibility.tolist(),
                "visible": (point_visibility >= self.visibility_threshold).tolist(),
            }
            if ellipse_values is not None and ellipse_boxes is not None:
                result["ellipse"] = ellipse_values[index].tolist()
                result["ellipse_bbox_xyxy"] = ellipse_boxes[index].tolist()
            results.append(result)
        return results
