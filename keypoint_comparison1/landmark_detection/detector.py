from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import (
    load_checkpoint,
    load_config,
    parse_image_size,
    prepare_image_tensor,
    scale_detections_to_original,
)


class DeimFaceDetector:
    """Read-only DEIMv2-L face detector used before top-down landmark crops."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        image_size: int | tuple[int, int] | list[int] = 960,
        resize_mode: str | None = None,
        pad_color: int | tuple[int, int, int] | list[int] | None = None,
        size_divisor: int | None = None,
        score_threshold: float = 0.33,
        maximum_detections: int = 100,
        crop_geometry: str = "ellipse",
        return_face_query_features: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        parsed_size = parse_image_size(image_size)
        if parsed_size is None:
            raise ValueError("image_size must be configured")
        self.image_size = parsed_size
        self.score_threshold = float(score_threshold)
        self.maximum_detections = int(maximum_detections)
        self.crop_geometry = str(crop_geometry)
        self.return_face_query_features = bool(return_face_query_features)
        if self.crop_geometry not in {"ellipse", "bbox"}:
            raise ValueError("crop_geometry must be ellipse or bbox")
        config = load_config(self.config_path)
        data_cfg = dict(config.get("data", {}) or {})
        self.resize_mode = str(
            resize_mode
            if resize_mode is not None
            else data_cfg.get("resize_mode", "stretch")
        )
        self.pad_color = (
            pad_color if pad_color is not None else data_cfg.get("pad_color")
        )
        self.size_divisor = int(
            size_divisor
            if size_divisor is not None
            else data_cfg.get("size_divisor", 32)
        )
        self.model = build_detector_from_config(config).to(self.device)
        if self.return_face_query_features:
            if not hasattr(self.model, "set_face_query_output"):
                raise ValueError("FOR requires an ellipse detector with face query output")
            self.model.set_face_query_output(True)
        missing, unexpected = load_checkpoint(self.model, self.checkpoint_path, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DEIM detector checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if hasattr(self.model, "score_threshold"):
            self.model.score_threshold = min(0.001, self.score_threshold)
        if hasattr(self.model, "postprocessor") and hasattr(self.model.postprocessor, "score_threshold"):
            self.model.postprocessor.score_threshold = min(0.001, self.score_threshold)
        self.model.eval()

    def _serialize_detections(
        self,
        detections: dict[str, torch.Tensor],
        *,
        input_hw: tuple[int, int],
        original_hw: tuple[int, int],
        geometry: Any | None = None,
    ) -> list[dict[str, Any]]:
        detections = scale_detections_to_original(
            detections,
            input_hw=input_hw,
            original_hw=original_hw,
            geometry=geometry,
        )
        boxes = detections["boxes"].detach().cpu()
        scores = detections["scores"].detach().cpu()
        labels = detections.get("labels")
        labels = (
            labels.detach().cpu()
            if isinstance(labels, torch.Tensor)
            else torch.zeros_like(scores, dtype=torch.long)
        )
        ellipses = detections.get("ellipses")
        ellipse_boxes = detections.get("ellipse_boxes")
        has_ellipses = (
            isinstance(ellipses, torch.Tensor)
            and isinstance(ellipse_boxes, torch.Tensor)
            and ellipses.shape[0] == boxes.shape[0]
            and ellipse_boxes.shape[0] == boxes.shape[0]
        )
        if self.crop_geometry == "ellipse" and not has_ellipses:
            raise RuntimeError("The configured ellipse crop requires ellipse detector outputs")
        ellipses = ellipses.detach().cpu() if has_ellipses else None
        ellipse_boxes = ellipse_boxes.detach().cpu() if has_ellipses else None
        order = torch.argsort(scores, descending=True)
        rows: list[dict[str, Any]] = []
        for index in order.tolist():
            score = float(scores[index])
            if score < self.score_threshold:
                continue
            box = [float(value) for value in boxes[index].tolist()]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            row: dict[str, Any] = {
                "bbox_xyxy": box,
                "detector_bbox_xyxy": box,
                "score": score,
                "label": int(labels[index]),
                "crop_geometry": self.crop_geometry,
            }
            if ellipses is not None and ellipse_boxes is not None:
                row["ellipse"] = [float(value) for value in ellipses[index].tolist()]
                row["ellipse_bbox_xyxy"] = [
                    float(value) for value in ellipse_boxes[index].tolist()
                ]
            if getattr(self, "return_face_query_features", False):
                row["face_query_index"] = int(detections["face_query_index"][index])
                row["face_query_features"] = detections["face_query_features"][index].detach().float().cpu().tolist()
            rows.append(row)
            if len(rows) >= self.maximum_detections:
                break
        return rows

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Sequence[Image.Image | np.ndarray],
    ) -> list[list[dict[str, Any]]]:
        if not images:
            return []
        tensors: list[torch.Tensor] = []
        geometries: list[tuple[tuple[int, int], tuple[int, int], Any | None]] = []
        padding_masks: list[torch.Tensor | None] = []
        for image in images:
            pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            tensor, input_hw, original_hw, geometry, padding_mask = prepare_image_tensor(
                pil.convert("RGB"),
                self.image_size,
                resize_mode=self.resize_mode,
                pad_color=self.pad_color,
                size_divisor=self.size_divisor,
            )
            tensors.append(tensor)
            geometries.append((input_hw, original_hw, geometry))
            padding_masks.append(padding_mask)
        shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(
                "DEIM batch inputs must share one tensor shape; configure a fixed image_size"
            )
        batch = torch.cat(tensors, dim=0)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            detections = self.model(
                batch.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                ),
                padding_mask=(
                    torch.stack(
                        [mask for mask in padding_masks if mask is not None], dim=0
                    ).to(self.device, non_blocking=self.device.type == "cuda")
                    if padding_masks and all(mask is not None for mask in padding_masks)
                    else None
                ),
            )
        if not isinstance(detections, list) or len(detections) != len(images):
            raise RuntimeError(
                "DEIM detector returned an invalid batch: "
                f"expected={len(images)}, actual="
                f"{len(detections) if isinstance(detections, list) else type(detections).__name__}"
            )
        return [
            self._serialize_detections(
                row,
                input_hw=input_hw,
                original_hw=original_hw,
                geometry=geometry,
            )
            for row, (input_hw, original_hw, geometry) in zip(
                detections,
                geometries,
                strict=True,
            )
        ]

    @torch.inference_mode()
    def predict(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        return self.predict_batch([image])[0]
