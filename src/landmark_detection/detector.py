from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import image_to_tensor, load_checkpoint, load_config, scale_detections_to_original


class DeimFaceDetector:
    """Read-only DEIMv2-L face detector used before top-down landmark crops."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        image_size: int = 960,
        score_threshold: float = 0.33,
        maximum_detections: int = 100,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.image_size = int(image_size)
        self.score_threshold = float(score_threshold)
        self.maximum_detections = int(maximum_detections)
        config = load_config(self.config_path)
        self.model = build_detector_from_config(config).to(self.device)
        missing, unexpected = load_checkpoint(self.model, self.checkpoint_path, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DEIM detector checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if hasattr(self.model, "score_threshold"):
            self.model.score_threshold = min(0.001, self.score_threshold)
        if hasattr(self.model, "postprocessor") and hasattr(self.model.postprocessor, "score_threshold"):
            self.model.postprocessor.score_threshold = min(0.001, self.score_threshold)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        tensor, input_hw, original_hw = image_to_tensor(pil.convert("RGB"), self.image_size)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            detections = self.model(tensor.to(self.device))[0]
        detections = scale_detections_to_original(detections, input_hw=input_hw, original_hw=original_hw)
        boxes = detections["boxes"].detach().cpu()
        scores = detections["scores"].detach().cpu()
        labels = detections.get("labels")
        labels = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.zeros_like(scores, dtype=torch.long)
        order = torch.argsort(scores, descending=True)
        rows: list[dict[str, Any]] = []
        for index in order.tolist():
            score = float(scores[index])
            if score < self.score_threshold:
                continue
            box = [float(value) for value in boxes[index].tolist()]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows.append({"bbox_xyxy": box, "score": score, "label": int(labels[index])})
            if len(rows) >= self.maximum_detections:
                break
        return rows
