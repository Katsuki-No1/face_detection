from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def padded_roi(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    padding: float = 0.25,
    min_side: int = 16,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < min_side or box_height < min_side:
        return None
    x1 = max(0.0, x1 - box_width * padding)
    y1 = max(0.0, y1 - box_height * padding)
    x2 = min(float(width), x2 + box_width * padding)
    y2 = min(float(height), y2 + box_height * padding)
    return (x1, y1, x2, y2) if x2 - x1 >= min_side and y2 - y1 >= min_side else None


def union_bbox(
    boxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Return the clipped spatial union of all valid xyxy boxes."""

    valid: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        if x2 > x1 and y2 > y1:
            valid.append((x1, y1, x2, y2))
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def crop_roi(frame: np.ndarray, bbox: Sequence[float], output_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError("ROI is empty after clipping")
    crop = frame[iy1:iy2, ix1:ix2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


class RoiPresenceClassifier:
    def __init__(self, checkpoint_path: Path, expected_dino_weights: Path | None = None) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"ROI presence classifier is missing: {checkpoint_path}. "
                "Run scripts/train_roi_presence_classifier.py first."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.metadata = dict(checkpoint.get("metadata", {}))
        if expected_dino_weights is not None:
            expected = self.metadata.get("dino_weights_sha256")
            actual = sha256_file(expected_dino_weights)
            if expected and expected != actual:
                raise RuntimeError("ROI presence classifier was trained with different DINO weights")
        input_dim = int(checkpoint["input_dim"])
        architecture = str(checkpoint.get("architecture", "linear"))
        if architecture == "mlp":
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 2)
            )
        else:
            self.model = torch.nn.Linear(input_dim, 2)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        temperature = checkpoint.get("temperature", [1.0, 1.0])
        self.temperature = torch.as_tensor(temperature, dtype=torch.float32).clamp_min(1e-3)
        thresholds = checkpoint.get("thresholds", {})
        self.face_threshold = float(thresholds.get("face", 0.5))
        self.head_threshold = float(thresholds.get("head", 0.5))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.empty((0, 2), dtype=np.float32)
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(np.asarray(embeddings, dtype=np.float32)))
            probabilities = self.torch.sigmoid(logits / self.temperature)
        return probabilities.cpu().numpy()


def write_presence_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
