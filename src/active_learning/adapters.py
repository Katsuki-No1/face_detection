from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.config_utils import load_yaml_config
from src.models import build_detector_from_config

from .config import ModelConfig
from .types import Prediction

LOGGER = logging.getLogger(__name__)


def restore_bbox_to_original(
    bbox_xyxy: Sequence[float],
    *,
    input_width: int,
    input_height: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    scale_x = original_width / float(max(1, input_width))
    scale_y = original_height / float(max(1, input_height))
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return (
        max(0.0, min(float(original_width), x1 * scale_x)),
        max(0.0, min(float(original_height), y1 * scale_y)),
        max(0.0, min(float(original_width), x2 * scale_x)),
        max(0.0, min(float(original_height), y2 * scale_y)),
    )


class ModelAdapter(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model_name = config.name
        self.inference_seconds = 0.0
        self.frames_processed = 0

    @abstractmethod
    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        video_id: str,
        timestamps: Sequence[float],
        return_embeddings: bool = False,
    ) -> tuple[list[list[Prediction]], np.ndarray | None]:
        raise NotImplementedError

    @property
    def effective_fps(self) -> float:
        return self.frames_processed / self.inference_seconds if self.inference_seconds > 0 else 0.0

    def reset(self) -> None:
        """Reset optional temporal/tracker state at an individual clip boundary.

        The three detector adapters are stateless, but this hook keeps clip
        processing safe for future tracker-backed adapters.
        """

        return None


class YoloAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(config.weights.as_posix())

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        video_id: str,
        timestamps: Sequence[float],
        return_embeddings: bool = False,
    ) -> tuple[list[list[Prediction]], np.ndarray | None]:
        started = time.perf_counter()
        results = self.model.predict(
            source=list(frames_bgr),
            imgsz=self.config.input_size,
            conf=self.config.confidence_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            batch=len(frames_bgr),
            verbose=False,
        )
        output: list[list[Prediction]] = []
        for frame, timestamp, result in zip(frames_bgr, timestamps, results, strict=True):
            height, width = frame.shape[:2]
            rows: list[Prediction] = []
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidence = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, label in zip(xyxy, confidence, classes, strict=True):
                    if label != self.config.source_class_id or float(score) < self.config.confidence_threshold:
                        continue
                    rows.append(Prediction(
                        model_name=self.model_name,
                        video_id=video_id,
                        timestamp_sec=float(timestamp),
                        class_id=self.config.class_id,
                        confidence=float(score),
                        bbox_xyxy_original=tuple(float(v) for v in box),
                        input_width=self.config.input_size,
                        input_height=self.config.input_size,
                        original_width=width,
                        original_height=height,
                    ))
            output.append(rows)
        self.inference_seconds += time.perf_counter() - started
        self.frames_processed += len(frames_bgr)
        return output, None


class RTMDetAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.config is None:
            raise ValueError("RTMDet adapter requires a model config")
        from mmdet.apis import init_detector  # type: ignore

        # MMEngine 0.x does not yet pass weights_only=False explicitly.  This
        # checkpoint was produced locally by the configured training run and
        # contains MMEngine HistoryBuffer metadata, so scope PyTorch's
        # compatibility switch to this trusted local load only.
        old_force_load = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            self.model = init_detector(config.config.as_posix(), config.weights.as_posix(), device=config.device)
        finally:
            if old_force_load is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_force_load

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        video_id: str,
        timestamps: Sequence[float],
        return_embeddings: bool = False,
    ) -> tuple[list[list[Prediction]], np.ndarray | None]:
        from mmdet.apis import inference_detector  # type: ignore

        started = time.perf_counter()
        results = inference_detector(self.model, list(frames_bgr))
        if not isinstance(results, list):
            results = [results]
        output: list[list[Prediction]] = []
        for frame, timestamp, result in zip(frames_bgr, timestamps, results, strict=True):
            height, width = frame.shape[:2]
            instances = result.pred_instances
            boxes = instances.bboxes.detach().cpu().numpy()
            scores = instances.scores.detach().cpu().numpy()
            labels = instances.labels.detach().cpu().numpy().astype(int)
            order = np.argsort(scores)[::-1][: self.config.max_detections]
            rows = [
                Prediction(
                    model_name=self.model_name,
                    video_id=video_id,
                    timestamp_sec=float(timestamp),
                    class_id=self.config.class_id,
                    confidence=float(scores[index]),
                    bbox_xyxy_original=tuple(float(v) for v in boxes[index]),
                    input_width=self.config.input_size,
                    input_height=self.config.input_size,
                    original_width=width,
                    original_height=height,
                )
                for index in order
                if int(labels[index]) == self.config.source_class_id and float(scores[index]) >= self.config.confidence_threshold
            ]
            output.append(rows)
        self.inference_seconds += time.perf_counter() - started
        self.frames_processed += len(frames_bgr)
        return output, None


class DinoDeimAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.config is None:
            raise ValueError("DINOv3+DEIMv2 adapter requires a model config")
        import torch

        self.torch = torch
        self.device = torch.device(config.device)
        detector_config = load_yaml_config(config.config)
        self.model = build_detector_from_config(detector_config).to(self.device)
        checkpoint = torch.load(config.weights, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_key = config.checkpoint_state
            if state_key == "auto":
                state_key = "ema" if isinstance(checkpoint.get("ema"), dict) else "model"
            if state_key not in checkpoint:
                raise RuntimeError(
                    f"DINOv3+DEIMv2 checkpoint has no {state_key!r} state: {config.weights}"
                )
            state = checkpoint[state_key]
        else:
            if config.checkpoint_state == "ema":
                raise RuntimeError(f"EMA state was requested but checkpoint is a plain state dict: {config.weights}")
            state = checkpoint
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"DINOv3+DEIMv2 checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.model.eval()
        if hasattr(self.model, "score_threshold"):
            self.model.score_threshold = config.confidence_threshold
        if hasattr(self.model, "detections_per_image"):
            self.model.detections_per_image = config.max_detections

    def _tensor(self, frames_bgr: Sequence[np.ndarray], input_size: int | None = None) -> Any:
        size = int(input_size or self.config.input_size)
        tensors = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            tensors.append(self.torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0))
        return self.torch.stack(tensors).to(self.device, non_blocking=True)

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        video_id: str,
        timestamps: Sequence[float],
        return_embeddings: bool = False,
    ) -> tuple[list[list[Prediction]], np.ndarray | None]:
        torch = self.torch
        images = self._tensor(frames_bgr)
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            features = self.model.backbone(images)
            encoded = self.model.encoder(list(features))
            raw = self.model.decoder(encoded, None)
            input_size = (self.config.input_size, self.config.input_size)
            sizes = torch.tensor(
                [[input_size[0], input_size[1]] for _ in frames_bgr],
                dtype=images.dtype,
                device=images.device,
            )
            detections = self.model.postprocessor(raw, sizes)
            detections = [self.model._clip_and_filter(row, (input_size[1], input_size[0])) for row in detections]
            embeddings_tensor = torch.cat([feature.mean(dim=(2, 3)) for feature in features], dim=1) if return_embeddings else None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.inference_seconds += time.perf_counter() - started
        self.frames_processed += len(frames_bgr)
        output: list[list[Prediction]] = []
        for frame, timestamp, detection in zip(frames_bgr, timestamps, detections, strict=True):
            height, width = frame.shape[:2]
            boxes = detection["boxes"].detach().float().cpu().numpy()
            scores = detection["scores"].detach().float().cpu().numpy()
            labels = detection["labels"].detach().long().cpu().numpy()
            rows: list[Prediction] = []
            for box, score, label in zip(boxes, scores, labels, strict=True):
                if int(label) != self.config.source_class_id or float(score) < self.config.confidence_threshold:
                    continue
                restored = restore_bbox_to_original(
                    box,
                    input_width=self.config.input_size,
                    input_height=self.config.input_size,
                    original_width=width,
                    original_height=height,
                )
                rows.append(Prediction(
                    model_name=self.model_name,
                    video_id=video_id,
                    timestamp_sec=float(timestamp),
                    class_id=self.config.class_id,
                    confidence=float(score),
                    bbox_xyxy_original=restored,
                    input_width=self.config.input_size,
                    input_height=self.config.input_size,
                    original_width=width,
                    original_height=height,
                ))
            output.append(rows)
        embeddings = embeddings_tensor.detach().float().cpu().numpy() if embeddings_tensor is not None else None
        return output, embeddings

    def embed_batch(self, frames_bgr: Sequence[np.ndarray], *, input_size: int | None = None) -> np.ndarray:
        """Return pooled frozen-backbone features without running the detector heads."""

        images = self._tensor(frames_bgr, input_size=input_size)
        started = time.perf_counter()
        with self.torch.inference_mode(), self.torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            features = self.model.backbone(images)
            embeddings = self.torch.cat([feature.mean(dim=(2, 3)) for feature in features], dim=1)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        self.inference_seconds += time.perf_counter() - started
        self.frames_processed += len(frames_bgr)
        return embeddings.detach().float().cpu().numpy()


def create_adapter(config: ModelConfig) -> ModelAdapter:
    adapter_type = config.type.lower()
    LOGGER.info("Loading model adapter %s (%s)", config.name, adapter_type)
    if adapter_type in {"yolo", "ultralytics", "yolo11", "yolo11s", "yolo26", "yolo26m"}:
        return YoloAdapter(config)
    if adapter_type in {"rtmdet", "mmdet"}:
        return RTMDetAdapter(config)
    if adapter_type in {"dinov3_deimv2", "dino_deimv2", "deimv2"}:
        return DinoDeimAdapter(config)
    raise ValueError(f"Unsupported model adapter type: {config.type}")
