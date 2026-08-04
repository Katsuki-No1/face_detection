from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .strongest_types import FaceObservation, StrongestModelAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Deimv2QueryRefinementAdapter(StrongestModelAdapter):
    """DEIMv2-L ellipse detector followed by DINOv3 Query Refinement."""

    capabilities = frozenset({"face", "ellipse", "keypoints"})

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        options: Mapping[str, Any],
    ) -> None:
        import torch

        from keypoint_comparison1.landmark_detection.detector import (
            DeimFaceDetector,
        )
        from keypoint_comparison1.landmark_detection.model import (
            build_landmark_model,
        )

        self.model_id = str(model_id)
        self.device = torch.device(
            device
            if torch.cuda.is_available() or not str(device).startswith("cuda")
            else "cpu"
        )
        self.detector_config = Path(options["detector_config"]).resolve()
        self.detector_checkpoint = Path(options["detector_checkpoint"]).resolve()
        self.query_checkpoint = Path(options["query_checkpoint"]).resolve()
        self.detector_batch_size = int(options.get("detector_batch_size", 2))
        self.query_batch_size = int(options.get("query_batch_size", 32))
        self.query_input_size = int(options.get("query_input_size", 256))
        self.crop_scale = float(options.get("crop_scale", 1.25))
        if self.detector_batch_size <= 0 or self.query_batch_size <= 0:
            raise ValueError("Detector and query batch sizes must be positive")
        if self.crop_scale <= 0:
            raise ValueError("crop_scale must be positive")

        self.detector = DeimFaceDetector(
            config_path=self.detector_config,
            checkpoint_path=self.detector_checkpoint,
            device=str(self.device),
            image_size=int(options.get("detector_input_size", 960)),
            score_threshold=float(options["inference_confidence_floor"]),
            maximum_detections=int(options.get("maximum_detections", 100)),
            crop_geometry="ellipse",
        )
        checkpoint = torch.load(
            self.query_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("config"),
            dict,
        ):
            raise RuntimeError(
                f"Query checkpoint has no embedded immutable config: "
                f"{self.query_checkpoint}"
            )
        self.query_raw = dict(checkpoint["config"])
        self.query_model, _ = build_landmark_model(
            self.query_raw,
            variant="query_refinement",
            initialize_from_detector=False,
        )
        self.query_model = self.query_model.to(self.device).eval()
        state = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(state, dict):
            raise RuntimeError(
                f"Query checkpoint has no EMA/model state: {self.query_checkpoint}"
            )
        missing, unexpected = self.query_model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Query checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        embedded_input_size = int(self.query_raw["data"]["input_size"])
        if self.query_input_size != embedded_input_size:
            raise RuntimeError(
                f"Configured query_input_size={self.query_input_size} does not "
                f"match checkpoint config input_size={embedded_input_size}"
            )
        del checkpoint, state
        self.mean = tuple(float(value) for value in self.query_raw["data"]["mean"])
        self.std = tuple(float(value) for value in self.query_raw["data"]["std"])

    def _predict_keypoints(
        self,
        frames_rgb: Sequence[np.ndarray],
        detections_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[list[FaceObservation]]:
        import torch

        from keypoint_comparison1.landmark_detection.dataset import image_to_tensor
        from keypoint_comparison1.landmark_detection.geometry import (
            build_crop_transform,
            normalized_crop_to_original,
            warp_crop,
        )
        from src.data.face_crop_cache import crop_geometry_from_detection

        crops: list[torch.Tensor] = []
        metadata: list[tuple[int, Mapping[str, Any], Any]] = []
        for frame_index, (image_rgb, detections) in enumerate(
            zip(frames_rgb, detections_by_frame, strict=True)
        ):
            for detection in detections:
                crop_box, rotation = crop_geometry_from_detection(
                    detection,
                    require_ellipse=True,
                )
                transform = build_crop_transform(
                    crop_box,
                    output_size=self.query_input_size,
                    scale_factor=self.crop_scale,
                    rotation_deg=rotation,
                    warp_bbox_aspect=False,
                )
                crop = warp_crop(image_rgb, transform)
                crops.append(image_to_tensor(crop, mean=self.mean, std=self.std))
                metadata.append((frame_index, detection, transform))

        output: list[list[FaceObservation]] = [
            [] for _ in detections_by_frame
        ]
        if not crops:
            return output
        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.query_batch_size):
                batch = torch.stack(
                    crops[start : start + self.query_batch_size]
                ).to(self.device)
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.device.type == "cuda",
                ):
                    prediction = self.query_model(batch)
                coordinates = (
                    prediction["decoded_coordinates"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                point_logits = prediction.get(
                    "presence_logits",
                    prediction["visibility_logits"],
                )
                scores = (
                    point_logits.sigmoid().detach().float().cpu().numpy()
                )
                decoded.extend(zip(coordinates, scores, strict=True))

        for (frame_index, detection, transform), (
            coordinates,
            point_scores,
        ) in zip(metadata, decoded, strict=True):
            ellipse = detection.get("ellipse")
            bbox = detection.get(
                "detector_bbox_xyxy",
                detection.get("bbox_xyxy"),
            )
            if not isinstance(ellipse, (list, tuple)) or len(ellipse) != 5:
                raise RuntimeError("DEIM ellipse prediction is missing ellipse geometry")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise RuntimeError("DEIM ellipse prediction is missing a face bbox")
            points = normalized_crop_to_original(coordinates, transform)
            output[frame_index].append(
                FaceObservation(
                    confidence=float(detection["score"]),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    ellipse=tuple(float(value) for value in ellipse),
                    keypoints=tuple(
                        (float(point[0]), float(point[1])) for point in points
                    ),
                    keypoint_scores=tuple(
                        float(value) for value in point_scores
                    ),
                )
            )
        return output

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
    ) -> list[list[FaceObservation]]:
        frames_rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr
        ]
        detections: list[list[dict[str, Any]]] = []
        for start in range(0, len(frames_rgb), self.detector_batch_size):
            detections.extend(
                self.detector.predict_batch(
                    frames_rgb[start : start + self.detector_batch_size]
                )
            )
        return self._predict_keypoints(frames_rgb, detections)

    def provenance(self) -> dict[str, Any]:
        assets = {
            "detector_config": self.detector_config,
            "detector_checkpoint": self.detector_checkpoint,
            "query_checkpoint": self.query_checkpoint,
        }
        return {
            "model_id": self.model_id,
            "adapter": f"{type(self).__module__}:{type(self).__qualname__}",
            "device": str(self.device),
            "capabilities": sorted(self.capabilities),
            "assets": {
                name: {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in assets.items()
            },
            "detector_batch_size": self.detector_batch_size,
            "query_batch_size": self.query_batch_size,
            "query_input_size": self.query_input_size,
            "crop_scale": self.crop_scale,
        }
