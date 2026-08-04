from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


REQUIRED_MODEL_CAPABILITIES = frozenset({"face", "ellipse", "keypoints"})


@dataclass(slots=True)
class FaceObservation:
    """Model-independent face, ellipse, and landmark prediction."""

    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    ellipse: tuple[float, float, float, float, float]
    keypoints: tuple[tuple[float, float], ...] = ()
    keypoint_scores: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["bbox_xyxy"] = list(self.bbox_xyxy)
        row["ellipse"] = list(self.ellipse)
        row["keypoints"] = [list(point) for point in self.keypoints]
        row["keypoint_scores"] = list(self.keypoint_scores)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FaceObservation":
        return cls(
            confidence=float(row["confidence"]),
            bbox_xyxy=tuple(float(value) for value in row["bbox_xyxy"]),
            ellipse=tuple(float(value) for value in row["ellipse"]),
            keypoints=tuple(
                (float(point[0]), float(point[1]))
                for point in row.get("keypoints", ())
            ),
            keypoint_scores=tuple(
                float(value) for value in row.get("keypoint_scores", ())
            ),
        )


@dataclass(slots=True)
class InferenceFrame:
    montage_frame_id: int
    montage_timestamp_sec: float
    source_segment: int
    source_timestamp_sec: float
    source_frame_id: int
    width: int
    height: int
    scene_id: int
    pixel_sha256: str
    observations: list[FaceObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["observations"] = [item.to_dict() for item in self.observations]
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "InferenceFrame":
        values = dict(row)
        values["observations"] = [
            FaceObservation.from_dict(dict(item))
            for item in row.get("observations", ())
        ]
        return cls(**values)


@dataclass(slots=True)
class FailureCandidate:
    frame: InferenceFrame
    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    assigned_category: str = ""
    selection_score: float = 0.0
    selection_note: str = ""

    @property
    def matched_categories(self) -> list[str]:
        return sorted(name for name, score in self.scores.items() if score > 0.0)


class StrongestModelAdapter(Protocol):
    """Contract implemented by the currently selected active-learning model."""

    model_id: str
    capabilities: frozenset[str]

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
    ) -> list[list[FaceObservation]]:
        ...

    def provenance(self) -> dict[str, Any]:
        ...
