from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Prediction:
    model_name: str
    video_id: str
    timestamp_sec: float
    class_id: int
    confidence: float
    bbox_xyxy_original: tuple[float, float, float, float]
    input_width: int
    input_height: int
    original_width: int
    original_height: int

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["bbox_xyxy_original"] = list(self.bbox_xyxy_original)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Prediction":
        data = dict(row)
        data["bbox_xyxy_original"] = tuple(float(v) for v in data["bbox_xyxy_original"])
        return cls(**data)


@dataclass(slots=True)
class VideoInfo:
    video_id: str
    filename: str
    path: str
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    frame_count: int = 0
    file_size: int = 0
    readable: bool = False
    excluded_split: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameFeature:
    video_id: str
    filename: str
    video_path: str
    timestamp_sec: float
    frame_index: int
    original_width: int
    original_height: int
    brightness: float
    contrast: float
    blur_score: float
    perceptual_hash: str
    scene_change_score: float
    scene_id: int
    dino_embedding_index: int = -1
    pixel_sha256: str = ""
    head_detection_count: int = 0
    model_detection_counts: dict[str, int] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        counts = row.pop("model_detection_counts")
        for name, value in counts.items():
            row[f"{name}_detection_count"] = int(value)
        return row


@dataclass(slots=True)
class Candidate:
    video_id: str
    filename: str
    video_path: str
    timestamp_sec: float
    scene_id: int
    perceptual_hash: str
    frame_index: int = -1
    video_duration_sec: float = 0.0
    pixel_sha256: str = ""
    uniform_score: float = 0.0
    disagreement_score: float = 0.0
    common_difficulty_score: float = 0.0
    temporal_anomaly_score: float = 0.0
    bbox_instability_score: float = 0.0
    low_confidence_score: float = 0.0
    confidence_boundary_score: float = 0.0
    confidence_boundary_model: str = ""
    confidence_boundary_value: float = 0.0
    confidence_boundary_threshold: float = 0.0
    confidence_boundary_ratio: float = 0.0
    confidence_boundary_side: str = ""
    other_instability_score: float = 0.0
    no_detection_evidence_score: float = 0.0
    low_presence_audit_score: float = 0.0
    novelty_score: float = 0.0
    temporal_random_score: float = 0.0
    dino_cluster_id: int = -1
    dino_cluster_by_bucket: dict[str, int] = field(default_factory=dict)
    duplicate_group_id: str = ""
    track_group_id: str = ""
    track_id: str = ""
    track_event_types: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    comparison_detection_count: int = 0
    head_score: float = 0.0
    p_face_present: float = 0.0
    p_head_present: float = 0.0
    roi_source: str = ""
    roi_sources: list[str] = field(default_factory=list)
    roi_bbox: tuple[float, float, float, float] | None = None
    roi_embedding_index: int = -1
    all_low_confidence: bool = False
    all_zero_detection: bool = False
    tracked_missing_detection: bool = False
    eligibility_reason: str = ""
    clip_start_sec: float = -1.0
    clip_end_sec: float = -1.0
    usable_quality: bool = True
    model_counts: dict[str, int] = field(default_factory=dict)
    model_predictions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    selection_bucket: str = ""
    selection_score: float = 0.0
    selection_reason: str = ""
    status: str = "candidate"
    rejection_reason: str = ""
    clip_id: str = ""

    def to_flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        counts = row.pop("model_counts")
        for name, value in counts.items():
            row[f"{name}_detection_count"] = int(value)
        row["track_event_types"] = ";".join(self.track_event_types)
        row["roi_sources"] = ";".join(self.roi_sources)
        row["score_components"] = ";".join(f"{key}={value:.6f}" for key, value in sorted(self.score_components.items()))
        row["model_predictions"] = json.dumps(self.model_predictions, ensure_ascii=False, separators=(",", ":"))
        row["dino_cluster_by_bucket"] = json.dumps(self.dino_cluster_by_bucket, sort_keys=True)
        row["roi_bbox"] = "" if self.roi_bbox is None else ",".join(f"{value:.3f}" for value in self.roi_bbox)
        return row


def path_text(path: Path) -> str:
    return path.resolve().as_posix()
