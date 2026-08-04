from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .rotations import geodesic_rotation_distance, project_to_so3

POSE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PseudoPoseFilterConfig:
    minimum_confidence: float = 0.45
    maximum_uncertainty: float = 1.25
    maximum_jitter_geodesic_degrees: float = 12.0
    maximum_flip_geodesic_degrees: float = 12.0
    maximum_temporal_geodesic_degrees: float = 18.0
    maximum_second_teacher_geodesic_degrees: float = 18.0
    minimum_consistency_checks: int = 2


@dataclass
class PseudoPoseRecord:
    image_id: str
    instance_id: str
    split: str
    source_video_id: str
    rotation_matrix: list[list[float]]
    teacher_model: str
    teacher_checkpoint_id: str
    teacher_checkpoint_sha256: str
    confidence: float
    uncertainty: float
    crop_jitter_consistency_radians: float
    horizontal_flip_consistency_radians: float
    temporal_consistency_radians: float | None
    second_teacher_consistency_radians: float | None
    accepted: bool
    rejection_reasons: list[str]


def _degrees(value: float) -> float:
    return float(value) * 180.0 / math.pi


def filter_pseudo_pose(
    *,
    confidence: float,
    uncertainty: float,
    crop_jitter_consistency_radians: float,
    horizontal_flip_consistency_radians: float,
    temporal_consistency_radians: float | None = None,
    second_teacher_consistency_radians: float | None = None,
    config: PseudoPoseFilterConfig | None = None,
) -> tuple[bool, list[str]]:
    """Apply multi-signal pseudo-label filtering.

    Crop-jitter and horizontal-flip consistency are always required; optional
    temporal and second-teacher checks participate when supplied.
    """

    cfg = config or PseudoPoseFilterConfig()
    reasons = []
    if not math.isfinite(confidence) or confidence < cfg.minimum_confidence:
        reasons.append("teacher_confidence")
    if not math.isfinite(uncertainty) or uncertainty > cfg.maximum_uncertainty:
        reasons.append("teacher_uncertainty")
    checks = [
        (
            "crop_jitter_consistency",
            crop_jitter_consistency_radians,
            cfg.maximum_jitter_geodesic_degrees,
        ),
        (
            "horizontal_flip_consistency",
            horizontal_flip_consistency_radians,
            cfg.maximum_flip_geodesic_degrees,
        ),
    ]
    if temporal_consistency_radians is not None:
        checks.append(
            (
                "temporal_consistency",
                temporal_consistency_radians,
                cfg.maximum_temporal_geodesic_degrees,
            )
        )
    if second_teacher_consistency_radians is not None:
        checks.append(
            (
                "second_teacher_consistency",
                second_teacher_consistency_radians,
                cfg.maximum_second_teacher_geodesic_degrees,
            )
        )
    passed_consistency = 0
    for name, value, maximum_degrees in checks:
        if not math.isfinite(float(value)) or _degrees(float(value)) > maximum_degrees:
            reasons.append(name)
        else:
            passed_consistency += 1
    if passed_consistency < cfg.minimum_consistency_checks:
        reasons.append("insufficient_consistency_checks")
    return not reasons, sorted(set(reasons))


def rotation_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    return geodesic_rotation_distance(
        project_to_so3(first),
        project_to_so3(second),
    )


def validate_pose_record(record: PseudoPoseRecord) -> None:
    if record.split.lower() != "train":
        raise ValueError(
            f"Pseudo-pose supervision may only be cached for train, got {record.split}"
        )
    rotation = torch.tensor(record.rotation_matrix, dtype=torch.float64)
    if rotation.shape != (3, 3) or not torch.isfinite(rotation).all():
        raise ValueError(f"Invalid rotation for {record.image_id}/{record.instance_id}")
    projected = project_to_so3(rotation)
    if float(torch.linalg.vector_norm(projected - rotation)) > 1.0e-3:
        raise ValueError(
            f"Non-orthonormal rotation for {record.image_id}/{record.instance_id}"
        )
    if bool(record.accepted) == bool(record.rejection_reasons):
        raise ValueError(
            f"Inconsistent acceptance/reasons for {record.image_id}/{record.instance_id}"
        )


def pose_cache_payload(
    records: list[PseudoPoseRecord],
    *,
    teacher: dict[str, Any],
    filter_config: PseudoPoseFilterConfig,
    seed: int,
) -> dict[str, Any]:
    for record in records:
        validate_pose_record(record)
    ordered = sorted(
        records,
        key=lambda item: (
            item.source_video_id,
            item.image_id,
            item.instance_id,
        ),
    )
    return {
        "schema_version": POSE_CACHE_SCHEMA_VERSION,
        "split": "train",
        "seed": int(seed),
        "teacher": dict(teacher),
        "filter": asdict(filter_config),
        "record_count": len(ordered),
        "accepted_count": sum(bool(item.accepted) for item in ordered),
        "records": [asdict(item) for item in ordered],
    }


def deterministic_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_pose_cache(path: str | Path, payload: dict[str, Any]) -> str:
    if int(payload.get("schema_version", 0)) != POSE_CACHE_SCHEMA_VERSION:
        raise ValueError("Invalid pose cache schema_version")
    if str(payload.get("split", "")).lower() != "train":
        raise ValueError("Refusing to write non-train pseudo-pose cache")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = deterministic_json_bytes(payload)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return hashlib.sha256(content).hexdigest()


def load_pose_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != POSE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pseudo-pose cache: {source}")
    if str(payload.get("split", "")).lower() != "train":
        raise ValueError(f"Pseudo-pose cache is not train-only: {source}")
    for row in payload.get("records", []):
        validate_pose_record(PseudoPoseRecord(**row))
    return payload


__all__ = [
    "POSE_CACHE_SCHEMA_VERSION",
    "PseudoPoseFilterConfig",
    "PseudoPoseRecord",
    "deterministic_json_bytes",
    "filter_pseudo_pose",
    "load_pose_cache",
    "pose_cache_payload",
    "rotation_consistency",
    "validate_pose_record",
    "write_pose_cache",
]
