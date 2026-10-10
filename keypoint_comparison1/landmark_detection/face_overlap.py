"""Face Overlap Robustness (FOR): target-face conditioning, version 1.

HEAD-free ordered cascade: full-image FACE/ellipse -> face-crop DINO ->
conditioned keypoint Query Refinement. IDs are image-local routing metadata,
not person identities. This module does not infer occlusion or depth order.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .geometry import CropTransform


CONTEXT_SCHEMA = "face-overlap-robustness-v1"
GEOMETRY_DIM = 10


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_context_cache(metadata: Mapping[str, Any], predictions: Mapping[int, Any],
                           *, annotation_path: Path, image_ids: set[int], query_dim: int,
                           detector_identity: Mapping[str, Any] | None = None) -> None:
    expected = {"face_context_schema": CONTEXT_SCHEMA,
                "annotation_sha256": file_sha256(annotation_path), **dict(detector_identity or {})}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"FOR cache provenance mismatch: {key}")
    if set(predictions) != image_ids:
        raise ValueError("FOR cache image inventory does not match the annotation split")
    for rows in predictions.values():
        seen = set()
        for row in rows:
            index = row.get("face_query_index")
            feature = np.asarray(row.get("face_query_features", []), dtype=np.float32)
            if (isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen
                    or feature.shape != (query_dim,) or not np.isfinite(feature).all()):
                raise ValueError("FOR cache has invalid/duplicate query IDs or invalid query features")
            seen.add(index)


def crop_face_context(detection: Mapping[str, Any], transform: CropTransform,
                      *, flipped: bool = False) -> dict[str, torch.Tensor]:
    """Use predicted geometry only; normalize by crop output_size - 1.

    Geometry: crop AABB xyxy, ellipse cx/cy/major/minor radii, cos(2 theta),
    sin(2 theta). No clipping: outside-crop geometry is useful context.
    Ellipse source angles are radians in image coordinates (x right, y down).
    """
    query = torch.as_tensor(detection["face_query_features"], dtype=torch.float32)
    query_id = detection["face_query_index"]
    if query.ndim != 1 or query.numel() == 0 or not torch.isfinite(query).all():
        raise ValueError("FOR requires a finite, nonempty face query vector")
    if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id < 0:
        raise ValueError("FOR face_query_index must be a nonnegative integer")
    box = np.asarray(detection["detector_bbox_xyxy"], dtype=np.float32)
    ellipse = np.asarray(detection["ellipse"], dtype=np.float32)
    if (box.shape != (4,) or ellipse.shape != (5,) or
            not np.isfinite(box).all() or not np.isfinite(ellipse).all() or
            np.any(box[2:] <= box[:2]) or np.any(ellipse[2:4] <= 0)):
        raise ValueError("FOR requires valid predicted bbox and ellipse")
    matrix = np.asarray(transform.matrix, dtype=np.float32).copy()
    side = float(transform.output_size - 1)
    if flipped:
        matrix[0] *= -1
        matrix[0, 2] += side
    corners = np.asarray([[box[0], box[1]], [box[2], box[1]],
                          [box[2], box[3]], [box[0], box[3]]])
    points = corners @ matrix[:, :2].T + matrix[:, 2]
    crop_box = np.concatenate([points.min(axis=0), points.max(axis=0)]) / side
    # Analytic affine ellipse transform: no clipping or minimum-radius floor.
    angle = ellipse[4]
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    axes = matrix[:, :2] @ rotation @ np.diag(ellipse[2:4])
    eigenvalues, eigenvectors = np.linalg.eigh(axes @ axes.T)
    center = matrix[:, :2] @ ellipse[:2] + matrix[:, 2]
    radii = np.sqrt(np.maximum(eigenvalues[::-1], 0))
    major_vector = eigenvectors[:, -1]
    crop_ellipse = np.concatenate([center, radii])
    theta = 2 * np.arctan2(major_vector[1], major_vector[0])
    geometry = np.concatenate([crop_box, crop_ellipse[:4] / side,
                               [np.cos(theta), np.sin(theta)]]).astype(np.float32)
    if not np.isfinite(geometry).all():
        raise ValueError("FOR crop geometry must be finite")
    return {"face_query_features": query,
            "face_geometry": torch.from_numpy(geometry),
            "face_query_index": torch.tensor(query_id, dtype=torch.long)}


class FaceOverlapConditioner(nn.Module):
    """Zero-initialized residual: enabled warm starts preserve head outputs."""
    def __init__(self, query_dim: int, dimension: int) -> None:
        super().__init__()
        if query_dim <= 0:
            raise ValueError("FOR query_dim must be positive")
        self.query_dim = query_dim
        self.query_projection = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, dimension))
        self.geometry_projection = nn.Sequential(nn.Linear(GEOMETRY_DIM, dimension), nn.GELU(),
                                                 nn.Linear(dimension, dimension))
        for projection in (self.query_projection, self.geometry_projection):
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)

    def forward(self, query: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or query.shape[1] != self.query_dim:
            raise ValueError(f"FOR expects face_query_features [B, {self.query_dim}]")
        if geometry.shape != (query.shape[0], GEOMETRY_DIM):
            raise ValueError(f"FOR expects face_geometry [B, {GEOMETRY_DIM}]")
        if not torch.isfinite(query).all() or not torch.isfinite(geometry).all():
            raise ValueError("FOR conditioning inputs must be finite")
        return self.query_projection(query) + self.geometry_projection(geometry)


def face_context_kwargs(model: nn.Module, targets: Sequence[Mapping[str, Any]],
                        device: torch.device) -> dict[str, torch.Tensor]:
    """One shared batch adapter for training and predicted-face evaluation."""
    if not getattr(model, "face_overlap_enabled", False):
        return {}
    if not targets or any("face_query_features" not in row or "face_geometry" not in row
                          or "face_query_index" not in row for row in targets):
        raise ValueError("FOR requires query-bearing predicted-face crops; prepare the FOR cache first")
    return {key: torch.stack([torch.as_tensor(row[key]) for row in targets]).to(device)
            for key in ("face_query_features", "face_geometry")}
