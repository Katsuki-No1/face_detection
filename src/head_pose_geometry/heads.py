from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


def _xyxy_tuple(values: Sequence[Any]) -> tuple[float, float, float, float]:
    parsed = [float(value) for value in values]
    if len(parsed) != 4:
        raise ValueError(f"Expected four xyxy values, got {len(parsed)}")
    return (parsed[0], parsed[1], parsed[2], parsed[3])


@dataclass(frozen=True)
class HeadDetection:
    """Detector-independent HEAD result in original-image coordinates."""

    box_xyxy: tuple[float, float, float, float]
    confidence: float
    instance_id: str | int | None = None
    valid_image_region_xyxy: tuple[float, float, float, float] | None = None
    source: str = "predicted"

    def validated(self, image_size: tuple[int, int]) -> HeadDetection:
        height, width = int(image_size[0]), int(image_size[1])
        values = torch.tensor(self.box_xyxy, dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError(f"Non-finite HEAD box: {self.box_xyxy}")
        x1, y1, x2, y2 = (float(value) for value in values)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid HEAD box: {self.box_xyxy}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"HEAD confidence is outside [0,1]: {self.confidence}")
        region = self.valid_image_region_xyxy or (0.0, 0.0, float(width), float(height))
        rx1, ry1, rx2, ry2 = (float(value) for value in region)
        if not (0.0 <= rx1 < rx2 <= width and 0.0 <= ry1 < ry2 <= height):
            raise ValueError(
                f"Invalid valid-image region {region} for image {(height, width)}"
            )
        return HeadDetection(
            box_xyxy=(x1, y1, x2, y2),
            confidence=float(self.confidence),
            instance_id=self.instance_id,
            valid_image_region_xyxy=(rx1, ry1, rx2, ry2),
            source=str(self.source),
        )


@runtime_checkable
class ExternalHeadDetector(Protocol):
    def detect(
        self,
        image: torch.Tensor,
        *,
        image_id: str | int | None = None,
    ) -> Sequence[HeadDetection]:
        """Return HEAD detections without exposing detector internals."""


class CallableHeadDetector:
    """Adapter for existing detector callables returning tensor dictionaries."""

    def __init__(
        self,
        function: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
        *,
        source: str,
    ) -> None:
        self.function = function
        self.source = str(source)

    def detect(
        self,
        image: torch.Tensor,
        *,
        image_id: str | int | None = None,
    ) -> list[HeadDetection]:
        del image_id
        result = self.function(image)
        boxes = torch.as_tensor(result["boxes"]).detach().cpu().reshape(-1, 4)
        scores = torch.as_tensor(result["scores"]).detach().cpu().reshape(-1)
        ids = result.get("instance_ids")
        if ids is None:
            instance_ids: list[str | int | None] = [None] * int(boxes.shape[0])
        else:
            instance_ids = list(torch.as_tensor(ids).detach().cpu().tolist())
        return [
            HeadDetection(
                box_xyxy=_xyxy_tuple(box.tolist()),
                confidence=float(score),
                instance_id=instance,
                source=self.source,
            ).validated((int(image.shape[-2]), int(image.shape[-1])))
            for box, score, instance in zip(
                boxes,
                scores,
                instance_ids,
                strict=True,
            )
        ]


class CachedHeadDetector:
    """Read deterministic cached HEAD predictions.

    Schema version 1 stores ``entries[image_id]`` as a list of detection
    mappings. It intentionally has no ground-truth fallback.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported HEAD cache schema in {self.path}: "
                f"{payload.get('schema_version')}"
            )
        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            raise TypeError(f"HEAD cache has no entries mapping: {self.path}")
        self.entries = dict(entries)
        self.metadata = dict(payload.get("metadata", {}))

    def detect(
        self,
        image: torch.Tensor,
        *,
        image_id: str | int | None = None,
    ) -> list[HeadDetection]:
        if image_id is None:
            raise ValueError("CachedHeadDetector requires image_id")
        key = str(image_id)
        if key not in self.entries:
            raise KeyError(f"No cached HEAD prediction for image_id={key}")
        height, width = int(image.shape[-2]), int(image.shape[-1])
        results = []
        for row in self.entries[key]:
            if not isinstance(row, Mapping):
                raise TypeError(f"Invalid HEAD cache row for image_id={key}")
            region = row.get("valid_image_region_xyxy")
            detection = HeadDetection(
                box_xyxy=_xyxy_tuple(row["box_xyxy"]),
                confidence=float(row["confidence"]),
                instance_id=row.get("instance_id"),
                valid_image_region_xyxy=(
                    _xyxy_tuple(region) if region is not None else None
                ),
                source=str(row.get("source", "cached_predicted")),
            )
            results.append(detection.validated((height, width)))
        return results


def require_predicted_heads(
    detections: Sequence[HeadDetection],
    *,
    split: str,
) -> None:
    if str(split).lower() not in {"val", "validation", "test"}:
        return
    forbidden = {
        "ground_truth",
        "gt",
        "jittered_ground_truth",
        "jittered_gt",
    }
    bad = [item.source for item in detections if item.source.lower() in forbidden]
    if bad:
        raise ValueError(
            f"Deployment-style {split} requires predicted HEAD detections; "
            f"found sources={bad[:5]}"
        )


def serialize_head_cache(
    entries: Mapping[str, Sequence[HeadDetection]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CachedHeadDetector.SCHEMA_VERSION,
        "metadata": dict(metadata),
        "entries": {
            str(image_id): [
                {
                    "box_xyxy": list(item.box_xyxy),
                    "confidence": float(item.confidence),
                    "instance_id": item.instance_id,
                    "valid_image_region_xyxy": (
                        list(item.valid_image_region_xyxy)
                        if item.valid_image_region_xyxy is not None
                        else None
                    ),
                    "source": item.source,
                }
                for item in rows
            ]
            for image_id, rows in sorted(entries.items(), key=lambda item: str(item[0]))
        },
    }


__all__ = [
    "CachedHeadDetector",
    "CallableHeadDetector",
    "ExternalHeadDetector",
    "HeadDetection",
    "require_predicted_heads",
    "serialize_head_cache",
]
