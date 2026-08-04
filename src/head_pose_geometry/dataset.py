from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cache import load_pose_cache


def _resolve_image(root: Path, file_name: str) -> Path:
    value = Path(file_name)
    candidates = (
        (value,) if value.is_absolute() else (root / value, root / "images" / value)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {file_name!r} below {root}")


def _canonical_ellipse(values: Sequence[float]) -> np.ndarray:
    ellipse = np.asarray(values, dtype=np.float64).reshape(5).copy()
    if not np.isfinite(ellipse).all() or min(ellipse[2:4]) <= 0.0:
        raise ValueError(f"Invalid ellipse: {values}")
    if ellipse[3] > ellipse[2]:
        ellipse[2], ellipse[3] = ellipse[3], ellipse[2]
        ellipse[4] += math.pi * 0.5
    ellipse[4] = (ellipse[4] + math.pi * 0.5) % math.pi - math.pi * 0.5
    return ellipse


def _scale_ellipse(
    values: Sequence[float],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    cx, cy, major, minor, theta = _canonical_ellipse(values)
    cosine, sine = math.cos(theta), math.sin(theta)
    axes = np.asarray(((cosine, -sine), (sine, cosine)))
    covariance = axes @ np.diag((major**2, minor**2)) @ axes.T
    scaling = np.diag((scale_x, scale_y))
    covariance = scaling @ covariance @ scaling
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    radii = np.sqrt(np.clip(eigenvalues[order], 1.0e-12, None))
    direction = eigenvectors[:, order[0]]
    return _canonical_ellipse(
        (
            cx * scale_x,
            cy * scale_y,
            radii[0],
            radii[1],
            math.atan2(float(direction[1]), float(direction[0])),
        )
    )


def ellipse_aabb(ellipse: Sequence[float]) -> np.ndarray:
    cx, cy, major, minor, theta = _canonical_ellipse(ellipse)
    cosine, sine = math.cos(theta), math.sin(theta)
    radius_x = math.sqrt((major * cosine) ** 2 + (minor * sine) ** 2)
    radius_y = math.sqrt((major * sine) ** 2 + (minor * cosine) ** 2)
    return np.asarray(
        (cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y),
        dtype=np.float32,
    )


def _xywh_to_xyxy(values: Sequence[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in values)
    return np.asarray((x, y, x + width, y + height), dtype=np.float32)


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    upper = np.maximum(first[:2], second[:2])
    lower = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(0.0, lower - upper)))
    first_area = float(np.prod(np.maximum(0.0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0.0, second[2:] - second[:2])))
    return intersection / max(first_area + second_area - intersection, 1.0e-12)


def audit_source_video_splits(
    split_to_annotation: Mapping[str, str | Path],
) -> dict[str, Any]:
    videos: dict[str, set[str]] = {}
    for split, path in split_to_annotation.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        videos[str(split)] = {
            str(row["source_video_id"])
            for row in payload.get("images", [])
            if row.get("source_video_id")
        }
    overlap: dict[str, list[str]] = {}
    names = sorted(videos)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            common = sorted(videos[first] & videos[second])
            overlap[f"{first}__{second}"] = common
            if common:
                raise ValueError(
                    f"Source-video leakage between {first} and {second}: {common[:10]}"
                )
    return {
        "source_video_counts": {split: len(values) for split, values in videos.items()},
        "overlap": overlap,
    }


class HeadCentricFaceDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """HEAD annotations with attached FACE ellipse/keypoint attributes.

    The direct FACE target remains the clipped ellipse AABB.  Valid FACE rows
    retain their legacy routing HEAD boxes, while ``head_detection_boxes``
    includes every valid category-1 HEAD bbox so an internal detector can be
    trained without dropping no-face or attribute-invalid heads.
    """

    def __init__(
        self,
        *,
        annotation: str | Path,
        image_root: str | Path,
        split: str,
        image_size: int = 960,
        target_category_id: int = 2,
        predicted_head_cache: str | Path | None = None,
        pose_pseudo_cache: str | Path | None = None,
        synthetic_rotation_supervision: bool = False,
        require_predicted_heads: bool = True,
    ) -> None:
        self.annotation = Path(annotation)
        self.image_root = Path(image_root)
        self.split = str(split).lower()
        self.image_size = int(image_size)
        self.target_category_id = int(target_category_id)
        self.synthetic_rotation_supervision = bool(synthetic_rotation_supervision)
        if (
            require_predicted_heads
            and self.split in {"val", "validation", "test"}
            and predicted_head_cache is None
        ):
            raise ValueError(
                f"Deployment-style {split} requires data.predicted_head_caches."
                f"{split}; GT HEAD fallback is forbidden"
            )
        data = json.loads(self.annotation.read_text(encoding="utf-8"))
        grouped: dict[int, list[dict[str, Any]]] = {}
        all_heads_grouped: dict[int, list[dict[str, Any]]] = {}
        for row in data.get("annotations", []):
            if int(row.get("category_id", -1)) == 1:
                bbox = row.get("bbox", ())
                if (
                    len(bbox) == 4
                    and np.isfinite(np.asarray(bbox, dtype=np.float64)).all()
                    and float(bbox[2]) > 0.0
                    and float(bbox[3]) > 0.0
                ):
                    all_heads_grouped.setdefault(
                        int(row["image_id"]),
                        [],
                    ).append(row)
            if (
                int(row.get("category_id", -1)) == 1
                and bool(
                    row.get(
                        "attribute_valid",
                        self.synthetic_rotation_supervision,
                    )
                )
                and len(row.get("ellipse", ())) == 5
            ):
                grouped.setdefault(int(row["image_id"]), []).append(row)
        head_entries: dict[str, list[dict[str, Any]]] | None = None
        if predicted_head_cache is not None:
            payload = json.loads(Path(predicted_head_cache).read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != 1:
                raise ValueError("Unsupported predicted HEAD cache schema")
            head_entries = dict(payload.get("entries", {}))
        pseudo: dict[tuple[str, str], dict[str, Any]] = {}
        if pose_pseudo_cache is not None:
            if self.split != "train":
                raise ValueError("Pose pseudo-labels are train-only")
            payload = load_pose_cache(pose_pseudo_cache)
            for row in payload["records"]:
                pseudo[(str(row["image_id"]), str(row["instance_id"]))] = row
        self.pseudo = pseudo
        self.samples: list[dict[str, Any]] = []
        for image in data.get("images", []):
            image_id = int(image["id"])
            rows = sorted(
                grouped.get(image_id, []),
                key=lambda row: int(row.get("id", 0)),
            )
            head_rows = sorted(
                all_heads_grouped.get(image_id, []),
                key=lambda row: int(row.get("id", 0)),
            )
            predicted = (
                list(head_entries.get(str(image_id), []))
                if head_entries is not None
                else None
            )
            self.samples.append(
                {
                    "image": image,
                    "path": _resolve_image(
                        self.image_root,
                        str(image["file_name"]),
                    ),
                    "annotations": rows,
                    "head_annotations": head_rows,
                    "predicted_heads": predicted,
                }
            )
        if not self.samples:
            raise ValueError(f"No images in {self.annotation}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _prediction_to_target_indices(
        rows: Sequence[Mapping[str, Any]],
        predicted: Sequence[Mapping[str, Any]],
        *,
        minimum_iou: float = 0.10,
    ) -> list[int]:
        mapping = [-1] * len(predicted)
        candidates = []
        for prediction_index, detection in enumerate(predicted):
            prediction_box = np.asarray(
                detection["box_xyxy"],
                dtype=np.float32,
            )
            for target_index, row in enumerate(rows):
                overlap = _box_iou(
                    prediction_box,
                    _xywh_to_xyxy(row["bbox"]),
                )
                candidates.append((-overlap, prediction_index, target_index))
        used_predictions: set[int] = set()
        used_targets: set[int] = set()
        for negative_iou, prediction_index, target_index in sorted(candidates):
            if -negative_iou < minimum_iou:
                break
            if prediction_index in used_predictions or target_index in used_targets:
                continue
            mapping[prediction_index] = target_index
            used_predictions.add(prediction_index)
            used_targets.add(target_index)
        return mapping

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image_info = sample["image"]
        image = Image.open(sample["path"]).convert("RGB")
        original_width, original_height = image.size
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        boxes, ellipses, keypoints = [], [], []
        labeled, visible, heads, head_confidence = [], [], [], []
        rotations, pose_mask, pose_weight = [], [], []
        template_vertices, target_scales = [], []
        target_translations, target_shapes = [], []
        instance_ids: list[int] = []
        for row in sample["annotations"]:
            ellipse = _scale_ellipse(row["ellipse"], scale_x, scale_y)
            face_box = ellipse_aabb(ellipse)
            face_box[0::2] = np.clip(face_box[0::2], 0.0, self.image_size)
            face_box[1::2] = np.clip(face_box[1::2], 0.0, self.image_size)
            points = np.asarray(row.get("keypoints", ()), dtype=np.float32).reshape(
                5, 2
            )
            points[:, 0] *= scale_x / self.image_size
            points[:, 1] *= scale_y / self.image_size
            states = np.asarray(
                row.get("keypoint_states", (0, 0, 0, 0, 0)),
                dtype=np.int64,
            ).reshape(5)
            annotation_id = str(row.get("id"))
            image_id = str(image_info["id"])
            pseudo = self.pseudo.get((image_id, annotation_id))
            if pseudo is None:
                pseudo = self.pseudo.get((image_id, str(row.get("source_head_index"))))
            if self.synthetic_rotation_supervision:
                if len(row.get("rotation_matrix", ())) != 3:
                    raise ValueError("Synthetic annotation requires rotation_matrix")
                rotations.append(np.asarray(row["rotation_matrix"], dtype=np.float32))
                pose_mask.append(True)
                pose_weight.append(float(row.get("pose_weight", 1.0)))
            else:
                rotations.append(
                    np.asarray(
                        pseudo["rotation_matrix"],
                        dtype=np.float32,
                    )
                    if pseudo is not None
                    else np.eye(3, dtype=np.float32)
                )
                pose_mask.append(bool(pseudo is not None and pseudo["accepted"]))
                pose_weight.append(
                    float(pseudo["confidence"]) if pseudo is not None else 0.0
                )
            if "template_vertices" in row:
                template_vertices.append(row["template_vertices"])
            if "weak_perspective_scale" in row:
                target_scales.append(float(row["weak_perspective_scale"]))
            if "translation" in row:
                target_translations.append(row["translation"])
            if "shape_code" in row:
                target_shapes.append(row["shape_code"])
            boxes.append(face_box)
            ellipses.append(ellipse)
            keypoints.append(points)
            labeled.append(states > 0)
            visible.append(states > 1)
            instance_ids.append(int(row.get("id", len(instance_ids))))
        count = len(boxes)
        face_index_by_id = {
            int(row.get("id", -1)): face_index
            for face_index, row in enumerate(sample["annotations"])
        }
        head_detection_boxes: list[np.ndarray] = []
        head_detection_to_face_indices: list[int] = []
        for row in sample["head_annotations"]:
            head = _xywh_to_xyxy(row["bbox"])
            head[0::2] *= scale_x
            head[1::2] *= scale_y
            head[0::2] = np.clip(
                head[0::2],
                0.0,
                float(self.image_size),
            )
            head[1::2] = np.clip(
                head[1::2],
                0.0,
                float(self.image_size),
            )
            if head[2] <= head[0] or head[3] <= head[1]:
                continue
            head_detection_boxes.append(head)
            head_detection_to_face_indices.append(
                face_index_by_id.get(int(row.get("id", -1)), -1)
            )
        if sample["predicted_heads"] is None:
            for row in sample["annotations"]:
                head = _xywh_to_xyxy(row["bbox"])
                head[0::2] *= scale_x
                head[1::2] *= scale_y
                heads.append(head)
                head_confidence.append(1.0)
            geometry_target_indices = list(range(count))
        else:
            geometry_target_indices = self._prediction_to_target_indices(
                sample["annotations"],
                sample["predicted_heads"],
            )
            for prediction in sample["predicted_heads"]:
                head = np.asarray(
                    prediction["box_xyxy"],
                    dtype=np.float32,
                ).copy()
                head[0::2] *= scale_x
                head[1::2] *= scale_y
                heads.append(head)
                head_confidence.append(float(prediction["confidence"]))
        head_count = len(heads)
        target: dict[str, Any] = {
            "boxes": torch.tensor(np.asarray(boxes), dtype=torch.float32).reshape(
                -1, 4
            ),
            "ellipses": torch.tensor(np.asarray(ellipses), dtype=torch.float32).reshape(
                -1, 5
            ),
            "ellipse_mask": torch.ones(count, dtype=torch.bool),
            "labels": torch.full((count,), self.target_category_id, dtype=torch.long),
            "keypoints": torch.tensor(
                np.asarray(keypoints), dtype=torch.float32
            ).reshape(-1, 5, 2),
            "labeled": torch.tensor(np.asarray(labeled), dtype=torch.bool).reshape(
                -1, 5
            ),
            "visible": torch.tensor(np.asarray(visible), dtype=torch.bool).reshape(
                -1, 5
            ),
            "head_boxes": torch.tensor(np.asarray(heads), dtype=torch.float32).reshape(
                -1, 4
            ),
            "head_detection_boxes": torch.tensor(
                np.asarray(head_detection_boxes),
                dtype=torch.float32,
            ).reshape(-1, 4),
            "head_detection_labels": torch.ones(
                len(head_detection_boxes),
                dtype=torch.long,
            ),
            "head_detection_to_face_indices": torch.tensor(
                head_detection_to_face_indices,
                dtype=torch.long,
            ),
            "head_confidence": torch.tensor(head_confidence, dtype=torch.float32),
            "geometry_target_indices": torch.tensor(
                geometry_target_indices,
                dtype=torch.long,
            ),
            "head_sources": [
                "ground_truth"
                if sample["predicted_heads"] is None
                else "cached_predicted"
            ]
            * head_count,
            "rotation_matrix": torch.tensor(
                np.asarray(rotations), dtype=torch.float32
            ).reshape(-1, 3, 3),
            "pose_supervision_mask": torch.tensor(pose_mask, dtype=torch.bool),
            "pose_weight": torch.tensor(pose_weight, dtype=torch.float32),
            "image_id": torch.tensor(int(image_info["id"]), dtype=torch.long),
            "source_video_id": str(image_info.get("source_video_id", "")),
            "instance_ids": torch.tensor(instance_ids, dtype=torch.long),
            "sample_is_synthetic": torch.full(
                (count,),
                self.synthetic_rotation_supervision,
                dtype=torch.bool,
            ),
        }
        optional_targets = (
            (
                "template_vertices",
                template_vertices,
                (count, -1, 3),
            ),
            ("weak_perspective_scale", target_scales, (count,)),
            ("translation", target_translations, (count, 2)),
            ("shape_code", target_shapes, (count, -1)),
        )
        for name, values, shape in optional_targets:
            if values:
                if len(values) != count:
                    raise ValueError(
                        f"Synthetic field {name} must be present for every instance"
                    )
                target[name] = torch.tensor(
                    np.asarray(values),
                    dtype=torch.float32,
                ).reshape(shape)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
        return tensor, target


class InterleavedRealSyntheticDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """One epoch contains every real row plus a fixed synthetic fraction."""

    def __init__(
        self,
        real: Dataset[tuple[torch.Tensor, dict[str, Any]]],
        synthetic: Dataset[tuple[torch.Tensor, dict[str, Any]]],
        *,
        synthetic_fraction: float,
        seed: int,
    ) -> None:
        if not 0.0 < synthetic_fraction < 1.0:
            raise ValueError("synthetic_fraction must be inside (0,1)")
        if len(synthetic) == 0:
            raise ValueError("Synthetic dataset is empty")
        self.real = real
        self.synthetic = synthetic
        self.synthetic_count = max(
            1,
            round(len(real) * synthetic_fraction / (1.0 - synthetic_fraction)),
        )
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.real) + self.synthetic_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for dataset in (self.real, self.synthetic):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if index < len(self.real):
            return self.real[index]
        synthetic_index = (
            index - len(self.real) + self.seed + self.epoch * self.synthetic_count
        ) % len(self.synthetic)
        return self.synthetic[synthetic_index]


__all__ = [
    "HeadCentricFaceDataset",
    "InterleavedRealSyntheticDataset",
    "audit_source_video_splits",
    "ellipse_aabb",
]
