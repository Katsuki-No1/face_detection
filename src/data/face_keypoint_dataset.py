from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .letterbox import letterbox_image, transform_target


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_KEYPOINT_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
YOLO_KEYPOINT_NAMES = ("e1", "e2", "n", "m1", "m2")


def ellipse_to_params(ellipse: dict[str, Any]) -> list[float]:
    """Return [cx, cy, rx, ry, theta_rad] from CVAT-like ellipse metadata."""

    if {"xtl", "ytl", "xbr", "ybr"}.issubset(ellipse):
        x1 = float(ellipse["xtl"])
        y1 = float(ellipse["ytl"])
        x2 = float(ellipse["xbr"])
        y2 = float(ellipse["ybr"])
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        rx = abs(x2 - x1) * 0.5
        ry = abs(y2 - y1) * 0.5
    else:
        cx = float(ellipse.get("cx", ellipse.get("center", {}).get("x", 0.0)))
        cy = float(ellipse.get("cy", ellipse.get("center", {}).get("y", 0.0)))
        rx = float(ellipse.get("rx", ellipse.get("radius", {}).get("x", 0.0)))
        ry = float(ellipse.get("ry", ellipse.get("radius", {}).get("y", 0.0)))

    if "theta" in ellipse:
        theta = float(ellipse["theta"])
    else:
        theta = math.radians(float(ellipse.get("rotation", ellipse.get("angle", 0.0))))
    return [cx, cy, max(0.0, rx), max(0.0, ry), theta]


def ellipse_to_xyxy(ellipse: dict[str, Any]) -> list[float]:
    cx, cy, rx, ry, theta = ellipse_to_params(ellipse)
    return ellipse_params_to_xyxy([cx, cy, rx, ry, theta])


def ellipse_params_to_xyxy(ellipse: list[float] | tuple[float, float, float, float, float]) -> list[float]:
    cx, cy, rx, ry, theta = [float(value) for value in ellipse]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    half_width = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
    half_height = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
    return [cx - half_width, cy - half_height, cx + half_width, cy + half_height]


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, theta: float, *, steps: int = 96) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    xs = rx * np.cos(angles)
    ys = ry * np.sin(angles)
    return np.stack([cx + xs * cos_theta - ys * sin_theta, cy + xs * sin_theta + ys * cos_theta], axis=1)


def _ellipse_from_points(points: np.ndarray) -> list[float]:
    center = points.mean(axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(1, points.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 1e-9, None)
    vectors = vectors[:, order]
    radii = np.sqrt(values * 2.0)
    theta = math.atan2(float(vectors[1, 0]), float(vectors[0, 0]))
    return [float(center[0]), float(center[1]), float(radii[0]), float(radii[1]), theta]


def _resize_ellipse_params(ellipse: list[float], scale_x: float, scale_y: float) -> list[float]:
    cx, cy, rx, ry, theta = [float(value) for value in ellipse]
    points = _ellipse_points(cx, cy, rx, ry, theta)
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    return _ellipse_from_points(points)


def clip_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x1 = max(0.0, min(float(width), float(box[0])))
    y1 = max(0.0, min(float(height), float(box[1])))
    x2 = max(0.0, min(float(width), float(box[2])))
    y2 = max(0.0, min(float(height), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_dict_to_xyxy(bbox: dict[str, Any]) -> list[float]:
    if {"xtl", "ytl", "xbr", "ybr"}.issubset(bbox):
        return [float(bbox["xtl"]), float(bbox["ytl"]), float(bbox["xbr"]), float(bbox["ybr"])]
    if {"x", "y", "w", "h"}.issubset(bbox):
        x = float(bbox["x"])
        y = float(bbox["y"])
        return [x, y, x + float(bbox["w"]), y + float(bbox["h"])]
    if {"x", "y", "width", "height"}.issubset(bbox):
        x = float(bbox["x"])
        y = float(bbox["y"])
        return [x, y, x + float(bbox["width"]), y + float(bbox["height"])]
    raise ValueError(f"Unsupported bbox dict keys: {sorted(bbox.keys())}")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_image_and_target(
    image: Image.Image,
    target: dict[str, Any],
    image_size: int | tuple[int, int] | None,
    *,
    resize_mode: str = "stretch",
    pad_color: int | tuple[int, int, int] | list[int] | None = None,
    size_divisor: int = 32,
) -> tuple[Image.Image, dict[str, Any]]:
    if image_size is None:
        return image, target
    resize_mode = str(resize_mode).lower()
    if resize_mode == "letterbox":
        canvas, geometry, padding_mask = letterbox_image(
            image,
            image_size,
            pad_color=pad_color,
            divisor=size_divisor,
        )
        return canvas, transform_target(target, geometry, padding_mask=padding_mask)
    if resize_mode != "stretch":
        raise ValueError(
            f"resize_mode must be 'stretch' or 'letterbox', got {resize_mode!r}"
        )
    if isinstance(image_size, int):
        new_h = new_w = int(image_size)
    else:
        new_h, new_w = int(image_size[0]), int(image_size[1])
    old_w, old_h = image.size
    image = image.resize((new_w, new_h), Image.BILINEAR)
    scale_x = new_w / float(old_w)
    scale_y = new_h / float(old_h)
    updated = dict(target)
    boxes = updated["boxes"].clone()
    keypoints = updated["keypoints"].clone()
    if keypoints.numel() > 0:
        keypoints[..., 0] *= scale_x
        keypoints[..., 1] *= scale_y
    ellipses = updated.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        resized_ellipses = [_resize_ellipse_params(item, scale_x, scale_y) for item in ellipses.detach().cpu().tolist()]
        ellipses = torch.as_tensor(resized_ellipses, dtype=ellipses.dtype, device=ellipses.device)
        resized_boxes = []
        for item in resized_ellipses:
            clipped = clip_xyxy(ellipse_params_to_xyxy(item), new_w, new_h)
            if clipped is None:
                clipped = [0.0, 0.0, 0.0, 0.0]
            resized_boxes.append(clipped)
        boxes = torch.as_tensor(
            resized_boxes,
            dtype=boxes.dtype,
            device=boxes.device,
        ).reshape(-1, 4)
        updated["ellipses"] = ellipses
    elif boxes.numel() > 0:
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y
    updated["boxes"] = boxes
    updated["keypoints"] = keypoints
    updated["orig_size"] = (old_h, old_w)
    updated["size"] = (new_h, new_w)
    return image, updated


def _tensor_target(
    *,
    boxes: list[list[float]],
    ellipses: list[list[float]] | None = None,
    labels: list[int] | None = None,
    keypoints: list[list[list[float]]],
    keypoint_mask: list[list[bool]],
    keypoint_visibility: list[list[int]] | None = None,
    face_visibility: list[int] | None = None,
    image_id: str,
    height: int,
    width: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if ellipses is None:
        ellipse_values = []
        for box in box_tensor.tolist():
            x1, y1, x2, y2 = box
            ellipse_values.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1) * 0.5, (y2 - y1) * 0.5, 0.0])
    else:
        ellipse_values = ellipses
    ellipse_tensor = torch.as_tensor(ellipse_values, dtype=torch.float32).reshape(box_tensor.shape[0], 5)
    kpt_tensor = torch.as_tensor(keypoints, dtype=torch.float32).reshape(box_tensor.shape[0], len(FACE_KEYPOINT_NAMES), 2)
    mask_tensor = torch.as_tensor(keypoint_mask, dtype=torch.bool).reshape(box_tensor.shape[0], len(FACE_KEYPOINT_NAMES))
    if keypoint_visibility is None:
        keypoint_visibility_tensor = mask_tensor.to(dtype=torch.long) * 2
    else:
        keypoint_visibility_tensor = torch.as_tensor(
            keypoint_visibility, dtype=torch.long
        ).reshape(box_tensor.shape[0], len(FACE_KEYPOINT_NAMES))
        if not bool(((0 <= keypoint_visibility_tensor) & (keypoint_visibility_tensor <= 2)).all()):
            raise ValueError("keypoint_visibility values must use COCO states 0, 1, or 2")
    if face_visibility is None:
        face_visibility_tensor = torch.full((box_tensor.shape[0],), 2, dtype=torch.long)
    else:
        face_visibility_tensor = torch.as_tensor(face_visibility, dtype=torch.long).reshape(-1)
        if face_visibility_tensor.numel() != box_tensor.shape[0]:
            raise ValueError(
                "face_visibility length must match boxes: "
                f"visibility={face_visibility_tensor.numel()}, boxes={box_tensor.shape[0]}"
            )
    if labels is None:
        label_tensor = torch.ones((box_tensor.shape[0],), dtype=torch.long)
    else:
        label_tensor = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        if label_tensor.numel() != box_tensor.shape[0]:
            raise ValueError(f"labels length must match boxes: labels={label_tensor.numel()}, boxes={box_tensor.shape[0]}")
    return {
        "boxes": box_tensor,
        "ellipses": ellipse_tensor,
        "labels": label_tensor,
        "keypoints": kpt_tensor,
        "keypoint_mask": mask_tensor,
        "keypoint_visibility": keypoint_visibility_tensor,
        "face_visibility": face_visibility_tensor,
        "image_id": str(image_id),
        "orig_size": (int(height), int(width)),
        "size": (int(height), int(width)),
        "metadata": metadata or {},
    }


def _empty_target(image_id: str, height: int, width: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return _tensor_target(
        boxes=[],
        ellipses=[],
        keypoints=[],
        keypoint_mask=[],
        image_id=image_id,
        height=height,
        width=width,
        metadata=metadata,
    )


def _load_metadata_index(data_root: Path, metadata_path: str | Path | None) -> dict[str, dict[str, Any]]:
    candidates: list[Path] = []
    if metadata_path:
        candidates.append(Path(metadata_path))
    candidates.append(data_root / "metadata" / "conversion_metadata.json")
    index: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            keys = set()
            for field in ("image_name", "output_image", "image_path", "image"):
                value = record.get(field)
                if isinstance(value, str) and value:
                    keys.add(value)
                    keys.add(Path(value).name)
                    keys.add(Path(value).stem)
            for key in keys:
                index[key] = record
        break
    return index


def _metadata_for_image(index: dict[str, dict[str, Any]], image_path: Path) -> dict[str, Any] | None:
    return index.get(str(image_path)) or index.get(image_path.name) or index.get(image_path.stem)


def _parse_yolo_label(
    label_path: Path,
    width: int,
    height: int,
) -> tuple[
    list[list[float]],
    list[list[list[float]]],
    list[list[bool]],
    list[list[float]],
    list[list[int]],
]:
    boxes: list[list[float]] = []
    keypoints: list[list[list[float]]] = []
    keypoint_mask: list[list[bool]] = []
    ellipses: list[list[float]] = []
    keypoint_visibility: list[list[int]] = []
    if not label_path.exists():
        return boxes, keypoints, keypoint_mask, ellipses, keypoint_visibility
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 5:
            continue
        cx, cy, bw, bh = values[1:5]
        box_w = bw * width
        box_h = bh * height
        x = cx * width
        y = cy * height
        clipped = clip_xyxy([x - box_w / 2.0, y - box_h / 2.0, x + box_w / 2.0, y + box_h / 2.0], width, height)
        if clipped is None:
            continue
        boxes.append(clipped)
        ellipses.append([x, y, box_w * 0.5, box_h * 0.5, 0.0])
        raw_keypoints = values[5:]
        sample_keypoints: list[list[float]] = []
        sample_mask: list[bool] = []
        sample_visibility: list[int] = []
        for index in range(len(FACE_KEYPOINT_NAMES)):
            offset = index * 3
            if offset + 2 >= len(raw_keypoints) or raw_keypoints[offset + 2] <= 0:
                sample_keypoints.append([0.0, 0.0])
                sample_mask.append(False)
                sample_visibility.append(0)
                continue
            sample_keypoints.append([raw_keypoints[offset] * width, raw_keypoints[offset + 1] * height])
            sample_mask.append(True)
            sample_visibility.append(min(2, max(1, int(round(raw_keypoints[offset + 2])))))
        keypoints.append(sample_keypoints)
        keypoint_mask.append(sample_mask)
        keypoint_visibility.append(sample_visibility)
    return boxes, keypoints, keypoint_mask, ellipses, keypoint_visibility


def _record_image_path(record: dict[str, Any], image_root: Path | None) -> Path:
    if record.get("image_path"):
        path = Path(record["image_path"])
        if path.is_file():
            return path
    if image_root is not None:
        return image_root / str(record.get("image_name", ""))
    return Path(str(record.get("image_path", record.get("image_name", ""))))


def _object_keypoints_from_metadata(
    meta_object: dict[str, Any] | None,
) -> tuple[list[list[float]], list[bool], list[int]]:
    sample_keypoints = [[0.0, 0.0] for _ in FACE_KEYPOINT_NAMES]
    sample_mask = [False for _ in FACE_KEYPOINT_NAMES]
    sample_visibility = [0 for _ in FACE_KEYPOINT_NAMES]
    if not meta_object:
        return sample_keypoints, sample_mask, sample_visibility
    source_names = YOLO_KEYPOINT_NAMES
    points = meta_object.get("keypoints", {})
    for target_index, source_name in enumerate(source_names):
        point = points.get(source_name) or points.get(FACE_KEYPOINT_NAMES[target_index])
        if not point or int(point.get("v", 0)) <= 0:
            continue
        sample_keypoints[target_index] = [float(point.get("x", 0.0)), float(point.get("y", 0.0))]
        sample_mask[target_index] = True
        sample_visibility[target_index] = min(2, max(1, int(point.get("v", 2))))
    return sample_keypoints, sample_mask, sample_visibility


def _metadata_objects_to_targets(
    metadata: dict[str, Any] | None,
    width: int,
    height: int,
) -> tuple[
    list[list[float]],
    list[list[list[float]]],
    list[list[bool]],
    list[list[float]],
    list[list[int]],
    list[int],
] | None:
    if not metadata:
        return None
    objects = metadata.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return None
    boxes: list[list[float]] = []
    keypoints: list[list[list[float]]] = []
    keypoint_mask: list[list[bool]] = []
    ellipses: list[list[float]] = []
    keypoint_visibility: list[list[int]] = []
    face_visibility: list[int] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        ellipse_meta = item.get("cvat_ellipse") or item.get("face", {}).get("cvat_ellipse")
        bbox = item.get("training_bbox") or item.get("raw_bbox") or item.get("bbox") or item.get("face", {}).get("bbox")
        if ellipse_meta:
            raw_box = ellipse_to_xyxy(ellipse_meta)
            ellipse = ellipse_to_params(ellipse_meta)
        elif isinstance(bbox, dict):
            raw_box = _bbox_dict_to_xyxy(bbox)
            x1, y1, x2, y2 = raw_box
            ellipse = [(x1 + x2) * 0.5, (y1 + y2) * 0.5, abs(x2 - x1) * 0.5, abs(y2 - y1) * 0.5, 0.0]
        else:
            continue
        clipped = clip_xyxy(raw_box, width, height)
        if clipped is None:
            continue
        sample_keypoints, sample_mask, sample_visibility = _object_keypoints_from_metadata(item)
        boxes.append(clipped)
        ellipses.append(ellipse)
        keypoints.append(sample_keypoints)
        keypoint_mask.append(sample_mask)
        keypoint_visibility.append(sample_visibility)
        face_meta = item.get("face", {}) if isinstance(item.get("face"), dict) else {}
        face_visibility.append(1 if bool(item.get("occluded", face_meta.get("occluded", False))) else 2)
    if not boxes:
        return None
    return boxes, keypoints, keypoint_mask, ellipses, keypoint_visibility, face_visibility


class FaceKeypointDataset(Dataset):
    """Face bbox + 5-keypoint dataset over YOLO Pose, COCO-like, or CVAT JSON."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        split: str = "train",
        annotation_path: str | Path | None = None,
        annotation_format: str = "auto",
        image_root: str | Path | None = None,
        metadata_path: str | Path | None = None,
        image_size: int | tuple[int, int] | None = None,
        resize_mode: str = "stretch",
        pad_color: int | tuple[int, int, int] | list[int] | None = None,
        size_divisor: int = 32,
        include_negative: bool = True,
        category_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.split = split
        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.annotation_format = annotation_format
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.resize_mode = str(resize_mode).lower()
        self.pad_color = pad_color
        self.size_divisor = int(size_divisor)
        self.include_negative = bool(include_negative)
        self.category_ids = {int(value) for value in category_ids} if category_ids is not None else None
        self.samples = self._build_samples(metadata_path)

    def _build_samples(self, metadata_path: str | Path | None) -> list[dict[str, Any]]:
        if self.annotation_path is not None:
            data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
            fmt = self.annotation_format
            if fmt == "auto":
                fmt = "coco" if "images" in data and "annotations" in data else "cvat"
            if fmt == "coco":
                return self._samples_from_coco(data)
            if fmt in {"cvat", "cvat_json", "converted_cvat"}:
                return self._samples_from_cvat_records(data)
            raise ValueError(f"Unsupported annotation_format: {self.annotation_format}")

        if self.data_root is None:
            raise ValueError("data_root or annotation_path must be provided")
        return self._samples_from_yolo(metadata_path)

    def _samples_from_yolo(self, metadata_path: str | Path | None) -> list[dict[str, Any]]:
        assert self.data_root is not None
        image_dir = self.data_root / "images" / self.split
        label_dir = self.data_root / "labels" / self.split
        metadata_index = _load_metadata_index(self.data_root, metadata_path)
        samples = []
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            samples.append(
                {
                    "format": "yolo",
                    "image_path": image_path,
                    "label_path": label_dir / f"{image_path.stem}.txt",
                    "metadata": _metadata_for_image(metadata_index, image_path) or {},
                    "image_id": image_path.stem,
                }
            )
        return samples

    def _samples_from_coco(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        image_root = self.image_root or self.data_root
        images = {int(item["id"]): item for item in data.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
        for annotation in data.get("annotations", []):
            if self.category_ids is not None and int(annotation.get("category_id", -1)) not in self.category_ids:
                continue
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        samples = []
        for image_id, image in images.items():
            path = Path(image.get("file_name", ""))
            if image_root is not None and not path.is_absolute():
                path = Path(image_root) / path
            annotations = annotations_by_image.get(image_id, [])
            if annotations or self.include_negative:
                samples.append({"format": "coco", "image_path": path, "image": image, "annotations": annotations, "image_id": image_id})
        return samples

    def _samples_from_cvat_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        samples = []
        for record in data.get("records", []):
            path = _record_image_path(record, self.image_root or self.data_root)
            faces = record.get("faces", [])
            objects = record.get("objects", [])
            if faces or objects or self.include_negative:
                samples.append({"format": "cvat", "image_path": path, "record": record, "image_id": record.get("image_name", path.stem)})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size
        if sample["format"] == "yolo":
            boxes, keypoints, keypoint_mask, ellipses, keypoint_visibility = _parse_yolo_label(sample["label_path"], width, height)
            face_visibility = [2 for _ in boxes]
            metadata_targets = _metadata_objects_to_targets(sample.get("metadata"), width, height)
            if metadata_targets is not None and len(metadata_targets[0]) == len(boxes):
                boxes, keypoints, keypoint_mask, ellipses, keypoint_visibility, face_visibility = metadata_targets
            target = _tensor_target(
                boxes=boxes,
                ellipses=ellipses,
                keypoints=keypoints,
                keypoint_mask=keypoint_mask,
                keypoint_visibility=keypoint_visibility,
                face_visibility=face_visibility,
                image_id=sample["image_id"],
                height=height,
                width=width,
                metadata=sample["metadata"],
            )
        elif sample["format"] == "coco":
            target = self._target_from_coco_sample(sample, width, height)
        else:
            target = self._target_from_cvat_sample(sample, width, height)

        image, target = _resize_image_and_target(
            image,
            target,
            self.image_size,
            resize_mode=self.resize_mode,
            pad_color=self.pad_color,
            size_divisor=self.size_divisor,
        )
        return _image_to_tensor(image), target

    def _target_from_coco_sample(self, sample: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        boxes: list[list[float]] = []
        ellipses: list[list[float]] = []
        labels: list[int] = []
        keypoints: list[list[list[float]]] = []
        keypoint_mask: list[list[bool]] = []
        keypoint_visibility: list[list[int]] = []
        face_visibility: list[int] = []
        metadata = {"annotations": sample.get("annotations", [])}
        for annotation in sample.get("annotations", []):
            bbox = annotation.get("bbox")
            if not bbox:
                continue
            clipped = clip_xyxy([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width, height)
            if clipped is None:
                continue
            boxes.append(clipped)
            raw_ellipse = annotation.get("ellipse")
            if isinstance(raw_ellipse, (list, tuple)) and len(raw_ellipse) == 5:
                values = [float(value) for value in raw_ellipse]
                if not all(math.isfinite(value) for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
                    raise ValueError(
                        f"Invalid ellipse in COCO annotation id={annotation.get('id')}: {raw_ellipse}"
                    )
                ellipses.append(values)
            else:
                x1, y1, x2, y2 = clipped
                ellipses.append(
                    [(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1) * 0.5, (y2 - y1) * 0.5, 0.0]
                )
            labels.append(int(annotation.get("category_id", 1)))
            raw = annotation.get("keypoints", [])
            sample_keypoints = []
            sample_mask = []
            sample_visibility = []
            for point_index in range(len(FACE_KEYPOINT_NAMES)):
                offset = point_index * 3
                if offset + 2 >= len(raw) or raw[offset + 2] <= 0:
                    sample_keypoints.append([0.0, 0.0])
                    sample_mask.append(False)
                    sample_visibility.append(0)
                    continue
                sample_keypoints.append([float(raw[offset]), float(raw[offset + 1])])
                sample_mask.append(True)
                sample_visibility.append(min(2, max(1, int(raw[offset + 2]))))
            keypoints.append(sample_keypoints)
            keypoint_mask.append(sample_mask)
            keypoint_visibility.append(sample_visibility)
            attributes = annotation.get("attributes", {})
            face_visibility.append(
                1
                if isinstance(attributes, dict) and bool(attributes.get("face_occluded", False))
                else 2
            )
        return _tensor_target(
            boxes=boxes,
            ellipses=ellipses,
            labels=labels,
            keypoints=keypoints,
            keypoint_mask=keypoint_mask,
            keypoint_visibility=keypoint_visibility,
            face_visibility=face_visibility,
            image_id=str(sample["image_id"]),
            height=height,
            width=width,
            metadata=metadata,
        )

    def _target_from_cvat_sample(self, sample: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        record = sample["record"]
        boxes: list[list[float]] = []
        ellipses: list[list[float]] = []
        keypoints: list[list[list[float]]] = []
        keypoint_mask: list[list[bool]] = []
        keypoint_visibility: list[list[int]] = []
        face_visibility: list[int] = []
        ellipse_metadata: list[dict[str, Any] | None] = []

        objects = record.get("objects", [])
        faces = record.get("faces", [])
        source_items = objects if objects else faces
        for item_index, item in enumerate(source_items):
            if objects:
                bbox = item.get("training_bbox") or item.get("raw_bbox") or item.get("bbox")
                face_meta = item.get("face", {})
                ellipse_meta = item.get("cvat_ellipse") or face_meta.get("cvat_ellipse")
            else:
                bbox = item.get("bbox")
                ellipse_meta = item.get("cvat_ellipse")
            if not bbox and ellipse_meta:
                raw_box = ellipse_to_xyxy(ellipse_meta)
            elif isinstance(bbox, dict):
                raw_box = _bbox_dict_to_xyxy(bbox)
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                raw_box = [float(value) for value in bbox]
            else:
                continue
            clipped = clip_xyxy(raw_box, width, height)
            if clipped is None:
                continue
            boxes.append(clipped)
            if ellipse_meta:
                ellipses.append(ellipse_to_params(ellipse_meta))
            else:
                x1, y1, x2, y2 = raw_box
                ellipses.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5, abs(x2 - x1) * 0.5, abs(y2 - y1) * 0.5, 0.0])
            sample_keypoints, sample_mask, sample_visibility = _object_keypoints_from_metadata(item if objects else None)
            keypoints.append(sample_keypoints)
            keypoint_mask.append(sample_mask)
            keypoint_visibility.append(sample_visibility)
            face_meta = item.get("face", {}) if isinstance(item.get("face"), dict) else {}
            face_visibility.append(1 if bool(item.get("occluded", face_meta.get("occluded", False))) else 2)
            ellipse_metadata.append(ellipse_meta)

        if not boxes and not self.include_negative:
            return _empty_target(str(sample["image_id"]), height, width, metadata={"record": record})
        return _tensor_target(
            boxes=boxes,
            ellipses=ellipses,
            keypoints=keypoints,
            keypoint_mask=keypoint_mask,
            keypoint_visibility=keypoint_visibility,
            face_visibility=face_visibility,
            image_id=str(sample["image_id"]),
            height=height,
            width=width,
            metadata={
                "record": record,
                "ellipse_metadata": ellipse_metadata,
                "head_boxes": record.get("heads", []),
            },
        )


class SyntheticFaceKeypointDataset(Dataset):
    """Tiny in-memory dataset for smoke tests when real data is unavailable."""

    def __init__(
        self,
        *,
        length: int = 2,
        image_size: int | tuple[int, int] = 56,
        positive: bool = True,
    ) -> None:
        self.length = int(length)
        self.image_size = image_size
        self.positive = bool(positive)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(self.image_size, int):
            height = width = int(self.image_size)
        else:
            height, width = int(self.image_size[0]), int(self.image_size[1])
        image = torch.zeros((3, height, width), dtype=torch.float32)
        if not self.positive or index % 2 == 1:
            return image, _empty_target(f"synthetic_{index}", height, width)
        box = [[width * 0.3, height * 0.25, width * 0.7, height * 0.75]]
        keypoints = [
            [
                [width * 0.42, height * 0.43],
                [width * 0.58, height * 0.43],
                [width * 0.50, height * 0.53],
                [width * 0.44, height * 0.63],
                [width * 0.56, height * 0.63],
            ]
        ]
        return image, _tensor_target(
            boxes=box,
            keypoints=keypoints,
            keypoint_mask=[[True, True, True, True, True]],
            image_id=f"synthetic_{index}",
            height=height,
            width=width,
            metadata={"synthetic": True},
        )


def collate_face_keypoint_batch(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
