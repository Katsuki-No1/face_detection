from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .face_keypoint_dataset import FaceKeypointDataset, _image_to_tensor, clip_xyxy


class CocoDetectionDataset(FaceKeypointDataset):
    """COCO detection dataset for head/face bbox training.

    COCO category ids are kept as 1-based labels in the returned target. The
    detector criterion converts them to 0-based class indices internally.
    """

    def __init__(
        self,
        *,
        annotation_path: str | Path,
        image_root: str | Path | None = None,
        data_root: str | Path | None = None,
        split: str = "train",
        image_size: int | tuple[int, int] | None = None,
        include_negative: bool = True,
        category_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            annotation_path=annotation_path,
            annotation_format="coco",
            image_root=image_root,
            image_size=image_size,
            include_negative=include_negative,
            category_ids=category_ids,
        )
        # Keep this adapter's private sample records detection-only as well as
        # its returned targets. In particular, a missing/invalid ellipse must
        # never invalidate an otherwise valid rectangular HEAD bbox.
        for sample in self.samples:
            if sample.get("format") != "coco":
                continue
            sample["annotations"] = [
                {
                    key: value
                    for key, value in annotation.items()
                    if key not in {"ellipse", "keypoints", "keypoint_states"}
                }
                for annotation in sample.get("annotations", [])
            ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        if sample.get("format") != "coco":
            raise RuntimeError(
                "CocoDetectionDataset accepts COCO records only; "
                f"got format={sample.get('format')!r}"
            )

        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size
        target = self._bbox_target_from_coco_sample(sample, width, height)
        image, target = self._resize_detection_target(image, target)
        return _image_to_tensor(image), target

    @staticmethod
    def _bbox_target_from_coco_sample(
        sample: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Parse COCO xywh boxes without consulting ellipse/keypoint fields."""

        boxes: list[list[float]] = []
        labels: list[int] = []
        annotations: list[dict[str, Any]] = []
        for annotation in sample.get("annotations", []):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x, y, box_width, box_height = (float(value) for value in bbox[:4])
            if (
                not all(math.isfinite(value) for value in (x, y, box_width, box_height))
                or box_width <= 0.0
                or box_height <= 0.0
            ):
                continue
            clipped = clip_xyxy(
                [x, y, x + box_width, y + box_height],
                width,
                height,
            )
            if clipped is None:
                continue
            boxes.append(clipped)
            labels.append(int(annotation["category_id"]))
            annotations.append(
                {
                    key: annotation[key]
                    for key in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")
                    if key in annotation
                }
            )

        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": str(sample["image_id"]),
            "orig_size": (height, width),
            "size": (height, width),
            "metadata": {
                "annotations": annotations,
                "coordinate_convention": "boxes=xyxy pixels; source bbox=COCO xywh pixels",
            },
        }

    def _resize_detection_target(
        self,
        image: Image.Image,
        target: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        if self.image_size is None:
            return image, target
        if isinstance(self.image_size, int):
            new_height = new_width = int(self.image_size)
        else:
            new_height, new_width = (int(value) for value in self.image_size)

        old_width, old_height = image.size
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, 0::2] *= new_width / float(old_width)
            boxes[:, 1::2] *= new_height / float(old_height)
        target = dict(target)
        target["boxes"] = boxes
        target["orig_size"] = (old_height, old_width)
        target["size"] = (new_height, new_width)
        return image, target


__all__ = ["CocoDetectionDataset"]
