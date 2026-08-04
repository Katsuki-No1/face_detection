from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from torch import nn

from .backbone import DINOv3ViTS16PlusBackbone
from .codino import CompactCoDINO
from .geometry import postprocess_head_detections
from .sfp import SimpleFeaturePyramid
from .structures import HeadImageBatch


EXPECTED_TOTAL_PARAMETERS = 38_529_172
PYRAMID_LEVELS = ("p3", "p4", "p5", "p6")
DETECTOR_LEVELS = ("p3", "p4", "p5")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class HeadCoDINODetector(nn.Module):
    """DINOv3 ViT-S+/16 + SFP + compact one-class Co-DINO detector."""

    def __init__(
        self,
        *,
        dinov3_source_root: str | Path | None = None,
        backbone_weights: str | Path | None = None,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        category_id: int = 1,
        score_threshold: float = 0.30,
        max_detections: int = 100,
    ) -> None:
        super().__init__()
        source_root = (
            Path(dinov3_source_root)
            if dinov3_source_root is not None
            else _repo_root() / "models" / "source" / "dinov3"
        )
        self.backbone = DINOv3ViTS16PlusBackbone(
            source_root=source_root,
            weights=backbone_weights,
            freeze=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self.sfp = SimpleFeaturePyramid()
        self.query_head = CompactCoDINO()
        self.category_id = int(category_id)
        if self.category_id <= 0:
            raise ValueError(f"category_id must be a positive COCO id, got {category_id}")
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be in [0,1], got {score_threshold}")
        if self.max_detections <= 0 or self.max_detections > self.query_head.num_queries:
            raise ValueError(
                f"max_detections must be in [1,{self.query_head.num_queries}], "
                f"got {max_detections}"
            )

        count = sum(parameter.numel() for parameter in self.parameters())
        if count != EXPECTED_TOTAL_PARAMETERS:
            raise RuntimeError(
                f"Unexpected total parameter count: expected={EXPECTED_TOTAL_PARAMETERS}, "
                f"actual={count}"
            )

    @staticmethod
    def _pad_feature_level(features: Sequence[torch.Tensor]) -> torch.Tensor:
        max_height = max(feature.shape[-2] for feature in features)
        max_width = max(feature.shape[-1] for feature in features)
        padded = [
            F.pad(
                feature,
                (0, max_width - feature.shape[-1], 0, max_height - feature.shape[-2]),
            )
            for feature in features
        ]
        return torch.cat(padded, dim=0)

    def extract_pyramid(
        self,
        batch: HeadImageBatch,
    ) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
        patch_features = self.backbone(batch.image_list())
        individual_pyramids = [self.sfp(feature[None]) for feature in patch_features]
        pyramid: OrderedDict[str, torch.Tensor] = OrderedDict()
        masks: OrderedDict[str, torch.Tensor] = OrderedDict()
        for level in PYRAMID_LEVELS:
            feature = self._pad_feature_level(
                [image_pyramid[level] for image_pyramid in individual_pyramids]
            )
            pyramid[level] = feature
            masks[level] = batch.level_mask(
                feature.shape[-2:],
                stride=self.sfp.strides[level],
            )
        return pyramid, masks

    def _build_data_samples(
        self,
        batch: HeadImageBatch,
        targets: Sequence[Mapping[str, Any]],
    ) -> list[DetDataSample]:
        if len(targets) != batch.batch_size:
            raise ValueError(
                f"targets length ({len(targets)}) does not match batch ({batch.batch_size})"
            )
        samples: list[DetDataSample] = []
        device = batch.tensors.device
        for index, target in enumerate(targets):
            boxes = torch.as_tensor(
                target.get("boxes", []),
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 4)
            source_labels = torch.as_tensor(
                target.get(
                    "labels",
                    torch.full((boxes.shape[0],), self.category_id, dtype=torch.long),
                ),
                dtype=torch.long,
                device=device,
            ).reshape(-1)
            if source_labels.shape[0] != boxes.shape[0]:
                raise ValueError("Each target must contain one label per box")
            # Accept already-internal label 0 as well as the configured COCO id.
            keep = (source_labels == self.category_id) | (source_labels == 0)
            boxes = boxes[keep]
            labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
            finite = torch.isfinite(boxes).all(dim=-1)
            positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[finite & positive_area]
            labels = labels[finite & positive_area]

            height, width = batch.original_size_list[index]
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            valid_after_clip = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            instances = InstanceData(
                bboxes=boxes[valid_after_clip],
                labels=labels[valid_after_clip],
            )
            sample = DetDataSample()
            sample.gt_instances = instances
            sample.set_metainfo(
                {
                    "img_id": target.get("image_id", index),
                    "img_shape": (height, width),
                    "ori_shape": (height, width),
                    "batch_input_shape": batch.padded_size,
                    "scale_factor": (1.0, 1.0),
                }
            )
            samples.append(sample)
        return samples

    def _run_query_head(
        self,
        batch: HeadImageBatch,
        pyramid: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        samples: Sequence[DetDataSample] | None,
    ) -> dict[str, Any]:
        detector_features = tuple(pyramid[level] for level in DETECTOR_LEVELS)
        detector_masks = tuple(masks[level] for level in DETECTOR_LEVELS)
        size_groups: dict[tuple[int, int], list[int]] = {}
        for index, size in enumerate(batch.individual_padded_size_list):
            size_groups.setdefault(size, []).append(index)
        # A mixed native-resolution batch changes MS-Deformable Attention's
        # spatial_shapes. Isolate different shapes (while still batching equal
        # shapes) so an image is numerically invariant to its batch partners.
        if self.training or len(size_groups) == 1:
            return self.query_head(
                detector_features,
                detector_masks,
                batch_data_samples=samples,
            )

        grouped_outputs: list[tuple[list[int], dict[str, Any]]] = []
        for (padded_height, padded_width), indices in size_groups.items():
            index_tensor = torch.tensor(indices, device=batch.tensors.device)
            group_features = []
            group_masks = []
            for level in DETECTOR_LEVELS:
                stride = self.sfp.strides[level]
                height, width = padded_height // stride, padded_width // stride
                group_features.append(
                    pyramid[level].index_select(0, index_tensor)[..., :height, :width]
                )
                group_masks.append(
                    masks[level].index_select(0, index_tensor)[..., :height, :width]
                )
            grouped_outputs.append(
                (
                    indices,
                    self.query_head(tuple(group_features), tuple(group_masks)),
                )
            )

        def restore_batch(key: str, *, layer_first: bool = False) -> torch.Tensor:
            slots: list[torch.Tensor | None] = [None] * batch.batch_size
            for indices, output in grouped_outputs:
                value = output[key]
                batch_dimension = 1 if layer_first else 0
                for local_index, original_index in enumerate(indices):
                    slots[original_index] = value.select(batch_dimension, local_index).unsqueeze(
                        batch_dimension
                    )
            return torch.cat([value for value in slots if value is not None], dim=batch_dimension)

        result: dict[str, Any] = {
            "pred_logits": restore_batch("pred_logits"),
            "pred_boxes": restore_batch("pred_boxes"),
            "enc_outputs": {
                "pred_logits": torch.cat(
                    [
                        next(
                            output["enc_outputs"]["pred_logits"][local : local + 1]
                            for indices, output in grouped_outputs
                            for local, original in enumerate(indices)
                            if original == index
                        )
                        for index in range(batch.batch_size)
                    ]
                ),
                "pred_boxes": torch.cat(
                    [
                        next(
                            output["enc_outputs"]["pred_boxes"][local : local + 1]
                            for indices, output in grouped_outputs
                            for local, original in enumerate(indices)
                            if original == index
                        )
                        for index in range(batch.batch_size)
                    ]
                ),
            },
            "_all_logits": restore_batch("_all_logits", layer_first=True),
            "_all_boxes": restore_batch("_all_boxes", layer_first=True),
            "_hidden_states": restore_batch("_hidden_states", layer_first=True),
            "_enc_logits": restore_batch("_enc_logits"),
            "_enc_boxes": restore_batch("_enc_boxes"),
        }
        result["aux_outputs"] = [
            {
                "pred_logits": torch.cat(
                    [
                        next(
                            output["aux_outputs"][layer]["pred_logits"][local : local + 1]
                            for indices, output in grouped_outputs
                            for local, original in enumerate(indices)
                            if original == index
                        )
                        for index in range(batch.batch_size)
                    ]
                ),
                "pred_boxes": torch.cat(
                    [
                        next(
                            output["aux_outputs"][layer]["pred_boxes"][local : local + 1]
                            for indices, output in grouped_outputs
                            for local, original in enumerate(indices)
                            if original == index
                        )
                        for index in range(batch.batch_size)
                    ]
                ),
            }
            for layer in range(self.query_head.num_decoder_layers - 1)
        ]
        reference_count = len(grouped_outputs[0][1]["_references"])
        result["_references"] = [
            torch.cat(
                [
                    next(
                        output["_references"][layer][local : local + 1]
                        for indices, output in grouped_outputs
                        for local, original in enumerate(indices)
                        if original == index
                    )
                    for index in range(batch.batch_size)
                ]
            )
            for layer in range(reference_count)
        ]
        encoded = []
        target_height = max(output["encoded_p6"].shape[-2] for _, output in grouped_outputs)
        target_width = max(output["encoded_p6"].shape[-1] for _, output in grouped_outputs)
        for index in range(batch.batch_size):
            value = next(
                output["encoded_p6"][local : local + 1]
                for indices, output in grouped_outputs
                for local, original in enumerate(indices)
                if original == index
            )
            encoded.append(
                F.pad(value, (0, target_width - value.shape[-1], 0, target_height - value.shape[-2]))
            )
        result["encoded_p6"] = torch.cat(encoded)
        return result

    def forward_raw(
        self,
        images: HeadImageBatch | Sequence[torch.Tensor] | torch.Tensor,
        *,
        targets: Sequence[Mapping[str, Any]] | None = None,
        return_features: bool = False,
    ) -> dict[str, Any]:
        batch = images if isinstance(images, HeadImageBatch) else HeadImageBatch.from_images(images)
        pyramid, masks = self.extract_pyramid(batch)
        samples = self._build_data_samples(batch, targets) if targets is not None else None
        raw = self._run_query_head(batch, pyramid, masks, samples)
        if return_features:
            raw["pyramid"] = pyramid
            raw["padding_masks"] = masks
            raw["original_sizes"] = batch.original_sizes
            raw["padded_size"] = batch.padded_size
            raw["individual_padded_sizes"] = batch.individual_padded_sizes
        return raw

    def loss(
        self,
        images: HeadImageBatch | Sequence[torch.Tensor] | torch.Tensor,
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        batch = images if isinstance(images, HeadImageBatch) else HeadImageBatch.from_images(images)
        samples = self._build_data_samples(batch, targets)
        pyramid, masks = self.extract_pyramid(batch)
        raw = self.query_head(
            tuple(pyramid[level] for level in DETECTOR_LEVELS),
            tuple(masks[level] for level in DETECTOR_LEVELS),
            batch_data_samples=samples,
        )
        return self.query_head.loss(raw, samples)

    def predict(
        self,
        images: HeadImageBatch | Sequence[torch.Tensor] | torch.Tensor,
        *,
        return_features: bool = False,
    ) -> list[dict[str, torch.Tensor]] | dict[str, Any]:
        batch = images if isinstance(images, HeadImageBatch) else HeadImageBatch.from_images(images)
        raw = self.forward_raw(batch, return_features=return_features)
        predictions = postprocess_head_detections(
            raw["pred_logits"],
            raw["pred_boxes"],
            batch.original_size_list,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
        )
        if not return_features:
            return predictions
        return {"predictions": predictions, "raw_outputs": raw}

    def forward(
        self,
        images: HeadImageBatch | Sequence[torch.Tensor] | torch.Tensor,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        return_raw: bool = False,
        return_features: bool = False,
    ) -> Any:
        if targets is not None:
            if return_raw:
                return self.forward_raw(images, targets=targets, return_features=return_features)
            return self.loss(images, targets)
        if return_raw:
            return self.forward_raw(images, return_features=return_features)
        return self.predict(images, return_features=return_features)


def build_head_codino_detector(
    config: Mapping[str, Any] | None = None,
    *,
    backbone_module: nn.Module | None = None,
) -> HeadCoDINODetector:
    if backbone_module is not None:
        raise ValueError("head_codino requires the official ViT-S+/16 wrapper")
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", cfg))
    model_cfg.pop("type", None)
    allowed = {
        "dinov3_source_root",
        "backbone_weights",
        "freeze_backbone",
        "unfreeze_last_n_blocks",
        "category_id",
        "score_threshold",
        "max_detections",
    }
    unknown = sorted(set(model_cfg) - allowed)
    if unknown:
        raise ValueError(f"Unsupported head_codino model options: {unknown}")
    return HeadCoDINODetector(**model_cfg)


__all__ = [
    "DETECTOR_LEVELS",
    "EXPECTED_TOTAL_PARAMETERS",
    "HeadCoDINODetector",
    "PYRAMID_LEVELS",
    "build_head_codino_detector",
]
