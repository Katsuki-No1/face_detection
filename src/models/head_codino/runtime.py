from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mmcv.ops import MultiScaleDeformableAttention
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler

from src.data import CocoDetectionDataset
from src.config_utils import load_yaml_config

from .model import HeadCoDINODetector, build_head_codino_detector
from .structures import collate_head_detection_batch


_FIXED_ARCHITECTURE = {
    "backbone": {
        "name": "dinov3_vits16plus",
        "patch_size": 16,
        "embed_dim": 384,
        "blocks": 12,
        "attention_heads": 6,
        "output": "x_norm_patchtokens",
    },
    "sfp": {
        "input_channels": 384,
        "output_channels": 256,
        "levels": ["p3", "p4", "p5", "p6"],
        "strides": [8, 16, 32, 64],
        "detector_levels": ["p3", "p4", "p5"],
        "conv_bias": False,
    },
    "codino": {
        "hidden_dim": 256,
        "feature_levels": 3,
        "encoder_layers": 1,
        "decoder_layers": 3,
        "attention_heads": 8,
        "sampling_points": 4,
        "ffn_dim": 1024,
        "activation": "ReLU",
        "matching_queries": 100,
        "two_stage": True,
        "mixed_selection": True,
        "iterative_box_refinement": True,
        "dn_query_budget": 50,
        "dn_label_noise": 0.5,
        "dn_box_noise": 0.4,
        "positional_encoding": {
            "num_feats": 128,
            "temperature": 20,
            "normalize": True,
            "dropout": 0.0,
        },
    },
    "matching_costs": {"focal": 2.0, "l1": 5.0, "giou": 2.0},
    "losses": {"quality_focal": 1.0, "l1": 5.0, "giou": 2.0},
}


def _validate_fixed_values(actual: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    for key, expected_value in expected.items():
        key_path = f"{path}.{key}"
        if key not in actual:
            raise ValueError(f"Missing fixed architecture value: {key_path}")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"{key_path} must be a mapping")
            _validate_fixed_values(actual_value, expected_value, key_path)
        elif actual_value != expected_value:
            raise ValueError(
                f"{key_path} is checkpoint-fixed: expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )


def validate_architecture_contract(config: Mapping[str, Any]) -> None:
    architecture = config.get("architecture")
    if architecture is not None:
        if not isinstance(architecture, Mapping):
            raise ValueError("architecture must be a mapping")
        _validate_fixed_values(architecture, _FIXED_ARCHITECTURE, "architecture")


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: str | Path, *, config: Mapping[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(str(config["_config_path"]))
    repository_root = config_path.parents[2]
    return (repository_root / path).resolve()


def build_model_from_runtime_config(config: Mapping[str, Any]) -> HeadCoDINODetector:
    validate_architecture_contract(config)
    model_config = dict(config.get("model", {}))
    for key in ("dinov3_source_root", "backbone_weights"):
        if model_config.get(key):
            model_config[key] = str(resolve_path(model_config[key], config=config))
    return build_head_codino_detector({"model": model_config})


def build_dataset(config: Mapping[str, Any], split: str) -> CocoDetectionDataset:
    data = dict(config.get("data", {}))
    split_cfg = dict(data.get(split, {}))
    annotation = split_cfg.get("annotation")
    image_root = split_cfg.get("image_root")
    if annotation is None:
        raise ValueError(f"data.{split}.annotation is required")
    return CocoDetectionDataset(
        annotation_path=resolve_path(annotation, config=config),
        image_root=resolve_path(image_root, config=config) if image_root else None,
        data_root=resolve_path(data["root"], config=config) if data.get("root") else None,
        split=split,
        # The shared upstream detector uses the same square input contract as
        # the integrated three-stage branch.  ``None`` preserves the original
        # native-resolution detector for historical configs.
        image_size=(
            int(data["image_size"])
            if data.get("image_size") is not None
            else None
        ),
        include_negative=True,
        category_ids=[1],
    )


def representative_subset_indices(
    dataset: Any,
    maximum_samples: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select a deterministic, source-video-balanced bounded subset.

    The fixed COCO splits are ordered by source video. Taking ``range(N)``
    therefore turns a small-scale gate into a one- or two-video overfit run.
    Allocate samples round-robin across source videos and then spread each
    group's selections over its full timeline.
    """

    total = len(dataset)
    requested = int(maximum_samples)
    if requested < 1:
        raise ValueError("maximum_samples must be positive")
    selected_count = min(requested, total)
    samples = getattr(dataset, "samples", None)
    grouped: dict[str, list[int]] = {}
    if isinstance(samples, Sequence) and len(samples) == total:
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                grouped = {}
                break
            image = sample.get("image")
            source = image.get("source_video_id") if isinstance(image, Mapping) else None
            if source in {None, ""}:
                grouped = {}
                break
            grouped.setdefault(str(source), []).append(index)

    if grouped:
        allocations = {source: 0 for source in grouped}
        remaining = selected_count
        while remaining:
            progressed = False
            for source, indices in grouped.items():
                if allocations[source] >= len(indices):
                    continue
                allocations[source] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                raise RuntimeError("Could not allocate the requested bounded subset")
        selected: list[int] = []
        for source, indices in grouped.items():
            count = allocations[source]
            if count == 0:
                continue
            if count == 1:
                selected.append(indices[len(indices) // 2])
                continue
            selected.extend(
                indices[round(position * (len(indices) - 1) / (count - 1))]
                for position in range(count)
            )
        selected.sort()
        strategy = "source_video_balanced_even_spacing"
        selected_per_group = {
            source: allocations[source]
            for source in grouped
            if allocations[source] > 0
        }
    else:
        if selected_count == 1:
            selected = [total // 2]
        else:
            selected = [
                round(position * (total - 1) / (selected_count - 1))
                for position in range(selected_count)
            ]
        strategy = "global_even_spacing"
        selected_per_group = {}

    if len(selected) != selected_count or len(set(selected)) != selected_count:
        raise RuntimeError("Representative subset selection produced duplicate indices")
    return selected, {
        "strategy": strategy,
        "total_samples": total,
        "requested_samples": requested,
        "selected_samples": selected_count,
        "available_source_groups": len(grouped),
        "selected_source_groups": len(selected_per_group),
        "selected_per_source_group": selected_per_group,
        "indices": selected,
    }


def build_dataloader(
    config: Mapping[str, Any],
    split: str,
    *,
    distributed: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = build_dataset(config, split)
    section = dict(config.get("train" if split == "train" else "evaluation", {}))
    sampler = (
        DistributedSampler(dataset, shuffle=split == "train", drop_last=False)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(section.get("batch_size", 1)),
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        num_workers=int(section.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(section.get("num_workers", 4)) > 0,
        collate_fn=collate_head_detection_batch,
        drop_last=False,
    )
    return loader, sampler


def move_targets_to_device(
    targets: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def seed_everything(seed: int, *, rank: int = 0) -> None:
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def cuda_operator_preflight(device: torch.device | str = "cuda") -> dict[str, Any]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production Head Co-DINO requires a CUDA device")
    module = MultiScaleDeformableAttention(
        embed_dims=256,
        num_heads=8,
        num_levels=3,
        num_points=4,
        batch_first=True,
    ).to(device)
    query = torch.zeros((1, 4, 256), device=device)
    value = torch.zeros((1, 3, 256), device=device)
    spatial_shapes = torch.tensor(((1, 1), (1, 1), (1, 1)), device=device)
    level_start_index = torch.tensor((0, 1, 2), device=device)
    reference_points = torch.full((1, 4, 3, 2), 0.5, device=device)
    with torch.no_grad():
        output = module(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
    if output.shape != query.shape or not torch.isfinite(output).all():
        raise RuntimeError("CUDA MultiScaleDeformableAttention preflight returned invalid output")
    return {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "operator": "mmcv.ops.MultiScaleDeformableAttention",
        "status": "ok",
    }


def read_rgb_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_predictions_to_coco(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prediction, target in zip(predictions, targets, strict=True):
        image_id: Any = target.get("image_id")
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            pass
        boxes = prediction["boxes"].detach().cpu()
        scores = prediction["scores"].detach().cpu()
        resized_height, resized_width = (
            int(value) for value in target.get("size", target.get("orig_size"))
        )
        original_height, original_width = (
            int(value) for value in target.get("orig_size", target.get("size"))
        )
        boxes = boxes.clone()
        if boxes.numel() and (
            resized_height != original_height or resized_width != original_width
        ):
            boxes[:, 0::2] *= original_width / float(resized_width)
            boxes[:, 1::2] *= original_height / float(resized_height)
        for box, score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = box.tolist()
            records.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return records


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


__all__ = [
    "build_dataloader",
    "build_dataset",
    "build_model_from_runtime_config",
    "cuda_operator_preflight",
    "load_runtime_config",
    "move_targets_to_device",
    "representative_subset_indices",
    "read_rgb_tensor",
    "resolve_path",
    "save_json",
    "seed_everything",
    "tensor_predictions_to_coco",
    "validate_architecture_contract",
]
