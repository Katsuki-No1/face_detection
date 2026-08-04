from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from .rotations import matrix_to_euler_xyz, rotation_6d_to_matrix

PoseTrainingMode = Literal[
    "frozen",
    "domain_adapter",
    "partial",
    "full",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialSixDRepNet360(nn.Module):
    """ResNet-50 topology and state keys used by official 6DRepNet360."""

    feature_dimension = 2048

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_reg = nn.Linear(self.feature_dimension, 6)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        value = self.conv1(image)
        value = self.bn1(value)
        value = self.relu(value)
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return self.avgpool(value).flatten(1)

    def forward_raw(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(image)
        return self.linear_reg(features), features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw, _ = self.forward_raw(image)
        return rotation_6d_to_matrix(raw)


class _FusedBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Conv2d(
                inplanes,
                planes * self.expansion,
                1,
                stride=stride,
                bias=True,
            )
            if downsample
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.relu(self.conv1(value))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        if self.downsample is not None:
            identity = self.downsample(value)
        return self.relu(output + identity)


class FusedSixDRepNet360(nn.Module):
    """Trainable ResNet-50 with BatchNorm fused into convolution weights."""

    feature_dimension = 2048

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_reg = nn.Linear(self.feature_dimension, 6)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        *,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = stride != 1 or self.inplanes != planes * _FusedBottleneck.expansion
        layers = [
            _FusedBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * _FusedBottleneck.expansion
        layers.extend(_FusedBottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.conv1(image))
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return self.avgpool(value).flatten(1)

    def forward_raw(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(image)
        return self.linear_reg(features), features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw, _ = self.forward_raw(image)
        return rotation_6d_to_matrix(raw)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state: Any = raw
    if isinstance(raw, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "ema"):
            if isinstance(raw.get(key), Mapping):
                state = raw[key]
                break
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint has no tensor state mapping: {path}")
    normalized = {}
    for raw_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        for prefix in ("module.", "_orig_mod.", "model."):
            key = key.removeprefix(prefix)
        if key == "fc.weight":
            key = "linear_reg.weight"
        elif key == "fc.bias":
            key = "linear_reg.bias"
        normalized[key] = value.detach().cpu()
    return normalized


def load_official_pytorch_checkpoint(
    model: OfficialSixDRepNet360,
    path: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"6DRepNet360 checkpoint was not found: {checkpoint}")
    source = _checkpoint_state(checkpoint)
    target = model.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = {
        key: {
            "checkpoint": tuple(source[key].shape),
            "model": tuple(target[key].shape),
        }
        for key in set(source) & set(target)
        if tuple(source[key].shape) != tuple(target[key].shape)
    }
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "6DRepNet360 checkpoint is not architecture-compatible: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={dict(list(shape_mismatch.items())[:20])}"
        )
    model.load_state_dict(source, strict=True)
    return {
        "format": "official_pytorch",
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "loaded_tensors": len(target),
        "loaded_elements": sum(int(value.numel()) for value in target.values()),
        "missing": [],
        "unexpected": [],
        "shape_mismatch": {},
        "key_mapping": {
            "module./_orig_mod./model.": "stripped when present",
            "fc": "linear_reg",
        },
    }


def load_fused_onnx_checkpoint(
    model: FusedSixDRepNet360,
    path: str | Path,
) -> dict[str, Any]:
    """Restore the fused trainable network from a converted ONNX graph."""

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:  # pragma: no cover - dependency report covers it
        raise RuntimeError(
            "Loading converted 6DRepNet360 ONNX weights requires `onnx`."
        ) from error

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"6DRepNet360 ONNX file was not found: {checkpoint}")
    graph = onnx.load(checkpoint, load_external_data=True).graph
    initializer = {
        item.name: torch.from_numpy(numpy_helper.to_array(item).copy())
        for item in graph.initializer
    }
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    convolution_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    diagnostics: list[dict[str, Any]] = []
    if len(conv_nodes) != len(convolution_modules):
        raise RuntimeError(
            "Converted 6DRepNet360 convolution count mismatch: "
            f"onnx={len(conv_nodes)}, model={len(convolution_modules)}"
        )
    with torch.no_grad():
        for node, (module_name, module) in zip(
            conv_nodes,
            convolution_modules,
            strict=True,
        ):
            if len(node.input) < 3:
                raise RuntimeError(f"ONNX Conv node has no fused bias: {node.name}")
            weight = initializer.get(node.input[1])
            bias = initializer.get(node.input[2])
            if weight is None or bias is None:
                raise RuntimeError(f"Missing ONNX initializer for Conv {node.name}")
            if tuple(weight.shape) != tuple(module.weight.shape):
                raise RuntimeError(
                    f"ONNX Conv shape mismatch for {module_name}: "
                    f"{tuple(weight.shape)} != {tuple(module.weight.shape)}"
                )
            if module.bias is None or tuple(bias.shape) != tuple(module.bias.shape):
                raise RuntimeError(f"ONNX Conv bias mismatch for {module_name}")
            module.weight.copy_(weight.to(module.weight))
            module.bias.copy_(bias.to(module.bias))
            diagnostics.append(
                {
                    "onnx_node": node.name,
                    "model_module": module_name,
                    "weight": node.input[1],
                    "bias": node.input[2],
                    "shape": tuple(weight.shape),
                }
            )
        for name in ("linear_reg.weight", "linear_reg.bias"):
            if name not in initializer:
                raise RuntimeError(f"Missing ONNX initializer {name}")
        if tuple(initializer["linear_reg.weight"].shape) != tuple(
            model.linear_reg.weight.shape
        ):
            raise RuntimeError("ONNX linear_reg.weight shape mismatch")
        model.linear_reg.weight.copy_(
            initializer["linear_reg.weight"].to(model.linear_reg.weight)
        )
        model.linear_reg.bias.copy_(
            initializer["linear_reg.bias"].to(model.linear_reg.bias)
        )
    return {
        "format": "converted_onnx_fused_resnet50",
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "producer": f"{onnx.load(checkpoint, load_external_data=False).producer_name}",
        "loaded_convolutions": len(diagnostics),
        "loaded_tensors": 2 * len(diagnostics) + 2,
        "loaded_elements": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "key_mapping": diagnostics,
        "caveat": (
            "BatchNorm was fused by the ONNX exporter; provenance is the "
            "converted full-rotation model, not byte identity with the missing .pth."
        ),
    }


class PoseEncoderAdapter(nn.Module):
    """6DRepNet360 adapter with configurable fine-tuning and uncertainty."""

    def __init__(
        self,
        backbone: OfficialSixDRepNet360 | FusedSixDRepNet360,
        *,
        training_mode: PoseTrainingMode = "frozen",
        embedding_dimension: int = 128,
        allow_full_finetune: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.training_mode: PoseTrainingMode = training_mode
        self.embedding_dimension = int(embedding_dimension)
        feature_dimension = int(backbone.feature_dimension)
        adapter_hidden = min(256, feature_dimension)
        self.domain_adapter = nn.Sequential(
            nn.Linear(feature_dimension, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, 6),
        )
        self.pose_embedding = nn.Sequential(
            nn.Linear(feature_dimension + 9, self.embedding_dimension),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dimension),
        )
        self.uncertainty_head = nn.Linear(feature_dimension, 1)
        output = self.domain_adapter[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)
        self.pretrained_load_info: dict[str, Any] = {}
        self._configure_training_mode(
            training_mode,
            allow_full_finetune=allow_full_finetune,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def _configure_training_mode(
        self,
        mode: PoseTrainingMode,
        *,
        allow_full_finetune: bool,
    ) -> None:
        if mode not in {"frozen", "domain_adapter", "partial", "full"}:
            raise ValueError(f"Unsupported pose training mode: {mode}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.domain_adapter.parameters():
            parameter.requires_grad_(mode != "frozen")
        if mode == "partial":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)
            for parameter in self.backbone.linear_reg.parameters():
                parameter.requires_grad_(True)
        elif mode == "full":
            if not allow_full_finetune:
                raise ValueError(
                    "Full 6DRepNet360 fine-tuning requires allow_full_finetune=true"
                )
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> PoseEncoderAdapter:
        super().train(mode)
        if self.training_mode in {"frozen", "domain_adapter"}:
            self.backbone.eval()
        elif self.training_mode == "partial":
            for name, module in self.backbone.named_children():
                module.train(mode and name in {"layer4", "linear_reg"})
        return self

    def preprocess(self, crop: torch.Tensor) -> torch.Tensor:
        if crop.ndim != 4 or crop.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB crops, got {tuple(crop.shape)}")
        value = crop
        if value.shape[-2:] != (224, 224):
            value = F.interpolate(
                value,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        return (value - self.image_mean.to(value)) / self.image_std.to(value)

    def forward(self, crop: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.preprocess(crop)
        raw, features = self.backbone.forward_raw(normalized)
        if self.training_mode != "frozen":
            raw = raw + self.domain_adapter(features)
        rotation = rotation_6d_to_matrix(raw)
        uncertainty = F.softplus(self.uncertainty_head(features)).squeeze(-1)
        confidence = torch.exp(-uncertainty).clamp(0.0, 1.0)
        embedding = self.pose_embedding(
            torch.cat((features, rotation.flatten(1)), dim=-1)
        )
        return {
            "rotation_matrix": rotation,
            "rotation_6d": raw,
            "pose_embedding": embedding,
            "pose_uncertainty": uncertainty,
            "pose_confidence": confidence,
            "euler_xyz": matrix_to_euler_xyz(rotation),
            "features": features,
        }


def build_pose_encoder(
    *,
    checkpoint: str | Path,
    training_mode: PoseTrainingMode = "frozen",
    embedding_dimension: int = 128,
    allow_full_finetune: bool = False,
) -> PoseEncoderAdapter:
    path = Path(checkpoint)
    if path.suffix.lower() == ".onnx":
        backbone: OfficialSixDRepNet360 | FusedSixDRepNet360 = FusedSixDRepNet360()
        load_info = load_fused_onnx_checkpoint(backbone, path)
    else:
        backbone = OfficialSixDRepNet360()
        load_info = load_official_pytorch_checkpoint(backbone, path)
    adapter = PoseEncoderAdapter(
        backbone,
        training_mode=training_mode,
        embedding_dimension=embedding_dimension,
        allow_full_finetune=allow_full_finetune,
    )
    adapter.pretrained_load_info = load_info
    return adapter


__all__ = [
    "FusedSixDRepNet360",
    "OfficialSixDRepNet360",
    "PoseEncoderAdapter",
    "PoseTrainingMode",
    "build_pose_encoder",
    "load_fused_onnx_checkpoint",
    "load_official_pytorch_checkpoint",
    "sha256_file",
]
