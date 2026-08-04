from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DynamicShape:
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def validate(self, *, name: str) -> None:
        ranks = {len(self.minimum), len(self.optimum), len(self.maximum)}
        if len(ranks) != 1 or not self.minimum:
            raise ValueError(f"{name} min/opt/max shapes must have one nonzero rank")
        for dimension, values in enumerate(
            zip(self.minimum, self.optimum, self.maximum, strict=True)
        ):
            low, optimum, high = values
            if low <= 0 or not low <= optimum <= high:
                raise ValueError(
                    f"{name} dimension {dimension} must satisfy "
                    f"0 < min <= opt <= max, got {values}"
                )


@dataclass(frozen=True, slots=True)
class TensorRTEngineBuild:
    name: str
    onnx_path: Path
    mixed_precision_onnx_path: Path
    engine_path: Path
    inputs: dict[str, DynamicShape]


@dataclass(frozen=True, slots=True)
class TensorRTBuildProfile:
    path: Path
    precision: str
    workspace_mib: int
    hardware_compatibility: str
    engines: tuple[TensorRTEngineBuild, ...]


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_tensorrt_build_profile(
    path: Path,
    *,
    require_onnx: bool = True,
) -> TensorRTBuildProfile:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("TensorRT profile must be a schema_version: 1 mapping")
    project_root = _resolve(raw.get("project_root", ".."), config_path.parent)
    precision = str(raw.get("precision", "")).lower()
    if precision != "fp16":
        raise ValueError("This active-learning TensorRT profile must use fp16")
    workspace_mib = int(raw.get("workspace_mib", 0))
    if workspace_mib <= 0:
        raise ValueError("workspace_mib must be positive")
    hardware_compatibility = str(
        raw.get("hardware_compatibility", "exact_gpu")
    )
    if hardware_compatibility not in {"exact_gpu", "ampere_plus"}:
        raise ValueError(
            "hardware_compatibility must be exact_gpu or ampere_plus"
        )
    raw_engines = raw.get("engines")
    if not isinstance(raw_engines, dict) or not raw_engines:
        raise ValueError("TensorRT profile must define at least one engine")
    engines: list[TensorRTEngineBuild] = []
    for engine_name, raw_engine_value in raw_engines.items():
        if not isinstance(raw_engine_value, dict):
            raise TypeError(f"engines.{engine_name} must be a mapping")
        raw_engine: dict[str, Any] = dict(raw_engine_value)
        onnx_path = _resolve(raw_engine["onnx"], project_root)
        mixed_precision_onnx_path = _resolve(
            raw_engine["mixed_precision_onnx"],
            project_root,
        )
        engine_path = _resolve(raw_engine["engine"], project_root)
        if require_onnx and not onnx_path.is_file():
            raise FileNotFoundError(
                f"TensorRT ONNX input does not exist: {onnx_path}"
            )
        raw_inputs = raw_engine.get("inputs")
        if not isinstance(raw_inputs, dict) or not raw_inputs:
            raise ValueError(f"engines.{engine_name}.inputs cannot be empty")
        inputs: dict[str, DynamicShape] = {}
        for input_name, raw_shape_value in raw_inputs.items():
            if not isinstance(raw_shape_value, dict):
                raise TypeError(
                    f"engines.{engine_name}.inputs.{input_name} "
                    "must be a mapping"
                )
            raw_shape = dict(raw_shape_value)
            shape = DynamicShape(
                minimum=tuple(int(value) for value in raw_shape["min"]),
                optimum=tuple(int(value) for value in raw_shape["opt"]),
                maximum=tuple(int(value) for value in raw_shape["max"]),
            )
            shape.validate(name=f"{engine_name}.{input_name}")
            inputs[str(input_name)] = shape
        engines.append(
            TensorRTEngineBuild(
                name=str(engine_name),
                onnx_path=onnx_path,
                mixed_precision_onnx_path=mixed_precision_onnx_path,
                engine_path=engine_path,
                inputs=inputs,
            )
        )
    return TensorRTBuildProfile(
        path=config_path,
        precision=precision,
        workspace_mib=workspace_mib,
        hardware_compatibility=hardware_compatibility,
        engines=tuple(engines),
    )
