from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from .strongest_types import REQUIRED_MODEL_CAPABILITIES, StrongestModelAdapter


CATEGORIES = (
    "temporal_ellipse_instability",
    "low_confidence",
    "face_without_keypoints",
    "extreme_screen_edge",
)


@dataclass(frozen=True, slots=True)
class MontageSettings:
    timeline_segments: int = 10
    clip_duration_sec: float = 60.0
    crf: int = 20
    preset: str = "fast"
    audio_bitrate: str = "192k"
    verify_decode: bool = False

    @property
    def duration_sec(self) -> float:
        return self.timeline_segments * self.clip_duration_sec


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    sample_fps: float = 5.0
    pipeline_batch_size: int = 4
    video_workers: int = 1
    scene_change_threshold: float = 0.32


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    max_frames_per_video: int = 50
    min_interval_sec: float = 10.0
    category_ratios: dict[str, float] = field(
        default_factory=lambda: {name: 0.25 for name in CATEGORIES}
    )
    confidence_min_ratio: float = 0.70
    confidence_max_ratio: float = 1.30
    extreme_edge_margin_ratio: float = 0.01
    temporal_max_gap_sec: float = 0.60
    temporal_min_track_hits: int = 3
    ellipse_center_jump_ratio: float = 0.75
    ellipse_axis_log_jump: float = 0.45
    ellipse_angle_jump_deg: float = 35.0


@dataclass(frozen=True, slots=True)
class OutputSettings:
    jpeg_quality: int = 95
    write_evidence_overlay: bool = True


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    adapter_class: str
    capabilities: frozenset[str]
    assets: dict[str, Path]
    options: dict[str, Any]
    face_confidence: float
    keypoint_confidence: float
    inference_confidence_floor: float
    threshold_provenance: dict[str, Any] = field(default_factory=dict)

    def build_adapter(self, device: str) -> StrongestModelAdapter:
        module_name, separator, class_name = self.adapter_class.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError(
                "adapter_class must use 'python.module:ClassName' syntax"
            )
        module = importlib.import_module(module_name)
        adapter_type = getattr(module, class_name)
        adapter = adapter_type(
            model_id=self.model_id,
            device=device,
            options={
                **self.options,
                **{name: path for name, path in self.assets.items()},
                "inference_confidence_floor": self.inference_confidence_floor,
            },
        )
        capabilities = frozenset(getattr(adapter, "capabilities", ()))
        if not REQUIRED_MODEL_CAPABILITIES <= capabilities:
            raise RuntimeError(
                f"{self.model_id} adapter lacks required capabilities: "
                f"{sorted(REQUIRED_MODEL_CAPABILITIES - capabilities)}"
            )
        return adapter


@dataclass(frozen=True, slots=True)
class StrongestPipelineConfig:
    project_root: Path
    config_path: Path
    registry_path: Path
    dataset_name: str
    input_dir: Path
    output_dir: Path
    model: ModelSpec
    device: str
    random_seed: int
    montage: MontageSettings
    inference: InferenceSettings
    selection: SelectionSettings
    output: OutputSettings
    recursive: bool = False
    video_include_globs: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Active-learning input does not exist: {self.input_dir}")
        for name, path in self.model.assets.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing active model asset {self.model.model_id}.{name}: {path}"
                )
        if not REQUIRED_MODEL_CAPABILITIES <= self.model.capabilities:
            raise ValueError(
                f"Model registry entry {self.model.model_id!r} must declare "
                f"{sorted(REQUIRED_MODEL_CAPABILITIES)}"
            )
        if self.montage.timeline_segments != 10:
            raise ValueError("The active-learning montage must divide the source into 10 segments")
        if abs(self.montage.duration_sec - 600.0) > 1e-6:
            raise ValueError("The active-learning montage must be exactly 10 minutes")
        if (
            self.inference.sample_fps <= 0
            or self.inference.pipeline_batch_size <= 0
            or self.inference.video_workers <= 0
        ):
            raise ValueError(
                "Inference fps, batch size, and video workers must be positive"
            )
        if self.selection.max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be positive")
        if self.selection.min_interval_sec < 0:
            raise ValueError("min_interval_sec cannot be negative")
        if tuple(self.selection.category_ratios) != CATEGORIES:
            raise ValueError(
                f"category_ratios must contain these categories in order: {CATEGORIES}"
            )
        if any(value < 0 for value in self.selection.category_ratios.values()):
            raise ValueError("category ratios cannot be negative")
        if sum(self.selection.category_ratios.values()) <= 0:
            raise ValueError("category ratios must sum to a positive value")
        if not 0 < self.selection.confidence_min_ratio < 1:
            raise ValueError("confidence_min_ratio must be between 0 and 1")
        if self.selection.confidence_max_ratio <= 1:
            raise ValueError("confidence_max_ratio must exceed 1")
        if not 0 <= self.selection.extreme_edge_margin_ratio <= 0.05:
            raise ValueError(
                "extreme_edge_margin_ratio must be between 0 and 0.05"
            )
        if not (
            0 < self.model.inference_confidence_floor
            <= self.model.face_confidence
            <= 1
        ):
            raise ValueError(
                "Model confidence thresholds must satisfy "
                "0 < inference_confidence_floor <= face_confidence <= 1"
            )
        if not 0 <= self.model.keypoint_confidence <= 1:
            raise ValueError("keypoint_confidence must be between 0 and 1")
        if not 1 <= self.output.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

    def resolved_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.resolve().as_posix()
            if isinstance(value, frozenset):
                return sorted(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _dataset_name(value: str) -> str:
    result = str(value).strip()
    if not result or result in {".", ".."} or Path(result).name != result:
        raise ValueError("dataset_name must be one directory name")
    return result


def _select_input_dir(
    roots: list[Path],
    dataset_name: str,
) -> Path:
    candidates = [root / dataset_name for root in roots]
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) > 1:
        raise RuntimeError(
            "The dataset exists below multiple input roots; keep one source of truth: "
            + ", ".join(path.as_posix() for path in existing)
        )
    if existing:
        return existing[0]
    return candidates[0]


def _parse_model(
    registry_path: Path,
    project_root: Path,
    model_ref: str | None,
) -> ModelSpec:
    registry = _mapping(registry_path)
    models = registry.get("models")
    if not isinstance(models, dict):
        raise TypeError(f"Model registry has no models mapping: {registry_path}")
    selected_ref = str(model_ref or registry.get("active_model", "")).strip()
    if selected_ref not in models:
        raise KeyError(f"Unknown strongest-model ref {selected_ref!r}")
    raw = models[selected_ref]
    if not isinstance(raw, dict):
        raise TypeError(f"Model registry entry {selected_ref!r} must be a mapping")
    thresholds = raw.get("thresholds", {})
    assets = raw.get("assets", {})
    if not isinstance(thresholds, dict) or not isinstance(assets, dict):
        raise TypeError(f"Invalid assets/thresholds for model {selected_ref!r}")
    return ModelSpec(
        model_id=selected_ref,
        adapter_class=str(raw["adapter_class"]),
        capabilities=frozenset(str(value) for value in raw.get("capabilities", ())),
        assets={
            str(name): _resolve(str(value), project_root)
            for name, value in assets.items()
        },
        options=dict(raw.get("options", {})),
        face_confidence=float(thresholds["face_confidence"]),
        keypoint_confidence=float(thresholds["keypoint_confidence"]),
        inference_confidence_floor=float(
            thresholds["inference_confidence_floor"]
        ),
        threshold_provenance={
            str(name): value
            for name, value in thresholds.items()
            if name
            not in {
                "face_confidence",
                "keypoint_confidence",
                "inference_confidence_floor",
            }
        },
    )


def load_strongest_pipeline_config(
    path: Path,
    *,
    dataset_name: str,
    model_ref: str | None = None,
) -> StrongestPipelineConfig:
    config_path = path.expanduser().resolve()
    raw = _mapping(config_path)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Strongest active-learning config must use schema_version: 1")
    project_root = _resolve(raw.get("project_root", ".."), config_path.parent)
    registry_path = _resolve(raw["model_registry"], project_root)
    name = _dataset_name(dataset_name)
    raw_roots = raw.get(
        "input_roots",
        ["data/input/active_learning"],
    )
    if not isinstance(raw_roots, list):
        raise TypeError("input_roots must be a list")
    input_roots = [_resolve(value, project_root) for value in raw_roots]
    input_dir = _select_input_dir(input_roots, name)
    output_root = _resolve(
        raw.get("output_root", "data/output/active_learning"),
        project_root,
    )
    config = StrongestPipelineConfig(
        project_root=project_root,
        config_path=config_path,
        registry_path=registry_path,
        dataset_name=name,
        input_dir=input_dir,
        output_dir=output_root / name,
        model=_parse_model(registry_path, project_root, model_ref),
        device=str(raw.get("device", "cuda:0")),
        random_seed=int(raw.get("random_seed", 20260728)),
        montage=MontageSettings(**dict(raw.get("montage", {}))),
        inference=InferenceSettings(**dict(raw.get("inference", {}))),
        selection=SelectionSettings(**dict(raw.get("selection", {}))),
        output=OutputSettings(**dict(raw.get("output", {}))),
        recursive=bool(raw.get("recursive", False)),
        video_include_globs=tuple(
            str(value) for value in raw.get("video_include_globs", ())
        ),
    )
    config.validate()
    return config
