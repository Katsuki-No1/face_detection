from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    name: str
    type: str
    weights: Path
    config: Path | None = None
    device: str = "cuda:0"
    input_size: int = 960
    confidence_threshold: float = 0.05
    analysis_threshold: float | None = None
    max_detections: int = 100
    batch_size: int = 8
    model_weight: float = 1.0
    class_id: int = 2
    class_name: str = "face"
    validation_map: float | None = None
    source_class_id: int = 0
    checkpoint_state: str = "model"
    embedding_source: bool = False
    model_ref: str = ""
    threshold_source: Path | None = None
    threshold_field: str = "fixed_threshold"


@dataclass(slots=True)
class ConfidenceBoundaryConfig:
    """Model-relative confidence band used for uncertainty sampling."""

    min_ratio: float = 0.70
    max_ratio: float = 1.30


@dataclass(slots=True)
class MatchingConfig:
    iou_threshold: float = 0.30
    center_distance_threshold: float = 0.25
    min_area_ratio: float = 0.33
    max_area_ratio: float = 3.0


@dataclass(slots=True)
class MontageConfig:
    width: int = 3840
    height: int = 2160
    fps: float = 30.0
    codec: str = "libx264"
    crf: int = 20
    preset: str = "medium"
    review_inference_fps: float = 2.0


@dataclass(slots=True)
class NoveltyConfig:
    enabled: bool = True
    reference_annotation: Path | None = None
    reference_image_root: Path | None = None
    head_reference_annotation: Path | None = None
    head_reference_image_root: Path | None = None
    reference_sample_count: int = 256
    cluster_count: int = 24
    roi_padding: float = 0.25
    roi_min_side: int = 16
    roi_input_size: int = 384


@dataclass(slots=True)
class DenseAnalysisConfig:
    enabled: bool = False
    fps: float = 5.0
    half_window_sec: float = 5.0
    proposal_multiplier: int = 5
    min_track_hits: int = 3
    max_track_gap_sec: float = 1.0


@dataclass(slots=True)
class PresenceClassifierConfig:
    enabled: bool = False
    checkpoint: Path | None = None
    metadata: Path | None = None
    face_threshold: float = 0.5
    head_threshold: float = 0.5
    roi_input_size: int = 384


@dataclass(slots=True)
class ClipWindowConfig:
    mode: str = "fixed"
    min_duration_sec: float = 4.0
    max_duration_sec: float = 10.0
    context_padding_sec: float = 1.0


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path
    config_path: Path
    input_dir: Path
    output_dir: Path
    models: list[ModelConfig]
    input_root: Path | None = None
    output_root: Path | None = None
    dataset_name: str = ""
    dataset_output_dir: Path | None = None
    frames_per_video: int = 100
    jpeg_quality: int = 95
    context_half_window_sec: float = 2.0
    sample_fps: float = 0.5
    clip_duration_sec: float = 10.0
    target_clip_count: int = 40
    allow_target_shortfall: bool = False
    max_clips_per_video: int = 5
    min_clips_per_video: int = 3
    min_interval_sec: float = 30.0
    max_clips_per_scene: int = 2
    max_clips_per_track: int = 2
    random_seed: int = 0
    selection_bucket_ratios: dict[str, float] = field(default_factory=lambda: {
        "uniform": 0.4,
        "model_disagreement": 0.2,
        "temporal_anomaly": 0.2,
        "other_instability": 0.2,
    })
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    montage: MontageConfig = field(default_factory=MontageConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    dense_analysis: DenseAnalysisConfig = field(default_factory=DenseAnalysisConfig)
    presence_classifier: PresenceClassifierConfig = field(default_factory=PresenceClassifierConfig)
    clip_window: ClipWindowConfig = field(default_factory=ClipWindowConfig)
    head_model: ModelConfig | None = None
    head_adoption_manifest: Path | None = None
    model_registry: Path | None = None
    confidence_boundary: ConfidenceBoundaryConfig = field(default_factory=ConfidenceBoundaryConfig)
    split_annotations: dict[str, Path] = field(default_factory=dict)
    recursive: bool = False
    video_include_globs: list[str] = field(default_factory=list)
    temporal_window_sec: float = 4.0
    scene_change_threshold: float = 0.32
    duplicate_phash_distance: int = 5
    dark_brightness_threshold: float = 8.0
    broken_contrast_threshold: float = 1.0

    def validate(self) -> None:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")
        if self.sample_fps <= 0 or self.clip_duration_sec <= 0:
            raise ValueError("sample_fps and clip_duration_sec must be positive")
        if self.target_clip_count <= 0 or self.max_clips_per_video <= 0 or self.frames_per_video <= 0:
            raise ValueError("clip counts must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.context_half_window_sec <= 0:
            raise ValueError("context_half_window_sec must be positive")
        if not self.models:
            raise ValueError("At least one model must be configured")
        supported = {
            "uniform", "other_instability",
            "model_disagreement", "common_difficulty", "novelty", "temporal_random",
            "temporal_anomaly", "no_detection_evidence", "roi_novelty", "low_presence_audit",
            "bbox_instability", "low_confidence", "head_evidence", "dino_novelty",
        }
        unknown = set(self.selection_bucket_ratios) - supported
        if unknown:
            raise ValueError(f"Unsupported selection bucket(s): {sorted(unknown)}")
        if any(float(value) < 0 for value in self.selection_bucket_ratios.values()):
            raise ValueError("selection_bucket_ratios cannot contain negative values")
        if sum(self.selection_bucket_ratios.values()) <= 0:
            raise ValueError("selection_bucket_ratios must sum to a positive value")
        frame_buckets = {
            "uniform", "model_disagreement", "temporal_anomaly", "other_instability",
            "bbox_instability", "low_confidence", "head_evidence", "dino_novelty",
        }
        if self.input_root is not None and not set(self.selection_bucket_ratios) <= frame_buckets:
            raise ValueError(
                "Frame dataset mode contains a non-frame selection bucket"
            )
        if any(not str(pattern).strip() for pattern in self.video_include_globs):
            raise ValueError("video_include_globs cannot contain empty patterns")
        configured_models = [*self.models, *([self.head_model] if self.head_model is not None else [])]
        names = [model.name for model in configured_models]
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique, including head_model")
        for model in configured_models:
            if not model.weights.is_file():
                raise FileNotFoundError(f"Missing {model.name} weights: {model.weights}")
            if model.config is not None and not model.config.is_file():
                raise FileNotFoundError(f"Missing {model.name} config: {model.config}")
            if model.analysis_threshold is not None and model.analysis_threshold < model.confidence_threshold:
                raise ValueError(f"{model.name} analysis_threshold must be >= confidence_threshold")
            if model.threshold_source is not None and not model.threshold_source.is_file():
                raise FileNotFoundError(f"Missing {model.name} threshold source: {model.threshold_source}")
            if model.source_class_id < 0:
                raise ValueError(f"{model.name} source_class_id must be non-negative")
            if model.checkpoint_state not in {"model", "ema", "auto"}:
                raise ValueError(f"{model.name} checkpoint_state must be model, ema, or auto")
        embedding_sources = [model.name for model in self.models if model.embedding_source]
        if self.dense_analysis.enabled and len(embedding_sources) != 1:
            raise ValueError(
                "Dense frame selection requires exactly one models[].embedding_source=true; "
                f"got {embedding_sources}"
            )
        if self.head_model is not None:
            if self.head_model.embedding_source:
                raise ValueError("head_model cannot be the DINO embedding source")
        if self.dense_analysis.enabled and self.dense_analysis.fps <= 0:
            raise ValueError("dense_analysis.fps must be positive")
        if not 0.0 < self.confidence_boundary.min_ratio < 1.0:
            raise ValueError("confidence_boundary.min_ratio must be between 0 and 1")
        if self.confidence_boundary.max_ratio <= 1.0:
            raise ValueError("confidence_boundary.max_ratio must be greater than 1")
        if self.clip_window.min_duration_sec <= 0 or self.clip_window.max_duration_sec < self.clip_window.min_duration_sec:
            raise ValueError("clip_window duration bounds are invalid")

    def resolved_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.resolve().as_posix()
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value
        return convert(asdict(self))


def _path(value: str | Path | None, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _nested_value(row: dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing threshold field {dotted_key!r}")
        value = value[part]
    return value


def _dataset_name(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("dataset_name must be one non-empty directory name")
    return name


def load_pipeline_config(path: Path, dataset_name: str | None = None) -> PipelineConfig:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("Pipeline config root must be a mapping")
    project_root = _path(raw.get("project_root", path.parent.parent), path.parent)
    assert project_root is not None
    model_registry = _path(raw.get("model_registry"), project_root)
    registry_models: dict[str, Any] = {}
    if model_registry is not None:
        registry_raw = _read_mapping(model_registry)
        registry_models = dict(registry_raw.get("models", registry_raw))

    def parse_model(row: dict[str, Any]) -> ModelConfig:
        override = dict(row)
        model_ref = str(override.pop("ref", override.pop("model_ref", "")))
        if model_ref:
            if model_ref not in registry_models:
                raise KeyError(f"Unknown model registry ref {model_ref!r} in {model_registry}")
            registered = registry_models[model_ref]
            if not isinstance(registered, dict):
                raise TypeError(f"Model registry entry {model_ref!r} must be a mapping")
            model_row = {**registered, **override}
        else:
            model_row = override
        model_row.pop("enabled", None)
        if "adapter" in model_row:
            model_row["type"] = model_row.pop("adapter")
        weights = _path(model_row.pop("weights"), project_root)
        model_config = _path(model_row.pop("config", None), project_root)
        threshold_source = _path(model_row.pop("threshold_source", None), project_root)
        threshold_field = str(model_row.pop("threshold_field", "fixed_threshold"))
        assert weights is not None
        if threshold_source is not None:
            threshold_metadata = _read_mapping(threshold_source)
            model_row["analysis_threshold"] = float(_nested_value(threshold_metadata, threshold_field))
            linked_weight = threshold_metadata.get("weights") or threshold_metadata.get("checkpoint")
            if linked_weight:
                linked_path = _path(str(linked_weight), project_root)
                if linked_path != weights:
                    raise RuntimeError(
                        f"{model_row.get('name', model_ref)} threshold source belongs to {linked_path}, "
                        f"not configured weights {weights}"
                    )
        return ModelConfig(
            weights=weights,
            config=model_config,
            model_ref=model_ref,
            threshold_source=threshold_source,
            threshold_field=threshold_field,
            **model_row,
        )

    models = [parse_model(row) for row in raw.get("models", [])]
    head_raw = raw.get("head_model")
    head_model = parse_model(dict(head_raw)) if isinstance(head_raw, dict) and head_raw.get("enabled", True) else None
    head_adoption_manifest = _path(raw.get("head_adoption_manifest"), project_root)
    if head_model is not None and head_adoption_manifest is not None and head_adoption_manifest.is_file():
        adoption = yaml.safe_load(head_adoption_manifest.read_text(encoding="utf-8")) or {}
        if adoption.get("selected_checkpoint") and adoption.get("selected_config"):
            head_model.weights = _path(adoption["selected_checkpoint"], project_root)  # type: ignore[assignment]
            head_model.config = _path(adoption["selected_config"], project_root)
            expected_hash = str(adoption.get("selected_checkpoint_sha256") or "")
            if expected_hash and head_model.weights is not None and head_model.weights.is_file() and _sha256(head_model.weights) != expected_hash:
                raise RuntimeError("Head adoption manifest checkpoint hash does not match the selected checkpoint")
    split_annotations = {
        str(name): resolved
        for name, value in dict(raw.get("split_annotations", {})).items()
        if (resolved := _path(value, project_root)) is not None
    }
    novelty_raw = dict(raw.get("novelty", {}))
    novelty_raw["reference_annotation"] = _path(novelty_raw.get("reference_annotation"), project_root)
    novelty_raw["reference_image_root"] = _path(novelty_raw.get("reference_image_root"), project_root)
    novelty_raw["head_reference_annotation"] = _path(novelty_raw.get("head_reference_annotation"), project_root)
    novelty_raw["head_reference_image_root"] = _path(novelty_raw.get("head_reference_image_root"), project_root)
    presence_raw = dict(raw.get("presence_classifier", {}))
    presence_raw["checkpoint"] = _path(presence_raw.get("checkpoint"), project_root)
    presence_raw["metadata"] = _path(presence_raw.get("metadata"), project_root)
    input_root = _path(raw.get("input_root"), project_root)
    output_root = _path(raw.get("output_root"), project_root)
    resolved_dataset_name = _dataset_name(dataset_name or raw.get("dataset_name", "")) if (input_root or output_root) else ""
    if (input_root is None) != (output_root is None):
        raise ValueError("input_root and output_root must be configured together")
    if input_root is not None and output_root is not None:
        input_dir = input_root / resolved_dataset_name
        dataset_output_dir = output_root / resolved_dataset_name
        output_dir = dataset_output_dir / "_active_learning"
    else:
        input_dir = _path(raw["input_dir"], project_root)
        output_dir = _path(raw["output_dir"], project_root)
        dataset_output_dir = output_dir
    config = PipelineConfig(
        project_root=project_root,
        config_path=path,
        input_dir=input_dir,  # type: ignore[arg-type]
        output_dir=output_dir,  # type: ignore[arg-type]
        models=models,
        input_root=input_root,
        output_root=output_root,
        dataset_name=resolved_dataset_name,
        dataset_output_dir=dataset_output_dir,
        frames_per_video=int(raw.get("frames_per_video", 100)),
        jpeg_quality=int(raw.get("jpeg_quality", 95)),
        context_half_window_sec=float(raw.get("context_half_window_sec", 2.0)),
        sample_fps=float(raw.get("sample_fps", 0.5)),
        clip_duration_sec=float(raw.get("clip_duration_sec", 10.0)),
        target_clip_count=int(raw.get("target_clip_count", 40)),
        allow_target_shortfall=bool(raw.get("allow_target_shortfall", False)),
        max_clips_per_video=int(raw.get("max_clips_per_video", 5)),
        min_clips_per_video=int(raw.get("min_clips_per_video", 3)),
        min_interval_sec=float(raw.get("min_interval_sec", 30.0)),
        max_clips_per_scene=int(raw.get("max_clips_per_scene", 2)),
        max_clips_per_track=int(raw.get("max_clips_per_track", 2)),
        random_seed=int(raw.get("random_seed", 0)),
        selection_bucket_ratios=dict(raw.get("selection_bucket_ratios", {})) or PipelineConfig.__dataclass_fields__["selection_bucket_ratios"].default_factory(),
        matching=MatchingConfig(**dict(raw.get("bbox_matching", {}))),
        montage=MontageConfig(**dict(raw.get("montage", {}))),
        novelty=NoveltyConfig(**novelty_raw),
        dense_analysis=DenseAnalysisConfig(**dict(raw.get("dense_analysis", {}))),
        presence_classifier=PresenceClassifierConfig(**presence_raw),
        clip_window=ClipWindowConfig(**dict(raw.get("clip_window", {}))),
        head_model=head_model,
        head_adoption_manifest=head_adoption_manifest,
        model_registry=model_registry,
        confidence_boundary=ConfidenceBoundaryConfig(**dict(raw.get("confidence_boundary", {}))),
        split_annotations=split_annotations,
        recursive=bool(raw.get("recursive", False)),
        video_include_globs=[str(value) for value in raw.get("video_include_globs", [])],
        temporal_window_sec=float(raw.get("temporal_window_sec", 4.0)),
        scene_change_threshold=float(raw.get("scene_change_threshold", 0.32)),
        duplicate_phash_distance=int(raw.get("duplicate_phash_distance", 5)),
        dark_brightness_threshold=float(raw.get("dark_brightness_threshold", 8.0)),
        broken_contrast_threshold=float(raw.get("broken_contrast_threshold", 1.0)),
    )
    assert isinstance(config.input_dir, Path) and isinstance(config.output_dir, Path)
    config.validate()
    return config
