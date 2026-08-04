from __future__ import annotations

import csv
import fnmatch
import hashlib
import html
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from script.script_data.clip_frame import (
    VideoFingerprint,
    build_video_history_rows,
    check_video_name_duplicates,
    find_videos,
    format_known_video_matches,
    merge_video_history,
    read_video_history,
    safe_name,
    search_past_clip_history,
    write_video_history,
)
from scripts.create_uniform_video_montage import (
    ClipRange,
    build_clip_ranges,
    create_montage,
)

from .media import (
    frame_quality,
    pixel_sha256,
    probe_video,
    read_video_frame,
    sample_frames_ffmpeg,
)
from .strongest_config import CATEGORIES, StrongestPipelineConfig
from .strongest_types import (
    FaceObservation,
    FailureCandidate,
    InferenceFrame,
    StrongestModelAdapter,
)
from .types import VideoInfo


LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
CATEGORY_LABELS = {
    "temporal_ellipse_instability": "時系列的な矛盾・楕円の不安定",
    "low_confidence": "低confidence",
    "face_without_keypoints": "faceのみ・keypointなし",
    "extreme_screen_edge": "画面の最外周1%に接する顔",
}


@dataclass(frozen=True, slots=True)
class MontageSegment:
    index: int
    source_start_sec: float
    source_end_sec: float
    montage_start_sec: float
    montage_end_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VideoWork:
    source: VideoInfo
    work_id: str
    work_dir: Path
    montage_path: Path
    montage_segments: tuple[MontageSegment, ...]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Content-aware fingerprint without reading a multi-GB video completely."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as source:
        digest.update(source.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            source.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(source.read(1024 * 1024))
    return {
        "path": path.resolve().as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampled_sha256": digest.hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def discover_source_videos(config: StrongestPipelineConfig) -> list[Path]:
    videos = [
        path
        for path in find_videos(config.input_dir, config.recursive)
        if path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if config.video_include_globs:
        videos = [
            path
            for path in videos
            if any(
                fnmatch.fnmatch(path.name, pattern)
                or fnmatch.fnmatch(path.stem, pattern)
                for pattern in config.video_include_globs
            )
        ]
    if not videos:
        raise FileNotFoundError(
            f"No input video found in {config.input_dir} "
            f"for globs={config.video_include_globs!r}"
        )
    return sorted(path.resolve() for path in videos)


def preflight_source_video_names(
    videos: Sequence[Path],
    *,
    history_csv: Path,
    project_root: Path,
    allow_known_names: bool,
) -> dict[str, Any]:
    """Reject duplicate current/past source names before montage creation."""

    resolved_history = history_csv.expanduser()
    if not resolved_history.is_absolute():
        resolved_history = project_root / resolved_history
    resolved_history = resolved_history.resolve()
    history_rows = read_video_history(resolved_history)
    recovered_rows = search_past_clip_history(project_root)
    searchable_rows, _ = merge_video_history(history_rows, recovered_rows)
    known_names = check_video_name_duplicates(
        list(videos),
        searchable_rows,
        allow_known_names=allow_known_names,
    )
    return {
        "history_csv": resolved_history.as_posix(),
        "input_videos": len(videos),
        "past_records": len(searchable_rows),
        "allowed_known_names": [
            {
                "source": path.name,
                "history": history_name,
            }
            for path, history_name in known_names
        ],
        "allowed_match_detail": (
            format_known_video_matches(known_names, [])
            if known_names
            else ""
        ),
    }


def record_completed_source_video_names(
    videos: Sequence[Path],
    *,
    input_dir: Path,
    output_dir: Path,
    history_csv: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Record source names only after a montage/all run completes cleanly."""

    resolved_history = history_csv.expanduser()
    if not resolved_history.is_absolute():
        resolved_history = project_root / resolved_history
    resolved_history = resolved_history.resolve()
    history_rows = read_video_history(resolved_history)
    recovered_rows = search_past_clip_history(project_root)
    merged_rows, _ = merge_video_history(history_rows, recovered_rows)
    fingerprints = {
        path: VideoFingerprint(
            content_sha256="",
            file_size_bytes=path.stat().st_size,
        )
        for path in videos
    }
    new_rows = build_video_history_rows(
        videos=list(videos),
        input_dir=input_dir,
        output_root=output_dir,
        recorded_at=datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        fingerprints=fingerprints,
        evidence_source="active_learning_strongest",
    )
    merged_rows, added_rows = merge_video_history(merged_rows, new_rows)
    write_video_history(resolved_history, merged_rows)
    run_history_path = output_dir / "video_names.csv"
    write_video_history(run_history_path, new_rows)
    return {
        "history_csv": resolved_history.as_posix(),
        "run_video_names_csv": run_history_path.resolve().as_posix(),
        "recorded_videos": len(new_rows),
        "new_global_records": added_rows,
    }


def _work_id(path: Path) -> str:
    slug = "_".join((safe_name(path.stem) or "video").split())[:80]
    digest = hashlib.sha256(
        path.resolve().as_posix().encode("utf-8")
    ).hexdigest()[:10]
    return f"{slug}__{digest}"


def _segments_from_ranges(ranges: Sequence[ClipRange]) -> tuple[MontageSegment, ...]:
    montage_cursor = 0.0
    result: list[MontageSegment] = []
    for row in ranges:
        duration = row.clip_end_sec - row.clip_start_sec
        result.append(
            MontageSegment(
                index=int(row.index),
                source_start_sec=float(row.clip_start_sec),
                source_end_sec=float(row.clip_end_sec),
                montage_start_sec=montage_cursor,
                montage_end_sec=montage_cursor + duration,
            )
        )
        montage_cursor += duration
    return tuple(result)


def _load_montage_segments(path: Path) -> tuple[MontageSegment, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(MontageSegment(**row) for row in payload["segments"])


def prepare_montage(
    config: StrongestPipelineConfig,
    source_path: Path,
    *,
    force: bool = False,
) -> VideoWork:
    source = probe_video(source_path)
    if not source.readable:
        raise RuntimeError(f"Unreadable source video {source_path}: {source.error}")
    work_id = _work_id(source_path)
    work_dir = config.output_dir / "_active_learning" / "videos" / work_id
    montage_path = work_dir / "montage" / f"{work_id}__uniform_10m.mp4"
    manifest_path = work_dir / "montage" / "montage_manifest.json"
    ranges = build_clip_ranges(
        source.duration,
        config.montage.timeline_segments,
        config.montage.clip_duration_sec,
    )
    segments = _segments_from_ranges(ranges)
    expected_signature = _signature(
        {
            "source": _file_fingerprint(source_path),
            "timeline_segments": config.montage.timeline_segments,
            "clip_duration_sec": config.montage.clip_duration_sec,
            "crf": config.montage.crf,
            "preset": config.montage.preset,
            "audio_bitrate": config.montage.audio_bitrate,
        }
    )
    if montage_path.is_file() and manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("signature") != expected_signature:
            raise RuntimeError(
                f"Montage provenance changed for {source.filename}; rerun with --force"
            )
        return VideoWork(
            source=source,
            work_id=work_id,
            work_dir=work_dir,
            montage_path=montage_path,
            montage_segments=_load_montage_segments(manifest_path),
        )
    montage_path.parent.mkdir(parents=True, exist_ok=True)
    create_montage(
        SimpleNamespace(
            input=source_path,
            output=montage_path,
            segments=config.montage.timeline_segments,
            clip_duration=config.montage.clip_duration_sec,
            crf=config.montage.crf,
            preset=config.montage.preset,
            audio_bitrate=config.montage.audio_bitrate,
            force=bool(force),
            verify_decode=config.montage.verify_decode,
        )
    )
    montage_media = probe_video(montage_path)
    if not montage_media.readable:
        raise RuntimeError(
            f"Generated montage is unreadable {montage_path}: {montage_media.error}"
        )
    payload = {
        "schema_version": 1,
        "signature": expected_signature,
        "source": _file_fingerprint(source_path),
        "source_video": source.to_dict(),
        "montage_path": montage_path.resolve().as_posix(),
        "montage_video": montage_media.to_dict(),
        "policy": {
            "timeline_segments": config.montage.timeline_segments,
            "clip_duration_sec": config.montage.clip_duration_sec,
            "output_duration_sec": config.montage.duration_sec,
            "sampling": "centered contiguous clip from every equal source interval",
        },
        "segments": [segment.to_dict() for segment in segments],
    }
    _write_json(manifest_path, payload)
    return VideoWork(
        source=source,
        work_id=work_id,
        work_dir=work_dir,
        montage_path=montage_path,
        montage_segments=segments,
    )


def map_montage_timestamp(
    timestamp_sec: float,
    segments: Sequence[MontageSegment],
) -> tuple[int, float]:
    if not segments:
        raise ValueError("Montage mapping has no segments")
    timestamp = max(0.0, float(timestamp_sec))
    segment = next(
        (
            row
            for row in segments
            if row.montage_start_sec
            <= timestamp
            < row.montage_end_sec - 1e-9
        ),
        segments[-1],
    )
    offset = min(
        timestamp - segment.montage_start_sec,
        segment.source_end_sec - segment.source_start_sec,
    )
    return segment.index, segment.source_start_sec + max(0.0, offset)


def align_source_timestamps_to_frame_grid(
    frames: Sequence[InferenceFrame],
    source_fps: float,
) -> list[InferenceFrame]:
    """Use the same source-frame time grid that final export will decode."""

    fps = float(source_fps)
    if fps <= 0:
        raise ValueError("source_fps must be positive")
    return [
        replace(
            frame,
            source_timestamp_sec=frame.source_frame_id / fps,
        )
        for frame in frames
    ]


def _model_provenance_signature(provenance: Mapping[str, Any]) -> str:
    return _signature(provenance)


def _prediction_model_contract(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only provenance fields that can change raw model predictions.

    The validation threshold-selection file is retained in full run
    provenance, but the adapter never reads it during inference. Its later
    documentation edits must not invalidate immutable raw predictions.
    """

    contract = json.loads(json.dumps(dict(provenance)))
    assets = dict(contract.get("assets", {}))
    assets.pop("threshold_selection_config", None)
    contract["assets"] = assets
    thresholds = dict(contract.get("thresholds", {}))
    contract["thresholds"] = {
        "inference_confidence_floor": thresholds.get(
            "inference_confidence_floor"
        )
    }
    return contract


def resolved_model_provenance(
    config: StrongestPipelineConfig,
) -> dict[str, Any]:
    return {
        "model_id": config.model.model_id,
        "adapter": config.model.adapter_class,
        "device": config.device,
        "capabilities": sorted(config.model.capabilities),
        "assets": {
            name: {
                "path": path.resolve().as_posix(),
                "sha256": _sha256_file(path),
            }
            for name, path in config.model.assets.items()
        },
        "options": config.model.options,
        "thresholds": {
            "inference_confidence_floor": (
                config.model.inference_confidence_floor
            ),
            "face_confidence": config.model.face_confidence,
            "keypoint_confidence": config.model.keypoint_confidence,
            "provenance": config.model.threshold_provenance,
        },
    }


def _inference_signature(
    config: StrongestPipelineConfig,
    work: VideoWork,
    model_provenance: Mapping[str, Any],
) -> str:
    return _signature(
        {
            "montage": _file_fingerprint(work.montage_path),
            "model": _model_provenance_signature(model_provenance),
            "sample_fps": config.inference.sample_fps,
            "scene_change_threshold": config.inference.scene_change_threshold,
            "mapping": [row.to_dict() for row in work.montage_segments],
        }
    )


def read_completed_inference(
    config: StrongestPipelineConfig,
    work: VideoWork,
    model_provenance: Mapping[str, Any],
) -> list[InferenceFrame]:
    prediction_path = work.work_dir / "inference" / "predictions.jsonl"
    status_path = work.work_dir / "inference" / "status.json"
    if not prediction_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(
            f"Run inference first for {work.source.filename}: {prediction_path}"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_signature = _inference_signature(config, work, model_provenance)
    signature_matches = status.get("signature") == expected_signature
    if not signature_matches:
        stored_model = status.get("model")
        signature_matches = (
            isinstance(stored_model, dict)
            and status.get("signature")
            == _inference_signature(config, work, stored_model)
            and _prediction_model_contract(stored_model)
            == _prediction_model_contract(model_provenance)
        )
        if signature_matches:
            LOGGER.warning(
                "Reusing immutable predictions for %s: only "
                "selection/documentation provenance changed",
                work.source.filename,
            )
    if status.get("status") != "complete" or not signature_matches:
        raise RuntimeError(
            f"Completed inference does not match the current model/config for "
            f"{work.source.filename}; rerun inference with --force"
        )
    rows = [
        InferenceFrame.from_dict(row) for row in _read_jsonl(prediction_path)
    ]
    if len(rows) != int(status.get("frames", -1)):
        raise RuntimeError(
            f"Inference cache row count mismatch for {work.source.filename}"
        )
    return rows


def run_inference(
    config: StrongestPipelineConfig,
    work: VideoWork,
    adapter: StrongestModelAdapter,
    model_provenance: Mapping[str, Any],
    *,
    force: bool = False,
    resume: bool = False,
) -> list[InferenceFrame]:
    prediction_path = work.work_dir / "inference" / "predictions.jsonl"
    status_path = work.work_dir / "inference" / "status.json"
    montage_fingerprint = _file_fingerprint(work.montage_path)
    expected_signature = _inference_signature(config, work, model_provenance)
    if status_path.is_file() and prediction_path.is_file() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("signature") != expected_signature:
            raise RuntimeError(
                f"Inference provenance changed for {work.source.filename}; "
                "rerun with --force"
            )
        if status.get("status") == "complete":
            return [
                InferenceFrame.from_dict(row) for row in _read_jsonl(prediction_path)
            ]
    if force:
        prediction_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    elif prediction_path.is_file() and not resume:
        raise RuntimeError(
            f"Partial inference exists; use --resume or --force: {prediction_path}"
        )

    previous_rows = _read_jsonl(prediction_path) if resume else []
    completed = len(previous_rows)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    montage_video = probe_video(work.montage_path)
    if not montage_video.readable:
        raise RuntimeError(f"Cannot decode montage: {montage_video.error}")
    batch_frames: list[np.ndarray] = []
    batch_metadata: list[dict[str, Any]] = []
    output_rows = list(previous_rows)
    previous_small: np.ndarray | None = None
    previous_segment = -1
    local_scene = 0

    def flush() -> None:
        if not batch_frames:
            return
        predictions = adapter.predict_batch(batch_frames)
        if len(predictions) != len(batch_frames):
            raise RuntimeError(
                f"Model output count mismatch: {len(predictions)} != {len(batch_frames)}"
            )
        for metadata, observations in zip(
            batch_metadata,
            predictions,
            strict=True,
        ):
            row = InferenceFrame(
                **metadata,
                observations=list(observations),
            ).to_dict()
            output_rows.append(row)
            with prediction_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        batch_frames.clear()
        batch_metadata.clear()

    if completed == 0:
        prediction_path.write_text("", encoding="utf-8")
    decoded_count = 0
    for montage_timestamp, sample_index, frame in sample_frames_ffmpeg(
        montage_video,
        config.inference.sample_fps,
    ):
        segment_index, source_timestamp = map_montage_timestamp(
            montage_timestamp,
            work.montage_segments,
        )
        if segment_index != previous_segment:
            previous_small = None
            local_scene = 0
            previous_segment = segment_index
        _, _, _, _, scene_change, current_small = frame_quality(
            frame,
            previous_small,
        )
        previous_small = current_small
        if scene_change >= config.inference.scene_change_threshold:
            local_scene += 1
        decoded_count += 1
        if sample_index < completed:
            continue
        batch_frames.append(frame)
        batch_metadata.append(
            {
                "montage_frame_id": int(
                    round(montage_timestamp * montage_video.fps)
                ),
                "montage_timestamp_sec": float(montage_timestamp),
                "source_segment": int(segment_index),
                "source_timestamp_sec": float(source_timestamp),
                "source_frame_id": int(
                    round(source_timestamp * max(work.source.fps, 1.0))
                ),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "scene_id": int(segment_index * 100000 + local_scene),
                "pixel_sha256": pixel_sha256(frame),
            }
        )
        if len(batch_frames) >= config.inference.pipeline_batch_size:
            flush()
    flush()
    if decoded_count != len(output_rows):
        raise RuntimeError(
            f"Inference row shortfall for {work.source.filename}: "
            f"decoded={decoded_count}, rows={len(output_rows)}"
        )
    status = {
        "schema_version": 1,
        "status": "complete",
        "signature": expected_signature,
        "frames": len(output_rows),
        "sample_fps": config.inference.sample_fps,
        "execution": {
            "pipeline_batch_size": config.inference.pipeline_batch_size,
            "video_workers": config.inference.video_workers,
        },
        "montage": montage_fingerprint,
        "model": dict(model_provenance),
        "coordinate_conventions": {
            "bbox_xyxy": "original montage pixels, [x1,y1,x2,y2]",
            "ellipse": (
                "original montage pixels, [cx,cy,semi_major,semi_minor,"
                "theta_radians], major>=minor, theta modulo pi"
            ),
            "keypoints": "original montage pixels, five fixed facial landmarks",
            "source_mapping": "source_timestamp = segment source_start + montage local time",
        },
    }
    _write_json(status_path, status)
    return [InferenceFrame.from_dict(row) for row in output_rows]


def _ellipse_aabb(
    ellipse: Sequence[float],
) -> tuple[float, float, float, float]:
    cx, cy, major, minor, angle = (float(value) for value in ellipse)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x_extent = math.sqrt((major * cosine) ** 2 + (minor * sine) ** 2)
    y_extent = math.sqrt((major * sine) ** 2 + (minor * cosine) ** 2)
    return (
        cx - x_extent,
        cy - y_extent,
        cx + x_extent,
        cy + y_extent,
    )


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0,
        min(ly2, ry2) - max(ly1, ry1),
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(1e-9, left_area + right_area - intersection)


def _angle_delta(left: float, right: float) -> float:
    return abs((float(right) - float(left) + math.pi / 2) % math.pi - math.pi / 2)


@dataclass(slots=True)
class _EllipseTrack:
    track_id: str
    scope: tuple[int, int]
    observation: FaceObservation
    frame_position: int
    timestamp_sec: float
    hits: int = 1
    missing_positions: list[int] = None  # type: ignore[assignment]
    event_positions: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.missing_positions = []
        self.event_positions = set()


def temporal_ellipse_events(
    frames: Sequence[InferenceFrame],
    config: StrongestPipelineConfig,
) -> tuple[dict[int, float], dict[int, list[str]], dict[int, list[str]]]:
    """Track reliable ellipses without crossing montage segments or scene cuts."""

    scores: dict[int, float] = defaultdict(float)
    reasons: dict[int, list[str]] = defaultdict(list)
    track_ids: dict[int, list[str]] = defaultdict(list)
    active: list[_EllipseTrack] = []
    finished: list[_EllipseTrack] = []
    next_track = 1
    previous_scope: tuple[int, int] | None = None
    max_gap = config.selection.temporal_max_gap_sec

    def finish_all() -> None:
        nonlocal active
        finished.extend(active)
        active = []

    for position, frame in enumerate(frames):
        scope = (frame.source_segment, frame.scene_id)
        if previous_scope is not None and scope != previous_scope:
            finish_all()
        previous_scope = scope
        retained: list[_EllipseTrack] = []
        for track in active:
            if frame.montage_timestamp_sec - track.timestamp_sec > max_gap + 1e-9:
                finished.append(track)
            else:
                retained.append(track)
        active = retained
        observations = [
            row
            for row in frame.observations
            if row.confidence >= config.model.face_confidence
        ]
        costs = np.full(
            (len(active), len(observations)),
            1e6,
            dtype=np.float64,
        )
        for track_index, track in enumerate(active):
            previous = track.observation
            previous_box = _ellipse_aabb(previous.ellipse)
            previous_scale = max(1.0, 2.0 * previous.ellipse[2])
            for observation_index, observation in enumerate(observations):
                current_box = _ellipse_aabb(observation.ellipse)
                center_distance = math.hypot(
                    observation.ellipse[0] - previous.ellipse[0],
                    observation.ellipse[1] - previous.ellipse[1],
                ) / previous_scale
                overlap = _bbox_iou(previous_box, current_box)
                axis_ratio = max(
                    observation.ellipse[2] / max(previous.ellipse[2], 1e-9),
                    previous.ellipse[2] / max(observation.ellipse[2], 1e-9),
                )
                if overlap >= 0.05 or (
                    center_distance <= 2.5 and axis_ratio <= 3.0
                ):
                    costs[track_index, observation_index] = (
                        0.55 * (1.0 - overlap)
                        + 0.35 * min(1.0, center_distance)
                        + 0.10 * min(1.0, math.log(axis_ratio))
                    )
        assigned_tracks: set[int] = set()
        assigned_observations: set[int] = set()
        if costs.size:
            row_indices, column_indices = linear_sum_assignment(costs)
            for track_index, observation_index in zip(
                row_indices.tolist(),
                column_indices.tolist(),
                strict=True,
            ):
                if costs[track_index, observation_index] >= 1e5:
                    continue
                track = active[track_index]
                current = observations[observation_index]
                previous = track.observation
                diameter = max(
                    1.0,
                    2.0 * previous.ellipse[2],
                    2.0 * current.ellipse[2],
                )
                center_jump = math.hypot(
                    current.ellipse[0] - previous.ellipse[0],
                    current.ellipse[1] - previous.ellipse[1],
                ) / diameter
                axis_jump = max(
                    abs(
                        math.log(
                            current.ellipse[2] / max(previous.ellipse[2], 1e-9)
                        )
                    ),
                    abs(
                        math.log(
                            current.ellipse[3] / max(previous.ellipse[3], 1e-9)
                        )
                    ),
                )
                circularity = min(
                    previous.ellipse[2] / max(previous.ellipse[3], 1e-9),
                    current.ellipse[2] / max(current.ellipse[3], 1e-9),
                )
                angle_jump = (
                    _angle_delta(previous.ellipse[4], current.ellipse[4])
                    if circularity >= 1.15
                    else 0.0
                )
                event_score = 0.0
                event_names: list[str] = []
                if center_jump >= config.selection.ellipse_center_jump_ratio:
                    event_names.append("ellipse_center_jump")
                    event_score = max(
                        event_score,
                        min(
                            1.0,
                            center_jump
                            / config.selection.ellipse_center_jump_ratio
                            - 0.25,
                        ),
                    )
                if axis_jump >= config.selection.ellipse_axis_log_jump:
                    event_names.append("ellipse_axis_jump")
                    event_score = max(
                        event_score,
                        min(
                            1.0,
                            axis_jump / config.selection.ellipse_axis_log_jump
                            - 0.25,
                        ),
                    )
                angle_limit = math.radians(
                    config.selection.ellipse_angle_jump_deg
                )
                if angle_jump >= angle_limit:
                    event_names.append("ellipse_angle_jump")
                    event_score = max(
                        event_score,
                        min(1.0, angle_jump / angle_limit - 0.25),
                    )
                if track.missing_positions:
                    for missing_position in track.missing_positions:
                        scores[missing_position] = 1.0
                        reasons[missing_position].append("middle_missing")
                        track_ids[missing_position].append(track.track_id)
                        track.event_positions.add(missing_position)
                    event_names.append("detection_reappeared")
                    event_score = max(event_score, 1.0)
                    track.missing_positions.clear()
                if event_score > 0:
                    scores[position] = max(scores[position], event_score)
                    reasons[position].extend(event_names)
                    track_ids[position].append(track.track_id)
                    track.event_positions.add(position)
                track.observation = current
                track.frame_position = position
                track.timestamp_sec = frame.montage_timestamp_sec
                track.hits += 1
                assigned_tracks.add(track_index)
                assigned_observations.add(observation_index)
        for track_index, track in enumerate(active):
            if track_index not in assigned_tracks:
                track.missing_positions.append(position)
        for observation_index, observation in enumerate(observations):
            if observation_index in assigned_observations:
                continue
            active.append(
                _EllipseTrack(
                    track_id=f"ellipse_track_{next_track:06d}",
                    scope=scope,
                    observation=observation,
                    frame_position=position,
                    timestamp_sec=frame.montage_timestamp_sec,
                )
            )
            next_track += 1
    finish_all()

    valid_event_positions = {
        position
        for track in finished
        if track.hits >= config.selection.temporal_min_track_hits
        for position in track.event_positions
    }
    return (
        {
            position: score
            for position, score in scores.items()
            if position in valid_event_positions
        },
        {
            position: sorted(set(values))
            for position, values in reasons.items()
            if position in valid_event_positions
        },
        {
            position: sorted(set(values))
            for position, values in track_ids.items()
            if position in valid_event_positions
        },
    )


def _low_confidence(
    observations: Sequence[FaceObservation],
    config: StrongestPipelineConfig,
) -> tuple[float, dict[str, Any]]:
    threshold = config.model.face_confidence
    lower = threshold * config.selection.confidence_min_ratio
    upper = threshold * config.selection.confidence_max_ratio
    eligible = [
        row for row in observations if lower <= row.confidence <= upper
    ]
    if not eligible:
        return 0.0, {}
    best = min(
        eligible,
        key=lambda row: (abs(row.confidence / threshold - 1.0), row.confidence),
    )
    maximum_delta = max(
        1.0 - config.selection.confidence_min_ratio,
        config.selection.confidence_max_ratio - 1.0,
    )
    score = 1.0 - abs(best.confidence / threshold - 1.0) / maximum_delta
    return max(1e-6, float(np.clip(score, 0.0, 1.0))), {
        "confidence": best.confidence,
        "threshold": threshold,
        "ratio": best.confidence / threshold,
        "band": [lower, upper],
    }


def _face_without_keypoints(
    observations: Sequence[FaceObservation],
    config: StrongestPipelineConfig,
) -> tuple[float, dict[str, Any]]:
    rows: list[tuple[float, FaceObservation, int]] = []
    for observation in observations:
        if observation.confidence < config.model.face_confidence:
            continue
        valid = sum(
            score >= config.model.keypoint_confidence
            for score in observation.keypoint_scores
        )
        if valid == 0:
            maximum = max(observation.keypoint_scores, default=0.0)
            rows.append((1.0 - maximum, observation, valid))
    if not rows:
        return 0.0, {}
    score, observation, valid = max(rows, key=lambda item: item[0])
    return max(1e-6, float(np.clip(score, 0.0, 1.0))), {
        "face_confidence": observation.confidence,
        "valid_keypoint_count": valid,
        "keypoint_threshold": config.model.keypoint_confidence,
        "keypoint_scores": list(observation.keypoint_scores),
    }


def _extreme_screen_edge(
    observations: Sequence[FaceObservation],
    *,
    width: int,
    height: int,
    config: StrongestPipelineConfig,
) -> tuple[float, dict[str, Any]]:
    margin = config.selection.extreme_edge_margin_ratio
    rows: list[tuple[float, float, FaceObservation, list[float]]] = []
    for observation in observations:
        if observation.confidence < config.model.face_confidence:
            continue
        x1, y1, x2, y2 = _ellipse_aabb(observation.ellipse)
        distances = [
            x1 / max(1.0, width),
            y1 / max(1.0, height),
            (width - x2) / max(1.0, width),
            (height - y2) / max(1.0, height),
        ]
        nearest = min(distances)
        if nearest <= margin:
            score = 1.0 if nearest <= 0 else 1.0 - nearest / max(margin, 1e-9)
            rows.append((max(1e-6, score), nearest, observation, distances))
    if not rows:
        return 0.0, {}
    score, nearest, observation, distances = max(
        rows,
        key=lambda item: (item[0], item[2].confidence),
    )
    return float(np.clip(score, 0.0, 1.0)), {
        "face_confidence": observation.confidence,
        "nearest_ellipse_edge_ratio": nearest,
        "edge_margin_ratio": margin,
        "edge_distances_ratio": distances,
    }


def build_failure_candidates(
    frames: Sequence[InferenceFrame],
    config: StrongestPipelineConfig,
) -> list[FailureCandidate]:
    temporal_scores, temporal_reasons, temporal_tracks = temporal_ellipse_events(
        frames,
        config,
    )
    candidates: list[FailureCandidate] = []
    for position, frame in enumerate(frames):
        low_score, low_detail = _low_confidence(frame.observations, config)
        no_point_score, no_point_detail = _face_without_keypoints(
            frame.observations,
            config,
        )
        edge_score, edge_detail = _extreme_screen_edge(
            frame.observations,
            width=frame.width,
            height=frame.height,
            config=config,
        )
        scores = {
            "temporal_ellipse_instability": temporal_scores.get(position, 0.0),
            "low_confidence": low_score,
            "face_without_keypoints": no_point_score,
            "extreme_screen_edge": edge_score,
        }
        if not any(score > 0 for score in scores.values()):
            continue
        candidates.append(
            FailureCandidate(
                frame=frame,
                scores=scores,
                details={
                    "temporal_ellipse_instability": {
                        "events": temporal_reasons.get(position, []),
                        "track_ids": temporal_tracks.get(position, []),
                    },
                    "low_confidence": low_detail,
                    "face_without_keypoints": no_point_detail,
                    "extreme_screen_edge": edge_detail,
                },
            )
        )

    # Exact decoded duplicates can occur around edits or frozen shots. Keep
    # the copy with the strongest failure evidence.
    by_hash: dict[str, FailureCandidate] = {}
    without_hash: list[FailureCandidate] = []
    for row in candidates:
        if not row.frame.pixel_sha256:
            without_hash.append(row)
            continue
        previous = by_hash.get(row.frame.pixel_sha256)
        if previous is None or (
            max(row.scores.values()),
            -row.frame.source_timestamp_sec,
        ) > (
            max(previous.scores.values()),
            -previous.frame.source_timestamp_sec,
        ):
            by_hash[row.frame.pixel_sha256] = row
    return sorted(
        [*by_hash.values(), *without_hash],
        key=lambda row: row.frame.source_timestamp_sec,
    )


def quota_counts(target: int, ratios: Mapping[str, float]) -> dict[str, int]:
    total = sum(float(value) for value in ratios.values())
    raw = {name: target * float(value) / total for name, value in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(
        ratios,
        key=lambda name: (
            raw[name] - counts[name],
            -list(ratios).index(name),
        ),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def maximum_spaced_candidate_count(
    candidates: Sequence[FailureCandidate],
    *,
    min_interval_sec: float,
    category: str | None = None,
) -> int:
    """Return the exact maximum cardinality for one fixed temporal spacing.

    All exclusion windows have the same length, so earliest-finish greedy is
    optimal. A category filter still counts candidates that may also match
    other categories.
    """

    rows = [
        row
        for row in candidates
        if category is None or row.scores.get(category, 0.0) > 0.0
    ]
    rows.sort(key=lambda row: row.frame.source_timestamp_sec)
    selected = 0
    previous_timestamp = float("-inf")
    for row in rows:
        timestamp = row.frame.source_timestamp_sec
        if timestamp - previous_timestamp + 1e-9 < min_interval_sec:
            continue
        selected += 1
        previous_timestamp = timestamp
    return selected


def select_failure_candidates(
    candidates: Sequence[FailureCandidate],
    config: StrongestPipelineConfig,
) -> tuple[list[FailureCandidate], dict[str, Any]]:
    target = min(config.selection.max_frames_per_video, len(candidates))
    quotas = quota_counts(target, config.selection.category_ratios)
    selected: list[FailureCandidate] = []
    selected_ids: set[int] = set()

    def separated(row: FailureCandidate) -> bool:
        return all(
            abs(
                row.frame.source_timestamp_sec
                - previous.frame.source_timestamp_sec
            )
            >= config.selection.min_interval_sec
            for previous in selected
        )

    eligible = {
        category: [row for row in candidates if row.scores[category] > 0]
        for category in CATEGORIES
    }
    # Allocate scarce conditions first so an overlap candidate does not get
    # consumed by an abundant bucket before its 25% target is attempted.
    category_order = sorted(
        CATEGORIES,
        key=lambda category: (len(eligible[category]), CATEGORIES.index(category)),
    )
    for category in category_order:
        ranked = sorted(
            eligible[category],
            key=lambda row: (
                row.scores[category],
                max(row.scores.values()),
                -row.frame.source_timestamp_sec,
            ),
            reverse=True,
        )
        for row in ranked:
            if sum(
                item.assigned_category == category for item in selected
            ) >= quotas[category]:
                break
            if id(row) in selected_ids or not separated(row):
                continue
            selected.append(
                replace(
                    row,
                    assigned_category=category,
                    selection_score=row.scores[category],
                    selection_note=f"25% quota: ranked by {category}",
                )
            )
            selected_ids.add(id(row))

    # Explicitly permitted quota refill: only frames satisfying at least one
    # of the four requested conditions are eligible.
    if len(selected) < target:
        refill = sorted(
            (row for row in candidates if id(row) not in selected_ids),
            key=lambda row: (
                max(row.scores.values()),
                sum(score > 0 for score in row.scores.values()),
                -row.frame.source_timestamp_sec,
            ),
            reverse=True,
        )
        for row in refill:
            if len(selected) >= target:
                break
            if not separated(row):
                continue
            category = max(
                CATEGORIES,
                key=lambda name: (row.scores[name], -CATEGORIES.index(name)),
            )
            selected.append(
                replace(
                    row,
                    assigned_category=category,
                    selection_score=row.scores[category],
                    selection_note="quota shortfall refill from another requested condition",
                )
            )
            selected_ids.add(id(row))

    selected.sort(key=lambda row: row.frame.source_timestamp_sec)
    actual = Counter(row.assigned_category for row in selected)
    raw_by_category = {
        category: sum(row.scores[category] > 0 for row in candidates)
        for category in CATEGORIES
    }
    spaced_by_category = {
        category: maximum_spaced_candidate_count(
            candidates,
            min_interval_sec=config.selection.min_interval_sec,
            category=category,
        )
        for category in CATEGORIES
    }
    summary = {
        "candidate_count": len(candidates),
        "candidate_count_by_category": raw_by_category,
        "maximum_after_exact_dedup_and_min_interval": (
            maximum_spaced_candidate_count(
                candidates,
                min_interval_sec=config.selection.min_interval_sec,
            )
        ),
        "maximum_by_category_after_min_interval": spaced_by_category,
        "target": config.selection.max_frames_per_video,
        "effective_target": target,
        "selected": len(selected),
        "quota": quotas,
        "actual": {category: int(actual[category]) for category in CATEGORIES},
        "quota_shortfall": {
            category: max(0, quotas[category] - int(actual[category]))
            for category in CATEGORIES
        },
        "refill_count": sum(
            row.selection_note.startswith("quota shortfall") for row in selected
        ),
        "total_shortfall": max(0, config.selection.max_frames_per_video - len(selected)),
        "min_interval_sec": config.selection.min_interval_sec,
    }
    return selected, summary


def _candidate_to_dict(row: FailureCandidate) -> dict[str, Any]:
    return {
        "frame": row.frame.to_dict(),
        "scores": row.scores,
        "details": row.details,
        "assigned_category": row.assigned_category,
        "selection_score": row.selection_score,
        "selection_note": row.selection_note,
    }


def _candidate_from_dict(row: Mapping[str, Any]) -> FailureCandidate:
    return FailureCandidate(
        frame=InferenceFrame.from_dict(dict(row["frame"])),
        scores={
            str(name): float(value)
            for name, value in dict(row["scores"]).items()
        },
        details=dict(row.get("details", {})),
        assigned_category=str(row.get("assigned_category", "")),
        selection_score=float(row.get("selection_score", 0.0)),
        selection_note=str(row.get("selection_note", "")),
    )


def write_selection_cache(
    config: StrongestPipelineConfig,
    work: VideoWork,
    selected: Sequence[FailureCandidate],
    summary: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
) -> None:
    selection_dir = work.work_dir / "selection"
    inference_path = work.work_dir / "inference" / "predictions.jsonl"
    status = {
        "schema_version": 1,
        "signature": _signature(
            {
                "inference": _file_fingerprint(inference_path),
                "selection": asdict(config.selection),
                "model": _model_provenance_signature(model_provenance),
                "seed": config.random_seed,
            }
        ),
        "summary": dict(summary),
    }
    _write_jsonl(
        selection_dir / "selected_candidates.jsonl",
        (_candidate_to_dict(row) for row in selected),
    )
    _write_json(selection_dir / "selection_status.json", status)


def read_selection_cache(
    config: StrongestPipelineConfig,
    work: VideoWork,
    model_provenance: Mapping[str, Any],
) -> tuple[list[FailureCandidate], dict[str, Any]]:
    selection_dir = work.work_dir / "selection"
    selected_path = selection_dir / "selected_candidates.jsonl"
    status_path = selection_dir / "selection_status.json"
    inference_path = work.work_dir / "inference" / "predictions.jsonl"
    if not selected_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(
            f"Run selection first for {work.source.filename}: {selected_path}"
        )
    expected = _signature(
        {
            "inference": _file_fingerprint(inference_path),
            "selection": asdict(config.selection),
            "model": _model_provenance_signature(model_provenance),
            "seed": config.random_seed,
        }
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("signature") != expected:
        raise RuntimeError(
            f"Selection cache does not match current inference/config for "
            f"{work.source.filename}; rerun selection"
        )
    return (
        [_candidate_from_dict(row) for row in _read_jsonl(selected_path)],
        dict(status.get("summary", {})),
    )


def _draw_evidence(
    frame: np.ndarray,
    candidate: FailureCandidate,
    config: StrongestPipelineConfig,
) -> np.ndarray:
    canvas = frame.copy()
    for observation in candidate.frame.observations:
        color = (
            (60, 220, 60)
            if observation.confidence >= config.model.face_confidence
            else (0, 165, 255)
        )
        cx, cy, major, minor, angle = observation.ellipse
        cv2.ellipse(
            canvas,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(major))), max(1, int(round(minor)))),
            math.degrees(angle),
            0,
            360,
            color,
            2,
            cv2.LINE_AA,
        )
        for point, score in zip(
            observation.keypoints,
            observation.keypoint_scores,
            strict=False,
        ):
            point_color = (
                (255, 220, 40)
                if score >= config.model.keypoint_confidence
                else (80, 80, 255)
            )
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                3,
                point_color,
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"face={observation.confidence:.3f}",
            (max(0, int(observation.bbox_xyxy[0])), max(18, int(observation.bbox_xyxy[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    lines = [
        candidate.assigned_category,
        f"score={candidate.selection_score:.3f}",
        "matched=" + " / ".join(candidate.matched_categories),
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 82), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


MANIFEST_FIELDS = (
    "frame_id",
    "source_video",
    "source_frame_id",
    "source_timestamp_sec",
    "montage_frame_id",
    "montage_timestamp_sec",
    "source_segment",
    "frame_path",
    "evidence_path",
    "primary_reason",
    "primary_reason_ja",
    "matched_reasons",
    "matched_reasons_ja",
    "selection_score",
    "selection_note",
    "scores_json",
    "details_json",
    "predictions_json",
    "model_id",
    "random_seed",
)


def export_selected_frames(
    config: StrongestPipelineConfig,
    work: VideoWork,
    selected: Sequence[FailureCandidate],
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    frame_dir = config.output_dir / "frames"
    evidence_dir = config.output_dir / "evidence"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if config.output.write_evidence_overlay:
        evidence_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    exported_hashes: set[str] = set()
    safe_stem = "_".join(
        (safe_name(Path(work.source.filename).stem) or "video").split()
    )[:100]
    if force:
        for stale_path in frame_dir.glob(f"{safe_stem}__frame_*.jpg"):
            stale_path.unlink()
        if config.output.write_evidence_overlay:
            for stale_path in evidence_dir.glob(
                f"{safe_stem}__frame_*__evidence.jpg"
            ):
                stale_path.unlink()
    for candidate in selected:
        frame, actual_frame_id, actual_timestamp = read_video_frame(
            Path(work.source.path),
            timestamp_sec=candidate.frame.source_timestamp_sec,
            frame_index=-1,
        )
        exported_hash = pixel_sha256(frame)
        if exported_hash in exported_hashes:
            continue
        exported_hashes.add(exported_hash)
        frame_id = f"{safe_stem}__frame_{actual_frame_id:010d}"
        frame_path = frame_dir / f"{frame_id}.jpg"
        evidence_path = evidence_dir / f"{frame_id}__evidence.jpg"
        if force or not frame_path.is_file():
            if not cv2.imwrite(
                frame_path.as_posix(),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.output.jpeg_quality],
            ):
                raise RuntimeError(f"Cannot write selected frame: {frame_path}")
        if config.output.write_evidence_overlay and (
            force or not evidence_path.is_file()
        ):
            evidence = _draw_evidence(frame, candidate, config)
            if not cv2.imwrite(
                evidence_path.as_posix(),
                evidence,
                [cv2.IMWRITE_JPEG_QUALITY, config.output.jpeg_quality],
            ):
                raise RuntimeError(f"Cannot write evidence frame: {evidence_path}")
        rows.append(
            {
                "frame_id": frame_id,
                "source_video": work.source.filename,
                "source_frame_id": actual_frame_id,
                "source_timestamp_sec": f"{actual_timestamp:.6f}",
                "montage_frame_id": candidate.frame.montage_frame_id,
                "montage_timestamp_sec": f"{candidate.frame.montage_timestamp_sec:.6f}",
                "source_segment": candidate.frame.source_segment,
                "frame_path": frame_path.resolve().as_posix(),
                "evidence_path": (
                    evidence_path.resolve().as_posix()
                    if config.output.write_evidence_overlay
                    else ""
                ),
                "primary_reason": candidate.assigned_category,
                "primary_reason_ja": CATEGORY_LABELS[candidate.assigned_category],
                "matched_reasons": ";".join(candidate.matched_categories),
                "matched_reasons_ja": "；".join(
                    CATEGORY_LABELS[name] for name in candidate.matched_categories
                ),
                "selection_score": f"{candidate.selection_score:.8f}",
                "selection_note": candidate.selection_note,
                "scores_json": json.dumps(
                    candidate.scores,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "details_json": json.dumps(
                    candidate.details,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "predictions_json": json.dumps(
                    [
                        observation.to_dict()
                        for observation in candidate.frame.observations
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "model_id": config.model.model_id,
                "random_seed": config.random_seed,
            }
        )
    per_video_dir = work.work_dir / "selection"
    _write_csv(per_video_dir / "selection_manifest.csv", rows, MANIFEST_FIELDS)
    _write_jsonl(per_video_dir / "selection_manifest.jsonl", rows)
    return rows


def _relative(path: str, report_path: Path) -> str:
    return Path(path).resolve().relative_to(report_path.parent.resolve()).as_posix()


def write_report(
    config: StrongestPipelineConfig,
    manifest: Sequence[Mapping[str, Any]],
    video_summaries: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
) -> None:
    report_path = config.output_dir / "selection_report.html"
    cards: list[str] = []
    for row in manifest:
        frame_src = _relative(str(row["frame_path"]), report_path)
        evidence_src = (
            _relative(str(row["evidence_path"]), report_path)
            if row.get("evidence_path")
            else ""
        )
        evidence_html = (
            f'<img loading="lazy" src="{html.escape(evidence_src)}" alt="evidence">'
            if evidence_src
            else ""
        )
        cards.append(
            f"""
<article>
  <h2>{html.escape(str(row['frame_id']))}</h2>
  <div class="images">
    <figure><img loading="lazy" src="{html.escape(frame_src)}" alt="clean frame"><figcaption>対象frame（原画像）</figcaption></figure>
    <figure>{evidence_html}<figcaption>選択根拠overlay</figcaption></figure>
  </div>
  <p><b>主理由:</b> {html.escape(str(row['primary_reason_ja']))}</p>
  <p><b>該当した全条件:</b> {html.escape(str(row['matched_reasons_ja']))}</p>
  <p><b>元動画:</b> {html.escape(str(row['source_video']))} /
     frame {row['source_frame_id']} / {row['source_timestamp_sec']} sec</p>
  <details><summary>スコア・判定詳細</summary>
    <pre>{html.escape(str(row['scores_json']))}</pre>
    <pre>{html.escape(str(row['details_json']))}</pre>
  </details>
</article>"""
        )
    counts = Counter(str(row["primary_reason"]) for row in manifest)
    summary_rows = "".join(
        f"<tr><td>{html.escape(CATEGORY_LABELS[category])}</td>"
        f"<td>{counts[category]}</td></tr>"
        for category in CATEGORIES
    )
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Active Learning 選択資料</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#202124}}
header,article{{background:white;border:1px solid #ddd;border-radius:10px;padding:16px;margin:0 0 18px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
figure{{margin:0}}img{{display:block;width:100%;height:auto;background:#111}}
figcaption{{font-size:.9rem;color:#555;margin-top:4px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body>
<header><h1>Active Learning 選択資料</h1>
<p>model: {html.escape(config.model.model_id)} / selected: {len(manifest)}</p>
<p>10等分した元動画の各区間中央60秒を連結した10分montageを5fpsで推論。
4条件を25%ずつ割り当て、不足枠は他の3条件を満たすframeで補完。
選択frameの元動画時刻は最低{config.selection.min_interval_sec:g}秒空ける。</p>
<table><thead><tr><th>選択理由</th><th>件数</th></tr></thead><tbody>{summary_rows}</tbody></table>
</header>
{''.join(cards)}
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "selected_frames": len(manifest),
        "by_primary_reason": {
            category: int(counts[category]) for category in CATEGORIES
        },
        "max_frames_per_video": config.selection.max_frames_per_video,
        "category_ratios": config.selection.category_ratios,
        "video_summaries": dict(video_summaries),
        "model": dict(model_provenance),
        "random_seed": config.random_seed,
        "selection_thresholds": {
            "face_confidence": config.model.face_confidence,
            "keypoint_confidence": config.model.keypoint_confidence,
            "min_interval_sec": config.selection.min_interval_sec,
            "confidence_ratio_band": [
                config.selection.confidence_min_ratio,
                config.selection.confidence_max_ratio,
            ],
            "extreme_screen_edge_margin_ratio": (
                config.selection.extreme_edge_margin_ratio
            ),
        },
        "selection_policy": (
            "25% x 4 per source video; category shortfalls may be refilled only "
            "by frames satisfying another requested category"
        ),
        "report": report_path.resolve().as_posix(),
    }
    _write_json(config.output_dir / "summary.json", summary)


def prepare_run_metadata(
    config: StrongestPipelineConfig,
    model_provenance: Mapping[str, Any],
) -> None:
    internal = config.output_dir / "_active_learning"
    internal.mkdir(parents=True, exist_ok=True)
    (internal / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            config.resolved_dict(),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    _write_json(
        internal / "run_provenance.json",
        {
            "schema_version": 1,
            "config": config.resolved_dict(),
            "model": dict(model_provenance),
            "coordinate_conventions": {
                "image_origin": "top-left",
                "x_axis": "right",
                "y_axis": "down",
                "angle": "radians modulo pi in image coordinates",
                "ellipse_axes": "semi-axis radii, major >= minor",
            },
            "random_seed": config.random_seed,
        },
    )


def aggregate_manifests(
    config: StrongestPipelineConfig,
    per_video_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    manifest = [dict(row) for rows in per_video_rows for row in rows]
    manifest.sort(
        key=lambda row: (
            str(row["source_video"]),
            int(row["source_frame_id"]),
        )
    )
    _write_csv(
        config.output_dir / "selection_manifest.csv",
        manifest,
        MANIFEST_FIELDS,
    )
    _write_jsonl(config.output_dir / "selection_manifest.jsonl", manifest)
    return manifest
