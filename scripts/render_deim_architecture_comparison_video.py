#!/usr/bin/env python3
"""Resumable 2x2 video comparison for four DEIM face architectures.

The inference stages are deliberately separate from rendering:

1. normal DEIMv2-L bounding boxes
2. DEIMv2-L ellipse detections
3. Query Refinement landmarks from the cached ellipse detections
4. ORFormer landmarks from the same cached ellipse detections

No inference is performed by ``--dry-run``. Each completed stage has a status
record containing checkpoint/config hashes and can be safely reused.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("deim_architecture_video")
MODEL_ROLES = ("normal_deim", "ellipse_deim", "query_refinement", "orformer")
LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)
LANDMARK_COLORS = (
    (255, 220, 40),
    (40, 150, 255),
    (40, 240, 240),
    (70, 220, 80),
    (230, 80, 230),
)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
TIMING_KEYS = (
    "decode_ms",
    "preprocess_cpu_ms",
    "host_to_device_ms",
    "model_gpu_ms",
    "postprocess_gpu_ms",
    "device_to_host_ms",
    "postprocess_cpu_ms",
    "render_ms",
    "video_write_ms",
    "total_ms",
)
DETECTOR_ROLES = ("normal_deim", "ellipse_deim")
DETECTOR_BATCH_TUNING_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    has_audio: bool


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 2560
    height: int = 1440
    codec: str = "libx264"
    crf: int = 20
    preset: str = "fast"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    landmark_radius: int = 5
    line_width: int = 3


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    path: Path
    project_root: Path
    requested_input: str
    input_path: Path
    recursive: bool
    extensions: tuple[str, ...]
    output_root: Path
    seed: int
    models: dict[str, dict[str, Any]]
    render: RenderConfig


@dataclass(frozen=True, slots=True)
class DetectorBatchSelection:
    requested: str | int
    selected_batch_size: int
    maximum_safe_batch_size: int
    selection_strategy: str
    tuning_path: Path | None
    source: str


class TimingAccumulator:
    def __init__(self, *, scope: str, frame_start: int) -> None:
        self.scope = scope
        self.frame_start = int(frame_start)
        self.frame_end = int(frame_start) - 1
        self.frames = 0
        self.active_frames = 0
        self.totals = {key: 0.0 for key in TIMING_KEYS}

    def add(self, frame_index: int, values: Mapping[str, float], *, active: bool) -> None:
        self.frame_end = int(frame_index)
        self.frames += 1
        self.active_frames += int(active)
        for key in TIMING_KEYS:
            self.totals[key] += float(values.get(key, 0.0))

    def payload(self) -> dict[str, Any]:
        denominator = max(1, self.frames)
        active_denominator = max(1, self.active_frames)
        return {
            "schema_version": 1,
            "scope": self.scope,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frames": self.frames,
            "active_frames": self.active_frames,
            "units": "milliseconds",
            "totals_ms": dict(self.totals),
            "mean_per_source_frame_ms": {
                key: value / denominator for key, value in self.totals.items()
            },
            "mean_per_active_frame_ms": {
                key: value / active_denominator for key, value in self.totals.items()
            },
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/video_comparison/deim_architecture_four_panel.yaml",
    )
    parser.add_argument("--input", type=Path, help="Override input file/directory")
    parser.add_argument("--output-dir", type=Path, help="Override configured output root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage",
        choices=("infer", "render", "all"),
        default="all",
        help="Run inference only, render cached JSONL only, or both",
    )
    parser.add_argument("--resume", action="store_true", help="Resume partial stage JSONL files")
    parser.add_argument("--force", action="store_true", help="Replace completed/partial stage outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the plan without opening video frames or loading models",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _detector_batch_candidates(model: Mapping[str, Any]) -> tuple[str | int, tuple[int, ...]]:
    requested = model.get("inference_batch_size", 1)
    if isinstance(requested, str) and requested.lower() == "auto":
        raw_candidates = model.get("batch_candidates", (1, 2, 4, 8, 16, 32))
        if (
            not isinstance(raw_candidates, Sequence)
            or isinstance(raw_candidates, (str, bytes))
        ):
            raise ValueError("batch_candidates must be a sequence of positive integers")
        try:
            candidates = tuple(sorted({int(value) for value in raw_candidates}))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "batch_candidates must be a sequence of positive integers"
            ) from error
        if not candidates or candidates[0] <= 0:
            raise ValueError("batch_candidates must contain positive integers")
        if candidates[0] != 1:
            raise ValueError("batch_candidates must include 1 as the safe fallback")
        return "auto", candidates
    try:
        fixed = int(requested)
    except (TypeError, ValueError) as error:
        raise ValueError("inference_batch_size must be 'auto' or a positive integer") from error
    if fixed <= 0:
        raise ValueError("inference_batch_size must be 'auto' or a positive integer")
    return fixed, (fixed,)


def load_config(path: Path) -> ComparisonConfig:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Video comparison config must be a schema_version=1 mapping")
    root = _resolve(raw.get("project_root", "../.."), config_path.parent)
    input_cfg = dict(raw.get("input", {}))
    output_cfg = dict(raw.get("output", {}))
    raw_models = raw.get("models")
    if not isinstance(raw_models, dict) or tuple(raw_models) != MODEL_ROLES:
        raise ValueError(f"models must appear in this exact order: {MODEL_ROLES}")

    models: dict[str, dict[str, Any]] = {}
    for role in MODEL_ROLES:
        model = deepcopy(dict(raw_models[role]))
        for key in ("config", "checkpoint", "threshold_source"):
            if model.get(key):
                model[key] = _resolve(model[key], root)
        models[role] = model
    if models["query_refinement"].get("detector_source") != "ellipse_deim":
        raise ValueError("query_refinement.detector_source must be ellipse_deim")
    if models["orformer"].get("detector_source") != "ellipse_deim":
        raise ValueError("orformer.detector_source must be ellipse_deim")
    if str(models["orformer"].get("model_name")) != "dinov3_orformer_rtmcc":
        raise ValueError("orformer.model_name must be dinov3_orformer_rtmcc")
    for role, model in models.items():
        if int(model.get("input_size", 0)) <= 0:
            raise ValueError(f"{role}.input_size must be positive")
        if float(model.get("score_threshold", 0.1)) <= 0:
            raise ValueError(f"{role}.score_threshold must be positive")
        if role in DETECTOR_ROLES:
            _detector_batch_candidates(model)
            strategy = str(model.get("auto_batch_selection", "throughput"))
            if strategy not in {"throughput", "largest_safe"}:
                raise ValueError(
                    f"{role}.auto_batch_selection must be throughput or largest_safe"
                )
            headroom = float(model.get("gpu_memory_headroom_ratio", 0.15))
            if not 0.0 <= headroom < 1.0:
                raise ValueError(
                    f"{role}.gpu_memory_headroom_ratio must be in [0, 1)"
                )
            if int(model.get("auto_batch_warmup_iterations", 1)) < 0:
                raise ValueError(
                    f"{role}.auto_batch_warmup_iterations must be non-negative"
                )
            if int(model.get("auto_batch_benchmark_iterations", 2)) <= 0:
                raise ValueError(
                    f"{role}.auto_batch_benchmark_iterations must be positive"
                )
        if role in {"query_refinement", "orformer"}:
            if float(model.get("crop_scale", 0.0)) != 1.25:
                raise ValueError(f"{role}.crop_scale must remain 1.25")
            if not 0 <= float(model.get("visibility_threshold", -1)) <= 1:
                raise ValueError(f"{role}.visibility_threshold must be in [0, 1]")

    render = RenderConfig(**dict(raw.get("render", {})))
    if render.width <= 0 or render.height <= 0 or render.width % 2 or render.height % 2:
        raise ValueError("render width/height must be positive even numbers")
    extensions = tuple(
        value.lower() if str(value).startswith(".") else f".{str(value).lower()}"
        for value in input_cfg.get("extensions", VIDEO_EXTENSIONS)
    )
    return ComparisonConfig(
        path=config_path,
        project_root=root,
        requested_input=str(input_cfg.get("requested_path", input_cfg.get("path", ""))),
        input_path=_resolve(input_cfg["path"], root),
        recursive=bool(input_cfg.get("recursive", False)),
        extensions=extensions,
        output_root=_resolve(output_cfg["root"], root),
        seed=int(raw.get("seed", 0)),
        models=models,
        render=render,
    )


def validate_model_assets(config: ComparisonConfig) -> None:
    for role, model in config.models.items():
        for key in ("config", "checkpoint"):
            path = model.get(key)
            if not isinstance(path, Path) or not path.is_file():
                raise FileNotFoundError(f"Missing {role}.{key}: {path}")
        source = model.get("threshold_source")
        if source is not None and (not isinstance(source, Path) or not source.is_file()):
            raise FileNotFoundError(f"Missing {role}.threshold_source: {source}")


def discover_videos(
    path: Path,
    *,
    recursive: bool,
    extensions: Sequence[str],
) -> list[Path]:
    resolved = path.expanduser().resolve()
    allowed = {value.lower() for value in extensions}
    if resolved.is_file():
        if resolved.suffix.lower() not in allowed:
            raise ValueError(f"Unsupported video extension: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input video path does not exist: {resolved}")
    iterator: Iterable[Path] = resolved.rglob("*") if recursive else resolved.iterdir()
    videos = sorted(
        (candidate.resolve() for candidate in iterator if candidate.is_file() and candidate.suffix.lower() in allowed),
        key=lambda value: value.as_posix(),
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found below: {resolved}")
    return videos


def _slug(text: str) -> str:
    result = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in result.split("_") if part) or "video"


def video_output_dir(root: Path, video: Path) -> Path:
    digest = hashlib.sha256(video.resolve().as_posix().encode("utf-8")).hexdigest()[:8]
    return root / f"{_slug(video.stem)}_{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> VideoMetadata:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[:4000]}")
    payload = json.loads(result.stdout)
    video = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"Input has no video stream: {path}")

    def ratio(value: str) -> float:
        numerator, separator, denominator = str(value).partition("/")
        return float(numerator) / float(denominator) if separator else float(value)

    fps = ratio(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    frame_count = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    metadata = VideoMetadata(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        has_audio=any(row.get("codec_type") == "audio" for row in payload.get("streams", [])),
    )
    if min(metadata.width, metadata.height, metadata.frame_count) <= 0 or metadata.fps <= 0:
        raise RuntimeError(f"Invalid video metadata: {metadata}")
    return metadata


def _last_jsonl_frame(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return -1
    last = ""
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = line
    try:
        return int(json.loads(last)["frame_index"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid final JSONL row; use --force: {path}") from error


def _stage_paths(output_dir: Path, role: str) -> tuple[Path, Path]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return prediction_dir / f"{role}.jsonl", prediction_dir / f"{role}.status.json"


def _stage_signature(
    *,
    video_sha256: str,
    model: Mapping[str, Any],
    dependency_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(model["checkpoint"])
    config = Path(model["config"])
    signature: dict[str, Any] = {
        "video_sha256": video_sha256,
        "config_sha256": sha256_file(config),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    if "score_threshold" in model:
        signature["score_threshold"] = float(model["score_threshold"])
    if dependency_sha256 is not None:
        signature["ellipse_predictions_sha256"] = dependency_sha256
    return signature


def _status_reusable(
    prediction_path: Path,
    status_path: Path,
    *,
    expected_frames: int,
    signature: Mapping[str, Any],
) -> bool:
    if not prediction_path.is_file() or not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return (
        status.get("status") == "complete"
        and int(status.get("frames", -1)) == expected_frames
        and status.get("signature") == dict(signature)
        and _last_jsonl_frame(prediction_path) == expected_frames - 1
    )


def _release_cuda(value: Any) -> None:
    candidates = [
        getattr(value, "model", None),
        getattr(getattr(value, "pipeline", None), "pose_model", None),
    ]
    for model in candidates:
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_cuda_oom(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _detector_batch_tuning_signature(
    detector: Any,
    model_cfg: Mapping[str, Any],
    *,
    candidates: Sequence[int],
) -> dict[str, Any]:
    import torch

    device = detector.device
    cuda_device_name = None
    cuda_capability = None
    total_memory_bytes = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        cuda_capability = list(torch.cuda.get_device_capability(device))
        total_memory_bytes = int(
            torch.cuda.get_device_properties(device).total_memory
        )
    return {
        "schema_version": DETECTOR_BATCH_TUNING_SCHEMA_VERSION,
        "config_sha256": sha256_file(Path(model_cfg["config"])),
        "checkpoint_sha256": sha256_file(Path(model_cfg["checkpoint"])),
        "input_size": int(model_cfg["input_size"]),
        "device": str(device),
        "cuda_device_name": cuda_device_name,
        "cuda_capability": cuda_capability,
        "cuda_runtime": torch.version.cuda,
        "total_memory_bytes": total_memory_bytes,
        "autocast": device.type == "cuda",
        "batch_candidates": [int(value) for value in candidates],
        "gpu_memory_headroom_ratio": float(
            model_cfg.get("gpu_memory_headroom_ratio", 0.15)
        ),
        "warmup_iterations": int(
            model_cfg.get("auto_batch_warmup_iterations", 1)
        ),
        "benchmark_iterations": int(
            model_cfg.get("auto_batch_benchmark_iterations", 2)
        ),
        "selection_strategy": str(
            model_cfg.get("auto_batch_selection", "throughput")
        ),
    }


def _read_detector_tuning_frames(
    video: Path,
    *,
    start_frame: int,
    count: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(video.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for DEIM batch tuning: {video}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < count:
            okay, frame_bgr = capture.read()
            if not okay or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _benchmark_detector_batch(
    detector: Any,
    images: Sequence[np.ndarray],
    *,
    warmup_iterations: int,
    benchmark_iterations: int,
    gpu_memory_headroom_ratio: float,
) -> dict[str, Any]:
    import torch

    device = detector.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        detector.predict_batch(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(benchmark_iterations):
        detector.predict_batch(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    peak_allocated_bytes = None
    peak_reserved_bytes = None
    total_memory_bytes = None
    memory_within_headroom = True
    if device.type == "cuda":
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        total_memory_bytes = int(
            torch.cuda.get_device_properties(device).total_memory
        )
        memory_limit = total_memory_bytes * (1.0 - gpu_memory_headroom_ratio)
        memory_within_headroom = peak_reserved_bytes <= memory_limit
    return {
        "batch_size": len(images),
        "iterations": int(benchmark_iterations),
        "elapsed_seconds": elapsed_seconds,
        "frames_per_second": (
            len(images) * benchmark_iterations / max(elapsed_seconds, 1e-9)
        ),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_memory_bytes": total_memory_bytes,
        "memory_within_headroom": memory_within_headroom,
    }


def _select_detector_batch_attempt(
    attempts: Sequence[Mapping[str, Any]],
    *,
    strategy: str,
) -> tuple[Mapping[str, Any], int]:
    safe = [row for row in attempts if row.get("status") == "ok"]
    if not safe:
        raise RuntimeError("DEIM auto batch tuning found no usable batch size")
    maximum_safe = max(int(row["batch_size"]) for row in safe)
    if strategy == "largest_safe":
        selected = max(safe, key=lambda row: int(row["batch_size"]))
    elif strategy == "throughput":
        selected = max(
            safe,
            key=lambda row: (
                float(row["frames_per_second"]),
                int(row["batch_size"]),
            ),
        )
    else:
        raise ValueError(f"Unsupported detector batch selection strategy: {strategy}")
    return selected, maximum_safe


def _auto_tune_detector_batch(
    detector: Any,
    *,
    video: Path,
    start_frame: int,
    remaining_frames: int,
    output_dir: Path,
    role: str,
    model_cfg: Mapping[str, Any],
    candidates: Sequence[int],
    force: bool,
) -> DetectorBatchSelection:
    tuning_path = output_dir / "logs" / f"{role}_batch_tuning.json"
    signature = _detector_batch_tuning_signature(
        detector,
        model_cfg,
        candidates=candidates,
    )
    if tuning_path.is_file() and not force:
        cached = json.loads(tuning_path.read_text(encoding="utf-8"))
        selected = int(cached.get("selected_batch_size", 0))
        maximum_safe = int(cached.get("maximum_safe_batch_size", 0))
        if (
            cached.get("signature") == signature
            and selected > 0
            and maximum_safe >= selected
        ):
            LOGGER.info(
                "[%s] reusing tuned detector batch=%d (maximum safe=%d)",
                role,
                selected,
                maximum_safe,
            )
            return DetectorBatchSelection(
                requested="auto",
                selected_batch_size=selected,
                maximum_safe_batch_size=maximum_safe,
                selection_strategy=str(
                    model_cfg.get("auto_batch_selection", "throughput")
                ),
                tuning_path=tuning_path,
                source="cache",
            )

    usable_candidates = [
        int(value) for value in candidates if int(value) <= remaining_frames
    ]
    if not usable_candidates and remaining_frames > 0:
        usable_candidates = [1]
    if not usable_candidates:
        return DetectorBatchSelection(
            requested="auto",
            selected_batch_size=1,
            maximum_safe_batch_size=1,
            selection_strategy=str(
                model_cfg.get("auto_batch_selection", "throughput")
            ),
            tuning_path=None,
            source="no_remaining_frames",
        )
    samples = _read_detector_tuning_frames(
        video,
        start_frame=start_frame,
        count=max(usable_candidates),
    )
    if not samples:
        raise RuntimeError(
            f"No frames available for {role} batch tuning at frame {start_frame}"
        )
    usable_candidates = [
        value for value in usable_candidates if value <= len(samples)
    ]
    warmup_iterations = int(
        model_cfg.get("auto_batch_warmup_iterations", 1)
    )
    benchmark_iterations = int(
        model_cfg.get("auto_batch_benchmark_iterations", 2)
    )
    headroom_ratio = float(
        model_cfg.get("gpu_memory_headroom_ratio", 0.15)
    )
    attempts: list[dict[str, Any]] = []
    for candidate in usable_candidates:
        LOGGER.info("[%s] probing detector batch=%d", role, candidate)
        try:
            result = _benchmark_detector_batch(
                detector,
                samples[:candidate],
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                gpu_memory_headroom_ratio=headroom_ratio,
            )
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            _clear_cuda_cache()
            attempts.append(
                {
                    "batch_size": candidate,
                    "status": "cuda_oom",
                    "error": str(error)[:1000],
                }
            )
            LOGGER.warning("[%s] detector batch=%d caused CUDA OOM", role, candidate)
            break
        within_headroom = bool(result["memory_within_headroom"])
        if not within_headroom and attempts:
            result["status"] = "headroom_exceeded"
            attempts.append(result)
            LOGGER.info(
                "[%s] detector batch=%d exceeds configured GPU headroom",
                role,
                candidate,
            )
            break
        result["status"] = "ok"
        if not within_headroom:
            result["headroom_override"] = (
                "batch 1 is retained because no smaller fallback exists"
            )
        attempts.append(result)
        LOGGER.info(
            "[%s] detector batch=%d: %.2f fps",
            role,
            candidate,
            float(result["frames_per_second"]),
        )
        if not within_headroom:
            break

    strategy = str(model_cfg.get("auto_batch_selection", "throughput"))
    selected_attempt, maximum_safe = _select_detector_batch_attempt(
        attempts,
        strategy=strategy,
    )
    selected = int(selected_attempt["batch_size"])
    payload = {
        "schema_version": DETECTOR_BATCH_TUNING_SCHEMA_VERSION,
        "role": role,
        "created_at_unix": time.time(),
        "signature": signature,
        "attempts": attempts,
        "selected_batch_size": selected,
        "maximum_safe_batch_size": maximum_safe,
        "selection_strategy": strategy,
    }
    _write_json_atomic(tuning_path, payload)
    LOGGER.info(
        "[%s] selected detector batch=%d (maximum safe=%d, strategy=%s)",
        role,
        selected,
        maximum_safe,
        strategy,
    )
    return DetectorBatchSelection(
        requested="auto",
        selected_batch_size=selected,
        maximum_safe_batch_size=maximum_safe,
        selection_strategy=strategy,
        tuning_path=tuning_path,
        source="benchmark",
    )


def _predict_detector_batch_with_fallback(
    detector: Any,
    images: Sequence[np.ndarray],
    *,
    batch_size: int,
) -> tuple[list[list[dict[str, Any]]], int, list[dict[str, int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not images:
        return [], batch_size, []
    working_batch_size = min(batch_size, len(images))
    fallback_events: list[dict[str, int]] = []
    while True:
        outputs: list[list[dict[str, Any]]] = []
        try:
            for start in range(0, len(images), working_batch_size):
                outputs.extend(
                    detector.predict_batch(
                        images[start : start + working_batch_size]
                    )
                )
        except BaseException as error:
            if not _is_cuda_oom(error) or working_batch_size <= 1:
                raise
            failed_batch_size = working_batch_size
            working_batch_size = max(1, working_batch_size // 2)
            outputs.clear()
            _clear_cuda_cache()
            fallback_events.append(
                {
                    "failed_batch_size": failed_batch_size,
                    "retry_batch_size": working_batch_size,
                }
            )
            LOGGER.warning(
                "DEIM runtime CUDA OOM at batch=%d; retrying unsaved frames at batch=%d",
                failed_batch_size,
                working_batch_size,
            )
            continue
        if len(outputs) != len(images):
            raise RuntimeError(
                "DEIM batch output count mismatch: "
                f"expected={len(images)}, actual={len(outputs)}"
            )
        effective = batch_size if not fallback_events else working_batch_size
        return outputs, effective, fallback_events


def infer_detector(
    video: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    *,
    role: str,
    model_cfg: Mapping[str, Any],
    device: str,
    video_sha256: str,
    resume: bool,
    force: bool,
) -> Path:
    prediction_path, status_path = _stage_paths(output_dir, role)
    signature = _stage_signature(video_sha256=video_sha256, model=model_cfg)
    if not force and _status_reusable(
        prediction_path, status_path, expected_frames=metadata.frame_count, signature=signature
    ):
        LOGGER.info("[%s] reusing completed predictions", role)
        return prediction_path
    if force:
        prediction_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    elif prediction_path.exists() and not resume:
        raise RuntimeError(f"Partial output exists; use --resume or --force: {prediction_path}")
    start_frame = _last_jsonl_frame(prediction_path) + 1 if resume else 0

    from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector

    detector = DeimFaceDetector(
        config_path=Path(model_cfg["config"]),
        checkpoint_path=Path(model_cfg["checkpoint"]),
        device=device,
        image_size=int(model_cfg["input_size"]),
        score_threshold=float(model_cfg["score_threshold"]),
        maximum_detections=int(model_cfg.get("maximum_detections", 100)),
        crop_geometry="ellipse" if role == "ellipse_deim" else "bbox",
    )
    requested_batch_size, candidates = _detector_batch_candidates(model_cfg)
    try:
        if requested_batch_size == "auto":
            batch_selection = _auto_tune_detector_batch(
                detector,
                video=video,
                start_frame=start_frame,
                remaining_frames=metadata.frame_count - start_frame,
                output_dir=output_dir,
                role=role,
                model_cfg=model_cfg,
                candidates=candidates,
                force=force,
            )
        else:
            batch_selection = DetectorBatchSelection(
                requested=requested_batch_size,
                selected_batch_size=int(requested_batch_size),
                maximum_safe_batch_size=int(requested_batch_size),
                selection_strategy="configured",
                tuning_path=None,
                source="config",
            )
    except BaseException:
        _release_cuda(detector)
        raise
    effective_batch_size = batch_selection.selected_batch_size
    runtime_fallbacks: list[dict[str, int]] = []
    capture = cv2.VideoCapture(video.as_posix())
    if not capture.isOpened():
        _release_cuda(detector)
        raise RuntimeError(f"Cannot open video: {video}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = start_frame
    started = time.perf_counter()
    next_log_frame = ((start_frame // 100) + 1) * 100
    try:
        with prediction_path.open("a" if start_frame else "w", encoding="utf-8") as sink:
            while processed < metadata.frame_count:
                batch_images: list[np.ndarray] = []
                while (
                    len(batch_images) < effective_batch_size
                    and processed + len(batch_images) < metadata.frame_count
                ):
                    okay, frame_bgr = capture.read()
                    if not okay or frame_bgr is None:
                        break
                    batch_images.append(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                if not batch_images:
                    break
                prediction_batch, updated_batch_size, fallback_events = (
                    _predict_detector_batch_with_fallback(
                        detector,
                        batch_images,
                        batch_size=effective_batch_size,
                    )
                )
                runtime_fallbacks.extend(fallback_events)
                effective_batch_size = updated_batch_size
                for offset, predictions in enumerate(prediction_batch):
                    frame_index = processed + offset
                    sink.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "timestamp_sec": frame_index / metadata.fps,
                                "predictions": predictions,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                processed += len(prediction_batch)
                if (
                    processed >= next_log_frame
                    or processed == metadata.frame_count
                ):
                    sink.flush()
                    rate = (processed - start_frame) / max(time.perf_counter() - started, 1e-9)
                    LOGGER.info(
                        "[%s] %d/%d (%.2f fps, batch=%d)",
                        role,
                        processed,
                        metadata.frame_count,
                        rate,
                        effective_batch_size,
                    )
                    while next_log_frame <= processed:
                        next_log_frame += 100
    finally:
        capture.release()
        _release_cuda(detector)
    if processed != metadata.frame_count:
        raise RuntimeError(f"Decoded frame shortfall for {role}: {processed}/{metadata.frame_count}")
    status = {
        "status": "complete",
        "role": role,
        "frames": processed,
        "prediction_path": prediction_path.as_posix(),
        "signature": signature,
        "threshold_selection_split": model_cfg.get("threshold_selection_split"),
        "threshold_source": str(model_cfg.get("threshold_source", "")),
        "batching": {
            "requested_batch_size": batch_selection.requested,
            "selected_batch_size": batch_selection.selected_batch_size,
            "maximum_safe_batch_size": batch_selection.maximum_safe_batch_size,
            "effective_final_batch_size": effective_batch_size,
            "selection_strategy": batch_selection.selection_strategy,
            "selection_source": batch_selection.source,
            "tuning_path": (
                batch_selection.tuning_path.as_posix()
                if batch_selection.tuning_path is not None
                else None
            ),
            "runtime_oom_fallbacks": runtime_fallbacks,
        },
    }
    _write_json_atomic(status_path, status)
    return prediction_path


class QueryRefinementPredictor:
    def __init__(self, model_cfg: Mapping[str, Any], *, device: str) -> None:
        import torch
        from keypoint_comparison1.landmark_detection.config import load_landmark_config
        from keypoint_comparison1.landmark_detection.model import build_landmark_model
        from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint

        experiment = load_landmark_config(Path(model_cfg["config"]), project_root=REPO_ROOT)
        self.raw = experiment.raw
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.model, _ = build_landmark_model(
            self.raw,
            variant="query_refinement",
            initialize_from_detector=False,
        )
        self.model = self.model.to(self.device).eval()
        load_landmark_checkpoint(
            self.model,
            Path(model_cfg["checkpoint"]),
            prefer_ema=True,
            strict=True,
        )
        self.input_size = int(model_cfg["input_size"])
        self.batch_size = int(model_cfg.get("batch_size", 32))
        self.crop_scale = float(model_cfg["crop_scale"])
        self.mean = tuple(float(value) for value in self.raw["data"]["mean"])
        self.std = tuple(float(value) for value in self.raw["data"]["std"])

    def predict(self, image_rgb: np.ndarray, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        import torch
        from keypoint_comparison1.landmark_detection.dataset import image_to_tensor
        from keypoint_comparison1.landmark_detection.geometry import (
            build_crop_transform,
            normalized_crop_to_original,
            warp_crop,
        )
        from src.data.face_crop_cache import crop_geometry_from_detection

        if not detections:
            return []
        crops = []
        transforms = []
        for detection in detections:
            crop_box, rotation = crop_geometry_from_detection(detection, require_ellipse=True)
            transform = build_crop_transform(
                crop_box,
                output_size=self.input_size,
                scale_factor=self.crop_scale,
                rotation_deg=rotation,
                warp_bbox_aspect=False,
            )
            crops.append(image_to_tensor(warp_crop(image_rgb, transform), mean=self.mean, std=self.std))
            transforms.append(transform)
        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.stack(crops[start : start + self.batch_size]).to(self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    output = self.model(batch)
                coordinates = output["decoded_coordinates"].detach().float().cpu().numpy()
                visibility = output["visibility_logits"].sigmoid().detach().float().cpu().numpy()
                decoded.extend(zip(coordinates, visibility, strict=True))
        return [
            {
                **dict(detection),
                "keypoints": normalized_crop_to_original(coordinates, transform).tolist(),
                "visibility": visibility.tolist(),
            }
            for detection, transform, (coordinates, visibility) in zip(
                detections, transforms, decoded, strict=True
            )
        ]


class OrformerPredictor:
    def __init__(self, model_cfg: Mapping[str, Any], *, device: str) -> None:
        import torch
        from keypoint_comparison2.config import load_config
        from keypoint_comparison2.training import load_trained_model

        experiment = load_config(Path(model_cfg["config"]), project_root=REPO_ROOT)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.model = load_trained_model(
            experiment,
            str(model_cfg["model_name"]),
            checkpoint=Path(model_cfg["checkpoint"]),
            device=self.device,
        )
        self.input_size = int(model_cfg["input_size"])
        self.crop_scale = float(model_cfg["crop_scale"])
        self.batch_size = int(model_cfg.get("batch_size", 32))
        self.flip_test = bool(model_cfg.get("flip_test", True))

    def predict(self, image_rgb: np.ndarray, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        predictions, _timing = self.predict_profiled(image_rgb, detections)
        return predictions

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def predict_profiled(
        self,
        image_rgb: np.ndarray,
        detections: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        import torch
        from keypoint_comparison2.data import build_affine_crop
        from src.data.face_crop_cache import crop_geometry_from_detection

        timing = {key: 0.0 for key in TIMING_KEYS}
        preprocess_started = time.perf_counter()
        crops = []
        transforms = []
        for detection in detections:
            crop_box, rotation = crop_geometry_from_detection(
                detection,
                require_ellipse=True,
            )
            transform = build_affine_crop(
                crop_box,
                output_size=self.input_size,
                scale=self.crop_scale,
                rotation=rotation,
                warp_bbox_aspect=False,
            )
            crop = cv2.warpAffine(
                image_rgb,
                transform.matrix,
                (self.input_size, self.input_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            crops.append(
                torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1)
                / 255.0
            )
            transforms.append(transform)
        timing["preprocess_cpu_ms"] = (
            time.perf_counter() - preprocess_started
        ) * 1000.0
        if not crops:
            return [], timing

        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                cpu_batch = torch.stack(crops[start : start + self.batch_size])

                self._synchronize()
                phase_started = time.perf_counter()
                batch = cpu_batch.to(self.device)
                self._synchronize()
                timing["host_to_device_ms"] += (
                    time.perf_counter() - phase_started
                ) * 1000.0

                self._synchronize()
                phase_started = time.perf_counter()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    output = self.model(batch)
                    flipped_output = (
                        self.model(torch.flip(batch, dims=(-1,)))
                        if self.flip_test
                        else None
                    )
                self._synchronize()
                timing["model_gpu_ms"] += (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                coordinates = output["coordinates"]
                visibility = output["visibility_logits"].sigmoid()
                if flipped_output is not None:
                    flipped_coordinates = flipped_output["coordinates"][
                        :, [1, 0, 2, 4, 3]
                    ].clone()
                    flipped_coordinates[..., 0] = 1.0 - flipped_coordinates[..., 0]
                    flipped_visibility = flipped_output[
                        "visibility_logits"
                    ].sigmoid()[:, [1, 0, 2, 4, 3]]
                    coordinates = 0.5 * (coordinates + flipped_coordinates)
                    visibility = 0.5 * (visibility + flipped_visibility)
                self._synchronize()
                timing["postprocess_gpu_ms"] += (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                coordinates_cpu = coordinates.detach().float().cpu()
                visibility_cpu = visibility.detach().float().cpu()
                self._synchronize()
                timing["device_to_host_ms"] += (
                    time.perf_counter() - phase_started
                ) * 1000.0
                decoded.extend(zip(coordinates_cpu, visibility_cpu, strict=True))

        postprocess_started = time.perf_counter()
        results = []
        for detection, transform, (coordinates, visibility) in zip(
            detections, transforms, decoded, strict=True
        ):
            points = coordinates.numpy() * (self.input_size - 1)
            homogeneous = np.concatenate(
                (points, np.ones((5, 1), dtype=np.float32)),
                axis=1,
            )
            original = homogeneous @ transform.inverse.T
            results.append(
                {
                    **dict(detection),
                    "keypoints": original.tolist(),
                    "visibility": visibility.tolist(),
                }
            )
        timing["postprocess_cpu_ms"] = (
            time.perf_counter() - postprocess_started
        ) * 1000.0
        return results, timing


def _iter_cached_rows(path: Path, *, start_frame: int) -> Iterable[dict[str, Any]]:
    expected = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            index = int(row["frame_index"])
            if index != expected:
                raise RuntimeError(f"Non-contiguous cached frame in {path}: expected {expected}, got {index}")
            expected += 1
            if index >= start_frame:
                yield row


def infer_landmarks(
    video: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    *,
    role: str,
    model_cfg: Mapping[str, Any],
    ellipse_prediction_path: Path,
    device: str,
    video_sha256: str,
    resume: bool,
    force: bool,
) -> Path:
    prediction_path, status_path = _stage_paths(output_dir, role)
    dependency_sha256 = sha256_file(ellipse_prediction_path)
    signature = _stage_signature(
        video_sha256=video_sha256,
        model=model_cfg,
        dependency_sha256=dependency_sha256,
    )
    if not force and _status_reusable(
        prediction_path, status_path, expected_frames=metadata.frame_count, signature=signature
    ):
        LOGGER.info("[%s] reusing completed predictions", role)
        return prediction_path
    if force:
        prediction_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    elif prediction_path.exists() and not resume:
        raise RuntimeError(f"Partial output exists; use --resume or --force: {prediction_path}")
    start_frame = _last_jsonl_frame(prediction_path) + 1 if resume else 0
    predictor: QueryRefinementPredictor | OrformerPredictor
    predictor = (
        QueryRefinementPredictor(model_cfg, device=device)
        if role == "query_refinement"
        else OrformerPredictor(model_cfg, device=device)
    )
    capture = cv2.VideoCapture(video.as_posix())
    if not capture.isOpened():
        _release_cuda(predictor)
        raise RuntimeError(f"Cannot open video: {video}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = start_frame
    started = time.perf_counter()
    timing_path = output_dir / "logs" / f"{role}_timing.json"
    timings = TimingAccumulator(scope=f"{role}_inference", frame_start=start_frame)
    try:
        cached_rows = _iter_cached_rows(ellipse_prediction_path, start_frame=start_frame)
        with prediction_path.open("a" if start_frame else "w", encoding="utf-8") as sink:
            for cached in cached_rows:
                if processed >= metadata.frame_count:
                    break
                frame_started = time.perf_counter()
                phase_started = time.perf_counter()
                okay, frame_bgr = capture.read()
                frame_timing = {key: 0.0 for key in TIMING_KEYS}
                frame_timing["decode_ms"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0
                if not okay or frame_bgr is None:
                    break
                if int(cached["frame_index"]) != processed:
                    raise RuntimeError(f"Ellipse cache/video mismatch at frame {processed}")
                preprocess_started = time.perf_counter()
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                color_conversion_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if isinstance(predictor, OrformerPredictor):
                    predictions, prediction_timing = predictor.predict_profiled(
                        image_rgb,
                        cached["predictions"],
                    )
                    for key, value in prediction_timing.items():
                        frame_timing[key] += value
                else:
                    predictions = predictor.predict(
                        image_rgb,
                        cached["predictions"],
                    )
                frame_timing["preprocess_cpu_ms"] += color_conversion_ms
                postprocess_started = time.perf_counter()
                serialized = json.dumps(
                    {
                        "frame_index": processed,
                        "timestamp_sec": processed / metadata.fps,
                        "predictions": predictions,
                        "detector_source": "ellipse_deim",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                sink.write(serialized + "\n")
                frame_timing["postprocess_cpu_ms"] += (
                    time.perf_counter() - postprocess_started
                ) * 1000.0
                frame_timing["total_ms"] = (
                    time.perf_counter() - frame_started
                ) * 1000.0
                timings.add(
                    processed,
                    frame_timing,
                    active=bool(cached["predictions"]),
                )
                processed += 1
                if processed % 100 == 0 or processed == metadata.frame_count:
                    sink.flush()
                    timings.write(timing_path)
                    rate = (processed - start_frame) / max(time.perf_counter() - started, 1e-9)
                    LOGGER.info("[%s] %d/%d (%.2f fps)", role, processed, metadata.frame_count, rate)
    finally:
        if timings.frames:
            timings.write(timing_path)
        capture.release()
        _release_cuda(predictor)
    if processed != metadata.frame_count:
        raise RuntimeError(f"Decoded frame shortfall for {role}: {processed}/{metadata.frame_count}")
    status = {
        "status": "complete",
        "role": role,
        "frames": processed,
        "prediction_path": prediction_path.as_posix(),
        "signature": signature,
        "detector_source": "ellipse_deim",
        "crop_geometry": "ellipse_aligned_long_axis_square",
        "crop_scale": float(model_cfg["crop_scale"]),
        "checkpoint_selection_split": model_cfg.get("checkpoint_selection_split"),
        "checkpoint_selection_metric": model_cfg.get("checkpoint_selection_metric"),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prediction_path


def load_prediction_rows(path: Path, expected_frames: int) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]] | None] = [None] * expected_frames
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            index = int(row["frame_index"])
            if not 0 <= index < expected_frames or output[index] is not None:
                raise RuntimeError(f"Invalid or duplicate frame index {index}: {path}")
            output[index] = [dict(item) for item in row["predictions"]]
    missing = next((index for index, row in enumerate(output) if row is None), None)
    if missing is not None:
        raise RuntimeError(f"Incomplete predictions at frame {missing}: {path}")
    return [row for row in output if row is not None]


def _clock(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def _transform_point(point: Sequence[float], scale: float, offset_x: float, offset_y: float) -> tuple[int, int]:
    return int(round(float(point[0]) * scale + offset_x)), int(round(float(point[1]) * scale + offset_y))


def _ellipse_polygon(
    ellipse: Sequence[float],
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    samples: int = 72,
) -> np.ndarray:
    cx, cy, major, minor, theta = (float(value) for value in ellipse)
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    cosine, sine = math.cos(theta), math.sin(theta)
    local_x, local_y = major * np.cos(angles), minor * np.sin(angles)
    points = np.stack(
        (
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        ),
        axis=1,
    )
    points[:, 0] = points[:, 0] * scale + offset_x
    points[:, 1] = points[:, 1] * scale + offset_y
    return np.rint(points).astype(np.int32)


def draw_panel(
    frame_bgr: np.ndarray,
    predictions: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    title: str,
    detail: str,
    mode: str,
    visibility_threshold: float,
    render: RenderConfig,
) -> np.ndarray:
    if mode not in {"bbox", "ellipse", "landmarks"}:
        raise ValueError(f"Unsupported panel mode: {mode}")
    source_height, source_width = frame_bgr.shape[:2]
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    panel[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = cv2.resize(
        frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    line_width = max(1, int(render.line_width))
    for prediction in predictions:
        if mode == "bbox":
            box = prediction.get("detector_bbox_xyxy", prediction.get("bbox_xyxy"))
            if isinstance(box, (list, tuple)) and len(box) == 4:
                p1 = _transform_point(box[:2], scale, offset_x, offset_y)
                p2 = _transform_point(box[2:], scale, offset_x, offset_y)
                cv2.rectangle(panel, p1, p2, (60, 230, 90), line_width, cv2.LINE_AA)
        else:
            ellipse = prediction.get("ellipse")
            if isinstance(ellipse, (list, tuple)) and len(ellipse) == 5:
                cv2.polylines(
                    panel,
                    [_ellipse_polygon(ellipse, scale=scale, offset_x=offset_x, offset_y=offset_y)],
                    True,
                    (60, 220, 255),
                    line_width,
                    cv2.LINE_AA,
                )
        if mode == "landmarks":
            keypoints = prediction.get("keypoints", ())
            visibility = prediction.get("visibility", ())
            for index, (point, score) in enumerate(zip(keypoints, visibility)):
                center = _transform_point(point, scale, offset_x, offset_y)
                color = LANDMARK_COLORS[index % len(LANDMARK_COLORS)]
                radius = max(2, int(render.landmark_radius))
                if float(score) >= visibility_threshold:
                    cv2.circle(panel, center, radius, color, -1, cv2.LINE_AA)
                    cv2.circle(panel, center, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.circle(panel, center, radius, color, 1, cv2.LINE_AA)
                    cv2.line(
                        panel,
                        (center[0] - radius, center[1] - radius),
                        (center[0] + radius, center[1] + radius),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (width, 66), (0, 0, 0), -1)
    panel = cv2.addWeighted(overlay, 0.62, panel, 0.38, 0)
    cv2.putText(panel, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, detail, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def render_video(
    video: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    config: ComparisonConfig,
    *,
    force: bool,
) -> Path:
    output_path = output_dir / "deim_architecture_four_panel.mp4"
    if output_path.is_file() and not force:
        LOGGER.info("[render] reusing %s", output_path)
        return output_path
    rows = {
        role: load_prediction_rows(_stage_paths(output_dir, role)[0], metadata.frame_count)
        for role in MODEL_ROLES
    }
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for rendering")
    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    partial.unlink(missing_ok=True)
    log_path = output_dir / "logs/ffmpeg_render.stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.render.width}x{config.render.height}",
        "-r", f"{metadata.fps:.12f}", "-i", "-",
    ]
    if metadata.has_audio:
        command.extend(["-i", video.as_posix(), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        command.extend(["-map", "0:v:0", "-an"])
    command.extend(
        [
            "-c:v", config.render.codec, "-crf", str(config.render.crf),
            "-preset", config.render.preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
    )
    if metadata.has_audio:
        command.extend(["-c:a", config.render.audio_codec, "-b:a", config.render.audio_bitrate])
    command.append(partial.as_posix())

    panel_width, panel_height = config.render.width // 2, config.render.height // 2
    panel_modes = {
        "normal_deim": "bbox",
        "ellipse_deim": "ellipse",
        "query_refinement": "landmarks",
        "orformer": "landmarks",
    }
    capture = cv2.VideoCapture(video.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for rendering: {video}")
    preview = output_dir / "deim_architecture_four_panel_center.jpg"
    rendered = 0
    started = time.perf_counter()
    timing_path = output_dir / "logs/render_timing.json"
    timings = TimingAccumulator(scope="four_panel_render", frame_start=0)
    with log_path.open("wb") as stderr:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
        assert encoder.stdin is not None
        try:
            while rendered < metadata.frame_count:
                frame_started = time.perf_counter()
                phase_started = time.perf_counter()
                okay, frame = capture.read()
                frame_timing = {key: 0.0 for key in TIMING_KEYS}
                frame_timing["decode_ms"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0
                if not okay or frame is None:
                    break
                render_started = time.perf_counter()
                panels = []
                for role in MODEL_ROLES:
                    model = config.models[role]
                    threshold = float(
                        model.get(
                            "visibility_threshold",
                            model.get(
                                "score_threshold",
                                config.models["ellipse_deim"]["score_threshold"],
                            ),
                        )
                    )
                    panels.append(
                        draw_panel(
                            frame,
                            rows[role][rendered],
                            width=panel_width,
                            height=panel_height,
                            title=str(model["title"]),
                            detail=(
                                f"{_clock(rendered / metadata.fps)} | "
                                f"faces {len(rows[role][rendered])} | "
                                f"{'vis' if panel_modes[role] == 'landmarks' else 'det'} thr {threshold:.2f}"
                            ),
                            mode=panel_modes[role],
                            visibility_threshold=float(model.get("visibility_threshold", 0.5)),
                            render=config.render,
                        )
                    )
                grid = cv2.vconcat((cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])))
                frame_timing["render_ms"] = (
                    time.perf_counter() - render_started
                ) * 1000.0
                write_started = time.perf_counter()
                encoder.stdin.write(grid.tobytes())
                frame_timing["video_write_ms"] = (
                    time.perf_counter() - write_started
                ) * 1000.0
                if rendered == metadata.frame_count // 2:
                    cv2.imwrite(preview.as_posix(), grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
                frame_timing["total_ms"] = (
                    time.perf_counter() - frame_started
                ) * 1000.0
                timings.add(
                    rendered,
                    frame_timing,
                    active=any(rows[role][rendered] for role in MODEL_ROLES),
                )
                rendered += 1
                if rendered % 300 == 0 or rendered == metadata.frame_count:
                    timings.write(timing_path)
                    LOGGER.info(
                        "[render] %d/%d (%.2f fps)",
                        rendered,
                        metadata.frame_count,
                        rendered / max(time.perf_counter() - started, 1e-9),
                    )
        finally:
            if timings.frames:
                timings.write(timing_path)
            capture.release()
            encoder.stdin.close()
            return_code = encoder.wait()
    if return_code != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"ffmpeg render failed: {detail}")
    if rendered != metadata.frame_count:
        raise RuntimeError(f"Rendered frame shortfall: {rendered}/{metadata.frame_count}")
    partial.replace(output_path)
    return output_path


def write_manifest(
    output_dir: Path,
    *,
    video: Path,
    video_sha256: str,
    metadata: VideoMetadata,
    config: ComparisonConfig,
) -> None:
    models = {}
    for role, model in config.models.items():
        models[role] = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in model.items()
            },
            "config_sha256": sha256_file(Path(model["config"])),
            "checkpoint_sha256": sha256_file(Path(model["checkpoint"])),
        }
    payload = {
        "schema_version": 1,
        "source_video": video.as_posix(),
        "source_video_sha256": video_sha256,
        "inference_seed": config.seed,
        "video_metadata": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "duration_sec": metadata.duration_sec,
            "has_audio": metadata.has_audio,
        },
        "coordinate_conventions": {
            "bbox": "absolute source pixels [x1,y1,x2,y2]",
            "ellipse": "absolute source pixels [cx,cy,major_radius,minor_radius,theta_radians]",
            "keypoints": f"absolute source pixels [x,y], order={list(LANDMARK_NAMES)}",
            "crop": "ellipse-aligned isotropic long-axis square, scale=1.25",
        },
        "shared_detector_contract": {
            "ellipse_prediction_stage": "ellipse_deim",
            "consumers": ["query_refinement", "orformer"],
        },
        "models": models,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _configure_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "run.log", encoding="utf-8")],
        force=True,
    )


def dry_run_plan(config: ComparisonConfig, videos: Sequence[Path], output_root: Path) -> dict[str, Any]:
    return {
        "mode": "dry-run (no frames decoded, no models loaded)",
        "config": config.path.as_posix(),
        "requested_input": config.requested_input,
        "resolved_input": config.input_path.as_posix(),
        "videos": [
            {
                "input": video.as_posix(),
                "output_dir": video_output_dir(output_root, video).as_posix(),
            }
            for video in videos
        ],
        "stage_order": list(MODEL_ROLES) + ["render"],
        "ellipse_detector_reused_by": ["query_refinement", "orformer"],
        "detector_batching": {
            role: {
                "inference_batch_size": model.get("inference_batch_size", 1),
                "batch_candidates": model.get("batch_candidates"),
                "selection_strategy": model.get(
                    "auto_batch_selection",
                    "configured",
                ),
            }
            for role, model in config.models.items()
            if role in DETECTOR_ROLES
        },
        "assets": {
            role: {
                "config": Path(model["config"]).as_posix(),
                "checkpoint": Path(model["checkpoint"]).as_posix(),
            }
            for role, model in config.models.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    validate_model_assets(config)
    input_path = args.input.expanduser().resolve() if args.input else config.input_path
    output_root = args.output_dir.expanduser().resolve() if args.output_dir else config.output_root
    videos = discover_videos(input_path, recursive=config.recursive, extensions=config.extensions)
    if args.dry_run:
        print(json.dumps(dry_run_plan(config, videos, output_root), ensure_ascii=False, indent=2))
        return 0
    _configure_logging(output_root / "logs")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required")
    np.random.seed(config.seed)
    if args.stage in {"infer", "all"}:
        import torch

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    for video in videos:
        output_dir = video_output_dir(output_root, video)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = probe_video(video)
        video_sha256 = sha256_file(video)
        if args.stage in {"infer", "all"}:
            infer_detector(
                video,
                output_dir,
                metadata,
                role="normal_deim",
                model_cfg=config.models["normal_deim"],
                device=args.device,
                video_sha256=video_sha256,
                resume=args.resume,
                force=args.force,
            )
            ellipse_path = infer_detector(
                video,
                output_dir,
                metadata,
                role="ellipse_deim",
                model_cfg=config.models["ellipse_deim"],
                device=args.device,
                video_sha256=video_sha256,
                resume=args.resume,
                force=args.force,
            )
            for role in ("query_refinement", "orformer"):
                infer_landmarks(
                    video,
                    output_dir,
                    metadata,
                    role=role,
                    model_cfg=config.models[role],
                    ellipse_prediction_path=ellipse_path,
                    device=args.device,
                    video_sha256=video_sha256,
                    resume=args.resume,
                    force=args.force,
                )
        write_manifest(
            output_dir,
            video=video,
            video_sha256=video_sha256,
            metadata=metadata,
            config=config,
        )
        if args.stage in {"render", "all"}:
            output = render_video(video, output_dir, metadata, config, force=args.force)
            LOGGER.info("completed: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
