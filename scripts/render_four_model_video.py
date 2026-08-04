#!/usr/bin/env python3
"""Run four detector adapters on every source frame and render a 2x2 video.

Inference results are persisted per model as JSONL before rendering. Completed
model results are reusable and interrupted JSONL files can be resumed, so a
long video does not need to restart from the first model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from typing import Any, Sequence

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.active_learning.adapters import create_adapter  # noqa: E402
from src.active_learning.config import ModelConfig  # noqa: E402
from src.active_learning.render import COLORS, _draw_panel  # noqa: E402
from src.active_learning.types import Prediction  # noqa: E402


LOGGER = logging.getLogger("four_model_video")


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


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    project_root: Path
    models: tuple[ModelConfig, ...]
    render: RenderConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("infer", "render", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> ComparisonConfig:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("Comparison config must be a mapping")
    project_root = _resolve(raw.get("project_root", "../.."), config_path.parent)
    models: list[ModelConfig] = []
    for raw_model in raw.get("models", []):
        row = dict(raw_model)
        row["weights"] = _resolve(row["weights"], project_root)
        if row.get("config"):
            row["config"] = _resolve(row["config"], project_root)
        models.append(ModelConfig(**row))
    if len(models) != 4:
        raise ValueError(f"Exactly four models are required for a 2x2 comparison, got {len(models)}")
    if len({model.name for model in models}) != 4:
        raise ValueError("Model names must be unique")
    for model in models:
        if not model.weights.is_file():
            raise FileNotFoundError(f"Missing weights for {model.name}: {model.weights}")
        if model.config is not None and not model.config.is_file():
            raise FileNotFoundError(f"Missing config for {model.name}: {model.config}")
        if model.confidence_threshold <= 0:
            raise ValueError(f"confidence_threshold must be positive for {model.name}")
    render = RenderConfig(**dict(raw.get("render", {})))
    if render.width <= 0 or render.height <= 0 or render.width % 2 or render.height % 2:
        raise ValueError("Render width and height must be positive even numbers")
    return ComparisonConfig(project_root=project_root, models=tuple(models), render=render)


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"Required command is unavailable: {command}")


def probe_video(path: Path) -> VideoMetadata:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:4000]}")
    payload = json.loads(result.stdout)
    video = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"Input has no video stream: {path}")

    def ratio(value: str) -> float:
        numerator, separator, denominator = str(value).partition("/")
        return float(numerator) / float(denominator) if separator else float(value)

    fps = ratio(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    frame_count = int(video.get("nb_frames") or 0)
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    if frame_count <= 0:
        frame_count = int(round(duration * fps))
    metadata = VideoMetadata(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        has_audio=any(row.get("codec_type") == "audio" for row in payload.get("streams", [])),
    )
    if metadata.width <= 0 or metadata.height <= 0 or metadata.fps <= 0 or metadata.frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {metadata}")
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str) -> str:
    value = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in value.split("_") if part)


def _last_jsonl_frame(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return -1
    last = ""
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = line
    return int(json.loads(last)["frame_index"]) if last else -1


def _status_matches(status: dict[str, Any], *, input_sha256: str, weight_sha256: str, model: ModelConfig) -> bool:
    return (
        status.get("status") == "complete"
        and status.get("input_sha256") == input_sha256
        and status.get("weight_sha256") == weight_sha256
        and math.isclose(float(status.get("confidence_threshold", -1)), model.confidence_threshold)
    )


def infer_model(
    input_path: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    model: ModelConfig,
    *,
    input_sha256: str,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(model.name)
    prediction_path = prediction_dir / f"{slug}.jsonl"
    status_path = prediction_dir / f"{slug}.status.json"
    weight_sha256 = sha256_file(model.weights)
    if status_path.is_file() and prediction_path.is_file() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if _status_matches(status, input_sha256=input_sha256, weight_sha256=weight_sha256, model=model):
            LOGGER.info("[infer:%s] reusing %s", model.name, prediction_path)
            return status
    if force:
        prediction_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    elif prediction_path.exists() and not resume:
        raise RuntimeError(f"Partial prediction file exists; use --resume or --force: {prediction_path}")

    start_frame = _last_jsonl_frame(prediction_path) + 1 if resume else 0
    if start_frame >= metadata.frame_count:
        start_frame = metadata.frame_count
    capture = cv2.VideoCapture(input_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    LOGGER.info("[infer:%s] loading adapter; start frame %d/%d", model.name, start_frame, metadata.frame_count)
    adapter = create_adapter(model)
    started = time.perf_counter()
    processed = start_frame
    mode = "a" if start_frame else "w"
    try:
        with prediction_path.open(mode, encoding="utf-8") as sink:
            while processed < metadata.frame_count:
                frames = []
                indices = []
                for _ in range(max(1, model.batch_size)):
                    okay, frame = capture.read()
                    if not okay or frame is None:
                        break
                    indices.append(processed + len(indices))
                    frames.append(frame)
                if not frames:
                    break
                timestamps = [index / metadata.fps for index in indices]
                batch_predictions, _ = adapter.predict_batch(
                    frames,
                    video_id=input_path.stem,
                    timestamps=timestamps,
                    return_embeddings=False,
                )
                for index, timestamp, predictions in zip(indices, timestamps, batch_predictions, strict=True):
                    sink.write(json.dumps({
                        "frame_index": index,
                        "timestamp_sec": timestamp,
                        "predictions": [prediction.to_dict() for prediction in predictions],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += len(frames)
                if processed % 300 < len(frames) or processed == metadata.frame_count:
                    sink.flush()
                    elapsed = time.perf_counter() - started
                    rate = max(0, processed - start_frame) / max(elapsed, 1e-9)
                    LOGGER.info("[infer:%s] %d/%d (%.1f%%, %.2f fps)", model.name, processed, metadata.frame_count, processed * 100 / metadata.frame_count, rate)
    finally:
        capture.release()
        del adapter
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if processed != metadata.frame_count:
        raise RuntimeError(f"Decoded frame shortfall for {model.name}: {processed}/{metadata.frame_count}")
    elapsed = time.perf_counter() - started
    status = {
        "status": "complete",
        "model_name": model.name,
        "input_sha256": input_sha256,
        "weight_path": model.weights.as_posix(),
        "weight_sha256": weight_sha256,
        "config_path": model.config.as_posix() if model.config else None,
        "confidence_threshold": model.confidence_threshold,
        "frames": processed,
        "inference_seconds_this_run": elapsed,
        "adapter_inference_seconds": None,
        "effective_fps_this_run": max(0, processed - start_frame) / max(elapsed, 1e-9),
        "prediction_path": prediction_path.as_posix(),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def load_predictions(path: Path, expected_frames: int) -> list[list[Prediction]]:
    output: list[list[Prediction] | None] = [None] * expected_frames
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            index = int(row["frame_index"])
            if not 0 <= index < expected_frames:
                raise RuntimeError(f"Prediction frame index is out of range: {index}")
            output[index] = [Prediction.from_dict(item) for item in row["predictions"]]
    missing = [index for index, rows in enumerate(output) if rows is None]
    if missing:
        raise RuntimeError(f"Prediction file is incomplete; first missing frame: {missing[0]}")
    return [rows for rows in output if rows is not None]


def _clock(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def render_video(
    input_path: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    config: ComparisonConfig,
    *,
    force: bool,
) -> tuple[Path, float]:
    output_path = output_dir / "four_model_face_comparison.mp4"
    if output_path.is_file() and not force:
        LOGGER.info("[render] reusing %s", output_path)
        return output_path, 0.0
    predictions_by_model: dict[str, list[list[Prediction]]] = {}
    for model in config.models:
        path = output_dir / "predictions" / f"{_slug(model.name)}.jsonl"
        predictions_by_model[model.name] = load_predictions(path, metadata.frame_count)

    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    partial.unlink(missing_ok=True)
    log_path = output_dir / "logs" / "ffmpeg_render.stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.render.width}x{config.render.height}", "-r", f"{metadata.fps:.12f}", "-i", "-",
    ]
    if metadata.has_audio:
        command.extend(["-i", input_path.as_posix(), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        command.extend(["-map", "0:v:0", "-an"])
    command.extend([
        "-c:v", config.render.codec, "-crf", str(config.render.crf), "-preset", config.render.preset,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ])
    if metadata.has_audio:
        command.extend(["-c:a", config.render.audio_codec, "-b:a", config.render.audio_bitrate])
    command.append(partial.as_posix())

    panel_width = config.render.width // 2
    panel_height = config.render.height // 2
    capture = cv2.VideoCapture(input_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video for rendering: {input_path}")
    started = time.perf_counter()
    center_image = output_dir / "four_model_face_comparison_center.jpg"
    with log_path.open("wb") as stderr:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
        assert encoder.stdin is not None
        rendered = 0
        try:
            for frame_index in range(metadata.frame_count):
                okay, frame = capture.read()
                if not okay or frame is None:
                    break
                grid = cv2.vconcat([
                    cv2.hconcat([
                        _draw_panel(
                            frame,
                            predictions_by_model[config.models[0].name][frame_index],
                            width=panel_width,
                            height=panel_height,
                            title=config.models[0].name,
                            detail=f"{_clock(frame_index / metadata.fps)} | thr {config.models[0].confidence_threshold:.2f} | detections {len(predictions_by_model[config.models[0].name][frame_index])}",
                            color=COLORS[0],
                            class_name="face",
                        ),
                        _draw_panel(
                            frame,
                            predictions_by_model[config.models[1].name][frame_index],
                            width=panel_width,
                            height=panel_height,
                            title=config.models[1].name,
                            detail=f"{_clock(frame_index / metadata.fps)} | thr {config.models[1].confidence_threshold:.2f} | detections {len(predictions_by_model[config.models[1].name][frame_index])}",
                            color=COLORS[1],
                            class_name="face",
                        ),
                    ]),
                    cv2.hconcat([
                        _draw_panel(
                            frame,
                            predictions_by_model[config.models[2].name][frame_index],
                            width=panel_width,
                            height=panel_height,
                            title=config.models[2].name,
                            detail=f"{_clock(frame_index / metadata.fps)} | thr {config.models[2].confidence_threshold:.2f} | detections {len(predictions_by_model[config.models[2].name][frame_index])}",
                            color=COLORS[2],
                            class_name="face",
                        ),
                        _draw_panel(
                            frame,
                            predictions_by_model[config.models[3].name][frame_index],
                            width=panel_width,
                            height=panel_height,
                            title=config.models[3].name,
                            detail=f"{_clock(frame_index / metadata.fps)} | thr {config.models[3].confidence_threshold:.2f} | detections {len(predictions_by_model[config.models[3].name][frame_index])}",
                            color=COLORS[3],
                            class_name="face",
                        ),
                    ]),
                ])
                encoder.stdin.write(grid.tobytes())
                rendered += 1
                if frame_index == metadata.frame_count // 2:
                    cv2.imwrite(center_image.as_posix(), grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if rendered % 300 == 0 or rendered == metadata.frame_count:
                    elapsed = time.perf_counter() - started
                    LOGGER.info("[render] %d/%d (%.1f%%, %.2f fps)", rendered, metadata.frame_count, rendered * 100 / metadata.frame_count, rendered / max(elapsed, 1e-9))
        finally:
            capture.release()
            encoder.stdin.close()
            return_code = encoder.wait()
    if return_code != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"ffmpeg rendering failed: {detail}")
    if rendered != metadata.frame_count:
        raise RuntimeError(f"Rendered frame shortfall: {rendered}/{metadata.frame_count}")
    partial.replace(output_path)
    return output_path, time.perf_counter() - started


def configure_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(log_dir / "run.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require_command("ffmpeg")
    _require_command("ffprobe")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    config = load_config(args.config)
    metadata = probe_video(input_path)
    input_sha256 = sha256_file(input_path)
    LOGGER.info("input=%s frames=%d fps=%.6f size=%dx%d", input_path, metadata.frame_count, metadata.fps, metadata.width, metadata.height)
    statuses: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    if args.stage in {"infer", "all"}:
        for model in config.models:
            statuses.append(infer_model(
                input_path,
                output_dir,
                metadata,
                model,
                input_sha256=input_sha256,
                resume=args.resume,
                force=args.force,
            ))
    else:
        for model in config.models:
            status_path = output_dir / "predictions" / f"{_slug(model.name)}.status.json"
            if not status_path.is_file():
                raise FileNotFoundError(f"Missing completed inference status for render stage: {status_path}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not _status_matches(
                status,
                input_sha256=input_sha256,
                weight_sha256=sha256_file(model.weights),
                model=model,
            ):
                raise RuntimeError(f"Inference status does not match input/model for {model.name}")
            statuses.append(status)
    output_video: Path | None = None
    render_seconds = 0.0
    if args.stage in {"render", "all"}:
        output_video, render_seconds = render_video(input_path, output_dir, metadata, config, force=args.force)
    manifest = {
        "status": "complete",
        "input": input_path.as_posix(),
        "input_sha256": input_sha256,
        "video": asdict(metadata),
        "config": args.config.expanduser().resolve().as_posix(),
        "models": [
            {
                "name": model.name,
                "type": model.type,
                "weights": model.weights.as_posix(),
                "config": model.config.as_posix() if model.config else None,
                "threshold": model.confidence_threshold,
                "threshold_selection": "validation max recall subject to precision > 0.90 at IoU 0.50",
            }
            for model in config.models
        ],
        "inference": statuses,
        "output_video": output_video.as_posix() if output_video else None,
        "render_seconds": render_seconds,
        "total_seconds_this_run": time.perf_counter() - total_started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump({
        "input": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "models": [
            {
                "name": model.name,
                "type": model.type,
                "weights": model.weights.as_posix(),
                "config": model.config.as_posix() if model.config else None,
                "confidence_threshold": model.confidence_threshold,
                "batch_size": model.batch_size,
                "source_class_id": model.source_class_id,
            }
            for model in config.models
        ],
        "render": asdict(config.render),
    }, sort_keys=False), encoding="utf-8")
    LOGGER.info("complete output=%s", output_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
