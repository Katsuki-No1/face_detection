#!/usr/bin/env python3
"""Run trained bbox detectors on every frame of a video and render overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.active_learning.adapters import ModelAdapter, create_adapter  # noqa: E402
from src.active_learning.config import ModelConfig, load_pipeline_config  # noqa: E402
from src.active_learning.types import Prediction  # noqa: E402


LOGGER = logging.getLogger("bbox_video_inference")
MODEL_COLORS = {
    "dino_deimv2": (0, 140, 255),
    "rtmdet": (255, 190, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DINOv3+DEIMv2 / RTMDet on every video frame and create annotated videos."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/active_learning_round_001.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dino_deimv2", "rtmdet"])
    parser.add_argument("--device", default=None, help="Override the configured device, for example cuda:0.")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to the input average FPS.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional leading duration (mainly for smoke tests).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _rational(value: str | None) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _video_metadata(path: Path) -> dict[str, Any]:
    data = _probe(path)
    video_stream = next((row for row in data.get("streams", []) if row.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"no video stream: {path}")
    duration = float(video_stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = _rational(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: duration={duration}, fps={fps}, size={width}x{height}")
    return {
        "path": path.resolve().as_posix(),
        "duration_sec": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "codec": str(video_stream.get("codec_name") or ""),
        "has_audio": any(row.get("codec_type") == "audio" for row in data.get("streams", [])),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_external(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(list(command), check=False, stdout=subprocess.DEVNULL, stderr=log)
    if result.returncode != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"external command failed ({result.returncode}): {' '.join(command[:10])}\n{detail}")


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_match_count(previous: Sequence[Sequence[float]], current: Sequence[Sequence[float]]) -> int:
    pairs = sorted(
        [(_bbox_iou(left, right), left_index, right_index) for left_index, left in enumerate(previous) for right_index, right in enumerate(current)],
        reverse=True,
    )
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches = 0
    for overlap, previous_index, current_index in pairs:
        if overlap < 0.30:
            break
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        matches += 1
    return matches


def _draw_predictions(frame: np.ndarray, predictions: Sequence[Prediction], model_name: str, timestamp: float) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    color = MODEL_COLORS.get(model_name, (0, 255, 255))
    thickness = max(2, int(round(min(width, height) / 360)))
    font_scale = max(0.5, min(width, height) / 1100.0)
    for row in predictions:
        x1, y1, x2, y2 = row.bbox_xyxy_original
        left = int(np.clip(round(x1), 0, width - 1))
        top = int(np.clip(round(y1), 0, height - 1))
        right = int(np.clip(round(x2), left + 1, width))
        bottom = int(np.clip(round(y2), top + 1, height))
        cv2.rectangle(output, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)
        label = f"face {row.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_top = max(0, top - text_height - baseline - 5)
        cv2.rectangle(output, (left, label_top), (min(width, left + text_width + 8), top), color, -1)
        cv2.putText(
            output,
            label,
            (left + 4, max(text_height + 1, top - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    header = f"{model_name}  t={timestamp:07.2f}s  faces={len(predictions)}"
    (header_width, header_height), header_baseline = cv2.getTextSize(
        header, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(output, (8, 8), (min(width - 1, header_width + 24), header_height + header_baseline + 20), (0, 0, 0), -1)
    cv2.putText(
        output,
        header,
        (16, header_height + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return output


def _prediction_record(prediction: Prediction) -> dict[str, Any]:
    row = prediction.to_dict()
    row["class_name"] = "face"
    return row


def _validate_video(path: Path, duration: float, expected_frames: int, log_dir: Path) -> dict[str, Any]:
    data = _probe(path)
    stream = next((row for row in data.get("streams", []) if row.get("codec_type") == "video"), None)
    if stream is None:
        raise RuntimeError(f"generated output has no video stream: {path}")
    actual_duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    frame_text = str(stream.get("nb_frames") or "")
    if abs(actual_duration - duration) > 0.15:
        raise RuntimeError(f"duration mismatch for {path.name}: {actual_duration:.6f} != {duration:.6f}")
    if not frame_text.isdigit() or int(frame_text) != expected_frames:
        raise RuntimeError(f"frame count mismatch for {path.name}: {frame_text} != {expected_frames}")
    _run_external(
        ["ffmpeg", "-v", "error", "-xerror", "-i", path.as_posix(), "-map", "0:v:0", "-f", "null", "-"],
        log_dir / "full_decode.log",
    )
    (log_dir / "full_decode.log").unlink(missing_ok=True)
    return {
        "duration_sec": actual_duration,
        "frames": int(frame_text),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": str(stream.get("codec_name") or ""),
        "has_audio": any(row.get("codec_type") == "audio" for row in data.get("streams", [])),
        "full_decode_ok": True,
    }


def _close_failed_processes(decoder: subprocess.Popen[bytes], encoder: subprocess.Popen[bytes]) -> None:
    decoder.kill()
    if encoder.stdin is not None:
        encoder.stdin.close()
    encoder.kill()
    encoder.wait()
    decoder.wait()


def run_model(
    model_config: ModelConfig,
    input_path: Path,
    output_root: Path,
    metadata: dict[str, Any],
    *,
    output_fps: float,
    duration: float,
    crf: int,
    preset: str,
    force: bool,
) -> dict[str, Any]:
    model_dir = output_root / model_config.name
    log_dir = model_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    final_video = model_dir / f"{model_config.name}_annotated_10m.mp4"
    predictions_path = model_dir / "predictions.jsonl"
    summary_path = model_dir / "summary.json"
    if not force and final_video.exists() and predictions_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            LOGGER.info("[%s] completed output already exists; skipping", model_config.name)
            return existing
    if not force and any(path.exists() for path in (final_video, predictions_path, summary_path)):
        raise FileExistsError(f"incomplete existing output for {model_config.name}; use --force after inspecting {model_dir}")

    silent_temporary = model_dir / "annotated_silent.partial.mp4"
    video_temporary = model_dir / f"{model_config.name}_annotated_10m.partial.mp4"
    predictions_temporary = model_dir / "predictions.jsonl.partial"
    summary_temporary = model_dir / "summary.json.partial"
    for path in (silent_temporary, video_temporary, predictions_temporary, summary_temporary):
        path.unlink(missing_ok=True)

    width = int(metadata["width"])
    height = int(metadata["height"])
    expected_frames = int(round(duration * output_fps))
    frame_bytes = width * height * 3
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            input_path.as_posix(),
            "-t",
            f"{duration:.9f}",
            "-map",
            "0:v:0",
            "-vf",
            f"fps={output_fps:.12g}:start_time=0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder_log = (log_dir / "encoder.log").open("w", encoding="utf-8")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{output_fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            silent_temporary.as_posix(),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=encoder_log,
    )
    assert decoder.stdout is not None and encoder.stdin is not None

    LOGGER.info(
        "[%s] loading model (threshold=%.3f, batch=%d)",
        model_config.name,
        model_config.confidence_threshold,
        model_config.batch_size,
    )
    adapter = create_adapter(model_config)
    frames_processed = 0
    frames_with_face = 0
    total_detections = 0
    face_counts: list[float] = []
    confidences: list[float] = []
    area_ratios: list[float] = []
    continuity_matches = 0
    continuity_opportunities = 0
    detection_drop_events = 0
    detection_reappearance_events = 0
    previous_boxes: list[tuple[float, float, float, float]] = []
    batch_frames: list[np.ndarray] = []
    started = time.perf_counter()

    def flush(prediction_file: TextIO, encoder_input: BinaryIO) -> None:
        nonlocal frames_processed, frames_with_face, total_detections
        nonlocal continuity_matches, continuity_opportunities
        nonlocal detection_drop_events, detection_reappearance_events, previous_boxes
        if not batch_frames:
            return
        timestamps = [(frames_processed + index) / output_fps for index in range(len(batch_frames))]
        prediction_batches, _ = adapter.predict_batch(
            batch_frames,
            video_id=input_path.stem,
            timestamps=timestamps,
        )
        for frame, timestamp, predictions in zip(batch_frames, timestamps, prediction_batches, strict=True):
            frame_index = frames_processed
            prediction_file.write(
                json.dumps(
                    {
                        "source": input_path.resolve().as_posix(),
                        "model": model_config.name,
                        "frame_index": frame_index,
                        "clip_time_sec": timestamp,
                        "objects": [_prediction_record(row) for row in predictions],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            boxes = [row.bbox_xyxy_original for row in predictions]
            frames_with_face += int(bool(boxes))
            total_detections += len(boxes)
            face_counts.append(float(len(boxes)))
            for row in predictions:
                confidences.append(row.confidence)
                x1, y1, x2, y2 = row.bbox_xyxy_original
                area_ratios.append(max(0.0, (x2 - x1) * (y2 - y1) / float(width * height)))
            if frame_index > 0:
                continuity_matches += _greedy_match_count(previous_boxes, boxes)
                continuity_opportunities += max(len(previous_boxes), len(boxes))
                detection_drop_events += int(bool(previous_boxes) and not boxes)
                detection_reappearance_events += int(not previous_boxes and bool(boxes))
            previous_boxes = boxes
            annotated = _draw_predictions(frame, predictions, model_config.name, timestamp)
            encoder_input.write(np.ascontiguousarray(annotated).tobytes())
            frames_processed += 1
            if frames_processed % max(1, int(round(output_fps * 10))) == 0:
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "[%s] %.0f/%.0fs, frames=%d, wall_fps=%.2f, model_fps=%.2f",
                    model_config.name,
                    frames_processed / output_fps,
                    duration,
                    frames_processed,
                    frames_processed / max(elapsed, 1e-9),
                    adapter.effective_fps,
                )
        batch_frames.clear()

    try:
        with predictions_temporary.open("w", encoding="utf-8") as prediction_file:
            while frames_processed + len(batch_frames) < expected_frames:
                raw = decoder.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"short decoded frame: {len(raw)}/{frame_bytes}")
                batch_frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
                if len(batch_frames) >= model_config.batch_size:
                    flush(prediction_file, encoder.stdin)
            flush(prediction_file, encoder.stdin)
    except BaseException:
        _close_failed_processes(decoder, encoder)
        encoder_log.close()
        raise

    encoder.stdin.close()
    encoder_return = encoder.wait()
    encoder_log.close()
    decoder_error = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
    decoder_return = decoder.wait()
    if decoder_return != 0:
        raise RuntimeError(f"decoder failed: {decoder_error[:4000]}")
    if encoder_return != 0:
        detail = (log_dir / "encoder.log").read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"encoder failed: {detail}")
    if frames_processed != expected_frames:
        raise RuntimeError(f"processed frame mismatch: {frames_processed} != {expected_frames}")

    _run_external(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            silent_temporary.as_posix(),
            "-i",
            input_path.as_posix(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            f"{duration:.9f}",
            "-movflags",
            "+faststart",
            video_temporary.as_posix(),
        ],
        log_dir / "audio_mux.log",
    )
    validation = _validate_video(video_temporary, duration, expected_frames, log_dir)
    prediction_lines = sum(1 for _ in predictions_temporary.open(encoding="utf-8"))
    if prediction_lines != expected_frames:
        raise RuntimeError(f"prediction line mismatch: {prediction_lines} != {expected_frames}")

    elapsed = time.perf_counter() - started
    summary = {
        "status": "completed",
        "purpose": "normal-operation qualitative/time-series smoke test; no ground-truth labels",
        "source": metadata,
        "model": {
            "name": model_config.name,
            "type": model_config.type,
            "config": model_config.config.resolve().as_posix() if model_config.config else None,
            "weights": model_config.weights.resolve().as_posix(),
            "weights_sha256": _sha256(model_config.weights),
            "device": model_config.device,
            "input_size": model_config.input_size,
            "confidence_threshold": model_config.confidence_threshold,
            "threshold_source": "validation best F1",
            "batch_size": model_config.batch_size,
        },
        "output": {
            "video": final_video.resolve().as_posix(),
            "predictions": predictions_path.resolve().as_posix(),
            "fps": output_fps,
            "duration_sec": duration,
            "frames_processed": frames_processed,
            "validation": validation,
        },
        "metrics": {
            "frames_with_face": frames_with_face,
            "face_detection_coverage": frames_with_face / frames_processed,
            "total_detections": total_detections,
            "face_count": _percentiles(face_counts),
            "confidence": _percentiles(confidences),
            "face_area_ratio": _percentiles(area_ratios),
            "continuity_iou_match_rate": continuity_matches / continuity_opportunities if continuity_opportunities else None,
            "detection_drop_events": detection_drop_events,
            "detection_reappearance_events": detection_reappearance_events,
            "wall_elapsed_sec": elapsed,
            "wall_effective_fps": frames_processed / max(elapsed, 1e-9),
            "model_inference_sec": adapter.inference_seconds,
            "model_effective_fps": adapter.effective_fps,
        },
    }
    summary_temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    video_temporary.replace(final_video)
    predictions_temporary.replace(predictions_path)
    summary_temporary.replace(summary_path)
    silent_temporary.unlink(missing_ok=True)
    LOGGER.info("[%s] completed: %s", model_config.name, final_video)
    return summary


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input video not found: {input_path}")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive")
    metadata = _video_metadata(input_path)
    output_fps = float(args.fps or metadata["fps"])
    duration = min(float(metadata["duration_sec"]), float(args.duration_sec or metadata["duration_sec"]))
    if output_fps <= 0 or duration <= 0 or not math.isfinite(output_fps + duration):
        raise ValueError(f"invalid output timing: fps={output_fps}, duration={duration}")

    pipeline_config = load_pipeline_config(args.config)
    configured = {row.name: row for row in pipeline_config.models}
    missing = [name for name in args.models if name not in configured]
    if missing:
        raise KeyError(f"models not found in {args.config}: {missing}; available={sorted(configured)}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.models:
        row = configured[name]
        threshold = row.analysis_threshold if row.analysis_threshold is not None else row.confidence_threshold
        row = replace(
            row,
            confidence_threshold=float(threshold),
            device=str(args.device or row.device),
            batch_size=int(args.batch_size or row.batch_size),
        )
        summaries.append(
            run_model(
                row,
                input_path,
                output_root,
                metadata,
                output_fps=output_fps,
                duration=duration,
                crf=args.crf,
                preset=args.preset,
                force=args.force,
            )
        )
    comparison = {
        "status": "completed",
        "source": metadata,
        "models": [row["model"]["name"] for row in summaries],
        "summaries": summaries,
    }
    temporary = output_root / "comparison_summary.json.partial"
    temporary.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_root / "comparison_summary.json")
    print(json.dumps({"status": "completed", "output_dir": output_root.as_posix(), "models": comparison["models"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
