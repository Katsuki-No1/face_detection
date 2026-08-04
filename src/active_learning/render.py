from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .adapters import ModelAdapter
from .config import PipelineConfig
from .media import run_checked
from .types import Candidate, Prediction, VideoInfo

LOGGER = logging.getLogger(__name__)
COLORS = [(70, 220, 255), (120, 255, 100), (255, 150, 80), (255, 100, 220)]
BELOW_THRESHOLD_COLOR = (0, 140, 255)  # orange in BGR


def _letterbox(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, int(round(frame.shape[1] * scale)))
    resized_height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, scale, left, top


def _draw_panel(
    frame: np.ndarray,
    predictions: Sequence[Prediction],
    *,
    width: int,
    height: int,
    title: str,
    detail: str,
    color: tuple[int, int, int],
    class_name: str = "object",
    roi_bbox: tuple[float, float, float, float] | None = None,
    analysis_threshold: float | None = None,
) -> np.ndarray:
    # Keep metadata outside the image. Drawing the title bar over the decoded
    # frame hid faces touching the top edge, which is exactly where reviewers
    # need to inspect detector recall. The image is therefore letterboxed into
    # a slightly smaller content area below a dedicated header.
    header_height = min(height - 1, max(72, int(round(height * 0.095))))
    content_height = max(1, height - header_height)
    content, scale, left, content_top = _letterbox(frame, width, content_height)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[header_height:height] = content
    top = header_height + content_top
    thickness = max(2, int(round(min(width, height) / 500)))
    font_scale = max(0.65, min(width, height) / 900)
    for row in predictions:
        below_threshold = analysis_threshold is not None and row.confidence < analysis_threshold
        box_color = BELOW_THRESHOLD_COLOR if below_threshold else color
        x1, y1, x2, y2 = row.bbox_xyxy_original
        point_a = (int(round(x1 * scale + left)), int(round(y1 * scale + top)))
        point_b = (int(round(x2 * scale + left)), int(round(y2 * scale + top)))
        cv2.rectangle(panel, point_a, point_b, box_color, thickness, cv2.LINE_AA)
        label = (
            f"{class_name}? {row.confidence:.2f} <T{analysis_threshold:.2f}"
            if below_threshold and analysis_threshold is not None
            else f"{class_name} {row.confidence:.2f}"
        )
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65, thickness)
        label_top = max(header_height, point_a[1] - text_size[1] - 10)
        cv2.rectangle(panel, (point_a[0], label_top), (min(width, point_a[0] + text_size[0] + 8), point_a[1]), box_color, -1)
        cv2.putText(panel, label, (point_a[0] + 4, max(text_size[1], point_a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65, (10, 10, 10), thickness, cv2.LINE_AA)
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        point_a = (int(round(x1 * scale + left)), int(round(y1 * scale + top)))
        point_b = (int(round(x2 * scale + left)), int(round(y2 * scale + top)))
        cv2.rectangle(panel, point_a, point_b, (255, 80, 255), thickness + 1, cv2.LINE_AA)
        cv2.putText(
            panel, "ROI", (point_a[0] + 4, min(height - 8, point_a[1] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.68, (255, 80, 255), thickness, cv2.LINE_AA,
        )
    cv2.rectangle(panel, (0, 0), (width, header_height - 1), (15, 15, 15), -1)
    cv2.putText(panel, title, (22, int(header_height * 0.43)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(panel, detail, (22, int(header_height * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.68, (240, 240, 240), max(1, thickness - 1), cv2.LINE_AA)
    return panel


def _nearest(rows: list[tuple[float, list[Prediction]]], timestamp: float) -> list[Prediction]:
    if not rows:
        return []
    return min(rows, key=lambda item: abs(item[0] - timestamp))[1]


def _sample_clip_for_inference(path: Path, fps: float) -> tuple[list[float], list[np.ndarray]]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source clip: {path}")
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(1e-6, capture.get(cv2.CAP_PROP_FPS))
    timestamps = np.arange(0.0, max(0.001, duration), 1.0 / fps).tolist()
    frames: list[np.ndarray] = []
    valid_timestamps: list[float] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            okay, frame = capture.read()
            if okay and frame is not None:
                valid_timestamps.append(float(timestamp))
                frames.append(frame)
    finally:
        capture.release()
    return valid_timestamps, frames


def _infer_clip(
    path: Path,
    adapters: Sequence[ModelAdapter],
    *,
    video_id: str,
    fps: float,
) -> dict[str, list[tuple[float, list[Prediction]]]]:
    timestamps, frames = _sample_clip_for_inference(path, fps)
    by_model: dict[str, list[tuple[float, list[Prediction]]]] = defaultdict(list)
    for adapter in adapters:
        adapter.reset()
        batch_size = max(1, adapter.config.batch_size)
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            batch_timestamps = timestamps[start : start + batch_size]
            predictions, _ = adapter.predict_batch(
                batch_frames,
                video_id=video_id,
                timestamps=batch_timestamps,
                return_embeddings=False,
            )
            by_model[adapter.model_name].extend(zip(batch_timestamps, predictions, strict=True))
    return by_model


def render_review_clip(
    source_clip: Path,
    output_path: Path,
    comparison_image: Path,
    candidate: Candidate,
    adapters: Sequence[ModelAdapter],
    config: PipelineConfig,
    source_start_sec: float,
    duration_sec: float | None = None,
) -> float:
    started = time.perf_counter()
    predictions = _infer_clip(
        source_clip,
        adapters,
        video_id=candidate.video_id,
        fps=config.montage.review_inference_fps,
    )
    capture = cv2.VideoCapture(source_clip.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source clip for rendering: {source_clip}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or config.montage.fps
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_duration = source_frames / max(source_fps, 1e-6)
    duration = float(duration_sec) if duration_sec is not None else decoded_duration
    panel_count = 1 + len(adapters)
    column_count = max(1, int(math.ceil(math.sqrt(panel_count))))
    row_count = max(1, int(math.ceil(panel_count / column_count)))
    x_edges = np.linspace(0, config.montage.width, column_count + 1, dtype=np.int32)
    y_edges = np.linspace(0, config.montage.height, row_count + 1, dtype=np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    stderr_path = config.output_dir / "logs" / "ffmpeg" / f"{candidate.clip_id}_grid.stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.montage.width}x{config.montage.height}", "-r", f"{config.montage.fps:.6f}", "-i", "-",
        "-an", "-c:v", config.montage.codec, "-crf", str(config.montage.crf), "-preset", config.montage.preset,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary_output.as_posix(),
    ]
    with stderr_path.open("wb") as error_file:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file)
        assert encoder.stdin is not None
        output_frame_count = max(1, int(round(duration * config.montage.fps)))
        comparison_written = False
        current_frame: np.ndarray | None = None
        current_source_index = -1
        try:
            for output_index in range(output_frame_count):
                timestamp = output_index / config.montage.fps
                target_source_index = min(max(0, source_frames - 1), int(timestamp * source_fps))
                while current_source_index < target_source_index:
                    okay, decoded = capture.read()
                    if not okay or decoded is None:
                        break
                    current_frame = decoded
                    current_source_index += 1
                if current_frame is None:
                    break
                frame = current_frame
                # OpenCV's built-in font is ASCII-only. The exact Unicode
                # filename remains in manifest/HTML; the overlay uses the
                # stable video ID plus original extension so every required
                # field stays legible within the panel width.
                source_label = f"{candidate.video_id}{Path(candidate.filename).suffix.lower()}"
                detail = (
                    f"{candidate.clip_id} | {source_label} | source {source_start_sec + timestamp:.2f}s | "
                    f"{candidate.selection_bucket} {candidate.selection_score:.3f}"
                )
                panel_specs: list[tuple[str, list[Prediction], tuple[int, int, int], str, float | None]] = [
                    ("ORIGINAL", [], (230, 230, 230), "frame", None)
                ]
                for index, adapter in enumerate(adapters):
                    rows = _nearest(predictions.get(adapter.model_name, []), timestamp)
                    threshold = adapter.config.analysis_threshold if adapter.config.analysis_threshold is not None else adapter.config.confidence_threshold
                    boundary_floor = max(
                        adapter.config.confidence_threshold,
                        threshold * config.confidence_boundary.min_ratio,
                    )
                    rows = [row for row in rows if row.confidence >= boundary_floor]
                    panel_specs.append(
                        (adapter.model_name, rows, COLORS[index % len(COLORS)], adapter.config.class_name, threshold)
                    )
                grid = np.zeros((config.montage.height, config.montage.width, 3), dtype=np.uint8)
                for panel_index, (title, rows, color, class_name, threshold) in enumerate(panel_specs):
                    row_index, column_index = divmod(panel_index, column_count)
                    left, right = int(x_edges[column_index]), int(x_edges[column_index + 1])
                    top, bottom = int(y_edges[row_index]), int(y_edges[row_index + 1])
                    if title == "ORIGINAL":
                        detection_detail = "N/A"
                    else:
                        accepted = sum(row.confidence >= float(threshold) for row in rows)
                        boundary = len(rows) - accepted
                        detection_detail = f">=T {accepted} | <T boundary {boundary} | orange=<T"
                    panel_detail = f"{detail} | detections {detection_detail}"
                    grid[top:bottom, left:right] = _draw_panel(
                        frame,
                        rows,
                        width=right - left,
                        height=bottom - top,
                        title=title,
                        detail=panel_detail,
                        color=color,
                        class_name=class_name,
                        roi_bbox=candidate.roi_bbox if title == "ORIGINAL" else None,
                        analysis_threshold=threshold,
                    )
                encoder.stdin.write(grid.tobytes())
                if not comparison_written and timestamp >= duration / 2.0:
                    comparison_image.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(comparison_image.as_posix(), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    comparison_written = True
        finally:
            capture.release()
            encoder.stdin.close()
            return_code = encoder.wait()
    if return_code != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"ffmpeg review encoder failed for {candidate.clip_id}: {detail}")
    temporary_output.replace(output_path)
    return time.perf_counter() - started


def cut_source_clip(
    source_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    duration_sec: float,
    config: PipelineConfig,
    clip_id: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    command = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{start_sec:.6f}", "-i", source_path.as_posix(),
        "-t", f"{duration_sec:.6f}", "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary_output.as_posix(),
    ]
    run_checked(command, config.output_dir / "logs" / "ffmpeg" / f"{clip_id}_source.stderr.log")
    temporary_output.replace(output_path)


def concatenate_review_clips(paths: Sequence[Path], output_path: Path, config: PipelineConfig) -> float:
    started = time.perf_counter()
    concat_path = config.output_dir / "logs" / "review_concat.txt"
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path.write_text("".join(f"file '{path.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in paths), encoding="utf-8")
    temporary_output = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_path.as_posix(),
        "-map", "0:v:0", "-c", "copy", "-an", "-movflags", "+faststart", temporary_output.as_posix(),
    ]
    run_checked(command, config.output_dir / "logs" / "ffmpeg" / "montage.stderr.log")
    temporary_output.replace(output_path)
    return time.perf_counter() - started
