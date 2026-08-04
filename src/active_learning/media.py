from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import hashlib
import numpy as np

from script.script_data.clip_frame import VIDEO_EXTENSIONS, command_error_message, find_videos, require_command

from .types import VideoInfo

LOGGER = logging.getLogger(__name__)


def pixel_sha256(frame_bgr: np.ndarray) -> str:
    """Hash decoded pixels together with shape and dtype for exact deduplication."""

    digest = hashlib.sha256()
    digest.update(str(frame_bgr.shape).encode("ascii"))
    digest.update(frame_bgr.dtype.str.encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(frame_bgr)))
    return digest.hexdigest()


def read_video_frame(
    path: Path,
    *,
    timestamp_sec: float,
    frame_index: int = -1,
) -> tuple[np.ndarray, int, float]:
    """Decode one original-resolution source frame and return actual position metadata."""

    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    try:
        if frame_index >= 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        okay, frame = capture.read()
        if not okay or frame is None:
            raise RuntimeError(
                f"Cannot decode source frame: {path} @ {timestamp_sec:.6f}s frame={frame_index}"
            )
        actual_index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_timestamp = actual_index / fps if fps > 0 else max(0.0, timestamp_sec)
        return frame, actual_index, actual_timestamp
    finally:
        capture.release()


def rational(value: str | None) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def video_id_from_path(path: Path) -> str:
    prefix = path.stem.split()[0].strip().replace("_", "-")
    return prefix or path.stem


def probe_video(path: Path) -> VideoInfo:
    row = VideoInfo(video_id=video_id_from_path(path), filename=path.name, path=path.resolve().as_posix(), file_size=path.stat().st_size)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        row.error = command_error_message(result)
        return row
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        row.duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
        row.fps = rational(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        row.width = int(stream.get("width", 0))
        row.height = int(stream.get("height", 0))
        row.codec = str(stream.get("codec_name", ""))
        count = str(stream.get("nb_frames", ""))
        row.frame_count = int(count) if count.isdigit() else int(round(row.duration * row.fps))
        if row.duration <= 0 or row.width <= 0 or row.height <= 0:
            raise ValueError("invalid duration or dimensions")
        # Lightweight first/middle/end decode check; a damaged video is logged
        # without aborting other inputs.
        for timestamp in (0.0, row.duration / 2.0, max(0.0, row.duration - 0.1)):
            check = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}", "-i", path.as_posix(), "-frames:v", "1", "-f", "null", "-"],
                check=False,
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                raise RuntimeError(command_error_message(check))
        row.readable = True
    except (KeyError, ValueError, TypeError, RuntimeError) as error:
        row.error = str(error)
    return row


def sample_frames_ffmpeg(video: VideoInfo, sample_fps: float) -> Iterator[tuple[float, int, np.ndarray]]:
    frame_bytes = video.width * video.height * 3
    command = [
        "ffmpeg", "-v", "error", "-i", video.path, "-map", "0:v:0", "-vf", f"fps={sample_fps:.12g}:start_time=0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    index = 0
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"Short raw frame for {video.filename}: {len(raw)}/{frame_bytes}; {stderr[:1000]}")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(video.height, video.width, 3).copy()
        yield index / sample_fps, index, frame
        index += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg sparse decode failed for {video.filename}: {stderr[:4000]}")


def sample_video_window(
    video: VideoInfo,
    *,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
) -> Iterator[tuple[float, int, np.ndarray]]:
    """Decode a bounded source interval and report absolute source timestamps."""

    start = max(0.0, float(start_sec))
    end = min(float(video.duration), float(end_sec))
    if end <= start:
        return
    frame_bytes = video.width * video.height * 3
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", video.path,
        "-t", f"{end - start:.6f}", "-map", "0:v:0", "-vf", f"fps={sample_fps:.12g}:start_time=0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    index = 0
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"Short dense frame for {video.filename}: {len(raw)}/{frame_bytes}; {stderr[:1000]}")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(video.height, video.width, 3).copy()
        yield start + index / sample_fps, index, frame
        index += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg dense decode failed for {video.filename}: {stderr[:4000]}")


def frame_quality(frame_bgr: np.ndarray, previous_small: np.ndarray | None) -> tuple[float, float, float, str, float, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8]
    median = float(np.median(low[1:]))
    bits = (low > median).reshape(-1)
    phash = f"{sum((int(bit) << index) for index, bit in enumerate(bits)):016x}"
    scene = float(np.mean(cv2.absdiff(small.astype(np.uint8), previous_small.astype(np.uint8))) / 255.0) if previous_small is not None else 0.0
    return brightness, contrast, blur, phash, scene, small


def run_checked(command: Sequence[str], stderr_path: Path) -> None:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as error_file:
        result = subprocess.run(list(command), check=False, stdout=subprocess.DEVNULL, stderr=error_file, text=True)
    if result.returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"External command failed ({result.returncode}): {' '.join(command[:8])}\n{detail}")


def check_media_commands() -> None:
    require_command("ffmpeg")
    require_command("ffprobe")
