#!/usr/bin/env python3
"""Review an inferred uniform montage and save source frames on pause.

The reviewed video is expected to have been created from the chronological,
uniform clips produced by ``create_uniform_video_montage.py``.  Pressing Space
or P pauses playback and immediately saves the corresponding unannotated frame
from the original source video.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.create_uniform_video_montage import build_clip_ranges  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEFAULT_INPUT_ROOT = Path("data/input/active_learning")
DEFAULT_OUTPUT_ROOT = Path("data/output/active_learning/active_montage")


class CliError(RuntimeError):
    """An error that should be printed without a traceback."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class ReviewPosition:
    montage_frame_index: int
    montage_time_sec: float
    segment_index: int
    source_time_sec: float
    source_frame_index: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play an inferred uniform montage and save the original source frame whenever playback is paused.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-prefix",
        default="MIDA",
        help="Beginning of the source filename used for recursive discovery.",
    )
    parser.add_argument("--source-video", type=Path, help="Original video; overrides --video-prefix discovery.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing original active-learning videos.",
    )
    parser.add_argument(
        "--inference-video",
        type=Path,
        help="Inferred montage. If omitted, a comparison/annotated MP4 is found under the dated output directory.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--review-date",
        default=None,
        help="Date used in the output name (YYYY-MM-DD); local today's date if omitted.",
    )
    parser.add_argument("--segments", type=int, default=60, help="Uniform montage segment count.")
    parser.add_argument("--clip-duration", type=float, default=10.0, help="Seconds per montage segment.")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Initial montage playback position.")
    parser.add_argument("--max-width", type=int, default=1920, help="Maximum review-window width; 0 keeps native size.")
    parser.add_argument("--max-height", type=int, default=1080, help="Maximum review-window height; 0 keeps native size.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args(argv)


def _rational_to_float(value: str | None) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise CliError(f"Video does not exist: {path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CliError(f"ffprobe failed for {path}: {result.stderr.strip()[:2000]}")
    try:
        payload: dict[str, Any] = json.loads(result.stdout)
        stream = next(row for row in payload.get("streams", []) if row.get("codec_type") == "video")
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
        fps = _rational_to_float(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        raw_frame_count = int(stream.get("nb_frames") or 0)
    except (StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CliError(f"Invalid video metadata: {path}") from error
    frame_count = raw_frame_count or int(round(duration * fps))
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise CliError(
            f"Invalid video metadata for {path}: duration={duration}, fps={fps}, size={width}x{height}"
        )
    return VideoInfo(duration, fps, width, height, frame_count)


def discover_source_video(input_root: Path, video_prefix: str) -> Path:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise CliError(f"Input root does not exist: {root}")
    prefix = video_prefix.casefold()
    matches = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES and path.stem.casefold().startswith(prefix)
    )
    if not matches:
        raise CliError(f"No source video beginning with {video_prefix!r} was found under {root}")
    if len(matches) > 1:
        detail = "\n".join(f"  - {path}" for path in matches)
        raise CliError(f"Multiple source videos begin with {video_prefix!r}; use --source-video:\n{detail}")
    return matches[0]


def review_date(value: str | None) -> str:
    if value is None:
        return datetime.now().astimezone().date().isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise CliError(f"--review-date must be YYYY-MM-DD: {value}")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise CliError(f"--review-date must be YYYY-MM-DD: {value}") from error


def source_short_name(source_path: Path) -> str:
    if len(source_path.stem) < 3:
        raise CliError(f"Source filename must contain at least three characters: {source_path.name}")
    short = source_path.stem[:3]
    return "".join(character if character.isalnum() else "_" for character in short)


def session_directory(source_path: Path, output_root: Path, date_text: str) -> Path:
    return output_root.expanduser().resolve() / f"{source_short_name(source_path)}_{date_text}"


def discover_inference_video(session_dir: Path) -> Path:
    candidates = [
        path.resolve()
        for path in session_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    ]

    def rank(path: Path) -> tuple[int, int, str]:
        name = path.stem.casefold()
        if name == "four_model_face_comparison":
            priority = 0
        elif "comparison" in name:
            priority = 1
        elif "annotated" in name or "inference" in name:
            priority = 2
        else:
            priority = 9
        return priority, -path.stat().st_mtime_ns, path.as_posix()

    ranked = sorted((path for path in candidates if rank(path)[0] < 9), key=rank)
    if not ranked:
        raise CliError(
            f"No inferred comparison/annotated video was found under {session_dir}. "
            "Specify it with --inference-video."
        )
    return ranked[0]


def map_montage_position(
    *,
    montage_frame_index: int,
    montage_fps: float,
    source_duration_sec: float,
    source_fps: float,
    segments: int,
    clip_duration_sec: float,
) -> ReviewPosition:
    if montage_frame_index < 0:
        raise ValueError("montage_frame_index must be non-negative")
    if montage_fps <= 0 or source_fps <= 0:
        raise ValueError("video FPS values must be positive")
    ranges = build_clip_ranges(source_duration_sec, segments, clip_duration_sec)
    montage_time = montage_frame_index / montage_fps
    expected_duration = segments * clip_duration_sec
    if montage_time >= expected_duration + max(0.25, 2.0 / montage_fps):
        raise CliError(
            f"Montage position {montage_time:.3f}s exceeds expected duration {expected_duration:.3f}s"
        )
    bounded_time = min(max(0.0, montage_time), math.nextafter(expected_duration, 0.0))
    segment_index = min(int(bounded_time // clip_duration_sec), segments - 1)
    offset = bounded_time - segment_index * clip_duration_sec
    source_time = min(ranges[segment_index].clip_start_sec + offset, source_duration_sec)
    source_frame = min(int(round(source_time * source_fps)), max(0, int(round(source_duration_sec * source_fps)) - 1))
    return ReviewPosition(
        montage_frame_index=montage_frame_index,
        montage_time_sec=montage_time,
        segment_index=segment_index,
        source_time_sec=source_time,
        source_frame_index=source_frame,
    )


def _next_sequence(output_dir: Path, basename: str) -> int:
    sequence = 1
    while any(output_dir.glob(f"{basename}_{sequence:03d}_*.jpg")):
        sequence += 1
    return sequence


def _write_jpeg(path: Path, frame: Any, quality: int) -> None:
    okay, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not okay:
        raise CliError(f"Could not encode frame as JPEG: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


class SourceFrameSaver:
    def __init__(
        self,
        *,
        source_path: Path,
        inference_path: Path,
        source_info: VideoInfo,
        montage_info: VideoInfo,
        output_dir: Path,
        date_text: str,
        segments: int,
        clip_duration_sec: float,
        jpeg_quality: int,
    ) -> None:
        self.source_path = source_path
        self.inference_path = inference_path
        self.source_info = source_info
        self.montage_info = montage_info
        self.output_dir = output_dir
        self.date_text = date_text
        self.segments = segments
        self.clip_duration_sec = clip_duration_sec
        self.jpeg_quality = jpeg_quality
        self.basename = f"{source_short_name(source_path)}_{date_text}"
        self.capture = cv2.VideoCapture(source_path.as_posix())
        if not self.capture.isOpened():
            raise CliError(f"Cannot open source video: {source_path}")

    def close(self) -> None:
        self.capture.release()

    def save(self, montage_frame_index: int) -> tuple[Path, ReviewPosition]:
        position = map_montage_position(
            montage_frame_index=montage_frame_index,
            montage_fps=self.montage_info.fps,
            source_duration_sec=self.source_info.duration_sec,
            source_fps=self.source_info.fps,
            segments=self.segments,
            clip_duration_sec=self.clip_duration_sec,
        )
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, position.source_frame_index)
        okay, frame = self.capture.read()
        if not okay or frame is None:
            raise CliError(
                f"Could not decode source frame {position.source_frame_index} at {position.source_time_sec:.3f}s"
            )
        actual_frame = max(0, int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sequence = _next_sequence(self.output_dir, self.basename)
        filename = (
            f"{self.basename}_{sequence:03d}_"
            f"f{actual_frame:09d}_t{position.source_time_sec:010.3f}s.jpg"
        )
        output_path = self.output_dir / filename
        _write_jpeg(output_path, frame, self.jpeg_quality)
        record = {
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image": output_path.name,
            "source_video": self.source_path.as_posix(),
            "inference_video": self.inference_path.as_posix(),
            **asdict(position),
            "decoded_source_frame_index": actual_frame,
            "uniform_montage": {
                "segments": self.segments,
                "clip_duration_sec": self.clip_duration_sec,
            },
        }
        with (self.output_dir / "review_manifest.jsonl").open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return output_path, position


def _display_frame(frame: Any, max_width: int, max_height: int) -> Any:
    height, width = frame.shape[:2]
    width_scale = max_width / width if max_width > 0 else 1.0
    height_scale = max_height / height if max_height > 0 else 1.0
    scale = min(1.0, width_scale, height_scale)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _seek_frame(capture: Any, frame_index: int) -> tuple[Any, int]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    okay, frame = capture.read()
    if not okay or frame is None:
        raise CliError(f"Could not decode inference frame {frame_index}")
    actual_index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
    return frame, actual_index


def review(
    *,
    inference_path: Path,
    montage_info: VideoInfo,
    saver: SourceFrameSaver,
    start_sec: float,
    max_width: int,
    max_height: int,
) -> int:
    capture = cv2.VideoCapture(inference_path.as_posix())
    if not capture.isOpened():
        raise CliError(f"Cannot open inferred montage: {inference_path}")
    start_frame = min(int(round(start_sec * montage_info.fps)), montage_info.frame_count - 1)
    frame, frame_index = _seek_frame(capture, start_frame)
    window_name = f"Inference montage review - {saver.source_path.stem[:24]}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as error:
        capture.release()
        raise CliError("OpenCV could not create a GUI window; run this command in a desktop session.") from error

    paused = False
    saves = 0
    frame_delay_ms = max(1, int(round(1000.0 / montage_info.fps)))
    print("Controls: Space/P=pause and auto-save, Space/P=resume, A/D=step while paused, S=save again, Q/Esc=quit")
    try:
        while True:
            cv2.imshow(window_name, _display_frame(frame, max_width, max_height))
            key = cv2.waitKey(0 if paused else frame_delay_ms) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
            if key in {32, ord("p"), ord("P")}:
                paused = not paused
                if paused:
                    output_path, position = saver.save(frame_index)
                    saves += 1
                    print(
                        f"Saved: {output_path} "
                        f"(montage={position.montage_time_sec:.3f}s, source={position.source_time_sec:.3f}s)"
                    )
                continue
            if key in {ord("s"), ord("S")}:
                output_path, position = saver.save(frame_index)
                saves += 1
                print(
                    f"Saved: {output_path} "
                    f"(montage={position.montage_time_sec:.3f}s, source={position.source_time_sec:.3f}s)"
                )
                continue
            if paused and key in {ord("a"), ord("A"), ord("d"), ord("D")}:
                direction = -1 if key in {ord("a"), ord("A")} else 1
                target = min(max(0, frame_index + direction), montage_info.frame_count - 1)
                frame, frame_index = _seek_frame(capture, target)
                continue
            if paused:
                continue
            okay, next_frame = capture.read()
            if not okay or next_frame is None:
                break
            frame = next_frame
            frame_index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return saves


def run(args: argparse.Namespace) -> int:
    if args.segments <= 0:
        raise CliError("--segments must be greater than zero")
    if not math.isfinite(args.clip_duration) or args.clip_duration <= 0:
        raise CliError("--clip-duration must be a positive finite value")
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        raise CliError("--start-sec must be a non-negative finite value")
    if not 1 <= args.jpeg_quality <= 100:
        raise CliError("--jpeg-quality must be between 1 and 100")

    source_path = (
        args.source_video.expanduser().resolve()
        if args.source_video is not None
        else discover_source_video(args.input_root, args.video_prefix)
    )
    if not source_path.is_file():
        raise CliError(f"Source video does not exist: {source_path}")
    date_text = review_date(args.review_date)
    output_dir = session_directory(source_path, args.output_root, date_text)
    inference_path = (
        args.inference_video.expanduser().resolve()
        if args.inference_video is not None
        else discover_inference_video(output_dir)
    )
    source_info = probe_video(source_path)
    montage_info = probe_video(inference_path)
    expected_duration = args.segments * args.clip_duration
    tolerance = max(0.25, 2.0 / montage_info.fps)
    if abs(montage_info.duration_sec - expected_duration) > tolerance:
        raise CliError(
            f"Inferred montage duration is {montage_info.duration_sec:.3f}s; expected "
            f"{expected_duration:.3f}s for {args.segments} x {args.clip_duration:.3f}s"
        )
    if args.start_sec >= montage_info.duration_sec:
        raise CliError(
            f"--start-sec {args.start_sec:.3f} is outside the montage ({montage_info.duration_sec:.3f}s)"
        )

    saver = SourceFrameSaver(
        source_path=source_path,
        inference_path=inference_path,
        source_info=source_info,
        montage_info=montage_info,
        output_dir=output_dir,
        date_text=date_text,
        segments=args.segments,
        clip_duration_sec=args.clip_duration,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Source: {source_path}")
    print(f"Inference: {inference_path}")
    print(f"Saved frames: {output_dir}")
    try:
        saves = review(
            inference_path=inference_path,
            montage_info=montage_info,
            saver=saver,
            start_sec=args.start_sec,
            max_width=args.max_width,
            max_height=args.max_height,
        )
    finally:
        saver.close()
    print(f"Review finished: {saves} frame(s) saved")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
