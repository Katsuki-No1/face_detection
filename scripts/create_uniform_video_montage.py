#!/usr/bin/env python3
"""Create a chronological montage from uniformly distributed video clips.

By default, the input timeline is divided into 60 equal intervals. A centered
10-second clip is taken from each interval and the clips are concatenated into
one 10-minute MP4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CliError(RuntimeError):
    """An error that should be shown without a Python traceback."""


@dataclass(frozen=True)
class MediaInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    has_audio: bool
    video_duration_sec: float | None = None
    audio_duration_sec: float | None = None
    video_frame_count: int | None = None


@dataclass(frozen=True)
class ClipRange:
    index: int
    interval_start_sec: float
    interval_end_sec: float
    clip_start_sec: float
    clip_end_sec: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide a video into equal timeline intervals, take a centered clip "
            "from each interval, and concatenate the clips chronologically."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/create_uniform_video_montage.py "
            "--input input.mp4 --output montage_10m.mp4"
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Source video.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--segments", type=int, default=60, help="Number of equal timeline intervals.")
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=10.0,
        help="Seconds extracted from the center of every interval.",
    )
    parser.add_argument("--crf", type=int, default=20, help="libx264 quality; lower values produce higher quality.")
    parser.add_argument("--preset", default="fast", help="libx264 encoding preset.")
    parser.add_argument("--audio-bitrate", default="192k", help="AAC output bitrate when the input has audio.")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing output file.")
    parser.add_argument(
        "--verify-decode",
        action="store_true",
        help="Decode the completed video fully before publishing it (slower).",
    )
    return parser.parse_args(argv)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise CliError(f"Required command is not available: {name}")


def rational_to_float(value: str | None) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def run_capture(command: Sequence[str], *, description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(list(command), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise CliError(f"{description} failed:\n{detail[:4000]}")
    return result


def probe_media(path: Path) -> MediaInfo:
    if not path.is_file():
        raise CliError(f"Input video does not exist: {path}")

    result = run_capture(
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
        description=f"ffprobe for {path}",
    )
    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CliError(f"ffprobe returned invalid JSON for {path}") from error

    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise CliError(f"Input has no video stream: {path}")

    raw_duration = payload.get("format", {}).get("duration") or video.get("duration")
    try:
        duration = float(raw_duration)
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError) as error:
        raise CliError(f"Input has invalid video metadata: {path}") from error
    fps = rational_to_float(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    if not math.isfinite(duration) or duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise CliError(
            f"Input has invalid video metadata: duration={duration}, fps={fps}, size={width}x{height}"
        )

    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    def optional_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0 else None

    def optional_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    return MediaInfo(
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=audio is not None,
        video_duration_sec=optional_float(video.get("duration")),
        audio_duration_sec=optional_float(audio.get("duration")) if audio is not None else None,
        video_frame_count=optional_int(video.get("nb_frames")),
    )


def build_clip_ranges(duration_sec: float, segments: int, clip_duration_sec: float) -> list[ClipRange]:
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise CliError("Input duration must be a positive finite number.")
    if segments <= 0:
        raise CliError("--segments must be greater than zero.")
    if not math.isfinite(clip_duration_sec) or clip_duration_sec <= 0:
        raise CliError("--clip-duration must be a positive finite number.")

    interval_duration = duration_sec / segments
    if interval_duration + 1e-9 < clip_duration_sec:
        required_duration = segments * clip_duration_sec
        raise CliError(
            f"Input is too short: {duration_sec:.3f}s. At least {required_duration:.3f}s is required "
            f"for {segments} non-overlapping clips of {clip_duration_sec:.3f}s."
        )

    margin = (interval_duration - clip_duration_sec) / 2.0
    ranges: list[ClipRange] = []
    for index in range(segments):
        interval_start = index * interval_duration
        interval_end = (index + 1) * interval_duration
        clip_start = interval_start + margin
        clip_end = clip_start + clip_duration_sec
        ranges.append(
            ClipRange(
                index=index,
                interval_start_sec=interval_start,
                interval_end_sec=interval_end,
                clip_start_sec=clip_start,
                clip_end_sec=clip_end,
            )
        )
    return ranges


def build_ffmpeg_command(
    *,
    input_path: Path,
    output_path: Path,
    media: MediaInfo,
    ranges: Sequence[ClipRange],
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-v",
        "warning",
        "-stats",
    ]

    for clip in ranges:
        clip_duration = clip.clip_end_sec - clip.clip_start_sec
        command.extend(
            [
                "-ss",
                f"{clip.clip_start_sec:.9f}",
                "-t",
                f"{clip_duration + 0.5:.9f}",
                "-threads",
                "1",
                "-i",
                input_path.as_posix(),
            ]
        )

    filter_parts: list[str] = []
    for input_index, clip in enumerate(ranges):
        clip_duration = clip.clip_end_sec - clip.clip_start_sec
        filter_parts.append(
            f"[{input_index}:v:0]trim=duration={clip_duration:.9f},setpts=PTS-STARTPTS[v{input_index}]"
        )
        if media.has_audio:
            filter_parts.append(
                f"[{input_index}:a:0]atrim=duration={clip_duration:.9f},"
                f"asetpts=PTS-STARTPTS[a{input_index}]"
            )

    if media.has_audio:
        concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(ranges)))
        filter_parts.append(f"{concat_inputs}concat=n={len(ranges)}:v=1:a=1[outv][outa]")
    else:
        concat_inputs = "".join(f"[v{index}]" for index in range(len(ranges)))
        filter_parts.append(f"{concat_inputs}concat=n={len(ranges)}:v=1:a=0[outv]")

    command.extend(["-filter_complex", ";".join(filter_parts), "-map", "[outv]"])
    if media.has_audio:
        command.extend(["-map", "[outa]"])

    output_duration_sec = sum(clip.clip_end_sec - clip.clip_start_sec for clip in ranges)
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            f"{media.fps:.12g}",
        ]
    )
    if media.has_audio:
        command.extend(["-c:a", "aac", "-b:a", audio_bitrate])
    command.extend(
        [
            "-t",
            f"{output_duration_sec:.9f}",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            output_path.as_posix(),
        ]
    )
    return command


def encode_montage(command: Sequence[str]) -> None:
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise CliError(f"ffmpeg montage encoding failed with exit code {result.returncode}.")


def verify_output(path: Path, expected_duration_sec: float, expect_audio: bool) -> MediaInfo:
    media = probe_media(path)
    tolerance = max(0.25, 2.0 / media.fps)
    if abs(media.duration_sec - expected_duration_sec) > tolerance:
        raise CliError(
            f"Unexpected output duration: {media.duration_sec:.3f}s; "
            f"expected {expected_duration_sec:.3f}s (+/- {tolerance:.3f}s)."
        )
    if media.video_duration_sec is None or abs(media.video_duration_sec - expected_duration_sec) > tolerance:
        actual = "unknown" if media.video_duration_sec is None else f"{media.video_duration_sec:.3f}s"
        raise CliError(
            f"Unexpected video-stream duration: {actual}; "
            f"expected {expected_duration_sec:.3f}s (+/- {tolerance:.3f}s)."
        )
    if media.has_audio != expect_audio:
        expected = "with audio" if expect_audio else "without audio"
        actual = "with audio" if media.has_audio else "without audio"
        raise CliError(f"Unexpected output streams: expected {expected}, got {actual}.")
    if expect_audio and (
        media.audio_duration_sec is None or abs(media.audio_duration_sec - expected_duration_sec) > tolerance
    ):
        actual = "unknown" if media.audio_duration_sec is None else f"{media.audio_duration_sec:.3f}s"
        raise CliError(
            f"Unexpected audio-stream duration: {actual}; "
            f"expected {expected_duration_sec:.3f}s (+/- {tolerance:.3f}s)."
        )
    expected_frame_count = round(expected_duration_sec * media.fps)
    if media.video_frame_count is not None and abs(media.video_frame_count - expected_frame_count) > 1:
        raise CliError(
            f"Unexpected video frame count: {media.video_frame_count}; expected about {expected_frame_count}."
        )
    return media


def verify_full_decode(path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-xerror", "-i", path.as_posix(), "-f", "null", "-"],
        check=False,
    )
    if result.returncode != 0:
        raise CliError(f"Full decode verification failed for {path}.")


def create_montage(args: argparse.Namespace) -> Path:
    require_command("ffmpeg")
    require_command("ffprobe")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise CliError("--input and --output must be different files.")
    if output_path.exists() and not args.force:
        raise CliError(f"Output already exists (use --force to replace it): {output_path}")
    if not 0 <= args.crf <= 51:
        raise CliError("--crf must be between 0 and 51.")

    source = probe_media(input_path)
    ranges = build_clip_ranges(source.duration_sec, args.segments, args.clip_duration)
    output_duration = args.segments * args.clip_duration
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Source: {input_path} ({source.duration_sec:.3f}s, "
        f"{source.width}x{source.height}, {source.fps:.6g}fps)"
    )
    print(
        f"Sampling: {args.segments} centered clips x {args.clip_duration:.3f}s "
        f"= {output_duration:.3f}s"
    )
    print(
        f"First clip: {ranges[0].clip_start_sec:.3f}s - {ranges[0].clip_end_sec:.3f}s; "
        f"last clip: {ranges[-1].clip_start_sec:.3f}s - {ranges[-1].clip_end_sec:.3f}s"
    )

    temporary_parent = output_path.parent
    temporary_prefix = f".{output_path.stem}.staging."
    with tempfile.TemporaryDirectory(dir=temporary_parent, prefix=temporary_prefix) as directory:
        staging_dir = Path(directory)
        staged_output = staging_dir / output_path.name
        command = build_ffmpeg_command(
            input_path=input_path,
            output_path=staged_output,
            media=source,
            ranges=ranges,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
        )
        encode_montage(command)
        output_media = verify_output(staged_output, output_duration, source.has_audio)
        if args.verify_decode:
            verify_full_decode(staged_output)
        os.replace(staged_output, output_path)

    print(
        f"Output: {output_path} ({output_media.duration_sec:.3f}s, "
        f"{output_media.width}x{output_media.height}, {output_media.fps:.6g}fps)"
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        create_montage(args)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
