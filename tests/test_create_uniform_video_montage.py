from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_uniform_video_montage.py"
SPEC = importlib.util.spec_from_file_location("create_uniform_video_montage", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
montage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = montage
SPEC.loader.exec_module(montage)


def test_builds_centered_ranges_for_sixty_intervals() -> None:
    ranges = montage.build_clip_ranges(duration_sec=7_200.0, segments=60, clip_duration_sec=10.0)

    assert len(ranges) == 60
    assert ranges[0].interval_start_sec == pytest.approx(0.0)
    assert ranges[0].clip_start_sec == pytest.approx(55.0)
    assert ranges[0].clip_end_sec == pytest.approx(65.0)
    assert ranges[-1].clip_start_sec == pytest.approx(7_135.0)
    assert ranges[-1].clip_end_sec == pytest.approx(7_145.0)
    assert sum(item.clip_end_sec - item.clip_start_sec for item in ranges) == pytest.approx(600.0)


def test_rejects_source_too_short_for_non_overlapping_clips() -> None:
    with pytest.raises(montage.CliError, match="At least 600.000s is required"):
        montage.build_clip_ranges(duration_sec=599.0, segments=60, clip_duration_sec=10.0)


def test_ffmpeg_command_uses_independently_seeked_inputs_and_filter_concat(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    ranges = montage.build_clip_ranges(duration_sec=8.0, segments=2, clip_duration_sec=1.0)
    media = montage.MediaInfo(duration_sec=8.0, fps=10.0, width=160, height=90, has_audio=True)

    command = montage.build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
        media=media,
        ranges=ranges,
        crf=20,
        preset="fast",
        audio_bitrate="192k",
    )
    filter_graph = command[command.index("-filter_complex") + 1]

    assert command.count("-i") == 2
    assert command.count(input_path.as_posix()) == 2
    assert "[0:v:0]trim=duration=1.000000000,setpts=PTS-STARTPTS[v0]" in filter_graph
    assert "[1:a:0]atrim=duration=1.000000000,asetpts=PTS-STARTPTS[a1]" in filter_graph
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]" in filter_graph


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_cli_creates_expected_montage_with_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "montage.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=6",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            source.as_posix(),
        ],
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH.as_posix(),
            "--input",
            source.as_posix(),
            "--output",
            output.as_posix(),
            "--segments",
            "3",
            "--clip-duration",
            "0.5",
            "--preset",
            "ultrafast",
            "--crf",
            "28",
            "--verify-decode",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            output.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    stream_types = {stream["codec_type"] for stream in payload["streams"]}
    video_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    assert {"video", "audio"}.issubset(stream_types)
    assert float(payload["format"]["duration"]) == pytest.approx(1.5, abs=0.1)
    assert int(video_stream["nb_frames"]) == 15
    assert float(audio_stream["duration"]) == pytest.approx(1.5, abs=0.1)
