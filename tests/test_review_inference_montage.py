from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.review_inference_montage import (
    CliError,
    discover_source_video,
    map_montage_position,
    review_date,
    session_directory,
)


def test_discovers_mida_recursively(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "MIDA-700 sample.mp4"
    source.parent.mkdir()
    source.touch()

    assert discover_source_video(tmp_path, "MIDA") == source.resolve()


def test_discovery_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    (tmp_path / "MIDA-001.mp4").touch()
    (tmp_path / "MIDA-002.mkv").touch()

    with pytest.raises(CliError, match="Multiple source videos"):
        discover_source_video(tmp_path, "MIDA")


def test_maps_montage_frame_back_to_centered_source_clip() -> None:
    first = map_montage_position(
        montage_frame_index=0,
        montage_fps=30.0,
        source_duration_sec=7_200.0,
        source_fps=30.0,
        segments=60,
        clip_duration_sec=10.0,
    )
    second = map_montage_position(
        montage_frame_index=300,
        montage_fps=30.0,
        source_duration_sec=7_200.0,
        source_fps=30.0,
        segments=60,
        clip_duration_sec=10.0,
    )
    last = map_montage_position(
        montage_frame_index=17_999,
        montage_fps=30.0,
        source_duration_sec=7_200.0,
        source_fps=30.0,
        segments=60,
        clip_duration_sec=10.0,
    )

    assert first.segment_index == 0
    assert first.source_time_sec == pytest.approx(55.0)
    assert first.source_frame_index == 1_650
    assert second.segment_index == 1
    assert second.source_time_sec == pytest.approx(175.0)
    assert last.segment_index == 59
    assert last.source_time_sec == pytest.approx(7_144.9666667)


def test_output_directory_uses_first_three_characters_and_date(tmp_path: Path) -> None:
    source = tmp_path / "MIDA-700 sample.mp4"
    output = session_directory(source, tmp_path / "active_montage", "2026-07-22")

    assert output == (tmp_path / "active_montage" / "MID_2026-07-22").resolve()


def test_review_date_validates_iso_format() -> None:
    assert review_date("2026-07-22") == date(2026, 7, 22).isoformat()
    with pytest.raises(CliError, match="YYYY-MM-DD"):
        review_date("20260722")
