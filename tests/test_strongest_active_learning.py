from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import pytest

from script.script_data.clip_frame import CliError, read_video_history
from src.active_learning.media import probe_video
from src.active_learning.strongest_config import (
    CATEGORIES,
    InferenceSettings,
    ModelSpec,
    MontageSettings,
    OutputSettings,
    SelectionSettings,
    StrongestPipelineConfig,
    load_strongest_pipeline_config,
)
from src.active_learning.strongest_pipeline import (
    MontageSegment,
    VideoWork,
    _prediction_model_contract,
    align_source_timestamps_to_frame_grid,
    build_failure_candidates,
    export_selected_frames,
    map_montage_timestamp,
    maximum_spaced_candidate_count,
    preflight_source_video_names,
    quota_counts,
    record_completed_source_video_names,
    select_failure_candidates,
    temporal_ellipse_events,
    write_report,
)
from src.active_learning.strongest_types import (
    FaceObservation,
    FailureCandidate,
    InferenceFrame,
)


def model_spec(tmp_path: Path) -> ModelSpec:
    return ModelSpec(
        model_id="fake_strongest",
        adapter_class="fake.module:Adapter",
        capabilities=frozenset({"face", "ellipse", "keypoints"}),
        assets={},
        options={},
        face_confidence=0.33,
        keypoint_confidence=0.50,
        inference_confidence_floor=0.05,
    )


def pipeline_config(
    tmp_path: Path,
    *,
    selection: SelectionSettings | None = None,
) -> StrongestPipelineConfig:
    return StrongestPipelineConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        registry_path=tmp_path / "registry.yaml",
        dataset_name="dataset",
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        model=model_spec(tmp_path),
        device="cpu",
        random_seed=17,
        montage=MontageSettings(),
        inference=InferenceSettings(),
        selection=selection or SelectionSettings(),
        output=OutputSettings(jpeg_quality=90),
    )


def observation(
    *,
    confidence: float = 0.9,
    ellipse: tuple[float, float, float, float, float] = (50, 50, 10, 7, 0),
    keypoint_scores: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9, 0.9),
) -> FaceObservation:
    return FaceObservation(
        confidence=confidence,
        bbox_xyxy=(
            ellipse[0] - ellipse[2],
            ellipse[1] - ellipse[2],
            ellipse[0] + ellipse[2],
            ellipse[1] + ellipse[2],
        ),
        ellipse=ellipse,
        keypoints=((45, 47), (55, 47), (50, 51), (46, 56), (54, 56)),
        keypoint_scores=keypoint_scores,
    )


def inference_frame(
    index: int,
    observations: list[FaceObservation],
    *,
    segment: int = 0,
    scene: int = 0,
) -> InferenceFrame:
    return InferenceFrame(
        montage_frame_id=index,
        montage_timestamp_sec=index * 0.2,
        source_segment=segment,
        source_timestamp_sec=index * 0.2,
        source_frame_id=index * 6,
        width=100,
        height=100,
        scene_id=scene,
        pixel_sha256=f"{index:064x}",
        observations=observations,
    )


def test_config_uses_canonical_active_learning_io() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_name = "active_learning08-02-4-4"
    config = load_strongest_pipeline_config(
        root / "configs/active_learning_strongest.yaml",
        dataset_name=dataset_name,
    )
    assert config.input_dir == (
        root
        / "data/input/active_learning"
        / dataset_name
    )
    assert config.output_dir == (
        root
        / "data/output/active_learning"
        / dataset_name
    )
    assert config.montage.timeline_segments == 10
    assert config.montage.duration_sec == 600
    assert config.selection.min_interval_sec == 10.0
    assert config.model.model_id == "deimv2_l_query_refinement"


def test_video_worker_count_must_be_positive(tmp_path: Path) -> None:
    config = pipeline_config(tmp_path)
    invalid = replace(
        config,
        inference=replace(config.inference, video_workers=0),
    )
    with pytest.raises(ValueError, match="video workers must be positive"):
        invalid.validate()


def test_source_video_names_are_recorded_after_completion(
    tmp_path: Path,
) -> None:
    input_dir = (
        tmp_path / "data/input/active_learning/new_active_learning_batch"
    )
    input_dir.mkdir(parents=True)
    videos = [input_dir / "Example-001.mp4", input_dir / "Example-002.mp4"]
    for index, path in enumerate(videos):
        path.write_bytes(f"video-{index}".encode("ascii"))
    output_dir = (
        tmp_path / "data/output/active_learning/new_active_learning_batch"
    )
    history_csv = tmp_path / "data/video_name_history.csv"

    initial = preflight_source_video_names(
        videos,
        history_csv=history_csv,
        project_root=tmp_path,
        allow_known_names=False,
    )
    assert initial["allowed_known_names"] == []
    assert not history_csv.exists()

    result = record_completed_source_video_names(
        videos,
        input_dir=input_dir,
        output_dir=output_dir,
        history_csv=history_csv,
        project_root=tmp_path,
    )
    assert result["recorded_videos"] == 2
    assert len(read_video_history(history_csv)) == 2
    run_rows = read_video_history(output_dir / "video_names.csv")
    assert {row["video_name"] for row in run_rows} == {
        "Example-001.mp4",
        "Example-002.mp4",
    }
    assert {
        row["evidence_source"] for row in run_rows
    } == {"active_learning_strongest"}

    with pytest.raises(CliError, match="過去に使用した動画名"):
        preflight_source_video_names(
            videos,
            history_csv=history_csv,
            project_root=tmp_path,
            allow_known_names=False,
        )
    allowed = preflight_source_video_names(
        videos,
        history_csv=history_csv,
        project_root=tmp_path,
        allow_known_names=True,
    )
    assert len(allowed["allowed_known_names"]) == 2


def test_source_video_name_preflight_rejects_current_batch_collisions(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first/Same Name.mp4"
    second = tmp_path / "second/same name.MOV"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    with pytest.raises(CliError, match="今回の入力内"):
        preflight_source_video_names(
            [first, second],
            history_csv=tmp_path / "data/video_name_history.csv",
            project_root=tmp_path,
            allow_known_names=False,
        )


def test_prediction_contract_ignores_only_threshold_source_document() -> None:
    base = {
        "model_id": "model",
        "assets": {
            "detector_checkpoint": {"sha256": "detector"},
            "query_checkpoint": {"sha256": "query"},
            "threshold_selection_config": {"sha256": "old-doc"},
        },
        "thresholds": {
            "inference_confidence_floor": 0.05,
            "face_confidence": 0.33,
            "keypoint_confidence": 0.50,
            "provenance": {"selection_split": "val"},
        },
    }
    documentation_changed = {
        **base,
        "assets": {
            **base["assets"],
            "threshold_selection_config": {"sha256": "new-doc"},
        },
    }
    assert _prediction_model_contract(base) == _prediction_model_contract(
        documentation_changed
    )
    checkpoint_changed = {
        **base,
        "assets": {
            **base["assets"],
            "query_checkpoint": {"sha256": "different-query"},
        },
    }
    assert _prediction_model_contract(base) != _prediction_model_contract(
        checkpoint_changed
    )


def test_montage_timestamp_maps_back_to_the_original_segment() -> None:
    segments = tuple(
        MontageSegment(
            index=index,
            source_start_sec=1000.0 + index * 200.0,
            source_end_sec=1060.0 + index * 200.0,
            montage_start_sec=index * 60.0,
            montage_end_sec=(index + 1) * 60.0,
        )
        for index in range(10)
    )
    assert map_montage_timestamp(0.0, segments) == (0, 1000.0)
    assert map_montage_timestamp(65.25, segments) == (1, 1205.25)
    assert map_montage_timestamp(599.8, segments) == pytest.approx((9, 2859.8))


def test_selection_timestamp_uses_the_source_frame_grid() -> None:
    frames = [
        inference_frame(0, []),
        inference_frame(1, []),
    ]
    frames[0].source_frame_id = 73071
    frames[0].source_timestamp_sec = 2438.12819615
    frames[1].source_frame_id = 73370
    frames[1].source_timestamp_sec = 2448.12819615
    aligned = align_source_timestamps_to_frame_grid(frames, 29.97)
    assert aligned[0].source_timestamp_sec == pytest.approx(2438.138138)
    assert (
        aligned[1].source_timestamp_sec
        - aligned[0].source_timestamp_sec
    ) == pytest.approx(9.9766433)


def test_middle_missing_and_ellipse_jump_are_temporal_candidates(
    tmp_path: Path,
) -> None:
    config = pipeline_config(tmp_path)
    frames = [
        inference_frame(0, [observation(ellipse=(30, 50, 10, 7, 0))]),
        inference_frame(1, [observation(ellipse=(31, 50, 10, 7, 0))]),
        inference_frame(2, [observation(ellipse=(32, 50, 10, 7, 0))]),
        inference_frame(3, []),
        inference_frame(4, [observation(ellipse=(33, 50, 10, 7, 0))]),
        inference_frame(5, [observation(ellipse=(65, 50, 18, 5, 1.0))]),
    ]
    scores, reasons, _ = temporal_ellipse_events(frames, config)
    assert scores[3] == 1.0
    assert "middle_missing" in reasons[3]
    assert scores[5] > 0
    assert {
        "ellipse_center_jump",
        "ellipse_axis_jump",
        "ellipse_angle_jump",
    } & set(reasons[5])


def test_temporal_tracking_does_not_cross_a_scene_or_montage_cut(
    tmp_path: Path,
) -> None:
    config = pipeline_config(tmp_path)
    frames = [
        inference_frame(0, [observation()]),
        inference_frame(1, [observation()]),
        inference_frame(2, [], scene=1),
        inference_frame(3, [observation()], scene=1),
        inference_frame(4, [], segment=1, scene=0),
        inference_frame(5, [observation()], segment=1, scene=0),
    ]
    scores, _, _ = temporal_ellipse_events(frames, config)
    assert scores == {}


def test_four_requested_failure_conditions_are_scored(tmp_path: Path) -> None:
    config = pipeline_config(tmp_path)
    frames = [
        inference_frame(
            0,
            [
                observation(
                    confidence=0.33,
                    ellipse=(4, 50, 5, 3, 0),
                    keypoint_scores=(0.1, 0.2, 0.1, 0.2, 0.1),
                )
            ],
        )
    ]
    candidates = build_failure_candidates(frames, config)
    assert len(candidates) == 1
    row = candidates[0]
    assert row.scores["low_confidence"] == pytest.approx(1.0)
    assert row.scores["face_without_keypoints"] > 0
    assert row.scores["extreme_screen_edge"] > 0
    assert row.details["extreme_screen_edge"]["edge_margin_ratio"] == 0.01


def _exclusive_candidate(index: int, category: str) -> FailureCandidate:
    scores = {name: 0.0 for name in CATEGORIES}
    scores[category] = 1.0 - index / 10000
    return FailureCandidate(
        frame=inference_frame(index, [observation()]),
        scores=scores,
    )


def test_selection_targets_25_percent_each_for_50_frames(
    tmp_path: Path,
) -> None:
    selection = replace(SelectionSettings(), min_interval_sec=0.0)
    config = pipeline_config(tmp_path, selection=selection)
    rows = [
        _exclusive_candidate(category_index * 100 + index, category)
        for category_index, category in enumerate(CATEGORIES)
        for index in range(20)
    ]
    selected, summary = select_failure_candidates(rows, config)
    assert len(selected) == 50
    assert quota_counts(50, selection.category_ratios) == {
        "temporal_ellipse_instability": 13,
        "low_confidence": 13,
        "face_without_keypoints": 12,
        "extreme_screen_edge": 12,
    }
    assert summary["actual"] == summary["quota"]


def test_missing_categories_are_refilled_only_from_requested_conditions(
    tmp_path: Path,
) -> None:
    selection = replace(SelectionSettings(), min_interval_sec=0.0)
    config = pipeline_config(tmp_path, selection=selection)
    rows = [
        _exclusive_candidate(index, "extreme_screen_edge") for index in range(50)
    ]
    selected, summary = select_failure_candidates(rows, config)
    assert len(selected) == 50
    assert all(row.assigned_category == "extreme_screen_edge" for row in selected)
    assert summary["refill_count"] > 0
    assert all(row.matched_categories for row in selected)


def test_maximum_spaced_capacity_is_not_limited_by_configured_50(
    tmp_path: Path,
) -> None:
    config = pipeline_config(
        tmp_path,
        selection=replace(SelectionSettings(), min_interval_sec=1.0),
    )
    rows = [
        _exclusive_candidate(index * 10, "low_confidence")
        for index in range(75)
    ]
    for index, row in enumerate(rows):
        row.frame.source_timestamp_sec = index * 1.1
    selected, summary = select_failure_candidates(rows, config)
    assert len(selected) == 50
    assert maximum_spaced_candidate_count(
        rows,
        min_interval_sec=1.0,
    ) == 75
    assert summary["maximum_after_exact_dedup_and_min_interval"] == 75


@pytest.mark.integration
def test_export_name_uses_original_video_and_frame_id(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    source_path = tmp_path / "source name.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=2",
            "-pix_fmt",
            "yuv420p",
            source_path.as_posix(),
        ],
        check=True,
    )
    source = probe_video(source_path)
    assert source.readable
    config = pipeline_config(tmp_path)
    work = VideoWork(
        source=source,
        work_id="source",
        work_dir=config.output_dir / "_active_learning/videos/source",
        montage_path=tmp_path / "unused.mp4",
        montage_segments=(),
    )
    frame = InferenceFrame(
        montage_frame_id=1,
        montage_timestamp_sec=1.0,
        source_segment=0,
        source_timestamp_sec=1.0,
        source_frame_id=10,
        width=160,
        height=90,
        scene_id=0,
        pixel_sha256="unique",
        observations=[observation(ellipse=(80, 45, 12, 8, 0))],
    )
    candidate = FailureCandidate(
        frame=frame,
        scores={name: float(name == "low_confidence") for name in CATEGORIES},
        details={"low_confidence": {"confidence": 0.33}},
        assigned_category="low_confidence",
        selection_score=1.0,
        selection_note="test",
    )
    manifest = export_selected_frames(config, work, [candidate])
    assert len(manifest) == 1
    assert manifest[0]["frame_id"].startswith("source_name__frame_")
    assert Path(manifest[0]["frame_path"]).is_file()
    assert Path(manifest[0]["evidence_path"]).is_file()
    write_report(
        config,
        manifest,
        {source.filename: {"selected": 1}},
        {"model_id": config.model.model_id},
    )
    document = (config.output_dir / "selection_report.html").read_text(
        encoding="utf-8"
    )
    assert "低confidence" in document
    assert "対象frame（原画像）" in document
