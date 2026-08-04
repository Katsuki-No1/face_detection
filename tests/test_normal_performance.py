from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.normal_performance.pipeline import (
    AnchorFeature,
    NormalPerformanceConfig,
    VideoMetadata,
    iter_motion_bursts,
    score_windows,
)
from src.normal_performance.scoring import (
    build_window_starts,
    invalid_penalty,
    normal_band_points,
    normalized_js_similarity,
    presence_points,
    representative_points,
    robust_variation,
    stable_percentiles,
)


def test_window_starts_include_exact_last_boundary() -> None:
    assert build_window_starts(1_200, 600, 60) == [float(value) for value in range(0, 601, 60)]
    assert build_window_starts(599.99, 600, 60) == []


def test_scoring_boundaries() -> None:
    assert presence_points(0.3999) == 0
    assert presence_points(0.40) == 1
    assert presence_points(0.60) == 2
    assert presence_points(0.80) == 3
    assert representative_points(0.2499) == 0
    assert representative_points(0.25) == 1
    assert representative_points(0.50) == 2
    assert representative_points(0.75) == 3
    assert normal_band_points(0.05) == 0
    assert normal_band_points(0.20) == 1
    assert normal_band_points(0.50) == 2
    assert normal_band_points(0.95) == 1
    assert normal_band_points(0.99) == 0
    assert invalid_penalty(0) == 0
    assert invalid_penalty(3) == -3
    assert invalid_penalty(10) == -5


def test_stable_percentiles_average_ties() -> None:
    assert stable_percentiles([0.2, 0.1, 0.2]).tolist() == [0.75, 0.0, 0.75]


def test_distribution_similarity_and_robust_variation() -> None:
    assert normalized_js_similarity(np.asarray([1, 1]), np.asarray([1, 1])) == pytest.approx(1.0)
    assert normalized_js_similarity(np.asarray([1, 0]), np.asarray([0, 1])) == pytest.approx(0.0)
    global_values = np.column_stack([np.arange(20), np.arange(20) * 2]).astype(np.float64)
    narrow = global_values[5:9]
    broad = global_values[2:18]
    assert robust_variation(broad, global_values) > robust_variation(narrow, global_values)


def _config(tmp_path: Path, **overrides: object) -> NormalPerformanceConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    weights = tmp_path / "weights.pt"
    weights.touch(exist_ok=True)
    values: dict[str, object] = {
        "project_root": tmp_path,
        "config_path": tmp_path / "config.yaml",
        "input_dir": input_dir,
        "output_dir": tmp_path / "output",
        "weights": weights,
        "anchor_interval_sec": 60.0,
        "motion_offsets_sec": (0.0, 0.5, 1.0),
        "window_duration_sec": 600.0,
        "window_step_sec": 60.0,
        "cluster_count": 3,
    }
    values.update(overrides)
    config = NormalPerformanceConfig(**values)
    config.validate()
    return config


def _anchor(index: int) -> AnchorFeature:
    return AnchorFeature(
        index=index,
        timestamp_sec=index * 60.0,
        image_path=f"selection/anchor_images/{index}.jpg",
        brightness=100.0,
        contrast=30.0,
        blur_score=100.0,
        max_scene_change=0.01,
        face_count=1,
        face_present=True,
        invalid=index == 0,
        invalid_reasons=["black"] if index == 0 else [],
        primary_pose_size=(math.log(0.05 + index * 0.001), 0.4 + index * 0.005, index * 0.01, index * 0.005),
        motion_raw=0.01 + index * 0.001,
        detections=[],
    )


def test_window_scoring_is_complete_and_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = VideoMetadata(
        path=tmp_path / "input" / "video.mp4",
        video_id="video",
        duration_sec=1_200.0,
        width=1280,
        height=720,
        fps=24.0,
        frame_count=28_800,
        codec="h264",
        file_size=1,
        mtime_ns=1,
    )
    anchors = [_anchor(index) for index in range(20)]
    embeddings = np.random.default_rng(7).normal(size=(20, 512)).astype(np.float32)
    first = score_windows(video, config, anchors, embeddings)
    second = score_windows(video, config, anchors, embeddings)
    assert len(first) == 11
    assert [row.start_sec for row in first] == [row.start_sec for row in second]
    assert sum(row.selected for row in first) == 1
    assert first[0].rank == 1 and first[0].selected
    assert all(len(row.anchor_indices) == 10 for row in first)


@pytest.mark.integration
def test_ffmpeg_motion_burst_sampling(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is required")
    config = _config(tmp_path)
    video_path = tmp_path / "input" / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=size=160x90:rate=10:duration=122.2", "-pix_fmt", "yuv420p", video_path.as_posix(),
        ],
        check=True,
    )
    video = VideoMetadata(
        path=video_path,
        video_id="sample",
        duration_sec=122.2,
        width=160,
        height=90,
        fps=10.0,
        frame_count=1_222,
        codec="h264",
        file_size=video_path.stat().st_size,
        mtime_ns=video_path.stat().st_mtime_ns,
    )
    bursts = list(iter_motion_bursts(video, config))
    assert [timestamp for timestamp, _ in bursts] == [0.0, 60.0, 120.0]
    assert all(len(frames) == 3 for _, frames in bursts)
    assert all(frames[0].shape == (90, 160, 3) for _, frames in bursts)


@pytest.mark.model_weights
def test_local_yolo_pose_model_has_face_keypoints_and_embedding() -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    weights = root / "runs/yolo11s_pose_face_pose_drive_v1_960_20260714/weights/best.pt"
    videos = list((root / "data/raw_data/nonactive(dataset=3000frame)7-17").glob("*.mp4"))
    if not weights.is_file() or not videos:
        pytest.skip("local model and video are required")
    capture = cv2.VideoCapture(videos[0].as_posix())
    capture.set(cv2.CAP_PROP_POS_MSEC, 3_600_000)
    okay, frame = capture.read()
    capture.release()
    assert okay and frame is not None
    model = YOLO(weights.as_posix())
    result = model.predict(source=[frame], imgsz=960, conf=0.44, device=0, verbose=False)[0]
    embedding = model.embed(source=[frame], imgsz=960, device=0, verbose=False)[0]
    assert model.task == "pose" and model.names == {0: "face"}
    assert result.keypoints is not None and result.keypoints.data.shape[1:] == (5, 3)
    assert tuple(embedding.shape) == (512,)
