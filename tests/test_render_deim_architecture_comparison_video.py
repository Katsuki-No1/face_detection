from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_deim_architecture_comparison_video.py"
SPEC = importlib.util.spec_from_file_location("render_deim_architecture_comparison_video", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_production_config_has_fixed_four_panel_contract() -> None:
    config = MODULE.load_config(
        ROOT / "configs/video_comparison/deim_architecture_four_panel.yaml"
    )

    assert tuple(config.models) == MODULE.MODEL_ROLES
    assert config.requested_input == "data/output/active_learning/active_montage"
    assert config.input_path == (
        ROOT
        / "data/output/active_learning/active_montage/MID_2026-07-22/MIDA-700_uniform_10m.mp4"
    ).resolve()
    assert config.models["query_refinement"]["detector_source"] == "ellipse_deim"
    assert config.models["orformer"]["detector_source"] == "ellipse_deim"
    assert config.models["normal_deim"]["score_threshold"] == pytest.approx(0.30)
    assert config.models["ellipse_deim"]["score_threshold"] == pytest.approx(0.33)
    assert config.models["normal_deim"]["inference_batch_size"] == "auto"
    assert config.models["ellipse_deim"]["inference_batch_size"] == "auto"
    assert config.models["normal_deim"]["batch_candidates"] == [1, 2, 4, 8, 16, 32]
    assert config.models["ellipse_deim"]["batch_candidates"] == [1, 2, 4, 8, 16, 32]


def test_discover_videos_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.mp4").touch()
    (tmp_path / "a.MOV").touch()
    (tmp_path / "ignored.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.mkv").touch()

    shallow = MODULE.discover_videos(
        tmp_path, recursive=False, extensions=(".mp4", ".mov", ".mkv")
    )
    recursive = MODULE.discover_videos(
        tmp_path, recursive=True, extensions=(".mp4", ".mov", ".mkv")
    )

    assert [path.name for path in shallow] == ["a.MOV", "b.mp4"]
    assert [path.name for path in recursive] == ["a.MOV", "b.mp4", "c.mkv"]


def test_ellipse_polygon_uses_radii_and_rotation() -> None:
    polygon = MODULE._ellipse_polygon(
        [50.0, 40.0, 20.0, 10.0, 0.0],
        scale=1.0,
        offset_x=0.0,
        offset_y=0.0,
    )

    assert tuple(polygon.min(axis=0)) == (30, 30)
    assert tuple(polygon.max(axis=0)) == (70, 50)


def test_draw_panel_supports_all_three_visualization_modes() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    prediction = {
        "detector_bbox_xyxy": [20.0, 10.0, 100.0, 90.0],
        "ellipse": [60.0, 50.0, 35.0, 25.0, 0.2],
        "keypoints": [[45.0, 40.0], [75.0, 40.0], [60.0, 55.0], [50.0, 70.0], [70.0, 70.0]],
        "visibility": [0.9, 0.8, 0.7, 0.4, 0.3],
    }
    render = MODULE.RenderConfig(width=640, height=360)

    panels = [
        MODULE.draw_panel(
            frame,
            [prediction],
            width=320,
            height=180,
            title=mode,
            detail="test",
            mode=mode,
            visibility_threshold=0.5,
            render=render,
        )
        for mode in ("bbox", "ellipse", "landmarks")
    ]

    assert all(panel.shape == (180, 320, 3) for panel in panels)
    assert all(np.count_nonzero(panel) > 0 for panel in panels)
    assert not np.array_equal(panels[0], panels[1])
    assert not np.array_equal(panels[1], panels[2])


def test_dry_run_plan_does_not_probe_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "fake.mp4"
    video.touch()
    config = MODULE.load_config(
        ROOT / "configs/video_comparison/deim_architecture_four_panel.yaml"
    )
    monkeypatch.setattr(MODULE, "probe_video", lambda _path: pytest.fail("must not probe"))

    plan = MODULE.dry_run_plan(config, [video], tmp_path / "out")

    assert plan["mode"].startswith("dry-run")
    assert plan["ellipse_detector_reused_by"] == ["query_refinement", "orformer"]
    assert plan["stage_order"][-1] == "render"
    assert plan["detector_batching"]["normal_deim"]["inference_batch_size"] == "auto"
    assert plan["detector_batching"]["ellipse_deim"]["batch_candidates"][-1] == 32


def test_deim_face_detector_predict_batch_preserves_single_image_contract() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, batch: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            self.batch_sizes.append(int(batch.shape[0]))
            return [
                {
                    "boxes": torch.tensor(
                        [[0.0, 0.0, 8.0, 8.0]],
                        device=batch.device,
                    ),
                    "scores": torch.tensor([0.9], device=batch.device),
                    "labels": torch.tensor([1], device=batch.device),
                }
                for _ in range(batch.shape[0])
            ]

    detector = object.__new__(DeimFaceDetector)
    detector.device = torch.device("cpu")
    detector.image_size = 8
    detector.score_threshold = 0.3
    detector.maximum_detections = 100
    detector.crop_geometry = "bbox"
    detector.model = FakeModel()
    images = [
        np.full((4, 6, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20)
    ]

    batch_rows = detector.predict_batch(images)
    single_rows = detector.predict(images[0])

    assert detector.model.batch_sizes == [2, 1]
    assert len(batch_rows) == 2
    assert batch_rows[0][0]["bbox_xyxy"] == pytest.approx([0.0, 0.0, 6.0, 4.0])
    assert batch_rows[0][0]["crop_geometry"] == "bbox"
    assert single_rows == batch_rows[0]


def test_detector_batch_selection_records_maximum_safe_separately() -> None:
    selected, maximum_safe = MODULE._select_detector_batch_attempt(
        [
            {"batch_size": 1, "status": "ok", "frames_per_second": 10.0},
            {"batch_size": 2, "status": "ok", "frames_per_second": 17.0},
            {"batch_size": 4, "status": "ok", "frames_per_second": 21.0},
            {"batch_size": 8, "status": "ok", "frames_per_second": 19.0},
            {"batch_size": 16, "status": "cuda_oom"},
        ],
        strategy="throughput",
    )

    assert selected["batch_size"] == 4
    assert maximum_safe == 8


def test_detector_runtime_oom_fallback_preserves_frame_order() -> None:
    class OomAboveTwoDetector:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def predict_batch(
            self,
            images: list[np.ndarray] | tuple[np.ndarray, ...],
        ) -> list[list[dict[str, int]]]:
            self.calls.append(len(images))
            if len(images) > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return [
                [{"frame_marker": int(image[0, 0, 0])}]
                for image in images
            ]

    detector = OomAboveTwoDetector()
    images = [
        np.full((2, 2, 3), fill_value=index, dtype=np.uint8)
        for index in range(8)
    ]

    outputs, effective_batch_size, events = (
        MODULE._predict_detector_batch_with_fallback(
            detector,
            images,
            batch_size=8,
        )
    )

    assert detector.calls == [8, 4, 2, 2, 2, 2]
    assert effective_batch_size == 2
    assert events == [
        {"failed_batch_size": 8, "retry_batch_size": 4},
        {"failed_batch_size": 4, "retry_batch_size": 2},
    ]
    assert [row[0]["frame_marker"] for row in outputs] == list(range(8))


def test_detector_auto_tuning_persists_and_reuses_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "detector.yaml"
    checkpoint_path = tmp_path / "detector.pt"
    config_path.write_text("model: {}\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    detector = type("FakeDetector", (), {"device": torch.device("cpu")})()
    model_cfg = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "input_size": 960,
        "auto_batch_selection": "throughput",
        "gpu_memory_headroom_ratio": 0.15,
        "auto_batch_warmup_iterations": 1,
        "auto_batch_benchmark_iterations": 2,
    }
    monkeypatch.setattr(
        MODULE,
        "_read_detector_tuning_frames",
        lambda *_args, **_kwargs: [
            np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(32)
        ],
    )
    throughput = {1: 10.0, 2: 17.0, 4: 21.0, 8: 19.0}

    def benchmark(_detector, images, **_kwargs):
        batch_size = len(images)
        if batch_size >= 16:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {
            "batch_size": batch_size,
            "iterations": 2,
            "elapsed_seconds": 1.0,
            "frames_per_second": throughput[batch_size],
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "total_memory_bytes": None,
            "memory_within_headroom": True,
        }

    monkeypatch.setattr(MODULE, "_benchmark_detector_batch", benchmark)
    first = MODULE._auto_tune_detector_batch(
        detector,
        video=tmp_path / "unused.mp4",
        start_frame=0,
        remaining_frames=100,
        output_dir=tmp_path / "output",
        role="normal_deim",
        model_cfg=model_cfg,
        candidates=(1, 2, 4, 8, 16, 32),
        force=False,
    )

    assert first.selected_batch_size == 4
    assert first.maximum_safe_batch_size == 8
    assert first.source == "benchmark"
    assert first.tuning_path is not None and first.tuning_path.is_file()

    monkeypatch.setattr(
        MODULE,
        "_read_detector_tuning_frames",
        lambda *_args, **_kwargs: pytest.fail("valid tuning result must be reused"),
    )
    second = MODULE._auto_tune_detector_batch(
        detector,
        video=tmp_path / "unused.mp4",
        start_frame=0,
        remaining_frames=100,
        output_dir=tmp_path / "output",
        role="normal_deim",
        model_cfg=model_cfg,
        candidates=(1, 2, 4, 8, 16, 32),
        force=False,
    )

    assert second.selected_batch_size == 4
    assert second.maximum_safe_batch_size == 8
    assert second.source == "cache"


def test_detector_batch_config_requires_batch_one_fallback() -> None:
    with pytest.raises(ValueError, match="include 1"):
        MODULE._detector_batch_candidates(
            {
                "inference_batch_size": "auto",
                "batch_candidates": [2, 4, 8],
            }
        )
