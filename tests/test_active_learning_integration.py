from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.active_learning.adapters import ModelAdapter
from src.active_learning.config import ModelConfig, MontageConfig, PipelineConfig
from src.active_learning.pipeline import run_export, run_inventory
from src.active_learning.render import BELOW_THRESHOLD_COLOR, _draw_panel, concatenate_review_clips, cut_source_clip, render_review_clip
from src.active_learning.types import Candidate, Prediction


def test_panel_header_does_not_cover_the_source_frame() -> None:
    frame = np.full((90, 160, 3), 255, dtype=np.uint8)
    panel = _draw_panel(
        frame, [], width=320, height=180, title="MODEL", detail="detail", color=(0, 255, 0)
    )
    # The 72-pixel header is separate, and the complete 16:9 frame is scaled
    # into the 108-pixel-tall content area below it.
    assert np.all(panel[72, 64] == 255)
    assert np.all(panel[179, 255] == 255)
    assert np.all(panel[72, 63] == 0)


def test_panel_draws_below_threshold_bbox_in_orange() -> None:
    frame = np.full((90, 160, 3), 255, dtype=np.uint8)
    prediction = Prediction(
        model_name="model", video_id="video", timestamp_sec=0, class_id=2,
        confidence=.4, bbox_xyxy_original=(10, 10, 40, 40), input_width=160,
        input_height=90, original_width=160, original_height=90,
    )
    panel = _draw_panel(
        frame, [prediction], width=320, height=180, title="MODEL", detail="detail",
        color=(0, 255, 0), analysis_threshold=.5,
    )
    assert np.any(np.all(panel == np.asarray(BELOW_THRESHOLD_COLOR), axis=2))


@pytest.mark.integration
def test_inventory_reads_small_generated_video(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required")
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    video = input_dir / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
        "-pix_fmt", "yuv420p", video.as_posix(),
    ], check=True)
    config = PipelineConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        models=[],
        montage=MontageConfig(width=320, height=180, fps=10, codec="libx264", crf=24, preset="ultrafast", review_inference_fps=1),
    )
    inventory = run_inventory(config)
    assert len(inventory) == 1
    assert inventory[0].readable
    assert inventory[0].width == 160 and inventory[0].height == 90
    source_clip = config.output_dir / "source_clips" / "clip_0001.mp4"
    review_clip = config.output_dir / "review_clips" / "clip_0001_grid.mp4"
    center_image = config.output_dir / "review_images" / "clip_0001_center.jpg"
    candidate = Candidate(
        video_id="sample", filename=video.name, video_path=video.as_posix(), timestamp_sec=0.5,
        scene_id=0, perceptual_hash="0" * 16, clip_id="clip_0001", selection_bucket="temporal_random",
    )
    cut_source_clip(video, source_clip, start_sec=0.0, duration_sec=1.0, config=config, clip_id=candidate.clip_id)
    render_review_clip(source_clip, review_clip, center_image, candidate, [], config, 0.0, duration_sec=1.0)
    montage = config.output_dir / "review_montage.mp4"
    concatenate_review_clips([review_clip], montage, config)
    assert source_clip.is_file() and review_clip.is_file() and center_image.is_file() and montage.is_file()
    import cv2
    capture = cv2.VideoCapture(review_clip.as_posix())
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    capture.release()


@pytest.mark.integration
def test_frame_export_uses_dataset_folder_and_original_resolution(monkeypatch, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required")

    class FakeAdapter(ModelAdapter):
        def predict_batch(self, frames_bgr, *, video_id, timestamps, return_embeddings=False):
            return [[] for _ in frames_bgr], None

    dataset_name = "manual_unlabelled"
    input_dir = tmp_path / "raw" / dataset_name
    dataset_output = tmp_path / "output" / dataset_name
    input_dir.mkdir(parents=True)
    video = input_dir / "sample source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "testsrc=size=160x90:rate=10:duration=2", "-pix_fmt", "yuv420p", video.as_posix(),
    ], check=True)
    dummy_weight = tmp_path / "fake.pt"
    dummy_weight.write_bytes(b"fake")
    model = ModelConfig(name="fake_model", type="fake", weights=dummy_weight)
    config = PipelineConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        input_dir=input_dir,
        output_dir=dataset_output / "_active_learning",
        dataset_name=dataset_name,
        dataset_output_dir=dataset_output,
        models=[model],
        context_half_window_sec=.5,
        montage=MontageConfig(
            width=320, height=180, fps=5, codec="libx264", crf=24,
            preset="ultrafast", review_inference_fps=1,
        ),
    )
    inventory = run_inventory(config)
    selected = [Candidate(
        video_id=inventory[0].video_id,
        filename=video.name,
        video_path=video.as_posix(),
        timestamp_sec=1.0,
        frame_index=10,
        scene_id=0,
        perceptual_hash="0" * 16,
        pixel_sha256="candidate-hash",
        clip_id="frame_000001",
        selection_bucket="uniform",
        uniform_score=1.0,
        selection_score=1.0,
        model_counts={"fake_model": 0},
        model_predictions={"fake_model": []},
        roi_source="full_frame",
        roi_sources=["full_frame"],
        roi_bbox=(0, 0, 160, 90),
    )]
    monkeypatch.setattr("src.active_learning.pipeline.create_adapter", lambda row: FakeAdapter(row))
    manifest = run_export(config, inventory, selected)
    frame_path = Path(manifest[0]["frame_path"])
    assert frame_path.parent == dataset_output
    assert "__f0000000010__t" in frame_path.name
    assert (config.output_dir / "selection_manifest.csv").is_file()
    import cv2
    frame = cv2.imread(frame_path.as_posix())
    assert frame is not None and frame.shape[:2] == (90, 160)
    user_cvat_zip = dataset_output / "cvat_export.zip"
    user_cvat_zip.write_bytes(b"user-owned")
    run_export(config, inventory, selected, force=True)
    assert user_cvat_zip.read_bytes() == b"user-owned"


@pytest.mark.model_weights
def test_configured_model_adapters_with_local_weights() -> None:
    import gc
    import cv2
    import torch

    from src.active_learning.adapters import DinoDeimAdapter, create_adapter
    from src.active_learning.config import load_pipeline_config

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the configured local model integration test")
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "active_learning_frame_dataset.yaml"
    if not config_path.is_file():
        pytest.skip("Local active-learning config is not available")
    dataset_name = "active_learning(dataset=3000frame)7-17"
    input_dir = (
        project_root / "data" / "input" / "active_learning" / dataset_name
    )
    if not input_dir.is_dir():
        pytest.skip("The configured legacy active-learning input was pruned locally")
    config = load_pipeline_config(config_path, dataset_name)
    video_paths = sorted(config.input_dir.glob("*.mp4"))
    if not video_paths:
        pytest.skip("A configured local input video is required")
    capture = cv2.VideoCapture(video_paths[0].as_posix())
    capture.set(cv2.CAP_PROP_POS_MSEC, 300_000.0)
    okay, frame = capture.read()
    capture.release()
    assert okay and frame is not None
    model_configs = [*config.models, *([config.head_model] if config.head_model is not None else [])]
    for model_config in model_configs:
        adapter = create_adapter(model_config)
        rows, embeddings = adapter.predict_batch(
            [frame],
            video_id="model_weight_test",
            timestamps=[300.0],
            return_embeddings=isinstance(adapter, DinoDeimAdapter),
        )
        assert len(rows) == 1
        for prediction in rows[0]:
            assert prediction.original_width == frame.shape[1]
            assert prediction.original_height == frame.shape[0]
            assert len(prediction.bbox_xyxy_original) == 4
        if isinstance(adapter, DinoDeimAdapter):
            assert embeddings is not None and embeddings.shape[0] == 1 and embeddings.shape[1] > 0
            roi_embeddings = adapter.embed_batch([frame], input_size=config.novelty.roi_input_size)
            assert roi_embeddings.shape[0] == 1 and roi_embeddings.shape[1] > 0
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
