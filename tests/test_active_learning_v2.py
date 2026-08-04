from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.active_learning.config import ClipWindowConfig, DenseAnalysisConfig, MatchingConfig, ModelConfig, PipelineConfig, PresenceClassifierConfig, load_pipeline_config
from src.active_learning.presence import padded_roi, union_bbox
from src.active_learning.report import write_html_report
from src.active_learning.scoring import build_candidates, confidence_boundary_components, disagreement_score
from src.active_learning.selection import adaptive_clip_bounds, quota_counts, select_candidates, select_frame_candidates
from src.active_learning.tracking import temporal_score, track_consensus_frames
from src.active_learning.types import Candidate, FrameFeature, Prediction, VideoInfo
from src.active_learning.stages import build_dense_candidates, run_dense_stage
from src.active_learning.adapters import DinoDeimAdapter, ModelAdapter


def feature(timestamp: float, scene: int = 0) -> FrameFeature:
    return FrameFeature(
        video_id="video", filename="video.mp4", video_path="/video.mp4", timestamp_sec=timestamp,
        frame_index=int(timestamp * 10), original_width=100, original_height=100,
        brightness=100, contrast=20, blur_score=10, perceptual_hash=f"{int(timestamp * 100):016x}",
        scene_change_score=0, scene_id=scene,
    )


def prediction(timestamp: float, confidence: float = 0.8, box=(10, 10, 20, 20), model="a") -> dict:
    return {
        "model_name": model, "video_id": "video", "timestamp_sec": timestamp, "class_id": 2,
        "confidence": confidence, "bbox_xyxy_original": list(box), "input_width": 100, "input_height": 100,
        "original_width": 100, "original_height": 100,
    }


def candidate(index: int, *, reason: str, comparison: int, roi: str = "face") -> Candidate:
    return Candidate(
        video_id=f"video_{index % 8}", filename="video.mp4", video_path="/video.mp4",
        timestamp_sec=index * 31.0, scene_id=index, perceptual_hash=f"{index:016x}",
        disagreement_score=1.0 - index / 1000, temporal_anomaly_score=0.8,
        no_detection_evidence_score=0.7, novelty_score=0.6, low_presence_audit_score=0.9,
        comparison_detection_count=comparison, model_counts={"a": comparison},
        eligibility_reason=reason, roi_source=roi, track_id=f"track_{index}", track_group_id=f"track_{index}",
    )


def test_all_zero_is_not_automatically_low_confidence() -> None:
    rows = build_candidates(
        [feature(0.0), feature(1.0)], [], model_weights={"a": 1.0, "b": 1.0},
        matching=MatchingConfig(), temporal_window_sec=4, random_seed=1,
    )
    assert all(row.all_zero_detection for row in rows)
    assert all(row.common_difficulty_score == 0 for row in rows)


def test_confidence_boundary_prefers_predictions_near_each_model_threshold() -> None:
    thresholds = {"yolo": .20, "deim": .50}
    far_below = confidence_boundary_components(
        [prediction(0, confidence=.05, model="deim")], thresholds, min_ratio=.7, max_ratio=1.3
    )
    just_below = confidence_boundary_components(
        [prediction(0, confidence=.48, model="deim")], thresholds, min_ratio=.7, max_ratio=1.3
    )
    just_above = confidence_boundary_components(
        [prediction(0, confidence=.21, model="yolo")], thresholds, min_ratio=.7, max_ratio=1.3
    )
    assert far_below["score"] == 0
    assert .8 < just_below["score"] < 1 and just_below["side"] == "below"
    assert .8 < just_above["score"] < 1 and just_above["side"] == "above"


def test_model_registry_resolves_checkpoint_and_threshold_from_metrics(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "weights": weights.as_posix(),
        "fixed_threshold": .42,
    }), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"models": {"model_v2": {
        "name": "model", "adapter": "fake", "weights": weights.as_posix(),
        "threshold_source": metrics.as_posix(), "threshold_field": "fixed_threshold",
    }}}), encoding="utf-8")
    config_path = tmp_path / "active_learning.yaml"
    config_path.write_text(json.dumps({
        "project_root": tmp_path.as_posix(),
        "model_registry": registry.as_posix(),
        "input_dir": input_dir.as_posix(),
        "output_dir": (tmp_path / "output").as_posix(),
        "models": [{"ref": "model_v2"}],
    }), encoding="utf-8")
    config = load_pipeline_config(config_path)
    assert config.models[0].weights == weights
    assert config.models[0].analysis_threshold == .42
    assert config.models[0].threshold_source == metrics


def test_sparse_temporal_neighbors_do_not_cross_scene() -> None:
    rows = build_candidates(
        [feature(0.0, 0), feature(1.0, 1)], [prediction(0.0)],
        model_weights={"a": 1.0, "b": 1.0}, matching=MatchingConfig(), temporal_window_sec=4, random_seed=1,
    )
    assert rows[1].temporal_anomaly_score == 0


def test_tracking_marks_middle_gap_disappearance_shape_and_confidence_drop() -> None:
    frames = []
    detections = {
        0.0: [{"bbox_xyxy_original": [10, 10, 20, 20], "confidence": 0.9}],
        0.2: [{"bbox_xyxy_original": [11, 10, 21, 20], "confidence": 0.9}],
        0.4: [{"bbox_xyxy_original": [12, 10, 22, 20], "confidence": 0.9}],
        1.0: [{"bbox_xyxy_original": [30, 10, 46, 26], "confidence": 0.2}],
    }
    for timestamp in np.arange(0, 1.01, 0.2):
        value = round(float(timestamp), 1)
        frames.append({
            "timestamp_sec": value, "scene_id": 0, "original_width": 100, "original_height": 100,
            "detections": detections.get(value, []),
        })
    observations, summaries = track_consensus_frames(frames, min_hits=3, max_gap_sec=1.0)
    events = {name for row in observations for name in row["events"]}
    assert summaries[0]["confirmed"]
    assert {"middle_missing", "disappearance", "position_jump", "shape_jump", "confidence_drop"} <= events
    assert max(temporal_score(row["events"]) for row in observations) > 0


def test_tracking_resets_at_scene_boundary() -> None:
    frames = [
        {"timestamp_sec": 0.0, "scene_id": 0, "original_width": 100, "original_height": 100, "detections": [{"bbox_xyxy_original": [10, 10, 20, 20], "confidence": .9}]},
        {"timestamp_sec": 0.2, "scene_id": 1, "original_width": 100, "original_height": 100, "detections": [{"bbox_xyxy_original": [10, 10, 20, 20], "confidence": .9}]},
    ]
    observations, _ = track_consensus_frames(frames, min_hits=1, max_gap_sec=1)
    assert len({row["track_id"] for row in observations if not row["predicted"]}) == 2
    assert not any(row["events"] for row in observations)


def test_tracking_marks_reconnectable_id_break() -> None:
    frames = []
    for timestamp in (0.0, 0.2, 0.4, 1.6, 1.8, 2.0):
        frames.append({
            "timestamp_sec": timestamp, "scene_id": 0, "original_width": 100, "original_height": 100,
            "detections": [{"bbox_xyxy_original": [10, 10, 20, 20], "confidence": .9}],
        })
    observations, summaries = track_consensus_frames(frames, min_hits=3, max_gap_sec=1.0)
    assert len([row for row in summaries if row["confirmed"]]) == 2
    assert any("id_break" in row["events"] for row in observations)


def test_trailing_scene_edge_gap_is_not_disappearance() -> None:
    frames = []
    for index, timestamp in enumerate((0.0, .2, .4, .6, .8)):
        frames.append({
            "timestamp_sec": timestamp, "scene_id": 0, "original_width": 100, "original_height": 100,
            "detections": [{"bbox_xyxy_original": [10, 10, 20, 20], "confidence": .9}] if index < 3 else [],
        })
    observations, _ = track_consensus_frames(frames, min_hits=3, max_gap_sec=1)
    assert not any("disappearance" in row["events"] for row in observations)


def test_disagreement_responds_to_iou_shape_and_confidence() -> None:
    weights = {"a": 1.0, "b": 1.0}
    same = {
        "a": [{**prediction(0, model="a"), "confidence_percentile": .8}],
        "b": [{**prediction(0, model="b"), "confidence_percentile": .8}],
    }
    varied = {
        "a": same["a"],
        "b": [{**prediction(0, box=(13, 10, 27, 24), model="b"), "confidence_percentile": .1}],
    }
    assert disagreement_score(varied, weights, MatchingConfig()) > disagreement_score(same, weights, MatchingConfig())


def test_new_bucket_quota_is_exact() -> None:
    assert quota_counts(40, {
        "model_disagreement": .35, "temporal_anomaly": .30, "no_detection_evidence": .20,
        "roi_novelty": .10, "low_presence_audit": .05,
    }) == {
        "model_disagreement": 14, "temporal_anomaly": 12, "no_detection_evidence": 8,
        "roi_novelty": 4, "low_presence_audit": 2,
    }


def test_head_only_is_excluded_from_normal_bucket_but_allowed_in_audit() -> None:
    head_only = candidate(1, reason="audit_only", comparison=0, roi="head")
    face = candidate(2, reason="face_presence", comparison=0, roi="head")
    selected, _, shortages = select_candidates(
        [head_only, face], target_count=2,
        ratios={"no_detection_evidence": .5, "low_presence_audit": .5},
        max_per_video=8, min_per_video=0, min_interval_sec=0, max_per_scene=2,
        max_per_track=2, duplicate_phash_distance=-1,
    )
    selected_by_bucket = {row.selection_bucket: row for row in selected}
    assert selected_by_bucket["no_detection_evidence"].eligibility_reason == "face_presence"
    assert selected_by_bucket["low_presence_audit"].eligibility_reason == "audit_only"
    assert not shortages


def test_shortfall_is_reported_without_fabricating_candidate() -> None:
    selected, _, shortages = select_candidates(
        [candidate(1, reason="audit_only", comparison=0, roi="")], target_count=4,
        ratios={"no_detection_evidence": .75, "low_presence_audit": .25},
        max_per_video=8, min_per_video=0, min_interval_sec=0, max_per_scene=2,
        max_per_track=2, duplicate_phash_distance=-1,
    )
    assert selected == []
    assert shortages["bucket:no_detection_evidence"].startswith("selected 0")


def test_adaptive_clip_is_bounded_by_scene() -> None:
    start, end = adaptive_clip_bounds(
        center=9, active_start=8.5, active_end=9.5, scene_start=5, scene_end=12,
        min_duration=4, max_duration=10, padding=1,
    )
    assert 5 <= start < end <= 12
    assert end - start >= 4


def test_roi_rejects_tiny_box_and_never_falls_back_to_full_frame() -> None:
    assert padded_roi((1, 1, 10, 10), width=100, height=100, min_side=16) is None
    assert padded_roi((10, 10, 30, 30), width=100, height=100, padding=.25, min_side=16) == (5.0, 5.0, 35.0, 35.0)


def test_review_csv_javascript_uses_bom_crlf_and_new_labels(tmp_path: Path) -> None:
    path = tmp_path / "review.html"
    write_html_report(path, [{
        "frame_id": "frame_000001", "frame_path": (tmp_path / "frame.jpg").as_posix(),
        "context_clip_path": (tmp_path / "context.mp4").as_posix(),
        "overlay_clip_path": (tmp_path / "overlay.mp4").as_posix(),
        "overlay_image_path": (tmp_path / "overlay.jpg").as_posix(), "source_video_id": "video",
        "source_filename": "video.mp4", "timestamp_sec": 1, "primary_bucket": "other_instability",
        "selection_score": 1, "selection_reason": "test",
        "all_reasons": "other/bbox_instability;ranked by bbox_instability_score",
        "model_detection_counts_json": json.dumps({"model_a": 0, "model_b": 1}),
        "model_predictions_json": json.dumps({"model_a": [], "model_b": [prediction(1, model="model_b")]}),
    }], {"models": ["model_a", "model_b"]})
    document = path.read_text(encoding="utf-8")
    assert "missed_detection" in document
    assert "bbox_localization" in document
    assert "model_a" in document and "model_b" in document
    assert "抽出根拠" in document
    assert "低confidenceまたはbboxが不安定" in document
    assert "採用カテゴリ:</b> その他の不安定（評価値 1.0000）" in document
    assert "該当した判定根拠:</b> bboxの位置・大きさが不安定、bboxの不安定度で優先" in document
    assert "教師データとしての判定" in document
    assert "annotation_decision" in document
    assert "<b>Bucket:</b>" not in document
    assert "<b>Reasons:</b>" not in document
    assert "text/csv;charset=utf-8" in document
    assert ".join('\\r\\n')" in document
    assert "setTimeout(()=>URL.revokeObjectURL" in document


def test_dense_candidate_distinguishes_zero_low_confidence_and_detected(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "dense").mkdir(parents=True)
    models = [
        ModelConfig(name=name, type="dummy", weights=tmp_path / f"{name}.pt", confidence_threshold=.05, analysis_threshold=.5)
        for name in ("a", "b", "c")
    ]
    config = PipelineConfig(
        project_root=tmp_path, config_path=tmp_path / "config.yaml", input_dir=tmp_path,
        output_dir=output, models=models,
        presence_classifier=PresenceClassifierConfig(face_threshold=.7, head_threshold=.7),
        clip_window=ClipWindowConfig(mode="adaptive_track", min_duration_sec=4, max_duration_sec=10, context_padding_sec=1),
    )
    base = {
        "video_id": "video", "filename": "video.mp4", "video_path": "/video.mp4", "scene_id": 0,
        "original_width": 100, "original_height": 100, "brightness": 100, "contrast": 20,
        "blur_score": 10, "scene_change_score": 0, "head_predictions": "[]", "track_id": "",
        "track_events": "{}", "roi_embedding_index": -1, "proposal_start_sec": 0, "proposal_end_sec": 10,
    }
    low = prediction(5, confidence=.2, model="a")
    detected = prediction(7, confidence=.8, model="a")
    rows = [
        {**base, "timestamp_sec": 3.0, "perceptual_hash": "0" * 16, "raw_predictions": "[]", "analysis_predictions": "[]", "roi_source": "", "roi_bbox": "", "p_face_present": 0, "p_head_present": 0},
        {**base, "timestamp_sec": 5.0, "perceptual_hash": "1" * 16, "raw_predictions": json.dumps([low]), "analysis_predictions": "[]", "roi_source": "head", "roi_bbox": "10,10,30,30", "p_face_present": .8, "p_head_present": .9},
        {**base, "timestamp_sec": 7.0, "perceptual_hash": "2" * 16, "raw_predictions": json.dumps([detected]), "analysis_predictions": json.dumps([detected]), "roi_source": "face", "roi_bbox": "10,10,30,30", "p_face_present": .1, "p_head_present": .1},
    ]
    pd.DataFrame(rows).to_parquet(output / "dense/dense_frames.parquet", index=False)
    pd.DataFrame([], columns=["row_type", "track_id"]).to_parquet(output / "dense/tracks.parquet", index=False)
    np.save(output / "dense/roi_embeddings.npy", np.empty((0, 0), dtype=np.float32))
    (output / "dense/dense_metrics.json").write_text(json.dumps({"presence_thresholds": {"face": .7, "head": .7}}))
    candidates = build_dense_candidates(config)
    zero, low_candidate, detected_candidate = candidates
    assert zero.all_zero_detection and not zero.all_low_confidence and zero.eligibility_reason == "full_frame_fallback"
    assert low_candidate.all_low_confidence and not low_candidate.all_zero_detection and low_candidate.eligibility_reason == "low_confidence"
    assert detected_candidate.comparison_detection_count == 1 and detected_candidate.eligibility_reason == "comparison_detection"


def test_dense_stage_writes_only_roi_embeddings(monkeypatch, tmp_path: Path) -> None:
    class FakeAdapter(ModelAdapter):
        def predict_batch(self, frames_bgr, *, video_id, timestamps, return_embeddings=False):
            groups = []
            for timestamp, frame in zip(timestamps, frames_bgr, strict=True):
                groups.append([Prediction(
                    model_name=self.model_name, video_id=video_id, timestamp_sec=timestamp, class_id=2,
                    confidence=.8, bbox_xyxy_original=(10 + timestamp, 10, 30 + timestamp, 30),
                    input_width=64, input_height=64, original_width=frame.shape[1], original_height=frame.shape[0],
                )])
            return groups, None

    class FakeDino(DinoDeimAdapter):
        def __init__(self, config):
            ModelAdapter.__init__(self, config)

        def predict_batch(self, frames_bgr, *, video_id, timestamps, return_embeddings=False):
            return FakeAdapter.predict_batch(self, frames_bgr, video_id=video_id, timestamps=timestamps, return_embeddings=return_embeddings)

        def embed_batch(self, frames_bgr, *, input_size=None):
            return np.asarray([[float(frame.mean()), 1, 2, 3] for frame in frames_bgr], dtype=np.float32)

    comparison = [
        ModelConfig(name="a", type="fake", weights=tmp_path / "a.pt", analysis_threshold=.5),
        ModelConfig(name="dino", type="dinov3_deimv2", weights=tmp_path / "dino.pt", analysis_threshold=.5, embedding_source=True),
        ModelConfig(name="c", type="fake", weights=tmp_path / "c.pt", analysis_threshold=.5),
    ]
    head = ModelConfig(name="head", type="fake", weights=tmp_path / "head.pt", analysis_threshold=.5, class_name="head")
    config = PipelineConfig(
        project_root=tmp_path, config_path=tmp_path / "config.yaml", input_dir=tmp_path,
        output_dir=tmp_path / "output", models=comparison, head_model=head,
        dense_analysis=DenseAnalysisConfig(enabled=True, fps=5, min_track_hits=3),
        presence_classifier=PresenceClassifierConfig(enabled=False, roi_input_size=32),
    )

    def fake_create(model):
        return FakeDino(model) if model.name == "dino" else FakeAdapter(model)

    def fake_frames(video, *, start_sec, end_sec, sample_fps):
        for index in range(30):
            yield index / sample_fps, index, np.full((64, 64, 3), index, dtype=np.uint8)

    monkeypatch.setattr("src.active_learning.stages.create_adapter", fake_create)
    monkeypatch.setattr("src.active_learning.stages.sample_video_window", fake_frames)
    inventory = [VideoInfo(video_id="video", filename="video.mp4", path="/video.mp4", duration=6, width=64, height=64, readable=True)]
    proposals = [{"video_id": "video", "window_id": "window_00001", "scene_id": 0, "start_sec": 0, "end_sec": 6}]
    metrics = run_dense_stage(config, inventory, proposals, force=True)
    assert metrics["frames"] == 30
    assert (config.output_dir / "dense/dense_frames.parquet").is_file()
    assert np.load(config.output_dir / "dense/roi_embeddings.npy").shape == (30, 4)
    assert not (config.output_dir / "scans/dino_embeddings.npy").exists()
    dense = pd.read_parquet(config.output_dir / "dense/dense_frames.parquet")
    assert dense["roi_source"].str.contains("face:a").all()
    assert dense["roi_source"].str.contains("head:head").all()


def _frame_candidate(video_id: str, index: int, **scores: float) -> Candidate:
    return Candidate(
        video_id=video_id,
        filename=f"{video_id}.mp4",
        video_path=f"/{video_id}.mp4",
        timestamp_sec=index * 31.0,
        frame_index=index,
        scene_id=index // 10,
        perceptual_hash=f"{index:016x}",
        pixel_sha256=f"{video_id}-{index}",
        usable_quality=True,
        dino_cluster_by_bucket={
            "model_disagreement": index % 4,
            "temporal_anomaly": index % 5,
            "other_instability": index % 6,
            "uniform": index % 7,
        },
        **scores,
    )


def test_frame_selection_reaches_per_video_40_20_20_20() -> None:
    rows = []
    for index in range(200):
        phase = index % 10
        rows.append(_frame_candidate(
            "video", index,
            disagreement_score=1.0 if phase == 0 else 0.0,
            temporal_anomaly_score=1.0 if phase == 1 else 0.0,
            other_instability_score=1.0 if phase == 2 else 0.0,
            uniform_score=1.0,
        ))
    selected, _, summary = select_frame_candidates(
        rows,
        frames_per_video=100,
        ratios={"uniform": .4, "model_disagreement": .2, "temporal_anomaly": .2, "other_instability": .2},
        min_interval_sec=30,
    )
    counts = {bucket: sum(row.selection_bucket == bucket for row in selected) for bucket in (
        "uniform", "model_disagreement", "temporal_anomaly", "other_instability"
    )}
    assert counts == {"uniform": 40, "model_disagreement": 20, "temporal_anomaly": 20, "other_instability": 20}
    assert summary["shortfalls"]["video"]["total_shortfall"] == 0


def test_priority_profile_reaches_35_25_15_10_10_5_with_relevance_gate() -> None:
    rows: list[Candidate] = []
    index = 0

    def add_many(count: int, **values: object) -> None:
        nonlocal index
        for _ in range(count):
            rows.append(_frame_candidate("video", index, **values))  # type: ignore[arg-type]
            index += 1

    evidence = {"head_score": .8, "model_counts": {"a": 1, "b": 0, "head": 1}}
    add_many(35, disagreement_score=1.0, comparison_detection_count=1, **evidence)
    add_many(25, temporal_anomaly_score=1.0, track_id="track", track_event_types=["middle_missing"], **evidence)
    add_many(15, bbox_instability_score=1.0, track_id="track", track_event_types=["shape_jump"], **evidence)
    add_many(10, confidence_boundary_score=1.0, **evidence)
    add_many(10, head_score=.8, comparison_detection_count=0, model_counts={"a": 0, "b": 0, "head": 1})
    add_many(5, novelty_score=1.0, roi_source="head:head", **evidence)

    ratios = {
        "model_disagreement": .35,
        "temporal_anomaly": .25,
        "bbox_instability": .15,
        "low_confidence": .10,
        "head_evidence": .10,
        "dino_novelty": .05,
    }
    selected, _, summary = select_frame_candidates(
        rows,
        frames_per_video=100,
        ratios=ratios,
        min_interval_sec=30,
        head_model_name="head",
        head_analysis_threshold=.25,
        require_relevance_evidence=True,
    )
    assert {bucket: sum(row.selection_bucket == bucket for row in selected) for bucket in ratios} == {
        "model_disagreement": 35,
        "temporal_anomaly": 25,
        "bbox_instability": 15,
        "low_confidence": 10,
        "head_evidence": 10,
        "dino_novelty": 5,
    }
    assert summary["shortfalls"]["video"]["total_shortfall"] == 0


def test_priority_profile_rejects_unsupported_single_model_and_full_frame_dino() -> None:
    rows = [
        _frame_candidate(
            "video", 0, disagreement_score=1.0, comparison_detection_count=1,
            model_counts={"dino": 1, "yolo": 0, "head": 0},
        ),
        _frame_candidate(
            "video", 1, novelty_score=1.0, roi_source="full_frame", head_score=.8,
            model_counts={"dino": 0, "yolo": 0, "head": 1},
        ),
    ]
    selected, _, summary = select_frame_candidates(
        rows,
        frames_per_video=2,
        ratios={"model_disagreement": .5, "dino_novelty": .5},
        min_interval_sec=0,
        head_model_name="head",
        head_analysis_threshold=.25,
        require_relevance_evidence=True,
    )
    assert selected == []
    assert summary["shortfalls"]["video"]["total_shortfall"] == 2


def test_active_shortfall_is_not_filled_with_weak_or_uniform_rows() -> None:
    rows = [
        _frame_candidate(
            "video", index,
            disagreement_score=1.0 if index < 10 else 0.0,
            uniform_score=1.0,
        )
        for index in range(200)
    ]
    selected, _, summary = select_frame_candidates(
        rows,
        frames_per_video=100,
        ratios={"uniform": .4, "model_disagreement": .2, "temporal_anomaly": .2, "other_instability": .2},
        min_interval_sec=30,
    )
    active = [row for row in selected if row.selection_bucket != "uniform"]
    assert len(active) == 10
    assert all(row.disagreement_score > 0 for row in active)
    assert len(selected) < 100
    assert summary["shortfalls"]["video"]["active_total"] == 10
    assert summary["shortfalls"]["video"]["total_shortfall"] > 0


def test_frame_selection_deduplicates_pixels_and_suppresses_near_timestamps() -> None:
    rows = [
        _frame_candidate("a", 0, disagreement_score=1.0, uniform_score=1.0),
        _frame_candidate("a", 1, disagreement_score=.9, uniform_score=1.0),
        _frame_candidate("b", 0, disagreement_score=.8, uniform_score=1.0),
    ]
    rows[1].timestamp_sec = 29.9
    rows[2].pixel_sha256 = rows[0].pixel_sha256
    selected, rejected, _ = select_frame_candidates(
        rows,
        frames_per_video=2,
        ratios={"model_disagreement": 1.0},
        min_interval_sec=30,
    )
    assert len(selected) == 1
    assert {row.rejection_reason for row in rejected} >= {"exact_pixel_duplicate", "min_interval_sec"}


def test_cluster_round_robin_and_selection_are_deterministic() -> None:
    rows = [
        _frame_candidate("video", 0, disagreement_score=1.0),
        _frame_candidate("video", 1, disagreement_score=.99),
        _frame_candidate("video", 2, disagreement_score=.8),
    ]
    rows[0].dino_cluster_by_bucket["model_disagreement"] = 0
    rows[1].dino_cluster_by_bucket["model_disagreement"] = 0
    rows[2].dino_cluster_by_bucket["model_disagreement"] = 1
    first, _, _ = select_frame_candidates(
        rows, frames_per_video=2, ratios={"model_disagreement": 1.0}, min_interval_sec=0
    )
    second, _, _ = select_frame_candidates(
        rows, frames_per_video=2, ratios={"model_disagreement": 1.0}, min_interval_sec=0
    )
    assert {row.dino_cluster_id for row in first} == {0, 1}
    assert [(row.video_id, row.timestamp_sec, row.selection_bucket) for row in first] == [
        (row.video_id, row.timestamp_sec, row.selection_bucket) for row in second
    ]


def test_roi_union_covers_head_and_all_model_boxes() -> None:
    assert union_bbox(
        [(10, 20, 30, 40), (5, 25, 50, 35), (15, 10, 20, 45)], width=100, height=80
    ) == (5.0, 10.0, 50.0, 45.0)
