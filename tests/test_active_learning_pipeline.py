from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np

from src.active_learning.adapters import restore_bbox_to_original
from src.active_learning.config import MatchingConfig
from src.active_learning.matching import match_bboxes
from src.active_learning.scoring import confidence_percentiles, disagreement_score
from src.active_learning.selection import clip_bounds, montage_ranges, quota_counts, select_candidates
from src.active_learning.pipeline import _matched_split
from src.active_learning.types import Candidate, VideoInfo


def candidate(video: str, timestamp: float, index: int) -> Candidate:
    return Candidate(
        video_id=video,
        filename=f"{video}.mp4",
        video_path=f"/{video}.mp4",
        timestamp_sec=timestamp,
        scene_id=index,
        perceptual_hash=f"{index * 0x9e3779b97f4a7c15 & ((1 << 64) - 1):016x}",
        disagreement_score=((index * 17) % 101) / 100,
        common_difficulty_score=((index * 29) % 103) / 102,
        novelty_score=((index * 43) % 107) / 106,
        temporal_random_score=((index * 59) % 109) / 108,
        model_counts={"yolo11s": 1, "rtmdet": 1, "dino_deimv2": 1},
    )


def test_restore_bbox_to_original_coordinates() -> None:
    assert restore_bbox_to_original(
        (96, 192, 480, 768), input_width=960, input_height=960, original_width=1920, original_height=1080
    ) == (192.0, 216.0, 960.0, 864.0)


def test_bbox_matching_uses_center_and_area_for_small_boxes() -> None:
    matches, left, right = match_bboxes([[10, 10, 20, 20]], [[12, 12, 22, 22]], iou_threshold=0.9, center_distance_threshold=0.3)
    assert len(matches) == 1
    assert left == [] and right == []


def test_disagreement_score_increases_for_unmatched_detection() -> None:
    weights = {"a": 1.0, "b": 1.0}
    same = {"a": [{"bbox_xyxy_original": [0, 0, 10, 10], "confidence_percentile": 0.5}], "b": [{"bbox_xyxy_original": [0, 0, 10, 10], "confidence_percentile": 0.5}]}
    different = {"a": same["a"], "b": []}
    assert disagreement_score(different, weights, MatchingConfig()) > disagreement_score(same, weights, MatchingConfig())


def test_confidence_percentiles_are_stable_and_average_ties() -> None:
    assert confidence_percentiles([0.2, 0.1, 0.2]) == [0.75, 0.0, 0.75]


def test_clip_bounds_clip_both_video_edges() -> None:
    assert clip_bounds(1.0, 10.0, 100.0) == (0.0, 10.0)
    assert clip_bounds(99.0, 10.0, 100.0) == (90.0, 100.0)
    assert clip_bounds(2.0, 10.0, 4.0) == (0.0, 4.0)


def test_minimum_interval_and_max_clips_per_video() -> None:
    pool = [candidate("a", index * 10.0, index) for index in range(30)] + [candidate("b", index * 10.0, index + 100) for index in range(30)]
    selected, _, _ = select_candidates(
        pool, target_count=8, ratios={"model_disagreement": .4, "common_difficulty": .2, "novelty": .2, "temporal_random": .2},
        max_per_video=4, min_per_video=2, min_interval_sec=30, max_per_scene=2, max_per_track=2, duplicate_phash_distance=-1,
    )
    counts = Counter(row.video_id for row in selected)
    assert max(counts.values()) <= 4
    for video in counts:
        timestamps = sorted(row.timestamp_sec for row in selected if row.video_id == video)
        assert all(right - left >= 30 for left, right in zip(timestamps, timestamps[1:]))


def test_bucket_quota() -> None:
    assert quota_counts(40, {"model_disagreement": .4, "common_difficulty": .2, "novelty": .2, "temporal_random": .2}) == {
        "model_disagreement": 16, "common_difficulty": 8, "novelty": 8, "temporal_random": 8
    }


def test_manifest_montage_timestamps_are_contiguous() -> None:
    assert montage_ranges([10, 10, 4]) == [(0.0, 10.0), (10.0, 20.0), (20.0, 24.0)]


def test_seed_reproducibility() -> None:
    from src.active_learning.scoring import deterministic_kmeans

    embeddings = np.random.default_rng(3).normal(size=(30, 8)).astype(np.float32)
    assert np.array_equal(deterministic_kmeans(embeddings, 5, 17), deterministic_kmeans(embeddings, 5, 17))


def test_validation_video_is_excluded_by_stable_source_hash() -> None:
    filename = "example original video.mp4"
    digest = hashlib.sha1(filename.removesuffix(".mp4").encode("utf-8")).hexdigest()[:8]
    video = VideoInfo(video_id="example", filename=filename, path=f"/{filename}")
    assert _matched_split(video, {"val": {f"2026_01_01_EXA_{digest}"}, "test": set()}) == "val"
