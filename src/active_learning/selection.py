from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from .types import Candidate


BUCKET_SCORE_FIELDS = {
    "uniform": "uniform_score",
    "other_instability": "other_instability_score",
    "model_disagreement": "disagreement_score",
    "common_difficulty": "common_difficulty_score",
    "novelty": "novelty_score",
    "temporal_random": "temporal_random_score",
    "temporal_anomaly": "temporal_anomaly_score",
    "no_detection_evidence": "no_detection_evidence_score",
    "roi_novelty": "novelty_score",
    "low_presence_audit": "low_presence_audit_score",
}
FILL_PRIORITY = [
    "model_disagreement", "temporal_anomaly", "no_detection_evidence", "roi_novelty",
    "low_presence_audit", "temporal_random", "novelty", "common_difficulty",
]


def quota_counts(target: int, ratios: dict[str, float]) -> dict[str, int]:
    normalized = {name: float(value) / sum(ratios.values()) for name, value in ratios.items()}
    raw = {name: target * ratio for name, ratio in normalized.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = target - sum(counts.values())
    for name in sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)[:remaining]:
        counts[name] += 1
    return counts


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count() if left and right else 64


def select_candidates(
    candidates: Iterable[Candidate],
    *,
    target_count: int,
    ratios: dict[str, float],
    max_per_video: int,
    min_per_video: int,
    min_interval_sec: float,
    max_per_scene: int,
    max_per_track: int,
    duplicate_phash_distance: int,
) -> tuple[list[Candidate], list[Candidate], dict[str, str]]:
    pool = list(candidates)
    quotas = quota_counts(target_count, ratios)
    selected: list[Candidate] = []
    rejected: list[Candidate] = []
    per_video: Counter[str] = Counter()
    per_scene: Counter[tuple[str, int]] = Counter()
    per_track: Counter[str] = Counter()
    reasons: dict[tuple[str, float], str] = {}
    duplicate_groups: dict[tuple[str, float], str] = {}
    bucket_order = [name for name in FILL_PRIORITY if name in quotas]

    def eligible_for_bucket(row: Candidate, bucket: str) -> bool:
        if bucket == "model_disagreement":
            return row.comparison_detection_count > 0 or any(row.model_counts.values())
        if bucket == "temporal_anomaly":
            return row.temporal_anomaly_score > 0 and bool(row.track_id or row.track_group_id)
        if bucket == "no_detection_evidence":
            return row.comparison_detection_count == 0 and row.eligibility_reason in {"face_presence", "head_with_temporal_face"}
        if bucket == "roi_novelty":
            return bool(row.roi_source) and row.eligibility_reason not in {"audit_only", "no_roi_evidence"}
        if bucket == "low_presence_audit":
            return bool(row.roi_source) and row.usable_quality
        return True

    def valid(row: Candidate) -> tuple[bool, str, str]:
        if per_video[row.video_id] >= max_per_video:
            return False, "max_clips_per_video", ""
        if per_scene[(row.video_id, row.scene_id)] >= max_per_scene:
            return False, "max_clips_per_scene", ""
        if row.track_group_id and per_track[row.track_group_id] >= max_per_track:
            return False, "max_clips_per_track", ""
        for previous in selected:
            if previous.video_id == row.video_id and abs(previous.timestamp_sec - row.timestamp_sec) < min_interval_sec:
                return False, "min_interval_sec", ""
            if _hamming(previous.perceptual_hash, row.perceptual_hash) <= duplicate_phash_distance:
                return False, "perceptual_duplicate", f"dup:{previous.video_id}:{previous.timestamp_sec:.6f}"
        return True, "", ""

    def add(row: Candidate, bucket: str) -> bool:
        if not eligible_for_bucket(row, bucket):
            return False
        okay, reason, duplicate_group = valid(row)
        if not okay:
            key = (row.video_id, row.timestamp_sec)
            if key not in reasons or reason == "perceptual_duplicate":
                reasons[key] = reason
            if duplicate_group:
                duplicate_groups[key] = duplicate_group
            return False
        score = float(getattr(row, BUCKET_SCORE_FIELDS[bucket]))
        selected.append(replace(
            row,
            selection_bucket=bucket,
            selection_score=score,
            selection_reason=f"ranked by {BUCKET_SCORE_FIELDS[bucket]}",
            status="selected",
        ))
        per_video[row.video_id] += 1
        per_scene[(row.video_id, row.scene_id)] += 1
        if row.track_group_id:
            per_track[row.track_group_id] += 1
        return True

    videos = sorted({row.video_id for row in pool})
    # First guarantee minimum representation where feasible, choosing the best
    # still-needed bucket in round-robin video order.
    for _ in range(min_per_video):
        for video_id in videos:
            for bucket in bucket_order:
                used = sum(row.selection_bucket == bucket for row in selected)
                if used >= quotas.get(bucket, 0):
                    continue
                ranked = sorted(
                    (row for row in pool if row.video_id == video_id and row not in selected),
                    key=lambda row: (float(getattr(row, BUCKET_SCORE_FIELDS[bucket])), -row.timestamp_sec),
                    reverse=True,
                )
                if next((True for row in ranked if add(row, bucket)), False):
                    break

    def ranked_for_bucket(bucket: str) -> list[Candidate]:
        if bucket != "temporal_random":
            return sorted(pool, key=lambda row: (float(getattr(row, BUCKET_SCORE_FIELDS[bucket])), row.video_id, -row.timestamp_sec), reverse=True)
        # Use seeded randomness within time strata so this bucket remains
        # independent of detector outputs while covering each video's timeline.
        temporal_pool = [row for row in pool if row.usable_quality]
        bins_per_video = max(1, (quotas.get(bucket, 1) + max(1, len(videos)) - 1) // max(1, len(videos)))
        primary: list[Candidate] = []
        remainder: list[Candidate] = []
        for video_id in videos:
            rows = sorted((row for row in temporal_pool if row.video_id == video_id), key=lambda row: row.timestamp_sec)
            if not rows:
                continue
            maximum = max(row.timestamp_sec for row in rows) + 1e-9
            groups: dict[int, list[Candidate]] = defaultdict(list)
            for row in rows:
                groups[min(bins_per_video - 1, int(row.timestamp_sec / maximum * bins_per_video))].append(row)
            for bin_index in range(bins_per_video):
                ranked = sorted(groups.get(bin_index, []), key=lambda row: row.temporal_random_score, reverse=True)
                if ranked:
                    primary.append(ranked[0])
                    remainder.extend(ranked[1:])
        primary.sort(key=lambda row: (row.timestamp_sec, row.video_id))
        remainder.sort(key=lambda row: row.temporal_random_score, reverse=True)
        return primary + remainder

    for bucket in bucket_order:
        needed = quotas.get(bucket, 0) - sum(row.selection_bucket == bucket for row in selected)
        ranked = ranked_for_bucket(bucket)
        for row in ranked:
            if needed <= 0:
                break
            if any(s.video_id == row.video_id and s.timestamp_sec == row.timestamp_sec for s in selected):
                continue
            if add(row, bucket):
                needed -= 1

    # Refill unavailable quotas in priority order, but cap any bucket at 60%.
    bucket_cap = max(quotas.values()) + max(1, target_count // 5)
    # Keep the audit quota fixed. Shortages are refilled from actionable
    # detector/temporal/evidence/ROI signals only.
    for bucket in [name for name in bucket_order if name != "low_presence_audit"]:
        if len(selected) >= target_count:
            break
        if sum(row.selection_bucket == bucket for row in selected) >= bucket_cap:
            continue
        ranked = ranked_for_bucket(bucket)
        for row in ranked:
            if len(selected) >= target_count or sum(s.selection_bucket == bucket for s in selected) >= bucket_cap:
                break
            if any(s.video_id == row.video_id and s.timestamp_sec == row.timestamp_sec for s in selected):
                continue
            add(row, bucket)

    selected.sort(key=lambda row: (row.video_id, row.timestamp_sec))
    selected = [replace(row, clip_id=f"clip_{index:04d}") for index, row in enumerate(selected, 1)]
    selected_keys = {(row.video_id, row.timestamp_sec) for row in selected}
    for row in pool:
        if (row.video_id, row.timestamp_sec) not in selected_keys:
            key = (row.video_id, row.timestamp_sec)
            rejected.append(replace(
                row,
                status="rejected",
                rejection_reason=reasons.get(key, "not_selected"),
                duplicate_group_id=duplicate_groups.get(key, row.duplicate_group_id),
            ))
    shortages = {
        video_id: f"selected {per_video[video_id]} < requested minimum {min_per_video}: insufficient non-overlapping/non-duplicate candidates"
        for video_id in videos if per_video[video_id] < min_per_video
    }
    for bucket, quota in quotas.items():
        actual = sum(row.selection_bucket == bucket for row in selected)
        if actual < quota:
            shortages[f"bucket:{bucket}"] = f"selected {actual} < requested {quota}: insufficient eligible candidates"
    return selected, rejected, shortages


FRAME_BUCKET_SCORE_FIELDS = {
    "model_disagreement": "disagreement_score",
    "temporal_anomaly": "temporal_anomaly_score",
    "bbox_instability": "bbox_instability_score",
    "low_confidence": "confidence_boundary_score",
    "head_evidence": "head_score",
    "dino_novelty": "novelty_score",
    "other_instability": "other_instability_score",
    "uniform": "uniform_score",
}
FRAME_BUCKET_PRIORITY = [
    "model_disagreement",
    "temporal_anomaly",
    "bbox_instability",
    "low_confidence",
    "head_evidence",
    "dino_novelty",
    "other_instability",
]
PRIORITY_PROFILE_BUCKETS = set(FRAME_BUCKET_PRIORITY[:6])


def _cluster_round_robin(rows: Iterable[Candidate], score_field: str, bucket: str) -> list[Candidate]:
    """Interleave DINO clusters while keeping the best rows first in each cluster."""

    groups: dict[int, list[Candidate]] = defaultdict(list)
    for row in rows:
        cluster = row.dino_cluster_by_bucket.get(bucket, row.dino_cluster_id)
        cluster = cluster if cluster >= 0 else -(row.scene_id + 1)
        groups[cluster].append(row)
    for cluster_rows in groups.values():
        cluster_rows.sort(
            key=lambda row: (float(getattr(row, score_field)), row.novelty_score, -row.timestamp_sec),
            reverse=True,
        )
    ordered: list[Candidate] = []
    cluster_order = sorted(
        groups,
        key=lambda cluster: (
            float(getattr(groups[cluster][0], score_field)),
            groups[cluster][0].novelty_score,
            -cluster,
        ),
        reverse=True,
    )
    depth = 0
    while True:
        added = False
        for cluster in cluster_order:
            if depth < len(groups[cluster]):
                ordered.append(groups[cluster][depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def select_frame_candidates(
    candidates: Iterable[Candidate],
    *,
    frames_per_video: int,
    ratios: dict[str, float],
    min_interval_sec: float,
    head_model_name: str = "",
    head_analysis_threshold: float = 0.0,
    require_relevance_evidence: bool = False,
) -> tuple[list[Candidate], list[Candidate], dict[str, object]]:
    """Select a bounded, priority-ordered frame dataset independently per video."""

    all_pool = [row for row in candidates if row.timestamp_sec >= 0 and row.usable_quality]
    quotas = quota_counts(frames_per_video, ratios)
    active_buckets = [bucket for bucket in FRAME_BUCKET_PRIORITY if bucket in quotas]
    priority_profile = bool(PRIORITY_PROFILE_BUCKETS & set(ratios)) and any(
        bucket in ratios for bucket in {"bbox_instability", "low_confidence", "head_evidence", "dino_novelty"}
    )
    selected: list[Candidate] = []
    selected_hashes: set[str] = set()
    rejection_reasons: dict[int, str] = {}
    shortages: dict[str, dict[str, int]] = {}

    # Resolve exact decoded-pixel duplicates across the complete input dataset
    # before per-video quota selection. Keep the copy with the strongest active
    # signal, then use stable video/timestamp ties for reproducibility.
    retained_hashes: set[str] = set()
    retained_ids: set[int] = set()
    active_score_fields = [FRAME_BUCKET_SCORE_FIELDS[bucket] for bucket in active_buckets]
    ranked_for_dedup = sorted(
        all_pool,
        key=lambda row: (
            max((float(getattr(row, field)) for field in active_score_fields), default=0.0),
            row.novelty_score,
            row.video_id,
            -row.timestamp_sec,
        ),
        reverse=True,
    )
    for row in ranked_for_dedup:
        if row.pixel_sha256 and row.pixel_sha256 in retained_hashes:
            rejection_reasons[id(row)] = "exact_pixel_duplicate"
            continue
        retained_ids.add(id(row))
        if row.pixel_sha256:
            retained_hashes.add(row.pixel_sha256)
    pool = [row for row in all_pool if id(row) in retained_ids]

    def active_face_model_count(row: Candidate) -> int:
        return sum(
            int(count) > 0
            for name, count in row.model_counts.items()
            if not head_model_name or name != head_model_name
        )

    def relevant(row: Candidate) -> bool:
        return (
            row.head_score >= head_analysis_threshold
            or active_face_model_count(row) >= 2
            or row.tracked_missing_detection
        )

    def eligible(row: Candidate, bucket: str) -> bool:
        if float(getattr(row, FRAME_BUCKET_SCORE_FIELDS[bucket])) <= 0.0:
            return False
        if not require_relevance_evidence:
            return True
        if bucket == "model_disagreement":
            return row.comparison_detection_count > 0 and relevant(row)
        if bucket in {"temporal_anomaly", "bbox_instability"}:
            return bool(row.track_id or row.track_event_types) and relevant(row)
        if bucket == "low_confidence":
            # Boundary candidates deliberately include scenery/hard negatives.
            # Human review decides whether the sample is positive or negative.
            return True
        if bucket == "head_evidence":
            return row.comparison_detection_count == 0 and row.head_score >= head_analysis_threshold
        if bucket == "dino_novelty":
            return row.roi_source != "full_frame" and relevant(row)
        return True

    def can_add(row: Candidate, video_selected: list[Candidate]) -> bool:
        if row.pixel_sha256 and row.pixel_sha256 in selected_hashes:
            rejection_reasons[id(row)] = "exact_pixel_duplicate"
            return False
        if any(abs(row.timestamp_sec - previous.timestamp_sec) < min_interval_sec for previous in video_selected):
            rejection_reasons.setdefault(id(row), "min_interval_sec")
            return False
        return True

    def add(row: Candidate, bucket: str, video_selected: list[Candidate], reason: str) -> bool:
        if not can_add(row, video_selected):
            return False
        score_field = FRAME_BUCKET_SCORE_FIELDS[bucket]
        chosen = replace(
            row,
            dino_cluster_id=row.dino_cluster_by_bucket.get(bucket, row.dino_cluster_id),
            selection_bucket=bucket,
            selection_score=float(getattr(row, score_field)),
            selection_reason=reason,
            status="selected",
        )
        video_selected.append(chosen)
        selected.append(chosen)
        if row.pixel_sha256:
            selected_hashes.add(row.pixel_sha256)
        return True

    videos = sorted({row.video_id for row in pool})
    for video_id in videos:
        rows = [row for row in pool if row.video_id == video_id]
        video_selected: list[Candidate] = []
        eligible_by_bucket = {bucket: [row for row in rows if eligible(row, bucket)] for bucket in active_buckets}
        # New six-signal profiles are assigned in the requested semantic
        # priority. Legacy profiles retain scarcity-first allocation.
        bucket_order = active_buckets if priority_profile else sorted(
            active_buckets, key=lambda bucket: (len(eligible_by_bucket[bucket]), bucket)
        )
        for bucket in bucket_order:
            needed = quotas.get(bucket, 0)
            if priority_profile and bucket != "dino_novelty":
                ranked = sorted(
                    eligible_by_bucket[bucket],
                    key=lambda row: (
                        float(getattr(row, FRAME_BUCKET_SCORE_FIELDS[bucket])),
                        row.video_id,
                        -row.timestamp_sec,
                    ),
                    reverse=True,
                )
            else:
                ranked = _cluster_round_robin(
                    eligible_by_bucket[bucket], FRAME_BUCKET_SCORE_FIELDS[bucket], bucket
                )
            for row in ranked:
                if sum(item.selection_bucket == bucket for item in video_selected) >= needed:
                    break
                if any(
                    item.video_id == row.video_id and abs(item.timestamp_sec - row.timestamp_sec) < 1e-6
                    for item in video_selected
                ):
                    continue
                add(row, bucket, video_selected, f"ranked by {FRAME_BUCKET_SCORE_FIELDS[bucket]}")

        active_target = sum(quotas.get(bucket, 0) for bucket in active_buckets)
        active_selected = sum(item.selection_bucket in active_buckets for item in video_selected)
        if active_selected < active_target and not priority_profile:
            remainder = [
                row for row in rows
                if any(eligible(row, bucket) for bucket in active_buckets)
                and not any(abs(item.timestamp_sec - row.timestamp_sec) < 1e-6 for item in video_selected)
            ]
            remainder.sort(
                key=lambda row: (
                    max(float(getattr(row, FRAME_BUCKET_SCORE_FIELDS[bucket])) for bucket in active_buckets),
                    row.novelty_score,
                    -row.timestamp_sec,
                ),
                reverse=True,
            )
            for row in remainder:
                if sum(item.selection_bucket in active_buckets for item in video_selected) >= active_target:
                    break
                scores = {
                    bucket: float(getattr(row, FRAME_BUCKET_SCORE_FIELDS[bucket]))
                    for bucket in active_buckets if eligible(row, bucket)
                }
                bucket = max(scores, key=lambda name: (scores[name], name))
                add(row, bucket, video_selected, "active quota refill")

        uniform_target = quotas.get("uniform", 0)
        if uniform_target > 0 and rows:
            maximum = max(
                max((row.video_duration_sec for row in rows), default=0.0),
                max(row.timestamp_sec for row in rows) + 1e-9,
            )
            width = maximum / uniform_target
            for bin_index in range(uniform_target):
                center = (bin_index + 0.5) * width
                bin_rows = [
                    row for row in rows
                    if min(uniform_target - 1, int(row.timestamp_sec / maximum * uniform_target)) == bin_index
                    and not any(abs(item.timestamp_sec - row.timestamp_sec) < 1e-6 for item in video_selected)
                ]
                ranked = sorted(
                    bin_rows,
                    key=lambda row: (abs(row.timestamp_sec - center), -row.novelty_score, row.timestamp_sec),
                )
                for row in ranked:
                    if add(row, "uniform", video_selected, f"uniform temporal bin {bin_index}"):
                        break

        actual = Counter(item.selection_bucket for item in video_selected)
        bucket_shortfalls = {
            bucket: max(0, quotas.get(bucket, 0) - int(actual.get(bucket, 0)))
            for bucket in quotas
        }
        shortages[video_id] = {
            **bucket_shortfalls,
            "active_total": sum(actual.get(bucket, 0) for bucket in active_buckets),
            "active_target": active_target,
            "total": len(video_selected),
            "total_shortfall": max(0, frames_per_video - len(video_selected)),
        }

    selected.sort(key=lambda row: (row.video_id, row.timestamp_sec))
    selected = [replace(row, clip_id=f"frame_{index:06d}") for index, row in enumerate(selected, 1)]
    selected_keys = {(row.video_id, round(row.timestamp_sec, 6)) for row in selected}
    rejected = [
        replace(
            row,
            status="rejected",
            rejection_reason=rejection_reasons.get(id(row), "not_selected"),
        )
        for row in all_pool
        if (row.video_id, round(row.timestamp_sec, 6)) not in selected_keys
    ]
    summary: dict[str, object] = {
        "target_per_video": frames_per_video,
        "quota_per_video": quotas,
        "selected": len(selected),
        "by_video": dict(Counter(row.video_id for row in selected)),
        "by_bucket": dict(Counter(row.selection_bucket for row in selected)),
        "shortfalls": shortages,
    }
    return selected, rejected, summary


def clip_bounds(center: float, duration: float, source_duration: float) -> tuple[float, float]:
    length = min(max(0.0, duration), max(0.0, source_duration))
    start = max(0.0, min(center - length / 2.0, source_duration - length))
    return start, min(source_duration, start + length)


def adaptive_clip_bounds(
    *,
    center: float,
    active_start: float,
    active_end: float,
    scene_start: float,
    scene_end: float,
    min_duration: float,
    max_duration: float,
    padding: float,
) -> tuple[float, float]:
    """Pad an active interval while respecting scene bounds and duration limits."""

    lower = min(scene_start, scene_end)
    upper = max(scene_start, scene_end)
    start = max(lower, min(active_start, center) - padding)
    end = min(upper, max(active_end, center) + padding)
    if end - start < min_duration:
        needed = min_duration - (end - start)
        start -= needed / 2.0
        end += needed / 2.0
        if start < lower:
            end = min(upper, end + lower - start)
            start = lower
        if end > upper:
            start = max(lower, start - (end - upper))
            end = upper
    if end - start > max_duration:
        start = max(lower, center - max_duration / 2.0)
        end = min(upper, start + max_duration)
        start = max(lower, end - max_duration)
    return float(start), float(end)


def montage_ranges(durations: Iterable[float]) -> list[tuple[float, float]]:
    cursor = 0.0
    result = []
    for duration in durations:
        length = max(0.0, float(duration))
        result.append((cursor, cursor + length))
        cursor += length
    return result
