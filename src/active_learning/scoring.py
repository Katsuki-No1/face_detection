from __future__ import annotations

import hashlib
from collections import defaultdict
from math import log
from typing import Any, Iterable, Sequence

import numpy as np

from .config import MatchingConfig
from .matching import match_bboxes
from .types import Candidate, FrameFeature, Prediction


def confidence_percentiles(values: Sequence[float]) -> list[float]:
    """Return stable empirical CDF ranks in [0, 1], averaging tied ranks."""
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    denominator = max(1, len(array) - 1)
    return (ranks / denominator).tolist()


def attach_confidence_percentiles(predictions: list[dict]) -> None:
    by_model: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(predictions):
        by_model[str(row["model_name"])].append(index)
    for indices in by_model.values():
        ranks = confidence_percentiles([float(predictions[index]["confidence"]) for index in indices])
        for index, rank in zip(indices, ranks, strict=True):
            predictions[index]["confidence_percentile"] = float(rank)


def confidence_boundary_components(
    predictions: Sequence[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    min_ratio: float = 0.70,
    max_ratio: float = 1.30,
) -> dict[str, Any]:
    """Return the prediction closest to its model-specific decision threshold.

    The score is triangular around confidence / threshold == 1. Values outside
    the configured band receive zero, so raw floor detections near 0.05 are not
    promoted merely because they exist.
    """

    best: dict[str, Any] = {
        "score": 0.0,
        "model": "",
        "confidence": 0.0,
        "threshold": 0.0,
        "ratio": 0.0,
        "side": "",
    }
    for row in predictions:
        model_name = str(row.get("model_name", ""))
        threshold = float(thresholds.get(model_name, 0.0))
        if threshold <= 0.0:
            continue
        confidence = float(row.get("confidence", 0.0))
        ratio = confidence / threshold
        if ratio < min_ratio or ratio > max_ratio:
            continue
        if ratio <= 1.0:
            score = (ratio - min_ratio) / max(1e-8, 1.0 - min_ratio)
            side = "below" if ratio < 1.0 else "at"
        else:
            score = (max_ratio - ratio) / max(1e-8, max_ratio - 1.0)
            side = "above"
        if score > float(best["score"]):
            best = {
                "score": float(np.clip(score, 0.0, 1.0)),
                "model": model_name,
                "confidence": confidence,
                "threshold": threshold,
                "ratio": ratio,
                "side": side,
            }
    return best


def disagreement_components(
    predictions_by_model: dict[str, Sequence[dict]],
    model_weights: dict[str, float],
    matching: MatchingConfig,
) -> dict[str, float]:
    names = sorted(predictions_by_model)
    if len(names) < 2:
        return {name: 0.0 for name in ("score", "count_difference", "unmatched", "iou_variance", "bbox_variance", "confidence_variance")}
    pair_values: list[float] = []
    pair_weights: list[float] = []
    pair_components: list[dict[str, float]] = []
    for left_index, left_name in enumerate(names):
        left = predictions_by_model[left_name]
        left_boxes = [row["bbox_xyxy_original"] for row in left]
        for right_name in names[left_index + 1 :]:
            right = predictions_by_model[right_name]
            right_boxes = [row["bbox_xyxy_original"] for row in right]
            matches, unmatched_left, unmatched_right = match_bboxes(
                left_boxes,
                right_boxes,
                iou_threshold=matching.iou_threshold,
                center_distance_threshold=matching.center_distance_threshold,
                min_area_ratio=matching.min_area_ratio,
                max_area_ratio=matching.max_area_ratio,
            )
            normalizer = max(1, len(left), len(right))
            count_difference = abs(len(left) - len(right)) / normalizer
            unmatched = (len(unmatched_left) + len(unmatched_right)) / max(1, len(left) + len(right))
            iou_variance = float(np.mean([
                0.5 * (1.0 - match.iou) for match in matches
            ])) + (0.5 * float(np.std([match.iou for match in matches])) if len(matches) > 1 else 0.0) if matches else (1.0 if left or right else 0.0)
            bbox_components: list[float] = []
            for match in matches:
                left_box = left_boxes[match.left_index]
                right_box = right_boxes[match.right_index]
                left_width = max(1e-6, float(left_box[2]) - float(left_box[0]))
                left_height = max(1e-6, float(left_box[3]) - float(left_box[1]))
                right_width = max(1e-6, float(right_box[2]) - float(right_box[0]))
                right_height = max(1e-6, float(right_box[3]) - float(right_box[1]))
                bbox_components.append(float(np.mean([
                    min(1.0, match.center_distance),
                    min(1.0, abs(log(right_width / left_width))),
                    min(1.0, abs(log(right_height / left_height))),
                    min(1.0, abs(log(max(match.area_ratio, 1e-6)))),
                ])))
            bbox_variance = float(np.mean(bbox_components)) if bbox_components else (1.0 if left or right else 0.0)
            confidence_variance = float(np.mean([
                abs(float(left[m.left_index].get("confidence_percentile", 0.5)) - float(right[m.right_index].get("confidence_percentile", 0.5)))
                for m in matches
            ])) if matches else (1.0 if left or right else 0.0)
            pair_values.append(float(np.clip(
                0.30 * count_difference + 0.20 * unmatched + 0.15 * iou_variance
                + 0.20 * bbox_variance + 0.15 * confidence_variance,
                0.0,
                1.0,
            )))
            pair_components.append({
                "count_difference": float(count_difference), "unmatched": float(unmatched),
                "iou_variance": float(iou_variance), "bbox_variance": float(bbox_variance),
                "confidence_variance": float(confidence_variance),
            })
            pair_weights.append(max(0.0, model_weights.get(left_name, 1.0)) * max(0.0, model_weights.get(right_name, 1.0)))
    weights = pair_weights if sum(pair_weights) > 0 else None
    result = {
        name: float(np.average([row[name] for row in pair_components], weights=weights))
        for name in pair_components[0]
    }
    result["score"] = float(np.average(pair_values, weights=weights))
    return result


def disagreement_score(
    predictions_by_model: dict[str, Sequence[dict]],
    model_weights: dict[str, float],
    matching: MatchingConfig,
) -> float:
    return disagreement_components(predictions_by_model, model_weights, matching)["score"]


def _centers(rows: Sequence[dict]) -> list[tuple[float, float, float]]:
    result = []
    for row in rows:
        x1, y1, x2, y2 = (float(v) for v in row["bbox_xyxy_original"])
        width = max(1.0, float(row.get("original_width", 1)))
        height = max(1.0, float(row.get("original_height", 1)))
        result.append(((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, ((x2 - x1) * (y2 - y1)) / (width * height)))
    return result


def build_candidates(
    features: Sequence[FrameFeature],
    predictions: list[dict],
    *,
    model_weights: dict[str, float],
    matching: MatchingConfig,
    temporal_window_sec: float,
    random_seed: int,
    dark_brightness_threshold: float = 8.0,
    broken_contrast_threshold: float = 1.0,
    analysis_thresholds: dict[str, float] | None = None,
    confidence_boundary_min_ratio: float = 0.70,
    confidence_boundary_max_ratio: float = 1.30,
) -> list[Candidate]:
    attach_confidence_percentiles(predictions)
    keyed_raw_predictions: dict[tuple[str, float], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    keyed_predictions: dict[tuple[str, float], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        key = (str(row["video_id"]), round(float(row["timestamp_sec"]), 6))
        model_name = str(row["model_name"])
        keyed_raw_predictions[key][model_name].append(row)
        if analysis_thresholds is None or float(row["confidence"]) >= analysis_thresholds.get(model_name, 0.0):
            keyed_predictions[key][model_name].append(row)
    by_video: dict[str, list[FrameFeature]] = defaultdict(list)
    for feature in features:
        by_video[feature.video_id].append(feature)
    result: list[Candidate] = []
    for video_id, rows in sorted(by_video.items()):
        rows.sort(key=lambda row: row.timestamp_sec)
        frame_scores: list[tuple[float, float]] = []
        for feature in rows:
            groups = keyed_predictions[(video_id, round(feature.timestamp_sec, 6))]
            for model_name in model_weights:
                groups.setdefault(model_name, [])
            relative_conf = [float(row.get("confidence_percentile", 0.0)) for model_rows in groups.values() for row in model_rows]
            mean_relative = float(np.mean(relative_conf)) if relative_conf else 0.0
            frame_scores.append((disagreement_score(groups, model_weights, matching), mean_relative))
        for index, feature in enumerate(rows):
            current_groups = keyed_predictions[(video_id, round(feature.timestamp_sec, 6))]
            current_raw_groups = keyed_raw_predictions[(video_id, round(feature.timestamp_sec, 6))]
            for model_name in model_weights:
                current_groups.setdefault(model_name, [])
                current_raw_groups.setdefault(model_name, [])
            lo = index
            hi = index
            while lo > 0 and feature.timestamp_sec - rows[lo - 1].timestamp_sec <= temporal_window_sec:
                lo -= 1
            while hi + 1 < len(rows) and rows[hi + 1].timestamp_sec - feature.timestamp_sec <= temporal_window_sec:
                hi += 1
            neighbor_indices = [j for j in range(lo, hi + 1) if j != index and rows[j].scene_id == feature.scene_id]
            current_counts = {name: len(model_rows) for name, model_rows in current_groups.items()}
            raw_detections = [row for model_rows in current_raw_groups.values() for row in model_rows]
            boundary = confidence_boundary_components(
                raw_detections,
                analysis_thresholds or {},
                min_ratio=confidence_boundary_min_ratio,
                max_ratio=confidence_boundary_max_ratio,
            )
            neighbor_counts = [
                sum(len(model_rows) for model_rows in keyed_predictions[(video_id, round(rows[j].timestamp_sec, 6))].values())
                for j in neighbor_indices
            ]
            current_count = sum(current_counts.values())
            count_drop = max(0.0, (float(np.median(neighbor_counts)) - current_count) / max(1.0, float(np.median(neighbor_counts)))) if neighbor_counts else 0.0
            # An absent score is not a low score. This was the source of the
            # old all-zero frames being promoted as common difficulty.
            low_confidence = 1.0 - frame_scores[index][1] if current_count > 0 else 0.0
            temporal_disappearance = 1.0 if current_count == 0 and any(value > 0 for value in neighbor_counts) else 0.0
            jitter_values: list[float] = []
            for model_name, current in current_groups.items():
                current_centers = _centers(current)
                if not current_centers:
                    continue
                center = np.mean(np.asarray(current_centers), axis=0)
                for j in neighbor_indices:
                    neighbor_group = keyed_predictions[(video_id, round(rows[j].timestamp_sec, 6))].get(model_name, [])
                    if neighbor_group:
                        jitter_values.append(float(np.linalg.norm(center - np.mean(np.asarray(_centers(neighbor_group)), axis=0))))
            jitter = min(1.0, float(np.mean(jitter_values)) * 5.0) if jitter_values else 0.0
            common = float(np.clip(0.35 * low_confidence + 0.25 * count_drop + 0.25 * temporal_disappearance + 0.15 * jitter, 0.0, 1.0))
            stable = int.from_bytes(hashlib.sha256(f"{random_seed}:{video_id}:{feature.timestamp_sec:.6f}".encode()).digest()[:8], "big")
            random_score = stable / float(2**64 - 1)
            all_detections = [row for model_rows in current_groups.values() for row in model_rows]
            if all_detections:
                representative = max(all_detections, key=lambda row: float(row.get("confidence_percentile", 0.0)))
                x1, y1, x2, y2 = (float(value) for value in representative["bbox_xyxy_original"])
                width = max(1.0, float(representative.get("original_width", feature.original_width)))
                height = max(1.0, float(representative.get("original_height", feature.original_height)))
                center_x = int(np.clip(((x1 + x2) / 2.0 / width) * 8, 0, 7))
                center_y = int(np.clip(((y1 + y2) / 2.0 / height) * 8, 0, 7))
                area = max(1e-8, ((x2 - x1) * (y2 - y1)) / (width * height))
                area_bin = int(np.clip(round(-np.log10(area)), 0, 7))
                track_group_id = f"{video_id}:s{feature.scene_id}:x{center_x}y{center_y}a{area_bin}"
            else:
                track_group_id = f"{video_id}:s{feature.scene_id}:no_detection"
            result.append(Candidate(
                video_id=video_id,
                filename=feature.filename,
                video_path=feature.video_path,
                timestamp_sec=feature.timestamp_sec,
                scene_id=feature.scene_id,
                perceptual_hash=feature.perceptual_hash,
                disagreement_score=frame_scores[index][0],
                common_difficulty_score=common,
                temporal_anomaly_score=float(np.clip(0.35 * count_drop + 0.35 * temporal_disappearance + 0.30 * jitter, 0.0, 1.0)),
                bbox_instability_score=jitter,
                low_confidence_score=float(boundary["score"]),
                confidence_boundary_score=float(boundary["score"]),
                confidence_boundary_model=str(boundary["model"]),
                confidence_boundary_value=float(boundary["confidence"]),
                confidence_boundary_threshold=float(boundary["threshold"]),
                confidence_boundary_ratio=float(boundary["ratio"]),
                confidence_boundary_side=str(boundary["side"]),
                other_instability_score=max(jitter, low_confidence),
                temporal_random_score=random_score,
                track_group_id=track_group_id,
                usable_quality=feature.brightness >= dark_brightness_threshold and feature.contrast >= broken_contrast_threshold,
                model_counts=current_counts,
                comparison_detection_count=current_count,
                all_zero_detection=current_count == 0,
                eligibility_reason="comparison_detection" if current_count > 0 else "sparse_all_zero",
            ))
    return result


def cosine_novelty(candidate_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    if candidate_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if len(candidate_embeddings) == 0 or len(reference_embeddings) == 0:
        return np.zeros((len(candidate_embeddings),), dtype=np.float32)
    candidates = candidate_embeddings / np.maximum(np.linalg.norm(candidate_embeddings, axis=1, keepdims=True), 1e-8)
    references = reference_embeddings / np.maximum(np.linalg.norm(reference_embeddings, axis=1, keepdims=True), 1e-8)
    result = np.empty((len(candidates),), dtype=np.float32)
    chunk = 512
    for start in range(0, len(candidates), chunk):
        similarities = candidates[start : start + chunk] @ references.T
        result[start : start + chunk] = 1.0 - similarities.max(axis=1)
    minimum, maximum = float(result.min()), float(result.max())
    return (result - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(result)


def deterministic_kmeans(embeddings: np.ndarray, clusters: int, seed: int, iterations: int = 30) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0,), dtype=np.int32)
    clusters = max(1, min(int(clusters), len(embeddings)))
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(seed)
    centroids = normalized[rng.choice(len(normalized), size=clusters, replace=False)].copy()
    labels = np.zeros((len(normalized),), dtype=np.int32)
    for _ in range(iterations):
        new_labels = np.argmax(normalized @ centroids.T, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster in range(clusters):
            members = normalized[labels == cluster]
            if len(members):
                centroid = members.mean(axis=0)
                centroids[cluster] = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
    return labels
