from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def build_window_starts(duration_sec: float, window_sec: float, step_sec: float) -> list[float]:
    """Return every full fixed-length window start, including an exact final boundary."""
    if duration_sec <= 0 or window_sec <= 0 or step_sec <= 0:
        raise ValueError("duration_sec, window_sec, and step_sec must be positive")
    if duration_sec + 1e-9 < window_sec:
        return []
    count = int(math.floor((duration_sec - window_sec + 1e-9) / step_sec)) + 1
    return [index * step_sec for index in range(count)]


def stable_percentiles(values: Sequence[float]) -> np.ndarray:
    """Return deterministic empirical ranks in [0, 1], averaging tied ranks."""
    if not values:
        return np.empty((0,), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / max(1, len(array) - 1)


def representative_points(percentile: float) -> int:
    if percentile >= 0.75:
        return 3
    if percentile >= 0.50:
        return 2
    if percentile >= 0.25:
        return 1
    return 0


def presence_points(ratio: float) -> int:
    if ratio >= 0.80:
        return 3
    if ratio >= 0.60:
        return 2
    if ratio >= 0.40:
        return 1
    return 0


def normal_band_points(percentile: float) -> int:
    """Reward normal moderate-to-high change, not frozen or extreme outliers."""
    if 0.35 <= percentile <= 0.90:
        return 2
    if 0.10 <= percentile < 0.35 or 0.90 < percentile <= 0.975:
        return 1
    return 0


def invalid_penalty(invalid_anchor_count: int) -> int:
    if invalid_anchor_count < 0:
        raise ValueError("invalid_anchor_count must be non-negative")
    return -min(5, invalid_anchor_count)


def normalized_js_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Jensen-Shannon similarity in [0, 1]."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("probability vectors must be one-dimensional with equal shapes")
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2.0

    def kl(values: np.ndarray, target: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / target[mask])))

    divergence = 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)
    return float(np.clip(1.0 - math.sqrt(max(0.0, divergence) / math.log(2.0)), 0.0, 1.0))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def representative_raw_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    cluster_count: int,
) -> float:
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("embedding and label dimensions do not match")
    if not indices:
        return 0.0
    global_counts = np.bincount(labels, minlength=cluster_count).astype(np.float64)
    window_counts = np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=cluster_count).astype(np.float64)
    distribution = normalized_js_similarity(window_counts, global_counts)
    global_center = embeddings.mean(axis=0)
    window_center = embeddings[np.asarray(indices, dtype=np.int64)].mean(axis=0)
    centroid = (cosine_similarity(window_center, global_center) + 1.0) / 2.0
    return float(np.clip(0.70 * distribution + 0.30 * centroid, 0.0, 1.0))


def robust_variation(values: np.ndarray, global_values: np.ndarray) -> float:
    """Mean window IQR in globally winsorized, robust-scaled feature space."""
    values = np.asarray(values, dtype=np.float64)
    global_values = np.asarray(global_values, dtype=np.float64)
    if values.ndim != 2 or global_values.ndim != 2 or values.shape[1] != global_values.shape[1]:
        raise ValueError("values must be two-dimensional with equal feature widths")
    if len(values) < 4 or len(global_values) < 4:
        return 0.0
    lower = np.nanpercentile(global_values, 5, axis=0)
    upper = np.nanpercentile(global_values, 95, axis=0)
    clipped_global = np.clip(global_values, lower, upper)
    clipped_window = np.clip(values, lower, upper)
    global_iqr = np.nanpercentile(clipped_global, 75, axis=0) - np.nanpercentile(clipped_global, 25, axis=0)
    window_iqr = np.nanpercentile(clipped_window, 75, axis=0) - np.nanpercentile(clipped_window, 25, axis=0)
    valid = np.isfinite(global_iqr) & np.isfinite(window_iqr) & (global_iqr > 1e-8)
    return float(np.nanmean(window_iqr[valid] / global_iqr[valid])) if np.any(valid) else 0.0
