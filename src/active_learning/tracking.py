from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_geometry(box: Sequence[float], width: float, height: float) -> tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    return (
        (x1 + x2) / 2.0 / max(1.0, width),
        (y1 + y2) / 2.0 / max(1.0, height),
        box_width / max(1.0, width),
        box_height / max(1.0, height),
        box_width * box_height,
    )


def association_metrics(left: Sequence[float], right: Sequence[float], width: float, height: float) -> tuple[float, float, float, float]:
    left_geo = bbox_geometry(left, width, height)
    right_geo = bbox_geometry(right, width, height)
    iou = bbox_iou(left, right)
    center = float(np.hypot(left_geo[0] - right_geo[0], left_geo[1] - right_geo[1]))
    area_ratio = right_geo[4] / max(1e-8, left_geo[4])
    cost = 0.5 * (1.0 - iou) + 0.3 * min(1.0, center) + 0.2 * min(1.0, abs(log(max(area_ratio, 1e-8))))
    return cost, iou, center, area_ratio


def consensus_detections(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping comparison-model boxes without including auxiliary head boxes."""

    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        box = row["bbox_xyxy_original"]
        width = float(row.get("original_width", 1))
        height = float(row.get("original_height", 1))
        best_index = -1
        best_iou = 0.0
        for index, cluster in enumerate(clusters):
            representative = np.median(np.asarray([item["bbox_xyxy_original"] for item in cluster]), axis=0)
            _, iou, center, area_ratio = association_metrics(representative, box, width, height)
            if (iou >= 0.30 or (center <= 0.12 and 0.33 <= area_ratio <= 3.0)) and iou >= best_iou:
                best_index, best_iou = index, iou
        if best_index < 0:
            clusters.append([row])
        elif str(row.get("model_name", "")) not in {str(item.get("model_name", "")) for item in clusters[best_index]}:
            clusters[best_index].append(row)

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        first = cluster[0]
        result.append({
            "bbox_xyxy_original": np.median(np.asarray([item["bbox_xyxy_original"] for item in cluster]), axis=0).tolist(),
            "confidence": float(np.mean([float(item.get("confidence_percentile", item.get("confidence", 0.0))) for item in cluster])),
            "model_names": sorted({str(item.get("model_name", "")) for item in cluster}),
            "original_width": int(first.get("original_width", 1)),
            "original_height": int(first.get("original_height", 1)),
        })
    return result


@dataclass(slots=True)
class _Track:
    track_id: str
    scene_id: int
    bbox: np.ndarray
    velocity: np.ndarray
    last_timestamp: float
    hits: int = 1
    misses: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, timestamp: float) -> np.ndarray:
        delta = max(0.0, timestamp - self.last_timestamp)
        return self.bbox + self.velocity * delta

    def update(self, detection: dict[str, Any], timestamp: float) -> None:
        new_box = np.asarray(detection["bbox_xyxy_original"], dtype=np.float64)
        delta = max(1e-6, timestamp - self.last_timestamp)
        measured_velocity = (new_box - self.bbox) / delta
        self.velocity = 0.7 * self.velocity + 0.3 * measured_velocity
        self.bbox = new_box
        self.last_timestamp = timestamp
        self.hits += 1
        self.misses = 0


def _robust_limit(values: Sequence[float], absolute: float) -> float:
    if len(values) < 4:
        return absolute
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(absolute, median + 3.0 * max(mad, 1e-6))


def _event_scores(observations: list[dict[str, Any]], min_hits: int) -> None:
    observed = [row for row in observations if not row["predicted"]]
    position_values: list[float] = []
    area_values: list[float] = []
    for previous, current in zip(observed, observed[1:]):
        px1, py1, px2, py2 = previous["bbox_xyxy_original"]
        cx1, cy1, cx2, cy2 = current["bbox_xyxy_original"]
        previous_area = max(1.0, (px2 - px1) * (py2 - py1))
        displacement = float(np.hypot((cx1 + cx2 - px1 - px2) / 2.0, (cy1 + cy2 - py1 - py2) / 2.0)) / sqrt(previous_area)
        area_ratio = max(1e-8, (cx2 - cx1) * (cy2 - cy1) / previous_area)
        position_values.append(displacement)
        area_values.append(abs(log(area_ratio)))
    position_limit = _robust_limit(position_values, 1.5)
    area_limit = _robust_limit(area_values, log(2.0))

    previous_observed: dict[str, Any] | None = None
    missing_run: list[dict[str, Any]] = []
    for row in observations:
        if row["predicted"]:
            missing_run.append(row)
            continue
        if previous_observed is not None:
            px1, py1, px2, py2 = previous_observed["bbox_xyxy_original"]
            cx1, cy1, cx2, cy2 = row["bbox_xyxy_original"]
            previous_area = max(1.0, (px2 - px1) * (py2 - py1))
            displacement = float(np.hypot((cx1 + cx2 - px1 - px2) / 2.0, (cy1 + cy2 - py1 - py2) / 2.0)) / sqrt(previous_area)
            area_ratio = max(1e-8, (cx2 - cx1) * (cy2 - cy1) / previous_area)
            previous_aspect = max(1e-8, (px2 - px1) / max(1e-8, py2 - py1))
            current_aspect = max(1e-8, (cx2 - cx1) / max(1e-8, cy2 - cy1))
            if displacement > position_limit:
                row["events"]["position_jump"] = min(1.0, displacement / max(position_limit, 1e-8) - 0.5)
            shape_change = max(abs(log(area_ratio)), abs(log(current_aspect / previous_aspect)))
            if shape_change > area_limit:
                row["events"]["shape_jump"] = min(1.0, shape_change / max(area_limit, 1e-8) - 0.5)
            confidence_drop = float(previous_observed["confidence"]) - float(row["confidence"])
            if confidence_drop >= 0.4:
                row["events"]["confidence_drop"] = min(1.0, confidence_drop)
        if missing_run and previous_observed is not None and len(observed) >= min_hits:
            for missing in missing_run:
                missing["events"]["middle_missing"] = 1.0
            if len(missing_run) >= 2:
                missing_run[-1]["events"]["disappearance"] = 1.0
        missing_run = []
        previous_observed = row
    # A trailing gap is classified after the complete scene is known. A gap
    # that merely reaches the scene/video edge must not become an anomaly.


def track_consensus_frames(
    frames: Sequence[dict[str, Any]],
    *,
    min_hits: int = 3,
    max_gap_sec: float = 1.0,
    track_prefix: str = "track",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Track consensus face boxes. Every input row needs timestamp, scene, dimensions and detections."""

    active: list[_Track] = []
    finished: list[_Track] = []
    next_id = 1
    previous_scene: int | None = None
    frame_step = float(np.median(np.diff([float(row["timestamp_sec"]) for row in frames]))) if len(frames) > 1 else 0.2

    for frame in sorted(frames, key=lambda row: float(row["timestamp_sec"])):
        timestamp = float(frame["timestamp_sec"])
        scene_id = int(frame["scene_id"])
        width = float(frame["original_width"])
        height = float(frame["original_height"])
        if previous_scene is not None and scene_id != previous_scene:
            finished.extend(active)
            active = []
        previous_scene = scene_id
        retained: list[_Track] = []
        for track in active:
            if timestamp - track.last_timestamp > max_gap_sec + frame_step / 2.0:
                finished.append(track)
            else:
                retained.append(track)
        active = retained
        detections = list(frame.get("detections", []))
        predictions = [track.predict(timestamp) for track in active]
        costs = np.full((len(active), len(detections)), 1e6, dtype=np.float64)
        for track_index, predicted in enumerate(predictions):
            for detection_index, detection in enumerate(detections):
                cost, iou, center, area_ratio = association_metrics(
                    predicted, detection["bbox_xyxy_original"], width, height
                )
                if iou >= 0.10 or (center <= 0.20 and 0.25 <= area_ratio <= 4.0):
                    costs[track_index, detection_index] = cost
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        if costs.size:
            rows, columns = linear_sum_assignment(costs)
            for track_index, detection_index in zip(rows.tolist(), columns.tolist(), strict=True):
                if costs[track_index, detection_index] >= 1e5:
                    continue
                track = active[track_index]
                detection = detections[detection_index]
                track.update(detection, timestamp)
                track.observations.append({
                    "track_id": track.track_id, "timestamp_sec": timestamp, "scene_id": scene_id,
                    "bbox_xyxy_original": track.bbox.tolist(), "confidence": float(detection.get("confidence", 0.0)),
                    "predicted": False, "confirmed": track.hits >= min_hits, "events": {},
                })
                assigned_tracks.add(track_index)
                assigned_detections.add(detection_index)
        survivors: list[_Track] = []
        for track_index, track in enumerate(active):
            if track_index not in assigned_tracks:
                predicted = predictions[track_index]
                track.misses += 1
                track.observations.append({
                    "track_id": track.track_id, "timestamp_sec": timestamp, "scene_id": scene_id,
                    "bbox_xyxy_original": predicted.tolist(), "confidence": 0.0, "predicted": True,
                    "confirmed": track.hits >= min_hits, "events": {},
                })
            survivors.append(track)
        active = survivors
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            track = _Track(
                track_id=f"{track_prefix}_{next_id:06d}", scene_id=scene_id,
                bbox=np.asarray(detection["bbox_xyxy_original"], dtype=np.float64),
                velocity=np.zeros(4, dtype=np.float64), last_timestamp=timestamp,
            )
            next_id += 1
            track.observations.append({
                "track_id": track.track_id, "timestamp_sec": timestamp, "scene_id": scene_id,
                "bbox_xyxy_original": track.bbox.tolist(), "confidence": float(detection.get("confidence", 0.0)),
                "predicted": False, "confirmed": False, "events": {},
            })
            active.append(track)
    finished.extend(active)

    scene_end = {
        int(scene_id): max(float(row["timestamp_sec"]) for row in frames if int(row["scene_id"]) == int(scene_id))
        for scene_id in {int(row["scene_id"]) for row in frames}
    }

    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for track in finished:
        _event_scores(track.observations, min_hits)
        observed_before_tail = [row for row in track.observations if not row["predicted"]]
        trailing = []
        for row in reversed(track.observations):
            if not row["predicted"]:
                break
            trailing.append(row)
        if (
            len(trailing) >= 2 and len(observed_before_tail) >= min_hits
            and scene_end.get(track.scene_id, track.observations[-1]["timestamp_sec"])
            - float(track.observations[-1]["timestamp_sec"]) >= frame_step - 1e-6
        ):
            trailing[0]["events"]["disappearance"] = 1.0
        observations.extend(track.observations)
        observed = [row for row in track.observations if not row["predicted"]]
        summaries.append({
            "track_id": track.track_id,
            "scene_id": track.scene_id,
            "hits": len(observed),
            "confirmed": len(observed) >= min_hits,
            "start_sec": min((float(row["timestamp_sec"]) for row in track.observations), default=0.0),
            "end_sec": max((float(row["timestamp_sec"]) for row in track.observations), default=0.0),
            "first_observed_sec": min((float(row["timestamp_sec"]) for row in observed), default=0.0),
            "last_observed_sec": max((float(row["timestamp_sec"]) for row in observed), default=0.0),
        })

    # A geometry-compatible new tracklet shortly after a confirmed one is a fragmentation event.
    confirmed = [row for row in summaries if row["confirmed"]]
    by_id = {row["track_id"]: [item for item in observations if item["track_id"] == row["track_id"]] for row in summaries}
    for left in confirmed:
        left_rows = [row for row in by_id[left["track_id"]] if not row["predicted"]]
        if not left_rows:
            continue
        for right in confirmed:
            gap = float(right["first_observed_sec"]) - float(left["last_observed_sec"])
            if left["track_id"] == right["track_id"] or left["scene_id"] != right["scene_id"] or not 0 < gap <= 1.5:
                continue
            right_rows = [row for row in by_id[right["track_id"]] if not row["predicted"]]
            if not right_rows:
                continue
            box_left = left_rows[-1]["bbox_xyxy_original"]
            box_right = right_rows[0]["bbox_xyxy_original"]
            width = max(max(box_left[2], box_right[2]), 1.0)
            height = max(max(box_left[3], box_right[3]), 1.0)
            _, iou, center, area_ratio = association_metrics(box_left, box_right, width, height)
            if iou >= 0.10 or (center <= 0.20 and 0.25 <= area_ratio <= 4.0):
                right_rows[0]["events"]["id_break"] = 1.0

    observations.sort(key=lambda row: (float(row["timestamp_sec"]), str(row["track_id"])))
    return observations, summaries


TEMPORAL_EVENT_WEIGHTS = {
    "middle_missing": 0.25,
    "disappearance": 0.20,
    "position_jump": 0.15,
    "shape_jump": 0.15,
    "confidence_drop": 0.15,
    "id_break": 0.10,
}


def temporal_score(events: dict[str, float]) -> float:
    return float(np.clip(sum(TEMPORAL_EVENT_WEIGHTS.get(name, 0.0) * float(value) for name, value in events.items()), 0.0, 1.0))


def temporal_presence_score(events: dict[str, float]) -> float:
    """Score detector flicker and track fragmentation without bbox jitter."""

    weights = {"middle_missing": 0.45, "disappearance": 0.30, "id_break": 0.25}
    return float(np.clip(sum(weights.get(name, 0.0) * float(value) for name, value in events.items()), 0.0, 1.0))


def bbox_instability_score(events: dict[str, float]) -> float:
    """Score only geometry instability while a detection remains present."""

    weights = {"position_jump": 0.50, "shape_jump": 0.50}
    return float(np.clip(sum(weights.get(name, 0.0) * float(value) for name, value in events.items()), 0.0, 1.0))
