from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .adapters import DinoDeimAdapter, create_adapter
from .config import PipelineConfig
from .media import frame_quality, pixel_sha256, sample_video_window
from .presence import crop_roi, padded_roi, union_bbox
from .scoring import (
    attach_confidence_percentiles,
    build_candidates,
    confidence_boundary_components,
    deterministic_kmeans,
    disagreement_components,
)
from .selection import adaptive_clip_bounds, quota_counts
from .tracking import (
    bbox_instability_score,
    consensus_detections,
    temporal_presence_score,
    temporal_score,
    track_consensus_frames,
)
from .types import Candidate, FrameFeature, VideoInfo

LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _stable_score(seed: int, *parts: object) -> float:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2**64 - 1)


def _load_sparse_features(config: PipelineConfig) -> list[FrameFeature]:
    table = pd.read_parquet(config.output_dir / "scans" / "frame_features.parquet")
    rows: list[FrameFeature] = []
    known = {name for name in FrameFeature.__dataclass_fields__ if name != "model_detection_counts"}
    for raw in table.to_dict(orient="records"):
        counts = {model.name: int(raw.get(f"{model.name}_detection_count", 0)) for model in config.models}
        values = {key: raw[key] for key in known if key in raw}
        rows.append(FrameFeature(**values, model_detection_counts=counts))
    return rows


def run_propose_stage(config: PipelineConfig, *, force: bool = False) -> list[dict[str, Any]]:
    output = config.output_dir / "proposals" / "dense_windows.json"
    csv_output = config.output_dir / "proposals" / "dense_windows.csv"
    if output.is_file() and not force:
        return json.loads(output.read_text(encoding="utf-8"))

    features = _load_sparse_features(config)
    raw_predictions: list[dict[str, Any]] = []
    for model in config.models:
        raw_predictions.extend(_read_jsonl(config.output_dir / "scans" / "model_predictions" / f"{model.name}.jsonl"))
    thresholds = {
        model.name: model.analysis_threshold if model.analysis_threshold is not None else model.confidence_threshold
        for model in config.models
    }
    sparse = build_candidates(
        features,
        raw_predictions,
        model_weights={model.name: model.model_weight for model in config.models},
        matching=config.matching,
        temporal_window_sec=config.temporal_window_sec,
        random_seed=config.random_seed,
        dark_brightness_threshold=config.dark_brightness_threshold,
        broken_contrast_threshold=config.broken_contrast_threshold,
        analysis_thresholds=thresholds,
        confidence_boundary_min_ratio=config.confidence_boundary.min_ratio,
        confidence_boundary_max_ratio=config.confidence_boundary.max_ratio,
    )
    raw_by_key: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_predictions:
        raw_by_key[(str(row["video_id"]), round(float(row["timestamp_sec"]), 6))].append(row)
    head_rows = _read_jsonl(config.output_dir / "scans" / "model_predictions" / f"{config.head_model.name}.jsonl") if config.head_model else []
    head_by_key: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        head_by_key[(str(row["video_id"]), round(float(row["timestamp_sec"]), 6))].append(row)

    rows_by_key = {(row.video_id, round(row.timestamp_sec, 6)): row for row in sparse}
    quotas = quota_counts(config.frames_per_video, config.selection_bucket_ratios)
    features_by_video: dict[str, list[FrameFeature]] = defaultdict(list)
    pools: dict[tuple[str, str], list[tuple[float, FrameFeature, str]]] = defaultdict(list)
    for feature in features:
        features_by_video[feature.video_id].append(feature)
        key = (feature.video_id, round(feature.timestamp_sec, 6))
        candidate = rows_by_key[key]
        raw = raw_by_key[key]
        analysis_count = candidate.comparison_detection_count
        heads = head_by_key.get(key, [])
        head_threshold = (
            config.head_model.analysis_threshold
            if config.head_model is not None and config.head_model.analysis_threshold is not None
            else config.head_model.confidence_threshold if config.head_model is not None else 0.0
        )
        head_score = max((float(row["confidence"]) for row in heads), default=0.0)
        boundary = confidence_boundary_components(
            raw,
            thresholds,
            min_ratio=config.confidence_boundary.min_ratio,
            max_ratio=config.confidence_boundary.max_ratio,
        )
        boundary_score = float(boundary["score"])
        pools[(feature.video_id, "model_disagreement")].append(
            (candidate.disagreement_score, feature, "sparse_disagreement")
        )
        pools[(feature.video_id, "temporal_anomaly")].append(
            (candidate.temporal_anomaly_score, feature, "sparse_temporal_change")
        )
        pools[(feature.video_id, "other_instability")].append(
            (max(boundary_score, candidate.common_difficulty_score), feature, "confidence_boundary_or_sparse_instability")
        )
        if candidate.bbox_instability_score > 0:
            pools[(feature.video_id, "bbox_instability")].append(
                (candidate.bbox_instability_score, feature, "sparse_bbox_instability")
            )
        if boundary_score > 0:
            pools[(feature.video_id, "low_confidence")].append(
                (boundary_score, feature, "confidence_near_model_threshold")
            )
        if analysis_count == 0 and head_score >= head_threshold:
            pools[(feature.video_id, "head_evidence")].append(
                (head_score, feature, "head_without_face_detection")
            )
        if analysis_count > 0 or head_score >= head_threshold:
            pools[(feature.video_id, "dino_novelty")].append(
                (
                    _stable_score(config.random_seed, "dino_proposal", feature.video_id, feature.timestamp_sec),
                    feature,
                    "relevance_gated_dino_proposal",
                )
            )

    # Uniform coverage is proposed independently of detector output. One
    # center-nearest sparse frame is retained for each temporal stratum.
    uniform_quota = quotas.get("uniform", 0)
    for video_id, video_features in features_by_video.items():
        ordered = sorted(video_features, key=lambda row: row.timestamp_sec)
        if not ordered or uniform_quota <= 0:
            continue
        maximum = max(ordered[-1].timestamp_sec + 1.0 / config.sample_fps, 1e-9)
        for bin_index in range(uniform_quota):
            center = (bin_index + 0.5) * maximum / uniform_quota
            members = [
                row for row in ordered
                if min(uniform_quota - 1, int(row.timestamp_sec / maximum * uniform_quota)) == bin_index
            ]
            if members:
                feature = min(members, key=lambda row: (abs(row.timestamp_sec - center), row.timestamp_sec))
                pools[(video_id, "uniform")].append(
                    (1.0 - min(1.0, abs(feature.timestamp_sec - center) / max(maximum / uniform_quota, 1e-9)), feature, f"uniform_bin_{bin_index:03d}")
                )

    selected: list[dict[str, Any]] = []
    for video_id in sorted(features_by_video):
        scene_count = max(1, len({row.scene_id for row in features_by_video[video_id]}))
        for bucket, quota in quotas.items():
            multiplier = 1 if bucket == "uniform" else config.dense_analysis.proposal_multiplier
            limit = max(1, quota * multiplier)
            ranked = sorted(
                pools.get((video_id, bucket), []),
                key=lambda item: (item[0], -item[1].timestamp_sec),
                reverse=True,
            )
            per_scene: dict[int, int] = defaultdict(int)
            for score, feature, reason in ranked:
                scene_cap = max(2, (limit + scene_count - 1) // scene_count)
                if per_scene[feature.scene_id] >= scene_cap and bucket != "uniform":
                    continue
                selected.append({
                    "video_id": feature.video_id,
                    "filename": feature.filename,
                    "video_path": feature.video_path,
                    "scene_id": int(feature.scene_id),
                    "center_sec": float(feature.timestamp_sec),
                    "start_sec": max(0.0, float(feature.timestamp_sec) - config.dense_analysis.half_window_sec),
                    "end_sec": float(feature.timestamp_sec) + config.dense_analysis.half_window_sec,
                    "proposal_bucket": bucket,
                    "proposal_score": float(score),
                    "proposal_reason": reason,
                })
                per_scene[feature.scene_id] += 1
                if sum(
                    row["video_id"] == video_id and row["proposal_bucket"] == bucket
                    for row in selected
                ) >= limit:
                    break

    # Merge overlapping windows only inside the same sparse scene.
    merged: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: (item["video_id"], item["scene_id"], item["start_sec"], item["end_sec"])):
        if merged and merged[-1]["video_id"] == row["video_id"] and merged[-1]["scene_id"] == row["scene_id"] and row["start_sec"] <= merged[-1]["end_sec"]:
            merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(row["end_sec"]))
            merged[-1]["proposal_score"] = max(float(merged[-1]["proposal_score"]), float(row["proposal_score"]))
            merged[-1]["proposal_buckets"] = sorted(set(merged[-1]["proposal_buckets"] + [row["proposal_bucket"]]))
            merged[-1]["proposal_reasons"] = sorted(set(merged[-1]["proposal_reasons"] + [row["proposal_reason"]]))
        else:
            merged.append({**row, "proposal_buckets": [row["proposal_bucket"]], "proposal_reasons": [row["proposal_reason"]]})
    for index, row in enumerate(merged, 1):
        row["window_id"] = f"window_{index:05d}"
        row["proposal_buckets"] = ";".join(row["proposal_buckets"])
        row["proposal_reasons"] = ";".join(row["proposal_reasons"])
        row.pop("proposal_bucket", None)
        row.pop("proposal_reason", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_output, merged)
    LOGGER.info("[propose] selected=%d merged_windows=%d", len(selected), len(merged))
    return merged


def run_dense_stage(
    config: PipelineConfig,
    inventory: Sequence[VideoInfo],
    proposals: Sequence[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    frame_output = config.output_dir / "dense" / "dense_frames.parquet"
    track_output = config.output_dir / "dense" / "tracks.parquet"
    embedding_output = config.output_dir / "dense" / "roi_embeddings.npy"
    if frame_output.is_file() and track_output.is_file() and embedding_output.is_file() and not force:
        return {"frames": len(pd.read_parquet(frame_output)), "reused": True}

    dino_model_config = next((model for model in config.models if model.embedding_source), None)
    if dino_model_config is None:
        # Directly constructed legacy test/config objects predate the explicit
        # embedding_source flag. YAML validation requires the explicit flag.
        dino_model_config = next(
            (model for model in config.models if model.type.lower() in {"dinov3_deimv2", "dino_deimv2", "deimv2"}),
            None,
        )
    if dino_model_config is None:
        raise RuntimeError("Dense ROI processing requires one configured DINOv3+DEIMv2 comparison model")
    adapters = [create_adapter(model) for model in config.models]
    head_adapter = create_adapter(config.head_model) if config.head_model is not None else None
    dino = next(
        (adapter for adapter in adapters if adapter.model_name == dino_model_config.name and isinstance(adapter, DinoDeimAdapter)),
        None,
    )
    if dino is None:
        raise RuntimeError("The configured DINO adapter did not expose ROI backbone embeddings")
    by_video = {row.video_id: row for row in inventory}
    dense_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    embedding_batches: list[np.ndarray] = []
    embedding_offset = 0
    face_thresholds = {
        model.name: model.analysis_threshold if model.analysis_threshold is not None else model.confidence_threshold
        for model in config.models
    }
    head_threshold = (
        config.head_model.analysis_threshold
        if config.head_model is not None and config.head_model.analysis_threshold is not None
        else config.head_model.confidence_threshold if config.head_model is not None else 0.0
    )

    for window_index, proposal in enumerate(proposals, 1):
        video = by_video[str(proposal["video_id"])]
        end = min(video.duration, float(proposal["end_sec"]))
        decoded = list(sample_video_window(
            video, start_sec=float(proposal["start_sec"]), end_sec=end, sample_fps=config.dense_analysis.fps
        ))
        if not decoded:
            continue
        timestamps = [float(row[0]) for row in decoded]
        frames = [row[2] for row in decoded]
        frame_meta: list[dict[str, Any]] = []
        previous_small: np.ndarray | None = None
        local_scene = 0
        for timestamp, _, frame in decoded:
            brightness, contrast, blur, phash, scene_score, previous_small = frame_quality(frame, previous_small)
            if scene_score >= config.scene_change_threshold:
                local_scene += 1
            frame_meta.append({
                "video_id": video.video_id, "filename": video.filename, "video_path": video.path,
                "window_id": proposal["window_id"], "timestamp_sec": float(timestamp),
                "frame_index": int(round(float(timestamp) * max(video.fps, 1.0))),
                "video_duration_sec": float(video.duration),
                "scene_id": int(proposal["scene_id"]) * 1000 + local_scene,
                "original_width": video.width, "original_height": video.height,
                "brightness": brightness, "contrast": contrast, "blur_score": blur,
                "perceptual_hash": phash, "pixel_sha256": pixel_sha256(frame),
                "scene_change_score": scene_score,
                "proposal_buckets": str(proposal.get("proposal_buckets", "")),
                "proposal_reasons": str(proposal.get("proposal_reasons", "")),
            })
        all_model_rows: dict[str, list[list[dict[str, Any]]]] = {}
        for adapter in [*adapters, *([head_adapter] if head_adapter is not None else [])]:
            assert adapter is not None
            batches: list[list[dict[str, Any]]] = []
            for start in range(0, len(frames), adapter.config.batch_size):
                predicted, _ = adapter.predict_batch(
                    frames[start:start + adapter.config.batch_size], video_id=video.video_id,
                    timestamps=timestamps[start:start + adapter.config.batch_size], return_embeddings=False,
                )
                batches.extend([[row.to_dict() for row in group] for group in predicted])
            all_model_rows[adapter.model_name] = batches

        comparison_flat = [row for adapter in adapters for group in all_model_rows[adapter.model_name] for row in group]
        attach_confidence_percentiles(comparison_flat)
        percentile_by_signature = {
            (str(row["model_name"]), round(float(row["timestamp_sec"]), 6), tuple(row["bbox_xyxy_original"])): row["confidence_percentile"]
            for row in comparison_flat
        }
        analysis_by_frame: list[list[dict[str, Any]]] = []
        raw_by_frame: list[list[dict[str, Any]]] = []
        head_by_frame: list[list[dict[str, Any]]] = []
        for frame_index in range(len(frames)):
            raw: list[dict[str, Any]] = []
            analysis: list[dict[str, Any]] = []
            for adapter in adapters:
                threshold = adapter.config.analysis_threshold if adapter.config.analysis_threshold is not None else adapter.config.confidence_threshold
                for row in all_model_rows[adapter.model_name][frame_index]:
                    signature = (str(row["model_name"]), round(float(row["timestamp_sec"]), 6), tuple(row["bbox_xyxy_original"]))
                    row["confidence_percentile"] = float(percentile_by_signature.get(signature, 0.0))
                    raw.append(row)
                    if float(row["confidence"]) >= threshold:
                        analysis.append(row)
            raw_by_frame.append(raw)
            analysis_by_frame.append(analysis)
            head_by_frame.append(all_model_rows[head_adapter.model_name][frame_index] if head_adapter is not None else [])

        tracking_input = [
            {**meta, "detections": consensus_detections(analysis)}
            for meta, analysis in zip(frame_meta, analysis_by_frame, strict=True)
        ]
        observations, summaries = track_consensus_frames(
            tracking_input, min_hits=config.dense_analysis.min_track_hits,
            max_gap_sec=config.dense_analysis.max_track_gap_sec,
            track_prefix=f"{video.video_id}_{proposal['window_id']}_track",
        )
        for row in observations:
            track_rows.append({"video_id": video.video_id, "window_id": proposal["window_id"], "track_scope": "consensus", "row_type": "observation", **row, "events": json.dumps(row["events"], sort_keys=True)})
        for row in summaries:
            track_rows.append({"video_id": video.video_id, "window_id": proposal["window_id"], "track_scope": "consensus", "row_type": "summary", **row})
        # Retain model-specific track evidence for diagnosing whether an
        # apparent fragmentation belongs to one detector or to the consensus.
        model_events_by_time: dict[float, dict[str, float]] = defaultdict(dict)
        for adapter in adapters:
            model_tracking_input = [
                {
                    **meta,
                    "detections": [
                        {
                            "bbox_xyxy_original": row["bbox_xyxy_original"],
                            "confidence": float(row.get("confidence_percentile", row.get("confidence", 0.0))),
                        }
                        for row in analysis if str(row["model_name"]) == adapter.model_name
                    ],
                }
                for meta, analysis in zip(frame_meta, analysis_by_frame, strict=True)
            ]
            model_observations, model_summaries = track_consensus_frames(
                model_tracking_input, min_hits=config.dense_analysis.min_track_hits,
                max_gap_sec=config.dense_analysis.max_track_gap_sec,
                track_prefix=f"{video.video_id}_{proposal['window_id']}_{adapter.model_name}_track",
            )
            for row in model_observations:
                timestamp_key = round(float(row["timestamp_sec"]), 6)
                for name, value in row["events"].items():
                    model_events_by_time[timestamp_key][name] = max(
                        model_events_by_time[timestamp_key].get(name, 0.0), float(value)
                    )
                track_rows.append({
                    "video_id": video.video_id, "window_id": proposal["window_id"],
                    "track_scope": adapter.model_name, "row_type": "observation", **row,
                    "events": json.dumps(row["events"], sort_keys=True),
                })
            for row in model_summaries:
                track_rows.append({
                    "video_id": video.video_id, "window_id": proposal["window_id"],
                    "track_scope": adapter.model_name, "row_type": "summary", **row,
                })
        observations_by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            observations_by_time[round(float(row["timestamp_sec"]), 6)].append(row)

        roi_crops: list[np.ndarray] = []
        roi_frame_indices: list[int] = []
        roi_details: list[tuple[list[str], tuple[float, float, float, float], str, dict[str, float]]] = []
        for frame_index, (frame, meta, analysis, heads) in enumerate(zip(frames, frame_meta, analysis_by_frame, head_by_frame, strict=True)):
            obs = observations_by_time.get(round(float(meta["timestamp_sec"]), 6), [])
            event_observation = max(obs, key=lambda row: temporal_score(row["events"]), default=None)
            sources: list[str] = []
            box: Sequence[float] | None = None
            track_id = ""
            events: dict[str, float] = {}
            current_boxes: list[Sequence[float]] = []
            for prediction in raw_by_frame[frame_index]:
                threshold = face_thresholds[str(prediction["model_name"])]
                if float(prediction["confidence"]) < threshold * config.confidence_boundary.min_ratio:
                    continue
                current_boxes.append(prediction["bbox_xyxy_original"])
                sources.append(f"face:{prediction['model_name']}")
            for prediction in heads:
                if float(prediction["confidence"]) < head_threshold * config.confidence_boundary.min_ratio:
                    continue
                current_boxes.append(prediction["bbox_xyxy_original"])
                sources.append(f"head:{prediction['model_name']}")
            box = union_bbox(current_boxes, width=video.width, height=video.height)
            if box is None:
                predicted = [row for row in obs if row["predicted"] and row["confirmed"]]
                if predicted:
                    best = max(predicted, key=lambda row: temporal_score(row["events"]))
                    sources, box, track_id, events = ["track"], best["bbox_xyxy_original"], str(best["track_id"]), dict(best["events"])
            if event_observation is not None:
                track_id = str(event_observation["track_id"])
                events = dict(event_observation["events"])
            for name, value in model_events_by_time.get(round(float(meta["timestamp_sec"]), 6), {}).items():
                events[name] = max(events.get(name, 0.0), float(value))
            roi = padded_roi(
                box, width=video.width, height=video.height,
                padding=config.novelty.roi_padding, min_side=config.novelty.roi_min_side,
            ) if box is not None else None
            if roi is None:
                roi = (0.0, 0.0, float(video.width), float(video.height))
                sources = ["full_frame"]
            roi_crops.append(crop_roi(frame, roi, config.novelty.roi_input_size))
            roi_frame_indices.append(frame_index)
            roi_details.append((sorted(set(sources)), roi, track_id, events))
            frame_meta[frame_index].update({
                "raw_predictions": json.dumps(raw_by_frame[frame_index], ensure_ascii=False, separators=(",", ":")),
                "analysis_predictions": json.dumps(analysis, ensure_ascii=False, separators=(",", ":")),
                "head_predictions": json.dumps(heads, ensure_ascii=False, separators=(",", ":")),
                "roi_source": "", "roi_sources": "", "roi_bbox": "", "roi_embedding_index": -1,
                "p_face_present": 0.0, "p_head_present": 0.0, "track_id": track_id,
                "track_events": json.dumps(events, sort_keys=True),
                "proposal_start_sec": float(proposal["start_sec"]), "proposal_end_sec": end,
            })
        embeddings = np.concatenate([
            dino.embed_batch(roi_crops[start:start + dino.config.batch_size], input_size=config.novelty.roi_input_size)
            for start in range(0, len(roi_crops), dino.config.batch_size)
        ], axis=0) if roi_crops else np.empty((0, 0), dtype=np.float32)
        for local_index, frame_index in enumerate(roi_frame_indices):
            sources, roi, track_id, events = roi_details[local_index]
            frame_meta[frame_index].update({
                "roi_source": "+".join(sources),
                "roi_sources": ";".join(sources),
                "roi_bbox": ",".join(f"{value:.3f}" for value in roi),
                "roi_embedding_index": embedding_offset + local_index,
                "p_face_present": 0.0,
                "p_head_present": 0.0,
                "track_id": track_id,
                "track_events": json.dumps(events, sort_keys=True),
            })
        if len(embeddings):
            embedding_batches.append(embeddings)
            embedding_offset += len(embeddings)
        dense_rows.extend(frame_meta)
        LOGGER.info("[dense] %d/%d %s frames=%d", window_index, len(proposals), proposal["window_id"], len(frames))

    frame_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dense_rows).to_parquet(frame_output, index=False)
    pd.DataFrame(track_rows).to_parquet(track_output, index=False)
    np.save(embedding_output, np.concatenate(embedding_batches, axis=0) if embedding_batches else np.empty((0, 0), dtype=np.float32))
    metrics = {
        "frames": len(dense_rows), "tracks": sum(row.get("row_type") == "summary" for row in track_rows),
        "windows": len(proposals),
        "roi_policy": "union_current_head_and_faces_then_track_then_full_frame",
    }
    (config.output_dir / "dense" / "dense_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def build_dense_candidates(config: PipelineConfig) -> list[Candidate]:
    table = pd.read_parquet(config.output_dir / "dense" / "dense_frames.parquet")
    if table.empty:
        return []
    track_table = pd.read_parquet(config.output_dir / "dense" / "tracks.parquet")
    summaries = {
        str(row["track_id"]): row
        for row in track_table.to_dict(orient="records")
        if row.get("row_type") == "summary" and row.get("track_scope") == "consensus"
    }
    scene_bounds = {
        (str(video_id), int(scene_id)): (float(group["timestamp_sec"].min()), float(group["timestamp_sec"].max()))
        for (video_id, scene_id), group in table.groupby(["video_id", "scene_id"])
    }
    presence_segments: dict[tuple[str, int, float], tuple[float, float]] = {}
    maximum_gap = 1.5 / max(config.dense_analysis.fps, 1e-6)
    for (video_id, scene_id), group in table.groupby(["video_id", "scene_id"]):
        active_rows = [
            row for row in group.sort_values("timestamp_sec").to_dict(orient="records")
            if str(row.get("roi_source") or "") != "full_frame"
        ]
        segment: list[dict[str, Any]] = []
        for row in active_rows:
            if segment and float(row["timestamp_sec"]) - float(segment[-1]["timestamp_sec"]) > maximum_gap:
                bounds = (float(segment[0]["timestamp_sec"]), float(segment[-1]["timestamp_sec"]))
                for member in segment:
                    presence_segments[(str(video_id), int(scene_id), round(float(member["timestamp_sec"]), 6))] = bounds
                segment = []
            segment.append(row)
        if segment:
            bounds = (float(segment[0]["timestamp_sec"]), float(segment[-1]["timestamp_sec"]))
            for member in segment:
                presence_segments[(str(video_id), int(scene_id), round(float(member["timestamp_sec"]), 6))] = bounds
    weights = {model.name: model.model_weight for model in config.models}
    thresholds = {
        model.name: model.analysis_threshold if model.analysis_threshold is not None else model.confidence_threshold
        for model in config.models
    }
    candidates: list[Candidate] = []
    for raw_row in table.to_dict(orient="records"):
        raw_predictions = json.loads(str(raw_row.get("raw_predictions") or "[]"))
        analysis_predictions = json.loads(str(raw_row.get("analysis_predictions") or "[]"))
        heads = json.loads(str(raw_row.get("head_predictions") or "[]"))
        events = json.loads(str(raw_row.get("track_events") or "{}"))
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in raw_predictions:
            raw_groups[str(prediction["model_name"])].append(prediction)
        for prediction in analysis_predictions:
            groups[str(prediction["model_name"])].append(prediction)
        for model in config.models:
            groups.setdefault(model.name, [])
            raw_groups.setdefault(model.name, [])
        comparison_count = len(analysis_predictions)
        model_counts = {name: len(rows) for name, rows in groups.items()}
        if config.head_model is not None:
            head_threshold = (
                config.head_model.analysis_threshold
                if config.head_model.analysis_threshold is not None
                else config.head_model.confidence_threshold
            )
            model_counts[config.head_model.name] = sum(
                float(row["confidence"]) >= head_threshold for row in heads
            )
        all_low = comparison_count == 0 and bool(raw_predictions)
        all_zero = not raw_predictions
        boundary = confidence_boundary_components(
            raw_predictions,
            thresholds,
            min_ratio=config.confidence_boundary.min_ratio,
            max_ratio=config.confidence_boundary.max_ratio,
        )
        confidence_boundary = float(boundary["score"])
        p_face = 0.0
        p_head = 0.0
        track_id = str(raw_row.get("track_id") or "")
        roi_sources = [item for item in str(raw_row.get("roi_sources") or "").split(";") if item]
        tracked_missing = "track" in roi_sources or "middle_missing" in events
        temporal = temporal_presence_score(events)
        bbox_instability = bbox_instability_score(events)
        other_instability = max(confidence_boundary, bbox_instability)
        no_detection = confidence_boundary if comparison_count == 0 else 0.0
        if comparison_count > 0:
            eligibility = "comparison_detection"
        elif raw_predictions:
            eligibility = "low_confidence"
        elif heads:
            eligibility = "head_evidence"
        elif tracked_missing:
            eligibility = "track_propagated"
        else:
            eligibility = "full_frame_fallback"
        roi_text = str(raw_row.get("roi_bbox") or "")
        roi_bbox = tuple(float(value) for value in roi_text.split(",")) if roi_text else None
        scene_key = (str(raw_row["video_id"]), int(raw_row["scene_id"]))
        scene_start, scene_end = scene_bounds[scene_key]
        summary = summaries.get(track_id, {})
        presence_start, presence_end = presence_segments.get(
            (str(raw_row["video_id"]), int(raw_row["scene_id"]), round(float(raw_row["timestamp_sec"]), 6)),
            (float(raw_row["timestamp_sec"]), float(raw_row["timestamp_sec"])),
        )
        clip_start, clip_end = adaptive_clip_bounds(
            center=float(raw_row["timestamp_sec"]),
            active_start=float(summary.get("start_sec", presence_start)),
            active_end=float(summary.get("end_sec", presence_end)),
            scene_start=scene_start,
            scene_end=scene_end,
            min_duration=config.clip_window.min_duration_sec,
            max_duration=config.clip_window.max_duration_sec,
            padding=config.clip_window.context_padding_sec,
        )
        disagreement = disagreement_components(groups, weights, config.matching)
        candidates.append(Candidate(
            video_id=str(raw_row["video_id"]), filename=str(raw_row["filename"]), video_path=str(raw_row["video_path"]),
            timestamp_sec=float(raw_row["timestamp_sec"]), scene_id=int(raw_row["scene_id"]),
            perceptual_hash=str(raw_row["perceptual_hash"]),
            frame_index=int(raw_row.get("frame_index", -1)),
            video_duration_sec=float(raw_row.get("video_duration_sec", 0.0)),
            pixel_sha256=str(raw_row.get("pixel_sha256") or ""),
            disagreement_score=disagreement["score"],
            common_difficulty_score=max(temporal, no_detection),
            temporal_anomaly_score=temporal,
            bbox_instability_score=bbox_instability,
            low_confidence_score=confidence_boundary,
            confidence_boundary_score=confidence_boundary,
            confidence_boundary_model=str(boundary["model"]),
            confidence_boundary_value=float(boundary["confidence"]),
            confidence_boundary_threshold=float(boundary["threshold"]),
            confidence_boundary_ratio=float(boundary["ratio"]),
            confidence_boundary_side=str(boundary["side"]),
            other_instability_score=other_instability,
            no_detection_evidence_score=no_detection,
            low_presence_audit_score=float(np.clip(1.0 - max(p_face, p_head), 0.0, 1.0)),
            temporal_random_score=_stable_score(config.random_seed, raw_row["video_id"], raw_row["timestamp_sec"]),
            track_group_id=track_id or f"{raw_row['video_id']}:s{raw_row['scene_id']}:no_track",
            track_id=track_id,
            track_event_types=sorted(events),
            score_components={
                **{f"temporal/{key}": float(value) for key, value in events.items()},
                **{f"disagreement/{key}": float(value) for key, value in disagreement.items() if key != "score"},
                "confidence_boundary/closeness": confidence_boundary,
                "confidence_boundary/ratio": float(boundary["ratio"]),
                "no_detection/low_confidence": confidence_boundary if comparison_count == 0 else 0.0,
                "no_detection/tracked_missing": float(tracked_missing),
                "other/bbox_instability": bbox_instability,
                "other/combined": other_instability,
            },
            comparison_detection_count=comparison_count,
            head_score=max((float(row["confidence"]) for row in heads), default=0.0),
            p_face_present=p_face, p_head_present=p_head,
            roi_source=str(raw_row.get("roi_source") or ""), roi_sources=roi_sources, roi_bbox=roi_bbox,
            roi_embedding_index=int(raw_row.get("roi_embedding_index", -1)),
            all_low_confidence=all_low, all_zero_detection=all_zero,
            tracked_missing_detection=tracked_missing, eligibility_reason=eligibility,
            clip_start_sec=clip_start, clip_end_sec=clip_end,
            usable_quality=float(raw_row["brightness"]) >= config.dark_brightness_threshold
            and float(raw_row["contrast"]) >= config.broken_contrast_threshold,
            model_counts=model_counts,
            model_predictions={
                **{name: list(rows) for name, rows in raw_groups.items()},
                **({config.head_model.name: list(heads)} if config.head_model is not None else {}),
            },
        ))

    embedding_path = config.output_dir / "dense" / "roi_embeddings.npy"
    if embedding_path.is_file():
        embeddings = np.load(embedding_path)
        for video_id in sorted({row.video_id for row in candidates}):
            score_by_bucket = {
                "model_disagreement": lambda row: row.disagreement_score,
                "temporal_anomaly": lambda row: row.temporal_anomaly_score,
                "bbox_instability": lambda row: row.bbox_instability_score,
                "low_confidence": lambda row: row.confidence_boundary_score,
                "head_evidence": lambda row: row.head_score if row.comparison_detection_count == 0 else 0.0,
                "dino_novelty": lambda row: 1.0 if row.roi_source != "full_frame" else 0.0,
                "other_instability": lambda row: row.other_instability_score,
                "uniform": lambda row: 1.0,
            }
            for bucket, score_of in score_by_bucket.items():
                indices = [
                    index for index, row in enumerate(candidates)
                    if row.video_id == video_id and row.roi_embedding_index >= 0 and score_of(row) > 0
                ]
                if not indices:
                    continue
                matrix = np.asarray([embeddings[candidates[index].roi_embedding_index] for index in indices])
                labels = deterministic_kmeans(
                    matrix,
                    config.novelty.cluster_count,
                    config.random_seed + sum(bucket.encode("utf-8")),
                )
                counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
                rarity = np.asarray([1.0 / counts[int(label)] for label in labels], dtype=np.float32)
                if rarity.max() > rarity.min():
                    rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min())
                else:
                    rarity = np.zeros_like(rarity)
                for candidate_index, score, label in zip(indices, rarity, labels, strict=True):
                    candidate = candidates[candidate_index]
                    candidate.novelty_score = max(candidate.novelty_score, float(score))
                    candidate.dino_cluster_by_bucket[bucket] = int(label)
    uniform_quota = quota_counts(config.frames_per_video, config.selection_bucket_ratios).get("uniform", 0)
    if uniform_quota > 0:
        for video_id in sorted({row.video_id for row in candidates}):
            rows = [row for row in candidates if row.video_id == video_id]
            if not rows:
                continue
            maximum = max(
                max((row.video_duration_sec for row in rows), default=0.0),
                max(row.timestamp_sec for row in rows) + 1.0 / max(config.dense_analysis.fps, 1e-6),
            )
            width = maximum / uniform_quota
            for row in rows:
                bin_index = min(uniform_quota - 1, int(row.timestamp_sec / max(maximum, 1e-9) * uniform_quota))
                center = (bin_index + 0.5) * width
                row.uniform_score = 1.0 - min(1.0, abs(row.timestamp_sec - center) / max(width, 1e-9))
    return candidates
