from __future__ import annotations

import csv
import fnmatch
import gc
import hashlib
import json
import logging
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

from script.script_data.clip_frame import find_videos, safe_name

from .adapters import DinoDeimAdapter, create_adapter
from .config import PipelineConfig
from .media import check_media_commands, frame_quality, pixel_sha256, probe_video, read_video_frame, sample_frames_ffmpeg
from .render import cut_source_clip, render_review_clip
from .report import selection_reason_label, write_html_report, write_review_template
from .scoring import build_candidates, cosine_novelty, deterministic_kmeans
from .selection import select_frame_candidates
from .stages import build_dense_candidates, run_dense_stage, run_propose_stage
from .types import Candidate, FrameFeature, VideoInfo

LOGGER = logging.getLogger(__name__)


def _scan_model_configs(config: PipelineConfig) -> list[Any]:
    return [*config.models, *([config.head_model] if config.head_model is not None else [])]


def _input_videos(config: PipelineConfig) -> list[Path]:
    videos = find_videos(config.input_dir, config.recursive)
    if not config.video_include_globs:
        return videos
    return [
        path for path in videos
        if any(
            fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path.stem, pattern)
            for pattern in config.video_include_globs
        )
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=names, extrasaction="ignore")
        if names:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _load_split_sources(paths: dict[str, Path]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split, path in paths.items():
        if not path.is_file():
            LOGGER.warning("Split annotation not found for exclusion audit: %s", path)
            result[split] = set()
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        values = set()
        for image in data.get("images", []):
            metadata = image.get("metadata") or {}
            value = metadata.get("source_video_id") or image.get("source_video_id")
            if value:
                values.add(str(value).lower())
        result[split] = values
    return result


def _matched_split(video: VideoInfo, split_sources: dict[str, set[str]]) -> str:
    stem = Path(video.filename).stem.lower()
    video_id = video.video_id.lower()
    stem_hash = hashlib.sha1(Path(video.filename).stem.encode("utf-8")).hexdigest()[:8]
    for split in ("test", "val", "validation", "train"):
        for source_id in split_sources.get(split, set()):
            if source_id in stem or video_id in source_id or source_id == video_id or source_id.endswith(stem_hash):
                return "val" if split == "validation" else split
    return ""


def run_inventory(config: PipelineConfig, *, force: bool = False) -> list[VideoInfo]:
    check_media_commands()
    csv_path = config.output_dir / "inventory" / "video_inventory.csv"
    json_path = config.output_dir / "inventory" / "video_inventory.json"
    if csv_path.exists() and json_path.exists() and not force:
        LOGGER.info("[inventory] reuse %s", json_path)
        return [VideoInfo(**row) for row in json.loads(json_path.read_text(encoding="utf-8"))]
    videos = _input_videos(config)
    if not videos:
        raise FileNotFoundError(
            f"No input videos matched video_include_globs={config.video_include_globs!r} in {config.input_dir}"
        )
    split_sources = _load_split_sources(config.split_annotations)
    rows: list[VideoInfo] = []
    started = time.perf_counter()
    for index, path in enumerate(videos, 1):
        item_started = time.perf_counter()
        info = probe_video(path)
        info.excluded_split = _matched_split(info, split_sources)
        if info.excluded_split:
            info.readable = False
            info.error = f"excluded because source video appears in {info.excluded_split} split"
        rows.append(info)
        if info.error:
            error_path = config.output_dir / "logs" / "inventory_errors" / f"{info.video_id}.log"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(info.error + "\n", encoding="utf-8")
        LOGGER.info("[inventory] %d/%d %s (%.2fs)", index, len(videos), info.video_id, time.perf_counter() - item_started)
    payload = [row.to_dict() for row in rows]
    _write_csv(csv_path, payload, [field.name for field in fields(VideoInfo)])
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("[inventory] completed %d video(s) in %.2fs", len(rows), time.perf_counter() - started)
    return rows


def _merge_scan_parts(config: PipelineConfig, inventory: Sequence[VideoInfo]) -> None:
    scan_dir = config.output_dir / "scans"
    feature_parts = scan_dir / ".parts" / "features"
    embedding_parts = scan_dir / ".parts" / "embeddings"
    frames: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []
    embedding_offset = 0
    completed_video_ids: set[str] = set()
    for video in inventory:
        if not video.readable or video.excluded_split:
            continue
        part_path = feature_parts / f"{video.video_id}.parquet"
        prediction_parts = [scan_dir / ".parts" / "predictions" / model.name / f"{video.video_id}.jsonl" for model in _scan_model_configs(config)]
        embedding_path = embedding_parts / f"{video.video_id}.npy"
        needs_embeddings = not config.dense_analysis.enabled and any(model.type.lower() in {"dinov3_deimv2", "dino_deimv2", "deimv2"} for model in config.models)
        if not part_path.exists() or not all(path.exists() for path in prediction_parts) or (needs_embeddings and not embedding_path.exists()):
            continue
        table = pd.read_parquet(part_path)
        if embedding_path.exists():
            array = np.load(embedding_path)
            if len(array) != len(table):
                raise RuntimeError(
                    f"DINO embedding/frame count mismatch for {video.video_id}: {len(array)} != {len(table)}"
                )
            # Embeddings are emitted once per sampled frame in exact table
            # order. Reconstruct indices here to make part merging robust to
            # interrupted runs and independently written part offsets.
            table["dino_embedding_index"] = np.arange(len(table), dtype=np.int64) + embedding_offset
            embedding_offset += len(array)
            embeddings.append(array)
        frames.append(table)
        completed_video_ids.add(video.video_id)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_parquet(scan_dir / "frame_features.parquet", index=False)
    if embeddings:
        np.save(scan_dir / "dino_embeddings.npy", np.concatenate(embeddings, axis=0))
    predictions_dir = scan_dir / "model_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for model in _scan_model_configs(config):
        destination = predictions_dir / f"{model.name}.jsonl"
        with destination.open("w", encoding="utf-8") as output:
            for video in inventory:
                if video.video_id not in completed_video_ids:
                    continue
                part = scan_dir / ".parts" / "predictions" / model.name / f"{video.video_id}.jsonl"
                if part.exists():
                    with part.open(encoding="utf-8") as source:
                        shutil.copyfileobj(source, output)


def _build_reference_embeddings(config: PipelineConfig, adapter: DinoDeimAdapter, *, force: bool) -> None:
    output = config.output_dir / "scans" / "reference_dino_embeddings.npy"
    metadata_output = config.output_dir / "scans" / "reference_dino_embeddings.json"
    annotation = config.novelty.reference_annotation
    root = config.novelty.reference_image_root
    if not config.novelty.enabled or annotation is None or root is None:
        LOGGER.warning("Novelty reference is disabled or not configured; reference-distance novelty cannot be computed")
        return
    if output.exists() and metadata_output.exists() and not force:
        return
    data = json.loads(annotation.read_text(encoding="utf-8"))
    images = list(data.get("images", []))
    rng = np.random.default_rng(config.random_seed)
    if len(images) > config.novelty.reference_sample_count:
        indices = sorted(rng.choice(len(images), config.novelty.reference_sample_count, replace=False).tolist())
        images = [images[index] for index in indices]
    embeddings: list[np.ndarray] = []
    retained: list[str] = []
    batch_frames: list[np.ndarray] = []
    batch_names: list[str] = []

    def flush() -> None:
        if not batch_frames:
            return
        _, values = adapter.predict_batch(
            batch_frames,
            video_id="reference_dataset",
            timestamps=[float(index) for index in range(len(batch_frames))],
            return_embeddings=True,
        )
        if values is None:
            raise RuntimeError("DINO adapter did not return real embeddings")
        embeddings.append(values)
        retained.extend(batch_names)
        batch_frames.clear()
        batch_names.clear()

    for image in images:
        path = root / str(image.get("file_name", ""))
        frame = cv2.imread(path.as_posix())
        if frame is None:
            LOGGER.warning("Cannot read novelty reference image: %s", path)
            continue
        batch_frames.append(frame)
        batch_names.append(path.as_posix())
        if len(batch_frames) >= adapter.config.batch_size:
            flush()
    flush()
    if not embeddings:
        raise RuntimeError("No valid images were available for real DINO novelty reference embeddings")
    np.save(output, np.concatenate(embeddings, axis=0))
    metadata_output.write_text(json.dumps({"annotation": annotation.as_posix(), "images": retained}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_scan(config: PipelineConfig, inventory: Sequence[VideoInfo], *, force: bool = False, resume: bool = False) -> dict[str, Any]:
    scan_dir = config.output_dir / "scans"
    metrics_path = scan_dir / "scan_metrics.json"
    final_features = scan_dir / "frame_features.parquet"
    final_predictions = [scan_dir / "model_predictions" / f"{model.name}.jsonl" for model in _scan_model_configs(config)]
    if force:
        for generated in [final_features, scan_dir / "dino_embeddings.npy", *final_predictions]:
            generated.unlink(missing_ok=True)
    if final_features.exists() and all(path.exists() for path in final_predictions) and not force and not resume:
        LOGGER.info("[scan] reuse completed scan outputs")
        return json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    adapters = [create_adapter(model) for model in _scan_model_configs(config)]
    dino_adapter = next((adapter for adapter in adapters if isinstance(adapter, DinoDeimAdapter)), None)
    store_full_frame_embeddings = not config.dense_analysis.enabled
    previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if resume and metrics_path.exists() and not force else {}
    metrics: dict[str, Any] = {
        "videos": dict(previous_metrics.get("videos", {})),
        "models": dict(previous_metrics.get("models", {})),
    }
    usable = [video for video in inventory if video.readable and not video.excluded_split]
    stage_started = time.perf_counter()
    for video_index, video in enumerate(usable, 1):
        feature_part = scan_dir / ".parts" / "features" / f"{video.video_id}.parquet"
        embedding_part = scan_dir / ".parts" / "embeddings" / f"{video.video_id}.npy"
        prediction_parts = {
            adapter.model_name: scan_dir / ".parts" / "predictions" / adapter.model_name / f"{video.video_id}.jsonl"
            for adapter in adapters
        }
        complete = feature_part.exists() and all(path.exists() for path in prediction_parts.values()) and (dino_adapter is None or not store_full_frame_embeddings or embedding_part.exists())
        if complete and not force:
            LOGGER.info("[scan] %d/%d reuse %s", video_index, len(usable), video.video_id)
            continue
        for path in [feature_part, embedding_part, *prediction_parts.values()]:
            path.parent.mkdir(parents=True, exist_ok=True)
        handles = {name: path.open("w", encoding="utf-8") for name, path in prediction_parts.items()}
        features: list[FrameFeature] = []
        embedding_batches: list[np.ndarray] = []
        batch_frames: list[np.ndarray] = []
        batch_timestamps: list[float] = []
        batch_features: list[FrameFeature] = []
        previous_small: np.ndarray | None = None
        scene_id = 0
        video_started = time.perf_counter()
        model_seconds_before = {adapter.model_name: adapter.inference_seconds for adapter in adapters}
        batch_size = max(1, min(adapter.config.batch_size for adapter in adapters))

        def flush() -> None:
            if not batch_frames:
                return
            for adapter in adapters:
                prediction_batches, embeddings = adapter.predict_batch(
                    batch_frames,
                    video_id=video.video_id,
                    timestamps=batch_timestamps,
                    return_embeddings=store_full_frame_embeddings and adapter is dino_adapter,
                )
                if embeddings is not None:
                    embedding_batches.append(embeddings)
                for feature, rows in zip(batch_features, prediction_batches, strict=True):
                    if config.head_model is not None and adapter.model_name == config.head_model.name:
                        feature.head_detection_count = len(rows)
                    else:
                        feature.model_detection_counts[adapter.model_name] = len(rows)
                    for row in rows:
                        handles[adapter.model_name].write(json.dumps(row.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            features.extend(batch_features)
            batch_frames.clear()
            batch_timestamps.clear()
            batch_features.clear()

        video_error: Exception | None = None
        try:
            for timestamp, frame_index, frame in sample_frames_ffmpeg(video, config.sample_fps):
                brightness, contrast, blur, phash, scene_score, small = frame_quality(frame, previous_small)
                previous_small = small
                if scene_score >= config.scene_change_threshold:
                    scene_id += 1
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                batch_features.append(FrameFeature(
                    video_id=video.video_id,
                    filename=video.filename,
                    video_path=video.path,
                    timestamp_sec=timestamp,
                    frame_index=frame_index,
                    original_width=video.width,
                    original_height=video.height,
                    brightness=brightness,
                    contrast=contrast,
                    blur_score=blur,
                    perceptual_hash=phash,
                    pixel_sha256=pixel_sha256(frame),
                    scene_change_score=scene_score,
                    scene_id=scene_id,
                    dino_embedding_index=len(features) + len(batch_features) if dino_adapter is not None and store_full_frame_embeddings else -1,
                ))
                if len(batch_frames) >= batch_size:
                    flush()
            flush()
        except Exception as error:
            video_error = error
        finally:
            for handle in handles.values():
                handle.close()
        if video_error is not None:
            error_path = config.output_dir / "logs" / "scan_errors" / f"{video.video_id}.log"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(video_error) + "\n", encoding="utf-8")
            metrics.setdefault("errors", {})[video.video_id] = str(video_error)
            LOGGER.exception("[scan] %d/%d failed %s; continuing with remaining videos", video_index, len(usable), video.video_id, exc_info=video_error)
            continue
        pd.DataFrame([row.to_flat_dict() for row in features]).to_parquet(feature_part, index=False)
        if dino_adapter is not None and store_full_frame_embeddings:
            if not embedding_batches:
                raise RuntimeError(f"No real DINO embeddings produced for {video.video_id}")
            np.save(embedding_part, np.concatenate(embedding_batches, axis=0))
        elapsed = time.perf_counter() - video_started
        metrics["videos"][video.video_id] = {
            "frames": len(features),
            "elapsed_sec": elapsed,
            "decode_and_all_models_fps": len(features) / elapsed if elapsed else 0.0,
            "model_inference_seconds": {
                adapter.model_name: adapter.inference_seconds - model_seconds_before[adapter.model_name]
                for adapter in adapters
            },
        }
        LOGGER.info("[scan] %d/%d %s frames=%d elapsed=%.1fs", video_index, len(usable), video.video_id, len(features), elapsed)
        for adapter in adapters:
            LOGGER.info(
                "[scan:%s] %d/%d cumulative_frames=%d cumulative_inference_sec=%.1f effective_fps=%.2f",
                adapter.model_name,
                video_index,
                len(usable),
                adapter.frames_processed,
                adapter.inference_seconds,
                adapter.effective_fps,
            )
    if dino_adapter is not None and store_full_frame_embeddings:
        _build_reference_embeddings(config, dino_adapter, force=force)
    _merge_scan_parts(config, usable)
    for adapter in adapters:
        prior = dict(metrics["models"].get(adapter.model_name, {}))
        prior_seconds = float(prior.get("inference_seconds", 0.0))
        prior_frames = int(prior.get("frames_processed", 0))
        total_seconds = prior_seconds + adapter.inference_seconds
        total_frames = prior_frames + adapter.frames_processed
        metrics["models"][adapter.model_name] = {
            "inference_seconds": total_seconds,
            "frames_processed": total_frames,
            "effective_inference_fps": total_frames / total_seconds if total_seconds else 0.0,
            "weights": adapter.config.weights.as_posix(),
        }
    metrics["elapsed_sec"] = float(previous_metrics.get("elapsed_sec", 0.0)) + time.perf_counter() - stage_started
    metrics["sample_fps"] = config.sample_fps
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del adapters
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return metrics


def _load_features(config: PipelineConfig) -> list[FrameFeature]:
    table = pd.read_parquet(config.output_dir / "scans" / "frame_features.parquet")
    model_columns = {model.name: f"{model.name}_detection_count" for model in config.models}
    rows: list[FrameFeature] = []
    for row in table.to_dict(orient="records"):
        counts = {name: int(row.get(column, 0)) for name, column in model_columns.items()}
        known = {field.name for field in fields(FrameFeature)} - {"model_detection_counts"}
        values = {key: row[key] for key in known if key in row}
        rows.append(FrameFeature(**values, model_detection_counts=counts))
    return rows


def _load_predictions(config: PipelineConfig, minimum_confidence: dict[str, float] | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in config.models:
        path = config.output_dir / "scans" / "model_predictions" / f"{model.name}.jsonl"
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                threshold = (minimum_confidence or {}).get(model.name, float("-inf"))
                if float(row["confidence"]) >= threshold:
                    result.append(row)
    return result


def run_propose(config: PipelineConfig, *, force: bool = False) -> list[dict[str, Any]]:
    return run_propose_stage(config, force=force)


def run_dense(
    config: PipelineConfig,
    inventory: Sequence[VideoInfo],
    proposals: Sequence[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    return run_dense_stage(config, inventory, proposals, force=force)


def run_score(config: PipelineConfig, *, force: bool = False) -> list[Candidate]:
    output = config.output_dir / "candidates" / "all_candidates.csv"
    cache = config.output_dir / "candidates" / "all_candidates.jsonl"
    if output.exists() and cache.exists() and not force:
        LOGGER.info("[score] reuse %s", output)
        return [Candidate(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines() if line]
    if config.dense_analysis.enabled and not (config.output_dir / "dense" / "dense_frames.parquet").is_file():
        raise FileNotFoundError("Dense analysis is enabled; run --stage propose and --stage dense before score")
    if config.dense_analysis.enabled:
        started = time.perf_counter()
        candidates = build_dense_candidates(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output, [row.to_flat_dict() for row in candidates])
        with cache.open("w", encoding="utf-8") as destination:
            for row in candidates:
                destination.write(json.dumps({field.name: getattr(row, field.name) for field in fields(Candidate)}, ensure_ascii=False) + "\n")
        metrics = {
            "candidate_count": len(candidates), "elapsed_sec": time.perf_counter() - started,
            "source": "dense_roi", "full_frame_novelty": False,
        }
        (config.output_dir / "candidates" / "score_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return candidates
    analysis_thresholds = {
        model.name: model.analysis_threshold if model.analysis_threshold is not None else model.confidence_threshold
        for model in config.models
    }
    features = _load_features(config)
    # Retain the uncertainty band plus all above-threshold detections. Raw
    # floor detections far below the decision boundary are intentionally
    # excluded from active-learning priority and score-stage memory.
    boundary_floors = {
        model.name: max(
            model.confidence_threshold,
            analysis_thresholds[model.name] * config.confidence_boundary.min_ratio,
        )
        for model in config.models
    }
    predictions = _load_predictions(config, boundary_floors)
    weights = {model.name: model.model_weight for model in config.models}
    started = time.perf_counter()
    candidates = build_candidates(
        features,
        predictions,
        model_weights=weights,
        matching=config.matching,
        temporal_window_sec=config.temporal_window_sec,
        random_seed=config.random_seed,
        dark_brightness_threshold=config.dark_brightness_threshold,
        broken_contrast_threshold=config.broken_contrast_threshold,
        analysis_thresholds=analysis_thresholds,
        confidence_boundary_min_ratio=config.confidence_boundary.min_ratio,
        confidence_boundary_max_ratio=config.confidence_boundary.max_ratio,
    )
    embedding_path = config.output_dir / "scans" / "dino_embeddings.npy"
    reference_path = config.output_dir / "scans" / "reference_dino_embeddings.npy"
    if embedding_path.exists():
        all_embeddings = np.load(embedding_path)
        feature_index = {(row.video_id, round(row.timestamp_sec, 6)): row.dino_embedding_index for row in features}
        indices = np.asarray([feature_index[(row.video_id, round(row.timestamp_sec, 6))] for row in candidates], dtype=np.int64)
        candidate_embeddings = all_embeddings[indices]
        labels = deterministic_kmeans(candidate_embeddings, config.novelty.cluster_count, config.random_seed)
        cluster_sizes = Counter(labels.tolist())
        rarity = np.asarray([1.0 / cluster_sizes[int(label)] for label in labels], dtype=np.float32)
        rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min()) if rarity.max() > rarity.min() else np.zeros_like(rarity)
        if reference_path.exists():
            novelty = 0.75 * cosine_novelty(candidate_embeddings, np.load(reference_path)) + 0.25 * rarity
        else:
            LOGGER.warning("Real DINO reference embeddings unavailable; novelty uses only real candidate-cluster rarity and pHash deduplication")
            novelty = rarity
        for candidate, score, label in zip(candidates, novelty, labels, strict=True):
            candidate.novelty_score = float(score)
            candidate.dino_cluster_id = int(label)
    else:
        LOGGER.warning("No DINO embeddings available; novelty score is disabled (no dummy embeddings used)")
    output.parent.mkdir(parents=True, exist_ok=True)
    flat = [row.to_flat_dict() for row in candidates]
    _write_csv(output, flat)
    with cache.open("w", encoding="utf-8") as destination:
        for row in candidates:
            destination.write(json.dumps(row.__dict__ if hasattr(row, "__dict__") else {
                field.name: getattr(row, field.name) for field in fields(Candidate)
            }, ensure_ascii=False) + "\n")
    metrics = {"candidate_count": len(candidates), "elapsed_sec": time.perf_counter() - started, "analysis_thresholds": analysis_thresholds, "dino_novelty_available": embedding_path.exists(), "reference_novelty_available": reference_path.exists()}
    (config.output_dir / "candidates" / "score_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("[score] %d/%d elapsed=%.1fs", len(candidates), len(candidates), metrics["elapsed_sec"])
    return candidates


def run_select(config: PipelineConfig, candidates: Sequence[Candidate], *, force: bool = False) -> list[Candidate]:
    output = config.output_dir / "candidates" / "selected_candidates.csv"
    cache = config.output_dir / "candidates" / "selected_candidates.jsonl"
    if output.exists() and cache.exists() and not force:
        LOGGER.info("[select] reuse %s", output)
        return [Candidate(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines() if line]
    selected, rejected, summary = select_frame_candidates(
        candidates,
        frames_per_video=config.frames_per_video,
        ratios=config.selection_bucket_ratios,
        min_interval_sec=config.min_interval_sec,
        head_model_name=config.head_model.name if config.head_model is not None else "",
        head_analysis_threshold=(
            config.head_model.analysis_threshold
            if config.head_model is not None and config.head_model.analysis_threshold is not None
            else config.head_model.confidence_threshold if config.head_model is not None else 0.0
        ),
        require_relevance_evidence=any(
            bucket in config.selection_bucket_ratios
            for bucket in {"bbox_instability", "low_confidence", "head_evidence", "dino_novelty"}
        ),
    )
    rows = [row.to_flat_dict() for row in selected]
    _write_csv(output, rows)
    duplicate_rows = [row.to_flat_dict() for row in rejected if row.rejection_reason == "exact_pixel_duplicate"]
    candidate_fields = list(rows[0].keys()) if rows else []
    _write_csv(
        config.output_dir / "candidates" / "rejected_duplicates.csv",
        duplicate_rows,
        candidate_fields,
    )
    with cache.open("w", encoding="utf-8") as destination:
        for row in selected:
            destination.write(json.dumps({field.name: getattr(row, field.name) for field in fields(Candidate)}, ensure_ascii=False) + "\n")
    (config.output_dir / "candidates" / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("[select] selected=%d buckets=%s videos=%s", len(selected), dict(Counter(row.selection_bucket for row in selected)), dict(Counter(row.video_id for row in selected)))
    return selected


MANIFEST_FIELDS = [
    "frame_id", "source_video_id", "source_filename", "source_hash", "source_frame_index",
    "timestamp_sec", "decoded_timestamp_sec", "frame_width", "frame_height", "frame_path",
    "candidate_pixel_sha256", "export_pixel_sha256", "primary_bucket", "extraction_reason", "selection_score",
    "selection_reason", "all_reasons", "disagreement_score", "temporal_anomaly_score",
    "bbox_instability_score", "low_confidence_score", "confidence_boundary_score",
    "confidence_boundary_model", "confidence_boundary_value", "confidence_boundary_threshold",
    "confidence_boundary_ratio", "confidence_boundary_side", "other_instability_score", "uniform_score",
    "head_score", "novelty_score", "dino_cluster_id", "dino_cluster_by_bucket_json", "roi_source", "roi_sources", "roi_bbox", "track_id",
    "track_event_types", "eligibility_reason", "model_detection_counts_json", "model_predictions_json",
    "context_start_sec", "context_end_sec", "context_clip_path", "overlay_clip_path",
    "overlay_image_path", "random_seed", "status", "review_error",
]


def _source_hash(path: Path) -> str:
    """Create a cheap content-aware source hash without reading an entire large video."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as source:
        digest.update(source.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            source.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(source.read(1024 * 1024))
    return digest.hexdigest()[:12]


def _generated_paths(row: dict[str, Any]) -> Iterable[Path]:
    for key in ("frame_path", "context_clip_path", "overlay_clip_path", "overlay_image_path"):
        value = str(row.get(key, "")).strip()
        if value:
            yield Path(value)


def _remove_previous_generated(
    rows: Sequence[dict[str, Any]],
    dataset_output: Path,
    retained_paths: set[Path],
) -> None:
    allowed_root = dataset_output.resolve()
    for row in rows:
        for path in _generated_paths(row):
            resolved = path.resolve()
            if (
                resolved not in retained_paths
                and resolved.is_relative_to(allowed_root)
                and resolved.is_file()
            ):
                resolved.unlink()


def run_export(
    config: PipelineConfig,
    inventory: Sequence[VideoInfo],
    selected: Sequence[Candidate],
    *,
    force: bool = False,
    resume: bool = False,
) -> list[dict[str, Any]]:
    dataset_output = (config.dataset_output_dir or config.output_dir).resolve()
    dataset_output.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "selection_manifest.csv"
    manifest_jsonl_path = config.output_dir / "selection_manifest.jsonl"
    if manifest_path.exists() and not force and not resume:
        LOGGER.info("[export] reuse %s", manifest_path)
        return _read_csv(manifest_path)
    previous_manifest = _read_csv(manifest_path) if force and manifest_path.is_file() else []
    by_video = {row.video_id: row for row in inventory}
    adapters = [create_adapter(model) for model in _scan_model_configs(config)] if selected else []
    source_hashes: dict[str, str] = {}
    exported_hashes: dict[str, str] = {}
    export_rejections: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    render_seconds = 0.0
    started = time.perf_counter()
    for index, candidate in enumerate(selected, 1):
        video = by_video[candidate.video_id]
        video_path = Path(video.path)
        if video.video_id not in source_hashes:
            source_hashes[video.video_id] = _source_hash(video_path)
        source_hash = source_hashes[video.video_id]
        frame, actual_index, actual_timestamp = read_video_frame(
            video_path,
            timestamp_sec=candidate.timestamp_sec,
            frame_index=candidate.frame_index,
        )
        export_hash = pixel_sha256(frame)
        if export_hash in exported_hashes:
            export_rejections.append({
                "frame_id": candidate.clip_id,
                "source_video_id": candidate.video_id,
                "source_filename": candidate.filename,
                "timestamp_sec": f"{candidate.timestamp_sec:.6f}",
                "export_pixel_sha256": export_hash,
                "duplicate_of_frame_id": exported_hashes[export_hash],
                "reason": "exact_export_pixel_duplicate",
            })
            LOGGER.warning(
                "[export frames] skip exact decoded duplicate %s (same as %s)",
                candidate.clip_id,
                exported_hashes[export_hash],
            )
            continue
        exported_hashes[export_hash] = candidate.clip_id
        safe_stem = (safe_name(Path(candidate.filename).stem) or "video")[:80]
        frame_name = (
            f"{safe_stem}__{source_hash}__f{actual_index:010d}"
            f"__t{actual_timestamp:012.3f}s.jpg"
        )
        frame_path = dataset_output / frame_name
        if force or not frame_path.is_file():
            temporary_frame = frame_path.with_name(f".{frame_path.stem}.partial.jpg")
            if not cv2.imwrite(
                temporary_frame.as_posix(), frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            ):
                raise RuntimeError(f"Failed to write dataset frame: {frame_path}")
            temporary_frame.replace(frame_path)
        start = max(0.0, candidate.timestamp_sec - config.context_half_window_sec)
        end = min(video.duration, candidate.timestamp_sec + config.context_half_window_sec)
        context_clip = config.output_dir / "context_clips" / f"{candidate.clip_id}.mp4"
        overlay_clip = config.output_dir / "overlays" / f"{candidate.clip_id}_models.mp4"
        overlay_image = config.output_dir / "overlays" / f"{candidate.clip_id}_models.jpg"
        review_error = ""
        try:
            if force or not context_clip.is_file():
                cut_source_clip(
                    video_path,
                    context_clip,
                    start_sec=start,
                    duration_sec=max(1e-3, end - start),
                    config=config,
                    clip_id=candidate.clip_id,
                )
            if force or not overlay_clip.is_file() or not overlay_image.is_file():
                render_seconds += render_review_clip(
                    context_clip,
                    overlay_clip,
                    overlay_image,
                    candidate,
                    adapters,
                    config,
                    start,
                    duration_sec=end - start,
                )
        except Exception as error:  # Frame export is intentionally not gated by review assets.
            review_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("[export review] failed for %s; original frame remains exported", candidate.clip_id)
        all_reasons = sorted(
            {
                candidate.selection_reason,
                candidate.eligibility_reason,
                *candidate.track_event_types,
                *(key for key, value in candidate.score_components.items() if float(value) > 0),
            }
            - {""}
        )
        manifest.append({
            "frame_id": candidate.clip_id,
            "source_video_id": candidate.video_id,
            "source_filename": candidate.filename,
            "source_hash": source_hash,
            "source_frame_index": actual_index,
            "timestamp_sec": f"{candidate.timestamp_sec:.6f}",
            "decoded_timestamp_sec": f"{actual_timestamp:.6f}",
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "frame_path": frame_path.resolve().as_posix(),
            "candidate_pixel_sha256": candidate.pixel_sha256,
            "export_pixel_sha256": export_hash,
            "primary_bucket": candidate.selection_bucket,
            "extraction_reason": selection_reason_label(candidate.selection_bucket),
            "selection_score": f"{candidate.selection_score:.8f}",
            "selection_reason": candidate.selection_reason,
            "all_reasons": ";".join(all_reasons),
            "disagreement_score": f"{candidate.disagreement_score:.8f}",
            "temporal_anomaly_score": f"{candidate.temporal_anomaly_score:.8f}",
            "bbox_instability_score": f"{candidate.bbox_instability_score:.8f}",
            "low_confidence_score": f"{candidate.low_confidence_score:.8f}",
            "confidence_boundary_score": f"{candidate.confidence_boundary_score:.8f}",
            "confidence_boundary_model": candidate.confidence_boundary_model,
            "confidence_boundary_value": f"{candidate.confidence_boundary_value:.8f}",
            "confidence_boundary_threshold": f"{candidate.confidence_boundary_threshold:.8f}",
            "confidence_boundary_ratio": f"{candidate.confidence_boundary_ratio:.8f}",
            "confidence_boundary_side": candidate.confidence_boundary_side,
            "other_instability_score": f"{candidate.other_instability_score:.8f}",
            "uniform_score": f"{candidate.uniform_score:.8f}",
            "head_score": f"{candidate.head_score:.8f}",
            "novelty_score": f"{candidate.novelty_score:.8f}",
            "dino_cluster_id": candidate.dino_cluster_id,
            "dino_cluster_by_bucket_json": json.dumps(candidate.dino_cluster_by_bucket, sort_keys=True),
            "roi_source": candidate.roi_source,
            "roi_sources": ";".join(candidate.roi_sources),
            "roi_bbox": "" if candidate.roi_bbox is None else ",".join(f"{value:.3f}" for value in candidate.roi_bbox),
            "track_id": candidate.track_id,
            "track_event_types": ";".join(candidate.track_event_types),
            "eligibility_reason": candidate.eligibility_reason,
            "model_detection_counts_json": json.dumps(candidate.model_counts, ensure_ascii=False, sort_keys=True),
            "model_predictions_json": json.dumps(candidate.model_predictions, ensure_ascii=False, separators=(",", ":")),
            "context_start_sec": f"{start:.6f}",
            "context_end_sec": f"{end:.6f}",
            "context_clip_path": context_clip.resolve().as_posix() if context_clip.is_file() else "",
            "overlay_clip_path": overlay_clip.resolve().as_posix() if overlay_clip.is_file() else "",
            "overlay_image_path": overlay_image.resolve().as_posix() if overlay_image.is_file() else "",
            "random_seed": config.random_seed,
            "status": "frame_exported" if review_error else "complete",
            "review_error": review_error,
        })
        _write_csv(manifest_path, manifest, MANIFEST_FIELDS)
        manifest_jsonl_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
        )
        LOGGER.info("[export frames] %d/%d %s", index, len(selected), frame_name)
    _write_csv(manifest_path, manifest, MANIFEST_FIELDS)
    manifest_jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    metrics = {
        "elapsed_sec": time.perf_counter() - started,
        "review_render_sec": render_seconds,
        "frames": len(manifest),
        "exact_export_duplicates": len(export_rejections),
        "review_failures": sum(bool(row["review_error"]) for row in manifest),
        "review_model_inference": {
            adapter.model_name: {
                "seconds": adapter.inference_seconds,
                "frames": adapter.frames_processed,
                "effective_fps": adapter.effective_fps,
            }
            for adapter in adapters
        },
    }
    (config.output_dir / "logs" / "export_metrics.json").parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        config.output_dir / "candidates" / "export_rejections.csv",
        export_rejections,
        [
            "frame_id", "source_video_id", "source_filename", "timestamp_sec",
            "export_pixel_sha256", "duplicate_of_frame_id", "reason",
        ],
    )
    (config.output_dir / "logs" / "export_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if force and previous_manifest:
        retained_paths = {
            path.resolve()
            for row in manifest
            for path in _generated_paths(row)
        }
        _remove_previous_generated(previous_manifest, dataset_output, retained_paths)
    del adapters
    gc.collect()
    return manifest


def run_report(
    config: PipelineConfig,
    manifest: Sequence[dict[str, Any]],
    scan_metrics: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    by_bucket = Counter(str(row["primary_bucket"]) for row in manifest)
    by_video = Counter(str(row["source_video_id"]) for row in manifest)
    selection_summary_path = config.output_dir / "candidates" / "selection_summary.json"
    selection_summary = (
        json.loads(selection_summary_path.read_text(encoding="utf-8"))
        if selection_summary_path.is_file() else {}
    )
    all_candidates_path = config.output_dir / "candidates" / "all_candidates.csv"
    all_candidate_rows = _read_csv(all_candidates_path) if all_candidates_path.is_file() else []
    export_rejections_path = config.output_dir / "candidates" / "export_rejections.csv"
    export_rejections = _read_csv(export_rejections_path) if export_rejections_path.is_file() else []
    eligibility_counts = Counter(str(row.get("eligibility_reason", "")) for row in all_candidate_rows)
    model_names = [model.name for model in _scan_model_configs(config)]
    summary = {
        "dataset_name": config.dataset_name,
        "input_video_count": len(by_video),
        "selected_frame_count": len(manifest),
        "target_frames_per_video": config.frames_per_video,
        "by_bucket": dict(by_bucket),
        "by_video": dict(by_video),
        "random_seed": config.random_seed,
        "models": model_names,
        "scan_metrics": scan_metrics or {},
        "shortfalls": selection_summary.get("shortfalls", {}),
        "candidate_eligibility_counts": dict(eligibility_counts),
        "review_failures": sum(bool(row.get("review_error")) for row in manifest),
        "export_shortfalls": {
            "exact_export_pixel_duplicate": len(export_rejections),
        },
        "dataset_output_dir": (config.dataset_output_dir or config.output_dir).as_posix(),
        "internal_output_dir": config.output_dir.as_posix(),
    }
    write_review_template(config.output_dir / "review_template.csv", manifest, model_names)
    write_html_report(config.output_dir / "review.html", manifest, summary, model_names)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("[report] frames=%d buckets=%s", len(manifest), dict(by_bucket))
    return summary


def _fingerprint_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _run_fingerprint(config: PipelineConfig) -> dict[str, Any]:
    videos = _input_videos(config)
    model_paths = [model.weights for model in _scan_model_configs(config)]
    model_paths.extend(model.config for model in _scan_model_configs(config) if model.config is not None)
    payload = {
        "config": config.resolved_dict(),
        "input_videos": [_fingerprint_entry(path) for path in videos],
        "model_files": [_fingerprint_entry(path) for path in model_paths],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(), **payload}


def prepare_output(
    config: PipelineConfig,
    *,
    force: bool = False,
    requested_stage: str = "all",
) -> None:
    dataset_output = config.dataset_output_dir or config.output_dir
    dataset_output.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "logs").mkdir(exist_ok=True)
    fingerprint_path = config.output_dir / "run_fingerprint.json"
    fingerprint = _run_fingerprint(config)
    if fingerprint_path.is_file():
        previous = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if previous.get("sha256") != fingerprint["sha256"]:
            if not force:
                raise RuntimeError(
                    "Input videos, model files, or resolved config changed since the previous run. "
                    "Inspect run_fingerprint.json and rerun with --stage all --force."
                )
            if requested_stage != "all":
                raise RuntimeError(
                    "The run fingerprint changed, so forcing only one stage could reuse stale downstream "
                    "artifacts. Rerun with --stage all --force."
                )
    fingerprint_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.resolved_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
