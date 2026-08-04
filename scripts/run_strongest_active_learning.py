#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import fnmatch
import gc
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from script.script_data.clip_frame import CliError  # noqa: E402
from src.active_learning.strongest_config import (  # noqa: E402
    load_strongest_pipeline_config,
)
from src.active_learning.strongest_pipeline import (  # noqa: E402
    aggregate_manifests,
    align_source_timestamps_to_frame_grid,
    build_failure_candidates,
    discover_source_videos,
    export_selected_frames,
    prepare_montage,
    prepare_run_metadata,
    preflight_source_video_names,
    read_completed_inference,
    read_selection_cache,
    record_completed_source_video_names,
    resolved_model_provenance,
    run_inference,
    select_failure_candidates,
    write_report,
    write_selection_cache,
)


LOGGER = logging.getLogger("strongest_active_learning")
STAGES = ("montage", "infer", "select", "export", "report", "all")
_INFERENCE_WORKER_CONFIG: Any | None = None
_INFERENCE_WORKER_PROVENANCE: dict[str, Any] | None = None
_INFERENCE_WORKER_ADAPTER: Any | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 10-minute uniform montages, run the registry-selected "
            "strongest face model, and export up to 50 failure frames per video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/active_learning_strongest.yaml",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="One directory below data/input/active_learning",
    )
    parser.add_argument(
        "--model-ref",
        help="Override model registry active_model for this run",
    )
    parser.add_argument(
        "--video-glob",
        action="append",
        default=[],
        help="Process only matching source filenames; may be repeated",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the requested stage (all regenerates every stage)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial inference JSONL",
    )
    parser.add_argument(
        "--video-history-csv",
        type=Path,
        default=PROJECT_ROOT / "data/video_name_history.csv",
        help="Persistent source-video name history shared with clip_frame.py",
    )
    parser.add_argument(
        "--allow-known-videos",
        "--allow-known-video-names",
        dest="allow_known_video_names",
        action="store_true",
        help=(
            "Allow an intentional rerun when a source video name already "
            "exists in clip/active-learning history"
        ),
    )
    parser.add_argument(
        "--pipeline-batch-size",
        type=int,
        help="Runtime inference batch override; model weights and thresholds are unchanged",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        help="Runtime detector microbatch override for constrained VRAM",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        help="Runtime keypoint-query microbatch override for constrained VRAM",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        help="Concurrent source-video inference workers on the configured GPU",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/assets and print the source/model plan without writing outputs",
    )
    return parser.parse_args(argv)


def _configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "_active_learning" / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _close_inference_worker_adapter() -> None:
    global _INFERENCE_WORKER_ADAPTER

    close_adapter = getattr(_INFERENCE_WORKER_ADAPTER, "close", None)
    if callable(close_adapter):
        close_adapter()
    _INFERENCE_WORKER_ADAPTER = None


def _initialize_inference_worker(
    config: Any,
    model_provenance: Mapping[str, Any],
) -> None:
    global _INFERENCE_WORKER_ADAPTER
    global _INFERENCE_WORKER_CONFIG
    global _INFERENCE_WORKER_PROVENANCE

    _INFERENCE_WORKER_CONFIG = config
    _INFERENCE_WORKER_PROVENANCE = dict(model_provenance)
    _INFERENCE_WORKER_ADAPTER = config.model.build_adapter(config.device)
    atexit.register(_close_inference_worker_adapter)


def _infer_work_in_worker(
    work: Any,
    *,
    force: bool,
    resume: bool,
) -> dict[str, Any]:
    if (
        _INFERENCE_WORKER_CONFIG is None
        or _INFERENCE_WORKER_PROVENANCE is None
        or _INFERENCE_WORKER_ADAPTER is None
    ):
        raise RuntimeError("Inference worker was not initialized")
    started = time.perf_counter()
    frames = run_inference(
        _INFERENCE_WORKER_CONFIG,
        work,
        _INFERENCE_WORKER_ADAPTER,
        _INFERENCE_WORKER_PROVENANCE,
        force=force,
        resume=resume,
    )
    elapsed = time.perf_counter() - started
    return {
        "work_id": work.work_id,
        "filename": work.source.filename,
        "frames": len(frames),
        "elapsed_sec": elapsed,
        "fps": len(frames) / elapsed if elapsed > 0 else None,
        "worker_pid": os.getpid(),
    }


def _run_parallel_inference(
    config: Any,
    works: Sequence[Any],
    model_provenance: Mapping[str, Any],
    *,
    force: bool,
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    worker_count = min(config.inference.video_workers, len(works))
    if worker_count < 2:
        raise ValueError("Parallel inference requires at least two workers")
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_inference_worker,
        initargs=(config, dict(model_provenance)),
    ) as executor:
        futures = {
            executor.submit(
                _infer_work_in_worker,
                work,
                force=force,
                resume=resume,
            ): work
            for work in works
        }
        for future in as_completed(futures):
            work = futures[future]
            try:
                result = future.result()
            except Exception as error:
                LOGGER.exception(
                    "[infer-worker] failed: %s",
                    work.source.filename,
                )
                failures[work.source.filename] = (
                    f"{type(error).__name__}: {error}"
                )
                continue
            results[work.work_id] = result
            LOGGER.info(
                "[infer-worker] %s frames=%d elapsed=%.1fs fps=%.2f pid=%d",
                result["filename"],
                result["frames"],
                result["elapsed_sec"],
                result["fps"],
                result["worker_pid"],
            )
    return results, failures


def _selection_for_work(
    config: Any,
    work: Any,
    model_provenance: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    frames = read_completed_inference(config, work, model_provenance)
    source_fps = float(work.source.fps)
    if source_fps > 0:
        # Selection must hold after source-frame quantization, not only on the
        # ideal montage-to-source timestamp mapping. At 29.97 fps, two ideal
        # timestamps 10.0 seconds apart can otherwise decode 299 frames apart
        # (9.9766 seconds).
        frames = align_source_timestamps_to_frame_grid(frames, source_fps)
    candidates = build_failure_candidates(frames, config)
    selected, summary = select_failure_candidates(candidates, config)
    summary["evaluated_frames"] = len(frames)
    maximum = int(summary["maximum_after_exact_dedup_and_min_interval"])
    if candidates:
        uncapped_config = replace(
            config,
            selection=replace(
                config.selection,
                max_frames_per_video=len(candidates),
            ),
        )
        _, uncapped_summary = select_failure_candidates(
            candidates,
            uncapped_config,
        )
        summary["current_policy_without_cap"] = {
            "requested": len(candidates),
            "selected": uncapped_summary["selected"],
            "actual": uncapped_summary["actual"],
            "refill_count": uncapped_summary["refill_count"],
        }
    else:
        summary["current_policy_without_cap"] = {
            "requested": 0,
            "selected": 0,
            "actual": {
                category: 0
                for category in config.selection.category_ratios
            },
            "refill_count": 0,
        }
    sensitivity: dict[str, Any] = {}
    for cap in sorted(
        {
            value
            for value in (25, 50, 70, 71, 75, 100, 150, 200, 300, maximum)
            if 0 < value <= maximum
        }
    ):
        audit_config = replace(
            config,
            selection=replace(config.selection, max_frames_per_video=cap),
        )
        _, cap_summary = select_failure_candidates(candidates, audit_config)
        sensitivity[str(cap)] = {
            "selected": cap_summary["selected"],
            "actual": cap_summary["actual"],
            "refill_count": cap_summary["refill_count"],
            "total_shortfall": cap_summary["total_shortfall"],
        }
    summary["cap_sensitivity"] = sensitivity
    write_selection_cache(
        config,
        work,
        selected,
        summary,
        model_provenance,
    )
    LOGGER.info(
        "[select] %s candidates=%d selected=%d categories=%s",
        work.source.filename,
        len(candidates),
        len(selected),
        summary["actual"],
    )
    return selected, summary


def _write_capacity_audit(
    config: Any,
    video_summaries: Mapping[str, Mapping[str, Any]],
    model_provenance: Mapping[str, Any],
) -> None:
    audit_dir = config.output_dir / "_active_learning"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for filename, summary in video_summaries.items():
        maximum = int(summary["maximum_after_exact_dedup_and_min_interval"])
        row: dict[str, Any] = {
            "source_video": filename,
            "evaluated_frames": int(summary.get("evaluated_frames", 0)),
            "candidate_count_after_exact_dedup": int(
                summary["candidate_count"]
            ),
            "mathematical_capacity_after_min_interval": maximum,
            "maximum_selected_by_current_policy": int(
                summary["current_policy_without_cap"]["selected"]
            ),
            "configured_cap": int(config.selection.max_frames_per_video),
            "selected_at_configured_cap": int(summary["selected"]),
            "cap_share_of_mathematical_capacity": (
                config.selection.max_frames_per_video / maximum
                if maximum
                else None
            ),
            "cap_share_of_current_policy_maximum": (
                config.selection.max_frames_per_video
                / int(summary["current_policy_without_cap"]["selected"])
                if int(summary["current_policy_without_cap"]["selected"])
                else None
            ),
            "refill_count": int(summary["refill_count"]),
            "total_shortfall": int(summary["total_shortfall"]),
        }
        for category in config.selection.category_ratios:
            row[f"raw_{category}"] = int(
                summary["candidate_count_by_category"][category]
            )
            row[f"spaced_{category}"] = int(
                summary["maximum_by_category_after_min_interval"][category]
            )
            row[f"selected_{category}"] = int(summary["actual"][category])
        rows.append(row)

    maxima = [
        int(row["mathematical_capacity_after_min_interval"])
        for row in rows
    ]
    policy_maxima = [
        int(row["maximum_selected_by_current_policy"])
        for row in rows
    ]
    sorted_maxima = sorted(maxima)
    middle = len(sorted_maxima) // 2
    median = (
        (
            sorted_maxima[middle - 1] + sorted_maxima[middle]
        )
        / 2
        if sorted_maxima and len(sorted_maxima) % 2 == 0
        else (sorted_maxima[middle] if sorted_maxima else 0)
    )
    payload = {
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "definition_of_maximum": (
            "mathematical capacity is the exact interval-scheduling ceiling "
            "after exact pixel deduplication; current-policy maximum is what "
            "the score/quota-first selector emits when its configured cap is removed"
        ),
        "sample_fps": config.inference.sample_fps,
        "minimum_interval_sec": config.selection.min_interval_sec,
        "configured_cap_per_video": config.selection.max_frames_per_video,
        "model": dict(model_provenance),
        "aggregate": {
            "videos": len(rows),
            "evaluated_frames": sum(
                int(row["evaluated_frames"]) for row in rows
            ),
            "candidate_count_after_exact_dedup": sum(
                int(row["candidate_count_after_exact_dedup"]) for row in rows
            ),
            "mathematical_capacity_after_min_interval_total": sum(maxima),
            "maximum_selected_by_current_policy_total": sum(policy_maxima),
            "configured_cap_total": (
                config.selection.max_frames_per_video * len(rows)
            ),
            "selected_at_configured_cap_total": sum(
                int(row["selected_at_configured_cap"]) for row in rows
            ),
            "mathematical_capacity_per_video_min": min(maxima, default=0),
            "mathematical_capacity_per_video_median": median,
            "mathematical_capacity_per_video_max": max(maxima, default=0),
            "videos_filling_configured_cap": sum(
                int(row["selected_at_configured_cap"])
                == config.selection.max_frames_per_video
                for row in rows
            ),
        },
        "videos": {
            filename: dict(summary)
            for filename, summary in video_summaries.items()
        },
    }
    audit_json = audit_dir / "capacity_audit.json"
    audit_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_csv = audit_dir / "capacity_audit.csv"
    if rows:
        with audit_csv.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    LOGGER.info(
        "[audit] policy_maximum_total=%d mathematical_capacity_total=%d "
        "configured_cap_total=%d report=%s",
        payload["aggregate"]["maximum_selected_by_current_policy_total"],
        payload["aggregate"][
            "mathematical_capacity_after_min_interval_total"
        ],
        payload["aggregate"]["configured_cap_total"],
        audit_json,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_strongest_pipeline_config(
        args.config,
        dataset_name=args.dataset_name,
        model_ref=args.model_ref,
    )
    runtime_sizes = {
        "pipeline_batch_size": args.pipeline_batch_size,
        "detector_batch_size": args.detector_batch_size,
        "query_batch_size": args.query_batch_size,
        "video_workers": args.video_workers,
    }
    for name, value in runtime_sizes.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.pipeline_batch_size is not None:
        config = replace(
            config,
            inference=replace(
                config.inference,
                pipeline_batch_size=args.pipeline_batch_size,
            ),
        )
    if args.video_workers is not None:
        config = replace(
            config,
            inference=replace(
                config.inference,
                video_workers=args.video_workers,
            ),
        )
    if (
        args.detector_batch_size is not None
        or args.query_batch_size is not None
    ):
        options = dict(config.model.options)
        if args.detector_batch_size is not None:
            options["detector_batch_size"] = args.detector_batch_size
        if args.query_batch_size is not None:
            options["query_batch_size"] = args.query_batch_size
        config = replace(
            config,
            model=replace(config.model, options=options),
        )
    config.validate()
    source_paths = discover_source_videos(config)
    if args.video_glob:
        source_paths = [
            path
            for path in source_paths
            if any(
                fnmatch.fnmatch(path.name, pattern)
                for pattern in args.video_glob
            )
        ]
        if not source_paths:
            raise FileNotFoundError(
                "No source video matched --video-glob: "
                + ", ".join(args.video_glob)
            )
    try:
        name_preflight = preflight_source_video_names(
            source_paths,
            history_csv=args.video_history_csv,
            project_root=config.project_root,
            allow_known_names=args.allow_known_video_names,
        )
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if name_preflight["allowed_known_names"]:
        print(
            "Known video names are allowed for this intentional rerun:\n"
            + name_preflight["allowed_match_detail"],
            flush=True,
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset_name": config.dataset_name,
                    "input_dir": config.input_dir.as_posix(),
                    "output_dir": config.output_dir.as_posix(),
                    "source_videos": [path.as_posix() for path in source_paths],
                    "video_name_preflight": name_preflight,
                    "model_id": config.model.model_id,
                    "adapter": config.model.adapter_class,
                    "montage": {
                        "timeline_segments": config.montage.timeline_segments,
                        "clip_duration_sec": config.montage.clip_duration_sec,
                        "duration_sec": config.montage.duration_sec,
                    },
                    "inference": {
                        "sample_fps": config.inference.sample_fps,
                        "pipeline_batch_size": (
                            config.inference.pipeline_batch_size
                        ),
                        "video_workers": config.inference.video_workers,
                        "expected_frames_per_video": int(
                            config.montage.duration_sec
                            * config.inference.sample_fps
                        ),
                    },
                    "selection": {
                        "max_frames_per_video": (
                            config.selection.max_frames_per_video
                        ),
                        "category_ratios": config.selection.category_ratios,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    _configure_logging(config.output_dir)
    started = time.perf_counter()
    works = []
    failures: dict[str, str] = {}
    for index, source_path in enumerate(source_paths, 1):
        try:
            LOGGER.info(
                "[montage] %d/%d %s",
                index,
                len(source_paths),
                source_path.name,
            )
            works.append(
                prepare_montage(
                    config,
                    source_path,
                    force=args.force and args.stage in {"montage", "all"},
                )
            )
        except Exception as error:
            LOGGER.exception("[montage] failed: %s", source_path)
            failures[source_path.name] = f"{type(error).__name__}: {error}"

    if args.stage == "montage":
        if failures:
            failure_path = config.output_dir / "_active_learning" / "failures.json"
            failure_path.write_text(
                json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        LOGGER.info(
            "Completed montage stage: success=%d failed=%d elapsed=%.1fs",
            len(works),
            len(failures),
            time.perf_counter() - started,
        )
        if not failures:
            history_result = record_completed_source_video_names(
                source_paths,
                input_dir=config.input_dir,
                output_dir=config.output_dir,
                history_csv=args.video_history_csv,
                project_root=config.project_root,
            )
            LOGGER.info(
                "[video names] recorded=%d run_csv=%s history_csv=%s",
                history_result["recorded_videos"],
                history_result["run_video_names_csv"],
                history_result["history_csv"],
            )
        return 1 if failures else 0

    model_provenance = resolved_model_provenance(config)
    prepare_run_metadata(config, model_provenance)
    adapter = None
    parallel_inference = (
        args.stage in {"infer", "all"}
        and len(works) > 1
        and config.inference.video_workers > 1
    )
    parallel_results: dict[str, dict[str, Any]] = {}
    parallel_failures: dict[str, str] = {}
    if parallel_inference:
        LOGGER.info(
            "Starting concurrent inference: videos=%d workers=%d device=%s",
            len(works),
            min(config.inference.video_workers, len(works)),
            config.device,
        )
        parallel_results, parallel_failures = _run_parallel_inference(
            config,
            works,
            model_provenance,
            force=args.force,
            resume=args.resume,
        )
        failures.update(parallel_failures)
    elif args.stage in {"infer", "all"} and works:
        LOGGER.info(
            "Loading active model %s via %s",
            config.model.model_id,
            config.model.adapter_class,
        )
        adapter = config.model.build_adapter(config.device)

    per_video_rows: list[list[dict[str, Any]]] = []
    video_summaries: dict[str, Any] = {}
    for index, work in enumerate(works, 1):
        if work.source.filename in parallel_failures:
            continue
        try:
            LOGGER.info(
                "[%s] %d/%d %s",
                args.stage,
                index,
                len(works),
                work.source.filename,
            )
            if args.stage in {"infer", "all"}:
                if parallel_inference:
                    result = parallel_results.get(work.work_id)
                    if result is None:
                        raise RuntimeError(
                            "Concurrent inference returned no result for "
                            f"{work.source.filename}"
                        )
                    LOGGER.info(
                        "[infer] %s frames=%d elapsed=%.1fs fps=%.2f pid=%d",
                        work.source.filename,
                        result["frames"],
                        result["elapsed_sec"],
                        result["fps"],
                        result["worker_pid"],
                    )
                else:
                    assert adapter is not None
                    frames = run_inference(
                        config,
                        work,
                        adapter,
                        model_provenance,
                        force=args.force,
                        resume=args.resume,
                    )
                    LOGGER.info(
                        "[infer] %s frames=%d",
                        work.source.filename,
                        len(frames),
                    )
            if args.stage in {"select", "all"}:
                selected, summary = _selection_for_work(
                    config,
                    work,
                    model_provenance,
                )
                video_summaries[work.source.filename] = summary
            if args.stage in {"export", "all"}:
                if args.stage == "export":
                    selected, summary = read_selection_cache(
                        config,
                        work,
                        model_provenance,
                    )
                    video_summaries[work.source.filename] = summary
                rows = export_selected_frames(
                    config,
                    work,
                    selected,
                    force=args.force,
                )
                per_video_rows.append(rows)
                LOGGER.info(
                    "[export] %s frames=%d",
                    work.source.filename,
                    len(rows),
                )
            if args.stage == "report":
                rows = _read_csv(
                    work.work_dir / "selection" / "selection_manifest.csv"
                )
                if not rows:
                    raise FileNotFoundError(
                        f"Run export first for {work.source.filename}"
                    )
                _, summary = read_selection_cache(
                    config,
                    work,
                    model_provenance,
                )
                per_video_rows.append(rows)
                video_summaries[work.source.filename] = summary
        except Exception as error:
            LOGGER.exception(
                "[%s] failed: %s",
                args.stage,
                work.source.filename,
            )
            failures[work.source.filename] = f"{type(error).__name__}: {error}"

    if args.stage in {"export", "report", "all"}:
        manifest = aggregate_manifests(config, per_video_rows)
        write_report(
            config,
            manifest,
            video_summaries,
            model_provenance,
        )
    if args.stage in {"select", "all"} and video_summaries:
        _write_capacity_audit(
            config,
            video_summaries,
            model_provenance,
        )
    failure_path = config.output_dir / "_active_learning" / "failures.json"
    if failures:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        failure_path.unlink(missing_ok=True)

    if not failures and args.stage == "all":
        history_result = record_completed_source_video_names(
            source_paths,
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            history_csv=args.video_history_csv,
            project_root=config.project_root,
        )
        LOGGER.info(
            "[video names] recorded=%d run_csv=%s history_csv=%s",
            history_result["recorded_videos"],
            history_result["run_video_names_csv"],
            history_result["history_csv"],
        )

    close_adapter = getattr(adapter, "close", None)
    if callable(close_adapter):
        close_adapter()
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    LOGGER.info(
        "Completed stage=%s videos=%d failed=%d elapsed=%.1fs output=%s",
        args.stage,
        max(0, len(source_paths) - len(failures)),
        len(failures),
        time.perf_counter() - started,
        config.output_dir,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
